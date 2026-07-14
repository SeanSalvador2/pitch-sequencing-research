r"""Bayesian contextual bandit -- the myopic prescriptive rung (SPEC ``12.5``; decisions D36-D39).

WS4 turns WS3's decomposed outcome model into an *uncertainty-aware* "best next pitch"
**target policy** and hands it to the OPE gate (:mod:`pitchseq.eval.ope`) for honest offline
evaluation. It never learns online and never re-fits an outcome or behavior model: it
**consumes** WS3's published artifacts through :func:`~workstreams.ws3_gbdt_stack.model.load_ws3_artifacts`
(decision D33) and evaluates **strictly** through :func:`pitchseq.eval.ope.evaluate_policy`
(decision D37). Phase B begins here: prescription, never trusted without OPE.

This module supplies the reusable pieces; :mod:`workstreams.ws4_bandit.run_ws4` wires them
into the checkpointed pipeline.

The pieces
==========

``thompson_policy`` (decision D36)
    Per-row Thompson probabilities :math:`P(a=\arg\max)` from independent per-action normals
    :math:`\mathcal N(\hat q(s,a),\,\sigma(s,a)^2)`, restricted to the leakage-safe
    **feasible** actions (``XX`` is never feasible; SPEC ``4``). Monte-Carlo, deterministic
    under a seed, numerically hygienic (rows sum to 1, zero mass on infeasible actions).
    Rows with an **empty** feasible mask (low-history pitchers) fall back to the
    **observed-action point mass** -- documented as *carrying no recommendation*.

``soften`` (SPEC ``9``)
    The conservative mixture :math:`\pi_\alpha=(1-\alpha)\mu+\alpha\tilde\pi`. Delegates to
    :func:`pitchseq.eval.ope.pi_alpha` -- **not** re-implemented.

``ambiguity_stats`` (decision D38)
    The uncertainty decomposition: per-row posterior :math:`P(\text{top beats runner-up})`
    and the share of rows whose recommended action is *ambiguous* at the 50 / 80 / 95%
    confidence levels -- "how often is the pick a real distinction versus a toss-up".

``deviation_map`` (decision D38)
    Mean total-variation distance of the target from behavior per count cell (and per any
    grouping) -- the "where does it dare to differ" exhibit.

``build_bandit_inputs``
    Loads the WS3 artifacts for one view and assembles the aligned
    ``(q, q_sd, mu, feasible_mask, align)`` bundle the pipeline evaluates, with shape /
    finiteness checks and the documented behavior-propensity floor.

The Thompson uncertainty (an important, documented modelling choice)
====================================================================

WS3's ``exp_reward_sd`` is a **residual** reward standard deviation -- the law-of-total-
variance spread of a *single pitch's* reward around :math:`\hat q(s,a)` (see
:meth:`~workstreams.ws3_gbdt_stack.model.OutcomeStack.exp_reward`). It is the outcome
**noise**, on the order of the reward scale (``|R|`` ~ 0.05-0.3), and it is *not* the
posterior standard error of the mean :math:`\hat q(s,a)` that a Thompson sampler needs:
used raw it dwarfs the inter-family :math:`\hat q` gaps (~0.005) and drives the policy to a
near-uniform draw over feasible actions. :func:`build_bandit_inputs` therefore returns
``q_sd = POSTERIOR_SCALE * exp_reward_sd``, mapping the per-pitch residual sd to a posterior
standard error of the mean via a single documented shrinkage
:math:`\kappa=` :data:`POSTERIOR_SCALE` :math:`\approx 1/\sqrt{n_\text{eff}}`, with
:math:`n_\text{eff}` the effective per-cell support a LightGBM leaf aggregates
(``min_child_samples`` is on the 10\ :sup:`2` scale, so :math:`\kappa\approx 0.1`).
:data:`POSTERIOR_SCALE` ``= 1.0`` recovers the literal "``exp_reward ± sd``" of decision
D36 but yields the near-uniform policy; ``kappa`` is the single exposed confidence knob and
:func:`thompson_policy` itself is agnostic to how ``q_sd`` was formed.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import ndtr

from pitchseq.config import load_config
from pitchseq.eval.ope import pi_alpha
from pitchseq.families import FAMILIES, OTHER_FAMILY, feasible_action_mask
from pitchseq.states import build_view
from workstreams.ws3_gbdt_stack.model import (
    ACTION_COL,
    WS3Artifacts,
    load_ws3_artifacts,
)

__all__ = [
    "FAMILIES",
    "ABLATION_VIEWS",
    "COMMON_EVAL_VIEW",
    "POSTERIOR_SCALE",
    "MU_FLOOR",
    "DEFAULT_N_SAMPLES",
    "XX_INDEX",
    "BanditInputs",
    "thompson_policy",
    "soften",
    "ambiguity_stats",
    "deviation_map",
    "build_bandit_inputs",
    "family_index",
    "feasible_matrix",
]

#: The D38 prescriptive-ablation view set: C (context only), L1 (previous pitch), O (full
#: ordered current-PA history). The C -> O value change is the sequencing-prescription
#: evidence; U / OM are out of scope for the myopic-bandit ablation.
ABLATION_VIEWS: tuple[str, ...] = ("C", "L1", "O")

#: The view whose behavior model and q-grid form the **common OPE evaluator** for the
#: ablation (the richest view in :data:`ABLATION_VIEWS`). Every view's target policy is
#: scored against this single (mu, q_hat) yardstick so the C -> O value change isolates the
#: policy's *information*, not the evaluator's.
COMMON_EVAL_VIEW = "O"

#: Shrinkage mapping WS3's residual ``exp_reward_sd`` to a posterior SE of the mean for the
#: Thompson draws (see the module docstring). ``0.05`` makes the Thompson uncertainty
#: commensurate with the inter-family ``q_hat`` gaps (~0.005), so the target is a *meaningful*
#: (concentrated, ~0.65-0.75 top-action posterior confidence) recommendation rather than a
#: near-uniform draw; it corresponds to an effective per-cell support ``n_eff ~ 1/kappa^2 ~
#: 400`` (a GBDT leaf aggregating a few hundred comparable pitches). ``1.0`` is the literal
#: D36 "exp_reward ± sd" and yields a near-uniform policy (residual noise >> the q_hat gaps).
POSTERIOR_SCALE = 0.05

#: Behavior-propensity floor applied (then renormalised) before importance ratios, so
#: ``mu(a|s) > 0`` for every action and the ratios stay finite (SPEC ``9`` numerical
#: hygiene). Well below the ``0.01`` OPE support threshold, so it never perturbs a
#: genuinely-supported action -- it only repairs a numerically-zero LightGBM output.
MU_FLOOR = 1e-6

#: Monte-Carlo draws per row for :func:`thompson_policy` (deterministic under the seed).
DEFAULT_N_SAMPLES = 1500

#: Index of the descriptive-only ``XX`` family (never a recommendable action; SPEC ``4``).
XX_INDEX = FAMILIES.index(OTHER_FAMILY)

_N_FAM = len(FAMILIES)
_FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}


class BanditInputs(NamedTuple):
    """The aligned per-row inputs for one state view (return of :func:`build_bandit_inputs`).

    Unpacks as the ``(q, q_sd, mu, feasible_mask, align)`` 5-tuple.

    Attributes
    ----------
    q : numpy.ndarray, shape (n, 8)
        ``E[R | s, a]`` for every family (WS3 q-grid, :data:`FAMILIES` order).
    q_sd : numpy.ndarray, shape (n, 8)
        Posterior standard error of ``q`` for the Thompson draws (WS3 residual
        ``exp_reward_sd`` scaled by :data:`POSTERIOR_SCALE`; see the module docstring).
    mu : numpy.ndarray, shape (n, 8)
        Behavior propensities ``mu(a | s)`` (floored at :data:`MU_FLOOR`, renormalised).
    feasible_mask : numpy.ndarray of bool, shape (n, 8)
        Leakage-safe feasible-action mask (``XX`` always ``False``; SPEC ``4``).
    align : pandas.DataFrame
        Row alignment: ``row_id``, ``observed_action`` (int family index), ``low_history``
        (bool) and ``any_feasible`` (bool), one row per decision in ``table`` order.
    """

    q: np.ndarray
    q_sd: np.ndarray
    mu: np.ndarray
    feasible_mask: np.ndarray
    align: pd.DataFrame


# -------------------------------------------------------------------------------------
# small helpers
# -------------------------------------------------------------------------------------


def _as_rng(rng) -> np.random.Generator:
    """Coerce ``None`` / int / :class:`numpy.random.Generator` to a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def family_index(families) -> np.ndarray:
    """Map an iterable of family codes to their integer index in :data:`FAMILIES`.

    Unknown / null codes map to the ``XX`` index (they are never a recommendable action).
    """
    return np.array(
        [_FAMILY_TO_IDX.get(f, XX_INDEX) for f in np.asarray(families, dtype=object)],
        dtype=np.int64,
    )


def feasible_matrix(mask_df: pd.DataFrame) -> np.ndarray:
    """Extract the ``(n, 8)`` boolean feasibility matrix from a
    :func:`~pitchseq.families.feasible_action_mask` frame (``FAMILIES`` column order)."""
    cols = [f"feasible_{f}" for f in FAMILIES]
    return mask_df[cols].to_numpy(dtype=bool)


# -------------------------------------------------------------------------------------
# D36: Thompson target policy
# -------------------------------------------------------------------------------------


def thompson_policy(
    q,
    q_sd,
    feasible_mask,
    n_samples: int = DEFAULT_N_SAMPLES,
    rng=None,
    observed_action=None,
    batch_rows: int = 4096,
) -> np.ndarray:
    r"""Per-row Thompson probabilities :math:`P(a=\arg\max)` over feasible actions (decision D36).

    For each decision row, draw ``n_samples`` independent samples of the per-action value
    from :math:`\mathcal N(q[i,a],\,q\_sd[i,a]^2)`, restrict the ``arg\max`` to the row's
    **feasible** actions, and return the empirical win frequency per action. The result is a
    valid distribution: it sums to exactly 1, places **zero** mass on infeasible actions, and
    is deterministic for a fixed ``rng``.

    * A **dominant** mean (one feasible action far above the rest relative to its sd) yields a
      near-point-mass; near-degenerate ``q_sd`` yields the exact ``arg\max`` point mass.
    * **Equal** means with equal sds yield a uniform distribution over the feasible actions.
    * A row with an **empty** feasible mask (a low-history pitcher whose repertoire cannot be
      established from pre-game history; SPEC ``4``) **falls back to the observed-action point
      mass** when ``observed_action`` is supplied -- these rows *carry no recommendation*, the
      target simply mirrors what was actually thrown. Without ``observed_action`` an empty row
      is left as an all-zero row (the caller must handle it).

    ``q_sd`` is taken as given: it is the *posterior* standard error of the mean, which
    :func:`build_bandit_inputs` forms by scaling WS3's residual ``exp_reward_sd`` (see the
    module docstring) -- this function does not re-scale it.

    Parameters
    ----------
    q : array-like, shape (n, A)
        Per-action posterior means ``E[R | s, a]``.
    q_sd : array-like, shape (n, A)
        Per-action posterior standard errors (``>= 0``).
    feasible_mask : array-like of bool, shape (n, A)
        Feasible actions per row (``XX`` must already be ``False``; SPEC ``4``).
    n_samples : int, optional
        Monte-Carlo draws per row (default :data:`DEFAULT_N_SAMPLES`).
    rng : None, int or numpy.random.Generator, optional
        Seed / generator for reproducible draws.
    observed_action : array-like of int, shape (n,), optional
        Observed (logged) action index per row, used as the empty-mask fallback.
    batch_rows : int, optional
        Row block size for the vectorised draw (bounds peak memory to
        ``batch_rows * n_samples * A`` floats); does not affect the result.

    Returns
    -------
    numpy.ndarray, shape (n, A)
        Per-row Thompson probabilities over feasible actions.
    """
    q = np.asarray(q, dtype=np.float64)
    q_sd = np.asarray(q_sd, dtype=np.float64)
    if q.shape != q_sd.shape:
        raise ValueError(f"q {q.shape} and q_sd {q_sd.shape} must match")
    feas = np.asarray(feasible_mask, dtype=bool)
    if feas.shape != q.shape:
        raise ValueError(f"feasible_mask {feas.shape} must match q {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("q contains non-finite values")
    if (q_sd < 0).any() or not np.isfinite(q_sd).all():
        raise ValueError("q_sd must be finite and non-negative")
    n, A = q.shape
    rng = _as_rng(rng)
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    out = np.zeros((n, A), dtype=np.float64)
    any_feasible = feas.any(axis=1)

    # Empty-mask rows: observed-action point mass (no recommendation).
    empty = ~any_feasible
    if empty.any():
        if observed_action is not None:
            obs = np.asarray(observed_action, dtype=np.int64)
            out[np.flatnonzero(empty), obs[empty]] = 1.0
        # else: left as zeros -- caller's responsibility.

    active = np.flatnonzero(any_feasible)
    neg_inf = -np.inf
    for start in range(0, len(active), batch_rows):
        rows = active[start:start + batch_rows]
        m = len(rows)
        mu_b = q[rows]                       # (m, A)
        sd_b = q_sd[rows]
        feas_b = feas[rows]
        # (m, n_samples, A) draws; infeasible actions pushed to -inf so they never win.
        draws = rng.standard_normal((m, n_samples, A)) * sd_b[:, None, :] + mu_b[:, None, :]
        draws = np.where(feas_b[:, None, :], draws, neg_inf)
        winners = draws.argmax(axis=2)        # (m, n_samples)
        # Count wins per action for each row.
        counts = np.zeros((m, A), dtype=np.float64)
        for a in range(A):
            counts[:, a] = (winners == a).sum(axis=1)
        out[rows] = counts / float(n_samples)
    return out


# -------------------------------------------------------------------------------------
# SPEC 9: conservative softening (delegates to eval/ope.pi_alpha)
# -------------------------------------------------------------------------------------


def soften(target_probs, mu_probs, alpha):
    r"""The SPEC ``9`` conservative mixture ``pi_alpha = (1-alpha) mu + alpha * target``.

    A thin, order-adapting wrapper over :func:`pitchseq.eval.ope.pi_alpha` (which takes
    ``(mu_probs, target_probs, alpha)``) -- WS4 does **not** re-implement the mixture.
    ``alpha=0`` returns behavior ``mu``; ``alpha=1`` returns the target; an array of alphas
    returns a ``(G, n, A)`` stack.

    Parameters
    ----------
    target_probs : array-like, shape (n, A)
        The proposed target policy ``pi_tilde(. | s)`` (e.g. :func:`thompson_policy`).
    mu_probs : array-like, shape (n, A)
        The behavior policy ``mu(. | s)``.
    alpha : float or array-like
        Mixing weight(s) in ``[0, 1]``.
    """
    return pi_alpha(mu_probs, target_probs, alpha)


# -------------------------------------------------------------------------------------
# D38: ambiguity decomposition
# -------------------------------------------------------------------------------------


def ambiguity_stats(q, q_sd, feasible_mask) -> dict:
    r"""The D38 uncertainty decomposition of the recommendation (per-row, analytic).

    For each **decidable** row (>= 2 feasible actions) let the *top* action be the feasible
    ``arg\max`` of the posterior mean ``q`` and the *runner-up* the next-best feasible mean.
    Treating the two posteriors as independent normals, the posterior probability that the top
    truly beats the runner-up is

    .. math::
       p_\text{beat} = \Phi\!\left(\frac{q_\text{top}-q_\text{ru}}
                                        {\sqrt{\sigma_\text{top}^2+\sigma_\text{ru}^2}}\right)
                     \in [0.5, 1],

    (:math:`\Phi` the standard-normal CDF; :math:`\ge 0.5` because the top is the mean-argmax).
    A row's recommendation is **ambiguous at the ``L`` confidence level** when
    :math:`p_\text{beat} < L`: we are less than ``L``-confident the recommended action beats
    its closest feasible alternative. Reported at ``L in {0.50, 0.80, 0.95}`` (higher ``L`` =>
    stricter => more rows ambiguous; ``L=0.50`` fires only on near-exact mean ties).

    Rows with a single feasible action (no runner-up) and empty-mask rows are *not decidable*
    and are counted separately, not included in the shares.

    Parameters
    ----------
    q, q_sd : array-like, shape (n, A)
        Per-action posterior means and standard errors.
    feasible_mask : array-like of bool, shape (n, A)
        Feasible actions per row.

    Returns
    -------
    dict
        ``n_rows``, ``n_decidable`` (>= 2 feasible), ``n_singleton`` (1 feasible),
        ``n_infeasible`` (0 feasible), ``mean_p_beat`` / ``median_p_beat`` (over decidable
        rows), ``ambiguous_50`` / ``ambiguous_80`` / ``ambiguous_95`` (share of decidable
        rows below each level) and ``mean_gap`` (mean posterior-mean margin top - runner-up).
    """
    q = np.asarray(q, dtype=np.float64)
    q_sd = np.asarray(q_sd, dtype=np.float64)
    feas = np.asarray(feasible_mask, dtype=bool)
    n, A = q.shape
    n_feas = feas.sum(axis=1)
    decidable = n_feas >= 2

    result = {
        "n_rows": int(n),
        "n_decidable": int(decidable.sum()),
        "n_singleton": int((n_feas == 1).sum()),
        "n_infeasible": int((n_feas == 0).sum()),
        "mean_p_beat": float("nan"),
        "median_p_beat": float("nan"),
        "mean_gap": float("nan"),
        "ambiguous_50": float("nan"),
        "ambiguous_80": float("nan"),
        "ambiguous_95": float("nan"),
    }
    if not decidable.any():
        return result

    qd = q[decidable]
    sdd = q_sd[decidable]
    fd = feas[decidable]
    masked = np.where(fd, qd, -np.inf)
    order = np.argsort(masked, axis=1)          # ascending; last two columns = runner-up, top
    rows = np.arange(qd.shape[0])
    top = order[:, -1]
    ru = order[:, -2]
    q_top = qd[rows, top]
    q_ru = qd[rows, ru]
    sd_top = sdd[rows, top]
    sd_ru = sdd[rows, ru]
    gap = q_top - q_ru
    pooled = np.sqrt(sd_top ** 2 + sd_ru ** 2)
    # Degenerate pooled sd -> deterministic winner (p_beat = 1 when gap > 0, else 0.5 tie).
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(pooled > 0, gap / pooled, np.where(gap > 0, np.inf, 0.0))
    p_beat = ndtr(z)

    result["mean_p_beat"] = float(p_beat.mean())
    result["median_p_beat"] = float(np.median(p_beat))
    result["mean_gap"] = float(gap.mean())
    result["ambiguous_50"] = float((p_beat < 0.50).mean())
    result["ambiguous_80"] = float((p_beat < 0.80).mean())
    result["ambiguous_95"] = float((p_beat < 0.95).mean())
    return result


# -------------------------------------------------------------------------------------
# D38: deviation-from-behavior map
# -------------------------------------------------------------------------------------


def deviation_map(target_probs, mu_probs, table, by=("balls", "strikes")) -> pd.DataFrame:
    r"""Mean total-variation distance of the target from behavior, per cell (decision D38).

    The per-row deviation is the total-variation distance
    :math:`\mathrm{TV}(\pi,\mu)=\tfrac12\sum_a|\pi(a|s)-\mu(a|s)|\in[0,1]`; this groups it by
    ``by`` (default the count cell ``balls x strikes``) and reports the mean and row count per
    cell -- the "where does it dare to differ" exhibit. Pass a different ``by`` (e.g.
    ``("state_view",)`` after concatenating views, or ``("balls","strikes","platoon")``) for
    other cuts.

    Parameters
    ----------
    target_probs, mu_probs : array-like, shape (n, A)
        Target and behavior policies per row (aligned to ``table``).
    table : pandas.DataFrame
        Decision rows carrying the ``by`` columns (aligned to the policy rows by position).
    by : sequence of str, optional
        Grouping columns (default ``("balls", "strikes")``).

    Returns
    -------
    pandas.DataFrame
        One row per cell: the ``by`` columns plus ``mean_tv`` and ``n``, sorted by the cell
        keys. (For the scalar overall mean TV, take
        ``float((0.5*np.abs(target-mu).sum(1)).mean())`` -- the pipeline reports it alongside
        this per-cell map.)
    """
    target = np.asarray(target_probs, dtype=np.float64)
    mu = np.asarray(mu_probs, dtype=np.float64)
    if target.shape != mu.shape:
        raise ValueError(f"target {target.shape} and mu {mu.shape} must match")
    tv = 0.5 * np.abs(target - mu).sum(axis=1)
    by = list(by)
    frame = table[by].copy().reset_index(drop=True)
    frame["_tv"] = tv
    grouped = (
        frame.groupby(by, observed=True)["_tv"]
        .agg(mean_tv="mean", n="size")
        .reset_index()
        .sort_values(by, kind="stable")
        .reset_index(drop=True)
    )
    return grouped


# -------------------------------------------------------------------------------------
# WS3-artifact consumption
# -------------------------------------------------------------------------------------


def _floor_renormalise(mu: np.ndarray, floor: float) -> np.ndarray:
    """Floor a probability matrix at ``floor`` and renormalise rows to sum to 1."""
    mu = np.clip(np.asarray(mu, dtype=np.float64), floor, None)
    return mu / mu.sum(axis=1, keepdims=True)


def _q_and_sd_grid(stack, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    r"""Counterfactual ``(q, exp_reward_sd)`` grids for all 8 families in **one** view build.

    Mirrors :meth:`~workstreams.ws3_gbdt_stack.model.OutcomeStack.q_grid`'s efficient pattern
    -- build the view matrix once, then swap the ``action_family`` column per family and
    re-run the stack's own stage-A/B probabilities + assembly -- but *retains* the residual
    ``exp_reward_sd`` that ``q_grid`` discards. It uses the loaded stack's own
    ``_stage_probs`` / ``_assemble`` (consuming WS3, not re-implementing it), so the means are
    identical to ``stack.q_grid`` / ``stack.exp_reward`` and the expensive view construction is
    done once per view instead of once per family (an ~8x speed-up on the real val+test rows).
    """
    balls = table["balls"].to_numpy()
    strikes = table["strikes"].to_numpy()
    X, _ = build_view(table, stack.view)
    X = X.copy()
    n = len(table)
    q = np.empty((n, _N_FAM), dtype=np.float64)
    sd = np.empty((n, _N_FAM), dtype=np.float64)
    for j, fam in enumerate(FAMILIES):
        X[ACTION_COL] = pd.Categorical([fam] * n, categories=list(FAMILIES))
        p_a, p_b = stack._stage_probs(X)
        q[:, j], sd[:, j] = stack._assemble(p_a, p_b, balls, strikes)
    return q, sd


def build_bandit_inputs(
    table: pd.DataFrame,
    ws3,
    view: str,
    config: dict | None = None,
    posterior_scale: float | None = None,
    feasible_mask: np.ndarray | None = None,
    low_history: np.ndarray | None = None,
) -> BanditInputs:
    r"""Assemble the aligned per-row bandit inputs for one state view from WS3's artifacts.

    Consumes WS3 through :func:`~workstreams.ws3_gbdt_stack.model.load_ws3_artifacts`
    (decision D33) -- it never re-fits a behavior or outcome model. The behavior propensities
    ``mu`` come from the view's :class:`~workstreams.ws3_gbdt_stack.model.BehaviorModel`; the
    per-family ``q`` (means) and residual ``exp_reward_sd`` come from the view's
    :class:`~workstreams.ws3_gbdt_stack.model.OutcomeStack` (one action-swept pass over the 8
    families). ``q_sd`` is the residual sd scaled by ``posterior_scale`` (see the module
    docstring); ``mu`` is floored at :data:`MU_FLOOR` and renormalised so importance ratios
    stay finite (SPEC ``9``). The feasible mask is the leakage-safe SPEC ``4`` mask (``XX``
    never feasible).

    Parameters
    ----------
    table : pandas.DataFrame
        Decision rows to evaluate on (typically the held-out val + test rows).
    ws3 : str, pathlib.Path or WS3Artifacts
        A directory written by the WS3 pipeline, or an already-loaded
        :class:`~workstreams.ws3_gbdt_stack.model.WS3Artifacts`.
    view : str
        One of :data:`ABLATION_VIEWS` (must be present in the artifacts).
    config : dict, optional
        Study config (for the feasible-mask thresholds); loaded from default when ``None``.
    posterior_scale : float, optional
        Override for :data:`POSTERIOR_SCALE` (the residual-sd -> posterior-SE shrinkage).
    feasible_mask, low_history : numpy.ndarray, optional
        A precomputed ``(n, 8)`` boolean feasibility matrix and ``(n,)`` low-history flag,
        aligned to ``table``. **Pass these** so the SPEC ``4`` trailing window is computed over
        the *full* table (train + val + test) and merely subset to ``table`` -- computing
        :func:`~pitchseq.families.feasible_action_mask` on held-out rows alone would undercount
        the pre-game repertoire (a 2024 game's trailing 365 days lie in the train seasons). When
        ``None`` the mask is (re)computed from ``table`` itself. The mask is view-independent, so
        the pipeline computes it once and shares it across views.

    Returns
    -------
    BanditInputs
        The ``(q, q_sd, mu, feasible_mask, align)`` bundle, all aligned to ``table`` order,
        shape- and finiteness-checked.
    """
    if config is None:
        config = load_config()
    if posterior_scale is None:
        posterior_scale = POSTERIOR_SCALE
    artifacts = ws3 if isinstance(ws3, WS3Artifacts) else load_ws3_artifacts(Path(ws3))
    if view not in artifacts.behavior or view not in artifacts.outcome:
        raise KeyError(
            f"view {view!r} not in WS3 artifacts (have behavior={sorted(artifacts.behavior)}, "
            f"outcome={sorted(artifacts.outcome)})"
        )
    n = len(table)

    mu = artifacts.propensities(table, view)                       # (n, 8), FAMILIES order
    if mu.shape != (n, _N_FAM) or not np.isfinite(mu).all():
        raise ValueError(f"behavior propensities for view {view!r} malformed: shape {mu.shape}")
    mu = _floor_renormalise(mu, MU_FLOOR)

    stack = artifacts.outcome[view]
    q, q_sd_resid = _q_and_sd_grid(stack, table)     # (n, 8) means + residual sds, one build
    if not np.isfinite(q).all() or not np.isfinite(q_sd_resid).all():
        raise ValueError(f"q / q_sd for view {view!r} contain non-finite values")
    q_sd = q_sd_resid * float(posterior_scale)

    if feasible_mask is None or low_history is None:
        mask_df = feasible_action_mask(table, config=config)
        feasible = feasible_matrix(mask_df)                        # (n, 8), XX always False
        low_hist = mask_df["low_history"].to_numpy(dtype=bool)
    else:
        feasible = np.asarray(feasible_mask, dtype=bool)
        low_hist = np.asarray(low_history, dtype=bool)
        if feasible.shape != (n, _N_FAM) or low_hist.shape != (n,):
            raise ValueError(
                f"precomputed feasible_mask {feasible.shape} / low_history {low_hist.shape} "
                f"must be ({n}, {_N_FAM}) / ({n},)"
            )
        if feasible[:, XX_INDEX].any():
            raise ValueError("feasible_mask must never mark XX feasible (SPEC 4)")
    any_feasible = feasible.any(axis=1)

    observed = family_index(table["family"].astype("object").to_numpy())
    align = pd.DataFrame(
        {
            "row_id": table["row_id"].to_numpy(),
            "observed_action": observed,
            "low_history": low_hist,
            "any_feasible": any_feasible,
        }
    )
    return BanditInputs(q=q, q_sd=q_sd, mu=mu, feasible_mask=feasible, align=align)
