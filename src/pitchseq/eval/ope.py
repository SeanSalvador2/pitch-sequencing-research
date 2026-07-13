"""Off-policy evaluation -- the prescriptive gate (SPEC ``9``).

Every prescriptive workstream (WS4 bandit, WS5 tabular MDP, WS7 offline RL) outputs a
*distribution over feasible pitches* and is judged here, never by a raw deterministic pick.
This module is the one OPE interface they all emit through. It provides

* the conservative mixture policy :func:`pi_alpha` (SPEC ``9``),
* six estimators -- Direct Method, IPS, self-normalised IPS, Doubly Robust, step-wise
  (per-decision) DR for full episodes, and Fitted-Q Evaluation,
* the :func:`ope_diagnostics` overlap / weight read-out,
* :func:`evaluate_policy`, the one-call wrapper that runs every applicable estimator,
  cluster-bootstraps confidence intervals, assembles the SPEC ``9`` report and applies the
  DM/DR/SNIPS agreement verdict, and
* :func:`behavior_policy_recovery`, SPEC ``9`` self-test #1.

Input contract
--------------
Inputs are plain arrays / frames. A *logged* frame has one row per decision with an integer
``action`` taken, its ``reward``, the behavior propensity ``mu_prob`` of the taken action
(and, when available, the full behavior distribution in ``mu_prob_0 ... mu_prob_{A-1}``), an
episode id ``pa_id`` and a within-episode ``step`` index. A *target* is supplied as a
per-row probability matrix ``target_probs`` of shape ``(n, A)`` giving ``pi(.|s_t)`` for the
state of each logged row. An outcome model, when available, is a per-row matrix ``q_hat`` of
shape ``(n, A)`` with ``q_hat[i, a] = E-hat[R | s_i, a]``.

Estimator formulas (undiscounted; :math:`w_i=\\pi(a_i|s_i)/\\mu(a_i|s_i)` is the taken-action
importance ratio, :math:`\\bar q^\\pi(s_i)=\\sum_a \\pi(a|s_i)\\,\\hat q(s_i,a)`):

.. math::
   V_{\\mathrm{DM}}   &= \\tfrac1n \\sum_i \\bar q^\\pi(s_i) \\\\
   V_{\\mathrm{IPS}}  &= \\tfrac1n \\sum_i w_i r_i \\\\
   V_{\\mathrm{SNIPS}}&= \\Big(\\sum_i w_i r_i\\Big)\\Big/\\Big(\\sum_i w_i\\Big) \\\\
   V_{\\mathrm{DR}}   &= \\tfrac1n \\sum_i \\big[\\bar q^\\pi(s_i) + w_i\\,(r_i-\\hat q(s_i,a_i))\\big]

Step-wise DR uses the per-decision recursion (Jiang & Li 2016), backward over an episode's
steps ``t = H-1 .. 0`` with per-step ratio :math:`\\rho_t=\\pi(a_t|s_t)/\\mu(a_t|s_t)`:

.. math::
   V^{\\mathrm{DR}}_H = 0, \\qquad
   V^{\\mathrm{DR}}_t = \\bar q^\\pi(s_t) + \\rho_t\\,\\big(r_t + V^{\\mathrm{DR}}_{t+1} - \\hat q(s_t,a_t)\\big),

and the estimate is the mean over episodes of :math:`V^{\\mathrm{DR}}_0`. For one-step
(bandit) episodes this is exactly :math:`V_{\\mathrm{DR}}`. FQE fits :math:`\\hat q` by
backward regression and reports :math:`\\tfrac1{|E|}\\sum_e \\sum_a \\pi(a|s_0^e)\\hat q(s_0^e,a)`;
for one-step episodic data it degenerates to the Direct Method with a fitted outcome model
(documented on :func:`fqe`).

Numerical hygiene
-----------------
Behavior propensities of *taken* actions are always strictly positive (the action was
sampled from ``mu``), so importance ratios need no clipping to stay finite; the code asserts
this rather than silently repairing it. Probabilities are clipped away from zero *only* in
the KL / total-variation diagnostics, where a target action with zero behavior support would
make :math:`\\log(\\pi/\\mu)` diverge -- that genuine lack of overlap is surfaced by the
support diagnostics instead. Optional IPS/SNIPS weight clipping caps :math:`w_i` at a fixed
threshold (config ``ope.weight_clip``); both clipped and unclipped values are reported.
Everything is deterministic under a fixed seed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..config import load_config
from ..splits import cluster_bootstrap_indices

__all__ = [
    "pi_alpha",
    "dm",
    "ips",
    "snips",
    "dr",
    "stepwise_dr",
    "fqe",
    "ope_diagnostics",
    "evaluate_policy",
    "behavior_policy_recovery",
    "outcome_model_means",
    "tabular_regressor",
    "default_regressor",
    "DEFAULT_WEIGHT_CLIP",
    "AGREEMENT_ESTIMATORS",
    "SEQUENTIAL_AGREEMENT",
]

#: Default cap for optional IPS/SNIPS weight clipping (config ``ope.weight_clip`` is a
#: boolean switch; this is the numeric threshold applied when it is on, overridable via
#: ``ope.weight_clip_value``). Chosen large enough that it does not bite on a well-overlapped
#: target (so behavior-policy recovery stays exact) but caps pathological ratios.
DEFAULT_WEIGHT_CLIP = 100.0

#: Estimators whose mutual agreement drives the SPEC ``9`` verdict on **bandit** (one-step)
#: data. The high-variance raw IPS is reported but excluded from the automatic gate (its wide
#: CI would make the tolerance vacuous). Matches the task's precise "DM vs DR vs SNIPS" rule.
AGREEMENT_ESTIMATORS = ("DM", "SNIPS", "DR")

#: Agreement set on **sequential** (multi-step) data, where only the trajectory-level
#: estimators evaluate the episode return.
SEQUENTIAL_AGREEMENT = ("stepwise_DR", "FQE")

#: Per-decision (bandit) estimators -- valid return estimators only when every episode is a
#: single step; skipped on multi-step data (they answer a per-decision, not per-return,
#: question). Step-wise DR and FQE handle both and degenerate to these on one-step data.
_BANDIT_ESTIMATORS = ("DM", "IPS", "SNIPS", "DR")

_KL_EPS = 1e-12  # floor for mu in log(pi/mu); see module "Numerical hygiene".


# -------------------------------------------------------------------------------------
# input helpers
# -------------------------------------------------------------------------------------


def _as_matrix(probs, name: str) -> np.ndarray:
    """Validate and return a ``(n, A)`` probability matrix (finite, non-negative, sums~1)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (n, A) matrix, got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"{name} contains non-finite values")
    if p.min() < -1e-9:
        raise ValueError(f"{name} contains negative probabilities")
    row_sums = p.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        worst = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(f"{name} rows must sum to 1 (max deviation {worst:.3g})")
    return p


def _taken(probs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Gather the taken-action entry from a ``(n, A)`` matrix."""
    n = probs.shape[0]
    return probs[np.arange(n), actions]


def _weights(target_probs: np.ndarray, actions: np.ndarray, mu_taken: np.ndarray) -> np.ndarray:
    """Taken-action importance ratios ``w = pi(a|s) / mu(a|s)``.

    Asserts ``mu_taken > 0`` (a logged action always had positive propensity), so the ratio
    is finite -- no silent clipping.
    """
    mu_taken = np.asarray(mu_taken, dtype=np.float64)
    if not np.isfinite(mu_taken).all() or (mu_taken <= 0).any():
        raise ValueError("behavior propensities of taken actions must be finite and > 0")
    pi_taken = _taken(target_probs, actions)
    return pi_taken / mu_taken


def _clip_weights(w: np.ndarray, clip) -> np.ndarray:
    """Cap importance weights at ``clip`` (no-op when ``clip`` is ``None``)."""
    if clip is None:
        return w
    return np.minimum(w, float(clip))


# -------------------------------------------------------------------------------------
# conservative mixture policy (SPEC 9)
# -------------------------------------------------------------------------------------


def pi_alpha(mu_probs, target_probs, alpha):
    r"""The SPEC ``9`` conservative mixture policy ``(1 - alpha) * mu + alpha * pi_tilde``.

    Parameters
    ----------
    mu_probs : array-like, shape (n, A)
        Behavior policy ``mu(.|s)`` per row.
    target_probs : array-like, shape (n, A)
        Proposed target policy ``pi_tilde(.|s)`` per row.
    alpha : float or array-like
        Mixing weight(s) in ``[0, 1]``. ``alpha=0`` returns ``mu``; ``alpha=1`` returns
        ``pi_tilde``. A scalar returns one ``(n, A)`` matrix; an array of length ``G``
        returns a ``(G, n, A)`` stack (vectorised over the config alpha grid).

    Returns
    -------
    numpy.ndarray
        ``(n, A)`` for scalar ``alpha``; ``(G, n, A)`` for a grid. Each row is a valid
        distribution (a convex combination of two distributions).
    """
    mu = _as_matrix(mu_probs, "mu_probs")
    pi = _as_matrix(target_probs, "target_probs")
    if mu.shape != pi.shape:
        raise ValueError(f"mu_probs {mu.shape} and target_probs {pi.shape} must match")
    a = np.asarray(alpha, dtype=np.float64)
    if a.ndim == 0:
        return (1.0 - float(a)) * mu + float(a) * pi
    # Vectorise over the grid: (G, 1, 1) broadcasting.
    a = a.reshape(-1, 1, 1)
    return (1.0 - a) * mu[None, :, :] + a * pi[None, :, :]


# -------------------------------------------------------------------------------------
# estimators -- each returns (value, per-unit contributions for the bootstrap)
# -------------------------------------------------------------------------------------


def dm(q_hat, target_probs) -> tuple[float, np.ndarray]:
    r"""Direct Method: plug the outcome model into the target policy.

    .. math:: V_{\mathrm{DM}} = \frac1n \sum_i \sum_a \pi(a\mid s_i)\,\hat q(s_i, a)

    Parameters
    ----------
    q_hat : array-like, shape (n, A)
        Outcome model ``q_hat[i, a] = E-hat[R | s_i, a]``.
    target_probs : array-like, shape (n, A)
        Target policy per row.

    Returns
    -------
    (float, numpy.ndarray)
        The value and the per-row contributions ``v_i = sum_a pi(a|s_i) q_hat[i, a]``
        (the plain mean of which is the value).
    """
    q = np.asarray(q_hat, dtype=np.float64)
    pi = _as_matrix(target_probs, "target_probs")
    if q.shape != pi.shape:
        raise ValueError(f"q_hat {q.shape} and target_probs {pi.shape} must match")
    if not np.isfinite(q).all():
        raise ValueError("q_hat contains non-finite values")
    v = (pi * q).sum(axis=1)
    return float(v.mean()), v


def ips(rewards, w, clip=None) -> tuple[float, np.ndarray]:
    r"""Inverse-propensity-scored estimator.

    .. math:: V_{\mathrm{IPS}} = \frac1n \sum_i w_i r_i

    Parameters
    ----------
    rewards : array-like, shape (n,)
        Observed rewards.
    w : array-like, shape (n,)
        Taken-action importance ratios ``pi(a_i|s_i)/mu(a_i|s_i)``.
    clip : float, optional
        If given, weights are capped at this value before scoring.

    Returns
    -------
    (float, numpy.ndarray)
        The value and the per-row contributions ``w_i r_i``.
    """
    r = np.asarray(rewards, dtype=np.float64)
    wc = _clip_weights(np.asarray(w, dtype=np.float64), clip)
    contrib = wc * r
    return float(contrib.mean()), contrib


def snips(rewards, w, clip=None) -> tuple[float, np.ndarray]:
    r"""Self-normalised IPS (a.k.a. the weighted / Hajek estimator).

    .. math:: V_{\mathrm{SNIPS}} = \frac{\sum_i w_i r_i}{\sum_i w_i}

    Lower variance than IPS and invariant to a constant reward shift; consistent but slightly
    biased in finite samples.

    Returns
    -------
    (float, numpy.ndarray)
        The value and per-row contributions as an ``(n, 2)`` array of ``[w_i r_i, w_i]`` --
        a *ratio-type* contribution whose value is ``sum(col 0) / sum(col 1)`` (the bootstrap
        recomputes the ratio on each resample).
    """
    r = np.asarray(rewards, dtype=np.float64)
    wc = _clip_weights(np.asarray(w, dtype=np.float64), clip)
    denom = wc.sum()
    value = float((wc * r).sum() / denom) if denom != 0 else float("nan")
    contrib = np.column_stack([wc * r, wc])
    return value, contrib


def dr(rewards, w, q_hat, target_probs, actions) -> tuple[float, np.ndarray]:
    r"""Doubly-robust estimator.

    .. math:: V_{\mathrm{DR}} = \frac1n \sum_i \big[\bar q^\pi(s_i) + w_i\,(r_i-\hat q(s_i,a_i))\big],
              \qquad \bar q^\pi(s_i)=\sum_a \pi(a\mid s_i)\,\hat q(s_i,a)

    Unbiased if *either* the propensities (through ``w``) *or* the outcome model ``q_hat`` is
    correct -- the double-robustness property.

    Parameters
    ----------
    rewards : array-like, shape (n,)
    w : array-like, shape (n,)
        Taken-action importance ratios.
    q_hat : array-like, shape (n, A)
        Outcome model.
    target_probs : array-like, shape (n, A)
    actions : array-like of int, shape (n,)
        Taken-action indices (to read ``q_hat[i, a_i]``).

    Returns
    -------
    (float, numpy.ndarray)
        The value and per-row contributions
        ``dr_i = bar_q_pi(s_i) + w_i (r_i - q_hat[i, a_i])``.
    """
    r = np.asarray(rewards, dtype=np.float64)
    ww = np.asarray(w, dtype=np.float64)
    q = np.asarray(q_hat, dtype=np.float64)
    pi = _as_matrix(target_probs, "target_probs")
    actions = np.asarray(actions, dtype=np.int64)
    if q.shape != pi.shape:
        raise ValueError(f"q_hat {q.shape} and target_probs {pi.shape} must match")
    if not np.isfinite(q).all():
        raise ValueError("q_hat contains non-finite values")
    baseline = (pi * q).sum(axis=1)  # bar q_pi(s_i)
    q_taken = _taken(q, actions)
    contrib = baseline + ww * (r - q_taken)
    return float(contrib.mean()), contrib


def stepwise_dr(
    rewards,
    w,
    q_hat,
    target_probs,
    actions,
    episode_id,
    step,
) -> tuple[float, np.ndarray]:
    r"""Per-decision (step-wise) doubly-robust estimator for full episodes.

    Backward recursion over each episode's steps ``t = H-1 .. 0`` (Jiang & Li 2016), with
    per-step ratio :math:`\rho_t=\pi(a_t|s_t)/\mu(a_t|s_t)` supplied through ``w``:

    .. math::
       V^{\mathrm{DR}}_H = 0,\qquad
       V^{\mathrm{DR}}_t = \bar q^\pi(s_t) + \rho_t\big(r_t + V^{\mathrm{DR}}_{t+1} - \hat q(s_t,a_t)\big)

    The estimate is the mean over episodes of :math:`V^{\mathrm{DR}}_0`. For one-step
    (bandit) episodes it reduces exactly to :func:`dr`.

    Parameters
    ----------
    rewards, w : array-like, shape (n,)
        Per-decision rewards and per-step importance ratios.
    q_hat, target_probs : array-like, shape (n, A)
        Per-decision outcome model and target policy (``q_hat`` should be the *return-to-go*
        Q-function under the target).
    actions : array-like of int, shape (n,)
    episode_id, step : array-like, shape (n,)
        Episode id and within-episode step index (ordering key).

    Returns
    -------
    (float, numpy.ndarray)
        The value and the per-*episode* contributions ``V_0^DR`` (one entry per episode, in
        first-appearance order of the episode ids).
    """
    r = np.asarray(rewards, dtype=np.float64)
    ww = np.asarray(w, dtype=np.float64)
    q = np.asarray(q_hat, dtype=np.float64)
    pi = _as_matrix(target_probs, "target_probs")
    if q.shape != pi.shape:
        raise ValueError(f"q_hat {q.shape} and target_probs {pi.shape} must match")
    if not np.isfinite(q).all():
        raise ValueError("q_hat contains non-finite values")
    actions = np.asarray(actions, dtype=np.int64)
    ep = np.asarray(episode_id)
    st = np.asarray(step)

    baseline = (pi * q).sum(axis=1)  # bar q_pi(s_t)
    q_taken = _taken(q, actions)

    df = pd.DataFrame(
        {
            "ep": ep,
            "step": st,
            "rho": ww,
            "r": r,
            "baseline": baseline,
            "q_taken": q_taken,
            "_order": np.arange(len(r)),
        }
    )
    # Preserve first-appearance episode order for reproducible per-episode contributions.
    ep_order = pd.unique(ep)
    values = np.empty(len(ep_order), dtype=np.float64)
    pos = {e: i for i, e in enumerate(ep_order)}
    for e, g in df.groupby("ep", sort=False):
        g = g.sort_values("step", kind="stable")
        v_next = 0.0
        # Iterate steps backward.
        for base, rho, rr, qtk in zip(
            g["baseline"].to_numpy()[::-1],
            g["rho"].to_numpy()[::-1],
            g["r"].to_numpy()[::-1],
            g["q_taken"].to_numpy()[::-1],
        ):
            v_next = base + rho * (rr + v_next - qtk)
        values[pos[e]] = v_next
    return float(values.mean()), values


# -------------------------------------------------------------------------------------
# fitted-Q evaluation
# -------------------------------------------------------------------------------------


class _TabularRegressor:
    """Exact group-mean regressor over identical feature rows (used by :func:`tabular_regressor`).

    Memorises the mean target for every distinct feature row seen in ``fit`` and predicts it;
    an unseen row falls back to the global training mean. On one-hot ``(state, action)``
    features this reproduces the exact conditional means, so FQE recovers the analytic value
    without model bias -- ideal for the deterministic self-tests.
    """

    def __init__(self) -> None:
        self._table: dict[bytes, float] = {}
        self._global = 0.0

    @staticmethod
    def _keys(X: np.ndarray):
        return [row.tobytes() for row in np.ascontiguousarray(X, dtype=np.float64)]

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._global = float(y.mean()) if len(y) else 0.0
        sums: dict[bytes, float] = {}
        counts: dict[bytes, int] = {}
        for key, val in zip(self._keys(X), y):
            sums[key] = sums.get(key, 0.0) + float(val)
            counts[key] = counts.get(key, 0) + 1
        self._table = {k: sums[k] / counts[k] for k in sums}
        return self

    def predict(self, X):
        return np.array([self._table.get(k, self._global) for k in self._keys(X)], dtype=np.float64)


def tabular_regressor() -> Callable[[], _TabularRegressor]:
    """Return a factory for an exact group-mean regressor (see :class:`_TabularRegressor`)."""
    return _TabularRegressor


def default_regressor(seed: int = 0) -> Callable[[], object]:
    """Return a factory for the default FQE regressor -- sklearn ``HistGradientBoostingRegressor``.

    A shallow, few-iteration gradient-boosted tree; deterministic for a fixed ``seed``.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    def factory():
        return HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )

    return factory


def _one_hot(idx: np.ndarray, n: int) -> np.ndarray:
    """One-hot encode integer indices into ``n`` columns."""
    out = np.zeros((len(idx), n), dtype=np.float64)
    out[np.arange(len(idx)), np.asarray(idx, dtype=np.int64)] = 1.0
    return out


def fqe(
    logged: pd.DataFrame,
    target_probs,
    n_actions: int,
    state_col: str = "state",
    action_col: str = "action",
    reward_col: str = "reward",
    episode_col: str = "pa_id",
    step_col: str = "step",
    episode_id=None,
    step=None,
    regressor_factory: Callable[[], object] | None = None,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    r"""Fitted-Q Evaluation of the target policy through a regressor callback.

    Fits the target's Q-function ``q_hat(s, a)`` by backward regression: at each step the
    regression target is ``r_t`` for terminal decisions and ``r_t + V_{t+1}(s_{t+1})`` for
    non-terminal ones, with ``V_{t+1}(s') = sum_a pi(a|s') q_hat_{t+1}(s', a)`` from the
    already-fitted next-step model. The reported value is the mean over episodes of the
    initial-state value ``sum_a pi(a|s_0) q_hat_0(s_0, a)``.

    For **one-step episodic data** (a bandit) there is a single step, the regression target is
    just ``r``, and the estimate is ``mean_i sum_a pi(a|s_i) q_hat(s_i, a)`` -- i.e. the
    Direct Method with a fitted outcome model. This degeneration is intentional and exact.

    Features are ``[one-hot(state), one-hot(action)]``. Supply ``regressor_factory`` (default
    :func:`default_regressor`, a gradient-boosted tree) to control the learner; the exact
    :func:`tabular_regressor` gives model-bias-free recovery for discrete states.

    Parameters
    ----------
    logged : pandas.DataFrame
        Logged decisions with ``state_col``, ``action_col``, ``reward_col``, ``episode_col``,
        ``step_col``.
    target_probs : array-like, shape (n, A)
        Per-row target policy (aligned to ``logged``).
    n_actions : int
        Action-space size ``A``.
    episode_id, step : array-like, optional
        Episode ids and step indices. When ``None`` they are read from ``episode_col`` /
        ``step_col``; pass them explicitly (as the wrapper does) to run FQE on frames that
        carry no episode/step columns -- a single-step interpretation then applies.
    regressor_factory : callable, optional
        ``() -> regressor`` with sklearn-style ``fit`` / ``predict``. Default gradient boosting.
    seed : int, optional
        Seed for the default regressor.

    Returns
    -------
    (float, numpy.ndarray)
        The value and one initial-state value ``V_0`` per episode (mean over these is the
        estimate; the ordering is irrelevant to the per-episode bootstrap).
    """
    pi = _as_matrix(target_probs, "target_probs")
    if regressor_factory is None:
        regressor_factory = default_regressor(seed)

    states = logged[state_col].to_numpy()
    actions = logged[action_col].to_numpy().astype(np.int64)
    rewards = logged[reward_col].to_numpy().astype(np.float64)
    ep = (logged[episode_col].to_numpy() if episode_id is None else np.asarray(episode_id))
    st = (logged[step_col].to_numpy() if step is None else np.asarray(step)).astype(np.int64)
    n_states = int(states.max()) + 1 if len(states) else 1

    order = np.arange(len(logged))
    # Successor row / state per decision (the next step within the same episode); NaN when
    # terminal. Steps are assumed contiguous within an episode (0, 1, 2, ...).
    df = pd.DataFrame({"ep": ep, "step": st, "state": states, "_pos": order})
    df = df.sort_values(["ep", "step"], kind="stable")
    next_state = df.groupby("ep", sort=False)["state"].shift(-1)
    next_pos = df.groupby("ep", sort=False)["_pos"].shift(-1)
    is_terminal = next_state.isna().to_numpy()
    pos_back = df["_pos"].to_numpy()
    next_state_arr = np.full(len(logged), -1, dtype=np.int64)
    next_state_arr[pos_back] = next_state.fillna(-1).to_numpy().astype(np.int64)
    next_row_arr = np.full(len(logged), -1, dtype=np.int64)
    next_row_arr[pos_back] = next_pos.fillna(-1).to_numpy().astype(np.int64)
    terminal_arr = np.zeros(len(logged), dtype=bool)
    terminal_arr[pos_back] = is_terminal

    def q_all_actions(model, state_idx: np.ndarray) -> np.ndarray:
        """Predict ``q_hat(s, a)`` for every action at each given state -> (m, A)."""
        m = len(state_idx)
        s_oh = _one_hot(state_idx, n_states)
        preds = np.empty((m, n_actions), dtype=np.float64)
        for a in range(n_actions):
            a_oh = _one_hot(np.full(m, a), n_actions)
            preds[:, a] = model.predict(np.hstack([s_oh, a_oh]))
        return preds

    steps_desc = sorted(np.unique(st).tolist(), reverse=True)
    models: dict[int, object] = {}
    for t in steps_desc:
        rows = np.flatnonzero(st == t)
        y = rewards[rows].copy()
        non_term = rows[~terminal_arr[rows]]
        if len(non_term) > 0:
            # V_{t+1}(s') = sum_a pi(a|s') q_hat_{t+1}(s', a) at the *successor* decisions,
            # using the successor rows' target probabilities and the next-step model.
            succ_rows = next_row_arr[non_term]
            succ_states = states[succ_rows]  # == next_state_arr[non_term]
            succ_model = models.get(t + 1)
            if succ_model is not None:
                q_succ = q_all_actions(succ_model, succ_states)
                v_succ = (pi[succ_rows] * q_succ).sum(axis=1)
                # Map back into y (aligned to `rows`).
                pos_in_rows = {r: i for i, r in enumerate(rows)}
                for r, v in zip(non_term, v_succ):
                    y[pos_in_rows[r]] += v
        X = np.hstack([_one_hot(states[rows], n_states), _one_hot(actions[rows], n_actions)])
        model = regressor_factory()
        model.fit(X, y)
        models[t] = model

    # Initial-state value per episode, at each episode's *own* first (minimum-step) decision
    # -- robust to episodes that do not all start at the same step. ``df`` is sorted by
    # (ep, step), so the first row per episode is its minimum-step decision.
    init_pos = df.drop_duplicates("ep", keep="first")["_pos"].to_numpy()
    init_steps = st[init_pos]
    values = np.full(len(init_pos), np.nan, dtype=np.float64)
    for t in np.unique(init_steps):
        m = init_steps == t
        q_init = q_all_actions(models[int(t)], states[init_pos[m]])
        values[m] = (pi[init_pos[m]] * q_init).sum(axis=1)
    return float(values.mean()), values


# -------------------------------------------------------------------------------------
# diagnostics
# -------------------------------------------------------------------------------------


def _kl_tv(mu: np.ndarray, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row KL(pi||mu) and total-variation distance (mu floored for the log only)."""
    mu_safe = np.clip(mu, _KL_EPS, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pi > 0, pi * (np.log(pi + _KL_EPS) - np.log(mu_safe)), 0.0)
    kl = ratio.sum(axis=1)
    tv = 0.5 * np.abs(pi - mu).sum(axis=1)
    return kl, tv


def ope_diagnostics(
    w,
    mu_probs,
    target_probs,
    support_threshold: float = 0.01,
    actions=None,
    rare_threshold: float | None = None,
) -> dict:
    r"""Overlap / weight / distance diagnostics for a target vs the behavior policy (SPEC ``9``).

    Parameters
    ----------
    w : array-like, shape (n,)
        Taken-action importance ratios ``pi(a_i|s_i)/mu(a_i|s_i)``.
    mu_probs : array-like, shape (n, A)
        Full behavior policy per row.
    target_probs : array-like, shape (n, A)
        Full target policy per row.
    support_threshold : float, optional
        Behavior-probability floor defining an "under-supported" action (default 0.01).
    actions : array-like of int, optional
        Taken-action indices; enables ``mu_taken_below_support_frac``.
    rare_threshold : float, optional
        Marginal-behavior floor defining a "rare" action (default: ``support_threshold``).

    Returns
    -------
    dict
        ``ess`` and ``ess_frac`` (:math:`(\sum w)^2/\sum w^2`, absolute and over ``n``),
        ``max_weight``, ``weight_q99`` / ``weight_q999`` (99th / 99.9th-percentile weights),
        ``oos_support_frac`` (mean target mass placed on actions with ``mu <
        support_threshold`` -- responds to a low-propensity target),
        ``mu_taken_below_support_frac`` (fraction of logged decisions whose taken-action
        propensity is below threshold; ``None`` without ``actions``),
        ``mean_kl`` (mean ``KL(pi||mu)``), ``mean_tv`` (mean total-variation distance),
        ``rare_action_frac`` (mean target mass on marginally-rare actions), and ``n``.
    """
    w = np.asarray(w, dtype=np.float64)
    mu = _as_matrix(mu_probs, "mu_probs")
    pi = _as_matrix(target_probs, "target_probs")
    if rare_threshold is None:
        rare_threshold = support_threshold
    n = len(w)

    sw = w.sum()
    sw2 = (w ** 2).sum()
    ess = float(sw * sw / sw2) if sw2 > 0 else 0.0

    low_support = mu < support_threshold  # (n, A)
    oos_support_frac = float((pi * low_support).sum(axis=1).mean())

    marginal_mu = mu.mean(axis=0)  # average behavior distribution over rows
    rare_actions = marginal_mu < rare_threshold  # (A,)
    rare_action_frac = float(pi[:, rare_actions].sum(axis=1).mean()) if rare_actions.any() else 0.0

    kl, tv = _kl_tv(mu, pi)

    mu_taken_below = None
    if actions is not None:
        actions = np.asarray(actions, dtype=np.int64)
        mu_taken = _taken(mu, actions)
        mu_taken_below = float((mu_taken < support_threshold).mean())

    return {
        "n": int(n),
        "ess": ess,
        "ess_frac": float(ess / n) if n else float("nan"),
        "max_weight": float(w.max()) if n else float("nan"),
        "weight_q99": float(np.quantile(w, 0.99)) if n else float("nan"),
        "weight_q999": float(np.quantile(w, 0.999)) if n else float("nan"),
        "oos_support_frac": oos_support_frac,
        "mu_taken_below_support_frac": mu_taken_below,
        "mean_kl": float(kl.mean()),
        "mean_tv": float(tv.mean()),
        "rare_action_frac": rare_action_frac,
        "support_threshold": float(support_threshold),
    }


# -------------------------------------------------------------------------------------
# bootstrap
# -------------------------------------------------------------------------------------


def _agg(contrib: np.ndarray) -> float:
    """Aggregate per-unit contributions: mean for 1-D, ratio of column sums for ``(n, 2)``."""
    contrib = np.asarray(contrib, dtype=np.float64)
    if contrib.ndim == 1:
        return float(contrib.mean()) if len(contrib) else float("nan")
    denom = contrib[:, 1].sum()
    return float(contrib[:, 0].sum() / denom) if denom != 0 else float("nan")


def _bootstrap_ci(
    contrib: np.ndarray,
    n_boot: int,
    seed: int,
    cluster_ids=None,
    alpha: float = 0.05,
) -> dict:
    """Cluster-bootstrap CI over units (episodes) for a contribution array.

    Reuses :func:`pitchseq.splits.cluster_bootstrap_indices` to resample units with
    replacement (each contribution row is its own unit by default, i.e. a per-episode
    bootstrap; pass ``cluster_ids`` to resample coarser clusters such as pitcher-game).
    Returns the point estimate, the two-sided ``1 - alpha`` percentile interval, the
    one-sided 95% lower bound (SPEC ``9``), the bootstrap SE and the replicate count.
    """
    contrib = np.asarray(contrib, dtype=np.float64)
    n = contrib.shape[0]
    point = _agg(contrib)
    if n == 0:
        return {"value": point, "lo": float("nan"), "hi": float("nan"),
                "lower_95": float("nan"), "se": float("nan"), "n_boot": 0}
    ids = np.arange(n) if cluster_ids is None else np.asarray(cluster_ids)
    boot_df = pd.DataFrame({"__unit__": ids})
    reps = []
    for idx in cluster_bootstrap_indices(boot_df, cluster="__unit__", n_boot=n_boot, seed=seed):
        reps.append(_agg(contrib[np.asarray(idx)]))
    reps = np.asarray([v for v in reps if np.isfinite(v)], dtype=np.float64)
    if len(reps) == 0:
        return {"value": point, "lo": float("nan"), "hi": float("nan"),
                "lower_95": float("nan"), "se": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    lower_95 = float(np.percentile(reps, 5.0))
    return {
        "value": point,
        "lo": float(lo),
        "hi": float(hi),
        "lower_95": lower_95,
        "se": float(reps.std(ddof=1)) if len(reps) > 1 else float("nan"),
        "n_boot": int(len(reps)),
    }


# -------------------------------------------------------------------------------------
# outcome-model helper (for DM / DR without an external q_hat)
# -------------------------------------------------------------------------------------


def outcome_model_means(
    logged: pd.DataFrame,
    n_actions: int,
    state_col: str = "state",
    action_col: str = "action",
    reward_col: str = "reward",
    prior_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Shrunk per-``(state, action)`` mean-reward table -- a simple outcome model ``q_hat``.

    .. math:: \hat q(s, a) = \frac{\sum_{i:(s_i,a_i)=(s,a)} r_i + \tau\,\bar r}{n_{s,a} + \tau}

    with global mean :math:`\bar r` and shrinkage :math:`\tau` (``prior_strength``), so
    unseen or thin ``(s, a)`` cells fall back smoothly to the global mean (no NaNs).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(per_row_q, table)`` -- ``per_row_q`` has shape ``(n, A)`` (the row's state
        broadcast over actions) and ``table`` has shape ``(n_states, A)``.
    """
    states = logged[state_col].to_numpy().astype(np.int64)
    actions = logged[action_col].to_numpy().astype(np.int64)
    rewards = logged[reward_col].to_numpy().astype(np.float64)
    n_states = int(states.max()) + 1 if len(states) else 1
    global_mean = float(rewards.mean()) if len(rewards) else 0.0

    sums = np.zeros((n_states, n_actions), dtype=np.float64)
    counts = np.zeros((n_states, n_actions), dtype=np.float64)
    np.add.at(sums, (states, actions), rewards)
    np.add.at(counts, (states, actions), 1.0)
    table = (sums + prior_strength * global_mean) / (counts + prior_strength)
    return table[states], table


# -------------------------------------------------------------------------------------
# one-call wrapper
# -------------------------------------------------------------------------------------


def _extract_mu_matrix(logged: pd.DataFrame, n_actions: int):
    """Full behavior matrix from ``mu_prob_0..A-1`` columns, or ``None`` if absent."""
    cols = [f"mu_prob_{a}" for a in range(n_actions)]
    if all(c in logged.columns for c in cols):
        return logged[cols].to_numpy(dtype=np.float64)
    return None


def _resolve_clip(config: dict) -> float | None:
    ope_cfg = config.get("ope", {}) if config else {}
    if not ope_cfg.get("weight_clip", False):
        return None
    return float(ope_cfg.get("weight_clip_value", DEFAULT_WEIGHT_CLIP))


def evaluate_policy(
    logged: pd.DataFrame,
    target_probs,
    q_hat=None,
    config: dict | None = None,
    n_boot: int = 300,
    seed: int | None = None,
    reward_col: str = "reward",
    action_col: str = "action",
    mu_col: str = "mu_prob",
    episode_col: str = "pa_id",
    step_col: str = "step",
    state_col: str = "state",
    support_threshold: float = 0.01,
    fqe_regressor_factory: Callable[[], object] | None = None,
    verdict_estimators=None,
) -> dict:
    """Run every applicable estimator, bootstrap CIs, and assemble the SPEC ``9`` report.

    Parameters
    ----------
    logged : pandas.DataFrame
        Logged decisions (see the module input contract).
    target_probs : array-like, shape (n, A)
        Per-row target policy ``pi(.|s_t)``.
    q_hat : array-like, shape (n, A), optional
        Outcome model. Required for DM, DR and step-wise DR; when absent those are skipped.
    config : dict, optional
        Study config; its ``ope`` block selects estimators, the alpha grid, weight clipping,
        the report fields and the disagreement switch. Loaded from default when ``None``.
    n_boot : int, optional
        Cluster-bootstrap replicates (default 300).
    seed : int, optional
        Bootstrap / FQE seed (default: config global seed).
    reward_col, action_col, mu_col, episode_col, step_col, state_col : str
        Column names in ``logged`` (defaults match the synthetic fixtures; workstreams pass
        their own, e.g. ``reward_col='R'``, ``action_col='family_idx'``).
    support_threshold : float, optional
        Support floor for the diagnostics.
    fqe_regressor_factory : callable, optional
        Regressor factory for FQE (default gradient boosting).
    verdict_estimators : sequence of str, optional
        Estimators whose agreement drives the verdict. When ``None`` (default) this is chosen
        automatically: :data:`AGREEMENT_ESTIMATORS` (DM/SNIPS/DR) for one-step data,
        :data:`SEQUENTIAL_AGREEMENT` (step-wise DR / FQE) for multi-step data.

    Notes
    -----
    The per-decision estimators DM/IPS/SNIPS/DR are valid return estimators only when every
    episode is a single step; on multi-step data they are **skipped** (step-wise DR and FQE
    evaluate the trajectory return instead). ``report['value']`` therefore comes from the
    plain DR on bandit data and from step-wise DR on sequential data.

    Returns
    -------
    dict
        The standard OPE report: ``estimators`` (per estimator: value, lower_95, ci, se),
        ``diagnostics`` (the :func:`ope_diagnostics` block), ``verdict``
        (``'CONSISTENT'`` / ``'INCONCLUSIVE'``) with ``verdict_detail``, ``report`` (the
        flat SPEC ``9`` headline per ``config['ope']['report']``), ``primary_estimator``,
        ``n`` and ``n_episodes``.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))
    ope_cfg = config.get("ope", {})
    requested = list(ope_cfg.get("estimators", ["DM", "IPS", "SNIPS", "DR", "stepwise_DR", "FQE"]))
    report_fields = list(ope_cfg.get("report", ["value", "lower_95", "ess", "oos_support_frac",
                                                "max_weight", "kl_from_behavior", "tv_from_behavior"]))
    clip = _resolve_clip(config)

    pi = _as_matrix(target_probs, "target_probs")
    n, n_actions = pi.shape
    rewards = logged[reward_col].to_numpy(dtype=np.float64)
    actions = logged[action_col].to_numpy().astype(np.int64)
    mu_taken = logged[mu_col].to_numpy(dtype=np.float64)
    w = _weights(pi, actions, mu_taken)
    episode_id = logged[episode_col].to_numpy() if episode_col in logged.columns else np.arange(n)
    step = logged[step_col].to_numpy() if step_col in logged.columns else np.zeros(n, dtype=np.int64)
    n_episodes = int(pd.unique(episode_id).shape[0])
    # Multi-step episodes -> only the trajectory-level estimators evaluate the return.
    sequential = bool(n_episodes < n)
    q = np.asarray(q_hat, dtype=np.float64) if q_hat is not None else None

    estimators: dict[str, dict] = {}

    def _record(name: str, contrib: np.ndarray, extra: dict | None = None) -> None:
        ci = _bootstrap_ci(contrib, n_boot=n_boot, seed=seed)
        entry = {
            "value": ci["value"],
            "lower_95": ci["lower_95"],
            "ci95": [ci["lo"], ci["hi"]],
            "se": ci["se"],
            "n_boot": ci["n_boot"],
        }
        if extra:
            entry.update(extra)
        estimators[name] = entry

    if not sequential:  # per-decision estimators are return estimators only on one-step data
        if "DM" in requested and q is not None:
            _, contrib = dm(q, pi)
            _record("DM", contrib)
        if "IPS" in requested:
            val_c, contrib_c = ips(rewards, w, clip=clip)
            val_u, _ = ips(rewards, w, clip=None)
            _record("IPS", contrib_c, extra={"value_unclipped": val_u})
        if "SNIPS" in requested:
            val_c, contrib_c = snips(rewards, w, clip=clip)
            val_u, _ = snips(rewards, w, clip=None)
            _record("SNIPS", contrib_c, extra={"value_unclipped": val_u})
        if "DR" in requested and q is not None:
            _, contrib = dr(rewards, w, q, pi, actions)
            _record("DR", contrib)
    if "stepwise_DR" in requested and q is not None:
        _, contrib = stepwise_dr(rewards, w, q, pi, actions, episode_id, step)
        _record("stepwise_DR", contrib)
    if "FQE" in requested and state_col in logged.columns:
        # Pass the wrapper's (possibly synthesised) episode/step arrays so FQE runs even when
        # the frame carries no episode/step columns (a one-step interpretation then applies).
        _, contrib = fqe(
            logged, pi, n_actions,
            state_col=state_col, action_col=action_col, reward_col=reward_col,
            episode_id=episode_id, step=step,
            regressor_factory=fqe_regressor_factory, seed=seed,
        )
        _record("FQE", contrib)

    mu_matrix = _extract_mu_matrix(logged, n_actions)
    if mu_matrix is not None:
        diagnostics = ope_diagnostics(w, mu_matrix, pi, support_threshold=support_threshold, actions=actions)
    else:  # no full behavior matrix -> weight diagnostics only
        diagnostics = {
            "n": int(n),
            "ess": float((w.sum() ** 2) / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0,
            "max_weight": float(w.max()),
            "weight_q99": float(np.quantile(w, 0.99)),
            "weight_q999": float(np.quantile(w, 0.999)),
            "oos_support_frac": None, "mean_kl": None, "mean_tv": None,
            "rare_action_frac": None, "mu_taken_below_support_frac": None,
        }
        diagnostics["ess_frac"] = diagnostics["ess"] / n if n else float("nan")

    if verdict_estimators is None:
        verdict_estimators = SEQUENTIAL_AGREEMENT if sequential else AGREEMENT_ESTIMATORS
    verdict, verdict_detail = _agreement_verdict(estimators, verdict_estimators, ope_cfg)

    primary = _primary_estimator(estimators)
    report = _headline_report(report_fields, estimators.get(primary, {}), diagnostics)

    return {
        "n": int(n),
        "n_episodes": n_episodes,
        "n_actions": int(n_actions),
        "sequential": sequential,
        "primary_estimator": primary,
        "estimators": estimators,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "weight_clip": clip,
        "report": report,
    }


def _primary_estimator(estimators: dict) -> str | None:
    """Choose the headline estimator: DR > stepwise_DR > SNIPS > IPS > DM > FQE.

    DR leads on one-step data; step-wise DR leads on sequential data (where the per-decision
    estimators are absent).
    """
    for name in ("DR", "stepwise_DR", "SNIPS", "IPS", "DM", "FQE"):
        if name in estimators:
            return name
    return None


def _agreement_verdict(estimators: dict, verdict_estimators, ope_cfg: dict) -> tuple[str, dict]:
    """CONSISTENT unless any monitored pair's point estimates differ by more than the wider
    of the two 95% CI half-widths (per config ``inconclusive_if_estimators_disagree``)."""
    enforce = bool(ope_cfg.get("inconclusive_if_estimators_disagree", True))
    present = [e for e in verdict_estimators if e in estimators]
    pairs = []
    disagree = False
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            va, vb = estimators[a]["value"], estimators[b]["value"]
            ha = (estimators[a]["ci95"][1] - estimators[a]["ci95"][0]) / 2.0
            hb = (estimators[b]["ci95"][1] - estimators[b]["ci95"][0]) / 2.0
            tol = max(ha, hb)
            diff = abs(va - vb)
            pair_disagree = bool(diff > tol) if np.isfinite(diff) and np.isfinite(tol) else False
            pairs.append({"pair": [a, b], "abs_diff": float(diff), "tolerance": float(tol),
                          "disagree": pair_disagree})
            disagree = disagree or pair_disagree
    verdict = "INCONCLUSIVE" if (enforce and disagree) else "CONSISTENT"
    return verdict, {"enforced": enforce, "compared": present, "pairs": pairs}


def _headline_report(report_fields, primary_entry: dict, diagnostics: dict) -> dict:
    """Flat SPEC ``9`` headline dict keyed exactly by ``config['ope']['report']``."""
    lookup = {
        "value": primary_entry.get("value"),
        "lower_95": primary_entry.get("lower_95"),
        "ess": diagnostics.get("ess"),
        "oos_support_frac": diagnostics.get("oos_support_frac"),
        "max_weight": diagnostics.get("max_weight"),
        "kl_from_behavior": diagnostics.get("mean_kl"),
        "tv_from_behavior": diagnostics.get("mean_tv"),
    }
    return {f: lookup.get(f) for f in report_fields}


# -------------------------------------------------------------------------------------
# self-test #1: behavior-policy recovery (SPEC 9)
# -------------------------------------------------------------------------------------


def behavior_policy_recovery(
    logged: pd.DataFrame,
    config: dict | None = None,
    n_actions: int | None = None,
    seed: int | None = None,
    n_boot: int = 300,
    tol: float | None = None,
    reward_col: str = "reward",
    action_col: str = "action",
    mu_col: str = "mu_prob",
    state_col: str = "state",
    episode_col: str = "pa_id",
    step_col: str = "step",
) -> dict:
    """SPEC ``9`` self-test #1: set the target equal to the behavior policy; every estimator
    must recover the observed held-out mean reward.

    The episodes are split 50/50 by seed; the outcome model ``q_hat`` for DM/DR is fit on the
    first half and applied to the held-out second half, on which the target is set to that
    half's own logged behavior distribution. Because ``pi == mu`` there, the importance
    weights are exactly 1 and IPS returns the held-out mean reward identically; every other
    estimator must land within its bootstrap CI of that mean.

    Parameters
    ----------
    logged : pandas.DataFrame
        Logged decisions with a full behavior matrix ``mu_prob_0..A-1``.
    config : dict, optional
        Study config (for the estimator list / seed).
    n_actions : int, optional
        Action count; inferred from the ``mu_prob_*`` columns when ``None``.
    seed : int, optional
        Split / bootstrap seed (default: config global seed).
    n_boot : int, optional
        Bootstrap replicates.
    tol : float, optional
        Absolute tolerance on ``|value - observed|`` (default: ``5 * SE`` of the held-out
        mean reward). The primary criterion is CI containment; ``tol`` is a secondary guard.

    Returns
    -------
    dict
        ``observed_mean``, per-estimator ``{value, abs_err, within_ci}``,
        ``ips_weights_unit`` (max ``|w - 1|`` below 1e-9), ``passed`` and ``tol``.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))
    if n_actions is None:
        n_actions = sum(c.startswith("mu_prob_") and c[len("mu_prob_"):].isdigit() for c in logged.columns)
    mu_cols = [f"mu_prob_{a}" for a in range(n_actions)]
    if not all(c in logged.columns for c in mu_cols):
        raise ValueError("behavior_policy_recovery needs the full behavior matrix mu_prob_0..A-1")

    logged = logged.reset_index(drop=True)
    episodes = pd.unique(logged[episode_col].to_numpy())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    cut = len(shuffled) // 2
    fit_eps = set(shuffled[:cut].tolist())
    is_fit = logged[episode_col].isin(fit_eps).to_numpy()
    fit_df = logged.loc[is_fit].reset_index(drop=True)
    eval_df = logged.loc[~is_fit].reset_index(drop=True)

    # Outcome model fit on the fit-half, applied to the eval-half.
    _, table = outcome_model_means(fit_df, n_actions, state_col=state_col,
                                   action_col=action_col, reward_col=reward_col)
    eval_states = eval_df[state_col].to_numpy().astype(np.int64)
    # Guard against eval states unseen in the fit half (rows beyond the fit table).
    if table.shape[0] <= int(eval_states.max()):
        pad = np.full((int(eval_states.max()) + 1 - table.shape[0], n_actions),
                      float(fit_df[reward_col].mean()))
        table = np.vstack([table, pad])
    q_hat_eval = table[eval_states]

    target = eval_df[mu_cols].to_numpy(dtype=np.float64)  # target == behavior on the eval half
    # The value to recover is the held-out mean *return*: mean reward per decision on bandit
    # (one-step) data, mean summed reward per episode on multi-step data.
    n_eval_ep = int(eval_df[episode_col].nunique())
    if n_eval_ep < len(eval_df):  # sequential
        returns = eval_df.groupby(episode_col)[reward_col].sum()
        observed = float(returns.mean())
        obs_sd = float(returns.std(ddof=1))
        obs_n = len(returns)
    else:
        observed = float(eval_df[reward_col].mean())
        obs_sd = float(eval_df[reward_col].std(ddof=1))
        obs_n = len(eval_df)
    if tol is None:
        tol = 5.0 * obs_sd / np.sqrt(obs_n)

    result = evaluate_policy(
        eval_df, target, q_hat=q_hat_eval, config=config, n_boot=n_boot, seed=seed,
        reward_col=reward_col, action_col=action_col, mu_col=mu_col,
        episode_col=episode_col, step_col=step_col, state_col=state_col,
    )

    # IPS weights must be exactly 1 (pi == mu on the eval half).
    w = _weights(target, eval_df[action_col].to_numpy().astype(np.int64),
                 eval_df[mu_col].to_numpy(dtype=np.float64))
    ips_unit = bool(np.max(np.abs(w - 1.0)) < 1e-9)

    per_est = {}
    all_ok = ips_unit
    for name, entry in result["estimators"].items():
        val = entry["value"]
        lo, hi = entry["ci95"]
        within = bool(lo <= observed <= hi) if np.isfinite(lo) and np.isfinite(hi) else False
        abs_err = float(abs(val - observed))
        ok = within or (abs_err <= tol)
        per_est[name] = {"value": float(val), "abs_err": abs_err, "within_ci": within, "ok": ok}
        all_ok = all_ok and ok

    return {
        "observed_mean": observed,
        "n_eval": int(len(eval_df)),
        "estimators": per_est,
        "ips_weights_unit": ips_unit,
        "tol": float(tol),
        "passed": bool(all_ok),
    }
