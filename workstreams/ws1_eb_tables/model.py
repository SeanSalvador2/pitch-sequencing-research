r"""Empirical-Bayes conditional tables for pitch selection and run value (SPEC ``12.1``).

This module implements the WS1 machinery: hierarchical shrinkage tables over the decision
table's context / history keys, with per-level concentrations fitted by empirical Bayes and
posterior uncertainty reported alongside the point estimates (decisions D25-D28).

Two table families share one key ladder
=======================================

The conditioning key deepens per view (decision **D25**). Writing
``ch = (balls, strikes, stand, p_throws)`` for the count x handedness cell and ``pit`` for
the pitcher id, each view's **deepest** cell is:

======  ==========================================  ========================================
view    selection key (next-pitch family)           history component
======  ==========================================  ========================================
``C``   ``(ch, pit)``                               none
``L1``  ``(ch, pit, prev_family)``                  the immediately preceding family
``U``   ``(ch, pit, multiset_sig)``                 order-invariant multiset of prior families
``O``   ``(ch, pit, (prev2, prev1))``               the ordered last-two prior families
``OM``  -- infeasible (see :func:`support_note`)    matchup memory cannot be tabulated
======  ==========================================  ========================================

The run-value tables use the **same** ladder but condition on the current action as well
(decision **D22**: outcome models condition on the pitch actually thrown), so every
run-value cell carries the current ``family`` as an extra key coordinate, and the estimand
is ``E[R | key cell, family]``.

Selection: hierarchical Dirichlet-multinomial (decision D26)
============================================================

Level ``l`` has a single shared concentration ``alpha_l``. A cell ``c`` with observed family
counts ``n_c = (n_{c,1}, ..., n_{c,8})`` and parent posterior mean ``pi_{parent(c)}``
(a distribution over the 8 families) has the closed-form Dirichlet posterior mean

.. math:: \hat p_c = \frac{n_c + \alpha_l\, \pi_{\mathrm{parent}(c)}}{N_c + \alpha_l},
          \qquad N_c = \sum_k n_{c,k}.

An **unseen** cell (``N_c = 0``) returns exactly the parent mean -- i.e. it *backs off* one
level up -- so at prediction time an unseen key resolves to the posterior mean of its
deepest observed ancestor. The posterior effective sample size is ``ESS_c = N_c + alpha_l``
and the per-family posterior variance is

.. math:: \mathrm{Var}(p_{c,k}) = \frac{\hat p_{c,k}\,(1 - \hat p_{c,k})}{N_c + \alpha_l + 1}.

Run value: hierarchical normal partial pooling (decision D26)
=============================================================

Level ``l`` has a single shared shrinkage strength ``kappa_l`` (a prior sample size). A cell
``c`` with ``n_c`` observations, sample mean ``\bar R_c`` and parent posterior mean
``m_{parent(c)}`` has the normal-normal posterior mean

.. math:: \hat m_c = \frac{n_c}{n_c + \kappa_l}\, \bar R_c
                     + \frac{\kappa_l}{n_c + \kappa_l}\, m_{\mathrm{parent}(c)},

again exactly the parent when ``n_c = 0``. With a pooled within-cell residual variance
``sigma^2`` the posterior standard deviation of the cell mean is

.. math:: \mathrm{sd}(m_c) = \frac{\sigma}{\sqrt{n_c + \kappa_l}}.

Fitting the concentrations (empirical Bayes, level by level)
============================================================

Each level's ``alpha_l`` / ``kappa_l`` is fitted **conditional on the posterior means of the
level above** (a top-down / level-wise empirical-Bayes pass), pooling the exchangeable cells
at that level. The primary estimator is the maximum marginal likelihood (the
Dirichlet-multinomial / normal-normal evidence, a smooth 1-D optimisation via
:mod:`scipy.optimize`); a documented method-of-moments estimator is the fallback when the
optimiser fails or a level is too sparse to identify the concentration. The fitted value and
the estimator actually used are recorded per level.

Everything is built from the shared decision table's existing columns (``pa_id``,
``pitch_number``, ``family``, count / handedness, ``pitcher``, ``R``); no history logic is
re-implemented here beyond forming the tabular keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from pitchseq.families import FAMILIES

__all__ = [
    "EBModel",
    "fit",
    "SELECTION_VIEWS",
    "dirichlet_posterior_mean",
    "normal_posterior_mean",
    "fit_dirichlet_concentration",
    "fit_normal_kappa",
    "count_hand_code",
    "history_signature",
    "support_table",
    "support_note",
]

# --- fixed encodings -------------------------------------------------------------------

_FAM_INDEX = {f: i for i, f in enumerate(FAMILIES)}
_N_FAM = len(FAMILIES)
_NONE_CODE = _N_FAM  # sentinel family code for "no prior pitch" (0..7 are the families)
_MULTISET_CAP = 6  # cap the U multiset signature at the PA's first 6 prior pitches (D25)

#: Views for which a pure conditional table is feasible (OM is not -- see support_note).
SELECTION_VIEWS = ("C", "U", "L1", "O")

# Concentration search bounds (the optimiser works on log10 of the concentration).
_ALPHA_MIN = 1e-3
_LOG10_LO = -3.0
_ROOT_ALPHA_DEFAULT = 1.0  # weak symmetric prior at the single-cell root (data swamps it)

#: Concentration ceiling per target. Both targets fit each level's concentration by maximum
#: marginal likelihood, but the fit is bounded above. Selection uses an effectively
#: unbounded ceiling because next-pitch prediction from flat conditional tables genuinely
#: gains almost nothing from within-PA history -- the MML pools the weak selection-order
#: signal toward the parent, which is the honest WS1 result (a flat table cannot resolve the
#: subtle ordering that WS2/WS3 are built to find). Run value uses a *moderate* ceiling so a
#: well-supported cell keeps some of its own reward signal instead of degenerating to its
#: parent: the run-value tables are the estimand later prescriptive workstreams consume
#: (D26), and the WS1 support exhibit needs the O table to express the family-pair reward
#: differences a planted ordered effect induces (D27). Both ceilings are reported per run.
MAX_CONCENTRATION = {"selection": 1e6, "run_value": 200.0}
_ALPHA_MAX = 1e6  # absolute clip for the method-of-moments fallback


# =======================================================================================
# Pure shrinkage math (documented formulas; exercised directly by the unit tests)
# =======================================================================================

def dirichlet_posterior_mean(counts, parent, alpha: float) -> np.ndarray:
    r"""Dirichlet posterior mean of a family distribution shrunk toward a parent.

    .. math:: \hat p = \frac{n + \alpha\,\pi}{\sum_k n_k + \alpha}

    Parameters
    ----------
    counts : array-like, shape (K,)
        Observed family counts ``n`` for the cell.
    parent : array-like, shape (K,)
        Parent posterior mean ``pi`` (a distribution, sums to 1, strictly positive).
    alpha : float
        Level concentration (prior pseudo-count strength).

    Returns
    -------
    numpy.ndarray, shape (K,)
        The posterior mean. Exactly ``parent`` when ``counts`` sums to 0.
    """
    counts = np.asarray(counts, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    return (counts + alpha * parent) / (counts.sum() + alpha)


def normal_posterior_mean(n: float, cell_mean: float, parent_mean: float, kappa: float) -> float:
    r"""Normal-normal partial-pooling posterior mean.

    .. math:: \hat m = \frac{n}{n + \kappa}\,\bar R + \frac{\kappa}{n + \kappa}\, m_{\pi}

    Returns exactly ``parent_mean`` when ``n = 0``.
    """
    if n <= 0:
        return float(parent_mean)
    w = n / (n + kappa)
    return float(w * cell_mean + (1.0 - w) * parent_mean)


def _dirichlet_neg_log_evidence(alpha: float, counts: np.ndarray, parents: np.ndarray) -> float:
    r"""Negative Dirichlet-multinomial log evidence summed over a level's cells.

    For cell ``c`` with prior ``Dir(alpha * pi_c)`` and counts ``n_c`` (``N_c = sum n_c``),
    the Pólya marginal likelihood (dropping the ``alpha``-independent multinomial
    coefficient) is

    .. math:: \log \Gamma(\alpha) - \log \Gamma(\alpha + N_c)
              + \sum_k \big[\log \Gamma(\alpha \pi_{c,k} + n_{c,k})
                            - \log \Gamma(\alpha \pi_{c,k})\big].
    """
    N = counts.sum(axis=1)
    a = alpha * parents
    term = gammaln(alpha) - gammaln(alpha + N) + (gammaln(a + counts) - gammaln(a)).sum(axis=1)
    return -float(term.sum())


def _dirichlet_mom(counts: np.ndarray, parents: np.ndarray, max_conc: float = _ALPHA_MAX) -> float:
    r"""Method-of-moments concentration from Pearson overdispersion of a level's cells.

    The Dirichlet-multinomial inflates the multinomial Pearson statistic
    ``X_c = sum_k (n_{c,k} - N_c pi_{c,k})^2 / (N_c pi_{c,k})`` in expectation by a factor
    ``1 + (N_c - 1) / (alpha + 1)`` over the multinomial baseline ``(K - 1)``. Matching the
    pooled statistic ``S = sum_c X_c`` to its expectation and solving for ``alpha`` gives

    .. math:: \alpha = \frac{\sum_c (N_c - 1)}{S / (K - 1) - C} - 1,

    where ``C`` is the number of cells. Under-dispersion (non-positive denominator) maps to
    heavy pooling (``alpha = alpha_max``). All parents are strictly positive, so ``X_c`` is
    always finite.
    """
    N = counts.sum(axis=1)
    use = N > 0
    counts, parents, N = counts[use], parents[use], N[use]
    if len(N) == 0:
        return _ROOT_ALPHA_DEFAULT
    expected = N[:, None] * parents
    X = ((counts - expected) ** 2 / expected).sum(axis=1)
    S = float(X.sum())
    C = len(N)
    M = float((N - 1).sum())
    K = counts.shape[1]
    denom = S / (K - 1) - C
    if denom <= 0 or M <= 0:
        return max_conc
    return float(np.clip(M / denom - 1.0, _ALPHA_MIN, max_conc))


def fit_dirichlet_concentration(
    counts: np.ndarray, parents: np.ndarray, max_concentration: float = _ALPHA_MAX
) -> tuple[float, str]:
    """Fit a level's Dirichlet concentration by maximum marginal likelihood (MoM fallback).

    Parameters
    ----------
    counts : numpy.ndarray, shape (C, K)
        Per-cell family counts for the ``C`` cells at this level.
    parents : numpy.ndarray, shape (C, K)
        Per-cell parent posterior means (strictly positive rows summing to 1).
    max_concentration : float, optional
        Upper bound on the fitted concentration (the ceiling of :data:`MAX_CONCENTRATION`).

    Returns
    -------
    (float, str)
        The fitted concentration and the estimator used (``"mml"``, ``"mom"`` or
        ``"default"``). Levels with fewer than two informative cells fall back to a weak
        default, as a single cell cannot identify a shared concentration.
    """
    counts = np.asarray(counts, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.float64)
    informative = (counts.sum(axis=1) > 0).sum()
    if informative < 2:
        return min(_ROOT_ALPHA_DEFAULT, max_concentration), "default"
    log_hi = np.log10(max_concentration)

    def objective(log10_alpha: float) -> float:
        return _dirichlet_neg_log_evidence(10.0 ** log10_alpha, counts, parents)

    try:
        res = minimize_scalar(objective, bounds=(_LOG10_LO, log_hi), method="bounded")
        alpha = float(10.0 ** res.x)
        if res.success and np.isfinite(alpha) and _ALPHA_MIN < alpha <= max_concentration:
            return alpha, "mml"
    except (ValueError, FloatingPointError):  # pragma: no cover - defensive
        pass
    return _dirichlet_mom(counts, parents, max_concentration), "mom"


def _normal_neg_log_likelihood(
    kappa: float, n: np.ndarray, mean: np.ndarray, parent: np.ndarray, sigma2: float
) -> float:
    r"""Negative normal-normal marginal log likelihood of a level's cell means.

    With between-cell variance ``tau^2 = sigma^2 / kappa`` the sample mean of cell ``c`` is
    marginally ``N(m_{parent(c)}, tau^2 + sigma^2 / n_c)``, giving

    .. math:: \tfrac12 \sum_c \Big[\log\!\big(2\pi(\tau^2 + \sigma^2/n_c)\big)
              + \frac{(\bar R_c - m_{\pi,c})^2}{\tau^2 + \sigma^2/n_c}\Big].
    """
    tau2 = sigma2 / kappa
    v = tau2 + sigma2 / n
    return 0.5 * float((np.log(2.0 * np.pi * v) + (mean - parent) ** 2 / v).sum())


def _normal_mom(
    n: np.ndarray, mean: np.ndarray, parent: np.ndarray, sigma2: float, max_conc: float = _ALPHA_MAX
) -> float:
    r"""DerSimonian-Laird-style moment shrinkage strength for a level.

    Matches the mean squared cell-to-parent deviation to its expectation
    ``tau^2 + sigma^2/n_c`` to estimate the between-cell variance
    ``tau^2 = mean_c[(\bar R_c - m_{\pi,c})^2 - sigma^2/n_c]`` (floored at a small positive
    value), then returns ``kappa = sigma^2 / tau^2``.
    """
    resid2 = (mean - parent) ** 2 - sigma2 / n
    tau2 = float(resid2.mean())
    if not np.isfinite(tau2) or tau2 <= 0:
        return max_conc
    return float(np.clip(sigma2 / tau2, _ALPHA_MIN, max_conc))


def fit_normal_kappa(
    n: np.ndarray,
    cell_means: np.ndarray,
    parent_means: np.ndarray,
    sigma2: float,
    max_concentration: float = _ALPHA_MAX,
) -> tuple[float, str]:
    """Fit a level's normal shrinkage strength by maximum marginal likelihood (MoM fallback).

    Parameters
    ----------
    n : numpy.ndarray, shape (C,)
        Per-cell observation counts (only cells with ``n >= 1`` should be passed).
    cell_means : numpy.ndarray, shape (C,)
        Per-cell sample means ``\\bar R_c``.
    parent_means : numpy.ndarray, shape (C,)
        Per-cell parent posterior means.
    sigma2 : float
        Pooled within-cell residual variance.
    max_concentration : float, optional
        Upper bound on the fitted shrinkage strength (the ceiling of
        :data:`MAX_CONCENTRATION`).

    Returns
    -------
    (float, str)
        The fitted shrinkage strength and the estimator used (``"mml"`` / ``"mom"`` /
        ``"default"``).
    """
    n = np.asarray(n, dtype=np.float64)
    cell_means = np.asarray(cell_means, dtype=np.float64)
    parent_means = np.asarray(parent_means, dtype=np.float64)
    if len(n) < 2 or sigma2 <= 0:
        return min(_ROOT_ALPHA_DEFAULT, max_concentration), "default"
    log_hi = np.log10(max_concentration)

    def objective(log10_kappa: float) -> float:
        return _normal_neg_log_likelihood(10.0 ** log10_kappa, n, cell_means, parent_means, sigma2)

    try:
        res = minimize_scalar(objective, bounds=(_LOG10_LO, log_hi), method="bounded")
        kappa = float(10.0 ** res.x)
        if res.success and np.isfinite(kappa) and _ALPHA_MIN < kappa <= max_concentration:
            return kappa, "mml"
    except (ValueError, FloatingPointError):  # pragma: no cover - defensive
        pass
    return _normal_mom(n, cell_means, parent_means, sigma2, max_concentration), "mom"


# =======================================================================================
# Tabular keys (built from the decision table's existing columns)
# =======================================================================================

def _family_codes(table: pd.DataFrame) -> np.ndarray:
    """Integer family code (0..7) per row from the decision table's ``family`` column."""
    fam = table["family"].astype("object").to_numpy()
    codes = np.full(len(table), _FAM_INDEX.get("XX", _N_FAM - 1), dtype=np.int64)
    for f, i in _FAM_INDEX.items():
        codes[fam == f] = i
    return codes


def count_hand_code(table: pd.DataFrame) -> np.ndarray:
    """Table-independent integer code for the ``(balls, strikes, stand, p_throws)`` cell.

    The encoding is fixed (not data-derived) so a cell has the same code in the training and
    prediction tables: ``(((balls * 3 + strikes) * 2 + stand_is_R) * 2 + p_throws_is_R)``.
    """
    balls = pd.to_numeric(table["balls"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    strikes = pd.to_numeric(table["strikes"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    stand_r = (table["stand"].astype("object").to_numpy() == "R").astype(np.int64)
    throw_r = (table["p_throws"].astype("object").to_numpy() == "R").astype(np.int64)
    return ((balls * 3 + strikes) * 2 + stand_r) * 2 + throw_r


def _prev_family_codes(table: pd.DataFrame, lag: int) -> np.ndarray:
    """Family code of the pitch ``lag`` positions before each row within its PA.

    ``_NONE_CODE`` where no such prior pitch exists. Uses the shared ``pa_id`` /
    ``pitch_number`` ordering (no re-implementation of history logic).
    """
    order = np.lexsort((table["pitch_number"].to_numpy(), table["pa_id"].to_numpy()))
    fam = _family_codes(table)
    pa = table["pa_id"].to_numpy()
    fam_s = fam[order]
    pa_s = pa[order]
    out_s = np.full(len(table), _NONE_CODE, dtype=np.int64)
    if lag < len(table):
        same_pa = pa_s[lag:] == pa_s[:-lag]
        vals = np.where(same_pa, fam_s[:-lag], _NONE_CODE)
        out_s[lag:] = vals
    out = np.empty(len(table), dtype=np.int64)
    out[order] = out_s
    return out


def _multiset_signature(table: pd.DataFrame) -> np.ndarray:
    r"""Order-invariant signature of the PA's first ``<= 6`` prior families (decision D25).

    For a pitch at position ``t`` the signature is the per-family count vector of the
    families thrown at absolute positions ``1 .. min(t - 1, 6)`` of the plate appearance,
    encoded as a single base-7 integer ``sum_k c_k 7^k`` (each count is ``0..6``, so the
    encoding is injective). It is invariant to the order of the prior pitches -- the
    unordered-collection question the U view isolates -- and caps at the first six priors so
    long plate appearances do not fragment support without bound.
    """
    n = len(table)
    pa = table["pa_id"].to_numpy()
    pitch_no = pd.to_numeric(table["pitch_number"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    fam = _family_codes(table)  # positional family codes (aligned to row order)

    counts = np.zeros((n, _N_FAM), dtype=np.int64)
    for pos in range(1, _MULTISET_CAP + 1):
        sel = pitch_no == pos
        if not sel.any():
            continue
        # Family thrown at absolute position ``pos`` within each PA (positional; no reliance
        # on the frame's index), mapped onto every row via its ``pa_id``.
        at_pos = pd.Series(fam[sel], index=pa[sel])
        at_pos = at_pos[~at_pos.index.duplicated()]
        mapped = pd.Series(pa).map(at_pos).to_numpy()
        present = ~np.isnan(mapped) & (pos < pitch_no)
        rows = np.flatnonzero(present)
        counts[rows, mapped[rows].astype(np.int64)] += 1
    powers = (7 ** np.arange(_N_FAM)).astype(np.int64)
    return counts @ powers


def history_signature(table: pd.DataFrame, view: str) -> np.ndarray:
    """Integer history-key coordinate for a view (``0`` for ``C`` -- no history).

    * ``L1`` -- previous family code (``0..8``).
    * ``U``  -- base-7 multiset signature (see :func:`_multiset_signature`).
    * ``O``  -- ordered pair ``prev2 * 9 + prev1`` of the last two prior family codes.
    """
    if view == "C":
        return np.zeros(len(table), dtype=np.int64)
    if view == "L1":
        return _prev_family_codes(table, 1)
    if view == "U":
        return _multiset_signature(table)
    if view == "O":
        prev1 = _prev_family_codes(table, 1)
        prev2 = _prev_family_codes(table, 2)
        return prev2 * (_N_FAM + 1) + prev1
    raise ValueError(f"unknown view {view!r}; expected one of {SELECTION_VIEWS}")


def _object_array(values) -> np.ndarray:
    """A 1-D object array of tuples (avoids numpy's 2-D coercion of a list of tuples)."""
    vals = list(values)
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr


def _level_keys(table: pd.DataFrame, view: str, target: str) -> list[np.ndarray]:
    """Per-row cell keys for every hierarchy level (root first, deepest last).

    Selection ladder: ``() -> (ch,) -> (ch, pit) -> (ch, pit, hist)``.
    Run-value ladder (conditions on the current family, D22):
    ``() -> (ch, fam) -> (ch, fam, pit) -> (ch, fam, pit, hist)``.
    The history level is present only for ``view != "C"``.
    """
    n = len(table)
    ch = count_hand_code(table)
    pit = pd.to_numeric(table["pitcher"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    root = _object_array([()] * n)

    if target == "selection":
        levels = [root, _object_array(zip(ch.tolist())), _object_array(zip(ch.tolist(), pit.tolist()))]
        if view != "C":
            hist = history_signature(table, view)
            levels.append(_object_array(zip(ch.tolist(), pit.tolist(), hist.tolist())))
        return levels

    fam = _family_codes(table)
    levels = [
        root,
        _object_array(zip(ch.tolist(), fam.tolist())),
        _object_array(zip(ch.tolist(), fam.tolist(), pit.tolist())),
    ]
    if view != "C":
        hist = history_signature(table, view)
        levels.append(_object_array(zip(ch.tolist(), fam.tolist(), pit.tolist(), hist.tolist())))
    return levels


# =======================================================================================
# The hierarchical table (one view, one target)
# =======================================================================================

@dataclass
class _Level:
    """One fitted level of the hierarchy."""

    name: str
    keys: dict = field(default_factory=dict)  # key tuple -> cell index
    post_mean: np.ndarray = None  # (C, K) selection distribution, or (C,) run-value mean
    ess: np.ndarray = None  # (C,) effective sample size (counts + concentration)
    train_n: np.ndarray = None  # (C,) observed pitch count per cell
    concentration: float = np.nan  # alpha (selection) / kappa (run value)
    method: str = ""  # estimator used


_LEVEL_NAMES = {
    "selection": ["global", "count_hand", "pitcher", "history"],
    "run_value": ["global", "count_hand_family", "pitcher_family", "history"],
}


class EBModel:
    """A fitted empirical-Bayes table for one state view and one target.

    Use :func:`fit` to construct one. For ``target="selection"`` :meth:`predict` returns an
    ``(n, 8)`` matrix of family probabilities (posterior means); for ``target="run_value"``
    it returns ``(exp_reward, exp_reward_sd)`` per row. Unseen keys back off up the hierarchy
    exactly as the shrinkage implies, and :attr:`backoff_counts_` records how often each
    level was the deepest observed ancestor on the most recent prediction.

    Attributes
    ----------
    view : str
        The state view (one of :data:`SELECTION_VIEWS`).
    target : str
        ``"selection"`` or ``"run_value"``.
    concentrations_ : dict
        Fitted ``alpha`` (selection) or ``kappa`` (run value) per level name.
    concentration_methods_ : dict
        The estimator used per level (``"mml"`` / ``"mom"`` / ``"default"``).
    sigma2_ : float
        Pooled within-cell residual variance (run value only; ``nan`` for selection).
    backoff_counts_ : dict
        Level-name -> number of prediction rows resolved at that level (deepest observed
        ancestor) on the most recent :meth:`predict` call.
    n_cells_ : dict
        Level-name -> number of fitted cells.
    """

    def __init__(self, view: str, target: str, max_concentration: float | None = None) -> None:
        if target not in ("selection", "run_value"):
            raise ValueError(f"target must be 'selection' or 'run_value', got {target!r}")
        self.view = view
        self.target = target
        self.max_concentration = float(
            MAX_CONCENTRATION[target] if max_concentration is None else max_concentration
        )
        self._levels: list[_Level] = []
        self.sigma2_ = float("nan")
        self.backoff_counts_: dict[str, int] = {}

    # --- fitting -----------------------------------------------------------------------

    def fit(self, table: pd.DataFrame) -> "EBModel":
        """Fit the hierarchy on a training decision table."""
        if self.view == "OM":
            raise NotImplementedError(_OM_MESSAGE)
        if self.view not in SELECTION_VIEWS:
            raise ValueError(f"unknown view {self.view!r}; expected one of {SELECTION_VIEWS} or 'OM'")
        level_keys = _level_keys(table, self.view, self.target)
        names = _LEVEL_NAMES[self.target][: len(level_keys)]
        if self.target == "selection":
            self._fit_selection(table, level_keys, names)
        else:
            self._fit_run_value(table, level_keys, names)
        return self

    def _factorize(self, level_keys):
        """Factorise each level's per-row keys; return (codes, uniques, parent_code) lists."""
        codes, uniques, parent_code = [], [], []
        for i, keys in enumerate(level_keys):
            c, u = pd.factorize(keys, sort=False)
            codes.append(c)
            uniques.append(u)
            if i == 0:
                parent_code.append(None)
            else:
                rep = pd.Series(np.arange(len(keys))).groupby(c).first().to_numpy()
                parent_code.append(codes[i - 1][rep])  # each cell's parent cell index
        return codes, uniques, parent_code

    def _fit_selection(self, table, level_keys, names) -> None:
        fam_code = _family_codes(table)
        codes, uniques, parent_code = self._factorize(level_keys)
        uniform = np.full(_N_FAM, 1.0 / _N_FAM)

        self._levels = []
        prev_mean = None
        for i, name in enumerate(names):
            C = len(uniques[i])
            counts = np.zeros((C, _N_FAM), dtype=np.float64)
            np.add.at(counts, (codes[i], fam_code), 1.0)
            N = counts.sum(axis=1)
            if i == 0:
                parents = np.tile(uniform, (C, 1))
                alpha, method = _ROOT_ALPHA_DEFAULT, "default"
            else:
                parents = prev_mean[parent_code[i]]
                alpha, method = fit_dirichlet_concentration(counts, parents, self.max_concentration)
            post = (counts + alpha * parents) / (N[:, None] + alpha)
            level = _Level(
                name=name,
                keys=dict(zip(uniques[i].tolist(), range(C))),
                post_mean=post,
                ess=N + alpha,
                train_n=N,
                concentration=alpha,
                method=method,
            )
            self._levels.append(level)
            prev_mean = post

    def _fit_run_value(self, table, level_keys, names) -> None:
        R = pd.to_numeric(table["R"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(R)
        self.sigma2_ = self._pooled_sigma2(table, R, finite)
        codes, uniques, parent_code = self._factorize(level_keys)
        grand_mean = float(R[finite].mean()) if finite.any() else 0.0

        self._levels = []
        prev_mean = None
        for i, name in enumerate(names):
            C = len(uniques[i])
            n = np.zeros(C, dtype=np.float64)
            sum_r = np.zeros(C, dtype=np.float64)
            np.add.at(n, codes[i][finite], 1.0)
            np.add.at(sum_r, codes[i][finite], R[finite])
            with np.errstate(invalid="ignore"):
                cell_mean = np.where(n > 0, sum_r / np.where(n > 0, n, 1.0), np.nan)
            if i == 0:
                parents = np.full(C, grand_mean)
                kappa, method = _ROOT_ALPHA_DEFAULT, "default"
                post = np.full(C, grand_mean)
            else:
                parents = prev_mean[parent_code[i]]
                seen = n > 0
                kappa, method = fit_normal_kappa(
                    n[seen], cell_mean[seen], parents[seen], self.sigma2_, self.max_concentration
                )
                w = np.where(n > 0, n / (n + kappa), 0.0)
                post = np.where(n > 0, w * np.nan_to_num(cell_mean) + (1.0 - w) * parents, parents)
            level = _Level(
                name=name,
                keys=dict(zip(uniques[i].tolist(), range(C))),
                post_mean=post,
                ess=n + kappa,
                train_n=n,
                concentration=kappa,
                method=method,
            )
            self._levels.append(level)
            prev_mean = post

    def _pooled_sigma2(self, table, R, finite) -> float:
        """Pooled within-``(count_hand, family)`` residual variance of the reward."""
        ch = count_hand_code(table)
        fam = _family_codes(table)
        key = ch.astype(np.int64) * (_N_FAM + 1) + fam
        df = pd.DataFrame({"key": key[finite], "R": R[finite]})
        grp = df.groupby("key")["R"]
        var = grp.var(ddof=1)
        cnt = grp.count()
        mask = cnt >= 2
        if mask.any():
            num = float((var[mask] * (cnt[mask] - 1)).sum())
            den = float((cnt[mask] - 1).sum())
            if den > 0 and num > 0:
                return num / den
        return float(np.var(R[finite])) if finite.any() else 1.0

    # --- prediction --------------------------------------------------------------------

    def _assign(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Deepest observed ancestor per row: returns (level_index, cell_index) arrays."""
        level_keys = _level_keys(table, self.view, self.target)
        n = len(table)
        used_level = np.full(n, -1, dtype=np.int64)
        used_cell = np.full(n, -1, dtype=np.int64)
        for i in range(len(self._levels) - 1, -1, -1):
            todo = np.flatnonzero(used_level < 0)
            if len(todo) == 0:
                break
            mapping = self._levels[i].keys
            mapped = pd.Series(level_keys[i][todo]).map(mapping).to_numpy()
            found = ~pd.isna(mapped)
            rows = todo[found]
            used_level[rows] = i
            used_cell[rows] = mapped[found].astype(np.int64)
        return used_level, used_cell

    def _record_backoff(self, used_level: np.ndarray) -> None:
        self.backoff_counts_ = {
            lvl.name: int(np.sum(used_level == i)) for i, lvl in enumerate(self._levels)
        }

    def predict(self, table: pd.DataFrame):
        """Predict for a table.

        Returns
        -------
        numpy.ndarray or tuple
            ``(n, 8)`` family probabilities for ``target="selection"``; the pair
            ``(exp_reward, exp_reward_sd)`` of ``(n,)`` arrays for ``target="run_value"``.
        """
        if self.target == "selection":
            return self.predict_proba(table)
        return self.predict_run_value(table)

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        """Posterior-mean family probabilities ``(n, 8)`` (selection target)."""
        used_level, used_cell = self._assign(table)
        self._record_backoff(used_level)
        out = np.empty((len(table), _N_FAM), dtype=np.float64)
        for i, level in enumerate(self._levels):
            rows = np.flatnonzero(used_level == i)
            if len(rows):
                out[rows] = level.post_mean[used_cell[rows]]
        return out

    def predict_run_value(self, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Posterior-mean reward and its posterior sd ``(exp_reward, exp_reward_sd)``."""
        used_level, used_cell = self._assign(table)
        self._record_backoff(used_level)
        mean = np.empty(len(table), dtype=np.float64)
        ess = np.empty(len(table), dtype=np.float64)
        for i, level in enumerate(self._levels):
            rows = np.flatnonzero(used_level == i)
            if len(rows):
                mean[rows] = level.post_mean[used_cell[rows]]
                ess[rows] = level.ess[used_cell[rows]]
        sd = np.sqrt(self.sigma2_ / np.maximum(ess, 1e-12))
        return mean, sd

    def posterior_stats(self, table: pd.DataFrame) -> dict:
        """Per-row posterior summaries for inspection / downstream uncertainty use.

        Returns a dict with ``ess`` (effective sample size of the resolved cell),
        ``backoff_level`` (the deepest observed ancestor's level index) and, for selection,
        ``proba`` plus per-family posterior ``variance`` ``p(1-p)/(ESS+1)``.
        """
        used_level, used_cell = self._assign(table)
        self._record_backoff(used_level)
        ess = np.empty(len(table), dtype=np.float64)
        for i, level in enumerate(self._levels):
            rows = np.flatnonzero(used_level == i)
            if len(rows):
                ess[rows] = level.ess[used_cell[rows]]
        out = {"ess": ess, "backoff_level": used_level}
        if self.target == "selection":
            proba = np.empty((len(table), _N_FAM), dtype=np.float64)
            for i, level in enumerate(self._levels):
                rows = np.flatnonzero(used_level == i)
                if len(rows):
                    proba[rows] = level.post_mean[used_cell[rows]]
            out["proba"] = proba
            out["variance"] = proba * (1.0 - proba) / (ess[:, None] + 1.0)
        else:
            mean, sd = self.predict_run_value(table)
            out["exp_reward"] = mean
            out["exp_reward_sd"] = sd
        return out

    # --- reporting ---------------------------------------------------------------------

    @property
    def concentrations_(self) -> dict:
        return {lvl.name: float(lvl.concentration) for lvl in self._levels}

    @property
    def concentration_methods_(self) -> dict:
        return {lvl.name: lvl.method for lvl in self._levels}

    @property
    def n_cells_(self) -> dict:
        return {lvl.name: int(len(lvl.keys)) for lvl in self._levels}

    @property
    def n_params(self) -> int:
        """Total stored cells across levels (selection: x8 family probabilities)."""
        total = sum(len(lvl.keys) for lvl in self._levels)
        return int(total * (_N_FAM if self.target == "selection" else 1))

    @property
    def deepest_level_name(self) -> str:
        return self._levels[-1].name

    def deepest_cell_train_n(self) -> np.ndarray:
        """Observed training pitch counts of the deepest-level cells."""
        return np.asarray(self._levels[-1].train_n, dtype=np.float64)

    def resolved_train_n(self, table: pd.DataFrame) -> np.ndarray:
        """Training support ``N`` of the deepest observed ancestor used for each row."""
        used_level, used_cell = self._assign(table)
        out = np.empty(len(table), dtype=np.float64)
        for i, level in enumerate(self._levels):
            rows = np.flatnonzero(used_level == i)
            if len(rows):
                out[rows] = level.train_n[used_cell[rows]]
        return out


def fit(
    train_table: pd.DataFrame,
    view: str,
    target: str = "selection",
    max_concentration: float | None = None,
) -> EBModel:
    """Fit an empirical-Bayes table (module-level convenience).

    Parameters
    ----------
    train_table : pandas.DataFrame
        Training decision table (SPEC ``3``).
    view : str
        State view: one of :data:`SELECTION_VIEWS` (``"C"``, ``"U"``, ``"L1"``, ``"O"``).
        ``"OM"`` raises :class:`NotImplementedError` -- matchup memory cannot be tabulated
        (see :func:`support_note`).
    target : {'selection', 'run_value'}
        ``"selection"`` fits the next-pitch family distribution; ``"run_value"`` fits
        ``E[R | key cell, family]`` conditioned on the current action (decision D22).
    max_concentration : float, optional
        Per-level concentration ceiling; defaults to :data:`MAX_CONCENTRATION` for the
        target.

    Returns
    -------
    EBModel
        The fitted table. Call :meth:`EBModel.predict` for predictions.
    """
    return EBModel(view, target, max_concentration=max_concentration).fit(train_table)


# =======================================================================================
# Support diagnostics -- the WS1 exhibit (SPEC 12.1)
# =======================================================================================

def support_table(
    train_table: pd.DataFrame,
    eval_table: pd.DataFrame | None = None,
    views=SELECTION_VIEWS,
    target: str = "selection",
) -> pd.DataFrame:
    """Per-view support diagnostics: the exhibit that exposes the support problem.

    For each view this reports how the conditioning key fragments the data and how badly the
    evaluation rows land in low-support cells -- the concrete evidence that deeper history
    keys cannot be estimated by naïve counting.

    Parameters
    ----------
    train_table : pandas.DataFrame
        Decision table the tables are fitted on (defines the cells and their support).
    eval_table : pandas.DataFrame, optional
        Rows whose support / backoff exposure is measured. Defaults to ``train_table``.
    views : sequence of str, optional
        Views to profile (default ``C, U, L1, O``).
    target : {'selection', 'run_value'}, optional
        Which key ladder to profile (default ``"selection"``).

    Returns
    -------
    pandas.DataFrame
        One row per view with: ``n_cells`` (distinct deepest cells), ``cell_n_p10 / p50 /
        p90 / mean`` (per-cell training pitch-count percentiles), ``frac_eval_lt5 / lt20 /
        lt50`` (fraction of eval rows whose resolved cell has training support below the
        threshold) and ``backoff_<level>`` rates (fraction of eval rows resolved at each
        level -- the smaller the deepest-level rate, the more the view relies on backoff).
    """
    if eval_table is None:
        eval_table = train_table
    rows = []
    for view in views:
        model = fit(train_table, view, target=target)
        cell_n = model.deepest_cell_train_n()
        resolved_n = model.resolved_train_n(eval_table)
        model.predict(eval_table)  # populates backoff_counts_
        n_eval = len(eval_table)
        record = {
            "view": view,
            "n_cells": int(len(cell_n)),
            "cell_n_p10": float(np.percentile(cell_n, 10)) if len(cell_n) else float("nan"),
            "cell_n_p50": float(np.percentile(cell_n, 50)) if len(cell_n) else float("nan"),
            "cell_n_p90": float(np.percentile(cell_n, 90)) if len(cell_n) else float("nan"),
            "cell_n_mean": float(cell_n.mean()) if len(cell_n) else float("nan"),
            "frac_eval_lt5": float(np.mean(resolved_n < 5)),
            "frac_eval_lt20": float(np.mean(resolved_n < 20)),
            "frac_eval_lt50": float(np.mean(resolved_n < 50)),
        }
        for name, count in model.backoff_counts_.items():
            record[f"backoff_{name}"] = float(count / n_eval) if n_eval else float("nan")
        rows.append(record)
    return pd.DataFrame(rows)


_OM_MESSAGE = (
    "OM (matchup-memory) tables are infeasible for pure conditional tables (decision D25). "
    "Keying on the specific batter-vs-pitcher history multiplies the already-fragmented O "
    "cells by the batter dimension, and an average pitcher-batter pair has on the order of "
    "20 career pitches (SPEC 3.4) -- almost every OM cell would have zero or near-zero "
    "support, so the table would back off to O for essentially every row and add nothing. "
    "Matchup memory is handled by the pooled / embedding models (WS3+), not by tables. "
    "Call support_note(table) for the numbers that justify this."
)


def support_note(table: pd.DataFrame | None = None) -> dict:
    """The honest support-problem explanation for OM, with the numbers that justify it.

    Parameters
    ----------
    table : pandas.DataFrame, optional
        A decision table. When given, the returned dict includes empirical
        pitcher-batter-pair support statistics computed from it.

    Returns
    -------
    dict
        ``message`` (the infeasibility explanation) and, when ``table`` is provided,
        ``n_pairs``, ``pair_pitches_median / mean``, ``frac_pairs_lt20 / lt50`` (fraction of
        pitcher-batter pairs with fewer than 20 / 50 pitches) and the same for
        pitcher-batter-count cells -- the fragmentation that makes an OM table hopeless.
    """
    note: dict = {"message": _OM_MESSAGE, "view": "OM", "feasible": False}
    if table is None:
        return note
    pair = table.groupby(["pitcher", "batter"], observed=True).size().to_numpy()
    note.update(
        {
            "n_pairs": int(len(pair)),
            "pair_pitches_median": float(np.median(pair)) if len(pair) else float("nan"),
            "pair_pitches_mean": float(pair.mean()) if len(pair) else float("nan"),
            "frac_pairs_lt20": float(np.mean(pair < 20)) if len(pair) else float("nan"),
            "frac_pairs_lt50": float(np.mean(pair < 50)) if len(pair) else float("nan"),
        }
    )
    cell = table.groupby(["pitcher", "batter", "balls", "strikes"], observed=True).size().to_numpy()
    note["n_pair_count_cells"] = int(len(cell))
    note["pair_count_cell_median"] = float(np.median(cell)) if len(cell) else float("nan")
    note["frac_pair_count_cells_lt5"] = float(np.mean(cell < 5)) if len(cell) else float("nan")
    return note
