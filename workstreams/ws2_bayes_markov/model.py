r"""Bayesian variable-order Markov grammar over pitch-family tokens (SPEC ``12.2``).

This module implements the WS2 "pitch grammar": a per-context variable-order Markov model of
next-pitch **selection** with hierarchical Dirichlet backoff and per-depth concentrations
fitted by empirical Bayes (decisions D29-D31). It fuses the Markov idea (order over pitch
tokens) with the Bayesian-pooling idea (shrinkage toward coarser contexts) so the depth of
context actually used is *earned* by the data.

Tokens, contexts, cells
=======================

* **Token** -- the pitch family (8 symbols :data:`~pitchseq.families.FAMILIES`) of the pitch
  thrown. The grammar models the next token.
* **Context at depth k** -- the ordered last ``k`` families of the *current plate
  appearance*, written chronologically ``ctx_k = (w_{t-k}, ..., w_{t-1})`` (depth 0 is the
  empty context). A row is eligible at depth ``k`` only when the PA supplies ``k`` real
  priors (``pitch_number >= k + 1``), so a context never contains a "no prior" sentinel.
* **Conditioning cell** -- ``ch = (balls, strikes, stand, p_throws)`` (count x hand), with a
  pitcher hierarchy level below it, exactly WS1's C-ladder.

Two-axis hierarchical Dirichlet backoff (decision D29)
======================================================

Hierarchical backoff runs along **two axes**, each a Dirichlet-multinomial chain of
posterior means, and the row prediction is their product-of-experts combination.

**Cell axis (WS1's C-ladder) -- the base rate.** global marginal -> count x hand -> count x
hand x pitcher:

.. math::

   P(a \mid ch) = \frac{n_{ch} + \beta_{ch}\,\pi_G}{N_{ch} + \beta_{ch}},
   \qquad
   b(a) \equiv P(a \mid ch, \mathrm{pit}) =
        \frac{n_{ch,\mathrm{pit}} + \beta_{\mathrm{pit}}\,P(a\mid ch)}{N_{ch,\mathrm{pit}} + \beta_{\mathrm{pit}}} ,

so the count x hand distribution is shrunk toward the global marginal ``pi_G`` and the
pitcher base ``b`` is a *further* level shrunk toward it -- used where support allows (at
depth 0 it does). ``b`` carries the count-conditioned repertoire.

**Depth axis (the primary grammar chain, conditioned on the pitcher) -- the ordered lift.**
The grammar is keyed by the **pitcher** (pooled over counts, so deep ordered contexts keep
support and the lift stays *pitcher-clean*): ``m(a) = P(a \mid \mathrm{pit})`` is the pitcher
marginal (shrunk toward ``pi_G``) and for ``k = 1 .. K_MAX`` a context ``(pit, ctx_k)`` with
next-family counts ``n`` (``N = sum n``) is shrunk toward its **suffix** parent

.. math::

   g_k(a \mid ctx_k, \mathrm{pit}) =
        \frac{n + \alpha_k\, g_{k-1}(a \mid ctx_{k-1}, \mathrm{pit})}{N + \alpha_k},
   \qquad g_0 \equiv m, \quad ctx_{k-1} = \operatorname{suffix}(ctx_k),

with ``suffix`` dropping the *oldest* family ``w_{t-k}`` (exact suffix-chain backoff).

**Product-of-experts combination.** The two axes meet multiplicatively: the base ``b``
supplies the count x pitcher repertoire, and the pitcher-conditioned grammar supplies the
ordered *lift* ``g_k / m`` -- how the ordered context reshapes the pitcher's own mix:

.. math::

   P(a \mid ch, \mathrm{pit}, ctx_k) \;\propto\; b(a)\,\frac{g_k(a \mid ctx_k, \mathrm{pit})}{m(a)}
   \;=\; \underbrace{P(a\mid ch,\mathrm{pit})}_{\text{count x pitcher base}}
         \underbrace{\frac{P(a\mid \mathrm{pit}, ctx_k)}{P(a\mid \mathrm{pit})}}_{\text{within-pitcher order lift}} ,

renormalised over the 8 families. When the context is uninformative (``g_k = m``) the
prediction is exactly the base ``b`` -- so the ``C`` view is ``b``, and ``L1`` / ``O`` apply
the depth-1 / depth-``k`` ordered lift on top of the same base. An unseen context makes
``g_k`` back off to its longest observed suffix; an unseen pitcher makes ``b`` and ``m`` back
off to the count x hand and global marginals.

The design is deliberate; combining the two axes is the crux (D29). Conditioning the ordered
lift on the pitcher is what keeps it *clean*: pooling the grammar over pitchers (keying it by
``ch`` only) lets the previous family stand in for pitcher identity -- ``prev = FC`` signals a
cutter pitcher -- so the lift would double-count the repertoire ``b`` already encodes and
*hurt*. Nesting the grammar under both count and pitcher (keying every depth by
``ch x pitcher``) removes that confound but fragments deep contexts until the ordered signal
is un-earnable. Conditioning the lift on the pitcher while pooling over the count keeps both a
count x pitcher base and a supported, confound-free ordered lift; the order effect is largely
count-independent (the count main-effect is carried by ``b``), which this trades on -- a
documented approximation.

Fitting the concentrations (empirical Bayes, level by level)
============================================================

Each level's concentration is fitted **conditional on the posterior means of its parents**
(a top-down pass), pooling the exchangeable cells at that level, by maximum marginal
likelihood of the Dirichlet-multinomial (Polya) evidence -- a smooth 1-D optimisation via
:mod:`scipy.optimize`; a documented method-of-moments estimator is the fallback when a level
is too sparse to identify the concentration. A depth carrying real ordered signal fits a
**moderate** ``alpha_k`` (its own counts win); a depth carrying none (e.g. after a history
permutation) fits ``alpha_k`` at the ceiling and pools away.

View mapping (decision D29)
===========================

======  ======================================  ==================================
view    max grammar depth                       note
======  ======================================  ==================================
``C``   0 (count x pitcher base only)            selection without current-PA order
``L1``  1                                        first-order (previous family)
``O``   ``K_MAX`` (configurable, default 4)      full variable-order grammar
``U``   -- not applicable                        a Markov grammar is inherently ordered
``OM``  -- not applicable                        no cross-PA pooling in a within-PA chain
======  ======================================  ==================================

``U`` and ``OM`` raise :class:`NotImplementedError`; see :func:`not_applicable_note`.

Everything is built from the shared decision table's existing columns (``pa_id``,
``pitch_number``, ``family``, count / handedness, ``pitcher``); no history logic is
re-implemented beyond forming the ordered token contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from pitchseq.families import FAMILIES

__all__ = [
    "VarOrderMarkov",
    "fit",
    "GRAMMAR_VIEWS",
    "VIEW_MAX_DEPTH",
    "K_MAX_DEFAULT",
    "EFFECTIVE_ORDER_TAU",
    "dirichlet_posterior_mean",
    "fit_dirichlet_concentration",
    "count_hand_code",
    "lag_family_codes",
    "permute_contexts_within_strata",
    "not_applicable_note",
]

# --- fixed encodings -------------------------------------------------------------------

_FAM_INDEX = {f: i for i, f in enumerate(FAMILIES)}
_N_FAM = len(FAMILIES)
_NONE_CODE = _N_FAM  # sentinel family code for "no prior pitch" (0..7 are the families)

#: Views for which the Markov grammar is defined (U / OM are not-applicable -- D29).
GRAMMAR_VIEWS = ("C", "L1", "O")

#: Default cap on the grammar order (decision D29; configurable per fit).
K_MAX_DEFAULT = 4

#: view -> grammar depth. ``C`` / ``L1`` are fixed at order 0 / 1; ``O`` uses the configurable
#: ``k_max`` (shown here at its default) so ``K_MAX`` is fully configurable per fit.
VIEW_MAX_DEPTH = {"C": 0, "L1": 1, "O": K_MAX_DEFAULT}

#: A depth is "earned" for a row when the resolved context's own-count weight
#: ``omega = N / (N + alpha_k)`` reaches this threshold -- the fraction of the posterior
#: effective sample size contributed by the context's own data rather than the backoff
#: prior. Documented default; configurable in :meth:`VarOrderMarkov.effective_order`.
EFFECTIVE_ORDER_TAU = 0.5

# Concentration search bounds (the optimiser works on log10 of the concentration).
_ALPHA_MIN = 1e-3
_LOG10_LO = -3.0
_ROOT_ALPHA = 1.0  # weak symmetric prior at the single-cell root (data swamps it)
#: Concentration ceiling. Selection from flat contexts genuinely gains little from deep
#: history, so the ceiling is effectively unbounded: a depth with no signal is pooled all the
#: way to its parent (``alpha_k`` at the ceiling), which is the honest variable-order result.
_ALPHA_MAX = 1e6


# =======================================================================================
# Pure Dirichlet shrinkage math (documented formulas; exercised directly by the tests)
# =======================================================================================

def dirichlet_posterior_mean(counts, parent, alpha: float) -> np.ndarray:
    r"""Dirichlet posterior mean of a family distribution shrunk toward a parent.

    .. math:: \hat p = \frac{n + \alpha\,\pi}{\sum_k n_k + \alpha}

    Parameters
    ----------
    counts : array-like, shape (K,)
        Observed next-family counts ``n`` for the context cell.
    parent : array-like, shape (K,)
        Parent (suffix) posterior mean ``pi`` (a distribution, sums to 1, strictly positive).
    alpha : float
        Concentration (prior pseudo-count strength) of this level.

    Returns
    -------
    numpy.ndarray, shape (K,)
        The posterior mean. Exactly ``parent`` when ``counts`` sums to 0 -- i.e. an unseen
        context backs off one level up.
    """
    counts = np.asarray(counts, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.float64)
    return (counts + alpha * parent) / (counts.sum() + alpha)


def _dirichlet_neg_log_evidence(alpha: float, counts: np.ndarray, parents: np.ndarray) -> float:
    r"""Negative Dirichlet-multinomial (Polya) log evidence summed over a level's cells.

    For cell ``c`` with prior ``Dir(alpha * pi_c)`` and counts ``n_c`` (``N_c = sum n_c``),
    the marginal likelihood (dropping the ``alpha``-independent multinomial coefficient) is

    .. math:: \log\Gamma(\alpha) - \log\Gamma(\alpha + N_c)
              + \sum_k\big[\log\Gamma(\alpha\pi_{c,k} + n_{c,k}) - \log\Gamma(\alpha\pi_{c,k})\big].
    """
    N = counts.sum(axis=1)
    a = alpha * parents
    term = gammaln(alpha) - gammaln(alpha + N) + (gammaln(a + counts) - gammaln(a)).sum(axis=1)
    return -float(term.sum())


def _dirichlet_mom(counts: np.ndarray, parents: np.ndarray, max_conc: float = _ALPHA_MAX) -> float:
    r"""Method-of-moments concentration from the Pearson overdispersion of a level's cells.

    The Dirichlet-multinomial inflates the multinomial Pearson statistic
    ``X_c = sum_k (n_{c,k} - N_c pi_{c,k})^2 / (N_c pi_{c,k})`` in expectation by a factor
    ``1 + (N_c - 1)/(alpha + 1)`` over the multinomial baseline ``K - 1``. Matching the pooled
    statistic ``S = sum_c X_c`` to its expectation and solving for ``alpha`` gives

    .. math:: \alpha = \frac{\sum_c (N_c - 1)}{S/(K - 1) - C} - 1,

    with ``C`` the number of cells. Under-dispersion (non-positive denominator) maps to heavy
    pooling (``alpha = alpha_max``). Parents are strictly positive, so ``X_c`` is finite.
    """
    N = counts.sum(axis=1)
    use = N > 0
    counts, parents, N = counts[use], parents[use], N[use]
    if len(N) == 0:
        return _ROOT_ALPHA
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
        Per-cell next-family counts for the ``C`` cells at this level.
    parents : numpy.ndarray, shape (C, K)
        Per-cell parent (suffix) posterior means (strictly positive rows summing to 1).
    max_concentration : float, optional
        Upper bound on the fitted concentration (default :data:`_ALPHA_MAX`).

    Returns
    -------
    (float, str)
        The fitted concentration and the estimator used (``"mml"`` / ``"mom"`` /
        ``"default"``). A level with fewer than two informative cells cannot identify a
        shared concentration and falls back to a weak default.
    """
    counts = np.asarray(counts, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.float64)
    informative = int((counts.sum(axis=1) > 0).sum())
    if informative < 2:
        return min(_ROOT_ALPHA, max_concentration), "default"
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


# =======================================================================================
# Token / context encodings (built from the decision table's existing columns)
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


def lag_family_codes(table: pd.DataFrame, k_max: int) -> np.ndarray:
    r"""Family code of the pitch at each lag ``1..k_max`` within every row's PA.

    ``out[r, j-1]`` is the family code of the pitch ``j`` positions before row ``r`` in the
    same plate appearance (lag 1 = the immediately preceding pitch), or :data:`_NONE_CODE`
    when the PA has fewer than ``j`` priors. Uses the shared ``pa_id`` / ``pitch_number``
    ordering (no history logic re-implemented); within a PA ``pitch_number`` increments by 1,
    so lag ``j`` is exactly ``j`` pitches back.

    Parameters
    ----------
    table : pandas.DataFrame
        Decision table (needs ``pa_id``, ``pitch_number``, ``family``).
    k_max : int
        Number of lags to compute.

    Returns
    -------
    numpy.ndarray, shape (n, k_max)
        Lagged family codes (``_NONE_CODE`` where no such prior exists).
    """
    n = len(table)
    order = np.lexsort((table["pitch_number"].to_numpy(), table["pa_id"].to_numpy()))
    fam_s = _family_codes(table)[order]
    pa_s = table["pa_id"].to_numpy()[order]
    out_s = np.full((n, k_max), _NONE_CODE, dtype=np.int64)
    for j in range(1, k_max + 1):
        if j < n:
            same = pa_s[j:] == pa_s[:-j]
            out_s[j:, j - 1] = np.where(same, fam_s[:-j], _NONE_CODE)
    out = np.empty((n, k_max), dtype=np.int64)
    out[order] = out_s
    return out


def _object_array(values) -> np.ndarray:
    """A 1-D object array of tuples (avoids numpy's 2-D coercion of a list of tuples)."""
    vals = list(values)
    arr = np.empty(len(vals), dtype=object)
    arr[:] = vals
    return arr


def permute_contexts_within_strata(
    lag_codes: np.ndarray, ch: np.ndarray, pitch_number: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Permute whole ordered-history vectors among rows sharing a ``(cell, pitch_number)``.

    Each row keeps its own next-family target but is reassigned another row's full lagged
    context vector, drawn from the same ``(count x hand cell, pitch_number)`` stratum. Because
    the stratum fixes ``pitch_number`` it fixes the number of real priors, so a permuted
    context has the right depth. This is the WS2 negative control (decision D30): it removes
    all within-PA ordered dependence while preserving the count x hand mix and PA depth, so
    the grammar's O-vs-L1 edge and its effective order must collapse.

    Parameters
    ----------
    lag_codes : numpy.ndarray, shape (n, k)
        Lagged family codes from :func:`lag_family_codes`.
    ch : numpy.ndarray, shape (n,)
        Count x hand codes (:func:`count_hand_code`).
    pitch_number : numpy.ndarray, shape (n,)
        Pitch number within the PA (defines the depth-preserving stratum).
    seed : int, optional
        Permutation seed (deterministic).

    Returns
    -------
    numpy.ndarray, shape (n, k)
        A row-permuted copy of ``lag_codes`` (contexts scrambled within strata).
    """
    rng = np.random.default_rng(seed)
    key = ch.astype(np.int64) * 1000 + np.asarray(pitch_number, dtype=np.int64)
    out = lag_codes.copy()
    order = np.argsort(key, kind="stable")
    sk = key[order]
    bounds = np.flatnonzero(np.diff(sk)) + 1
    for grp in np.split(order, bounds):
        if len(grp) > 1:
            out[grp] = lag_codes[grp[rng.permutation(len(grp))]]
    return out


# =======================================================================================
# One fitted level of a backoff chain
# =======================================================================================

@dataclass
class _Level:
    """One fitted level: a set of cells with posterior-mean family distributions.

    ``role`` names the level: ``"global"`` / ``"cell"`` / ``"pitcher"`` for the cell axis
    (the base rate ``b``), ``"pit"`` for the pitcher marginal ``m`` (the ordered lift's
    reference), and ``"depth"`` for a grammar level ``g_k``. ``depth`` is the grammar order
    (``>= 1``) for grammar levels and ``0`` otherwise. ``parent`` names the level each cell
    shrinks toward, and ``parent_cell`` maps each cell to its parent's cell index (for the
    motif suffix lookups).
    """

    name: str
    role: str
    depth: int
    keys: dict = field(default_factory=dict)  # key -> cell index
    post_mean: np.ndarray = None  # (C, 8) posterior-mean family distribution
    train_n: np.ndarray = None  # (C,) observed next-token count in the cell
    parent: str = ""  # parent level name
    parent_cell: np.ndarray = None  # (C,) parent cell index (for suffix / motif lookups)
    concentration: float = np.nan  # alpha for this level
    method: str = ""  # estimator used


class VarOrderMarkov:
    """A fitted variable-order Markov selection grammar for one state view.

    Construct with :func:`fit`. :meth:`predict_proba` returns the product-of-experts
    posterior-mean family probabilities; the grammar exhibits (:meth:`effective_order`,
    :meth:`top_motifs`, :meth:`order_usage_summary`) and the calibrated-uncertainty read-outs
    (:meth:`predictive_variance`, :meth:`posterior_stats`) support the notebook's calibration
    and grammar displays.

    Attributes
    ----------
    view : str
        One of :data:`GRAMMAR_VIEWS`.
    k_max : int
        Grammar-order cap.
    max_depth : int
        The view's maximum grammar depth actually built.
    concentrations_ : dict
        Fitted concentration per level name (``beta_cell`` / ``beta_pit`` / ``alpha_k``).
    concentration_methods_ : dict
        The estimator used per level (``"mml"`` / ``"mom"`` / ``"default"``).
    n_cells_ : dict
        Level name -> number of fitted cells.
    grammar_depth_counts_ : dict
        Grammar depth -> number of prediction rows whose deepest observed ordered context was
        that depth, on the most recent :meth:`predict_proba` (depth 0 = base only).
    """

    def __init__(self, view: str, k_max: int = K_MAX_DEFAULT) -> None:
        if view in ("U", "OM"):
            raise NotImplementedError(not_applicable_note(view)["message"])
        if view not in GRAMMAR_VIEWS:
            raise ValueError(f"unknown view {view!r}; expected one of {GRAMMAR_VIEWS} (U/OM not applicable)")
        self.view = view
        self.k_max = int(k_max)
        # C / L1 are fixed at order 0 / 1; O's grammar depth is the configurable K_MAX (D29).
        self.max_depth = self.k_max if view == "O" else VIEW_MAX_DEPTH[view]
        self._levels: list[_Level] = []
        self._by_name: dict[str, _Level] = {}
        self.grammar_depth_counts_: dict[int, int] = {}

    # --- fitting -----------------------------------------------------------------------

    def fit(self, table: pd.DataFrame, lag_codes: np.ndarray | None = None) -> "VarOrderMarkov":
        """Fit both axes on a training decision table.

        Parameters
        ----------
        table : pandas.DataFrame
            Training decision table (SPEC ``3``).
        lag_codes : numpy.ndarray, optional
            Precomputed ``(n, max_depth)`` lagged family codes (:func:`lag_family_codes`).
            Supplied by the permutation control to inject scrambled contexts; derived from
            ``table`` when ``None``.
        """
        fam = _family_codes(table)
        ch = count_hand_code(table)
        pit = pd.to_numeric(table["pitcher"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        if lag_codes is None:
            lag_codes = lag_family_codes(table, max(self.max_depth, 1))

        self._levels = []
        # --- cell axis (the base rate b): global -> cell(ch) -> pitcher(ch, pit) ---
        self._fit_global(fam)
        self._fit_child("cell", "cell", 0, keys=_object_array(zip(ch.tolist())),
                        parent_keys=_object_array([()] * len(fam)), parent_level="global", fam=fam)
        self._fit_child("pitcher", "pitcher", 0, keys=_object_array(zip(ch.tolist(), pit.tolist())),
                        parent_keys=_object_array(zip(ch.tolist())), parent_level="cell", fam=fam)

        # --- depth axis (the ordered lift): pitcher marginal m -> (pit, ctx_1) -> ... ---
        # Conditioned on the pitcher and pooled over counts (so the lift is pitcher-clean and
        # deep contexts keep support); the suffix parent drops the oldest lag ``l_k``.
        if self.max_depth >= 1:
            self._fit_child("pit", "pit", 0, keys=_object_array(zip(pit.tolist())),
                            parent_keys=_object_array([()] * len(fam)), parent_level="global", fam=fam)
            for k in range(1, self.max_depth + 1):
                rows = np.flatnonzero(lag_codes[:, k - 1] != _NONE_CODE)  # PA supplies k real priors
                ctx_cols = [pit[rows].tolist()] + [lag_codes[rows, j].tolist() for j in range(k)]
                keys = _object_array(zip(*ctx_cols))
                parent_cols = [pit[rows].tolist()] + [lag_codes[rows, j].tolist() for j in range(k - 1)]
                parent_keys = _object_array(zip(*parent_cols))
                parent_level = f"depth{k - 1}" if k > 1 else "pit"
                self._fit_child(f"depth{k}", "depth", k, keys=keys, parent_keys=parent_keys,
                                parent_level=parent_level, fam=fam[rows])

        self._by_name = {lvl.name: lvl for lvl in self._levels}
        return self

    def _fit_global(self, fam: np.ndarray) -> None:
        counts = np.zeros(_N_FAM, dtype=np.float64)
        np.add.at(counts, fam, 1.0)
        uniform = np.full(_N_FAM, 1.0 / _N_FAM)
        post = (counts + _ROOT_ALPHA * uniform) / (counts.sum() + _ROOT_ALPHA)
        self._levels.append(_Level(
            name="global", role="global", depth=0, keys={(): 0}, post_mean=post[None, :],
            train_n=np.array([counts.sum()]), parent="", parent_cell=np.array([-1]),
            concentration=_ROOT_ALPHA, method="default",
        ))

    def _fit_child(self, name, role, depth, keys, parent_keys, parent_level, fam) -> None:
        """Fit one non-root level: count, look up parents, fit the concentration, shrink."""
        parent = self._find(parent_level)
        codes, uniques = pd.factorize(keys, sort=False)
        C = len(uniques)
        counts = np.zeros((C, _N_FAM), dtype=np.float64)
        np.add.at(counts, (codes, fam), 1.0)
        N = counts.sum(axis=1)

        first_row = pd.Series(np.arange(len(keys))).groupby(codes).first().to_numpy()
        parent_key_of_cell = parent_keys[first_row]
        parent_cell = np.array([parent.keys.get(pk, 0) for pk in parent_key_of_cell.tolist()], dtype=np.int64)
        parents = parent.post_mean[parent_cell]

        alpha, method = fit_dirichlet_concentration(counts, parents, _ALPHA_MAX)
        post = (counts + alpha * parents) / (N[:, None] + alpha)
        self._levels.append(_Level(
            name=name, role=role, depth=depth, keys=dict(zip(uniques.tolist(), range(C))),
            post_mean=post, train_n=N, parent=parent_level, parent_cell=parent_cell,
            concentration=alpha, method=method,
        ))

    def _find(self, name: str) -> _Level:
        for lvl in self._levels:
            if lvl.name == name:
                return lvl
        raise KeyError(f"level {name!r} not fitted")  # pragma: no cover - defensive

    # --- row-key + backoff machinery ---------------------------------------------------

    def _coords(self, table: pd.DataFrame, lag_codes: np.ndarray | None):
        """Per-row conditioning coordinates (ch, pitcher, pitch_number, lagged contexts)."""
        ch = count_hand_code(table)
        pit = pd.to_numeric(table["pitcher"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        pn = pd.to_numeric(table["pitch_number"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        if lag_codes is None:
            lag_codes = lag_family_codes(table, max(self.max_depth, 1))
        return ch, pit, pn, lag_codes

    def _row_keys(self, level: _Level, ch, pit, pn, lag_codes):
        """Per-row key + eligibility mask for one level."""
        n = len(ch)
        if level.role == "global":
            return _object_array([()] * n), np.ones(n, bool)
        if level.role == "cell":
            return _object_array(zip(ch.tolist())), np.ones(n, bool)
        if level.role == "pitcher":
            return _object_array(zip(ch.tolist(), pit.tolist())), np.ones(n, bool)
        if level.role == "pit":
            return _object_array(zip(pit.tolist())), np.ones(n, bool)
        k = level.depth
        eligible = (lag_codes[:, k - 1] != _NONE_CODE) & (pn >= k + 1)
        cols = [pit.tolist()] + [lag_codes[:, j].tolist() for j in range(k)]
        return _object_array(zip(*cols)), eligible

    def _backoff(self, pref: list[_Level], ch, pit, pn, lag_codes):
        """Deepest observed level per row over ``pref`` (deepest first).

        Returns ``(post_mean (n, 8), used_pref_index (n,), cell_index (n,))``. ``pref`` must
        end in a level that resolves every row (the global level).
        """
        n = len(ch)
        used = np.full(n, -1, dtype=np.int64)
        cell = np.full(n, -1, dtype=np.int64)
        for i, level in enumerate(pref):
            todo = np.flatnonzero(used < 0)
            if len(todo) == 0:
                break
            keys, eligible = self._row_keys(level, ch, pit, pn, lag_codes)
            cand = todo[eligible[todo]]
            if len(cand) == 0:
                continue
            mapped = pd.Series(keys[cand]).map(level.keys).to_numpy()
            found = ~pd.isna(mapped)
            hit = cand[found]
            used[hit] = i
            cell[hit] = mapped[found].astype(np.int64)
        out = np.empty((n, _N_FAM), dtype=np.float64)
        for i, level in enumerate(pref):
            rows = np.flatnonzero(used == i)
            if len(rows):
                out[rows] = level.post_mean[cell[rows]]
        return out, used, cell

    def _grammar_pref(self) -> list[_Level]:
        """Grammar levels deepest-first, backing off to the pitcher marginal then global."""
        gram = sorted((lvl for lvl in self._levels if lvl.role == "depth"),
                      key=lambda lvl: lvl.depth, reverse=True)
        return gram + [self._by_name["pit"], self._by_name["global"]]

    @staticmethod
    def _resolved_ess(pref: list[_Level], used: np.ndarray, cell: np.ndarray, n: int) -> np.ndarray:
        """Per-row posterior ESS ``N + alpha`` of the resolved level (vectorised over levels)."""
        ess = np.empty(n, dtype=np.float64)
        for i, level in enumerate(pref):
            rows = np.flatnonzero(used == i)
            if len(rows):
                ess[rows] = level.train_n[cell[rows]] + level.concentration
        return ess

    # --- prediction --------------------------------------------------------------------

    def predict_proba(self, table: pd.DataFrame, lag_codes: np.ndarray | None = None) -> np.ndarray:
        r"""Product-of-experts posterior-mean next-family probabilities ``(n, 8)``.

        For each row the count x pitcher base ``b`` (backing off pitcher -> cell -> global) is
        combined with the within-pitcher order lift ``g_k / m`` of the deepest observed ordered
        context: ``P(a) \propto b(a)\, g_k(a) / m(a)``, renormalised. With no grammar levels
        (``C`` view) or an uninformative context this is exactly the base ``b``; rows always
        sum to 1.
        """
        return self.posterior_stats(table, lag_codes)["proba"]

    # --- calibrated uncertainty --------------------------------------------------------

    def posterior_stats(self, table: pd.DataFrame, lag_codes: np.ndarray | None = None) -> dict:
        r"""Per-row product-of-experts prediction plus the calibration read-outs.

        The base ``b`` and ordered ``g`` factors are each Dirichlet posterior means; the
        per-row effective sample size is the **smaller** of the two resolved nodes' ESS
        (``N + alpha``) -- the sparser factor bounds the confidence -- and the per-family
        posterior variance is ``p(1 - p)/(ESS + 1)`` on the combined ``p``.

        Returns
        -------
        dict
            ``proba`` ``(n, 8)``, ``variance`` ``(n, 8)``, ``ess`` ``(n,)`` (min of the base
            and grammar node ESS), ``base_ess`` ``(n,)``, ``grammar_ess`` ``(n,)``,
            ``resolved_depth`` ``(n,)`` (deepest observed ordered depth; 0 = base only).
        """
        ch, pit, pn, lag_codes = self._coords(table, lag_codes)
        n = len(ch)

        base_pref = [self._by_name["pitcher"], self._by_name["cell"], self._by_name["global"]]
        b, b_used, b_cell = self._backoff(base_pref, ch, pit, pn, lag_codes)
        base_ess = self._resolved_ess(base_pref, b_used, b_cell, n)

        if self.max_depth >= 1:
            m_pref = [self._by_name["pit"], self._by_name["global"]]
            m, _, _ = self._backoff(m_pref, ch, pit, pn, lag_codes)
            gram_pref = self._grammar_pref()
            g, g_used, g_cell = self._backoff(gram_pref, ch, pit, pn, lag_codes)
            depth = np.zeros(n, dtype=np.int64)  # "pit"/"global" levels carry depth 0
            for i, level in enumerate(gram_pref):
                rows = np.flatnonzero(g_used == i)
                if len(rows):
                    depth[rows] = level.depth
            grammar_ess = self._resolved_ess(gram_pref, g_used, g_cell, n)
            prod = b * g / m
            proba = prod / prod.sum(axis=1, keepdims=True)
        else:  # C view: base only, no ordered lift
            proba = b / b.sum(axis=1, keepdims=True)
            depth = np.zeros(n, dtype=np.int64)
            grammar_ess = np.full(n, np.inf)

        self.grammar_depth_counts_ = {int(k): int(np.sum(depth == k)) for k in range(self.max_depth + 1)}
        ess = np.minimum(base_ess, grammar_ess)
        variance = proba * (1.0 - proba) / (ess[:, None] + 1.0)
        return {"proba": proba, "variance": variance, "ess": ess, "base_ess": base_ess,
                "grammar_ess": grammar_ess, "resolved_depth": depth}

    def predictive_variance(self, table: pd.DataFrame, lag_codes: np.ndarray | None = None) -> np.ndarray:
        """Per-row per-family posterior predictive variance ``p(1-p)/(ESS+1)`` ``(n, 8)``."""
        return self.posterior_stats(table, lag_codes)["variance"]

    # --- grammar exhibits --------------------------------------------------------------

    def effective_order(self, table: pd.DataFrame, tau: float = EFFECTIVE_ORDER_TAU,
                        lag_codes: np.ndarray | None = None) -> dict:
        r"""Per-row grammar depth actually earned, and its distribution.

        A row's effective order is the deepest grammar depth ``k`` (``<= max_depth`` and the
        available priors) whose context ``(pit, ctx_k)`` was observed in training **and** whose
        own-count weight

        .. math:: \omega_k = \frac{N_k}{N_k + \alpha_k}

        reaches ``tau`` -- i.e. the context's own data, not the backoff prior, drives its
        distribution. A depth with no ordered signal fits ``alpha_k`` at the ceiling, so
        ``omega_k`` collapses and the effective order drops to the coarser context. Rows with
        no qualifying depth have effective order 0.

        Returns
        -------
        dict
            ``per_row`` ``(n,)``, ``distribution`` (order -> fraction of rows), ``mean``,
            ``frac_ge2`` (fraction with order ``>= 2``), ``tau``.
        """
        ch, pit, pn, lag_codes = self._coords(table, lag_codes)
        n = len(ch)
        order = np.zeros(n, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)
        for k in range(self.max_depth, 0, -1):
            level = self._by_name[f"depth{k}"]
            eligible = (~assigned) & (lag_codes[:, k - 1] != _NONE_CODE) & (pn >= k + 1)
            rows = np.flatnonzero(eligible)
            if len(rows) == 0:
                continue
            cols = [pit[rows].tolist()] + [lag_codes[rows, j].tolist() for j in range(k)]
            mapped = pd.Series(_object_array(zip(*cols))).map(level.keys).to_numpy()
            found = ~pd.isna(mapped)
            cells = mapped[found].astype(np.int64)
            omega = level.train_n[cells] / (level.train_n[cells] + level.concentration)
            hit = rows[found][omega >= tau]
            order[hit] = k
            assigned[hit] = True
        dist = {int(k): float(np.mean(order == k)) for k in range(self.max_depth + 1)}
        return {"per_row": order, "distribution": dist,
                "mean": float(order.mean()) if n else float("nan"),
                "frac_ge2": float(np.mean(order >= 2)) if n else float("nan"), "tau": float(tau)}

    def top_motifs(self, min_support: int = 30, top_n: int = 20) -> list[dict]:
        r"""The "grammar rules": ordered contexts whose next-token distribution moves most.

        For every grammar context (aggregated over pitchers that share the ordered family
        context) the readout compares the context's next-family distribution ``p_ctx`` to its
        suffix's ``p_suffix`` (drop the oldest family), ranking by the Kullback-Leibler
        divergence

        .. math:: D_{\mathrm{KL}}(p_{\text{ctx}} \,\|\, p_{\text{suffix}})
                  = \sum_a p_{\text{ctx}}(a)\,\log_2 \frac{p_{\text{ctx}}(a)}{p_{\text{suffix}}(a)}.

        The single family with the largest absolute probability change is reported with its
        before/after probabilities, ``log2`` ratio and direction (``"suppress"`` when its
        probability drops, ``"promote"`` when it rises) -- e.g. a same-family run
        ``FF FF -> P(FF)`` dropping is the no-three-in-a-row rule.

        Parameters
        ----------
        min_support : int, optional
            Minimum pooled training support for a context to be reported (default 30).
        top_n : int, optional
            Number of motifs to return (default 20).

        Returns
        -------
        list of dict
            Ranked motifs, each with ``context`` (families oldest->newest), ``depth``,
            ``support``, ``kl``, ``top_family``, ``p_ctx``, ``p_suffix``, ``log2_ratio``,
            ``direction`` and a human-readable ``text``.
        """
        agg: dict[tuple, dict] = {}
        for level in self._levels:
            if level.role != "depth":
                continue
            k = level.depth
            parent = self._by_name[level.parent]
            for key, ci in level.keys.items():
                fam_ctx = tuple(int(x) for x in key[1:])  # drop the pitcher code
                N = float(level.train_n[ci])
                if N <= 0:
                    continue
                rec = agg.setdefault((k, fam_ctx),
                                     {"num_ctx": np.zeros(_N_FAM), "num_suf": np.zeros(_N_FAM), "N": 0.0})
                rec["num_ctx"] += N * level.post_mean[ci]
                rec["num_suf"] += N * parent.post_mean[level.parent_cell[ci]]
                rec["N"] += N

        motifs = []
        for (k, fam_ctx), rec in agg.items():
            if rec["N"] < min_support:
                continue
            p_ctx = rec["num_ctx"] / rec["N"]
            p_suf = rec["num_suf"] / rec["N"]
            kl = float(np.sum(p_ctx * np.log2(np.clip(p_ctx, 1e-12, 1.0) / np.clip(p_suf, 1e-12, 1.0))))
            j = int(np.argmax(np.abs(p_ctx - p_suf)))
            log2_ratio = float(np.log2(np.clip(p_ctx[j], 1e-12, 1.0) / np.clip(p_suf[j], 1e-12, 1.0)))
            direction = "suppress" if p_ctx[j] < p_suf[j] else "promote"
            fams = [FAMILIES[c] for c in reversed(fam_ctx)]  # oldest -> newest (chronological)
            top_fam = FAMILIES[j]
            text = (f"{' '.join(fams)} -> P({top_fam}) {p_suf[j]:.2f}->{p_ctx[j]:.2f} "
                    f"({direction}, KL={kl:.3f})")
            motifs.append({"context": fams, "depth": int(k), "support": int(rec["N"]), "kl": kl,
                           "top_family": top_fam, "p_ctx": float(p_ctx[j]), "p_suffix": float(p_suf[j]),
                           "log2_ratio": log2_ratio, "direction": direction, "text": text})
        motifs.sort(key=lambda m: m["kl"], reverse=True)
        return motifs[:top_n]

    def order_usage_summary(self) -> dict:
        """Grammar-depth usage of the most recent :meth:`predict_proba` call.

        Returns
        -------
        dict
            ``depth_distribution`` (deepest observed ordered depth -> fraction of rows),
            ``grammar_frac`` (fraction with an ordered context of depth ``>= 1``), the fitted
            ``concentrations`` / ``methods`` per level and ``n_levels``.
        """
        total = sum(self.grammar_depth_counts_.values()) or 1
        dist = {k: c / total for k, c in self.grammar_depth_counts_.items()}
        grammar_frac = sum(c for k, c in self.grammar_depth_counts_.items() if k >= 1) / total
        return {"depth_distribution": dist, "grammar_frac": float(grammar_frac),
                "concentrations": self.concentrations_, "methods": self.concentration_methods_,
                "n_levels": len(self._levels)}

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
        """Total stored family distributions across levels (each is 8 probabilities)."""
        return int(sum(len(lvl.keys) for lvl in self._levels) * _N_FAM)


def fit(
    train_table: pd.DataFrame,
    view: str,
    k_max: int = K_MAX_DEFAULT,
    lag_codes: np.ndarray | None = None,
) -> VarOrderMarkov:
    """Fit a variable-order Markov grammar (module-level convenience).

    Parameters
    ----------
    train_table : pandas.DataFrame
        Training decision table (SPEC ``3``).
    view : str
        State view: one of :data:`GRAMMAR_VIEWS` (``"C"`` depth 0, ``"L1"`` depth 1, ``"O"``
        depth ``<= k_max``). ``"U"`` / ``"OM"`` raise :class:`NotImplementedError` -- a Markov
        grammar is inherently ordered and has no cross-PA pooling (see
        :func:`not_applicable_note`).
    k_max : int, optional
        Grammar-order cap for the ``O`` view (decision D29; default :data:`K_MAX_DEFAULT`).
    lag_codes : numpy.ndarray, optional
        Precomputed lagged contexts (permutation control); derived from ``train_table`` when
        ``None``.

    Returns
    -------
    VarOrderMarkov
        The fitted grammar. Call :meth:`VarOrderMarkov.predict_proba` for predictions.
    """
    return VarOrderMarkov(view, k_max=k_max).fit(train_table, lag_codes=lag_codes)


# =======================================================================================
# Not-applicable views (D29) -- U and OM
# =======================================================================================

_NA_MESSAGES = {
    "U": (
        "The U (unordered) view is not applicable to a variable-order Markov grammar "
        "(decision D29). A Markov context is an *ordered* string of preceding tokens; the "
        "grammar's entire object of study is order. An order-invariant multiset of prior "
        "families is not a Markov context, and collapsing the ordered context to its multiset "
        "would erase exactly the structure WS2 measures. The unordered question is answered by "
        "the tabular workstreams (WS1's U table) and the flexible models (WS3+), not by the "
        "grammar."
    ),
    "OM": (
        "The OM (matchup-memory) view is not applicable to a variable-order Markov grammar "
        "(decision D29). The grammar is a *within-PA* chain over the current plate "
        "appearance's tokens; matchup memory is cross-PA (earlier PAs this game / season / "
        "career), which the Markov context does not span. Cross-PA adaptation is handled by "
        "the pooled / embedding models (WS3+, WS6), not by the within-PA grammar."
    ),
}


def not_applicable_note(view: str) -> dict:
    """The honest explanation for a not-applicable view (``U`` or ``OM``; decision D29).

    Parameters
    ----------
    view : str
        ``"U"`` or ``"OM"``.

    Returns
    -------
    dict
        ``{'view', 'applicable': False, 'message'}`` explaining why the grammar does not fit
        the view.
    """
    if view not in _NA_MESSAGES:
        raise ValueError(f"{view!r} is applicable to the grammar; not-applicable views are U / OM")
    return {"view": view, "applicable": False, "message": _NA_MESSAGES[view]}
