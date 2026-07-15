r"""Conservative offline RL (FQI) + exploitability + the study frontier (SPEC 12.7 / 10).

This is the WS7 capstone model module. It supplies four reusable pieces, in build order:

1. **Conservative fitted-Q iteration** (decision D48) -- :class:`ConservativeFQI`. A batch
   fitted-Q iteration over the decision-table feature views (the rich ``O`` view via
   :func:`pitchseq.states.build_view` with the current action appended per decision D22, or a
   coarse ``count`` state for the comparison rung) with a **behavior-support pessimism penalty**.
   Unlike :func:`pitchseq.eval.ope.fqe` (which evaluates a *fixed* target by a policy-weighted
   backup), FQI iterates the **Bellman-optimality** backup to a fixed point to *improve* the
   policy:

   .. math::
      Q_{k+1}(s,a) \;\leftarrow\; r(s,a) \;+\;
        \mathbf{1}[s\ \text{non-terminal}]\;
        \max_{a'\in\mathcal{F}(s')}\!\big[\,Q_k(s',a') - \lambda\,\mathrm{pen}(s',a')\,\big],

   backward over the plate-appearance steps (each iteration propagates value one pitch back;
   ``n_iter`` >= the PA horizon reaches the finite-horizon fixed point). The **behavior-support
   penalty** is the documented indicator variant

   .. math:: \mathrm{pen}(s,a) = \mathbf{1}\!\big[\hat\mu(a\mid s) < \text{floor}\big],

   penalising actions the logging policy rarely takes (their bootstrapped Q is unreliable and
   off-support). ``lambda`` scales it in reward units (config). This is the CQL-lite pessimism
   that keeps the improved policy near behaviour support -- the SPEC 9 conservative discipline
   pushed inside the Bellman backup, not only the final softening. The alternative smooth variant
   ``pen(s,a) = c / sqrt(max(n_eff(s,a), 1))`` is documented on :func:`behavior_support_penalty`;
   the indicator is chosen as primary because it *is* the SPEC 9 support diagnostic (a hard
   behaviour-probability floor), is interpretable, and needs no extra count bookkeeping. ``mu-hat``
   is WS3's contextual propensities when supplied (decision D33), else the empirical
   state-conditional behaviour from train counts.

2. **The conservative policy** -- :func:`conservative_greedy` (feasible arg-max of the penalised
   Q, one-hot), softened toward behaviour through :func:`pitchseq.eval.ope.pi_alpha` at the
   pipeline level; :func:`fqi_diagnostics` (Q drift per iteration, penalty share, support
   histogram).

3. **The exploitability read-out** (decision D49, SPEC 10) -- a fixed interpretable
   :class:`BatterResponseModel` (a tabular swing/take response by ``count x family x
   location-bucket``), the per-count **zero-sum payoff game** it induces with the common outcome
   model (:func:`payoff_by_count`), its :func:`equilibrium_value` by LP (``scipy.linprog``), and
   each policy's :func:`policy_exploitability` / :func:`exploitability` vs that equilibrium, with
   bootstrapped CIs.

4. **The study frontier** (decision D49, SPEC 10) -- :func:`assemble_frontier` (the plotting-ready
   table over every policy: OPE value x predictability-in-bits x exploitability x
   distance-from-behaviour x compute), :func:`write_frontier_csv` and :func:`plot_frontier` (the
   final matplotlib figure of the study).

Plus persistence (:func:`save_ws7_artifacts` / :func:`load_ws7_artifacts`). LightGBM is the
``ml`` extra (as in WS3); the ``count`` rung and everything else is LightGBM-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:  # LightGBM powers the rich O-view Q-regressor (the WS3 ``ml`` extra).
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "Workstream 7 requires LightGBM for the O-view FQI. Install the ML extra:\n"
        "    pip install -e '.[ml]'\n"
    ) from exc

import joblib
from scipy.optimize import linprog

from pitchseq.eval import ope
from pitchseq.families import FAMILIES, OTHER_FAMILY
from pitchseq.states import build_view

__all__ = [
    "FAMILIES",
    "FQI_VIEWS",
    "XX_INDEX",
    "N_COUNT",
    "DEFAULT_LAMBDA",
    "DEFAULT_SUPPORT_FLOOR",
    "DEFAULT_N_ITER",
    "DEFAULT_DISRUPTION_BETA",
    "DEFAULT_FQI_PARAMS",
    "ACTION_COL",
    "count_state",
    "family_indices",
    "behavior_support_penalty",
    "effective_feasible_mask",
    "conservative_greedy",
    "successor_index",
    "fitted_q_iteration",
    "ConservativeFQI",
    "fqi_diagnostics",
    "BatterResponseModel",
    "payoff_by_count",
    "payoff_matrices",
    "equilibrium_value",
    "policy_exploitability",
    "exploitability",
    "FRONTIER_COLUMNS",
    "assemble_frontier",
    "write_frontier_csv",
    "plot_frontier",
    "save_ws7_artifacts",
    "load_ws7_artifacts",
]

# -------------------------------------------------------------------------------------
# constants
# -------------------------------------------------------------------------------------

_N_FAM = len(FAMILIES)  # 8
#: Index of the descriptive-only ``XX`` family (never a feasible / recommendable action; SPEC 4).
XX_INDEX = FAMILIES.index(OTHER_FAMILY)  # 7
_FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}

#: The two FQI state representations. ``O`` is the rich ordered view (LightGBM function
#: approximation); ``count`` is the coarse ``balls x strikes`` cell (exact tabular) -- the C-analog
#: comparison rung whose value gap with ``O`` isolates the *sequencing* contribution.
FQI_VIEWS: tuple[str, ...] = ("O", "count")

#: 12 count cells (balls 0..3 x strikes 0..2); the ``count``-view FQI state and the exploitability
#: game's per-state index.
N_COUNT = 12

#: Pessimism penalty coefficient ``lambda`` (reward units); scales the behaviour-support penalty
#: inside the Bellman backup and the final greedy. Config-exposed (``--lambda`` / ``lam``).
DEFAULT_LAMBDA = 0.05

#: Behaviour-probability floor for the support penalty (matches SPEC 9's support_threshold): an
#: action with ``mu-hat(a|s) < floor`` is treated as off-support and penalised.
DEFAULT_SUPPORT_FLOOR = 0.02

#: FQI iterations (Bellman-optimality backups). The PA horizon is short (<= a dozen pitches), so a
#: handful of iterations reaches the finite-horizon fixed point; iteration stops early on small Q
#: drift. Kept modest because each iteration is a full regressor refit.
DEFAULT_N_ITER = 6

#: Anticipation-disruption coefficient ``beta`` for the exploitability game (run units): how much a
#: correctly-anticipating batter reduces the pitcher's reward on a family, scaled by that family's
#: swing propensity (see :func:`payoff_by_count`).
DEFAULT_DISRUPTION_BETA = 0.12

#: Small deterministic LightGBM defaults for the O-view Q-regressor (the CLI may raise
#: ``num_threads`` / ``n_estimators`` on real data). Mirrors WS3's reproducibility settings.
DEFAULT_FQI_PARAMS: dict = {
    "n_estimators": 200,
    "num_leaves": 31,
    "min_child_samples": 40,
    "learning_rate": 0.05,
    "max_depth": -1,
    "reg_lambda": 1.0,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
    "random_state": 20260715,
}

#: The current-action feature appended to the O view (decision D22): ``Q(s, a)`` conditions on the
#: chosen family, which is the *decision*, not execution -- legitimate on the leakage-audited view.
ACTION_COL = "action_family"

_NEG_INF = -1e18


# -------------------------------------------------------------------------------------
# small shared helpers
# -------------------------------------------------------------------------------------


def count_state(table: pd.DataFrame) -> np.ndarray:
    """Coarse discrete state = count cell ``balls * 3 + strikes`` (0..11) -- the ``count`` FQI state.

    The same discretisation WS4 used for the FQE cross-check; it is also the exploitability game's
    per-state index. A pre-pitch count never reaches 4 balls / 3 strikes (those terminate the PA).
    """
    b = table["balls"].to_numpy(dtype=np.int64)
    s = table["strikes"].to_numpy(dtype=np.int64)
    return b * 3 + s


def family_indices(table: pd.DataFrame, action_col: str = "family") -> np.ndarray:
    """Map the observed family label per row to its :data:`FAMILIES` index (``XX`` for unknowns)."""
    return np.array(
        [_FAMILY_TO_IDX.get(str(f), XX_INDEX) for f in table[action_col].astype(object).to_numpy()],
        dtype=np.int64,
    )


def successor_index(pa_id, order_key) -> tuple[np.ndarray, np.ndarray]:
    r"""Within-episode successor row index and terminal flag per decision (for the FQI backup).

    Within each PA (``pa_id``, ordered by ``order_key`` = ``pitch_number``) a non-last pitch's
    successor is the *original row index* of the next pitch; the last pitch of a PA is terminal
    (``succ = -1``). Fully vectorised (a lexsort + a same-episode shift); the returned arrays are in
    the input row order.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``succ`` (int, original index of the successor row or ``-1``) and ``terminal`` (bool).
    """
    pa = np.asarray(pa_id)
    order = np.lexsort((np.asarray(order_key), pa))  # positions sorted by (pa_id, order_key)
    pa_s = pa[order]
    n = len(pa)
    succ = np.full(n, -1, dtype=np.int64)
    same = pa_s[:-1] == pa_s[1:]
    succ[order[:-1][same]] = order[1:][same]
    terminal = succ == -1
    return succ, terminal


# -------------------------------------------------------------------------------------
# the behaviour-support pessimism penalty (decision D48)
# -------------------------------------------------------------------------------------


def behavior_support_penalty(mu_probs, floor: float = DEFAULT_SUPPORT_FLOOR) -> np.ndarray:
    r"""The behaviour-support penalty ``pen(s,a) = 1[mu-hat(a|s) < floor]`` (decision D48).

    A per-row ``(n, A)`` indicator that is ``1`` on actions the estimated logging policy takes with
    probability below ``floor`` -- exactly SPEC 9's out-of-support flag. Multiplied by ``lambda``
    (in reward units) it becomes the CQL-lite pessimism subtracted from ``Q`` inside the FQI backup
    and the final greedy: the improved policy is pushed away from actions where the logged data
    cannot support a reliable value estimate.

    The chosen indicator form is interpretable (a hard behaviour-probability floor) and reuses the
    SPEC 9 diagnostic verbatim. The documented smooth alternative is a count-based bonus
    ``pen(s,a) = c / sqrt(max(n_eff(s,a), 1))`` (effective sample size of the ``(s, a)`` cell); it
    trades interpretability for a graded penalty and needs per-cell counts -- not used here.

    Parameters
    ----------
    mu_probs : array-like, shape (n, A)
        Estimated behaviour policy ``mu-hat(.|s)`` per row (WS3 propensities or empirical counts).
    floor : float, optional
        Support floor (default :data:`DEFAULT_SUPPORT_FLOOR`).

    Returns
    -------
    numpy.ndarray, shape (n, A)
        The 0/1 penalty (float).
    """
    mu = np.asarray(mu_probs, dtype=np.float64)
    if mu.ndim != 2:
        raise ValueError(f"mu_probs must be 2-D (n, A), got shape {mu.shape}")
    return (mu < float(floor)).astype(np.float64)


def effective_feasible_mask(feasible_mask) -> np.ndarray:
    r"""SPEC-4 mask with empty rows filled to all non-``XX`` families (the WS4/WS5 discipline).

    A decision row with no SPEC-4-feasible family (a low-history pitcher) would leave the FQI backup
    and the greedy with no action; as in WS4/WS5 such rows fall back to every non-``XX`` family being
    feasible, so a policy is always defined. ``XX`` stays infeasible everywhere (SPEC 4).
    """
    feas = np.asarray(feasible_mask, dtype=bool).copy()
    empty = ~feas.any(axis=1)
    if empty.any():
        feas[empty] = True
        feas[empty, XX_INDEX] = False
    return feas


def conservative_greedy(q, penalty, feasible_mask, lam: float = DEFAULT_LAMBDA) -> np.ndarray:
    r"""Feasible arg-max of the **penalised** Q -- the conservative greedy policy (one-hot).

    .. math:: \pi(\cdot\mid s) = \operatorname*{one\_hot\,arg\,max}_{a\in\mathcal{F}(s)}
              \big[\,Q(s,a) - \lambda\,\mathrm{pen}(s,a)\,\big]

    Infeasible actions and off-support actions (through the penalty) are pushed out of the arg-max;
    a state with no feasible action falls back to the observed-action-agnostic uniform-over-nothing
    guard (all mass on family 0 as a placeholder, never surfaced -- the pipeline only scores rows
    with a feasible action). The pipeline softens this toward behaviour with
    :func:`pitchseq.eval.ope.pi_alpha` (the SPEC 9 conservative mixture).

    Parameters
    ----------
    q : array-like, shape (n, A)
        Q-values ``Q(s, a)``.
    penalty : array-like, shape (n, A)
        The support penalty from :func:`behavior_support_penalty` (0/1 grid).
    feasible_mask : array-like of bool, shape (n, A)
        Per-row feasible action set (SPEC 4).
    lam : float, optional
        Penalty coefficient ``lambda`` (default :data:`DEFAULT_LAMBDA`).

    Returns
    -------
    numpy.ndarray, shape (n, A)
        One-hot greedy policy per row (rows sum to 1).
    """
    q = np.asarray(q, dtype=np.float64)
    pen = np.asarray(penalty, dtype=np.float64)
    feas = np.asarray(feasible_mask, dtype=bool)
    scores = np.where(feas, q - lam * pen, _NEG_INF)
    # A row with no feasible action: keep the arg-max well-defined (placeholder on family 0).
    no_feas = ~feas.any(axis=1)
    idx = scores.argmax(axis=1)
    idx[no_feas] = 0
    out = np.zeros_like(q)
    out[np.arange(len(q)), idx] = 1.0
    return out


# -------------------------------------------------------------------------------------
# Q-regressors (two backends): rich O-view LightGBM, coarse count-state tabular
# -------------------------------------------------------------------------------------


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce a state-view frame to LightGBM-friendly dtypes (categoricals preserved).

    Object -> ``category`` (native categorical handling), bool -> ``int8``, else ``float64``.
    Existing ``category`` columns (the view's family / outcome / pitch-type tokens and the appended
    ``action_family``) are left untouched so their integer codes -- and predictions -- are stable
    across a save / load round-trip. Mirrors WS3's ``_prep``.
    """
    X = X.copy()
    for c in X.columns:
        dt = X[c].dtype
        if isinstance(dt, pd.CategoricalDtype):
            continue
        if dt == bool:
            X[c] = X[c].astype("int8")
        elif dt == object:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


class _LGBMQ:
    """O-view LightGBM Q-regressor: ``Q(s, a)`` over ``build_view(O) + action_family`` (decision D22).

    ``fit`` trains on the base view frame with the observed action appended; :meth:`grid_predict`
    sweeps ``action_family`` over the 8 families (building the view **once** and swapping only the
    action column -- the WS3 ``q_grid`` pattern), returning the ``(n, 8)`` counterfactual Q-grid the
    FQI backup and the greedy consume.
    """

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**DEFAULT_FQI_PARAMS, **(params or {})}
        self.model: LGBMRegressor | None = None

    @staticmethod
    def _with_action(base: pd.DataFrame, action_idx: np.ndarray) -> pd.DataFrame:
        X = base.copy()
        fams = np.array([FAMILIES[i] for i in np.asarray(action_idx, dtype=np.int64)], dtype=object)
        X[ACTION_COL] = pd.Categorical(fams, categories=list(FAMILIES))
        return X

    def fit(self, base: pd.DataFrame, action_idx: np.ndarray, y: np.ndarray) -> "_LGBMQ":
        X = _prep(self._with_action(base, action_idx))
        self.model = LGBMRegressor(**self.params)
        self.model.fit(X, np.asarray(y, dtype=np.float64))
        return self

    def grid_predict(self, base: pd.DataFrame) -> np.ndarray:
        Xp = _prep(base.copy())
        out = np.empty((len(base), _N_FAM), dtype=np.float64)
        for j, fam in enumerate(FAMILIES):
            Xp[ACTION_COL] = pd.Categorical([fam] * len(base), categories=list(FAMILIES))
            out[:, j] = self.model.predict(Xp)
        return out

    @property
    def n_params(self) -> int:
        if self.model is None:
            return 0
        booster = self.model.booster_
        return int(booster.num_trees() * int(self.params.get("num_leaves", 31)))


# -------------------------------------------------------------------------------------
# conservative fitted-Q iteration (decision D48)
# -------------------------------------------------------------------------------------


def _penalized_state_value(q_grid, feasible, penalty, lam) -> np.ndarray:
    r"""Penalised optimal state value ``max_{a in F(s)}[Q(s,a) - lam*pen(s,a)]`` per row.

    A row with **no** feasible action (an empty SPEC-4 mask -- possible for low-history pitchers)
    falls back to the max over *all* actions, so the ``-inf`` feasibility sentinel never leaks into
    the FQI bootstrap target (which would diverge). The pipeline additionally passes an *effective*
    feasible mask (empty rows -> all non-``XX`` families), so this fallback is a defensive guard.
    """
    q = np.asarray(q_grid, dtype=np.float64)
    pen = np.asarray(penalty, dtype=np.float64)
    feas = np.asarray(feasible, dtype=bool)
    scored = q - lam * pen
    v = np.where(feas, scored, _NEG_INF).max(axis=1)
    no_feas = ~feas.any(axis=1)
    if no_feas.any():
        v[no_feas] = scored[no_feas].max(axis=1)
    return v


def _tabular_fit(state_idx, actions, y, n_states, n_actions) -> np.ndarray:
    """Exact per-``(state, action)`` group-mean table with a global-mean fallback (unseen cells)."""
    s = np.asarray(state_idx, dtype=np.int64)
    a = np.asarray(actions, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    g = float(y.mean()) if len(y) else 0.0
    sums = np.zeros((n_states, n_actions), dtype=np.float64)
    counts = np.zeros((n_states, n_actions), dtype=np.float64)
    np.add.at(sums, (s, a), y)
    np.add.at(counts, (s, a), 1.0)
    return np.divide(sums, counts, out=np.full_like(sums, g), where=counts > 0)


def fitted_q_iteration(state_idx, actions, rewards, succ, terminal, feasible, penalty,
                       n_states, n_actions, lam: float = 0.0, n_iter: int = DEFAULT_N_ITER,
                       drift_tol: float = 1e-4) -> tuple[np.ndarray, list[float]]:
    r"""Tabular conservative FQI over a discrete state -- the reusable core (decision D48).

    The exact-tabular engine of the ``count`` FQI rung, also the directly-testable core (it recovers
    the analytic optimal Q on the synth two-step MDP). Iterates the penalised Bellman-optimality
    backup

    .. math:: Q_{k+1}(s,a) \leftarrow r(s,a) + \mathbf{1}[\text{non-terminal}]
              \max_{a'\in\mathcal F(s')}\!\big[Q_k(s',a') - \lambda\,\mathrm{pen}(s',a')\big],

    fitting each ``Q_{k+1}`` as the exact per-``(state, action)`` group-mean of the target (unseen
    cells fall back to the global target mean). Iteration stops at ``n_iter`` backups or when the mean
    ``|Q_{k+1} - Q_k|`` at the taken action drops below ``drift_tol``.

    Parameters
    ----------
    state_idx : array-like of int, shape (n,)
        Discrete state id per decision (``0 .. n_states-1``).
    actions : array-like of int, shape (n,)
        Action index per decision (``0 .. n_actions-1``).
    rewards : array-like, shape (n,)
        Immediate reward per decision.
    succ : array-like of int, shape (n,)
        Successor **row** index per decision (``-1`` for terminal); see :func:`successor_index`.
    terminal : array-like of bool, shape (n,)
        Terminal flag (no bootstrap).
    feasible : array-like of bool, shape (n, n_actions)
        Per-row feasible action mask (restricts the backup's ``max``).
    penalty : array-like, shape (n, n_actions)
        Support penalty grid (:func:`behavior_support_penalty`).
    n_states, n_actions : int
        Table dimensions.
    lam : float, optional
        Penalty coefficient (default 0 -- unpenalised, recovers the analytic optimal Q).
    n_iter : int, optional
        Maximum Bellman backups.
    drift_tol : float, optional
        Early-stop threshold on the taken-action Q drift.

    Returns
    -------
    (numpy.ndarray, list of float)
        The ``(n_states, n_actions)`` Q-table and the per-iteration drift trace.
    """
    state_idx = np.asarray(state_idx, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    succ = np.asarray(succ, dtype=np.int64)
    nonterm = ~np.asarray(terminal, dtype=bool)
    feasible = np.asarray(feasible, dtype=bool)
    penalty = np.asarray(penalty, dtype=np.float64)
    ar = np.arange(len(state_idx))

    Q = _tabular_fit(state_idx, actions, rewards, n_states, n_actions)  # Q_0 = reward regression
    q_grid = Q[state_idx]
    drifts: list[float] = []
    for _ in range(n_iter):
        v_state = _penalized_state_value(q_grid, feasible, penalty, lam)
        y = rewards.copy()
        y[nonterm] = rewards[nonterm] + v_state[succ[nonterm]]
        Q_new = _tabular_fit(state_idx, actions, y, n_states, n_actions)
        q_new = Q_new[state_idx]
        drift = float(np.abs(q_new[ar, actions] - q_grid[ar, actions]).mean())
        drifts.append(drift)
        Q, q_grid = Q_new, q_new
        if drift < drift_tol:
            break
    return Q, drifts


class ConservativeFQI:
    r"""Conservative fitted-Q iteration for the improvement policy (decision D48).

    One state representation (``O`` LightGBM function approximation or ``count`` exact tabular).
    :meth:`fit` iterates the penalised Bellman-optimality backup on the *training* transitions;
    :meth:`q_grid` / :meth:`policy` apply the fitted Q to any rows (typically the held-out eval rows).
    See the module docstring for the backup and the penalty. The ``count`` view delegates to the exact
    :func:`fitted_q_iteration`; the ``O`` view runs the same backup with a LightGBM regressor refit per
    iteration (the counterfactual q-grid is swept by swapping ``action_family``, the WS3 pattern). The
    penalty grid, feasibility and behaviour policy are supplied per row (so ``O`` uses WS3's contextual
    propensities and the O-view feasible set, while ``count`` uses the empirical count-state
    behaviour) -- the class is agnostic to their source.

    Attributes
    ----------
    view : str
        ``"O"`` or ``"count"``.
    lam, floor, n_iter : float, float, int
        Penalty coefficient, support floor, requested Bellman backups.
    diagnostics_ : dict
        :func:`fqi_diagnostics` output (Q drift per iteration, penalty share, support histogram).
    """

    def __init__(self, view: str = "O", lam: float = DEFAULT_LAMBDA,
                 floor: float = DEFAULT_SUPPORT_FLOOR, n_iter: int = DEFAULT_N_ITER,
                 params: dict | None = None, drift_tol: float = 1e-4) -> None:
        if view not in FQI_VIEWS:
            raise ValueError(f"unknown view {view!r}; choose from {FQI_VIEWS}")
        self.view = view
        self.lam = float(lam)
        self.floor = float(floor)
        self.n_iter = int(n_iter)
        self.params = params
        self.drift_tol = float(drift_tol)
        self._reg: _LGBMQ | None = None  # O view
        self._q_table: np.ndarray | None = None  # count view
        self.diagnostics_: dict = {}

    def fit(self, train_table: pd.DataFrame, mu_train, feasible_train,
            reward_col: str = "R") -> "ConservativeFQI":
        """Iterate the penalised Bellman-optimality backup on the training transitions.

        Parameters
        ----------
        train_table : pandas.DataFrame
            Training decision rows (needs ``pa_id``, ``pitch_number``, ``family``, ``reward_col`` and
            the columns :func:`pitchseq.states.build_view` needs for the ``O`` view).
        mu_train : array-like, shape (n, 8)
            Behaviour policy ``mu-hat(.|s)`` per train row (WS3 propensities or empirical counts).
        feasible_train : array-like of bool, shape (n, 8)
            Per-row SPEC 4 feasible mask on the train rows.
        reward_col : str, optional
            Reward column (default ``"R"``).
        """
        table = train_table.reset_index(drop=True)
        actions = family_indices(table)
        rewards = np.nan_to_num(
            pd.to_numeric(table[reward_col], errors="coerce").to_numpy(dtype=np.float64), nan=0.0
        )
        succ, terminal = successor_index(table["pa_id"].to_numpy(), table["pitch_number"].to_numpy())
        penalty = behavior_support_penalty(mu_train, self.floor)
        feasible = np.asarray(feasible_train, dtype=bool)

        if self.view == "count":
            state_idx = count_state(table)
            self._q_table, drifts = fitted_q_iteration(
                state_idx, actions, rewards, succ, terminal, feasible, penalty,
                N_COUNT, _N_FAM, self.lam, self.n_iter, self.drift_tol,
            )
            q_grid = self._q_table[state_idx]
        else:  # O view: LightGBM function approximation
            base, _ = build_view(table, "O")
            base = base.reset_index(drop=True)
            reg = _LGBMQ(self.params).fit(base, actions, rewards)  # Q_0 = reward regression
            q_grid = reg.grid_predict(base)
            drifts = []
            nonterm = ~terminal
            ar = np.arange(len(table))
            for _ in range(self.n_iter):
                v_state = _penalized_state_value(q_grid, feasible, penalty, self.lam)
                y = rewards.copy()
                y[nonterm] = rewards[nonterm] + v_state[succ[nonterm]]
                reg = _LGBMQ(self.params).fit(base, actions, y)
                q_new = reg.grid_predict(base)
                drift = float(np.abs(q_new[ar, actions] - q_grid[ar, actions]).mean())
                drifts.append(drift)
                q_grid = q_new
                if drift < self.drift_tol:
                    break
            self._reg = reg

        self.diagnostics_ = fqi_diagnostics(drifts, penalty, feasible, q_grid, actions)
        return self

    def q_grid(self, table: pd.DataFrame) -> np.ndarray:
        """Counterfactual Q-grid ``Q(s, a)`` for all 8 families on ``table``, shape ``(n, 8)``."""
        table = table.reset_index(drop=True)
        if self.view == "count":
            return self._q_table[count_state(table)]
        base, _ = build_view(table, "O")
        return self._reg.grid_predict(base.reset_index(drop=True))

    def policy(self, table: pd.DataFrame, mu, feasible) -> np.ndarray:
        """The conservative greedy one-hot policy on ``table`` (penalised feasible arg-max)."""
        q = self.q_grid(table)
        pen = behavior_support_penalty(mu, self.floor)
        return conservative_greedy(q, pen, feasible, self.lam)

    @property
    def n_params(self) -> int:
        if self.view == "count":
            return int(N_COUNT * _N_FAM)
        return int(self._reg.n_params) if self._reg is not None else 0


def fqi_diagnostics(drifts, penalty, feasible, q_grid, actions) -> dict:
    r"""FQI convergence / conservatism diagnostics (decision D48).

    Parameters
    ----------
    drifts : sequence of float
        Mean ``|Q_{k+1} - Q_k|`` (at the taken action) per iteration -- the fixed-point convergence
        trace. A decreasing, small tail is convergence; a growing trace flags function-approximation
        divergence (surfaced, not hidden).
    penalty : array-like, shape (n, 8)
        The support penalty grid (``behavior_support_penalty`` output).
    feasible : array-like of bool, shape (n, 8)
        Per-row feasible mask.
    q_grid : array-like, shape (n, 8)
        The final Q-grid (for the greedy-support histogram).
    actions : array-like, shape (n,)
        Observed action indices (unused beyond shape symmetry; kept for signature clarity).

    Returns
    -------
    dict
        ``drift_per_iter``, ``final_drift``, ``n_iter_run``, ``converged`` (final drift below the
        first), ``penalty_share`` (mean fraction of feasible ``(s,a)`` cells that are off-support),
        ``mean_feasible`` (mean feasible actions per row) and ``greedy_support_hist`` (histogram over
        rows of whether the *unpenalised* feasible arg-max is itself off-support -- the share of
        decisions where pessimism actually bites).
    """
    pen = np.asarray(penalty, dtype=np.float64)
    feas = np.asarray(feasible, dtype=bool)
    q = np.asarray(q_grid, dtype=np.float64)
    drifts = [float(d) for d in drifts]
    # Penalty share among feasible cells only (infeasible cells are irrelevant).
    feas_cells = feas.sum()
    penalty_share = float((pen * feas).sum() / feas_cells) if feas_cells > 0 else float("nan")
    # Does the unpenalised feasible arg-max land on an off-support action? (pessimism-bites share)
    q_feas = np.where(feas, q, _NEG_INF)
    greedy_a = q_feas.argmax(axis=1)
    ar = np.arange(len(q))
    greedy_offsupport = pen[ar, greedy_a] > 0
    has_feas = feas.any(axis=1)
    bite = float(greedy_offsupport[has_feas].mean()) if has_feas.any() else float("nan")
    return {
        "drift_per_iter": drifts,
        "final_drift": drifts[-1] if drifts else float("nan"),
        "n_iter_run": len(drifts),
        "converged": bool(len(drifts) >= 2 and drifts[-1] <= drifts[0]),
        "penalty_share": penalty_share,
        "mean_feasible": float(feas.sum(axis=1).mean()),
        "pessimism_bites_frac": bite,
        "greedy_support_hist": {
            "n_rows": int(len(q)),
            "n_greedy_offsupport": int(greedy_offsupport[has_feas].sum()),
        },
    }


# -------------------------------------------------------------------------------------
# exploitability read-out (decision D49, SPEC 10)
# -------------------------------------------------------------------------------------


class BatterResponseModel:
    r"""A fixed, interpretable tabular batter swing/take response (decision D49, SPEC 10).

    Models ``P(swing | count, family, location-bucket)`` from **train** data as beta-smoothed
    empirical swing rates. ``swing`` is the ready-made ``is_swing`` label (batter offered at the
    pitch: whiff / foul / in-play), ``count`` is the 12-cell ``balls x strikes``, ``family`` the 8
    pitch families, and the ``location-bucket`` is three bands of the batter-relative vertical
    location ``exec_plate_z_norm`` (low / middle / high, plus an out-of-band catch-all).

    The location bucket is an *execution* quantity, so it is used only to **fit** this descriptive
    opponent model on train and is **marginalised away at prediction** (a pitch's location is not
    known pre-decision): the game payoff uses the family's train location-bucket mixture, which by
    total probability equals the raw ``(count, family)`` swing rate. :meth:`swing_prob` therefore
    returns the pre-pitch, leakage-safe marginal ``P(swing | count, family)``; the by-bucket table is
    retained for interpretability (:attr:`by_bucket_`). The model is **fixed** (fit once on train,
    never adapted) -- the "fixed interpretable batter response" SPEC 10 asks for.

    Attributes
    ----------
    swing_table_ : numpy.ndarray, shape (12, 8)
        Marginal ``P(swing | count, family)`` (the game input).
    by_bucket_ : numpy.ndarray, shape (12, 8, n_buckets)
        By-location-bucket swing rates (interpretability only).
    prior_swing_ : float
        Global swing rate (the smoothing target / fallback for empty cells).
    """

    N_BUCKETS = 4  # low, middle, high, out-of-band

    def __init__(self, alpha: float = 5.0) -> None:
        self.alpha = float(alpha)  # beta-smoothing pseudo-count toward the global swing rate
        self.swing_table_ = np.full((N_COUNT, _N_FAM), 0.5, dtype=np.float64)
        self.by_bucket_ = np.full((N_COUNT, _N_FAM, self.N_BUCKETS), 0.5, dtype=np.float64)
        self.prior_swing_ = 0.5

    @staticmethod
    def _loc_bucket(z_norm) -> np.ndarray:
        """Vertical-location bucket from ``exec_plate_z_norm``: 0 low, 1 middle, 2 high, 3 out."""
        z = pd.to_numeric(z_norm, errors="coerce").to_numpy(dtype=np.float64)
        out = np.full(len(z), 3, dtype=np.int64)  # out-of-band / missing
        out[(z >= 0.0) & (z < 1.0 / 3)] = 0
        out[(z >= 1.0 / 3) & (z < 2.0 / 3)] = 1
        out[(z >= 2.0 / 3) & (z <= 1.0)] = 2
        return out

    def fit(self, train_table: pd.DataFrame) -> "BatterResponseModel":
        """Fit the swing-rate tables from train ``is_swing`` (falls back to deriving swing from
        ``outcome1`` = whiff/foul/in_play when ``is_swing`` is absent)."""
        t = train_table
        if "is_swing" in t.columns:
            swing = t["is_swing"].astype(bool).to_numpy()
        else:
            o1 = t["outcome1"].astype(object).to_numpy()
            swing = np.isin(o1, ("whiff", "foul", "in_play"))
        counts = count_state(t)
        fams = family_indices(t)
        buckets = self._loc_bucket(t["exec_plate_z_norm"]) if "exec_plate_z_norm" in t.columns \
            else np.zeros(len(t), dtype=np.int64)
        self.prior_swing_ = float(swing.mean()) if len(swing) else 0.5

        # Marginal (count, family) beta-smoothed swing rate.
        n_sw = np.zeros((N_COUNT, _N_FAM)); n_tot = np.zeros((N_COUNT, _N_FAM))
        np.add.at(n_sw, (counts, fams), swing.astype(np.float64))
        np.add.at(n_tot, (counts, fams), 1.0)
        self.swing_table_ = (n_sw + self.alpha * self.prior_swing_) / (n_tot + self.alpha)

        # By-bucket (interpretability).
        b_sw = np.zeros((N_COUNT, _N_FAM, self.N_BUCKETS)); b_tot = np.zeros((N_COUNT, _N_FAM, self.N_BUCKETS))
        np.add.at(b_sw, (counts, fams, buckets), swing.astype(np.float64))
        np.add.at(b_tot, (counts, fams, buckets), 1.0)
        self.by_bucket_ = (b_sw + self.alpha * self.prior_swing_) / (b_tot + self.alpha)
        return self

    def swing_prob(self, count_ids=None, family_ids=None) -> np.ndarray:
        """Marginal ``P(swing | count, family)``.

        With no arguments returns the full ``(12, 8)`` table; with ``count_ids`` (and optionally
        ``family_ids``) returns the per-row swing propensity."""
        if count_ids is None:
            return self.swing_table_.copy()
        c = np.asarray(count_ids, dtype=np.int64)
        if family_ids is None:
            return self.swing_table_[c]  # (n, 8) over families
        return self.swing_table_[c, np.asarray(family_ids, dtype=np.int64)]


def payoff_by_count(q_by_count, swing_by_count, beta: float = DEFAULT_DISRUPTION_BETA) -> np.ndarray:
    r"""Per-count zero-sum payoff games ``R(s, a, k)`` (decision D49, SPEC 10).

    A per-count-state game between the **pitcher** (row player, 8 families, maximising run value) and
    the **batter** (column player, 8 "sit-on-family" guesses, minimising it). Throwing family ``a``
    against a batter sitting on family ``k`` pays the pitcher

    .. math:: R(s, a, k) = \hat q(s, a) \;-\; \beta\,p_{\text{swing}}(s, a)\,\mathbf{1}[a = k],

    where :math:`\hat q(s,a)` is the common outcome model's ``E[R|s,a]`` (the average-batter reward)
    and the anticipation penalty applies only when the batter guessed the thrown family (``a = k``):
    a correctly-anticipating batter converts more of their swings into damage, so the pitcher's
    reward drops by ``beta`` scaled by the family's swing propensity :math:`p_{\text{swing}}(s,a)`
    (from the fixed :class:`BatterResponseModel`). ``beta`` is the fixed disruption coefficient. The
    game is model-dependent by necessity (SPEC 10) -- reported as a light game-theoretic capstone.

    A predictable pitcher (mass on one family) lets the batter sit on it and eat the full penalty; an
    unpredictable (mixed) pitcher spreads the anticipation risk -- so the equilibrium is mixed and
    concentrated policies are exploitable.

    Parameters
    ----------
    q_by_count : array-like, shape (12, 8)
        Per-count mean outcome value ``E[R|s,a]`` (aggregated from the row-level q-grid).
    swing_by_count : array-like, shape (12, 8)
        Per-count swing propensity ``P(swing|count,family)`` (:meth:`BatterResponseModel.swing_prob`).
    beta : float, optional
        Anticipation-disruption coefficient (default :data:`DEFAULT_DISRUPTION_BETA`).

    Returns
    -------
    numpy.ndarray, shape (12, 8, 8)
        ``payoff[c, a, k]`` -- the pitcher's expected run value in count ``c`` throwing ``a`` against
        a batter sitting on ``k``.
    """
    q = np.asarray(q_by_count, dtype=np.float64)  # (C, A)
    sw = np.asarray(swing_by_count, dtype=np.float64)  # (C, A)
    C, A = q.shape
    payoff = np.repeat(q[:, :, None], A, axis=2)  # (C, A, K): base q̂(s,a), constant over k
    diag = np.arange(A)
    payoff[:, diag, diag] -= beta * sw[:, diag]  # anticipation penalty on the matched family
    return payoff


def payoff_matrices(table: pd.DataFrame, q_grid, response_model: "BatterResponseModel",
                    beta: float = DEFAULT_DISRUPTION_BETA) -> tuple[np.ndarray, np.ndarray]:
    r"""Per-state (per-count) 8x8 payoff games from the outcome model + the batter response (D49).

    The named SPEC 10 / D49 entry point: aggregate the row-level outcome ``q-hat`` grid (the common
    outcome model's ``E[R|s,a]`` -- WS3's grid) and the fixed :class:`BatterResponseModel`'s swing
    propensity to per-count means, then build the per-count zero-sum games with
    :func:`payoff_by_count`. Aggregating to the 12 count states (rather than solving a game per row)
    keeps the equilibrium LP count small and the game interpretable, while every decision row is still
    scored in *its own count's* game by :func:`exploitability`.

    Parameters
    ----------
    table : pandas.DataFrame
        Decision rows (needs ``balls`` / ``strikes`` for the count state).
    q_grid : array-like, shape (n, 8)
        Per-row outcome model ``E[R|s,a]`` for all 8 families (the common ``q-hat``).
    response_model : BatterResponseModel
        The fitted fixed batter response (supplies ``P(swing|count,family)``).
    beta : float, optional
        Anticipation-disruption coefficient (default :data:`DEFAULT_DISRUPTION_BETA`).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``payoffs`` of shape ``(12, 8, 8)`` (the per-count games) and ``count_per_row`` of shape
        ``(n,)`` (each row's count-state id, for :func:`exploitability`).
    """
    counts = count_state(table)
    q = np.asarray(q_grid, dtype=np.float64)
    q_by_c = np.zeros((N_COUNT, _N_FAM), dtype=np.float64)
    cnt = np.zeros(N_COUNT, dtype=np.float64)
    np.add.at(q_by_c, counts, q)
    np.add.at(cnt, counts, 1.0)
    q_by_c = np.divide(q_by_c, cnt[:, None], out=np.zeros_like(q_by_c), where=cnt[:, None] > 0)
    payoffs = payoff_by_count(q_by_c, response_model.swing_prob(), beta=beta)
    return payoffs, counts


def equilibrium_value(payoff) -> dict:
    r"""Maximin value + optimal pitcher strategy of one zero-sum game by LP (``scipy.linprog``).

    Solves the row player's (pitcher, maximiser) program

    .. math:: \max_{x,\,v}\; v \quad\text{s.t.}\quad \sum_a x_a\,R[a,k] \ge v\ \forall k,\;
              \sum_a x_a = 1,\; x \ge 0,

    cast for ``linprog`` (a minimiser) as ``min -v`` with variables ``[x_0..x_{A-1}, v]``, the column
    constraints ``-R[:,k]^T x + v <= 0`` and the simplex equality. By the minimax theorem the optimal
    ``v`` is the game value (equal to the batter's minimax); a fictitious-play iteration would
    converge to the same value and is the documented alternative when an LP solver is unavailable.

    Returns
    -------
    dict
        ``value`` (the equilibrium run value), ``strategy`` (the ``(A,)`` optimal pitcher mixed
        strategy) and ``status`` (the solver status; on failure ``value`` falls back to the pure
        maximin ``max_a min_k R[a,k]`` and ``strategy`` to its arg-max one-hot).
    """
    R = np.asarray(payoff, dtype=np.float64)
    A, K = R.shape
    # Variables: x (A) then v (1). Objective: minimise -v.
    c = np.zeros(A + 1); c[-1] = -1.0
    # Column constraints: for each k, -sum_a x_a R[a,k] + v <= 0.
    A_ub = np.zeros((K, A + 1)); A_ub[:, :A] = -R.T; A_ub[:, A] = 1.0
    b_ub = np.zeros(K)
    A_eq = np.zeros((1, A + 1)); A_eq[0, :A] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * A + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        x = np.clip(res.x[:A], 0.0, None)
        x = x / x.sum() if x.sum() > 0 else np.full(A, 1.0 / A)
        return {"value": float(res.x[A]), "strategy": x, "status": "optimal"}
    # Fallback: pure maximin (never expected on a finite game, but keeps the read-out defined).
    row_min = R.min(axis=1)
    a_star = int(row_min.argmax())
    x = np.zeros(A); x[a_star] = 1.0
    return {"value": float(row_min[a_star]), "strategy": x, "status": f"linprog:{res.status}"}


def policy_exploitability(policy_probs, payoff, eq_value: float | None = None) -> float:
    r"""Exploitability of one pitcher policy in one game: ``V_eq - min_k (pi^T R)_k`` (>= 0).

    If the pitcher commits to mixed strategy ``pi``, the batter best-responds by the column minimising
    the pitcher's expected value, so the pitcher's guaranteed value is ``min_k (pi^T R)_k``. The
    exploitability is the shortfall from the equilibrium value ``V_eq`` -- zero for a maximin-optimal
    (equilibrium) policy, positive and growing for predictable / concentrated policies. ``eq_value``
    is computed via :func:`equilibrium_value` when not supplied.

    Parameters
    ----------
    policy_probs : array-like, shape (A,)
        The pitcher's mixed strategy over families (need not exclude ``XX``; a zero there is typical).
    payoff : array-like, shape (A, K)
        The game payoff (:func:`payoff_by_count` slice for the relevant count).
    eq_value : float, optional
        Precomputed equilibrium value (avoids re-solving the LP per policy).

    Returns
    -------
    float
        The (non-negative) exploitability; small negative numerical noise is clipped to 0.
    """
    pi = np.asarray(policy_probs, dtype=np.float64)
    R = np.asarray(payoff, dtype=np.float64)
    if eq_value is None:
        eq_value = equilibrium_value(R)["value"]
    best_response_value = float((pi @ R).min())  # batter best-responds (minimiser)
    return float(max(eq_value - best_response_value, 0.0)) + 0.0  # +0.0 normalises -0.0 -> 0.0


def exploitability(policy_rows, payoff_by_count_arr, count_per_row, eq_values=None,
                   cluster_ids=None, n_boot: int = 0, seed: int = 0) -> dict:
    r"""Average exploitability of a per-row policy against the per-count equilibrium (SPEC 10).

    Each decision row is scored in *its own count's* game: the row's exploitability is
    ``V_eq(count) - min_k (pi_row^T R_count)_k`` (:func:`policy_exploitability`). The read-out is the
    mean over rows, optionally with a cluster (pitcher-game) bootstrap CI.

    Parameters
    ----------
    policy_rows : array-like, shape (n, 8)
        Per-row pitcher policy.
    payoff_by_count_arr : array-like, shape (12, 8, 8)
        The per-count games (:func:`payoff_by_count`).
    count_per_row : array-like, shape (n,)
        Each row's count-state id (:func:`count_state`).
    eq_values : array-like, shape (12,), optional
        Precomputed per-count equilibrium values (computed here when ``None``).
    cluster_ids : array-like, shape (n,), optional
        Cluster ids for the bootstrap (pitcher-game); no CI when ``None`` / ``n_boot < 1``.
    n_boot : int, optional
        Cluster-bootstrap replicates for the CI (default 0 -- point only).
    seed : int, optional
        Bootstrap seed.

    Returns
    -------
    dict
        ``mean`` (average exploitability), ``per_row`` (the per-row array), ``lower_95`` / ``ci95``
        (cluster-bootstrap bounds, ``NaN`` when no bootstrap), ``n``, ``eq_values`` and
        ``by_count`` (mean exploitability per count).
    """
    pol = np.asarray(policy_rows, dtype=np.float64)
    payoffs = np.asarray(payoff_by_count_arr, dtype=np.float64)
    counts = np.asarray(count_per_row, dtype=np.int64)
    C = payoffs.shape[0]
    if eq_values is None:
        eq_values = np.array([equilibrium_value(payoffs[c])["value"] for c in range(C)])
    eq_values = np.asarray(eq_values, dtype=np.float64)
    # Per-row exploitability: V_eq(c) - min_k (pi_row @ R_c)_k.
    per_row = np.empty(len(pol), dtype=np.float64)
    for c in range(C):
        m = counts == c
        if not m.any():
            continue
        # (n_c, K) = pi_rows (n_c, A) @ R_c (A, K); best response = min over k.
        br = (pol[m] @ payoffs[c]).min(axis=1)
        per_row[m] = np.maximum(eq_values[c] - br, 0.0)
    mean = float(per_row.mean()) if len(per_row) else float("nan")

    lower_95 = float("nan"); ci95 = [float("nan"), float("nan")]
    if n_boot >= 1 and cluster_ids is not None:
        from pitchseq.splits import cluster_bootstrap_indices
        frame = pd.DataFrame({"pg": np.asarray(cluster_ids)})
        reps = [float(per_row[np.asarray(idx)].mean())
                for idx in cluster_bootstrap_indices(frame, cluster="pg", n_boot=n_boot, seed=seed)]
        reps = np.asarray([r for r in reps if np.isfinite(r)], dtype=np.float64)
        if reps.size:
            lo, hi = np.percentile(reps, [2.5, 97.5])
            lower_95 = float(np.percentile(reps, 5.0)); ci95 = [float(lo), float(hi)]

    by_count = {}
    for c in range(C):
        m = counts == c
        if m.any():
            by_count[int(c)] = float(per_row[m].mean())
    return {"mean": mean, "per_row": per_row, "lower_95": lower_95, "ci95": ci95,
            "n": int(len(per_row)), "eq_values": eq_values, "by_count": by_count}


# -------------------------------------------------------------------------------------
# the study frontier (decision D49, SPEC 10)
# -------------------------------------------------------------------------------------

#: The frontier schema -- one row per policy across the whole study (SPEC 10 final figure).
FRONTIER_COLUMNS = [
    "policy_id", "ope_value", "ope_lower95", "b_seq_bits",
    "exploitability", "tv_from_behavior", "params", "wall_clock_s",
]


def assemble_frontier(entries) -> pd.DataFrame:
    r"""Assemble the study frontier table from per-policy entries (decision D49, SPEC 10).

    Each entry is a dict with (a superset of) :data:`FRONTIER_COLUMNS`. Entries come from every
    policy in the study -- the behaviour policy, the WS4 bandit, the WS5 MDP designs and the WS7
    conservative-FQI policies at each ``alpha`` -- so the frontier is the study-wide trade-off of OPE
    run value against predictability-in-bits (``b_seq_bits`` -- the ordered-history bits the policy's
    state view exploits, per SPEC 10; ``0`` for context/count-only policies), exploitability,
    distance-from-behaviour (``tv_from_behavior``) and compute (``params`` / ``wall_clock_s``).

    Parameters
    ----------
    entries : sequence of dict
        Per-policy records; missing frontier columns are filled with ``NaN``.

    Returns
    -------
    pandas.DataFrame
        Columns exactly :data:`FRONTIER_COLUMNS` (plus any extra keys), one row per entry, in input
        order.
    """
    rows = []
    for e in entries:
        row = {c: e.get(c, np.nan) for c in FRONTIER_COLUMNS}
        for k, v in e.items():  # keep any extra annotations (e.g. view, alpha)
            if k not in row:
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered = FRONTIER_COLUMNS + [c for c in df.columns if c not in FRONTIER_COLUMNS]
    return df[ordered]


def write_frontier_csv(df: pd.DataFrame, path) -> Path:
    """Persist the frontier table as a plotting-ready CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def plot_frontier(df: pd.DataFrame, path, title: str = "WS7 study frontier") -> Path:
    r"""The final frontier figure of the study (decision D49, SPEC 10) -- a labelled two-panel scatter.

    Renders with the non-interactive ``Agg`` backend (headless pipeline) and saves a PNG. Two panels
    over the shared policy set:

    * **left** -- OPE run value (y, with the one-sided 95% lower bound as a down-error whisker) vs
      distance-from-behaviour ``tv_from_behavior`` (x); point colour = exploitability. The
      prescriptive trade-off: value bought with deviation from behaviour and predictability cost.
    * **right** -- exploitability (y) vs predictability-in-bits ``b_seq_bits`` (x); point size scales
      with ``params`` (compute). The game-theoretic capstone axis.

    Every policy is a labelled point in both panels, so all five SPEC 10 axes (value, bits,
    exploitability, distance, compute) are legible in one figure.

    Parameters
    ----------
    df : pandas.DataFrame
        A frontier table (:func:`assemble_frontier`).
    path : str or pathlib.Path
        Output PNG path.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display, write straight to a file
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    val = pd.to_numeric(d["ope_value"], errors="coerce").to_numpy()
    lo = pd.to_numeric(d["ope_lower95"], errors="coerce").to_numpy()
    tv = pd.to_numeric(d["tv_from_behavior"], errors="coerce").to_numpy()
    bits = pd.to_numeric(d["b_seq_bits"], errors="coerce").to_numpy()
    expl = pd.to_numeric(d["exploitability"], errors="coerce").to_numpy()
    params = pd.to_numeric(d["params"], errors="coerce").to_numpy()
    labels = d["policy_id"].astype(str).to_numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))

    # Left: value vs distance-from-behaviour, coloured by exploitability, lower-95 whisker.
    yerr = np.clip(val - lo, 0.0, None)
    finite_expl = expl[np.isfinite(expl)]
    sc = axL.scatter(tv, val, c=expl, cmap="viridis", s=90, zorder=3,
                     vmin=(finite_expl.min() if finite_expl.size else 0.0),
                     vmax=(finite_expl.max() if finite_expl.size else 1.0), edgecolors="k", linewidths=0.5)
    axL.errorbar(tv, val, yerr=[yerr, np.zeros_like(yerr)], fmt="none", ecolor="0.5",
                 elinewidth=1.0, capsize=2, zorder=2)
    for x, y, lab in zip(tv, val, labels):
        if np.isfinite(x) and np.isfinite(y):
            axL.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axL.set_xlabel("total-variation distance from behavior")
    axL.set_ylabel("OPE run value (whisker = one-sided 95% lower bound)")
    axL.set_title("value vs deviation from behavior")
    axL.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=axL, fraction=0.046, pad=0.04)
    cb.set_label("exploitability")

    # Right: exploitability vs predictability-in-bits, size = params (compute).
    ps = params.astype(np.float64)
    finite_ps = ps[np.isfinite(ps) & (ps > 0)]
    if finite_ps.size:
        sizes = 40 + 260 * (np.nan_to_num(ps, nan=0.0) - finite_ps.min()) / (np.ptp(finite_ps) + 1e-9)
    else:
        sizes = np.full(len(ps), 90.0)
    axR.scatter(bits, expl, s=sizes, c="#4C72B0", alpha=0.8, edgecolors="k", linewidths=0.5, zorder=3)
    for x, y, lab in zip(bits, expl, labels):
        if np.isfinite(x) and np.isfinite(y):
            axR.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    axR.set_xlabel("predictability in bits $B_{seq}$ (ordered-history bits the view exploits)")
    axR.set_ylabel("average exploitability vs equilibrium")
    axR.set_title("exploitability vs predictability (point size = params)")
    axR.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# -------------------------------------------------------------------------------------
# persistence (house pattern: joblib blobs + JSON sidecar)
# -------------------------------------------------------------------------------------


def save_ws7_artifacts(out_dir, world_tag: str, fqi_policies: dict | None = None,
                       response_model: "BatterResponseModel | None" = None) -> dict:
    """Persist the fitted WS7 pieces (FQI Q-models per view + the batter-response model).

    Parameters
    ----------
    out_dir : str or pathlib.Path
        Output directory.
    world_tag : str
        Tag for the filenames (``real`` / ``null`` / ``positive``).
    fqi_policies : dict, optional
        ``view -> ConservativeFQI`` fitted models.
    response_model : BatterResponseModel, optional
        The fitted response model.

    Returns
    -------
    dict
        The written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for view, fqi in (fqi_policies or {}).items():
        p = out_dir / f"fqi_{world_tag}_{view}.joblib"
        joblib.dump(fqi, p)
        written[f"fqi_{view}"] = str(p)
    if response_model is not None:
        p = out_dir / f"response_{world_tag}.joblib"
        joblib.dump(response_model, p)
        written["response"] = str(p)
    return written


def load_ws7_artifacts(directory, world_tag: str) -> dict:
    """Load persisted WS7 FQI models + response model for ``world_tag`` (companion to
    :func:`save_ws7_artifacts`)."""
    directory = Path(directory)
    fqi: dict = {}
    for p in sorted(directory.glob(f"fqi_{world_tag}_*.joblib")):
        view = p.stem.split("_")[-1]
        fqi[view] = joblib.load(p)
    response = None
    rp = directory / f"response_{world_tag}.joblib"
    if rp.is_file():
        response = joblib.load(rp)
    return {"fqi": fqi, "response": response, "directory": str(directory)}
