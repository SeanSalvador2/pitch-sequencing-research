r"""GBDT behavior model and decomposed outcome stack (SPEC ``12.3``; decisions D32-D35).

This module is the WS3 centerpiece. It supplies three reusable pieces, each fitted per
state view (all five: ``C`` / ``U`` / ``L1`` / ``O`` / ``OM``):

1. **Behavior model** ``mu(a | s)`` -- a multiclass LightGBM over the 8 pitch families
   (:data:`~pitchseq.families.FAMILIES`), trained on the view matrices from
   :func:`pitchseq.states.build_view`. It answers *selection* (finding #1): does ordered
   history sharpen the next-pitch prediction? It carries **no current-pitch execution
   feature** -- the views guarantee this and :func:`~pitchseq.decision_table.leakage_audit`
   is asserted anyway.

2. **Decomposed outcome model** (decision D32) -- an interpretable event-tree run-value
   estimator assembled from three separately-fitted pieces, each conditioned on the current
   action (decision D22, ``action_family`` appended to every view):

   * **Stage A** ``P(outcome1 | s, a)`` -- multiclass LightGBM over the 6 level-1 outcomes
     :data:`~pitchseq.outcomes.OUTCOME1` ``(ball, called_strike, whiff, foul, hbp, in_play)``.
   * **Stage B** ``P(outcome2 | s, a)`` -- multiclass LightGBM over the 5 level-2 outcomes
     :data:`~pitchseq.outcomes.OUTCOME2` ``(single, double, triple, home_run, out_or_other)``,
     trained on the **in-play rows only** but evaluable for any row (the counterfactual
     "if put in play, what happens").
   * **Stage C** count-conditional node values ``V(node, balls, strikes)`` -- the train-fold
     mean reward for each event-tree node at each count, a lookup with a global-node
     fallback (:class:`NodeValueTable`).

   **Assembly (the exact formula).** With the count ``c = (balls, strikes)``, the leaves of
   the two-level event tree are the 5 non-in-play level-1 outcomes plus the 5 level-2
   refinements of ``in_play``, and the expected reward is

   .. math::

      \mathbb{E}[R \mid s, a, c] =
        \sum_{k \in \{\text{ball},\text{called\_strike},\text{whiff},\text{foul},\text{hbp}\}}
             P_A(o_1 = k \mid s, a)\; V_1(k, c)
        \;+\; P_A(o_1 = \text{in\_play} \mid s, a)
             \sum_{j \in \text{OUTCOME2}} P_B(o_2 = j \mid s, a)\; V_2(j, c),

   where :math:`V_1(k, c)` is the count-conditional mean reward of level-1 node ``k`` and
   :math:`V_2(j, c)` the count-conditional mean reward of level-2 node ``j`` (each with a
   global-node fallback when the ``(node, count)`` cell is empty). This factorization keeps
   every piece interpretable and independently reusable by WS4/WS5/WS7 (decision D33).

   **Predictive uncertainty (``exp_reward_sd``, residual-based).** Treating the event-tree
   leaf as a latent node, the law of total variance gives the per-row reward variance

   .. math::

      \mathrm{Var}[R \mid s, a, c] =
        \sum_{\ell} p_\ell\,\big(S_\ell(c) + V_\ell(c)^2\big) - \mathbb{E}[R \mid s, a, c]^2,

   summed over the 10 leaves :math:`\ell` (leaf probability :math:`p_\ell` and value
   :math:`V_\ell` as in the assembly), where :math:`S_\ell(c)` is the train-fold **within-node
   reward variance**. ``exp_reward_sd`` is :math:`\sqrt{\max(\mathrm{Var}, 0)}` -- a pure
   residual-based uncertainty, no bootstrap needed.

3. **Direct** ``E[R | s, a]`` **regressor** -- a LightGBM regression on the same
   (view + action) features (decision D32 cross-check). :meth:`OutcomeStack.disagreement`
   reports ``mean |decomposed - direct|`` and flags it when large.

Two derived artifacts feed the prescriptive workstreams (decision D33):

* :func:`q_grid` -- the counterfactual value grid ``qhat(s, a)`` for all 8 families per row
  (swap the action feature, re-run stages A/B, re-assemble).
* :func:`propensities` -- the behavior propensities ``mu(a | s)`` per row.

Both, plus the fitted boosters and node-value lookups, are persisted under an output
directory and reloaded by :func:`load_ws3_artifacts` (imported by WS4/5/7).

Hyperparameters (decision D34)
==============================

:data:`DEFAULT_PARAMS` is the predeclared LightGBM default; :data:`HP_GRID` is a small
``num_leaves x min_child_samples x learning_rate`` grid (``<= 12`` combos), identical across
views. When tuning is enabled a model selects its grid combo by **validation log loss** on an
internal holdout carved from the training seasons (never the reported validation / test
folds), then refits on the full train fold; the chosen params are logged.

LightGBM is an optional dependency (the ``ml`` extra). It is imported at module load with a
clear install hint if absent (see the module note below): the core ``pitchseq`` package and
WS1/WS2 are deliberately LightGBM-free, so WS3 opts into it via ``pip install -e '.[ml]'``
(the RUNBOOK's environment step already installs the extra).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:  # LightGBM is the WS3 core dependency, gated behind the ``ml`` extra.
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "Workstream 3 requires LightGBM. Install the ML extra:\n"
        "    pip install -e '.[ml]'\n"
        "(the core pitchseq package and WS1/WS2 are intentionally LightGBM-free)."
    ) from exc

import joblib

from pitchseq.decision_table import leakage_audit
from pitchseq.families import FAMILIES
from pitchseq.outcomes import OUTCOME1, OUTCOME2
from pitchseq.states import STATE_VIEWS, build_view

__all__ = [
    "BEHAVIOR_VIEWS",
    "OUTCOME_VIEWS",
    "DEFAULT_PARAMS",
    "HP_GRID",
    "ACTION_COL",
    "DISAGREEMENT_FLAG_ABS",
    "BehaviorModel",
    "NodeValueTable",
    "OutcomeStack",
    "WS3Artifacts",
    "q_grid",
    "propensities",
    "save_ws3_artifacts",
    "load_ws3_artifacts",
    "make_ws3_model_factory",
    "select_hyperparams",
    "outcome_view_matrix",
]

# --- constants -------------------------------------------------------------------------

#: WS3 runs every one of the five nested views (the first workstream that can -- D34).
BEHAVIOR_VIEWS = tuple(STATE_VIEWS)
OUTCOME_VIEWS = tuple(STATE_VIEWS)

#: The current-action feature appended to every outcome view (decision D22).
ACTION_COL = "action_family"

_N_FAM = len(FAMILIES)
_O1 = list(OUTCOME1)
_O2 = list(OUTCOME2)
_I_INPLAY = _O1.index("in_play")
_NON_INPLAY = [i for i in range(len(_O1)) if i != _I_INPLAY]

#: Predeclared LightGBM defaults (decision D34). ``num_threads`` / ``deterministic`` are set
#: for reproducibility in tests; the CLI may raise ``num_threads`` for speed on real data.
DEFAULT_PARAMS: dict = {
    "num_leaves": 31,
    "min_child_samples": 50,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "max_depth": -1,
    "reg_lambda": 1.0,
    "num_threads": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
    "random_state": 20260713,
}

#: The validation grid (decision D34): num_leaves x min_child_samples x learning_rate = 12
#: combos, identical across views. Selection is by validation log loss.
HP_GRID: dict = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 100],
    "learning_rate": [0.03, 0.1],
}

#: Rows above which the HP grid subsamples the training fold for the search (the final model
#: is always refit on the full fold). Keeps the 12-combo search cheap on the full data.
HP_SAMPLE_CAP = 250_000

#: ``mean |decomposed - direct|`` above this absolute reward magnitude flags the D32
#: cross-check as a material disagreement (the reward scale is |R| ~ 0.05-0.3).
DISAGREEMENT_FLAG_ABS = 0.03

_EARLY_STOPPING_ROUNDS = 30


# --- dtype prep + probability alignment ------------------------------------------------

def _prep(X: pd.DataFrame) -> pd.DataFrame:
    """Coerce a state-view frame to LightGBM-friendly dtypes (categoricals preserved).

    Object columns become ``category`` (LightGBM native categorical handling), booleans
    become ``int8`` and everything else becomes ``float64``. Existing ``category`` columns
    (the view's family / outcome / pitch-type tokens and the appended ``action_family``) are
    left untouched so their integer codes -- and therefore predictions -- are stable across
    a save / load round-trip.
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


def _align(proba: np.ndarray, classes, labels: list) -> np.ndarray:
    """Widen a ``(n, n_observed)`` probability matrix to ``(n, len(labels))`` in label order.

    Classes the model never saw in training get a zero column. The observed classes already
    sum to 1 per row, so the widened rows still sum to 1.
    """
    proba = np.asarray(proba, dtype=np.float64)
    col = {c: i for i, c in enumerate(list(classes))}
    out = np.zeros((proba.shape[0], len(labels)), dtype=np.float64)
    for j, lab in enumerate(labels):
        if lab in col:
            out[:, j] = proba[:, col[lab]]
    return out


# --- thin LightGBM wrappers (predict via the sklearn estimator; joblib-persisted) ------

class _RawClassifier:
    """A LightGBM multiclass classifier exposing ``fit`` / ``predict_proba`` / ``classes_``.

    ``predict_proba`` returns the ``(n, n_observed_classes)`` matrix in ``classes_`` order --
    the raw contract the shared falsification code (:mod:`pitchseq.eval.falsification`)
    re-aligns via ``classes_``. Higher-level WS3 models widen it to a fixed label order with
    :func:`_align`.
    """

    def __init__(self, **params) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self.model: LGBMClassifier | None = None
        self.classes_ = None
        self.best_iteration_: int | None = None

    def fit(self, X, y, eval_set=None):
        self.model = LGBMClassifier(**self.params)
        callbacks = None
        if eval_set is not None:
            ev = [(_prep(ex), ey) for ex, ey in eval_set]
            callbacks = [lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)]
            self.model.fit(_prep(X), np.asarray(y), eval_set=ev, eval_metric="multi_logloss", callbacks=callbacks)
        else:
            self.model.fit(_prep(X), np.asarray(y))
        self.classes_ = self.model.classes_
        self.best_iteration_ = getattr(self.model, "best_iteration_", None)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(_prep(X))

    def feature_gain(self) -> dict:
        booster = self.model.booster_
        names = booster.feature_name()
        gains = booster.feature_importance(importance_type="gain")
        return {n: float(g) for n, g in zip(names, gains)}

    @property
    def n_params(self) -> int:
        booster = self.model.booster_
        return int(booster.num_trees() * int(self.params.get("num_leaves", 31)))


class _RawRegressor:
    """A LightGBM regressor exposing ``fit`` / ``predict`` (the direct E[R|s,a] cross-check)."""

    def __init__(self, **params) -> None:
        self.params = {**DEFAULT_PARAMS, **params, "objective": "regression"}
        self.model: LGBMRegressor | None = None

    def fit(self, X, y):
        self.model = LGBMRegressor(**self.params)
        self.model.fit(_prep(X), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(_prep(X)), dtype=np.float64)


# --- hyperparameter selection (decision D34) -------------------------------------------

def _carve_tune_holdout(table: pd.DataFrame, min_rows: int = 200) -> np.ndarray | None:
    """Boolean mask for an internal HP-validation holdout carved from the training fold.

    The latest training season is held out when at least two seasons are present and both
    sides are large enough; otherwise a deterministic ``game_pk`` hash (25%) is used. Returns
    ``None`` when neither side clears ``min_rows`` (too little data to tune -- use defaults).
    The reported validation / test seasons are never touched: this holdout is entirely inside
    the train fold.
    """
    n = len(table)
    if "season" in table.columns and table["season"].nunique() >= 2:
        latest = int(table["season"].max())
        hold = (table["season"].to_numpy() == latest)
        if hold.sum() >= min_rows and (n - hold.sum()) >= min_rows:
            return hold
    gp = pd.to_numeric(table["game_pk"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    hold = (gp % 4 == 0)
    if hold.sum() >= min_rows and (n - hold.sum()) >= min_rows:
        return hold
    return None


def _subsample_rows(n: int, cap: int, seed: int) -> np.ndarray:
    """Deterministic row positions capped at ``cap`` (all rows when ``n <= cap``)."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


def select_hyperparams(
    X_tr: pd.DataFrame,
    y_tr,
    X_val: pd.DataFrame,
    y_val,
    base_params: dict,
    grid: dict = HP_GRID,
    seed: int = 0,
) -> tuple[dict, list[dict]]:
    """Select a grid combo by validation multiclass log loss (decision D34).

    Each combo is trained on ``(X_tr, y_tr)`` with early stopping against ``(X_val, y_val)``;
    the combo with the lowest validation ``multi_logloss`` (at its best iteration) wins. The
    returned params carry ``n_estimators`` set to that best iteration.

    Parameters
    ----------
    X_tr, y_tr : training features / labels for the candidate fits.
    X_val, y_val : the internal HP-validation holdout.
    base_params : dict
        Params every combo starts from (:data:`DEFAULT_PARAMS`-like).
    grid : dict, optional
        ``{param: [values]}`` (default :data:`HP_GRID`); the Cartesian product is searched.
    seed : int, optional
        Unused placeholder for API symmetry (LightGBM determinism comes from ``base_params``).

    Returns
    -------
    (dict, list of dict)
        The chosen params and a log with one record per combo (params, ``val_log_loss``,
        ``best_iteration``).
    """
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]
    log: list[dict] = []
    best_params, best_ll = None, np.inf
    for combo in combos:
        params = {**base_params, **combo}
        clf = _RawClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        best_score = clf.model.best_score_
        try:
            val_ll = float(best_score["valid_0"]["multi_logloss"])
        except (KeyError, TypeError):  # pragma: no cover - single-class fold
            proba = _align(clf.predict_proba(X_val), clf.classes_, sorted(set(np.asarray(y_val).tolist())))
            val_ll = float(-np.log(np.clip(proba.max(axis=1), 1e-12, 1.0)).mean())
        best_iter = clf.best_iteration_ or params["n_estimators"]
        log.append({"params": dict(combo), "val_log_loss": val_ll, "best_iteration": int(best_iter)})
        if val_ll < best_ll:
            best_ll, best_params = val_ll, {**params, "n_estimators": int(max(best_iter, 20))}
    return best_params, log


def _resolve_params(train: pd.DataFrame, X: pd.DataFrame, y, tune: bool, params: dict | None,
                    seed: int) -> tuple[dict, list[dict]]:
    """Return the params to fit with: an explicit ``params``, a grid selection, or defaults."""
    base = {**DEFAULT_PARAMS, **(params or {})}
    if not tune:
        return base, []
    hold = _carve_tune_holdout(train)
    if hold is None:
        return base, []
    tr_pos = np.flatnonzero(~hold)
    va_pos = np.flatnonzero(hold)
    tr_pos = tr_pos[_subsample_rows(len(tr_pos), HP_SAMPLE_CAP, seed)]
    va_pos = va_pos[_subsample_rows(len(va_pos), HP_SAMPLE_CAP, seed + 1)]
    y = np.asarray(y)
    if len(np.unique(y[tr_pos])) < 2:
        return base, []
    return select_hyperparams(X.iloc[tr_pos], y[tr_pos], X.iloc[va_pos], y[va_pos], base, HP_GRID, seed)


# --- outcome-view feature matrix (view + current action) -------------------------------

def outcome_view_matrix(table: pd.DataFrame, view: str, action=None) -> pd.DataFrame:
    """Build the outcome feature matrix: the state view plus the current action (decision D22).

    Parameters
    ----------
    table : pandas.DataFrame
        Decision table.
    view : str
        One of :data:`~pitchseq.states.STATE_VIEWS`.
    action : str or array-like, optional
        The action family appended as ``action_family``. A scalar sets every row to that
        family (the counterfactual used by :func:`q_grid`); an array sets it per row; ``None``
        uses the observed ``family`` column.

    Returns
    -------
    pandas.DataFrame
        The view matrix with a categorical ``action_family`` column over
        :data:`~pitchseq.families.FAMILIES`. The view itself is leakage-audited by
        :func:`~pitchseq.states.build_view`; appending the chosen action is legitimate (it is
        the decision, not execution).
    """
    X, _ = build_view(table, view)
    X = X.copy()
    if action is None:
        vals = table["family"].astype("object").to_numpy()
    elif np.isscalar(action):
        vals = np.array([action] * len(table), dtype=object)
    else:
        vals = np.asarray(action, dtype=object)
    X[ACTION_COL] = pd.Categorical(vals, categories=list(FAMILIES))
    return X


# --- behavior model mu(a|s) ------------------------------------------------------------

class BehaviorModel:
    """Multiclass LightGBM behavior model ``mu(a | s)`` for one state view.

    Predicts the next-pitch family distribution from the state view alone (no action, no
    execution). :meth:`predict_proba` returns an ``(n, 8)`` matrix in
    :data:`~pitchseq.families.FAMILIES` order.

    Attributes
    ----------
    view : str
        The state view.
    params_ : dict
        The LightGBM params actually fitted (after any grid selection).
    hp_log_ : list of dict
        The per-combo grid-selection log (empty when tuning was skipped).
    feature_names_ : list of str
        The view's feature names.
    """

    def __init__(self, view: str) -> None:
        if view not in STATE_VIEWS:
            raise ValueError(f"unknown view {view!r}; expected one of {STATE_VIEWS}")
        self.view = view
        self._clf: _RawClassifier | None = None
        self.params_: dict = {}
        self.hp_log_: list[dict] = []
        self.feature_names_: list[str] = []

    def fit(self, train_table: pd.DataFrame, tune: bool = False, params: dict | None = None,
            seed: int = 0) -> "BehaviorModel":
        """Fit the behavior model on a training decision table.

        Parameters
        ----------
        train_table : pandas.DataFrame
            Training decision table (SPEC ``3``).
        tune : bool, optional
            When ``True`` select params from :data:`HP_GRID` by internal-holdout validation
            log loss (decision D34); otherwise use ``params`` / :data:`DEFAULT_PARAMS`.
        params : dict, optional
            Explicit LightGBM params (overrides defaults; ignored per-key by a grid search).
        seed : int, optional
            Seed for the HP-search subsampling.
        """
        X, meta = build_view(train_table, self.view)
        leakage_audit(X)  # no execution/label column may be a behavior feature (SPEC 0/11)
        self.feature_names_ = list(meta["feature_names"])
        y = train_table["family"].astype("object").to_numpy()
        self.params_, self.hp_log_ = _resolve_params(train_table, X, y, tune, params, seed)
        self._clf = _RawClassifier(**self.params_).fit(X, y)
        return self

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        """Family probabilities ``(n, 8)`` aligned to :data:`~pitchseq.families.FAMILIES`."""
        X, _ = build_view(table, self.view)
        return _align(self._clf.predict_proba(X), self._clf.classes_, list(FAMILIES))

    def feature_gain(self) -> dict:
        """Gain-based feature importances ``{feature: gain}`` (descending), for the exhibit."""
        gains = self._clf.feature_gain()
        return dict(sorted(gains.items(), key=lambda kv: kv[1], reverse=True))

    @property
    def n_params(self) -> int:
        return self._clf.n_params


# --- stage C: count-conditional node values --------------------------------------------

def _count_key(balls, strikes) -> np.ndarray:
    """Integer count key ``balls * 3 + strikes`` (unique per legal (balls, strikes) cell)."""
    b = np.asarray(balls, dtype=np.int64)
    s = np.asarray(strikes, dtype=np.int64)
    return b * 3 + s


class NodeValueTable:
    r"""Count-conditional event-tree node values ``V(node, balls, strikes)`` (decision D32).

    For each level-1 outcome ``k`` and each level-2 outcome ``j`` this stores the train-fold
    mean reward and within-node reward variance, both keyed by the count cell with a
    global-node fallback:

    .. math::

       V_1(k, c) = \operatorname{mean}\{R_i : o_{1,i} = k,\; c_i = c\},\qquad
       V_2(j, c) = \operatorname{mean}\{R_i : o_{2,i} = j,\; c_i = c,\; \text{in play}\},

    falling back to the count-marginal node mean when a ``(node, count)`` cell is empty, and
    to ``0`` when a node is entirely absent. The within-node variances :math:`S_1, S_2` feed
    the residual-based ``exp_reward_sd`` (see the module docstring).

    Attributes
    ----------
    v1_global, s1_global : numpy.ndarray, shape (6,)
        Global level-1 node means / within-node variances (:data:`~pitchseq.outcomes.OUTCOME1`
        order).
    v2_global, s2_global : numpy.ndarray, shape (5,)
        Global level-2 node means / variances (:data:`~pitchseq.outcomes.OUTCOME2` order).
    v1_by_count, s1_by_count, v2_by_count, s2_by_count : dict
        ``count_key -> (resolved node vector)`` with per-node global fallback baked in.
    """

    def __init__(self) -> None:
        self.v1_global = np.zeros(len(_O1))
        self.s1_global = np.zeros(len(_O1))
        self.v2_global = np.zeros(len(_O2))
        self.s2_global = np.zeros(len(_O2))
        self.v1_by_count: dict[int, np.ndarray] = {}
        self.s1_by_count: dict[int, np.ndarray] = {}
        self.v2_by_count: dict[int, np.ndarray] = {}
        self.s2_by_count: dict[int, np.ndarray] = {}

    def fit(self, train_table: pd.DataFrame) -> "NodeValueTable":
        """Compute the node value / variance lookups from a training decision table."""
        R = pd.to_numeric(train_table["R"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(R)
        o1 = train_table["outcome1"].astype("object").to_numpy()
        o2 = train_table["outcome2"].astype("object").to_numpy()
        ck = _count_key(train_table["balls"].to_numpy(), train_table["strikes"].to_numpy())

        self.v1_global, self.s1_global = self._global_stats(R, finite, o1, _O1)
        ip = finite & (o1 == "in_play")
        self.v2_global, self.s2_global = self._global_stats(R, ip, o2, _O2)

        for c in np.unique(ck):
            cmask = finite & (ck == c)
            self.v1_by_count[int(c)], self.s1_by_count[int(c)] = self._cell_stats(
                R, cmask, o1, _O1, self.v1_global, self.s1_global
            )
            ipc = ip & (ck == c)
            self.v2_by_count[int(c)], self.s2_by_count[int(c)] = self._cell_stats(
                R, ipc, o2, _O2, self.v2_global, self.s2_global
            )
        return self

    @staticmethod
    def _global_stats(R, mask, labels_col, labels) -> tuple[np.ndarray, np.ndarray]:
        v = np.zeros(len(labels))
        s = np.zeros(len(labels))
        for i, lab in enumerate(labels):
            sel = mask & (labels_col == lab)
            if sel.any():
                vals = R[sel]
                v[i] = float(vals.mean())
                s[i] = float(vals.var()) if len(vals) > 1 else 0.0
        return v, s

    @staticmethod
    def _cell_stats(R, mask, labels_col, labels, v_global, s_global) -> tuple[np.ndarray, np.ndarray]:
        v = v_global.copy()
        s = s_global.copy()
        for i, lab in enumerate(labels):
            sel = mask & (labels_col == lab)
            if sel.any():
                vals = R[sel]
                v[i] = float(vals.mean())
                s[i] = float(vals.var()) if len(vals) > 1 else float(s_global[i])
        return v, s

    def resolve(self, balls, strikes) -> dict:
        """Per-row node vectors for a batch of counts.

        Returns a dict with ``v1``/``s1`` of shape ``(n, 6)`` and ``v2``/``s2`` of shape
        ``(n, 5)``; unseen counts use the global node vectors.
        """
        ck = _count_key(balls, strikes)
        n = len(ck)
        v1 = np.empty((n, len(_O1)))
        s1 = np.empty((n, len(_O1)))
        v2 = np.empty((n, len(_O2)))
        s2 = np.empty((n, len(_O2)))
        for c in np.unique(ck):
            rows = np.flatnonzero(ck == c)
            ci = int(c)
            v1[rows] = self.v1_by_count.get(ci, self.v1_global)
            s1[rows] = self.s1_by_count.get(ci, self.s1_global)
            v2[rows] = self.v2_by_count.get(ci, self.v2_global)
            s2[rows] = self.s2_by_count.get(ci, self.s2_global)
        return {"v1": v1, "s1": s1, "v2": v2, "s2": s2}

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict (arrays -> lists; count keys -> strings)."""
        def enc(d):
            return {str(k): v.tolist() for k, v in d.items()}

        return {
            "v1_global": self.v1_global.tolist(), "s1_global": self.s1_global.tolist(),
            "v2_global": self.v2_global.tolist(), "s2_global": self.s2_global.tolist(),
            "v1_by_count": enc(self.v1_by_count), "s1_by_count": enc(self.s1_by_count),
            "v2_by_count": enc(self.v2_by_count), "s2_by_count": enc(self.s2_by_count),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeValueTable":
        """Reconstruct from :meth:`to_dict`."""
        t = cls()
        t.v1_global = np.asarray(d["v1_global"], dtype=np.float64)
        t.s1_global = np.asarray(d["s1_global"], dtype=np.float64)
        t.v2_global = np.asarray(d["v2_global"], dtype=np.float64)
        t.s2_global = np.asarray(d["s2_global"], dtype=np.float64)

        def dec(x):
            return {int(k): np.asarray(v, dtype=np.float64) for k, v in x.items()}

        t.v1_by_count = dec(d["v1_by_count"])
        t.s1_by_count = dec(d["s1_by_count"])
        t.v2_by_count = dec(d["v2_by_count"])
        t.s2_by_count = dec(d["s2_by_count"])
        return t


# --- decomposed outcome stack ----------------------------------------------------------

class OutcomeStack:
    """The decomposed event-tree outcome model for one state view (decision D32).

    Holds the stage-A / stage-B classifiers, the stage-C :class:`NodeValueTable`, and the
    direct-regression cross-check. :meth:`exp_reward` assembles ``E[R | s, a]`` by the exact
    formula in the module docstring; :meth:`q_grid` sweeps the action over all 8 families.

    Attributes
    ----------
    view : str
        The state view.
    params_ : dict
        The LightGBM params fitted for stage A (reused by stage B and the direct regressor to
        keep the equal-budget grid cost bounded -- decision D34).
    hp_log_ : list of dict
        Stage-A grid-selection log.
    """

    def __init__(self, view: str) -> None:
        if view not in STATE_VIEWS:
            raise ValueError(f"unknown view {view!r}; expected one of {STATE_VIEWS}")
        self.view = view
        self._stage_a: _RawClassifier | None = None
        self._stage_b: _RawClassifier | None = None
        self._direct: _RawRegressor | None = None
        self.nodes = NodeValueTable()
        self.params_: dict = {}
        self.hp_log_: list[dict] = []
        self.feature_names_: list[str] = []

    def fit(self, train_table: pd.DataFrame, tune: bool = False, params: dict | None = None,
            seed: int = 0) -> "OutcomeStack":
        """Fit stages A/B, the node values and the direct regressor on a training table.

        Stage A tunes (when ``tune``) by internal-holdout outcome-1 log loss; the chosen
        params are reused for stage B (fitted on in-play rows) and the direct regressor.
        """
        X = outcome_view_matrix(train_table, self.view, action=None)
        # Only ``action_family`` may be added to the audited view; assert nothing leaked.
        leakage_audit([c for c in X.columns if c != ACTION_COL])
        self.feature_names_ = list(X.columns)

        y1 = train_table["outcome1"].astype("object").to_numpy()
        self.params_, self.hp_log_ = _resolve_params(train_table, X, y1, tune, params, seed)
        self._stage_a = _RawClassifier(**self.params_).fit(X, y1)

        ip = train_table["outcome2"].notna().to_numpy()
        Xb = X.iloc[ip]
        yb = train_table.loc[ip, "outcome2"].astype("object").to_numpy()
        self._stage_b = _RawClassifier(**self.params_).fit(Xb, yb)

        R = pd.to_numeric(train_table["R"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(R)
        self._direct = _RawRegressor(**self.params_).fit(X.iloc[finite], R[finite])

        self.nodes.fit(train_table)
        return self

    # --- stage predictions -------------------------------------------------------------

    def _stage_probs(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Stage-A / stage-B probabilities from an already-built outcome-view matrix.

        Building the view once and reusing it here is what keeps :meth:`q_grid` (which sweeps
        the action over 8 families) from rebuilding the view matrix 8x -- the assembly's
        dominant cost on the full data.
        """
        p_a = _align(self._stage_a.predict_proba(X), self._stage_a.classes_, _O1)
        p_b = _align(self._stage_b.predict_proba(X), self._stage_b.classes_, _O2)
        return p_a, p_b

    def predict_outcome1(self, table: pd.DataFrame, action=None) -> np.ndarray:
        """``P(outcome1 | s, a)`` as ``(n, 6)`` in :data:`~pitchseq.outcomes.OUTCOME1` order."""
        X = outcome_view_matrix(table, self.view, action=action)
        return _align(self._stage_a.predict_proba(X), self._stage_a.classes_, _O1)

    def predict_outcome2(self, table: pd.DataFrame, action=None) -> np.ndarray:
        """``P(outcome2 | s, a)`` as ``(n, 5)`` in :data:`~pitchseq.outcomes.OUTCOME2` order."""
        X = outcome_view_matrix(table, self.view, action=action)
        return _align(self._stage_b.predict_proba(X), self._stage_b.classes_, _O2)

    def predict_direct(self, table: pd.DataFrame, action=None) -> np.ndarray:
        """The direct-regression ``E[R | s, a]`` (the D32 cross-check), shape ``(n,)``."""
        X = outcome_view_matrix(table, self.view, action=action)
        return self._direct.predict(X)

    # --- assembly ----------------------------------------------------------------------

    def _assemble(self, p_a: np.ndarray, p_b: np.ndarray, balls, strikes) -> tuple[np.ndarray, np.ndarray]:
        r"""Assemble ``E[R]`` and its residual-based sd from stage probabilities + node values.

        Implements the two formulas in the module docstring: the expected reward is the
        leaf-probability-weighted sum of node values, and the variance is the law-of-total-
        variance over the 10 event-tree leaves.
        """
        nv = self.nodes.resolve(balls, strikes)
        v1, s1, v2, s2 = nv["v1"], nv["s1"], nv["v2"], nv["s2"]

        p_inplay = p_a[:, _I_INPLAY]
        # Leaf probabilities: 5 non-in-play level-1 nodes, then 5 in-play level-2 refinements.
        p_leaf = np.concatenate([p_a[:, _NON_INPLAY], p_inplay[:, None] * p_b], axis=1)
        v_leaf = np.concatenate([v1[:, _NON_INPLAY], v2], axis=1)
        s_leaf = np.concatenate([s1[:, _NON_INPLAY], s2], axis=1)

        er = (p_leaf * v_leaf).sum(axis=1)
        second_moment = (p_leaf * (s_leaf + v_leaf ** 2)).sum(axis=1)
        var = np.maximum(second_moment - er ** 2, 0.0)
        return er, np.sqrt(var)

    def exp_reward(self, table: pd.DataFrame, action=None, with_sd: bool = False):
        """Assembled ``E[R | s, a]`` (decision D32).

        Parameters
        ----------
        table : pandas.DataFrame
            Decision table.
        action : str or array-like, optional
            Action family (scalar / per-row / observed ``None``); see
            :func:`outcome_view_matrix`.
        with_sd : bool, optional
            When ``True`` also return the residual-based ``exp_reward_sd``.

        Returns
        -------
        numpy.ndarray or tuple
            ``E[R]`` of shape ``(n,)``, or ``(E[R], sd)`` when ``with_sd``.
        """
        X = outcome_view_matrix(table, self.view, action=action)  # one view build for both stages
        p_a, p_b = self._stage_probs(X)
        er, sd = self._assemble(p_a, p_b, table["balls"].to_numpy(), table["strikes"].to_numpy())
        return (er, sd) if with_sd else er

    def q_grid(self, table: pd.DataFrame) -> np.ndarray:
        """Counterfactual value grid ``E[R | s, a]`` for all 8 families, shape ``(n, 8)``.

        Builds the state view **once**, then swaps the ``action_family`` column to each family
        in turn, re-runs stages A/B and re-assembles -- so the expensive view construction is
        not repeated per family.
        """
        balls = table["balls"].to_numpy()
        strikes = table["strikes"].to_numpy()
        X, _ = build_view(table, self.view)
        X = X.copy()
        out = np.empty((len(table), _N_FAM), dtype=np.float64)
        for j, fam in enumerate(FAMILIES):
            X[ACTION_COL] = pd.Categorical([fam] * len(table), categories=list(FAMILIES))
            p_a, p_b = self._stage_probs(X)
            out[:, j], _ = self._assemble(p_a, p_b, balls, strikes)
        return out

    def disagreement(self, table: pd.DataFrame, action=None) -> dict:
        """The D32 decomposed-vs-direct cross-check on a table.

        Returns ``mean_abs_diff`` = ``mean |E_decomposed - E_direct|``, ``max_abs_diff``,
        ``flag`` (``mean_abs_diff > DISAGREEMENT_FLAG_ABS``) and ``threshold``.
        """
        dec = self.exp_reward(table, action=action)
        dirv = self.predict_direct(table, action=action)
        diff = np.abs(dec - dirv)
        mad = float(diff.mean()) if len(diff) else float("nan")
        return {
            "mean_abs_diff": mad,
            "max_abs_diff": float(diff.max()) if len(diff) else float("nan"),
            "flag": bool(np.isfinite(mad) and mad > DISAGREEMENT_FLAG_ABS),
            "threshold": DISAGREEMENT_FLAG_ABS,
        }

    def feature_gain(self) -> dict:
        """Stage-A gain-based feature importances ``{feature: gain}`` (descending)."""
        gains = self._stage_a.feature_gain()
        return dict(sorted(gains.items(), key=lambda kv: kv[1], reverse=True))

    @property
    def n_params(self) -> int:
        return int(self._stage_a.n_params + self._stage_b.n_params)


# --- derived artifacts (decision D33) --------------------------------------------------

def q_grid(model: OutcomeStack, table: pd.DataFrame, view: str | None = None) -> np.ndarray:
    """Counterfactual value grid ``qhat(s, a)`` for all 8 families (decision D33).

    Parameters
    ----------
    model : OutcomeStack
        A fitted outcome stack (its own ``view`` is used).
    table : pandas.DataFrame
        Decision table.
    view : str, optional
        Ignored when it matches ``model.view``; raises on a mismatch (guards against feeding
        an artifact from the wrong view).

    Returns
    -------
    numpy.ndarray, shape (n, 8)
        ``E[R | s, a]`` for every family (:data:`~pitchseq.families.FAMILIES` order).
    """
    if view is not None and view != model.view:
        raise ValueError(f"view {view!r} does not match the model's view {model.view!r}")
    return model.q_grid(table)


def propensities(behavior_model: BehaviorModel, table: pd.DataFrame) -> np.ndarray:
    """Behavior propensities ``mu(a | s)`` for all 8 families (decision D33), shape ``(n, 8)``."""
    return behavior_model.predict_proba(table)


# --- persistence -----------------------------------------------------------------------

def _behavior_paths(out_dir: Path, view: str) -> dict:
    return {"model": out_dir / f"behavior_{view}.joblib", "meta": out_dir / f"behavior_{view}.json"}


def _outcome_paths(out_dir: Path, view: str) -> dict:
    return {"model": out_dir / f"outcome_{view}.joblib", "meta": out_dir / f"outcome_{view}.json"}


def save_behavior(out_dir: str | Path, model: BehaviorModel) -> None:
    """Persist a fitted :class:`BehaviorModel` (joblib blob + JSON metadata sidecar)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _behavior_paths(out_dir, model.view)
    joblib.dump(model, paths["model"])
    meta = {
        "view": model.view, "kind": "behavior", "params": model.params_,
        "hp_log": model.hp_log_, "feature_names": model.feature_names_,
        "n_params": model.n_params, "feature_gain_top": _top_gain(model.feature_gain()),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")


def save_outcome(out_dir: str | Path, model: OutcomeStack) -> None:
    """Persist a fitted :class:`OutcomeStack` (joblib blob + JSON metadata + node lookups)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _outcome_paths(out_dir, model.view)
    joblib.dump(model, paths["model"])
    meta = {
        "view": model.view, "kind": "outcome", "params": model.params_,
        "hp_log": model.hp_log_, "feature_names": model.feature_names_,
        "n_params": model.n_params, "node_values": model.nodes.to_dict(),
        "feature_gain_top": _top_gain(model.feature_gain()),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")


def _top_gain(gains: dict, k: int = 15) -> dict:
    return dict(list(gains.items())[:k])


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class WS3Artifacts:
    """Loaded WS3 artifacts: the behavior + outcome models per view (decision D33).

    Consumed by WS4/WS5/WS7 -- they read these fitted pieces and never refit their own
    behavior propensity or outcome model.

    Attributes
    ----------
    behavior : dict
        ``view -> BehaviorModel``.
    outcome : dict
        ``view -> OutcomeStack``.
    directory : str
        The directory the artifacts were loaded from.
    """

    behavior: dict = field(default_factory=dict)
    outcome: dict = field(default_factory=dict)
    directory: str = ""

    @property
    def views(self) -> list[str]:
        return sorted(set(self.behavior) | set(self.outcome))

    def propensities(self, table: pd.DataFrame, view: str) -> np.ndarray:
        """``mu(a | s)`` ``(n, 8)`` from the loaded behavior model for ``view``."""
        return propensities(self.behavior[view], table)

    def q_grid(self, table: pd.DataFrame, view: str) -> np.ndarray:
        """``qhat(s, a)`` ``(n, 8)`` from the loaded outcome stack for ``view``."""
        return q_grid(self.outcome[view], table, view)

    def exp_reward(self, table: pd.DataFrame, view: str, action=None, with_sd: bool = False):
        """Assembled ``E[R | s, a]`` from the loaded outcome stack for ``view``."""
        return self.outcome[view].exp_reward(table, action=action, with_sd=with_sd)


def save_ws3_artifacts(out_dir: str | Path, behavior: dict | None = None,
                       outcome: dict | None = None) -> None:
    """Persist per-view behavior and/or outcome models under ``out_dir`` (decision D33)."""
    for model in (behavior or {}).values():
        save_behavior(out_dir, model)
    for model in (outcome or {}).values():
        save_outcome(out_dir, model)


def load_ws3_artifacts(directory: str | Path) -> WS3Artifacts:
    """Load all persisted WS3 behavior + outcome models from ``directory`` (decision D33).

    Parameters
    ----------
    directory : str or pathlib.Path
        A directory written by :func:`save_ws3_artifacts` / the WS3 pipeline.

    Returns
    -------
    WS3Artifacts
        The loaded models keyed by view, with :meth:`WS3Artifacts.propensities`,
        :meth:`WS3Artifacts.q_grid` and :meth:`WS3Artifacts.exp_reward` helpers for WS4/5/7.
    """
    directory = Path(directory)
    behavior: dict = {}
    outcome: dict = {}
    for p in sorted(directory.glob("behavior_*.joblib")):
        model = joblib.load(p)
        behavior[model.view] = model
    for p in sorted(directory.glob("outcome_*.joblib")):
        model = joblib.load(p)
        outcome[model.view] = model
    return WS3Artifacts(behavior=behavior, outcome=outcome, directory=str(directory))


# --- falsification factory (decision D17 interface) ------------------------------------

def make_ws3_model_factory(params: dict | None = None):
    """Return a ``model_factory(view) -> classifier`` for the shared falsification battery.

    The shared falsification code (:mod:`pitchseq.eval.falsification`) builds the view
    matrices itself (appending ``action_family`` for outcome targets) and re-aligns via
    ``classes_``, so this hands back a raw LightGBM multiclass classifier with WS3's params --
    the same model family the behavior / stage-A models use.

    Parameters
    ----------
    params : dict, optional
        LightGBM params (merged over :data:`DEFAULT_PARAMS`); a small / fast override keeps
        the permutation refits cheap in tests.
    """
    merged = {**DEFAULT_PARAMS, **(params or {})}

    def factory(view: str) -> _RawClassifier:
        return _RawClassifier(**merged)

    return factory
