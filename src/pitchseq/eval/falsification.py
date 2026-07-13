"""Sequence falsification through a model-callback interface (SPEC ``8.3``; decision D17).

The caller supplies a ``model_factory`` -- ``model_factory(view: str) -> model`` where
``model`` has ``fit(X, y)`` and ``predict_proba(X)`` (and, ideally, ``classes_``) -- plus
the decision table. This module builds the state views internally via
:func:`pitchseq.states.build_view`, constructs the permuted / pseudo-history / ablated
feature sets, retrains through the callback, and compares losses. It never inspects model
internals, so any learner (LightGBM, logistic regression, ...) plugs in.

Conditioning on the current pitch. For an **outcome** target the model must condition on the
pitch actually thrown, or ordered history would help merely by predicting *which* pitch is
thrown next (selection structure, finding #1) rather than the outcome (finding #2). So for
any non-``family`` target the chosen family is appended as the ``action_family`` feature to
every view (this is the "after conditioning on the current pitch" of SPEC ``0``). For the
``family`` target the family is the label and is not added.

Implemented (SPEC ``8.3``):

* :func:`order_ablation` -- O must beat U and L1, not just C (returns per-view losses +
  the SPEC ``6`` deltas).
* :func:`history_permutation_test` -- permute ordered histories among PAs matched on
  strata **and prior-family multiset**, keeping the unordered content fixed, and refit;
  an order-sensitive edge collapses.
* :func:`pseudo_history_control` -- swap in a history from another PA by the *same pitcher*
  in a similar count; tests whether the model merely identifies the pitcher.
* :func:`mechanism_ablation` -- drop named feature groups (family slots / location /
  velo-diff / outcome history) and report the loss increase.
* :func:`run_null_world_acceptance` / :func:`run_positive_world_acceptance` -- the SPEC
  ``11`` structured verdicts used by the acceptance tests.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..families import FAMILIES
from ..outcomes import OUTCOME1
from ..states import build_view
from .metrics import ablation_deltas, clustered_ci, log_loss_per_row

__all__ = [
    "order_ablation",
    "history_permutation_test",
    "pseudo_history_control",
    "mechanism_ablation",
    "run_null_world_acceptance",
    "run_positive_world_acceptance",
    "velo_gap_prev_transition",
]

_ACTION_COL = "action_family"


# --- small helpers ---------------------------------------------------------------------

def _target_labels(target: str) -> list:
    if target == "family":
        return list(FAMILIES)
    if target == "outcome1":
        return list(OUTCOME1)
    raise ValueError(f"target must be 'family' or 'outcome1', got {target!r}")


def _target_values(table: pd.DataFrame, target: str) -> np.ndarray:
    col = "family" if target == "family" else "outcome1"
    return table[col].astype("object").to_numpy()


def _mask_positions(mask, n: int) -> np.ndarray:
    if mask is None:
        return np.arange(n)
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return np.flatnonzero(mask)
    return mask.astype(np.int64)


def _default_train_eval(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """A sensible default split: by season if available, else by game parity.

    Synthetic worlds are stationary, so a game-level split is leakage-safe; when seasons
    are present the SPEC temporal ordering (earlier seasons train, latest eval) is used.
    """
    n = len(table)
    if "season" in table.columns and table["season"].nunique() >= 2:
        seasons = np.sort(table["season"].unique())
        cut = seasons[-1]
        train = (table["season"] < cut).to_numpy()
        evalm = (table["season"] == cut).to_numpy()
        if train.any() and evalm.any():
            return train, evalm
    gp = table["game_pk"].to_numpy()
    train = (gp % 2 == 0)
    evalm = ~train
    return train, evalm


def _build_view_matrix(table: pd.DataFrame, view: str, condition_on_action: bool) -> pd.DataFrame:
    """Build a view and optionally append the chosen family as ``action_family``."""
    X, _ = build_view(table, view)
    if condition_on_action:
        X = X.copy()
        X[_ACTION_COL] = pd.Categorical(
            table["family"].astype("object").to_numpy(), categories=list(FAMILIES)
        )
    return X


def _align_proba(model, X: pd.DataFrame, labels: list) -> np.ndarray:
    """Predict and align columns to ``labels`` order via ``model.classes_`` if present."""
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is None:
        if proba.shape[1] != len(labels):
            raise ValueError(
                f"model has no classes_ and predict_proba width {proba.shape[1]} != "
                f"len(labels) {len(labels)}"
            )
        return proba
    col = {c: i for i, c in enumerate(list(classes))}
    out = np.zeros((proba.shape[0], len(labels)), dtype=np.float64)
    for j, lab in enumerate(labels):
        if lab in col:
            out[:, j] = proba[:, col[lab]]
    return out


def _fit_eval(
    model_factory,
    view: str,
    X: pd.DataFrame,
    y: np.ndarray,
    labels: list,
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Fit ``model_factory(view)`` on train rows, return per-row eval log loss."""
    model = model_factory(view)
    model.fit(X.iloc[train_pos], y[train_pos])
    proba = _align_proba(model, X.iloc[eval_pos], labels)
    return log_loss_per_row(y[eval_pos], proba, labels, eps)


# --- order ablation --------------------------------------------------------------------

def order_ablation(
    table: pd.DataFrame,
    model_factory,
    train_mask=None,
    eval_mask=None,
    target: str = "outcome1",
    views=("C", "U", "L1", "O", "OM"),
    condition_on_action: bool | None = None,
    eps: float = 1e-12,
) -> dict:
    """Fit one model per view and return per-view losses plus the SPEC ``6`` deltas.

    Parameters
    ----------
    table : pandas.DataFrame
        Decision table.
    model_factory : callable
        ``model_factory(view) -> model`` with ``fit`` / ``predict_proba``.
    train_mask, eval_mask : array-like of bool, optional
        Row masks; defaults to :func:`_default_train_eval`.
    target : {'outcome1', 'family'}
        Prediction target. ``'outcome1'`` conditions on the current family (finding #2).
    views : sequence of str
        Views to fit (default all five).
    condition_on_action : bool, optional
        Override the default (append current family iff target is not ``'family'``).
    eps : float
        Log-loss clip floor.

    Returns
    -------
    dict
        ``losses`` (per view), ``per_row`` (per-view eval loss arrays, aligned to
        ``eval_pos``), ``deltas`` (``delta_order`` / ``delta_matchup`` when U/L1/O present),
        ``eval_pos``, ``train_pos``, ``target``, ``condition_on_action``.
    """
    n = len(table)
    train_pos = _mask_positions(train_mask, n)
    eval_pos = _mask_positions(eval_mask, n)
    if train_mask is None and eval_mask is None:
        tm, em = _default_train_eval(table)
        train_pos, eval_pos = np.flatnonzero(tm), np.flatnonzero(em)
    cond = (target != "family") if condition_on_action is None else condition_on_action
    labels = _target_labels(target)
    y = _target_values(table, target)

    per_row: dict[str, np.ndarray] = {}
    losses: dict[str, float] = {}
    for v in views:
        X = _build_view_matrix(table, v, cond)
        pr = _fit_eval(model_factory, v, X, y, labels, train_pos, eval_pos, eps)
        per_row[v] = pr
        losses[v] = float(pr.mean())

    if {"U", "L1", "O"} <= set(losses):
        deltas = ablation_deltas(losses)
    else:
        deltas = {"delta_order": None, "delta_matchup": None}

    return {
        "target": target,
        "condition_on_action": cond,
        "views": list(views),
        "losses": losses,
        "per_row": per_row,
        "eval_pos": eval_pos,
        "train_pos": train_pos,
        "deltas": deltas,
    }


# --- stratified history permutation ----------------------------------------------------

def _strata_codes(table: pd.DataFrame, strata_cols, multiset_cols_frame: pd.DataFrame | None) -> np.ndarray:
    """Integer stratum id per row from strata columns + optional multiset signature."""
    parts = [table[c].astype("object").astype(str).to_numpy() for c in strata_cols if c in table.columns]
    if multiset_cols_frame is not None and multiset_cols_frame.shape[1] > 0:
        ms = multiset_cols_frame.round().astype("int64").astype(str)
        parts.append(ms.agg("|".join, axis=1).to_numpy())
    key = np.array(["|".join(vals) for vals in zip(*parts)], dtype=object)
    return pd.factorize(key)[0]


def _grouped_permutation(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation of ``arange(n)`` that shuffles only *within* each stratum code."""
    n = len(codes)
    perm = np.arange(n)
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    bounds = np.flatnonzero(np.diff(sc)) + 1
    for grp in np.split(order, bounds):
        if len(grp) > 1:
            perm[grp] = grp[rng.permutation(len(grp))]
    return perm


def _permute_block(X: pd.DataFrame, block_cols: list, perm: np.ndarray) -> pd.DataFrame:
    """Return ``X`` with ``block_cols`` rows reordered by ``perm`` (dtypes preserved)."""
    other = X.drop(columns=block_cols).reset_index(drop=True)
    block = X[block_cols].iloc[perm].reset_index(drop=True)
    out = pd.concat([other, block], axis=1)
    return out[list(X.columns)]


def history_permutation_test(
    table: pd.DataFrame,
    model_factory,
    strata_cols=("pitcher", "balls", "strikes", "pitch_number", "stand"),
    n_permutations: int = 24,
    seed: int = 0,
    target: str = "outcome1",
    reference_view: str = "U",
    ordered_view: str = "O",
    condition_on_action: bool | None = None,
    restrict_min_pitch: int = 3,
    train_mask=None,
    eval_mask=None,
    alpha: float = 0.05,
    eps: float = 1e-12,
) -> dict:
    r"""Stratified history-permutation test for an order-sensitive outcome edge.

    The observed statistic is the **order edge** ``edge = Loss(reference_view) -
    Loss(ordered_view)`` on the evaluation rows (positive when the ordered view predicts
    outcomes better). Under the null of no ordered dependence, permuting the ordered-history
    features among rows matched on ``strata_cols`` **and the prior-family multiset** (so the
    unordered content and count/handedness/depth are preserved -- only *order* is scrambled)
    leaves the edge unchanged in expectation. The one-sided p-value is

    .. math:: p = \frac{1 + \#\{b : \text{edge}^{(b)} \ge \text{edge}_{\text{obs}}\}}{1 + B}.

    On synthetic data the strata (including the multiset) match exactly, approximating the
    SPEC ``8.3`` multiset-matched permutation. Only the ordered-only columns (those in the O
    view but not the reference view) are permuted; the reference view carries no order
    information, so its loss is constant across permutations and is computed once.

    Returns
    -------
    dict
        ``observed_edge``, ``loss_reference``, ``loss_ordered``, ``null_edges``,
        ``p_value``, ``fired`` (``p_value < alpha``), plus bookkeeping counts.
    """
    n = len(table)
    if train_mask is None and eval_mask is None:
        tm, em = _default_train_eval(table)
    else:
        tm = np.zeros(n, bool) if train_mask is None else np.asarray(train_mask, bool)
        em = np.zeros(n, bool) if eval_mask is None else np.asarray(eval_mask, bool)
    restrict = table["pitch_number"].to_numpy() >= restrict_min_pitch
    train_pos = np.flatnonzero(tm & restrict)
    eval_pos = np.flatnonzero(em & restrict)

    cond = (target != "family") if condition_on_action is None else condition_on_action
    labels = _target_labels(target)
    y = _target_values(table, target)

    X_ref = _build_view_matrix(table, reference_view, cond)
    X_ord = _build_view_matrix(table, ordered_view, cond)
    order_only = [c for c in X_ord.columns if c not in set(X_ref.columns)]
    if not order_only or len(train_pos) == 0 or len(eval_pos) == 0:
        return {
            "observed_edge": float("nan"),
            "p_value": 1.0,
            "fired": False,
            "null_edges": [],
            "order_only_n": len(order_only),
            "n_train": int(len(train_pos)),
            "n_eval": int(len(eval_pos)),
        }

    # Reference loss (order-independent -> computed once).
    ref_loss = float(
        _fit_eval(model_factory, reference_view, X_ref, y, labels, train_pos, eval_pos, eps).mean()
    )
    obs_ord_loss = float(
        _fit_eval(model_factory, ordered_view, X_ord, y, labels, train_pos, eval_pos, eps).mean()
    )
    observed_edge = ref_loss - obs_ord_loss

    # Strata (+ prior-family multiset from the U-view counts, if present in the reference).
    ms_cols = [c for c in X_ref.columns if c.startswith("u_count_")]
    ms_frame = X_ref[ms_cols] if ms_cols else None
    codes = _strata_codes(table, strata_cols, ms_frame)

    rng = np.random.default_rng(seed)
    null_edges: list[float] = []
    for _ in range(n_permutations):
        perm = _grouped_permutation(codes, rng)
        X_perm = _permute_block(X_ord, order_only, perm)
        loss_perm = float(
            _fit_eval(model_factory, ordered_view, X_perm, y, labels, train_pos, eval_pos, eps).mean()
        )
        null_edges.append(ref_loss - loss_perm)

    null = np.asarray(null_edges, dtype=np.float64)
    p_value = float((1 + int(np.sum(null >= observed_edge))) / (1 + n_permutations))
    return {
        "observed_edge": observed_edge,
        "loss_reference": ref_loss,
        "loss_ordered": obs_ord_loss,
        "null_edges": null_edges,
        "null_edge_mean": float(null.mean()),
        "p_value": p_value,
        "fired": p_value < alpha,
        "order_only_n": len(order_only),
        "n_train": int(len(train_pos)),
        "n_eval": int(len(eval_pos)),
    }


# --- pseudo-history control ------------------------------------------------------------

def pseudo_history_control(
    table: pd.DataFrame,
    model_factory,
    seed: int = 0,
    target: str = "outcome1",
    ordered_view: str = "O",
    condition_on_action: bool | None = None,
    restrict_min_pitch: int = 2,
    train_mask=None,
    eval_mask=None,
    eps: float = 1e-12,
) -> dict:
    """Swap each PA's history for another PA's by the same pitcher in a similar count.

    All history-derived features (everything in the ordered view that is not context and
    not the current action) are permuted among rows matched on ``(pitcher, balls,
    strikes)``; context and the current pitch are left intact. If the model was genuinely
    using ordered history the loss rises (positive ``delta``); if it was only identifying
    the pitcher (same pitcher, same count) the loss barely moves.

    Returns
    -------
    dict
        ``loss_true``, ``loss_pseudo``, ``delta`` (``loss_pseudo - loss_true``).
    """
    n = len(table)
    if train_mask is None and eval_mask is None:
        tm, em = _default_train_eval(table)
    else:
        tm = np.asarray(train_mask, bool)
        em = np.asarray(eval_mask, bool)
    restrict = table["pitch_number"].to_numpy() >= restrict_min_pitch
    train_pos = np.flatnonzero(tm & restrict)
    eval_pos = np.flatnonzero(em & restrict)

    cond = (target != "family") if condition_on_action is None else condition_on_action
    labels = _target_labels(target)
    y = _target_values(table, target)

    X_ctx = _build_view_matrix(table, "C", cond)  # context (+ action)
    X_ord = _build_view_matrix(table, ordered_view, cond)
    hist_cols = [c for c in X_ord.columns if c not in set(X_ctx.columns)]
    if not hist_cols or len(train_pos) == 0 or len(eval_pos) == 0:
        return {"loss_true": float("nan"), "loss_pseudo": float("nan"), "delta": float("nan")}

    loss_true = float(
        _fit_eval(model_factory, ordered_view, X_ord, y, labels, train_pos, eval_pos, eps).mean()
    )
    codes = _strata_codes(table, ("pitcher", "balls", "strikes"), None)
    rng = np.random.default_rng(seed)
    perm = _grouped_permutation(codes, rng)
    X_pseudo = _permute_block(X_ord, hist_cols, perm)
    loss_pseudo = float(
        _fit_eval(model_factory, ordered_view, X_pseudo, y, labels, train_pos, eval_pos, eps).mean()
    )
    return {"loss_true": loss_true, "loss_pseudo": loss_pseudo, "delta": loss_pseudo - loss_true}


# --- mechanism ablation ----------------------------------------------------------------

def _default_mechanism_groups(columns) -> dict:
    """Partition O-view columns into interpretable mechanism groups by name pattern."""
    groups = {
        "family_slots": re.compile(r"(family|pitch_type)$|^u_count_|^action_family$"),
        "location": re.compile(r"plate_x_br|plate_z_norm|loc_delta"),
        "velo_diff": re.compile(r"release_speed|velo_delta|dvelo"),
        "outcome_history": re.compile(r"outcome1$|^u_n_"),
    }
    out: dict[str, list] = {g: [] for g in groups}
    for c in columns:
        for g, pat in groups.items():
            if pat.search(c):
                out[g].append(c)
    return {g: cols for g, cols in out.items() if cols}


def mechanism_ablation(
    table: pd.DataFrame,
    model_factory,
    groups: dict | None = None,
    target: str = "outcome1",
    ordered_view: str = "O",
    condition_on_action: bool | None = None,
    restrict_min_pitch: int = 1,
    train_mask=None,
    eval_mask=None,
    eps: float = 1e-12,
) -> dict:
    """Drop named feature groups from the O view and report the loss increase.

    Parameters
    ----------
    groups : dict, optional
        ``{group_name: [column, ...]}``. Defaults to family-slots / location / velo-diff /
        outcome-history derived from the O-view column names.

    Returns
    -------
    dict
        ``full_loss``, ``group_losses`` (loss with the group removed), ``deltas``
        (``loss_without - full_loss``; larger means the group mattered more), ``groups``.
    """
    n = len(table)
    if train_mask is None and eval_mask is None:
        tm, em = _default_train_eval(table)
    else:
        tm = np.asarray(train_mask, bool)
        em = np.asarray(eval_mask, bool)
    restrict = table["pitch_number"].to_numpy() >= restrict_min_pitch
    train_pos = np.flatnonzero(tm & restrict)
    eval_pos = np.flatnonzero(em & restrict)

    cond = (target != "family") if condition_on_action is None else condition_on_action
    labels = _target_labels(target)
    y = _target_values(table, target)

    X = _build_view_matrix(table, ordered_view, cond)
    if groups is None:
        groups = _default_mechanism_groups(X.columns)

    full_loss = float(_fit_eval(model_factory, ordered_view, X, y, labels, train_pos, eval_pos, eps).mean())
    group_losses: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for g, cols in groups.items():
        drop = [c for c in cols if c in X.columns]
        Xg = X.drop(columns=drop)
        loss_g = float(_fit_eval(model_factory, ordered_view, Xg, y, labels, train_pos, eval_pos, eps).mean())
        group_losses[g] = loss_g
        deltas[g] = loss_g - full_loss
    return {"full_loss": full_loss, "group_losses": group_losses, "deltas": deltas, "groups": groups}


# --- ordered velo transition (used by positive-world recovery) -------------------------

def velo_gap_prev_transition(table: pd.DataFrame) -> np.ndarray:
    r"""``|velo_{t-1} - velo_{t-2}|`` per row (NaN when fewer than two priors).

    This is the quantity the positive world keys its planted effect on and that the O view
    represents (via ``o_velo_delta_last`` / the slot speeds). Computed from the realised
    ``exec_release_speed`` of the prior pitches within each PA.
    """
    t = table.sort_values(["pa_id", "pitch_number"], kind="stable")
    g = t.groupby("pa_id", sort=False)["exec_release_speed"]
    prev1 = g.shift(1)
    prev2 = g.shift(2)
    gap = (prev1 - prev2).abs()
    return gap.reindex(table.index).to_numpy()


# --- acceptance verdicts (SPEC 11) -----------------------------------------------------

def _delta_order_ci(ab: dict, table: pd.DataFrame, seed: int, n_boot: int) -> dict:
    """Cluster-bootstrap CI for ``Delta_order = min(Loss_U, Loss_L1) - Loss_O`` on eval."""
    per_row = ab["per_row"]
    eval_pos = ab["eval_pos"]
    eval_df = table.iloc[eval_pos][["pitcher", "game_pk"]].reset_index(drop=True)
    lu, ll, lo = per_row["U"], per_row["L1"], per_row["O"]

    def metric(idx):
        return float(min(lu[idx].mean(), ll[idx].mean()) - lo[idx].mean())

    return clustered_ci(metric, eval_df, cluster="pitcher_game", n_boot=n_boot, seed=seed)


def run_null_world_acceptance(
    table: pd.DataFrame,
    model_factory,
    train_mask=None,
    eval_mask=None,
    n_permutations: int = 24,
    seed: int = 0,
    target: str = "outcome1",
    delta_threshold: float = 0.01,
    alpha: float = 0.05,
    ci_boot: int = 200,
) -> dict:
    """SPEC ``11`` null-world verdict: no ordered outcome dependence should be found.

    Detection requires **both** a permutation test that fires (``p < alpha``) **and** a
    ``Delta_order`` cluster-CI lower bound above 0. The null world should trigger neither.

    Returns
    -------
    dict
        ``verdict`` (``'NULL_CONFIRMED'`` / ``'NULL_VIOLATED'``), ``delta_order``,
        ``delta_order_ci``, ``permutation_p``, ``permutation_fired``,
        ``order_effect_detected``, ``losses``, ``abs_delta_below_threshold``.
    """
    ab = order_ablation(
        table, model_factory, train_mask, eval_mask, target=target, views=("C", "U", "L1", "O")
    )
    ci = _delta_order_ci(ab, table, seed=seed, n_boot=ci_boot)
    perm = history_permutation_test(
        table, model_factory, n_permutations=n_permutations, seed=seed, target=target,
        train_mask=train_mask, eval_mask=eval_mask, alpha=alpha,
    )
    delta_order = ab["deltas"]["delta_order"]
    detected = bool(perm["fired"] and ci["lo"] > 0)
    return {
        "world": "null",
        "verdict": "NULL_CONFIRMED" if not detected else "NULL_VIOLATED",
        "order_effect_detected": detected,
        "delta_order": delta_order,
        "delta_order_ci": ci,
        "abs_delta_below_threshold": bool(abs(delta_order) < delta_threshold),
        "permutation_p": perm["p_value"],
        "permutation_fired": bool(perm["fired"]),
        "losses": ab["losses"],
    }


def run_positive_world_acceptance(
    table: pd.DataFrame,
    model_factory,
    train_mask=None,
    eval_mask=None,
    n_permutations: int = 24,
    seed: int = 0,
    target: str = "outcome1",
    velo_gap_threshold: float = 5.0,
    alpha: float = 0.05,
    ci_boot: int = 200,
) -> dict:
    """SPEC ``11`` positive-world verdict: the planted ordered effect should be recovered.

    Detection (as in the null verdict) needs the permutation test to fire **and** the
    ``Delta_order`` CI to exclude 0 from below. Additionally the effect is *recovered*: the
    O-view outcome model's predicted whiff probability is compared between rows whose
    ordered velo transition clears the threshold and those that do not; the lift should be
    positive (matching the planted sign) and of comparable magnitude.

    Returns
    -------
    dict
        ``verdict`` (``'POSITIVE_CONFIRMED'`` / ``'POSITIVE_MISSED'``), ``delta_order``,
        ``delta_order_ci``, ``permutation_p``, ``permutation_fired``,
        ``order_effect_detected``, ``recovered_whiff_lift``, ``recovered_sign_ok``,
        ``losses``.
    """
    ab = order_ablation(
        table, model_factory, train_mask, eval_mask, target=target, views=("C", "U", "L1", "O")
    )
    ci = _delta_order_ci(ab, table, seed=seed, n_boot=ci_boot)
    perm = history_permutation_test(
        table, model_factory, n_permutations=n_permutations, seed=seed, target=target,
        train_mask=train_mask, eval_mask=eval_mask, alpha=alpha,
    )
    delta_order = ab["deltas"]["delta_order"]
    detected = bool(perm["fired"] and ci["lo"] > 0)

    # Recover the effect from the O-view outcome model's predicted whiff probability.
    recovered = float("nan")
    if target == "outcome1":
        n = len(table)
        if train_mask is None and eval_mask is None:
            tm, em = _default_train_eval(table)
            train_pos, eval_pos = np.flatnonzero(tm), np.flatnonzero(em)
        else:
            train_pos = _mask_positions(train_mask, n)
            eval_pos = _mask_positions(eval_mask, n)
        labels = _target_labels("outcome1")
        y = _target_values(table, "outcome1")
        X = _build_view_matrix(table, "O", True)
        model = model_factory("O")
        model.fit(X.iloc[train_pos], y[train_pos])
        proba = _align_proba(model, X.iloc[eval_pos], labels)
        whiff = proba[:, labels.index("whiff")]
        gap = velo_gap_prev_transition(table)[eval_pos]
        trig = gap >= velo_gap_threshold
        if trig.any() and (~trig).any():
            recovered = float(whiff[trig].mean() - whiff[~trig].mean())

    sign_ok = bool(np.isfinite(recovered) and recovered > 0)
    return {
        "world": "positive",
        "verdict": "POSITIVE_CONFIRMED" if (detected and sign_ok) else "POSITIVE_MISSED",
        "order_effect_detected": detected,
        "delta_order": delta_order,
        "delta_order_ci": ci,
        "permutation_p": perm["p_value"],
        "permutation_fired": bool(perm["fired"]),
        "recovered_whiff_lift": recovered,
        "recovered_sign_ok": sign_ok,
        "losses": ab["losses"],
    }
