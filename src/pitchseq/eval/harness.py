"""The one evaluation entry point every workstream uses (SPEC ``8``).

``evaluate_predictions`` joins a standard prediction table (:mod:`.predictions`) to the
decision table on ``row_id``, validates it, and computes the applicable metric block for
whichever prediction groups are present -- for every reporting slice in
``config['report_slices']`` (``all`` / ``seq_eligible`` / ``long_pa`` / ``two_strike`` /
``three_ball`` / ``first_pitch``) -- with cluster-bootstrap CIs on the headline metrics.

``compare_views`` takes the same model's predictions under several state views and produces
the SPEC ``6`` ablation-delta table (``Delta_order`` / ``Delta_matchup``) with clustered
CIs. ``write_report`` serialises a report dict to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..families import FAMILIES
from ..outcomes import OUTCOME1, OUTCOME2
from . import metrics as M
from .predictions import (
    ACTION_PROB_COLS,
    OUTCOME1_PROB_COLS,
    OUTCOME2_PROB_COLS,
    prediction_groups_present,
    validate_predictions,
)

__all__ = ["evaluate_predictions", "compare_views", "write_report", "slice_masks"]

_JOIN_COLS = ["row_id", "family", "outcome1", "outcome2", "R", "pitcher", "game_pk", "balls", "strikes", "pitch_number"]


def slice_masks(df: pd.DataFrame, slices) -> dict:
    """Boolean row masks for the requested SPEC ``6`` reporting slices."""
    pn = df["pitch_number"].to_numpy()
    balls = df["balls"].to_numpy()
    strikes = df["strikes"].to_numpy()
    all_masks = {
        "all": np.ones(len(df), bool),
        "seq_eligible": pn >= 2,
        "long_pa": pn >= 3,
        "two_strike": strikes >= 2,
        "three_ball": balls >= 3,
        "first_pitch": pn == 1,
    }
    return {s: all_masks[s] for s in slices if s in all_masks}


def _join(pred_df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    tab = table[[c for c in _JOIN_COLS if c in table.columns]].copy()
    merged = pred_df.merge(tab, on="row_id", how="inner")
    return merged


def _proba(df: pd.DataFrame, cols) -> np.ndarray:
    return df[list(cols)].to_numpy(dtype=np.float64)


def _marginal_baseline_loss(y_labels, labels) -> float:
    """Log loss of predicting the empirical class frequency for every row (skill anchor)."""
    y = np.asarray(y_labels)
    n = len(y)
    if n == 0:
        return float("nan")
    counts = np.array([np.sum(y == lab) for lab in labels], dtype=np.float64)
    freq = (counts + 1e-9) / (counts.sum() + 1e-9 * len(labels))
    proba = np.tile(freq, (n, 1))
    return M.log_loss(y, proba, labels=labels)


def _headline_ci(per_row: np.ndarray, sub: pd.DataFrame, n_boot: int, seed: int) -> dict:
    def metric(idx):
        return float(per_row[idx].mean())

    return M.clustered_ci(metric, sub[["pitcher", "game_pk"]].reset_index(drop=True), n_boot=n_boot, seed=seed)


def _action_block(sub: pd.DataFrame, cols, labels, n_boot: int, seed: int) -> dict:
    proba = _proba(sub, cols)
    y = sub["family"].astype("object").to_numpy()
    per_row = M.log_loss_per_row(y, proba, labels)
    pred_lab = np.array(labels)[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)
    correct = (pred_lab == y).astype(float)
    base = _marginal_baseline_loss(y, labels)
    ll = float(per_row.mean())
    return {
        "n": int(len(sub)),
        "log_loss": ll,
        "brier": M.brier_multiclass(y, proba, labels),
        "top1_acc": M.top_k_accuracy(y, proba, 1, labels),
        "top2_acc": M.top_k_accuracy(y, proba, 2, labels),
        "macro_f1": M.macro_f1(y, pred_lab, labels),
        "log_loss_per_pitcher_macro": M.per_pitcher_macro(per_row, sub["pitcher"].to_numpy()),
        "log_loss_micro": M.micro_mean(per_row),
        "ece": M.ece(correct, conf),
        "skill_vs_marginal": M.skill_vs_baseline(ll, base),
        "log_loss_ci": _headline_ci(per_row, sub, n_boot, seed),
    }


def _outcome1_block(sub: pd.DataFrame, cols, labels, n_boot: int, seed: int) -> dict:
    proba = _proba(sub, cols)
    y = sub["outcome1"].astype("object").to_numpy()
    per_row = M.log_loss_per_row(y, proba, labels)
    pred_lab = np.array(labels)[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)
    correct = (pred_lab == y).astype(float)
    base = _marginal_baseline_loss(y, labels)
    ll = float(per_row.mean())
    return {
        "n": int(len(sub)),
        "log_loss": ll,
        "brier": M.brier_multiclass(y, proba, labels),
        "top1_acc": M.top_k_accuracy(y, proba, 1, labels),
        "ece": M.ece(correct, conf),
        "skill_vs_marginal": M.skill_vs_baseline(ll, base),
        "log_loss_ci": _headline_ci(per_row, sub, n_boot, seed),
    }


def _outcome2_block(sub: pd.DataFrame, cols, labels, n_boot: int, seed: int) -> dict:
    ip = sub["outcome2"].notna().to_numpy()
    sub_ip = sub.loc[ip]
    if len(sub_ip) == 0:
        return {"n": 0}
    proba = _proba(sub_ip, cols)
    y = sub_ip["outcome2"].astype("object").to_numpy()
    per_row = M.log_loss_per_row(y, proba, labels)
    return {
        "n": int(len(sub_ip)),
        "log_loss": float(per_row.mean()),
        "brier": M.brier_multiclass(y, proba, labels),
        "log_loss_ci": _headline_ci(per_row, sub_ip, n_boot, seed),
    }


def _exp_reward_block(sub: pd.DataFrame, n_boot: int, seed: int) -> dict:
    ok = sub["R"].notna().to_numpy() & np.isfinite(sub["exp_reward"].to_numpy())
    s = sub.loc[ok]
    if len(s) == 0:
        return {"n": 0}
    observed = s["R"].to_numpy(dtype=float)
    predicted = s["exp_reward"].to_numpy(dtype=float)
    err = observed - predicted
    per_row_sq = err ** 2

    def rmse_metric(idx):
        return float(np.sqrt(per_row_sq[idx].mean()))

    ci = M.clustered_ci(rmse_metric, s[["pitcher", "game_pk"]].reset_index(drop=True), n_boot=n_boot, seed=seed)
    return {
        "n": int(len(s)),
        **M.run_value_mae_rmse(observed, predicted),
        "calibration": M.run_value_calibration(observed, predicted),
        "rmse_ci": ci,
    }


def evaluate_predictions(
    pred_df: pd.DataFrame,
    table: pd.DataFrame,
    config: dict | None = None,
    n_boot: int = 200,
    seed: int | None = None,
) -> dict:
    """Score a prediction table against the decision table over every reporting slice.

    Parameters
    ----------
    pred_df : pandas.DataFrame
        Standard prediction table (SPEC ``8.1``); validated internally.
    table : pandas.DataFrame
        Decision table (must contain ``row_id`` and the label / cluster columns).
    config : dict, optional
        Study config (for ``report_slices``); loaded from default when ``None``.
    n_boot : int, optional
        Cluster-bootstrap replicates for headline CIs (default 200).
    seed : int, optional
        Bootstrap seed (default: config global seed).

    Returns
    -------
    dict
        ``{'model_id', 'state_view', 'groups_present', 'n_joined', 'slices': {slice:
        {group: metrics}}}``.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))
    validate_predictions(pred_df)

    merged = _join(pred_df, table)
    groups = prediction_groups_present(pred_df)
    has_reward = "exp_reward" in pred_df.columns
    report_slices = config.get("report_slices", ["all"])
    masks = slice_masks(merged, report_slices)

    out: dict = {
        "model_id": str(pred_df["model_id"].iloc[0]) if "model_id" in pred_df and len(pred_df) else None,
        "state_view": str(pred_df["state_view"].iloc[0]) if "state_view" in pred_df and len(pred_df) else None,
        "groups_present": groups + (["exp_reward"] if has_reward else []),
        "n_joined": int(len(merged)),
        "slices": {},
    }

    for sname, mask in masks.items():
        sub = merged.loc[mask]
        block: dict = {"n": int(len(sub))}
        if len(sub) == 0:
            out["slices"][sname] = block
            continue
        if "action_prob" in groups:
            block["action_prob"] = _action_block(sub, ACTION_PROB_COLS, list(FAMILIES), n_boot, seed)
        if "outcome1_prob" in groups:
            block["outcome1_prob"] = _outcome1_block(sub, OUTCOME1_PROB_COLS, list(OUTCOME1), n_boot, seed)
        if "outcome2_prob" in groups:
            block["outcome2_prob"] = _outcome2_block(sub, OUTCOME2_PROB_COLS, list(OUTCOME2), n_boot, seed)
        if has_reward:
            block["exp_reward"] = _exp_reward_block(sub, n_boot, seed)
        out["slices"][sname] = block

    return out


def _view_per_row_loss(pred_df: pd.DataFrame, table: pd.DataFrame, target: str):
    """Per-row loss + join keys for one view's predictions (aligned by row_id)."""
    if target == "outcome1":
        cols, labels, ycol = OUTCOME1_PROB_COLS, list(OUTCOME1), "outcome1"
    elif target == "family":
        cols, labels, ycol = ACTION_PROB_COLS, list(FAMILIES), "family"
    else:
        raise ValueError(f"target must be 'outcome1' or 'family', got {target!r}")
    merged = _join(pred_df[["row_id", *cols]], table)
    merged = merged.sort_values("row_id", kind="stable").reset_index(drop=True)
    proba = _proba(merged, cols)
    y = merged[ycol].astype("object").to_numpy()
    per_row = M.log_loss_per_row(y, proba, labels)
    return merged, per_row


def compare_views(
    preds_by_view: dict,
    table: pd.DataFrame,
    config: dict | None = None,
    target: str = "outcome1",
    n_boot: int = 200,
    seed: int | None = None,
) -> dict:
    """Ablation-delta table (SPEC ``6``) from one model's predictions across views.

    Parameters
    ----------
    preds_by_view : dict
        ``{view: prediction_df}`` for at least ``U``, ``L1`` and ``O`` (``OM`` optional).
        Each is joined to ``table`` and reduced to a common set of ``row_id``s.
    table : pandas.DataFrame
        Decision table.
    config : dict, optional
        Study config.
    target : {'outcome1', 'family'}
        Which prediction group to score.
    n_boot : int, optional
        Bootstrap replicates for the delta CIs.
    seed : int, optional
        Bootstrap seed.

    Returns
    -------
    dict
        ``losses`` (per view), ``deltas`` (``delta_order`` / ``delta_matchup``),
        ``delta_order_ci``, ``delta_matchup_ci`` (when ``OM`` present), ``n_common``.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))

    merged_by_view = {}
    per_row_by_view = {}
    for v, pdf in preds_by_view.items():
        merged, per_row = _view_per_row_loss(pdf, table, target)
        merged_by_view[v] = merged
        per_row_by_view[v] = per_row

    # Restrict to the common row_ids across all views (aligned, sorted).
    common = None
    for merged in merged_by_view.values():
        ids = set(merged["row_id"].tolist())
        common = ids if common is None else (common & ids)
    common = sorted(common) if common else []

    common_set = set(common)
    aligned_loss = {}
    ref_keys = None
    for v, merged in merged_by_view.items():
        # ``merged`` is already sorted by row_id, so the common-row mask keeps row_id order.
        keep = merged["row_id"].isin(common_set).to_numpy()
        aligned_loss[v] = per_row_by_view[v][keep]
        if ref_keys is None:
            ref_keys = merged.loc[keep, ["pitcher", "game_pk"]].reset_index(drop=True)

    losses = {v: float(arr.mean()) for v, arr in aligned_loss.items()}
    deltas = M.ablation_deltas(losses)

    def order_metric(idx):
        return float(min(aligned_loss["U"][idx].mean(), aligned_loss["L1"][idx].mean()) - aligned_loss["O"][idx].mean())

    delta_order_ci = M.clustered_ci(order_metric, ref_keys, n_boot=n_boot, seed=seed)

    result = {
        "target": target,
        "losses": losses,
        "deltas": deltas,
        "delta_order_ci": delta_order_ci,
        "n_common": int(len(common)),
    }
    if "OM" in aligned_loss:
        def matchup_metric(idx):
            return float(aligned_loss["O"][idx].mean() - aligned_loss["OM"][idx].mean())

        result["delta_matchup_ci"] = M.clustered_ci(matchup_metric, ref_keys, n_boot=n_boot, seed=seed)
    return result


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_report(report: dict, path: str | Path) -> Path:
    """Write an evaluation report dict to JSON.

    Parameters
    ----------
    report : dict
        The report (from :func:`evaluate_predictions` or :func:`compare_views`).
    path : str or pathlib.Path
        Destination file (parent directories created).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=_json_default)
    return out
