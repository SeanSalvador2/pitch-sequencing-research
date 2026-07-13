"""The shared evaluation harness (SPEC 8): evaluate_predictions / compare_views / report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from model_factories import make_lgbm_factory

from pitchseq.decision_table import build_decision_table
from pitchseq.eval.baselines import TransitionBaseline
from pitchseq.eval.falsification import _align_proba, _build_view_matrix  # internal reuse
from pitchseq.eval.harness import compare_views, evaluate_predictions, write_report
from pitchseq.families import FAMILIES
from pitchseq.outcomes import OUTCOME1
from pitchseq.synth import make_null_world

_PARAMS = dict(num_leaves=8, min_child_samples=40, reg_lambda=5.0, n_estimators=40, max_depth=4)


def _table():
    raw, _ = make_null_world(n_games=14, seed=2, innings_per_game=5)
    return build_decision_table(raw)


def _meta(df, view, n_params=10):
    df["model_id"] = f"test_{view}"
    df["state_view"] = view
    df["seconds"] = 0.5
    df["peak_mem_mb"] = 50.0
    df["n_params"] = n_params
    return df


def _action_pred_df(table):
    proba = TransitionBaseline().fit(table).predict_proba(table)
    df = pd.DataFrame({"row_id": table["row_id"].to_numpy()})
    for j, f in enumerate(FAMILIES):
        df[f"action_prob_{f}"] = proba[:, j]
    return _meta(df, "L1")


def _outcome_pred_df(table, view, factory):
    X = _build_view_matrix(table, view, condition_on_action=True)
    y = table["outcome1"].astype("object").to_numpy()
    model = factory(view)
    model.fit(X, y)
    proba = _align_proba(model, X, list(OUTCOME1))
    df = pd.DataFrame({"row_id": table["row_id"].to_numpy()})
    for j, c in enumerate(OUTCOME1):
        df[f"outcome1_prob_{c}"] = proba[:, j]
    df["exp_reward"] = 0.0  # placeholder E[R]; exercises the reward metric block
    return _meta(df, view)


def test_evaluate_action_predictions_and_report(tmp_path):
    table = _table()
    pred = _action_pred_df(table)
    report = evaluate_predictions(pred, table, n_boot=50, seed=0)

    assert report["state_view"] == "L1"
    assert "action_prob" in report["groups_present"]
    # Every configured reporting slice is present.
    for s in ("all", "seq_eligible", "long_pa", "two_strike", "three_ball", "first_pitch"):
        assert s in report["slices"]
    allblk = report["slices"]["all"]["action_prob"]
    assert np.isfinite(allblk["log_loss"])
    assert 0.0 <= allblk["top1_acc"] <= 1.0
    assert 0.0 <= allblk["top2_acc"] <= 1.0
    assert allblk["top2_acc"] >= allblk["top1_acc"] - 1e-9  # top-2 dominates top-1
    assert allblk["log_loss_ci"]["lo"] <= allblk["log_loss"] <= allblk["log_loss_ci"]["hi"]
    # log-loss skill vs the marginal baseline is defined.
    assert "loss_ratio" in allblk["skill_vs_marginal"]

    path = write_report(report, tmp_path / "report.json")
    assert path.is_file()


def test_evaluate_outcome_and_reward():
    table = _table()
    factory = make_lgbm_factory(**_PARAMS)
    pred = _outcome_pred_df(table, "O", factory)
    report = evaluate_predictions(pred, table, n_boot=50, seed=0)
    assert set(report["groups_present"]) >= {"outcome1_prob", "exp_reward"}
    o1 = report["slices"]["all"]["outcome1_prob"]
    assert np.isfinite(o1["log_loss"]) and np.isfinite(o1["brier"])
    rew = report["slices"]["all"]["exp_reward"]
    assert np.isfinite(rew["rmse"]) and "calibration" in rew


def test_compare_views_produces_ablation_deltas():
    table = _table()
    factory = make_lgbm_factory(**_PARAMS)
    preds = {v: _outcome_pred_df(table, v, factory) for v in ("U", "L1", "O")}
    result = compare_views(preds, table, target="outcome1", n_boot=50, seed=0)

    assert set(result["losses"]) == {"U", "L1", "O"}
    assert np.isfinite(result["deltas"]["delta_order"])
    assert result["deltas"]["delta_matchup"] is None  # OM not provided
    ci = result["delta_order_ci"]
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert result["n_common"] == len(table)
