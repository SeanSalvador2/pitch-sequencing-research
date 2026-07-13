"""Scoring metrics (SPEC 8.2) -- hand-computed checks on tiny arrays."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pitchseq.eval import metrics as M


def test_log_loss_hand_computed():
    proba = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = [0, 1]
    expected = (-math.log(0.8) - math.log(0.7)) / 2
    assert M.log_loss(y, proba, labels=[0, 1]) == pytest.approx(expected)
    per_row = M.log_loss_per_row(y, proba, labels=[0, 1])
    assert per_row[0] == pytest.approx(-math.log(0.8))
    assert per_row[1] == pytest.approx(-math.log(0.7))


def test_log_loss_clips_zero_probability():
    proba = np.array([[1.0, 0.0]])
    # True class has probability 0 -> clipped to eps, giving -log(eps).
    val = M.log_loss([1], proba, labels=[0, 1], eps=1e-12)
    assert val == pytest.approx(-math.log(1e-12))


def test_brier_multiclass_hand_computed():
    proba = np.array([[0.7, 0.2, 0.1]])
    # class 0 true: (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.09+0.04+0.01 = 0.14
    assert M.brier_multiclass([0], proba, labels=[0, 1, 2]) == pytest.approx(0.14)


def test_top_k_accuracy():
    proba = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    y = [1, 2]
    assert M.top_k_accuracy(y, proba, k=1, labels=[0, 1, 2]) == pytest.approx(0.5)  # row0 wrong, row1 right
    assert M.top_k_accuracy(y, proba, k=2, labels=[0, 1, 2]) == pytest.approx(1.0)  # both in top-2


def test_macro_f1_hand_computed():
    # 2 classes, perfect on class 1, one miss on class 0.
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    # class 0: tp=1, fp=0, fn=1 -> f1 = 2/(2+0+1)=2/3 ; class 1: tp=2, fp=1, fn=0 -> f1=4/5
    assert M.macro_f1(y_true, y_pred, labels=[0, 1]) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_ece_hand_computed():
    # Perfectly calibrated: a single bin of predictions 0.5 with a 50% positive rate.
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.full(6, 0.5)
    assert M.ece(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)
    # Miscalibrated: predict 0.9 but only half are positive -> |0.5-0.9| = 0.4.
    y2 = np.array([1, 0, 1, 0])
    p2 = np.array([0.9, 0.9, 0.9, 0.9])
    assert M.ece(y2, p2, n_bins=10) == pytest.approx(0.4)


def test_run_value_calibration_recovers_line():
    x = np.linspace(-1, 1, 50)
    y = 2.0 * x + 1.0  # perfect line
    cal = M.run_value_calibration(y, x)
    assert cal["intercept"] == pytest.approx(1.0)
    assert cal["slope"] == pytest.approx(2.0)
    assert cal["r2"] == pytest.approx(1.0)


def test_run_value_mae_rmse():
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 5.0])  # errors 0,0,-2
    res = M.run_value_mae_rmse(observed, predicted)
    assert res["mae"] == pytest.approx(2 / 3)
    assert res["rmse"] == pytest.approx(math.sqrt(4 / 3))


def test_ablation_deltas_formula():
    losses = {"C": 1.0, "U": 0.9, "L1": 0.85, "O": 0.8, "OM": 0.78}
    d = M.ablation_deltas(losses)
    assert d["delta_order"] == pytest.approx(min(0.9, 0.85) - 0.8)  # 0.05
    assert d["delta_matchup"] == pytest.approx(0.8 - 0.78)  # 0.02
    # OM optional.
    d2 = M.ablation_deltas({"U": 0.9, "L1": 0.85, "O": 0.8})
    assert d2["delta_matchup"] is None
    with pytest.raises(KeyError):
        M.ablation_deltas({"U": 0.9, "O": 0.8})  # missing L1


def test_skill_vs_baseline():
    s = M.skill_vs_baseline(0.8, 1.0)
    assert s["loss_ratio"] == pytest.approx(0.8)
    assert s["loss_diff"] == pytest.approx(0.2)
    assert s["skill"] == pytest.approx(0.2)


def test_per_pitcher_macro_vs_micro():
    per_row = np.array([0.0, 0.0, 3.0])  # pitcher A has two 0s, pitcher B has one 3
    pitcher = np.array([1, 1, 2])
    # macro: mean over pitchers of within-pitcher mean = (0 + 3)/2 = 1.5
    assert M.per_pitcher_macro(per_row, pitcher) == pytest.approx(1.5)
    # micro: pitch-weighted mean = 1.0
    assert M.micro_mean(per_row) == pytest.approx(1.0)


def test_clustered_ci_reproducible_and_sane():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "pitcher": rng.integers(0, 8, n),
            "game_pk": rng.integers(0, 20, n),
            "val": rng.normal(1.0, 0.5, n),
        }
    ).reset_index(drop=True)
    vals = df["val"].to_numpy()

    def metric(idx):
        return float(vals[idx].mean())

    ci1 = M.clustered_ci(metric, df, n_boot=200, seed=42)
    ci2 = M.clustered_ci(metric, df, n_boot=200, seed=42)
    assert ci1 == ci2  # identical under the same seed
    assert ci1["lo"] <= ci1["point"] <= ci1["hi"]
    assert ci1["point"] == pytest.approx(vals.mean())
    # Different seed -> generally different interval bounds.
    ci3 = M.clustered_ci(metric, df, n_boot=200, seed=7)
    assert (ci3["lo"], ci3["hi"]) != (ci1["lo"], ci1["hi"])
