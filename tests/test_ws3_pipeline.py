"""WS3 end-to-end pipeline tests (SPEC 12.3; decisions D32-D35).

The stack is exercised on the shared synthetic worlds and scored only through the shared
harness. The null world must stay quiet (``NULL_QUIET`` -- no ordered outcome dependence
found) and the positive world must recover the planted previous-transition effect nearly
directly (``MECHANISM_RECOVERED``): the outcome-target ``Delta_order`` CI excludes 0, the
history-permutation test fires, the mechanism ablation ranks the velo-difference group top,
and the recovered whiff-lift is a large fraction of the planted truth -- far above WS1's
~0.03 family-proxy attenuation. Per-view checkpointing (build one stage, resume) is checked,
as is standard-schema validity and the persisted-artifact contract WS4/5/7 consume.

Sizes / seeds / permutation counts are calibrated small so the assertions are reproducible
and the whole suite stays fast; LightGBM runs single-threaded and deterministic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pitchseq.eval.predictions import (
    ACTION_PROB_COLS,
    OUTCOME1_PROB_COLS,
    OUTCOME2_PROB_COLS,
    load_predictions,
    validate_predictions,
)
from workstreams.ws3_gbdt_stack.model import load_ws3_artifacts
from workstreams.ws3_gbdt_stack.run_ws3 import run_ws3

# Fast, deterministic LightGBM for both the main models and the falsification factory.
_FAST = {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.1}
_COMMON = dict(params=_FAST, fals_params=_FAST, n_boot=30, ci_boot=30, seed=7)


@pytest.fixture(scope="module")
def null_result():
    """A full null-world run (written to a module temp dir for the artifact tests)."""
    out = tempfile.mkdtemp(prefix="ws3_null_")
    return run_ws3(synth="null", stage="all", out=out, n_games=80, n_perm=12,
                   write_outputs=True, **_COMMON)


@pytest.fixture(scope="module")
def positive_result():
    """A full positive-world run (24 permutations so the p-value can clear 0.05)."""
    return run_ws3(synth="positive", stage="all", out=tempfile.mkdtemp(prefix="ws3_pos_"),
                   n_games=140, n_perm=24, write_outputs=False, **_COMMON)


# --- D35 null world: quiet -------------------------------------------------------------

def test_null_world_quiet(null_result):
    """The null world yields no ordered outcome dependence: NULL_QUIET (D35)."""
    fal = null_result["falsification"]
    assert fal["d35_verdict"] == "NULL_QUIET"
    assert fal["d35_pass"] is True
    assert null_result["d35_verdict"] == "NULL_QUIET"
    assert fal["acceptance"]["order_effect_detected"] is False


def test_null_world_delta_order_not_significant(null_result):
    """Neither the selection nor the outcome1 Delta_order is significantly positive (D21)."""
    d = null_result["val"]["deltas"]
    assert d["selection"]["delta_order_ci"]["lo"] <= 0
    assert d["outcome1"]["delta_order_ci"]["lo"] <= 0


def test_null_world_permutation_does_not_fire(null_result):
    acc = null_result["falsification"]["acceptance"]
    assert acc["permutation_fired"] is False


def test_behavior_O_beats_count_references_on_null_selection(null_result):
    """The GBDT with O features exploits the planted order-dependent selection habit:
    the O behavior model beats the count-based references on next-pitch log loss."""
    sel = null_result["val"]["central"]["selection"]
    refs = null_result["baseline_selection_loss"]
    o_ll = sel["O"]["log_loss"]
    assert o_ll < refs["pitcher_count_prev"]
    assert o_ll < refs["transition"]
    assert o_ll < refs["global_count_hand"]


# --- D35 positive world: mechanism recovered -------------------------------------------

def test_positive_world_mechanism_recovered(positive_result):
    """The planted previous-transition effect is recovered: MECHANISM_RECOVERED (D35)."""
    fal = positive_result["falsification"]
    assert fal["d35_verdict"] == "MECHANISM_RECOVERED"
    assert fal["d35_pass"] is True
    assert positive_result["d35_verdict"] == "MECHANISM_RECOVERED"


def test_positive_outcome_delta_order_significant(positive_result):
    """O beats U / L1 on the OUTCOME target with a clustered CI strictly above 0 (finding #2)."""
    do = positive_result["val"]["deltas"]["outcome1"]
    assert do["deltas"]["delta_order"] > 0
    assert do["delta_order_ci"]["lo"] > 0


def test_positive_locked_test_delta_order_significant(positive_result):
    """The ordered outcome edge also holds on the locked 2025 test season."""
    do = positive_result["test"]["deltas"]["outcome1"]
    assert do["delta_order_ci"]["lo"] > 0


def test_positive_permutation_fires(positive_result):
    """The stratified history-permutation test fires on the planted effect (p < 0.05)."""
    acc = positive_result["falsification"]["acceptance"]
    assert acc["permutation_fired"] is True
    assert acc["permutation_p"] < 0.05


def test_positive_mechanism_ablation_isolates_velo(positive_result):
    """Removing the velo-difference group hurts most -- the mechanism is isolated."""
    mech = positive_result["falsification"]["mechanism_ablation"]
    assert mech["top_group"] == "velo_diff"
    assert mech["ranking"][0] == "velo_diff"


def test_positive_recovery_ratio_far_above_ws1(positive_result):
    """The recovered whiff-lift is a large fraction of the planted truth -- far above WS1's
    ~0.03 family-proxy attenuation (the D35 contrast exhibit)."""
    fal = positive_result["falsification"]
    ratio = fal["recovery_ratio"]
    assert ratio > fal["recovery_ratio_floor"]  # documented floor (0.30)
    assert ratio > 10 * fal["ws1_attenuation_reference"]  # >> WS1's 0.03 attenuation
    assert np.isfinite(fal["recovered_whiff_lift"])


# --- decomposed vs direct cross-check (D32) --------------------------------------------

def test_decomposed_vs_direct_agreement_reported(null_result):
    """Every view reports the decomposed-vs-direct disagreement and none is flagged large."""
    dis = null_result["disagreement"]
    assert set(dis) == set(null_result["views"])
    for view, d in dis.items():
        assert np.isfinite(d["mean_abs_diff"])
        assert d["flag"] is False


# --- standard-schema validity ----------------------------------------------------------

def test_predictions_valid_and_row_id_aligned(null_result):
    """Per-view behavior + outcome predictions conform to SPEC 8.1."""
    out_dir = Path(null_result["outputs"]["artifacts_dir"])
    for view in null_result["views"]:
        bp = load_predictions(out_dir / f"pred_behavior_null_{view}.parquet")
        assert validate_predictions(bp) is True
        assert all(c in bp.columns for c in ACTION_PROB_COLS)
        op = load_predictions(out_dir / f"pred_outcome_null_{view}.parquet")
        assert validate_predictions(op) is True
        assert all(c in op.columns for c in OUTCOME1_PROB_COLS + OUTCOME2_PROB_COLS)
        assert "exp_reward" in op.columns and "exp_reward_sd" in op.columns


def test_pipeline_writes_all_artifacts(null_result):
    """A full run writes models, predictions, q-grids, propensities, a report and runmeta."""
    out_dir = Path(null_result["outputs"]["artifacts_dir"])
    assert (out_dir / "ws3_report_null.json").is_file()
    assert (out_dir / "ws3_null.runmeta.json").is_file()
    for view in null_result["views"]:
        assert (out_dir / f"behavior_{view}.joblib").is_file()
        assert (out_dir / f"outcome_{view}.joblib").is_file()
        assert (out_dir / f"qgrid_null_{view}.parquet").is_file()
        assert (out_dir / f"propensity_null_{view}.parquet").is_file()


def test_load_ws3_artifacts_roundtrip(null_result):
    """The persisted artifacts reload into a bundle WS4/5/7 can query (decision D33)."""
    out_dir = Path(null_result["outputs"]["artifacts_dir"])
    art = load_ws3_artifacts(out_dir)
    assert set(art.views) == set(null_result["views"])
    from pitchseq.synth import make_null_world
    from pitchseq.decision_table import build_decision_table

    raw, _ = make_null_world(n_games=10, seed=7, innings_per_game=6)
    probe = build_decision_table(raw)
    mu = art.propensities(probe, "O")
    qg = art.q_grid(probe, "O")
    assert mu.shape == (len(probe), 8) and np.allclose(mu.sum(axis=1), 1.0, atol=1e-9)
    assert qg.shape == (len(probe), 8) and np.isfinite(qg).all()


# --- per-view checkpointing (interrupt / resume) ---------------------------------------

def test_checkpoint_resume_skips_completed(tmp_path):
    """Building one stage and re-running ``--stage all`` skips the completed views (D12)."""
    out = str(tmp_path)
    views = ["C", "U", "L1", "O"]
    first = run_ws3(synth="null", stage="behavior", out=out, views=views, n_games=40,
                    write_outputs=True, **_COMMON)
    assert sorted(first["checkpoints"]["behavior"]["built"]) == sorted(views)

    second = run_ws3(synth="null", stage="all", out=out, views=views, n_games=40,
                     n_perm=8, write_outputs=True, **_COMMON)
    # Behavior was already done -> skipped, not rebuilt; the rest builds.
    assert sorted(second["checkpoints"]["behavior"]["skipped"]) == sorted(views)
    assert second["checkpoints"]["behavior"]["built"] == []
    assert sorted(second["checkpoints"]["outcome"]["built"]) == sorted(views)


def test_force_rebuilds(tmp_path):
    """``force=True`` ignores the .done markers and rebuilds."""
    out = str(tmp_path)
    run_ws3(synth="null", stage="behavior", out=out, views=["C"], n_games=30,
            write_outputs=True, **_COMMON)
    again = run_ws3(synth="null", stage="behavior", out=out, views=["C"], n_games=30,
                    write_outputs=True, force=True, **_COMMON)
    assert again["checkpoints"]["behavior"]["built"] == ["C"]
