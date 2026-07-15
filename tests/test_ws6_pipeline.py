"""WS6 end-to-end pipeline tests (SPEC 12.4; decisions D45-D47).

The GRU ladder is exercised on the shared synthetic worlds and scored only through the
shared harness. On the **null** world the D47 verdicts must fire: the selection GRU-O
rediscovers the planted no-three-in-a-row grammar (``GRU_GRAMMAR_DETECTED`` -- the
capacity-matched L1-vs-O twin delta on the repeat-context rows is CI-positive) while the
outcome target stays quiet (``GRU_OUTCOME_QUIET`` -- D21), and the motif probe shows repeat
suppression. Standard-schema validity, the L1/O capacity match, per-view checkpointing, and
the factory-driven falsification plumbing are checked too. The **positive** world's mechanism
recovery is asserted directionally at this lean scale (honest about the deep model's sample
cost; the magnitude is a Phase-2 full-data question).

Sizes / seeds / permutation counts are calibrated small (n_games=150, a compact GRU, few
epochs) so the assertions are reproducible and the whole suite stays fast; torch runs
single-threaded.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from pitchseq.eval.predictions import (  # noqa: E402
    ACTION_PROB_COLS,
    OUTCOME1_PROB_COLS,
    load_predictions,
    validate_predictions,
)
from workstreams.ws6_deep_seq.run_ws6 import run_ws6  # noqa: E402

# Compact, deterministic GRU calibrated so the null D47 gates fire with margin at lean scale.
_NULL_KW = dict(synth="null", n_games=150, seed=0, epochs=30, hidden=64, embed=16, batch=512,
                max_len=12, device="cpu", n_boot=60, n_perm=8, ci_boot=60)


@pytest.fixture(scope="module")
def null_result():
    out = tempfile.mkdtemp(prefix="ws6_null_")
    return run_ws6(out=out, write_outputs=True, **_NULL_KW)


# --- D47 null world: grammar detected, outcome quiet -----------------------------------

def test_null_grammar_detected(null_result):
    """The selection GRU-O rediscovers the grammar: GRU_GRAMMAR_DETECTED (D47)."""
    fal = null_result["falsification"]
    assert fal["grammar_verdict"] == "GRU_GRAMMAR_DETECTED"
    assert fal["grammar_detected"] is True
    assert fal["grammar_repeat_ci"]["lo"] > 0  # repeat-context L1-O twin delta is CI-positive


def test_null_outcome_quiet(null_result):
    """No ordered *outcome* dependence is found on the null world: GRU_OUTCOME_QUIET (D21)."""
    fal = null_result["falsification"]
    assert fal["outcome_verdict"] == "GRU_OUTCOME_QUIET"
    assert fal["outcome_delta_order_ci"]["lo"] <= 0
    assert fal["permutation"]["fired"] is False


def test_null_d47_pass(null_result):
    assert null_result["d47_pass"] is True
    assert null_result["falsification"]["d47_verdict"] == "GRU_GRAMMAR_DETECTED+GRU_OUTCOME_QUIET"


def test_null_aggregate_deltas_not_significant(null_result):
    """The *aggregate* selection and outcome Delta_order are not significantly positive
    (the grammar signal concentrates in the repeat-context slice; SPEC 13's 'O barely beats
    L1' on the whole set)."""
    sc = null_result["scores"]
    assert sc["selection"]["val_deltas"]["delta_order_ci"]["lo"] <= 0
    assert sc["outcome1"]["val_deltas"]["delta_order_ci"]["lo"] <= 0


def test_motif_probe_shows_repeat_suppression(null_result):
    """The trained selection GRU-O suppresses P(same family third time) after [X,X]."""
    motif = null_result["motif"]
    assert motif["mean_suppression"] > 0
    assert len(motif["per_family"]) >= 5


# --- selection skill vs references -----------------------------------------------------

def test_selection_O_beats_order_blind_references(null_result):
    """The O behavior GRU beats the order-blind count references on next-pitch log loss."""
    o_ll = null_result["scores"]["selection"]["central"]["O"]["log_loss"]
    refs = null_result["baseline_selection_loss"]
    assert o_ll < refs["global_count_hand"]
    assert o_ll < refs["transition"]


# --- capacity match --------------------------------------------------------------------

def test_L1_and_O_are_capacity_matched(null_result):
    """L1 and O are the *same* network (same GRU + head) -- identical parameter counts."""
    par = null_result["pareto"]["selection"]
    assert par["L1"]["n_params"] == par["O"]["n_params"]
    assert par["O"]["n_params"] > par["U"]["n_params"] > par["C"]["n_params"]


# --- standard-schema predictions -------------------------------------------------------

def test_predictions_valid_and_schema(null_result):
    out_dir = Path(null_result["outputs"]["artifacts_dir"])
    for view in null_result["views"]:
        sp = load_predictions(out_dir / f"pred_selection_null_{view}_val.parquet")
        assert validate_predictions(sp) is True
        assert all(c in sp.columns for c in ACTION_PROB_COLS)
        op = load_predictions(out_dir / f"pred_outcome1_null_{view}_val.parquet")
        assert validate_predictions(op) is True
        assert all(c in op.columns for c in OUTCOME1_PROB_COLS)


def test_pipeline_writes_report_and_runmeta(null_result):
    out_dir = Path(null_result["outputs"]["artifacts_dir"])
    assert (out_dir / "ws6_report_null.json").is_file()
    assert (out_dir / "ws6_null.runmeta.json").is_file()
    for view in null_result["views"]:
        assert (out_dir / f"model_null_selection_{view}.pt").is_file()


# --- per-view checkpoint resume (tiny, isolated) ---------------------------------------

def test_checkpoint_resume_loads(tmp_path):
    """A completed (view, target) is loaded from its checkpoint on re-run (D12)."""
    kw = dict(synth="null", n_games=20, seed=0, epochs=3, hidden=16, embed=8, batch=256,
              max_len=6, device="cpu", n_boot=10, n_perm=0, ci_boot=10,
              views=["C", "O"], targets=["selection"])
    first = run_ws6(out=str(tmp_path), write_outputs=True, **kw)
    assert sorted(first["checkpoints"]["selection"]["fitted"]) == ["C", "O"]
    second = run_ws6(out=str(tmp_path), write_outputs=True, **kw)
    assert sorted(second["checkpoints"]["selection"]["loaded"]) == ["C", "O"]
    assert second["checkpoints"]["selection"]["fitted"] == []


def test_force_refits(tmp_path):
    kw = dict(synth="null", n_games=20, seed=0, epochs=3, hidden=16, embed=8, batch=256,
              max_len=6, device="cpu", n_boot=10, n_perm=0, ci_boot=10,
              views=["C"], targets=["selection"])
    run_ws6(out=str(tmp_path), write_outputs=True, **kw)
    again = run_ws6(out=str(tmp_path), write_outputs=True, force=True, **kw)
    assert again["checkpoints"]["selection"]["fitted"] == ["C"]


# --- D47 positive world: mechanism recovered (directional at lean scale) ---------------

@pytest.fixture(scope="module")
def positive_result():
    out = tempfile.mkdtemp(prefix="ws6_pos_")
    return run_ws6(synth="positive", out=out, write_outputs=False, n_games=100, seed=0, epochs=20,
                   hidden=64, embed=16, batch=512, max_len=12, device="cpu", n_boot=40, n_perm=4,
                   ci_boot=40, effect_size=0.30)


def test_positive_mechanism_recovered_directionally(positive_result):
    """The GRU-O recovers the planted ordered whiff mechanism: the predicted whiff lift on
    triggered rows is positive and a substantial fraction of the planted truth.

    At this lean scale the *aggregate* outcome Delta_order need not clear its clustered CI
    (the effect lives in the rare large-velo-transition rows, diluting the whole-set delta);
    the honest verdict is GRU_MECHANISM_DIRECTIONAL and certification of the magnitude is a
    Phase-2 full-data question (documented -- mirrors WS4/WS5's lean-scale honesty)."""
    fal = positive_result["falsification"]
    assert fal["recovered_sign_ok"] is True
    assert fal["recovered_whiff_lift"] > 0
    assert fal["recovery_ratio"] > 0.3  # a substantial fraction of the planted lift
    assert fal["d47_verdict"] in ("GRU_MECHANISM_RECOVERED", "GRU_MECHANISM_DIRECTIONAL")


def test_positive_O_not_worse_than_L1_on_outcome(positive_result):
    """The ordered O outcome model is not materially worse than its L1 twin (the planted
    effect keys on |velo_{t-1} - velo_{t-2}|, which only O can represent). A small tolerance
    keeps this robust to torch-build float variation at lean scale."""
    fal = positive_result["falsification"]
    assert fal["outcome_losses"]["O"] <= fal["outcome_losses"]["L1"] + 5e-3


# --- factory-driven falsification plumbing (cheap smoke) -------------------------------

def test_order_ablation_and_permutation_smoke():
    """The factory-driven order ablation + token-order permutation run and return sane shapes."""
    from pitchseq.decision_table import build_decision_table
    from pitchseq.synth import make_null_world
    from workstreams.ws6_deep_seq.model import (
        temporal_split_masks, ws6_order_ablation, ws6_permutation_test,
    )
    raw, _ = make_null_world(n_games=16, seed=7, innings_per_game=6)
    tab = build_decision_table(raw)
    m = temporal_split_masks(tab)
    p = dict(hidden=16, embed_dim=8, max_epochs=3, max_len=6, seed=0, device="cpu", batch_size=256)
    ab = ws6_order_ablation(tab, "selection", p, m["train"], m["val"], views=("C", "U", "L1", "O"))
    assert set(ab["losses"]) == {"C", "U", "L1", "O"}
    assert np.isfinite(list(ab["losses"].values())).all()
    perm = ws6_permutation_test(tab, "outcome1", p, n_permutations=2, seed=0,
                                train_mask=m["train"], eval_mask=m["val"])
    assert 0.0 < perm["p_value"] <= 1.0
    assert len(perm["null_edges"]) == 2
