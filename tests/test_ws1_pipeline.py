"""WS1 end-to-end pipeline tests (SPEC 12.1; decisions D21, D25-D28).

The pipeline is exercised on the shared synthetic worlds and scored **only** through the
shared harness. The null world checks that WS1's flat tables do not manufacture an ordered
selection edge and that the serious hierarchical version does not lose to the quick
count-based references; the positive world checks that the ordered (O) run-value table
recovers the planted velo-transition effect in the right direction, with the heavy
attenuation decision D27 anticipates; the support diagnostics check the support explosion
that is the WS1 exhibit.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchseq.decision_table import build_decision_table
from pitchseq.eval.predictions import ACTION_PROB_COLS, validate_predictions
from pitchseq.families import FAMILIES
from pitchseq.synth import make_null_world
from workstreams.ws1_eb_tables import model as W
from workstreams.ws1_eb_tables.run_ws1 import run_ws1

# Deterministic world sizes / seeds locked so the assertions below are reproducible.
_NULL = dict(synth="null", n_games=90, seed=7, target="selection", n_boot=100)
_POS = dict(synth="positive", n_games=300, seed=7, target="run_value", n_boot=50)


@pytest.fixture(scope="module")
def null_result():
    return run_ws1(write_outputs=False, **_NULL)


@pytest.fixture(scope="module")
def positive_result():
    return run_ws1(write_outputs=False, **_POS)


# --- null world: honest selection behaviour + baseline comparison ----------------------

def test_null_pipeline_delta_order_not_significantly_positive(null_result):
    """Null world: WS1's selection ``Delta_order`` must not be significantly positive (D21).

    WS1's flat Dirichlet shrinkage pools the weak within-PA selection-order signal toward
    the pitcher x count cell (the maximum-marginal-likelihood concentration is at its
    ceiling), so ``Delta_order = Loss(min[U, L1]) - Loss(O)`` collapses to ~0. Per D21 the
    criterion is "not significantly positive" (the statistic is negatively biased under the
    null); a small negative value reads as consistent with no ordering effect.
    """
    do = null_result["delta_order"]
    assert do is not None
    assert do["ci"]["lo"] <= 0.0
    assert do["significantly_positive"] is False
    assert abs(do["point"]) < 0.03


def test_null_pipeline_ws1_not_worse_than_reference_baselines(null_result):
    """Null world: WS1 selection log loss <= its corresponding count-based reference.

    The history views (L1/U/O) are compared to ``pitcher_count_prev`` (the prev-family
    reference), which WS1 beats outright because its hierarchical shrinkage does not pay the
    quick baseline's fragmentation cost. The C view is compared to ``pitcher_count``; on the
    null world WS1-C also splits by batter handedness (uninformative here), so it is asserted
    within a small documented epsilon -- a tie within noise, per the task -- and would win on
    real data where handedness matters.
    """
    sel = null_result["selection"]
    bl = null_result["baseline_selection_loss"]
    for view in null_result["views"]:
        ll = sel[view]["log_loss"]
        ref = bl[null_result["reference_for"][view]]
        eps = 0.02 if view == "C" else 1e-9
        assert ll <= ref + eps, f"{view}: {ll:.4f} > ref {ref:.4f}"
    # WS1 beats the three references it is not merely tying (global / transition / prev).
    assert min(sel[v]["log_loss"] for v in null_result["views"]) < bl["pitcher_count_prev"]
    assert min(sel[v]["log_loss"] for v in null_result["views"]) < bl["transition"]


def test_null_pipeline_predictions_validate_through_harness():
    """A WS1 selection prediction table conforms to the SPEC 8.1 contract."""
    raw, _ = make_null_world(n_games=20, seed=7, innings_per_game=6)
    table = build_decision_table(raw)
    proba = W.fit(table, "L1", "selection").predict_proba(table)
    import pandas as pd

    df = pd.DataFrame({"row_id": table["row_id"].to_numpy()})
    for j, col in enumerate(ACTION_PROB_COLS):
        df[col] = proba[:, j]
    df["model_id"] = "ws1_eb_tables"
    df["state_view"] = "L1"
    df["seconds"] = 0.1
    df["peak_mem_mb"] = 1.0
    df["n_params"] = 10
    assert validate_predictions(df) is True


# --- positive world: ordered run-value recovery (D27) ----------------------------------

def test_positive_pipeline_ordered_run_value_edge_recovered(positive_result):
    """Positive world: the O run-value table recovers the planted velo-transition effect.

    The effect boosts whiffs (hence reward) when ``|velo_{t-1} - velo_{t-2}| >= 5`` mph,
    which the ordered family-pair key of the O table proxies (distant velo-band pairs). The
    edge is measured **within long plate appearances** (``pitch_number >= 3``, where the
    transition is defined) as the predicted-reward difference between triggered and
    non-triggered rows, minus the same difference for the no-history C view (which isolates
    the ordered contribution from count/family composition).

    Measured (n_games=300, seed=7): empirical reward lift ~ +0.016; O predicted edge
    ~ +0.0005 with ~ +0.0003 of it beyond C -> **attenuation ~ 0.03**. The heavy attenuation
    is expected and reported per D27 (flat tables barely resolve the small ordered outcome
    effect against reward noise); the test asserts only the direction and a modest floor.
    """
    rec = positive_result["positive_recovery"]
    assert rec and rec["true_lift"] > 0
    o = rec["per_view"]["O"]
    c = rec["per_view"]["C"]
    l1 = rec["per_view"]["L1"]
    # Direction: the O table predicts higher reward on triggered rows.
    assert o["edge"] > 0
    # The ordered contribution (beyond composition) is positive and above a modest floor,
    # and the ordered pair recovers more than the single previous family (L1).
    assert o["edge_vs_C"] > 1e-4
    assert o["edge_vs_C"] > l1["edge_vs_C"]
    # Attenuated, not exaggerated: recovered edge is a small fraction of the true lift.
    assert 0.0 < o["attenuation"] < 1.0
    assert c["edge_vs_C"] == pytest.approx(0.0, abs=1e-12)  # C-relative edge of C is 0


# --- support diagnostics: the WS1 exhibit ----------------------------------------------

def test_support_cells_explode_and_low_support_fraction_rises(null_result):
    """Support fragments as the key deepens: the WS1 support-problem exhibit.

    The number of distinct cells explodes from C to the history views (each history key
    multiplies the pitcher x count cells), and the fraction of eval rows landing in cells
    with fewer than 5 training pitches rises monotonically C -> L1 -> U -> O. (The distinct
    count of U vs O is world-dependent: U keys the full capped multiset of priors while O
    keys only the ordered last two, so on longer plate appearances U can fragment more than
    O -- reported honestly; the monotone low-support fraction is the robust support signal.)
    """
    sup = {s["view"]: s for s in null_result["support"]}
    n = {v: sup[v]["n_cells"] for v in ("C", "L1", "U", "O")}
    # Explosion: every history view fragments far beyond C, and beyond L1.
    assert n["C"] < n["L1"]
    assert n["L1"] < n["U"]
    assert n["L1"] < n["O"]
    assert n["U"] > 2 * n["C"] and n["O"] > 2 * n["C"]
    # Low-support fraction rises with view depth (robustly monotone C -> L1 -> U -> O).
    lt5 = [sup[v]["frac_eval_lt5"] for v in ("C", "L1", "U", "O")]
    assert all(lt5[i] < lt5[i + 1] for i in range(3)), lt5
    # Per-cell support collapses: the median C cell has many more pitches than a history cell.
    assert sup["C"]["cell_n_p50"] > sup["O"]["cell_n_p50"]


# --- output artefacts ------------------------------------------------------------------

def test_pipeline_writes_valid_artifacts(tmp_path):
    """A full run writes per-view predictions, a report, a support CSV and run metadata."""
    from pitchseq.eval.predictions import load_predictions

    result = run_ws1(
        synth="null", n_games=30, seed=7, target="both", n_boot=25, out=str(tmp_path),
        write_outputs=True,
    )
    assert (tmp_path / "ws1_report_null.json").is_file()
    assert (tmp_path / "support_null.csv").is_file()
    assert (tmp_path / "ws1_null.runmeta.json").is_file()
    for view in result["views"]:
        pred_path = tmp_path / f"predictions_null_{view}.parquet"
        assert pred_path.is_file()
        loaded = load_predictions(pred_path)
        assert validate_predictions(loaded) is True
        # Both prediction groups present for target="both".
        assert all(c in loaded.columns for c in ACTION_PROB_COLS)
        assert "exp_reward" in loaded.columns and "exp_reward_sd" in loaded.columns
