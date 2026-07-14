"""WS2 end-to-end pipeline tests (SPEC 12.2; decisions D29-D31).

The grammar is exercised on the shared synthetic null world -- WS2's **positive control**
(decision D30: it plants an order-2 no-three-in-a-row selection habit) -- and scored only
through the shared harness. The tests check that the grammar is detected (O beats L1 on
selection log loss, effective order >= 2 mass is material, the repeat-suppression motif is
present and correctly signed), that a stratified history permutation collapses the ordered
edge, the effective order and ``B_seq``, that ``B_seq`` is positive and hand-computable, and
that the standard-schema predictions are valid and row_id-aligned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.decision_table import build_decision_table
from pitchseq.eval.predictability import bits_of_predictability
from pitchseq.eval.predictions import ACTION_PROB_COLS, validate_predictions
from pitchseq.families import FAMILIES
from pitchseq.synth import make_null_world
from workstreams.ws2_bayes_markov import model as W
from workstreams.ws2_bayes_markov.run_ws2 import run_ws2

# Deterministic world size / seed / bootstrap locked so the assertions are reproducible.
_NULL = dict(synth="null", n_games=200, seed=7, n_boot=60)


@pytest.fixture(scope="module")
def null_result():
    return run_ws2(write_outputs=False, **_NULL)


# --- D30 detection: the null world is WS2's positive control ----------------------------

def test_null_world_grammar_detected(null_result):
    """The planted order-2 habit is detected: O < L1 loss, order-2 mass, repeat motif (D30)."""
    v = null_result["verdicts"]
    assert v["grammar_detected"] is True
    assert v["grammar_detected_verdict"] == "GRAMMAR_DETECTED"
    comps = v["detected_components"]
    assert comps["o_beats_l1"] is True
    assert comps["order_ge2_mass"] > 0.05  # effective order >= 2 mass is material
    assert comps["repeat_suppression_motif"] is True


def test_null_world_O_beats_L1_on_selection_log_loss(null_result):
    """The full ordered grammar has lower selection log loss than the previous-pitch view."""
    sel = null_result["selection"]
    assert sel["O"]["log_loss"] < sel["L1"]["log_loss"]
    assert sel["L1"]["log_loss"] <= sel["C"]["log_loss"] + 1e-9  # order-1 adds nothing here, ties C


def test_null_world_delta_order_L1_significantly_positive(null_result):
    """``delta_order_L1 = Loss(L1) - Loss(O)`` is positive with a CI lower bound above 0."""
    do = null_result["delta_order_L1"]
    assert do["point"] > 0
    assert do["ci"]["lo"] > 0  # read directly (no min-bias, unlike SPEC 6 Delta_order)
    assert do["significantly_positive"] is True


def test_null_world_repeat_suppression_motif_present_and_signed(null_result):
    """A same-family-run motif tops the grammar rules and suppresses the repeated family."""
    motifs = null_result["top_motifs"]
    runs = [m for m in motifs if m["depth"] >= 2 and len(set(m["context"])) == 1]
    assert runs, "expected a same-family-run motif"
    m = runs[0]
    assert m["direction"] == "suppress"
    assert m["top_family"] == m["context"][-1]
    assert m["p_ctx"] < m["p_suffix"]


def test_null_world_effective_order_distribution(null_result):
    """The effective-order distribution puts material mass at depth 2 (the planted order)."""
    eo = null_result["grammar_exhibits"]["effective_order"]
    dist = eo["distribution"]
    assert dist[2] > 0.05
    assert eo["frac_ge2"] > 0.05
    assert eo["mean"] > 0.0


# --- D30 negative control: permutation collapses the grammar ---------------------------

def test_null_world_collapses_under_permutation(null_result):
    """A stratified history permutation collapses the ordered edge and the effective order."""
    v = null_result["verdicts"]
    assert v["collapses_under_permutation"] is True
    assert v["collapse_verdict"] == "COLLAPSES_UNDER_PERMUTATION"
    cc = v["collapse_components"]
    assert cc["perm_edge"] < 0.5 * cc["real_edge"]  # edge collapses
    assert cc["perm_order_ge2_mass"] < 0.03          # effective order concentrates at <= 1


def test_null_world_permutation_effective_order_at_most_one(null_result):
    """After permutation the effective order concentrates at <= 1 (no earned depth-2)."""
    perm = null_result["permutation_control"]
    assert perm["effective_order"]["frac_ge2"] < 0.03


# --- B_seq: bits of predictability (D31) -----------------------------------------------

def test_null_world_B_seq_positive_and_collapses(null_result):
    """``B_seq`` is positive on the null world and collapses toward 0 after permutation."""
    bits = null_result["bits_of_predictability"]
    assert bits["overall"] > 0.0  # ordered history IS predictive of selection here
    # Deeper counts (more history) carry more bits than the overall average.
    assert bits["three_ball"] > bits["overall"]
    perm_b = null_result["permutation_control"]["b_seq"]
    assert abs(perm_b) < 0.5 * bits["overall"]  # collapses toward 0


def test_bits_of_predictability_hand_computed():
    """``B_seq = mean log2(q_O(a)/q_C(a))`` on a tiny constructed case."""
    # Two rows; O concentrates on the realised next family, C is uniform-ish.
    q_o = np.array([[0.8, 0.2 / 7, 0.2 / 7, 0.2 / 7, 0.2 / 7, 0.2 / 7, 0.2 / 7, 0.2 / 7],
                    [0.1, 0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10]])
    q_c = np.array([[0.25, 0.25, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05],
                    [0.25, 0.25, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05]])
    actions = np.array([FAMILIES[0], FAMILIES[1]])  # realised next families
    bits = bits_of_predictability(q_o, q_c, actions)
    assert np.isclose(bits[0], np.log2(0.8 / 0.25))
    assert np.isclose(bits[1], np.log2(0.6 / 0.25))
    assert bits.mean() > 0  # O forecasts the realised pitch better than C


def test_ws2_bits_positive_on_constructed_order_world():
    """On a handcrafted order-2 world the O grammar earns positive bits over the C base."""
    rows = []
    fams = ["FF", "SL", "CH", "CU"]
    rng = np.random.default_rng(5)
    recs = []
    for i in range(400):
        a = fams[rng.integers(4)]
        b = a if rng.random() < 0.5 else fams[rng.integers(4)]
        third = (a if rng.random() < 0.05 else fams[rng.integers(4)]) if a == b else fams[rng.integers(4)]
        for j, f in enumerate([a, b, third]):
            recs.append(dict(game_pk=1, at_bat_number=i + 1, pitch_number=j + 1, family=f,
                             pitcher=1, balls=0, strikes=0, R=0.0))
    df = pd.DataFrame(recs)
    df["family"] = pd.Categorical(df["family"], categories=list(FAMILIES))
    df["pa_id"] = df["game_pk"] * 1000 + df["at_bat_number"]
    df["row_id"] = df["game_pk"].astype(str) + "_" + df["at_bat_number"].astype(str) + "_" + df["pitch_number"].astype(str)
    for c, d in (("stand", "R"), ("p_throws", "R"), ("batter", 2)):
        df[c] = d
    q_o = W.fit(df, "O").predict_proba(df)
    q_c = W.fit(df, "C").predict_proba(df)
    y = df["family"].astype("object").to_numpy()
    assert bits_of_predictability(q_o, q_c, y).mean() > 0.0


# --- standard-schema validity ----------------------------------------------------------

def test_predictions_validate_and_row_id_aligned():
    """Per-view WS2 predictions conform to SPEC 8.1 and share the eval rows' row_id order."""
    raw, _ = make_null_world(n_games=20, seed=7, innings_per_game=6)
    table = build_decision_table(raw)
    proba = W.fit(table, "O").predict_proba(table)
    df = pd.DataFrame({"row_id": table["row_id"].to_numpy()})
    for j, col in enumerate(ACTION_PROB_COLS):
        df[col] = proba[:, j]
    df["model_id"] = "ws2_bayes_markov"
    df["state_view"] = "O"
    df["seconds"] = 0.1
    df["peak_mem_mb"] = 1.0
    df["n_params"] = 100
    assert validate_predictions(df) is True
    assert list(df["row_id"]) == list(table["row_id"])  # row_id-aligned to the eval table


def test_pipeline_writes_valid_artifacts(tmp_path):
    """A full run writes per-view predictions, a report, a motifs CSV and run metadata."""
    from pitchseq.eval.predictions import load_predictions

    result = run_ws2(synth="null", n_games=60, seed=7, n_boot=25, out=str(tmp_path), write_outputs=True)
    assert (tmp_path / "ws2_report_null.json").is_file()
    assert (tmp_path / "motifs_null.csv").is_file()
    assert (tmp_path / "ws2_null.runmeta.json").is_file()
    for view in result["views"]:
        pred_path = tmp_path / f"predictions_null_{view}.parquet"
        assert pred_path.is_file()
        loaded = load_predictions(pred_path)
        assert validate_predictions(loaded) is True
        assert all(c in loaded.columns for c in ACTION_PROB_COLS)


def test_not_applicable_views_reported(null_result):
    """The report documents U / OM as not-applicable for a Markov grammar (D29)."""
    na = null_result["not_applicable"]
    assert set(na) == {"U", "OM"}
    assert na["U"]["applicable"] is False and na["OM"]["applicable"] is False
