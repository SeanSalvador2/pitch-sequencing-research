"""WS5 tabular-MDP pipeline tests (SPEC 12.6; decisions D40-D43, D24, D39).

The MDP ladder is exercised end-to-end on the shared synthetic worlds and evaluated **only**
through the OPE gate (:mod:`pitchseq.eval.ope`). Pinned robustly: the behavior-recovery gate
passes *first* (D37 / SPEC 0.3); the D43 state-ladder / D42 cross-check blocks are complete with
**real** CIs (the FQE CI comes from the refit cluster bootstrap -- the per-episode contributions
are constant in this constant-initial-state domain, so a resampling bootstrap of them would be
structurally degenerate); ``alpha=0`` recovers behavior; ESS falls as ``alpha`` rises; degenerate
CIs are flagged, never rendered as fake intervals.

The verdicts (the honest, deterministic ones at test scale): the **null** world reports
``SEQ_NEUTRAL_MDP`` (no alpha satisfies the triple gate). The **positive** world reports the
D39-first-class ``SETUP_INCONCLUSIVE``: the held-out FQE trigger-count gap point is positive and
larger than the null world's (the ladder *discriminates* the worlds), and the in-sample
model-based gap and setup diagnostics agree directionally, but the refit-bootstrap lower-95
cannot clear the +0.003 ceiling at synthetic scale -- certification is a data-scale question for
Phase 2. The unit-level proof that the machinery *can* cash a setup is
``test_ws5_model.test_setup_representable_trigger_design_sets_up_and_wins`` (a constructed world
where the trigger-design policy provably sets up and wins); the pipeline test here asserts the
verdict machinery itself (the SETUP_EXPLOITED path fires on an engineered clear -- see
``test_verdict_triple_gate_fires_on_engineered_clear``).

Sizes / seeds are deterministic; a trimmed ``alpha in {0, 1}`` grid and low bootstrap counts keep
the suite fast without changing any point estimate.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from workstreams.ws5_tabular_mdp.run_ws5 import (
    _cluster_boot,
    _d42_agreement,
    _fmt_ci,
    _ladder_verdict,
    _refit_ci,
    run_ws5,
)
from workstreams.ws5_tabular_mdp.model import STATE_DESIGNS, RICHEST_DESIGN

# fqe_boot=12 keeps the refit bootstrap real (deterministic, non-degenerate CIs) while holding
# the suite in budget; the demo runs use 150-200 replicates for resolution. Point estimates and
# verdicts do not depend on the replicate count.
_COMMON = dict(alphas=[0.0, 1.0], n_boot=40, fqe_boot=12, sim_episodes=1000, seed=7,
               n_games=1600, alpha_r=12.0, write_outputs=False)


@pytest.fixture(scope="module")
def null_result():
    return run_ws5(synth="null", out=tempfile.mkdtemp(prefix="ws5_null_"), **_COMMON)


@pytest.fixture(scope="module")
def positive_result():
    return run_ws5(synth="positive", out=tempfile.mkdtemp(prefix="ws5_pos_"), **_COMMON)


# --- the OPE gate (SPEC 0.3 / D37) -----------------------------------------------------

def test_behavior_recovery_gate_passes_first(null_result, positive_result):
    for r in (null_result, positive_result):
        assert r["gate"] == "PASSED"
        rec = r["behavior_recovery"]
        assert rec["passed"] is True
        assert rec["ips_weights_unit"] is True  # pi == mu -> importance weights exactly 1


# --- D43 ladder + D42 cross-check complete, with REAL CIs -------------------------------

def test_ladder_and_d42_blocks_complete(null_result):
    """Every (design, alpha) cell carries model-based + both OPE estimators (real CIs) + the
    per-alpha like-for-like D42 agreement verdict."""
    for d in STATE_DESIGNS:
        for a in null_result["alphas"]:
            cell = null_result["ladder"][d][a]
            assert "model_based" in cell
            for est in ("stepwise_DR", "FQE"):
                assert cell[est]["value"] is not None
                assert "lower_95" in cell[est] and "ci95" in cell[est]
                assert "degenerate" in cell[est]
            # The FQE CI is the refit bootstrap -- real (non-degenerate), finite bounds.
            fq = cell["FQE"]
            assert fq["method"] == "fqe_refit_cluster_bootstrap"
            assert fq["degenerate"] is False
            lo, hi = fq["ci95"]
            assert np.isfinite(lo) and np.isfinite(hi) and lo < hi  # a REAL interval
            assert cell["d42"]["verdict"] in {"CONSISTENT", "DIVERGES", "n/a"}
        ss = null_result["state_summary"][d]
        assert "model_based_greedy" in ss and "simulated_greedy" in ss


def test_simulator_matches_model_based_greedy(positive_result):
    """D42: the Monte-Carlo simulator reproduces the analytic model-based greedy value per design."""
    for d in STATE_DESIGNS:
        ss = positive_result["state_summary"][d]
        assert ss["simulated_greedy"] == pytest.approx(ss["model_based_greedy"], abs=0.02)


def test_alpha_zero_recovers_behavior(null_result, positive_result):
    """pi_0 == mu: full ESS and a ~zero trigger-count gap (both designs evaluate behavior)."""
    for r in (null_result, positive_result):
        for d in STATE_DESIGNS:
            assert r["ladder"][d][0.0]["ess_frac"] == pytest.approx(1.0, abs=1e-6)
        g0 = r["gaps"][0.0]["trigger_minus_count"]["FQE"]["value"]
        assert abs(g0) < 5e-3


def test_ess_falls_as_alpha_rises(null_result, positive_result):
    for r in (null_result, positive_result):
        for d in STATE_DESIGNS:
            ess = [r["ladder"][d][a]["ess_frac"] for a in sorted(r["alphas"])]
            assert ess[-1] < ess[0]  # deviating from behavior costs overlap


# --- the greedy-optimism exhibit (separate from the D42 verdict) ------------------------

def test_greedy_optimism_exhibit(positive_result):
    """The in-sample greedy model-based value is optimistic vs its own held-out FQE value,
    sharper on the richest (sparsest) design -- the labeled exhibit, not a verdict input."""
    opt = {}
    for d in STATE_DESIGNS:
        mb = positive_result["state_summary"][d]["model_based_greedy"]
        fqe = positive_result["ladder"][d][1.0]["FQE"]["value"]  # alpha=1 IS the greedy target
        opt[d] = mb - fqe
        assert opt[d] > 0  # in-sample optimism everywhere
    assert opt[RICHEST_DESIGN] > opt["count"]  # sharper on the sparser design


# --- degenerate-CI honesty ---------------------------------------------------------------

def test_cluster_boot_flags_constant_contributions():
    """A constant contribution array is structurally degenerate: flagged, CI = NaN, and the
    formatter renders 'n/a (constant contributions)' rather than a fake interval."""
    out = _cluster_boot(np.full(50, 0.123), np.arange(50).astype(str), n_boot=20, seed=0)
    assert out["degenerate"] is True
    assert not np.isfinite(out["ci95"][0]) and not np.isfinite(out["lower_95"])
    assert _fmt_ci(out) == "n/a (constant contributions)"
    # A varying array is NOT degenerate and yields a real interval.
    rng = np.random.default_rng(0)
    ok = _cluster_boot(rng.normal(size=200), np.repeat(np.arange(20), 10).astype(str),
                       n_boot=50, seed=0)
    assert ok["degenerate"] is False
    assert ok["ci95"][0] < ok["value"] < ok["ci95"][1]


def test_refit_ci_degenerate_and_real():
    """_refit_ci: empty/NaN replicates -> degenerate n/a; varying replicates -> percentiles."""
    empty = _refit_ci(0.5, np.full(10, np.nan))
    assert empty["degenerate"] is True and _fmt_ci(empty) == "n/a (constant contributions)"
    rng = np.random.default_rng(1)
    reps = 0.01 * rng.standard_normal(200) + 0.05
    real = _refit_ci(0.05, reps)
    assert real["degenerate"] is False
    assert real["ci95"][0] < 0.05 < real["ci95"][1]
    assert real["lower_95"] == pytest.approx(np.percentile(reps, 5.0))


def test_d42_agreement_rule():
    """The like-for-like D42 rule: diverge iff |diff| > max half-width of the REAL CIs."""
    sw_ok = {"value": 0.05, "ci95": [0.03, 0.07], "degenerate": False}    # halfwidth 0.02
    fqe_ok = {"value": 0.055, "ci95": [0.045, 0.065], "degenerate": False}  # halfwidth 0.01
    agree = _d42_agreement(0.06, sw_ok, fqe_ok)   # MB within both tolerances
    assert agree["verdict"] == "CONSISTENT"
    diverge = _d42_agreement(0.12, sw_ok, fqe_ok)  # MB far outside both
    assert diverge["verdict"] == "DIVERGES"
    # Degenerate CIs cannot be assessed -> pair marked None, not fabricated.
    fqe_bad = {"value": 0.055, "ci95": [float("nan")] * 2, "degenerate": True}
    partial = _d42_agreement(0.06, sw_ok, fqe_bad)
    assert partial["pairs"]["model_based|FQE"]["diverges"] is None


# --- the triple gate: the SETUP_EXPLOITED path is reachable ------------------------------

def test_verdict_triple_gate_fires_on_engineered_clear():
    """On an engineered gap whose refit-FQE lower-95 clears the ceiling with stepDR and MB
    directionally positive, the verdict machinery fires SETUP_EXPLOITED -- proving the shipped
    SETUP_INCONCLUSIVE is a statement about the data scale, not a blind alley in the code."""
    gaps = {1.0: {"trigger_minus_count": {
        "FQE": {"value": 0.010, "lower_95": 0.006, "ci95": [0.005, 0.015], "degenerate": False},
        "stepwise_DR": {"value": 0.008, "lower_95": -0.05, "ci95": [-0.06, 0.08], "degenerate": False},
        "model_based": 0.02,
    }}}
    v = _ladder_verdict("positive", gaps, ceiling=0.003, setup_diag={"setup_gap_mean": 0.01})
    assert v["verdict"] == "SETUP_EXPLOITED"
    assert v["fired_alphas"] == [1.0]
    # Any one leg failing (FQE lower-95, stepDR sign, MB sign) blocks the verdict.
    for broken in (
        {"FQE": {"value": 0.010, "lower_95": 0.001, "ci95": [0.0, 0.015], "degenerate": False}},
        {"stepwise_DR": {"value": -0.001, "lower_95": -0.05, "ci95": [-0.06, 0.08], "degenerate": False}},
        {"model_based": -0.01},
    ):
        g = {1.0: {"trigger_minus_count": {**gaps[1.0]["trigger_minus_count"], **broken}}}
        v = _ladder_verdict("positive", g, ceiling=0.003, setup_diag=None)
        assert v["verdict"] == "SETUP_INCONCLUSIVE"


# --- verdicts: the worlds are discriminated ----------------------------------------------

def test_null_world_seq_neutral(null_result):
    """Null world: no alpha satisfies the triple gate -> SEQ_NEUTRAL_MDP."""
    assert null_result["verdict"]["verdict"] == "SEQ_NEUTRAL_MDP"
    for a in null_result["alphas"]:
        g = null_result["gaps"][a]["trigger_minus_count"]
        gate = (not g["FQE"]["degenerate"]) and np.isfinite(g["FQE"]["lower_95"]) \
            and g["FQE"]["lower_95"] > null_result["ceiling"] \
            and g["stepwise_DR"]["value"] > 0 and g["model_based"] > 0
        assert not gate


def test_positive_world_honest_verdict_and_discrimination(null_result, positive_result):
    """Positive world at synthetic scale: the honest deterministic verdict is the D39
    first-class SETUP_INCONCLUSIVE -- the held-out FQE trigger-count gap point is positive and
    clearly larger than the null world's (the ladder discriminates), the in-sample model-based
    gap agrees, but the refit-bootstrap lower-95 cannot clear the +0.003 ceiling at this scale
    (a data-scale question deferred to Phase 2)."""
    assert positive_result["verdict"]["verdict"] == "SETUP_INCONCLUSIVE"
    top_a = max(positive_result["alphas"])
    pos_gap = positive_result["gaps"][top_a]["trigger_minus_count"]
    null_gap = null_result["gaps"][top_a]["trigger_minus_count"]
    # Discrimination: the positive world's held-out FQE gap point exceeds the null world's.
    assert pos_gap["FQE"]["value"] > null_gap["FQE"]["value"]
    assert pos_gap["FQE"]["value"] > 0
    # The in-sample model-based gap agrees directionally (the setup is present in the MDP).
    assert pos_gap["model_based"] > 0
    # The evidence block is populated for the headline story.
    ev = positive_result["verdict"]["evidence"]
    assert "fqe_value" in ev and "stepdr_value" in ev and "model_based" in ev


def test_setup_diagnostics_discriminate(null_result, positive_result):
    """The setup Q-gap (create vs not-create a trigger) is larger in the positive world."""
    pos = positive_result["setup_diagnostics"]["setup_gap_mean"]
    null = null_result["setup_diagnostics"]["setup_gap_mean"]
    assert pos > null
    assert pos > 0.0


def test_gap_reported_with_refit_ci(positive_result):
    """Both gaps carry the refit-bootstrap FQE CI, the stepwise-DR contribution CI and the
    model-based gap; the FQE gap CI is real (finite, non-degenerate) at every alpha."""
    for a in positive_result["alphas"]:
        for key in ("trigger_minus_count", "prev_minus_count"):
            g = positive_result["gaps"][a][key]
            assert "FQE" in g and "stepwise_DR" in g and "model_based" in g
            fq = g["FQE"]
            assert fq["method"] == "fqe_refit_cluster_bootstrap"
            assert fq["degenerate"] is False
            lo, hi = fq["ci95"]
            assert np.isfinite(lo) and np.isfinite(hi) and lo < hi
            assert lo <= fq["value"] <= hi or abs(fq["value"] - np.clip(fq["value"], lo, hi)) < 5e-3
