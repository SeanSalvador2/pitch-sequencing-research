"""WS4 bandit model unit tests (SPEC 12.5; decisions D36-D38).

Pins the model-level contracts on small, hand-checkable, deterministic inputs: the Thompson
target policy (dominant mean -> near point mass; equal means -> uniform over feasible;
infeasible actions get zero mass; empty mask -> observed-action fallback; deterministic
under a seed; rows sum to 1); the ambiguity decomposition and deviation map on tiny
constructed cases; the soften delegation to eval/ope.pi_alpha; and build_bandit_inputs'
shape / finiteness / behaviour-floor contract against a tiny WS3 fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from workstreams.ws4_bandit import model as M


# --- thompson_policy -------------------------------------------------------------------

def test_thompson_dominant_q_near_point_mass():
    """One feasible action far above the rest (relative to sd) -> near point mass."""
    q = np.array([[10.0, 0.0, 0.0]])
    q_sd = np.array([[0.1, 0.1, 0.1]])
    feas = np.array([[True, True, True]])
    p = M.thompson_policy(q, q_sd, feas, n_samples=2000, rng=0)
    assert p.shape == (1, 3)
    assert p[0, 0] > 0.999
    assert np.isclose(p.sum(), 1.0)


def test_thompson_equal_q_equal_sd_uniform_over_feasible():
    """Equal means and sds -> uniform over the feasible actions (MC, tolerant)."""
    q = np.array([[1.0, 1.0, 1.0, 1.0]])
    q_sd = np.array([[1.0, 1.0, 1.0, 1.0]])
    feas = np.array([[True, True, True, True]])
    p = M.thompson_policy(q, q_sd, feas, n_samples=8000, rng=123)
    assert np.allclose(p[0], 0.25, atol=0.03)
    assert np.isclose(p.sum(), 1.0)


def test_thompson_infeasible_actions_get_zero_mass():
    q = np.array([[3.0, 5.0, 4.0, 9.0]])          # action 3 has the highest mean ...
    q_sd = np.array([[0.5, 0.5, 0.5, 0.5]])
    feas = np.array([[True, True, False, False]])  # ... but 3 (and 2) are infeasible
    p = M.thompson_policy(q, q_sd, feas, n_samples=2000, rng=1)
    assert p[0, 2] == 0.0 and p[0, 3] == 0.0
    assert np.isclose(p[0, :2].sum(), 1.0)


def test_thompson_empty_mask_falls_back_to_observed_action():
    """A low-history row with no feasible action -> observed-action point mass (no rec)."""
    q = np.array([[1.0, 2.0, 3.0]])
    q_sd = np.array([[1.0, 1.0, 1.0]])
    feas = np.array([[False, False, False]])
    p = M.thompson_policy(q, q_sd, feas, n_samples=500, rng=0, observed_action=np.array([1]))
    assert np.array_equal(p[0], [0.0, 1.0, 0.0])


def test_thompson_deterministic_under_seed():
    q = np.array([[0.1, 0.2, 0.15], [0.3, 0.1, 0.2]])
    q_sd = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    feas = np.array([[True, True, True], [True, True, False]])
    a = M.thompson_policy(q, q_sd, feas, n_samples=1000, rng=42)
    b = M.thompson_policy(q, q_sd, feas, n_samples=1000, rng=42)
    assert np.array_equal(a, b)


def test_thompson_rows_sum_to_one_and_nonneg():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(20, 8))
    q_sd = np.abs(rng.normal(size=(20, 8))) + 0.01
    feas = rng.random((20, 8)) > 0.4
    feas[:, M.XX_INDEX] = False  # XX never feasible (SPEC 4)
    obs = rng.integers(0, 8, size=20)
    p = M.thompson_policy(q, q_sd, feas, n_samples=500, rng=7, observed_action=obs)
    assert np.all(p >= 0)
    assert np.allclose(p.sum(axis=1), 1.0)
    for i in range(len(p)):
        if feas[i].any():
            assert np.all(p[i][~feas[i]] == 0.0)          # zero mass on every infeasible action
        else:
            assert p[i, obs[i]] == 1.0                    # empty mask -> observed-action fallback


def test_thompson_zero_sd_is_exact_argmax():
    """Degenerate (zero) uncertainty -> the exact feasible argmax point mass."""
    q = np.array([[0.2, 0.5, 0.4]])
    q_sd = np.zeros((1, 3))
    feas = np.array([[True, True, False]])
    p = M.thompson_policy(q, q_sd, feas, n_samples=64, rng=0)
    assert np.array_equal(p[0], [0.0, 1.0, 0.0])  # action 1 is the feasible argmax


def test_thompson_batching_matches_unbatched():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(50, 5))
    q_sd = np.abs(rng.normal(size=(50, 5))) + 0.05
    feas = rng.random((50, 5)) > 0.3
    a = M.thompson_policy(q, q_sd, feas, n_samples=400, rng=9, batch_rows=7)
    b = M.thompson_policy(q, q_sd, feas, n_samples=400, rng=9, batch_rows=10_000)
    assert np.array_equal(a, b)


# --- soften ----------------------------------------------------------------------------

def test_soften_delegates_to_pi_alpha_endpoints():
    mu = np.array([[0.5, 0.3, 0.2]])
    tgt = np.array([[0.1, 0.1, 0.8]])
    assert np.allclose(M.soften(tgt, mu, 0.0), mu)     # alpha=0 -> behavior
    assert np.allclose(M.soften(tgt, mu, 1.0), tgt)    # alpha=1 -> target
    assert np.allclose(M.soften(tgt, mu, 0.25), 0.75 * mu + 0.25 * tgt)


# --- ambiguity_stats -------------------------------------------------------------------

def test_ambiguity_stats_hand_computed():
    """p_beat = Phi((q_top-q_ru)/sqrt(sd_top^2+sd_ru^2)); shares over decidable rows."""
    q = np.array([[0.10, 0.06, 0.0],    # gap 0.04, pooled sd sqrt(2)*0.02
                  [0.10, 0.02, 0.0]])   # gap 0.08, pooled sd sqrt(2)*0.01
    q_sd = np.array([[0.02, 0.02, 0.02],
                     [0.01, 0.01, 0.01]])
    feas = np.array([[True, True, False],
                     [True, True, False]])
    s = M.ambiguity_stats(q, q_sd, feas)
    p_a = float(ndtr(0.04 / np.sqrt(2 * 0.02 ** 2)))   # ~0.9214
    p_b = float(ndtr(0.08 / np.sqrt(2 * 0.01 ** 2)))   # ~1.0
    assert s["n_decidable"] == 2
    assert np.isclose(s["mean_p_beat"], (p_a + p_b) / 2)
    assert np.isclose(s["mean_gap"], 0.06)
    assert np.isclose(s["ambiguous_95"], 0.5)   # only row A is below 0.95
    assert np.isclose(s["ambiguous_80"], 0.0)
    assert np.isclose(s["ambiguous_50"], 0.0)   # p_beat >= 0.5 always


def test_ambiguity_stats_counts_singleton_and_infeasible():
    q = np.array([[0.1, 0.05, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    q_sd = np.full((3, 3), 0.02)
    feas = np.array([[True, True, False],    # decidable (2 feasible)
                     [True, False, False],   # singleton (1 feasible)
                     [False, False, False]])  # infeasible (0)
    s = M.ambiguity_stats(q, q_sd, feas)
    assert s["n_decidable"] == 1
    assert s["n_singleton"] == 1
    assert s["n_infeasible"] == 1
    assert s["n_rows"] == 3


# --- deviation_map ---------------------------------------------------------------------

def test_deviation_map_hand_computed():
    """Mean TV per count cell on a tiny constructed case."""
    target = np.array([[0.7, 0.3], [0.9, 0.1], [0.5, 0.5]])
    mu = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    # per-row TV = 0.2, 0.4, 0.0
    table = pd.DataFrame({"balls": [0, 0, 1], "strikes": [0, 0, 0]})
    dev = M.deviation_map(target, mu, table, by=("balls", "strikes"))
    cell00 = dev[(dev.balls == 0) & (dev.strikes == 0)].iloc[0]
    cell10 = dev[(dev.balls == 1) & (dev.strikes == 0)].iloc[0]
    assert np.isclose(cell00["mean_tv"], 0.3) and cell00["n"] == 2   # (0.2+0.4)/2
    assert np.isclose(cell10["mean_tv"], 0.0) and cell10["n"] == 1


# --- build_bandit_inputs (D33 consumption) ---------------------------------------------

@pytest.fixture(scope="module")
def tiny_ws3(tmp_path_factory):
    """A tiny WS3 fit (C + O) on a small null world, persisted for the consumption test."""
    from pitchseq.decision_table import build_decision_table
    from pitchseq.synth import make_null_world
    from workstreams.ws3_gbdt_stack.model import (
        BehaviorModel, OutcomeStack, save_ws3_artifacts,
    )

    raw, _ = make_null_world(n_games=25, seed=7, innings_per_game=6)
    table = build_decision_table(raw)
    params = {"n_estimators": 40, "num_leaves": 15, "min_child_samples": 20}
    behavior, outcome = {}, {}
    for v in ("C", "O"):
        behavior[v] = BehaviorModel(v).fit(table, params=params)
        outcome[v] = OutcomeStack(v).fit(table, params=params)
    d = tmp_path_factory.mktemp("ws3")
    save_ws3_artifacts(d, behavior=behavior, outcome=outcome)
    return str(d), table


def test_build_bandit_inputs_shapes_and_floor(tiny_ws3):
    ws3_dir, table = tiny_ws3
    bi = M.build_bandit_inputs(table, ws3_dir, "O")
    n = len(table)
    assert bi.q.shape == (n, 8) and bi.q_sd.shape == (n, 8) and bi.mu.shape == (n, 8)
    assert np.isfinite(bi.q).all() and np.isfinite(bi.q_sd).all()
    assert (bi.q_sd >= 0).all()
    # mu floored + renormalised: strictly positive, sums to 1.
    assert (bi.mu >= M.MU_FLOOR - 1e-12).all()
    assert np.allclose(bi.mu.sum(axis=1), 1.0)
    # feasible mask: XX never feasible; bool; aligned.
    assert bi.feasible_mask.dtype == bool
    assert not bi.feasible_mask[:, M.XX_INDEX].any()
    assert set(bi.align.columns) == {"row_id", "observed_action", "low_history", "any_feasible"}
    assert len(bi.align) == n
    # q_sd is the residual sd scaled by POSTERIOR_SCALE (unpack as a 5-tuple too).
    q, q_sd, mu, feas, align = bi
    assert q_sd.shape == q.shape


def test_build_bandit_inputs_accepts_loaded_artifacts(tiny_ws3):
    from workstreams.ws3_gbdt_stack.model import load_ws3_artifacts
    ws3_dir, table = tiny_ws3
    art = load_ws3_artifacts(ws3_dir)
    bi = M.build_bandit_inputs(table, art, "C", posterior_scale=0.1)
    assert bi.q.shape == (len(table), 8)
    # posterior_scale override scales the sd: q_sd = 0.1 * residual sd.
    bi_default = M.build_bandit_inputs(table, art, "C")
    # both finite; the override is a pure scalar multiple of the residual sd
    assert np.isfinite(bi.q_sd).all()
    ratio = bi.q_sd[bi_default.q_sd > 0] / bi_default.q_sd[bi_default.q_sd > 0]
    assert np.allclose(ratio, 0.1 / M.POSTERIOR_SCALE)
