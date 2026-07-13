"""Off-policy evaluation on the logged-bandit fixture (SPEC 9 / 11).

Covers the SPEC 9 self-tests and the acceptance checks of the 0c unit: estimator formulas,
the conservative pi_alpha path, behavior-policy recovery, known-value recovery for both
canonical targets, the double-robustness demonstrations, the overlap diagnostics and the
DM/DR/SNIPS agreement verdict (CONSISTENT vs INCONCLUSIVE).
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchseq.eval import ope
from pitchseq.synth import make_logged_bandit

SEED = 20260713


@pytest.fixture(scope="module")
def bandit():
    """A logged-bandit fixture with its ground truth, built once for the module."""
    logged, truth = make_logged_bandit(n_rounds=8000, seed=SEED)
    return logged, truth


def _pack(logged, truth, target_name):
    """Return (target_probs, q_hat_rows, rewards, actions, mu_taken, w) for a named target."""
    states = logged["state"].to_numpy()
    target = truth.target_probs_for(states, target_name)
    q_row, _ = ope.outcome_model_means(logged, truth.n_actions)
    r = logged["reward"].to_numpy()
    a = logged["action"].to_numpy()
    mu = logged["mu_prob"].to_numpy()
    w = ope._weights(target, a, mu)
    return target, q_row, r, a, mu, w


# --------------------------------------------------------------------------------------
# estimator formulas (hand checks on the fixture arrays)
# --------------------------------------------------------------------------------------


def test_estimator_formulas_match_definitions(bandit):
    logged, truth = bandit
    target, q_row, r, a, mu, w = _pack(logged, truth, "greedy")

    v_dm, c_dm = ope.dm(q_row, target)
    assert v_dm == pytest.approx((target * q_row).sum(axis=1).mean())
    assert c_dm == pytest.approx((target * q_row).sum(axis=1))

    v_ips, c_ips = ope.ips(r, w)
    assert v_ips == pytest.approx((w * r).mean())
    assert c_ips == pytest.approx(w * r)

    v_snips, _ = ope.snips(r, w)
    assert v_snips == pytest.approx((w * r).sum() / w.sum())

    v_dr, c_dr = ope.dr(r, w, q_row, target, a)
    baseline = (target * q_row).sum(axis=1)
    q_taken = q_row[np.arange(len(a)), a]
    assert v_dr == pytest.approx((baseline + w * (r - q_taken)).mean())


def test_dr_reduces_to_ips_with_zero_q_and_to_dm_with_zero_w(bandit):
    logged, truth = bandit
    target, q_row, r, a, mu, w = _pack(logged, truth, "greedy")
    # q_hat == 0 -> baseline 0, q_taken 0 -> dr_i = w_i r_i == IPS.
    v_dr0, _ = ope.dr(r, w, np.zeros_like(q_row), target, a)
    v_ips, _ = ope.ips(r, w)
    assert v_dr0 == pytest.approx(v_ips)
    # w == 0 -> correction vanishes -> dr == DM.
    v_drw, _ = ope.dr(r, np.zeros_like(w), q_row, target, a)
    v_dm, _ = ope.dm(q_row, target)
    assert v_drw == pytest.approx(v_dm)


def test_ips_weight_clipping_caps_weights(bandit):
    logged, truth = bandit
    # A low-support target (mass on the least-likely behavior action) inflates some weights.
    states = logged["state"].to_numpy()
    low = truth.behavior_policy.argmin(axis=1)
    pi_low = np.full_like(truth.behavior_policy, 0.02)
    pi_low[np.arange(truth.n_states), low] = 1.0 - 0.02 * (truth.n_actions - 1)
    target = pi_low[states]
    r = logged["reward"].to_numpy()
    w = ope._weights(target, logged["action"].to_numpy(), logged["mu_prob"].to_numpy())
    v_unclipped, _ = ope.ips(r, w)
    v_clipped, contrib = ope.ips(r, w, clip=5.0)
    assert (np.abs(contrib) <= 5.0 * (np.abs(r).max() + 1)).all()
    assert w.max() > 5.0  # the cap actually bites
    assert v_clipped != pytest.approx(v_unclipped)


# --------------------------------------------------------------------------------------
# pi_alpha conservative mixture
# --------------------------------------------------------------------------------------


def test_pi_alpha_endpoints_and_linearity(bandit):
    logged, truth = bandit
    mu = truth.behavior_policy
    pi = truth.targets["greedy"]

    assert ope.pi_alpha(mu, pi, 0.0) == pytest.approx(mu)
    assert ope.pi_alpha(mu, pi, 1.0) == pytest.approx(pi)
    mid = ope.pi_alpha(mu, pi, 0.25)
    assert mid == pytest.approx(0.75 * mu + 0.25 * pi)
    assert np.allclose(mid.sum(axis=1), 1.0)

    # Grid form -> (G, S, A).
    grid = ope.pi_alpha(mu, pi, [0.0, 0.5, 1.0])
    assert grid.shape == (3, *mu.shape)
    assert grid[0] == pytest.approx(mu)
    assert grid[2] == pytest.approx(pi)


def test_pi_alpha_value_is_exactly_linear_and_monotone(bandit):
    """V(pi_alpha) is exactly (1-a)V(mu) + a V(pi_tilde): V(mu) at 0, V(pi_tilde) at 1."""
    logged, truth = bandit
    mu = truth.behavior_policy
    pi = truth.targets["greedy"]
    v_mu = truth.behavior_value
    v_pi = truth.target_value("greedy")
    alphas = [0.0, 0.1, 0.25, 0.5, 1.0]
    vals = [truth.policy_value(ope.pi_alpha(mu, pi, a)) for a in alphas]
    assert vals[0] == pytest.approx(v_mu)
    assert vals[-1] == pytest.approx(v_pi)
    for a, v in zip(alphas, vals):
        assert v == pytest.approx((1 - a) * v_mu + a * v_pi)
    # Greedy improves on behavior, so the path is monotone increasing.
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# --------------------------------------------------------------------------------------
# SPEC 9 self-test 1: behavior-policy recovery
# --------------------------------------------------------------------------------------


def test_behavior_policy_recovery_all_estimators(bandit):
    logged, truth = bandit
    rec = ope.behavior_policy_recovery(logged, n_actions=truth.n_actions, seed=SEED, n_boot=200)
    assert rec["ips_weights_unit"] is True  # pi == mu -> weights exactly 1
    assert rec["passed"] is True
    for name, e in rec["estimators"].items():
        assert e["ok"], f"{name} failed to recover held-out mean: {e}"
    # IPS/SNIPS recover the held-out mean *exactly* (weights are 1).
    assert rec["estimators"]["IPS"]["value"] == pytest.approx(rec["observed_mean"])
    assert rec["estimators"]["SNIPS"]["value"] == pytest.approx(rec["observed_mean"])


# --------------------------------------------------------------------------------------
# SPEC 9 self-test 2: known-value recovery + double robustness
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("target_name", ["greedy", "shift"])
def test_known_value_recovery_within_ci(bandit, target_name):
    logged, truth = bandit
    target, q_row, *_ = _pack(logged, truth, target_name)
    v_true = truth.target_value(target_name)
    res = ope.evaluate_policy(logged, target, q_hat=q_row, n_boot=300, seed=SEED)
    # The unbiased estimators must cover the exact analytic value.
    for name in ("IPS", "SNIPS", "DR"):
        lo, hi = res["estimators"][name]["ci95"]
        assert lo <= v_true <= hi, f"{name} CI [{lo:.4f},{hi:.4f}] misses V={v_true:.4f}"
    # Point estimates are close to truth as well.
    assert res["estimators"]["DR"]["value"] == pytest.approx(v_true, abs=0.03)
    assert res["verdict"] == "CONSISTENT"


def test_double_robustness_biased_outcome_model(bandit):
    """Misspecified q_hat (constant offset) biases DM but DR stays near truth."""
    logged, truth = bandit
    target, q_row, r, a, mu, w = _pack(logged, truth, "greedy")
    v_true = truth.target_value("greedy")
    q_bad = q_row + 0.5  # additive misspecification

    v_dm, _ = ope.dm(q_bad, target)
    v_dr, _ = ope.dr(r, w, q_bad, target, a)
    assert abs(v_dm - v_true) > 0.3  # DM is badly biased (~ +0.5)
    assert abs(v_dr - v_true) < 0.05  # DR corrects it via the IPS term


def test_double_robustness_corrupted_propensities(bandit):
    """Corrupted propensities bias IPS but DR stays near truth when q_hat is correct."""
    logged, truth = bandit
    states = logged["state"].to_numpy()
    target = truth.target_probs_for(states, "greedy")
    v_true = truth.target_value("greedy")
    # Correct q_hat = the true mean-reward table; corrupted weights = true weights * 3.
    q_true_row = truth.q_table[states]
    r = logged["reward"].to_numpy()
    a = logged["action"].to_numpy()
    w = ope._weights(target, a, logged["mu_prob"].to_numpy())
    w_bad = w * 3.0

    v_ips, _ = ope.ips(r, w_bad)
    v_dr, _ = ope.dr(r, w_bad, q_true_row, target, a)
    assert abs(v_ips - v_true) > 0.3  # IPS is badly biased
    assert abs(v_dr - v_true) < 0.05  # DR is protected by the correct outcome model


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------


def test_diagnostics_behavior_equals_target_is_degenerate(bandit):
    logged, truth = bandit
    mu_mat = logged[[f"mu_prob_{i}" for i in range(truth.n_actions)]].to_numpy()
    a = logged["action"].to_numpy()
    w = ope._weights(mu_mat, a, logged["mu_prob"].to_numpy())
    d = ope.ope_diagnostics(w, mu_mat, mu_mat, support_threshold=0.01, actions=a)
    assert d["ess_frac"] == pytest.approx(1.0)  # weights all 1
    assert d["max_weight"] == pytest.approx(1.0)
    assert d["mean_kl"] == pytest.approx(0.0, abs=1e-9)
    assert d["mean_tv"] == pytest.approx(0.0, abs=1e-9)


def test_diagnostics_ess_drops_for_a_different_target(bandit):
    logged, truth = bandit
    states = logged["state"].to_numpy()
    mu_mat = logged[[f"mu_prob_{i}" for i in range(truth.n_actions)]].to_numpy()
    target = truth.target_probs_for(states, "greedy")
    a = logged["action"].to_numpy()
    w = ope._weights(target, a, logged["mu_prob"].to_numpy())
    d = ope.ope_diagnostics(w, mu_mat, target, support_threshold=0.01)
    assert d["ess_frac"] < 0.9  # pi != mu -> effective sample size shrinks
    assert d["mean_kl"] > 0.0
    assert d["mean_tv"] > 0.0


def test_support_fraction_responds_to_low_propensity_target(bandit):
    logged, truth = bandit
    states = logged["state"].to_numpy()
    mu_mat = logged[[f"mu_prob_{i}" for i in range(truth.n_actions)]].to_numpy()
    a = logged["action"].to_numpy()
    # Threshold that some behavior probabilities fall below.
    thr = float(np.quantile(truth.behavior_policy, 0.4))

    # Target concentrating on the lowest- vs highest-propensity action per state.
    def concentrate(idx_per_state):
        pol = np.full_like(truth.behavior_policy, 1e-6)
        pol[np.arange(truth.n_states), idx_per_state] = 1.0 - 1e-6 * (truth.n_actions - 1)
        return pol[states]

    low = concentrate(truth.behavior_policy.argmin(axis=1))
    high = concentrate(truth.behavior_policy.argmax(axis=1))
    w_low = ope._weights(low, a, logged["mu_prob"].to_numpy())
    w_high = ope._weights(high, a, logged["mu_prob"].to_numpy())
    d_low = ope.ope_diagnostics(w_low, mu_mat, low, support_threshold=thr)
    d_high = ope.ope_diagnostics(w_high, mu_mat, high, support_threshold=thr)
    assert d_low["oos_support_frac"] > d_high["oos_support_frac"]


# --------------------------------------------------------------------------------------
# agreement verdict
# --------------------------------------------------------------------------------------


def test_inconclusive_verdict_fires_on_inconsistent_inputs(bandit):
    """A wrong q_hat scale drags DM far from IPS/DR with tight CIs -> INCONCLUSIVE."""
    logged, truth = bandit
    states = logged["state"].to_numpy()
    target = truth.target_probs_for(states, "greedy")
    q_row, _ = ope.outcome_model_means(logged, truth.n_actions)
    q_scaled = q_row * 4.0 + 1.0  # badly mis-scaled outcome model

    res = ope.evaluate_policy(logged, target, q_hat=q_scaled, n_boot=300, seed=SEED)
    assert res["verdict"] == "INCONCLUSIVE"
    # The DM/DR pair (or DM/SNIPS) is the culprit.
    disagreeing = [p["pair"] for p in res["verdict_detail"]["pairs"] if p["disagree"]]
    assert any("DM" in pair for pair in disagreeing)


def test_evaluate_policy_runs_fqe_without_episode_columns(bandit):
    """A bandit frame with no pa_id/step columns still evaluates (one-step interpretation)."""
    logged, truth = bandit
    bare = logged.drop(columns=["pa_id", "step"])
    target = truth.target_probs_for(bare["state"].to_numpy(), "greedy")
    q_row, _ = ope.outcome_model_means(bare, truth.n_actions)
    res = ope.evaluate_policy(bare, target, q_hat=q_row, n_boot=80, seed=SEED)
    # FQE (and the per-decision estimators) all run without the episode/step columns.
    assert "FQE" in res["estimators"]
    assert res["estimators"]["FQE"]["value"] == pytest.approx(truth.target_value("greedy"), abs=0.05)
    assert res["sequential"] is False


def test_evaluate_policy_report_has_spec9_fields(bandit):
    logged, truth = bandit
    states = logged["state"].to_numpy()
    target = truth.target_probs_for(states, "greedy")
    q_row, _ = ope.outcome_model_means(logged, truth.n_actions)
    res = ope.evaluate_policy(logged, target, q_hat=q_row, n_boot=100, seed=SEED)
    # Exactly the SPEC 9 report fields from config ope.report.
    for field in ("value", "lower_95", "ess", "oos_support_frac", "max_weight",
                  "kl_from_behavior", "tv_from_behavior"):
        assert field in res["report"]
    assert res["report"]["lower_95"] <= res["report"]["value"]
    # Every configured estimator ran.
    for name in ("DM", "IPS", "SNIPS", "DR", "stepwise_DR", "FQE"):
        assert name in res["estimators"]
