"""Sequential OPE estimators on the two-step MDP fixture (SPEC 9 / 11).

Step-wise (per-decision) DR and Fitted-Q Evaluation must recover an analytically computable
target-policy value on a tiny two-step episodic decision process, and FQE must degenerate to
the Direct Method on one-step (bandit) data.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchseq.eval import ope
from pitchseq.synth import make_logged_bandit, make_two_step_mdp

SEED = 20260713


@pytest.fixture(scope="module")
def mdp():
    logged, truth = make_two_step_mdp(n_episodes=5000, seed=SEED)
    return logged, truth


def _mdp_arrays(logged, truth, name="lookahead_greedy"):
    target = truth.per_row_target(logged, name)
    q_hat = truth.per_row_q(logged, name)  # exact target Q-function as the outcome model
    r = logged["reward"].to_numpy()
    a = logged["action"].to_numpy()
    mu = logged["mu_prob"].to_numpy()
    w = ope._weights(target, a, mu)
    return target, q_hat, r, a, w


def test_stepwise_dr_recovers_two_step_value(mdp):
    logged, truth = mdp
    target, q_hat, r, a, w = _mdp_arrays(logged, truth)
    v_true = truth.target_value("lookahead_greedy")

    value, contrib = ope.stepwise_dr(
        r, w, q_hat, target, a, logged["pa_id"].to_numpy(), logged["step"].to_numpy()
    )
    assert len(contrib) == logged["pa_id"].nunique()  # one contribution per episode
    assert value == pytest.approx(v_true, abs=0.05)
    # Within a bootstrap CI too.
    ci = ope._bootstrap_ci(contrib, n_boot=300, seed=SEED)
    assert ci["lo"] <= v_true <= ci["hi"]


def test_stepwise_dr_reduces_to_dr_on_one_step_episodes():
    """On one-step (bandit) episodes step-wise DR must equal the plain DR estimate."""
    logged, truth = make_logged_bandit(n_rounds=4000, seed=SEED)
    states = logged["state"].to_numpy()
    target = truth.target_probs_for(states, "greedy")
    q_row, _ = ope.outcome_model_means(logged, truth.n_actions)
    r = logged["reward"].to_numpy()
    a = logged["action"].to_numpy()
    w = ope._weights(target, a, logged["mu_prob"].to_numpy())
    v_dr, _ = ope.dr(r, w, q_row, target, a)
    v_sdr, _ = ope.stepwise_dr(
        r, w, q_row, target, a, logged["pa_id"].to_numpy(), logged["step"].to_numpy()
    )
    assert v_sdr == pytest.approx(v_dr)


@pytest.mark.parametrize("factory_name", ["tabular", "hgbr"])
def test_fqe_recovers_two_step_value(mdp, factory_name):
    logged, truth = mdp
    target = truth.per_row_target(logged, "lookahead_greedy")
    v_true = truth.target_value("lookahead_greedy")
    factory = ope.tabular_regressor() if factory_name == "tabular" else None  # None -> HGBR

    value, contrib = ope.fqe(logged, target, truth.n_actions, regressor_factory=factory, seed=SEED)
    assert len(contrib) == logged["pa_id"].nunique()
    tol = 0.02 if factory_name == "tabular" else 0.06
    assert value == pytest.approx(v_true, abs=tol)


def test_fqe_recovers_behavior_return(mdp):
    """FQE with the target set to the behavior policy recovers the behavior return."""
    logged, truth = mdp
    n_a = truth.n_actions
    mu_mat = logged[[f"mu_prob_{i}" for i in range(n_a)]].to_numpy()
    v_behavior = truth.policy_value(truth.mu0, truth.mu1)
    value, _ = ope.fqe(logged, mu_mat, n_a, regressor_factory=ope.tabular_regressor(), seed=SEED)
    assert value == pytest.approx(v_behavior, abs=0.03)


def test_fqe_degenerates_to_direct_method_on_bandit():
    """On one-step data, FQE(tabular) equals the Direct Method with fitted group means."""
    logged, truth = make_logged_bandit(n_rounds=6000, seed=SEED)
    states = logged["state"].to_numpy()
    target = truth.target_probs_for(states, "greedy")

    v_fqe, _ = ope.fqe(logged, target, truth.n_actions, regressor_factory=ope.tabular_regressor())
    # Group-mean outcome model (no shrinkage) == the tabular FQE fit on 1-step data.
    q_row, _ = ope.outcome_model_means(logged, truth.n_actions, prior_strength=1e-12)
    v_dm, _ = ope.dm(q_row, target)
    assert v_fqe == pytest.approx(v_dm, abs=1e-6)


def test_behavior_policy_recovery_on_sequential_data(mdp):
    """SPEC 9 self-test 1 on multi-step data: recover the held-out mean *episode return*."""
    logged, truth = mdp
    rec = ope.behavior_policy_recovery(logged, n_actions=truth.n_actions, seed=SEED, n_boot=200)
    assert rec["ips_weights_unit"] is True  # every per-step ratio is 1
    assert rec["passed"] is True
    # The observed value is the mean episode return, near the analytic behavior value.
    assert rec["observed_mean"] == pytest.approx(truth.policy_value(truth.mu0, truth.mu1), abs=0.03)
    # Only the trajectory estimators are checked on sequential data.
    assert set(rec["estimators"]) == {"stepwise_DR", "FQE"}


def test_stepwise_dr_validates_q_hat(mdp):
    """stepwise_dr rejects a wrong-shaped or non-finite q_hat (parity with dm / dr)."""
    logged, truth = mdp
    target, q_hat, r, a, w = _mdp_arrays(logged, truth)
    ep, st = logged["pa_id"].to_numpy(), logged["step"].to_numpy()
    with pytest.raises(ValueError):
        ope.stepwise_dr(r, w, q_hat[:, :1], target, a, ep, st)  # shape mismatch
    q_nan = q_hat.copy()
    q_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        ope.stepwise_dr(r, w, q_nan, target, a, ep, st)  # non-finite


def test_fqe_handles_nonuniform_episode_starts(mdp):
    """Episodes that start at different steps each get their own initial-state value (no
    uninitialised slots)."""
    logged, truth = mdp
    # Drop the step-0 row of every even episode, so those episodes start at step 1.
    drop = (logged["step"].to_numpy() == 0) & (logged["pa_id"].to_numpy() % 2 == 0)
    frame = logged.loc[~drop].reset_index(drop=True)
    target = truth.per_row_target(frame, "lookahead_greedy")
    value, contrib = ope.fqe(frame, target, truth.n_actions, regressor_factory=ope.tabular_regressor())
    assert len(contrib) == frame["pa_id"].nunique()
    assert np.isfinite(value)
    assert np.isfinite(contrib).all()  # no garbage from uninitialised episode slots


def test_evaluate_policy_uses_sequential_estimators_on_mdp(mdp):
    logged, truth = mdp
    target = truth.per_row_target(logged, "lookahead_greedy")
    q_hat = truth.per_row_q(logged, "lookahead_greedy")
    v_true = truth.target_value("lookahead_greedy")
    res = ope.evaluate_policy(
        logged, target, q_hat=q_hat, n_boot=200, seed=SEED,
        fqe_regressor_factory=ope.tabular_regressor(),
    )
    assert res["sequential"] is True
    assert res["n_episodes"] == logged["pa_id"].nunique()
    # Only the trajectory-level estimators run on multi-step data; the per-decision bandit
    # estimators are skipped (they do not evaluate the episode return).
    assert set(res["estimators"]) == {"stepwise_DR", "FQE"}
    assert res["primary_estimator"] == "stepwise_DR"
    for name in ("stepwise_DR", "FQE"):
        assert res["estimators"][name]["value"] == pytest.approx(v_true, abs=0.06)
    # The verdict compares the sequential estimators and finds them consistent.
    assert res["verdict"] == "CONSISTENT"
    assert set(res["verdict_detail"]["compared"]) == {"stepwise_DR", "FQE"}
