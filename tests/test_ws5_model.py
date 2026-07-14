"""WS5 tabular-MDP model unit tests (SPEC 12.6; decisions D40-D43).

Pins the model-level contracts on small, hand-checkable, deterministic inputs: the state
encoder ids for all three designs and the leakage-safe trigger flag (missing priors, prior-
pitch-only execution, terminal mapping); the MDP estimator's exact Dirichlet-smoothed
transitions, two-level-shrunk rewards and support counts on tiny handcrafted episodes;
undiscounted policy iteration on a hand-solvable 3-state MDP; the simulator's agreement with
the analytic value and its seeded reproducibility; and -- the unit-level proof the machinery
can *represent a setup* -- a minimal constructed world where one family creates the next
pitch's trigger, so the trigger-design optimal policy chooses it (and is worth more) while the
count design cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.families import FAMILIES
from workstreams.ws5_tabular_mdp import model as M

FIDX = {f: i for i, f in enumerate(FAMILIES)}


def _table(rows):
    """Build a tiny decision table from a list of dicts (fills the columns the MDP needs)."""
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], categories=list(FAMILIES))
    df["outcome1"] = df["outcome1"].astype(object)
    return df


# --- encode_states: hand-checked ids ---------------------------------------------------

def test_count_state_ids_hand_checked():
    """count design: state = balls*3 + strikes; terminals appended at 12..15."""
    tbl = _table([
        dict(pa_id=1, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=95.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=2, balls=0, strikes=1, family="SL", exec_release_speed=86.0, outcome1="in_play"),
        dict(pa_id=2, pitch_number=1, balls=2, strikes=1, family="FF", exec_release_speed=95.0, outcome1="ball"),
        dict(pa_id=2, pitch_number=2, balls=3, strikes=1, family="FF", exec_release_speed=95.0, outcome1="ball"),
    ])
    ss = M.encode_states(tbl, "count")
    assert ss.n_nonterminal == 12 and ss.n_states == 16
    assert list(ss.state_ids) == [0, 1, 7, 10]  # (0,0)->0 (0,1)->1 (2,1)->7 (3,1)->10
    assert ss.terminal_ids == {"walk": 12, "strikeout": 13, "hbp": 14, "in_play_end": 15}


def test_count_prev_state_ids_hand_checked():
    """count_prev: state = count_id*9 + prev_idx; prev NONE = slot 8."""
    tbl = _table([
        dict(pa_id=1, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=95.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=2, balls=0, strikes=1, family="SL", exec_release_speed=86.0, outcome1="in_play"),
    ])
    ss = M.encode_states(tbl, "count_prev")
    assert ss.n_nonterminal == 108 and ss.n_states == 112
    # pitch 1: (0,0) prev NONE -> 0*9 + 8 = 8
    assert ss.state_ids[0] == 8
    # pitch 2: (0,1) prev FF(idx 0) -> 1*9 + 0 = 9
    assert ss.state_ids[1] == 9
    assert ss.terminal_ids["walk"] == 108


def test_count_prev_trigger_state_ids_hand_checked():
    """count_prev_trigger: state = (count_id*9 + prev_idx)*2 + trigger."""
    tbl = _table([
        dict(pa_id=1, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=95.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=2, balls=0, strikes=1, family="CU", exec_release_speed=79.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=3, balls=0, strikes=2, family="FF", exec_release_speed=95.0, outcome1="in_play"),
    ])
    ss = M.encode_states(tbl, "count_prev_trigger")
    assert ss.n_nonterminal == 216 and ss.n_states == 220
    # pitch1: (0,0) prev NONE trig0 -> (0*9+8)*2+0 = 16
    assert ss.state_ids[0] == 16
    # pitch2: (0,1) prev FF trig0 (only 1 prior) -> (1*9+0)*2+0 = 18
    assert ss.state_ids[1] == 18
    # pitch3: (0,2) prev CU(idx4) trig1 (|79-95|=16>=5) -> (2*9+4)*2+1 = 45
    assert ss.state_ids[2] == 45
    assert ss.terminal_ids["in_play_end"] == 219


# --- trigger flag: missing priors, leakage, threshold ----------------------------------

def test_trigger_needs_two_priors():
    """The trigger flag is 0 for the first two pitches of every PA (both priors must exist)."""
    tbl = _table([
        dict(pa_id=1, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=95.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=2, balls=0, strikes=1, family="CU", exec_release_speed=79.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=3, balls=0, strikes=2, family="FF", exec_release_speed=95.0, outcome1="foul"),
    ])
    _, trig, has2 = M.prev_and_trigger(tbl)
    assert list(trig) == [0, 0, 1]        # only pitch 3 has two priors and a >=5 gap
    assert list(has2) == [False, False, True]


def test_trigger_uses_prior_pitches_only_no_leakage():
    """The pitch-t trigger uses velo_{t-1}, velo_{t-2} only -- changing the CURRENT pitch's
    execution velocity must not change its trigger (leakage guard, SPEC 0)."""
    base = [
        dict(pa_id=1, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=95.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=2, balls=0, strikes=1, family="FF", exec_release_speed=95.0, outcome1="called_strike"),
        dict(pa_id=1, pitch_number=3, balls=0, strikes=2, family="SL", exec_release_speed=86.0, outcome1="foul"),
    ]
    _, trig_a, _ = M.prev_and_trigger(_table(base))
    # velo_2 - velo_1 = 0 -> pitch-3 trigger is 0, regardless of pitch-3's own velo.
    assert trig_a[2] == 0
    wild = [dict(r) for r in base]
    wild[2]["exec_release_speed"] = 40.0  # absurd current-pitch velo
    _, trig_b, _ = M.prev_and_trigger(_table(wild))
    assert trig_b[2] == 0                  # unchanged: current execution never enters the flag


def test_trigger_threshold_boundary():
    """Gap exactly at the threshold fires; just below does not."""
    def trig_for(v2):
        tbl = _table([
            dict(pa_id=1, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=90.0, outcome1="ball"),
            dict(pa_id=1, pitch_number=2, balls=1, strikes=0, family="FF", exec_release_speed=v2, outcome1="ball"),
            dict(pa_id=1, pitch_number=3, balls=2, strikes=0, family="FF", exec_release_speed=88.0, outcome1="ball"),
        ])
        _, trig, _ = M.prev_and_trigger(tbl, threshold=5.0)
        return trig[2]
    assert trig_for(95.0) == 1   # |95 - 90| = 5.0 >= 5 -> fires
    assert trig_for(94.9) == 0   # |94.9 - 90| = 4.9 < 5 -> no


# --- terminal mapping ------------------------------------------------------------------

def test_terminal_type_mapping():
    assert M.terminal_type("in_play", 0, 0) == "in_play_end"
    assert M.terminal_type("hbp", 1, 1) == "hbp"
    assert M.terminal_type("ball", 3, 0) == "walk"        # ball four
    assert M.terminal_type("ball", 2, 0) is None          # ball three: PA continues
    assert M.terminal_type("called_strike", 0, 2) == "strikeout"  # strike three
    assert M.terminal_type("whiff", 0, 2) == "strikeout"
    assert M.terminal_type("called_strike", 0, 1) is None
    assert M.terminal_type("foul", 0, 2) is None          # fouls never terminate


# --- estimate_mdp: exact smoothed values + support -------------------------------------

@pytest.fixture
def tiny_mdp_table():
    """Two 2-pitch PAs, count design, everything hand-traceable."""
    return _table([
        dict(pa_id=1, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=95.0, outcome1="called_strike", R=0.045),
        dict(pa_id=1, pitch_number=2, balls=0, strikes=1, family="FF", exec_release_speed=95.0, outcome1="in_play", R=-0.10),
        dict(pa_id=2, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=90.0, outcome1="ball", R=-0.04),
        dict(pa_id=2, pitch_number=2, balls=1, strikes=0, family="FF", exec_release_speed=90.0, outcome1="in_play", R=0.05),
    ])


def test_estimate_mdp_transitions_and_support(tiny_mdp_table):
    """Exact Dirichlet-smoothed transitions over the per-source reachable set, and support counts."""
    mdp = M.estimate_mdp(tiny_mdp_table, "count", alpha_t=1.0, alpha_r=1.0)
    ff = FIDX["FF"]
    s0, s1, s3, term = 0, 1, 3, mdp.terminal_ids["in_play_end"]  # (0,0),(0,1),(1,0)
    # From s0: FF observed -> {s1, s3}; succ(s0) size 2; each (1+1)/(2+1*2)=0.5.
    assert mdp.support[s0, ff] == 2
    assert mdp.P[s0, ff, s1] == pytest.approx(0.5)
    assert mdp.P[s0, ff, s3] == pytest.approx(0.5)
    # From s1: FF -> in_play_end; succ size 1; (1+1)/(1+1)=1.0.
    assert mdp.P[s1, ff, term] == pytest.approx(1.0)
    assert mdp.support[s1, ff] == 1
    # Unobserved action SI at s0 backs off to the state action-marginal (same successors here).
    si = FIDX["SI"]
    assert mdp.support[s0, si] == 0
    assert mdp.P[s0, si, s1] == pytest.approx(0.5)
    # Terminal is absorbing.
    assert mdp.P[term, ff, term] == pytest.approx(1.0)


def test_estimate_mdp_two_level_reward(tiny_mdp_table):
    """R_hat(s,a) shrinks the cell mean toward the state mean, then the global mean (alpha_r=1)."""
    mdp = M.estimate_mdp(tiny_mdp_table, "count", alpha_t=1.0, alpha_r=1.0)
    ff = FIDX["FF"]
    r_global = (0.045 - 0.10 - 0.04 + 0.05) / 4.0
    # s0 has two FF rows (0.045, -0.04): state sum 0.005, n 2.
    r_state_s0 = (0.005 + 1.0 * r_global) / (2 + 1.0)
    r_hat_s0_ff = (0.005 + 1.0 * r_state_s0) / (2 + 1.0)
    assert mdp.R[0, ff] == pytest.approx(r_hat_s0_ff)
    # s1 has one FF row (-0.10).
    r_state_s1 = (-0.10 + 1.0 * r_global) / (1 + 1.0)
    r_hat_s1_ff = (-0.10 + 1.0 * r_state_s1) / (1 + 1.0)
    assert mdp.R[1, ff] == pytest.approx(r_hat_s1_ff)
    # Terminals carry zero reward.
    assert mdp.R[mdp.terminal_ids["in_play_end"], ff] == pytest.approx(0.0)


# --- policy_iteration: hand-solvable 3-state MDP ---------------------------------------

def test_policy_iteration_hand_solvable():
    r"""A 3-state MDP (s0, s1, terminal t): from s0, A -> reward 0 then s1; B -> reward 1 then t.
    From s1, A -> reward 2 then t. Undiscounted optimum: s0 takes A (0 + V(s1)=2 > 1)."""
    S, A = 3, 2
    P = np.zeros((S, A, S)); R = np.zeros((S, A))
    t = 2
    P[0, 0, 1] = 1.0; R[0, 0] = 0.0     # s0,A -> s1
    P[0, 1, t] = 1.0; R[0, 1] = 1.0     # s0,B -> terminal, reward 1
    P[1, 0, t] = 1.0; R[1, 0] = 2.0     # s1,A -> terminal, reward 2
    P[1, 1, t] = 1.0; R[1, 1] = 0.5     # s1,B -> terminal, reward 0.5
    P[t, :, t] = 1.0                    # absorbing terminal
    feasible = np.array([[True, True], [True, True], [False, False]])
    policy, Q, V = M.policy_iteration(P, R, feasible, gamma=1.0)
    assert policy[0] == 0 and policy[1] == 0
    assert V[0] == pytest.approx(2.0) and V[1] == pytest.approx(2.0)
    assert Q[0, 0] == pytest.approx(2.0) and Q[0, 1] == pytest.approx(1.0)


def test_policy_iteration_convergence_guard():
    """A pathological improper policy is caught by the max_iter guard (never hangs silently)."""
    S, A = 2, 1
    P = np.zeros((S, A, S)); R = np.zeros((S, A))
    P[0, 0, 0] = 1.0  # s0 self-loops forever (never terminates) -- improper
    P[1, 0, 1] = 1.0
    R[0, 0] = 1.0
    feasible = np.array([[True], [False]])
    # Iterative fallback yields a (large/degenerate) value; the guard bounds the work.
    policy, Q, V = M.policy_iteration(P, R, feasible, gamma=1.0, max_iter=50)
    assert policy.shape == (S,)


# --- simulator: equals the analytic value; reproducible --------------------------------

def test_simulate_matches_analytic_deterministic_mdp():
    """On a deterministic MDP the simulated return equals the analytic policy value exactly."""
    S, A = 3, 2
    P = np.zeros((S, A, S)); R = np.zeros((S, A))
    t = 2
    P[0, 0, 1] = 1.0; R[0, 0] = 0.3
    P[0, 1, t] = 1.0; R[0, 1] = 0.1
    P[1, 0, t] = 1.0; R[1, 0] = 0.7
    P[1, 1, t] = 1.0; R[1, 1] = 0.2
    P[t, :, t] = 1.0
    policy = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # always action A
    term = np.array([False, False, True])
    V, _ = M.policy_evaluation(P, R, policy, terminal_mask=term)
    start = np.array([1.0, 0.0, 0.0])
    analytic = float(start @ V)  # 0.3 + 0.7 = 1.0
    sim_v, traces = M.simulate(P, R, policy, 100, np.random.default_rng(0), start, terminal_mask=term)
    assert analytic == pytest.approx(1.0)
    assert sim_v == pytest.approx(analytic)          # deterministic dynamics -> exact
    assert traces[0][0] == [0, 1]                    # visited s0 then s1


def test_simulate_seeded_reproducible_stochastic():
    """A stochastic MDP: two seeds match themselves; the MC mean approaches the analytic value."""
    S, A = 2, 1
    P = np.zeros((S, A, S)); R = np.zeros((S, A))
    P[0, 0, 0] = 0.5; P[0, 0, 1] = 0.5; R[0, 0] = 1.0  # s0 -> stay (r=1) or terminate
    P[1, 0, 1] = 1.0
    term = np.array([False, True])
    pol = np.array([[1.0], [1.0]])
    start = np.array([1.0, 0.0])
    a = M.simulate(P, R, pol, 4000, np.random.default_rng(3), start, terminal_mask=term)[0]
    b = M.simulate(P, R, pol, 4000, np.random.default_rng(3), start, terminal_mask=term)[0]
    assert a == b                                     # seeded reproducibility
    V, _ = M.policy_evaluation(P, R, pol, terminal_mask=term)
    assert a == pytest.approx(float(start @ V), abs=0.1)  # geometric-length return ~ analytic


# --- the setup proof: the machinery can represent a setup pitch ------------------------

def _setup_world(n_pa=120, boost=0.5, x_cost=0.05):
    r"""A minimal world where one family (CU, slow) *creates* the next pitch's trigger.

    Every PA is 3 pitches: pitch1 = FF (fast, 92); pitch2 in {FF, CU} (CU is slow, 78, so
    choosing CU at pitch2 makes ``|velo_2 - velo_1| >= 5`` -> pitch3's trigger fires); pitch3 =
    FF terminal (in_play). Pitch3 reward is ``boost`` when triggered else 0; pitch2's CU carries
    a small immediate cost ``x_cost`` (so a *myopic/count* policy avoids it). Half the PAs take
    CU at pitch2 so the estimator sees both. The trigger design must learn to *set up* (take CU);
    the count design cannot see the trigger and avoids CU's cost.
    """
    rows = []
    for i in range(n_pa):
        take_cu = (i % 2 == 0)
        p2_fam, p2_velo, p2_R = ("CU", 78.0, -x_cost) if take_cu else ("FF", 92.0, 0.0)
        p3_trig = abs(p2_velo - 92.0) >= 5.0
        rows += [
            dict(pa_id=i, pitch_number=1, balls=0, strikes=0, family="FF", exec_release_speed=92.0, outcome1="called_strike", R=0.0),
            dict(pa_id=i, pitch_number=2, balls=0, strikes=1, family=p2_fam, exec_release_speed=p2_velo, outcome1="called_strike", R=p2_R),
            dict(pa_id=i, pitch_number=3, balls=0, strikes=2, family="FF", exec_release_speed=92.0, outcome1="in_play", R=(boost if p3_trig else 0.0)),
        ]
    return _table(rows)


def test_setup_representable_trigger_design_sets_up_and_wins():
    """Unit-level proof: the trigger design's optimal policy takes the setup family CU at pitch 2
    (creating pitch 3's trigger) while the count design avoids it; the trigger design's value is
    higher and its simulator agrees."""
    tbl = _setup_world()
    mdp_c = M.estimate_mdp(tbl, "count", alpha_t=0.5, alpha_r=0.5)
    mdp_t = M.estimate_mdp(tbl, "count_prev_trigger", alpha_t=0.5, alpha_r=0.5)
    pol_c, Qc, Vc = M.policy_iteration(mdp_c.P, mdp_c.R, mdp_c.feasible)
    pol_t, Qt, Vt = M.policy_iteration(mdp_t.P, mdp_t.R, mdp_t.feasible)

    # The pitch-2 decision state: (0,1). count id = 1.
    cu, ff = FIDX["CU"], FIDX["FF"]
    # count design: state (0,1) -> id 1; greedy avoids CU (its cost, no visible setup value).
    assert pol_c[1] == ff
    # trigger design: state (0,1, prev=FF, trig=0) -> (1*9 + 0)*2 + 0 = 18; greedy takes CU (setup).
    s_trig = (1 * M.N_PREV + ff) * 2 + 0
    assert pol_t[s_trig] == cu

    v_c = float(mdp_c.start_dist @ Vc)
    v_t = float(mdp_t.start_dist @ Vt)
    assert v_t > v_c + 0.1          # the setup is worth clearly more than the count policy

    # The transparent simulator reproduces the trigger design's advantage.
    sim_t, _ = M.simulate(mdp_t.P, mdp_t.R, M.greedy_policy_matrix(pol_t, mdp_t.n_actions),
                          5000, np.random.default_rng(1), mdp_t.start_dist)
    sim_c, _ = M.simulate(mdp_c.P, mdp_c.R, M.greedy_policy_matrix(pol_c, mdp_c.n_actions),
                          5000, np.random.default_rng(1), mdp_c.start_dist)
    assert sim_t > sim_c
    assert sim_t == pytest.approx(v_t, abs=0.03)


def test_onehot_regressor_matches_ope_tabular_regressor():
    """The fast one-hot group-mean regressor is numerically identical to ope.tabular_regressor
    on FQE-style one-hot [state | action] features -- exact cell means, global-mean fallback for
    unseen cells -- so the refit bootstrap changes speed, never values."""
    from pitchseq.eval import ope

    rng = np.random.default_rng(0)
    n_states, n_actions, n = 5, 3, 400
    s = rng.integers(0, n_states, size=n)
    a = rng.integers(0, n_actions, size=n)
    # leave cell (4, 2) unseen to exercise the fallback
    keep = ~((s == 4) & (a == 2))
    s, a = s[keep], a[keep]
    y = rng.normal(size=len(s))
    X = np.zeros((len(s), n_states + n_actions))
    X[np.arange(len(s)), s] = 1.0
    X[np.arange(len(s)), n_states + a] = 1.0

    slow = ope.tabular_regressor()().fit(X, y)
    fast = M.onehot_tabular_regressor(n_actions)().fit(X, y)
    assert np.allclose(slow.predict(X), fast.predict(X))

    # every (state, action) combo, including the unseen one (global-mean fallback)
    grid_s, grid_a = np.meshgrid(np.arange(n_states), np.arange(n_actions), indexing="ij")
    gs, ga = grid_s.ravel(), grid_a.ravel()
    G = np.zeros((len(gs), n_states + n_actions))
    G[np.arange(len(gs)), gs] = 1.0
    G[np.arange(len(gs)), n_states + ga] = 1.0
    assert np.allclose(slow.predict(G), fast.predict(G))
    unseen = (gs == 4) & (ga == 2)
    assert fast.predict(G)[unseen][0] == pytest.approx(float(y.mean()))


def test_onehot_regressor_inside_fqe_matches():
    """End-to-end: ope.fqe with the fast factory returns exactly the same value as with the
    generic tabular factory on a small two-step logged frame."""
    from pitchseq.eval import ope
    from pitchseq.synth import make_two_step_mdp

    logged, truth = make_two_step_mdp(n_episodes=300, seed=5)
    pi = truth.per_row_target(logged, "lookahead_greedy")
    v_slow, c_slow = ope.fqe(logged, pi, truth.n_actions, regressor_factory=ope.tabular_regressor())
    v_fast, c_fast = ope.fqe(logged, pi, truth.n_actions,
                             regressor_factory=M.onehot_tabular_regressor(truth.n_actions))
    assert v_fast == pytest.approx(v_slow, abs=1e-12)
    assert np.allclose(c_fast, c_slow)


def test_setup_diagnostics_positive_gap():
    """setup_diagnostics reports a positive setup Q-gap and a policy change from trigger knowledge."""
    tbl = _setup_world()
    mdp_t = M.estimate_mdp(tbl, "count_prev_trigger", alpha_t=0.5, alpha_r=0.5)
    mdp_p = M.estimate_mdp(tbl, "count_prev", alpha_t=0.5, alpha_r=0.5)
    diag = M.setup_diagnostics(mdp_t, mdp_p)
    assert diag["setup_gap_mean"] > 0.0            # the MDP values creating a trigger
    assert diag["n_states_with_gap"] >= 1
    assert 0.0 <= diag["optimal_action_change_frac"] <= 1.0
