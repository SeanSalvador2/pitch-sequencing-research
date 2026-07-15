"""WS7 conservative-offline-RL model unit tests (SPEC 12.7 / 10; decisions D48-D50).

Pins the model-level contracts on small, hand-checkable, deterministic inputs: unpenalised FQI
recovers the analytic optimal Q on the shared synthetic two-step MDP (and its greedy is optimal),
a harsh support penalty makes the greedy retreat to high-support actions; the behaviour-support
penalty and the FQI diagnostics on constructed counts; the exploitability game (an exact
hand-solvable 2x2 equilibrium, a dominated policy's positive exploitability, an equilibrium
policy's ~0); the successor / effective-feasibility helpers; and the frontier schema + a smoke
figure render (Agg backend, file exists).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitchseq.synth import make_two_step_mdp
from workstreams.ws7_offline_rl import model as M


# --- FQI recovers the analytic optimal Q on the two-step MDP ----------------------------

def test_fqi_recovers_analytic_optimal_q_two_step():
    r"""Unpenalised tabular FQI on the synth two-step MDP recovers the analytic optimal Q on every
    *reached* (state, action) cell, and its greedy policy is optimal there.

    Optimal Q: ``Q1*(s1,a) = R1(s1,a)`` (terminal) and ``Q0*(s0,a) = R0(s0,a) + max_a' R1(T(s0,a),a')``
    -- computed directly from the ground truth. With zero reward noise the exact group-mean fit
    recovers the mean rewards, so FQI reaches the fixed point (unreached states carry no data and are
    excluded from the comparison)."""
    logged, truth = make_two_step_mdp(n_episodes=4000, seed=5, reward_noise=0.0)
    n_s0, A = truth.n_s0, truth.n_actions
    n_states = truth.n_states_total
    succ, terminal = M.successor_index(logged["pa_id"].to_numpy(), logged["step"].to_numpy())
    feas = np.ones((len(logged), A), dtype=bool)
    pen = np.zeros((len(logged), A))
    Q, drifts = M.fitted_q_iteration(
        logged["state"].to_numpy(), logged["action"].to_numpy(), logged["reward"].to_numpy(),
        succ, terminal, feas, pen, n_states, A, lam=0.0, n_iter=6,
    )
    v1 = truth.R1.max(axis=1)
    Q0_star = truth.R0 + v1[truth.T]
    Q1_star = truth.R1

    reached = np.unique(logged["state"].to_numpy())
    reached_s0 = reached[reached < n_s0]
    reached_s1 = reached[reached >= n_s0] - n_s0
    assert np.abs(Q[reached_s0] - Q0_star[reached_s0]).max() < 1e-9
    assert np.abs(Q[n_s0 + reached_s1] - Q1_star[reached_s1]).max() < 1e-9
    # Greedy is optimal on the reached states.
    assert (Q[reached_s0].argmax(1) == Q0_star[reached_s0].argmax(1)).all()
    assert (Q[n_s0 + reached_s1].argmax(1) == Q1_star[reached_s1].argmax(1)).all()
    # Converged (drift decays; the 2-step horizon reaches the fixed point in one backup).
    assert drifts[-1] < 1e-9


def test_harsh_penalty_makes_greedy_retreat_to_support():
    r"""A high-Q, low-support action is chosen unpenalised; a harsh support penalty retreats the
    greedy to the lower-Q, high-support action (the CQL-lite conservatism)."""
    q = np.array([[1.0, 0.3]])             # action 0 has higher Q
    mu = np.array([[0.005, 0.6]])          # action 0 is below the support floor
    feas = np.array([[True, True]])
    pen = M.behavior_support_penalty(mu, floor=0.02)
    assert pen.tolist() == [[1.0, 0.0]]
    pol_free = M.conservative_greedy(q, pen, feas, lam=0.0)
    pol_harsh = M.conservative_greedy(q, pen, feas, lam=2.0)
    assert pol_free.argmax() == 0          # unpenalised -> the high-Q action
    assert pol_harsh.argmax() == 1         # harsh penalty -> retreat to the high-support action


def test_behavior_support_penalty_math():
    """The penalty is exactly the 1[mu < floor] indicator per (row, action)."""
    mu = np.array([[0.5, 0.01, 0.02, 0.019], [0.3, 0.3, 0.3, 0.1]])
    pen = M.behavior_support_penalty(mu, floor=0.02)
    # floor is strict '<': 0.02 is NOT below 0.02.
    assert pen.tolist() == [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]


def test_fqi_diagnostics_on_constructed():
    """fqi_diagnostics reports the drift trace, penalty share among feasible cells, and the
    pessimism-bites share (unpenalised greedy landing on an off-support action)."""
    # 3 rows, 3 actions. Feasible: all but action 2 on row 0.
    feas = np.array([[True, True, False], [True, True, True], [True, True, True]])
    pen = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    q = np.array([[0.1, 0.9, 5.0],    # greedy feasible = action 1 (off-support -> bites)
                  [0.2, 0.1, 0.05],   # greedy = action 0 (on-support)
                  [9.0, 0.1, 0.2]])   # greedy = action 0 (off-support -> bites)
    diag = M.fqi_diagnostics([0.5, 0.2, 0.05], pen, feas, q, np.array([0, 1, 2]))
    assert diag["drift_per_iter"] == [0.5, 0.2, 0.05]
    assert diag["final_drift"] == 0.05 and diag["n_iter_run"] == 3
    # Penalty share over feasible cells: feasible cells = 8, penalised-and-feasible = row1 a2? no (feasible),
    # pen&feasible: row0 a1(1), row1 a2(1), row2 a0(1) = 3 of 8.
    assert diag["penalty_share"] == pytest.approx(3.0 / 8.0)
    # pessimism bites: rows 0 and 2 greedy land off-support -> 2/3.
    assert diag["pessimism_bites_frac"] == pytest.approx(2.0 / 3.0)


# --- successor / feasibility helpers ---------------------------------------------------

def test_successor_index_within_pa():
    """Successor is the next pitch of the same PA (original index); the last pitch is terminal."""
    pa = np.array([1, 1, 1, 2, 2])
    pitch = np.array([1, 2, 3, 1, 2])
    succ, terminal = M.successor_index(pa, pitch)
    assert succ.tolist() == [1, 2, -1, 4, -1]
    assert terminal.tolist() == [False, False, True, False, True]
    # Robust to unsorted input order.
    order = np.array([4, 0, 2, 1, 3])
    succ2, term2 = M.successor_index(pa[order], pitch[order])
    # row for (pa1,pitch1) is at position 1 here; its successor is (pa1,pitch2) at position 3.
    assert term2[np.where((pa[order] == 1) & (pitch[order] == 3))[0][0]]  # last pitch terminal


def test_effective_feasible_mask_fills_empty_rows():
    """A row with no feasible family is filled to all non-XX; XX stays infeasible."""
    feas = np.zeros((2, 8), dtype=bool)
    feas[0] = [True, False, False, False, False, False, False, False]  # only FF
    # row 1 is all-empty -> filled
    out = M.effective_feasible_mask(feas)
    assert out[0].tolist() == feas[0].tolist()          # non-empty row untouched
    assert out[1, M.XX_INDEX] == False                  # XX never feasible
    assert out[1, :M.XX_INDEX].all()                    # the rest filled True


# --- exploitability game ---------------------------------------------------------------

def test_equilibrium_value_matching_pennies_exact():
    """Matching pennies: value 0, equilibrium strategy (.5, .5) (exact via LP)."""
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    eq = M.equilibrium_value(payoff)
    assert eq["value"] == pytest.approx(0.0, abs=1e-9)
    assert eq["strategy"] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert eq["status"] == "optimal"


def test_exploitability_dominated_positive_equilibrium_zero():
    """A dominated (pure) policy has positive exploitability; the equilibrium policy ~0."""
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    v = M.equilibrium_value(payoff)["value"]
    assert M.policy_exploitability([1.0, 0.0], payoff, v) == pytest.approx(1.0, abs=1e-6)
    assert M.policy_exploitability([0.0, 1.0], payoff, v) == pytest.approx(1.0, abs=1e-6)
    assert M.policy_exploitability([0.5, 0.5], payoff, v) == pytest.approx(0.0, abs=1e-9)


def test_equilibrium_asymmetric_game():
    """A non-symmetric 2x2 game: the LP value equals the analytic mixed-strategy value."""
    # R = [[3,1],[0,2]] : pitcher maximiser. Mixed value = 3/2 (x=(.5,.5) gives col mins [1.5,1.5]).
    payoff = np.array([[3.0, 1.0], [0.0, 2.0]])
    eq = M.equilibrium_value(payoff)
    assert eq["value"] == pytest.approx(1.5, abs=1e-6)
    # A pure "always row 0" is exploitable: batter picks column 1 -> value 1 < 1.5.
    assert M.policy_exploitability([1.0, 0.0], payoff, eq["value"]) == pytest.approx(0.5, abs=1e-6)


def test_payoff_by_count_anticipation_penalty():
    """payoff_by_count: base q̂ off the diagonal, q̂ - beta*swing on the matched (a==k) diagonal."""
    q = np.zeros((M.N_COUNT, 8)); q[0] = np.arange(8) * 0.01
    sw = np.zeros((M.N_COUNT, 8)); sw[0] = 0.5
    payoff = M.payoff_by_count(q, sw, beta=0.2)
    assert payoff.shape == (M.N_COUNT, 8, 8)
    # off-diagonal: base q̂(a)
    assert payoff[0, 3, 1] == pytest.approx(q[0, 3])
    # diagonal: q̂(a) - beta*swing(a)
    assert payoff[0, 3, 3] == pytest.approx(q[0, 3] - 0.2 * 0.5)


def test_payoff_matrices_from_grid_and_response():
    """payoff_matrices (the named D49 entry point) aggregates a row-level q-grid + the fitted response
    model to per-count 8x8 games, and returns each row's count for the exploitability pass."""
    rng = np.random.default_rng(0)
    n = 240
    tbl = pd.DataFrame({
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
        "family": pd.Categorical(rng.choice(list(M.FAMILIES[:7]), n), categories=list(M.FAMILIES)),
        "is_swing": rng.random(n) < 0.45, "pa_id": np.arange(n), "pitch_number": 1,
        "exec_plate_z_norm": rng.random(n),
    })
    q = rng.normal(0, 0.05, size=(n, 8))
    resp = M.BatterResponseModel().fit(tbl)
    payoffs, counts = M.payoff_matrices(tbl, q, resp, beta=0.1)
    assert payoffs.shape == (M.N_COUNT, 8, 8)
    assert counts.shape == (n,) and int(counts.max()) < M.N_COUNT
    ex = M.exploitability(np.full((n, 8), 1.0 / 8), payoffs, counts)
    assert ex["mean"] >= 0 and ex["per_row"].shape == (n,)


def test_exploitability_aggregate_per_count():
    """exploitability aggregates per-row over each row's own count game; a concentrated policy is
    more exploitable than a uniform one."""
    rng = np.random.default_rng(0)
    q = rng.normal(0, 0.05, size=(M.N_COUNT, 8))
    sw = np.clip(rng.uniform(0.2, 0.7, size=(M.N_COUNT, 8)), 0, 1)
    payoffs = M.payoff_by_count(q, sw, beta=0.15)
    counts = rng.integers(0, M.N_COUNT, size=300)
    uniform = np.full((300, 8), 1.0 / 8)
    concentrated = np.zeros((300, 8)); concentrated[:, 0] = 1.0  # always FF
    ex_u = M.exploitability(uniform, payoffs, counts)
    ex_c = M.exploitability(concentrated, payoffs, counts)
    assert ex_u["mean"] >= 0 and ex_c["mean"] >= 0
    assert ex_c["mean"] > ex_u["mean"]        # predictable -> more exploitable
    assert ex_u["per_row"].shape == (300,)


# --- frontier assembly + figure --------------------------------------------------------

def test_frontier_assembly_schema_and_figure():
    """assemble_frontier yields the fixed schema; the figure renders to a PNG (Agg, file exists)."""
    entries = [
        dict(policy_id="behavior", ope_value=0.010, ope_lower95=0.004, b_seq_bits=0.0,
             exploitability=0.02, tv_from_behavior=0.0, params=0, wall_clock_s=0.0),
        dict(policy_id="fqi_O@1", ope_value=0.030, ope_lower95=-0.005, b_seq_bits=0.05,
             exploitability=0.08, tv_from_behavior=0.4, params=6000, wall_clock_s=12.0),
        dict(policy_id="fqi_count@1", ope_value=0.025, ope_lower95=0.0, b_seq_bits=0.0,
             exploitability=0.06, tv_from_behavior=0.35, params=96, wall_clock_s=1.0),
    ]
    df = M.assemble_frontier(entries)
    assert list(df.columns)[:len(M.FRONTIER_COLUMNS)] == M.FRONTIER_COLUMNS
    assert len(df) == 3
    d = Path(tempfile.mkdtemp(prefix="ws7_frontier_"))
    csv = M.write_frontier_csv(df, d / "frontier.csv")
    png = M.plot_frontier(df, d / "frontier.png", title="test frontier")
    assert csv.is_file() and png.is_file()
    assert png.stat().st_size > 1000  # a real PNG was written


def test_frontier_fills_missing_columns():
    """Missing frontier columns are filled with NaN (a cross-workstream row need not carry all)."""
    df = M.assemble_frontier([dict(policy_id="ws5_trigger", ope_value=0.02)])
    assert set(M.FRONTIER_COLUMNS).issubset(df.columns)
    assert np.isnan(df.loc[0, "exploitability"])
