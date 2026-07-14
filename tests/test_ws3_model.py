"""WS3 GBDT model unit tests (SPEC 12.3; decisions D32-D35).

These pin the model-level contracts on small, fast, deterministic inputs: behavior
probabilities are valid on every view and carry no execution leakage; the decomposed
outcome assembly reproduces a hand-computed ``E[R]`` from known node probabilities and
values; the count-conditional node table falls back to the global node when a cell is
empty; the direct regressor agrees with the decomposition within tolerance; the q-grid is
well shaped and finite; swapping the action changes the prediction (counterfactual
sensitivity); and a save / load round-trip reproduces predictions exactly. LightGBM runs
single-threaded and deterministic so the assertions are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.decision_table import EXECUTION_COLS, LABEL_COLS, build_decision_table
from pitchseq.families import FAMILIES
from pitchseq.outcomes import OUTCOME1, OUTCOME2
from pitchseq.states import build_view
from pitchseq.synth import make_null_world
from workstreams.ws3_gbdt_stack import model as M

# A small deterministic world shared across the model tests; fast LightGBM params.
_FAST = {"n_estimators": 50, "num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def null_table() -> pd.DataFrame:
    raw, _ = make_null_world(n_games=30, seed=7, innings_per_game=6)
    return build_decision_table(raw)


# --- behavior model --------------------------------------------------------------------

def test_behavior_probs_valid_all_views(null_table):
    """The behavior model returns a valid (n, 8) family distribution on every view."""
    for view in M.BEHAVIOR_VIEWS:
        bm = M.BehaviorModel(view).fit(null_table, params=_FAST)
        proba = bm.predict_proba(null_table)
        assert proba.shape == (len(null_table), len(FAMILIES))
        assert np.all(proba >= 0)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_behavior_unknown_view_raises():
    with pytest.raises(ValueError):
        M.BehaviorModel("ZZ")


def test_leakage_audit_passes_every_view_matrix(null_table):
    """No execution / label column appears in any behavior or outcome feature matrix."""
    banned = set(EXECUTION_COLS) | set(LABEL_COLS)
    for view in M.BEHAVIOR_VIEWS:
        X, _ = build_view(null_table, view)  # build_view audits internally; re-check here
        assert not (set(X.columns) & banned)
        assert not any(c.startswith("exec_") for c in X.columns)
        Xo = M.outcome_view_matrix(null_table, view, action="FF")
        # Only the current-action feature may be added on top of the audited view (D22).
        assert set(Xo.columns) - set(X.columns) == {M.ACTION_COL}
        assert not (set(Xo.columns) & banned)


# --- decomposed outcome: hand-computed assembly ----------------------------------------

def _controlled_node_table() -> M.NodeValueTable:
    """A NodeValueTable with a known reward per (outcome node, count=(0,0)) cell."""
    rows = []

    def add(o1, o2, R, n=20, b=0, s=0):
        rows.extend([dict(outcome1=o1, outcome2=o2, R=R, balls=b, strikes=s) for _ in range(n)])

    add("ball", None, 0.04); add("called_strike", None, -0.05); add("whiff", None, -0.06)
    add("foul", None, -0.02); add("hbp", None, 0.30)
    add("in_play", "single", 0.45); add("in_play", "double", 0.75); add("in_play", "triple", 1.0)
    add("in_play", "home_run", 1.4); add("in_play", "out_or_other", -0.25)
    tr = pd.DataFrame(rows)
    tr["outcome1"] = pd.Categorical(tr.outcome1, categories=list(OUTCOME1))
    tr["outcome2"] = pd.Categorical(tr.outcome2, categories=list(OUTCOME2))
    return M.NodeValueTable().fit(tr)


def test_node_values_recover_known_means():
    nv = _controlled_node_table()
    res = nv.resolve(np.array([0]), np.array([0]))
    v1 = dict(zip(OUTCOME1, res["v1"][0]))
    v2 = dict(zip(OUTCOME2, res["v2"][0]))
    assert np.isclose(v1["ball"], 0.04) and np.isclose(v1["called_strike"], -0.05)
    assert np.isclose(v1["whiff"], -0.06) and np.isclose(v1["hbp"], 0.30)
    assert np.isclose(v2["single"], 0.45) and np.isclose(v2["home_run"], 1.4)


def test_node_value_global_fallback_for_unseen_count():
    """An unseen count resolves to the global node vectors (no crash, sensible values)."""
    nv = _controlled_node_table()
    seen = nv.resolve(np.array([0]), np.array([0]))["v1"][0]
    unseen = nv.resolve(np.array([3]), np.array([2]))["v1"][0]  # count (3,2) never in train
    assert np.allclose(unseen, nv.v1_global)
    # With a single count in training, the global equals the (0,0) cell.
    assert np.allclose(seen, unseen)


def test_assembly_reproduces_hand_computed_expected_reward():
    r"""``E[R] = sum_k P_A(k) V1(k) + P_A(in_play) sum_j P_B(j) V2(j)`` on a known case."""
    nv = _controlled_node_table()
    stack = M.OutcomeStack.__new__(M.OutcomeStack)  # bypass fit; inject the controlled nodes
    stack.nodes = nv

    idx = {c: i for i, c in enumerate(OUTCOME1)}
    jdx = {c: i for i, c in enumerate(OUTCOME2)}
    p_a = np.zeros((1, len(OUTCOME1)))
    p_a[0, idx["ball"]] = 0.2; p_a[0, idx["called_strike"]] = 0.1; p_a[0, idx["whiff"]] = 0.05
    p_a[0, idx["foul"]] = 0.05; p_a[0, idx["in_play"]] = 0.6
    p_b = np.zeros((1, len(OUTCOME2)))
    p_b[0, jdx["single"]] = 0.5; p_b[0, jdx["out_or_other"]] = 0.5

    er, sd = stack._assemble(p_a, p_b, np.array([0]), np.array([0]))
    hand = (0.2 * 0.04 + 0.1 * (-0.05) + 0.05 * (-0.06) + 0.05 * (-0.02)
            + 0.6 * (0.5 * 0.45 + 0.5 * (-0.25)))
    assert np.isclose(er[0], hand)
    assert np.isfinite(sd[0]) and sd[0] >= 0


# --- decomposed outcome on a fitted stack ----------------------------------------------

@pytest.fixture(scope="module")
def outcome_stack(null_table) -> M.OutcomeStack:
    return M.OutcomeStack("O").fit(null_table, params=_FAST)


def test_outcome_probabilities_valid(outcome_stack, null_table):
    p1 = outcome_stack.predict_outcome1(null_table)
    p2 = outcome_stack.predict_outcome2(null_table)
    assert p1.shape == (len(null_table), len(OUTCOME1))
    assert p2.shape == (len(null_table), len(OUTCOME2))
    assert np.allclose(p1.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(p2.sum(axis=1), 1.0, atol=1e-9)


def test_exp_reward_and_sd_finite_and_nonneg(outcome_stack, null_table):
    er, sd = outcome_stack.exp_reward(null_table, with_sd=True)
    assert np.isfinite(er).all()
    assert np.isfinite(sd).all() and (sd >= 0).all()


def test_q_grid_shape_and_finite(outcome_stack, null_table):
    qg = M.q_grid(outcome_stack, null_table, "O")
    assert qg.shape == (len(null_table), len(FAMILIES))
    assert np.isfinite(qg).all()


def test_q_grid_view_mismatch_raises(outcome_stack, null_table):
    with pytest.raises(ValueError):
        M.q_grid(outcome_stack, null_table, "C")


def test_counterfactual_action_changes_prediction(outcome_stack, null_table):
    """Swapping the current action changes the outcome distribution and E[R] (D33)."""
    er_ff = outcome_stack.exp_reward(null_table, action="FF")
    er_sl = outcome_stack.exp_reward(null_table, action="SL")
    assert not np.allclose(er_ff, er_sl)
    p_ff = outcome_stack.predict_outcome1(null_table, action="FF")
    p_sl = outcome_stack.predict_outcome1(null_table, action="SL")
    assert not np.allclose(p_ff, p_sl)


def test_decomposed_vs_direct_within_tolerance(outcome_stack, null_table):
    """The D32 cross-check: decomposed and direct E[R] agree within the flag tolerance."""
    dis = outcome_stack.disagreement(null_table)
    assert np.isfinite(dis["mean_abs_diff"])
    assert dis["mean_abs_diff"] < 0.05  # comfortably below a material disagreement
    assert dis["flag"] is False


# --- persistence: save -> load -> identical predictions --------------------------------

def test_persistence_identical_predictions(null_table, tmp_path):
    """A save / load round-trip reproduces behavior, q-grid and E[R] predictions exactly."""
    bm = M.BehaviorModel("L1").fit(null_table, params=_FAST)
    os_ = M.OutcomeStack("L1").fit(null_table, params=_FAST)
    M.save_ws3_artifacts(tmp_path, behavior={"L1": bm}, outcome={"L1": os_})

    art = M.load_ws3_artifacts(tmp_path)
    assert art.views == ["L1"]
    assert np.array_equal(bm.predict_proba(null_table), art.propensities(null_table, "L1"))
    assert np.array_equal(os_.q_grid(null_table), art.q_grid(null_table, "L1"))
    assert np.array_equal(os_.exp_reward(null_table), art.exp_reward(null_table, "L1"))
    assert np.array_equal(os_.predict_outcome1(null_table),
                          art.outcome["L1"].predict_outcome1(null_table))


# --- falsification factory (the shared D17 interface) ----------------------------------

def test_ws3_model_factory_is_usable_classifier(null_table):
    """The falsification factory yields a fitted classifier with classes_ / predict_proba."""
    factory = M.make_ws3_model_factory(_FAST)
    X, _ = build_view(null_table, "C")
    y = null_table["family"].astype("object").to_numpy()
    clf = factory("C").fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape[0] == len(null_table)
    assert set(clf.classes_).issubset(set(FAMILIES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_propensities_match_behavior_predict(null_table):
    bm = M.BehaviorModel("C").fit(null_table, params=_FAST)
    assert np.array_equal(M.propensities(bm, null_table), bm.predict_proba(null_table))
