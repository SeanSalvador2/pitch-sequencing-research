"""WS2 variable-order Markov grammar unit tests (SPEC 12.2; decisions D29-D31).

These pin the Dirichlet backoff mathematics against hand computations, the concentration
fitting against data simulated from the model, the suffix-chain backoff (unseen deep context
-> longest seen suffix, not global), order sensitivity (AB != BA), the ``K_MAX`` cap, the
effective-order read-out on a pure order-2 rule, the top-motif direction, and the D29
not-applicable views. Everything runs on tiny handcrafted frames where the truth is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.families import FAMILIES
from workstreams.ws2_bayes_markov import model as W

_FAMI = {f: i for i, f in enumerate(FAMILIES)}


# --- tiny table factory ----------------------------------------------------------------

def _rows_to_table(rows: list[dict]) -> pd.DataFrame:
    """Minimal decision-table columns the WS2 grammar needs."""
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], categories=list(FAMILIES))
    df["pa_id"] = df["game_pk"] * 1000 + df["at_bat_number"]
    df["row_id"] = (
        df["game_pk"].astype(str) + "_" + df["at_bat_number"].astype(str) + "_" + df["pitch_number"].astype(str)
    )
    for col, default in (("stand", "R"), ("p_throws", "R"), ("balls", 0), ("strikes", 0),
                         ("pitcher", 1), ("batter", 2), ("R", 0.0)):
        if col not in df.columns:
            df[col] = default
    return df


def _pa(fams, pa=1, pitcher=1, balls=0, strikes=0, game_pk=1):
    """One PA's rows from a family sequence (pitch_number 1..len)."""
    return [
        dict(game_pk=game_pk, at_bat_number=pa, pitch_number=i + 1, family=f, pitcher=pitcher,
             balls=balls, strikes=strikes, R=0.0)
        for i, f in enumerate(fams)
    ]


# --- Dirichlet shrinkage math ----------------------------------------------------------

def test_dirichlet_posterior_zero_counts_is_parent_exactly():
    """An unseen context (zero counts) backs off to its suffix parent exactly."""
    parent = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
    post = W.dirichlet_posterior_mean(np.zeros(8), parent, alpha=7.0)
    assert np.allclose(post, parent, atol=1e-15)


def test_dirichlet_posterior_hand_computed_convex_combo():
    counts = np.zeros(8)
    counts[0], counts[1] = 8.0, 2.0
    parent = np.full(8, 1.0 / 8)
    post = W.dirichlet_posterior_mean(counts, parent, alpha=10.0)
    expected = (counts + 10.0 * parent) / (10.0 + 10.0)
    assert np.allclose(post, expected)
    assert np.isclose(post.sum(), 1.0)
    assert 0.125 < post[0] < 0.8  # between the parent (0.125) and the raw freq (0.8)


def test_dirichlet_posterior_large_counts_approach_empirical():
    counts = np.zeros(8)
    counts[3], counts[4] = 9000.0, 1000.0
    parent = np.full(8, 1.0 / 8)
    post = W.dirichlet_posterior_mean(counts, parent, alpha=8.0)
    assert abs(post[3] - 0.9) < 1e-3 and abs(post[4] - 0.1) < 1e-3


def test_fit_dirichlet_concentration_recovers_known_alpha():
    """Simulate cells from ``Dir(alpha0 * parent) -> Multinomial`` and recover ``alpha0``."""
    rng = np.random.default_rng(0)
    alpha0 = 25.0
    parent = np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04])
    n_cells = 600
    counts = np.zeros((n_cells, 8))
    parents = np.tile(parent, (n_cells, 1))
    for c in range(n_cells):
        p = rng.dirichlet(alpha0 * parent)
        N = int(rng.integers(40, 120))
        counts[c] = rng.multinomial(N, p)
    alpha_hat, method = W.fit_dirichlet_concentration(counts, parents)
    assert method == "mml"
    assert 0.6 * alpha0 < alpha_hat < 1.6 * alpha0


def test_fit_dirichlet_single_cell_falls_back_to_default():
    counts = np.zeros((1, 8))
    counts[0, 0] = 5.0
    parents = np.full((1, 8), 1.0 / 8)
    _, method = W.fit_dirichlet_concentration(counts, parents)
    assert method == "default"


# --- encodings -------------------------------------------------------------------------

def test_lag_family_codes_within_pa():
    table = _rows_to_table(_pa(["FF", "SL", "CH"]))
    lag = W.lag_family_codes(table, 3)
    order = np.argsort(table["pitch_number"].to_numpy())
    lag = lag[order]
    # pitch 1: no priors.
    assert lag[0, 0] == W._NONE_CODE and lag[0, 1] == W._NONE_CODE
    # pitch 2: lag1 = FF, no lag2.
    assert lag[1, 0] == _FAMI["FF"] and lag[1, 1] == W._NONE_CODE
    # pitch 3: lag1 = SL (prev1), lag2 = FF (prev2).
    assert lag[2, 0] == _FAMI["SL"] and lag[2, 1] == _FAMI["FF"]


def test_count_hand_code_is_fixed_and_distinct():
    a = _rows_to_table(_pa(["FF"], balls=1, strikes=2))
    b = _rows_to_table(_pa(["FF"], balls=1, strikes=2))
    c = _rows_to_table([dict(game_pk=1, at_bat_number=1, pitch_number=1, family="FF",
                             pitcher=1, balls=0, strikes=0, R=0.0)])
    assert W.count_hand_code(a)[0] == W.count_hand_code(b)[0]  # data-independent
    assert W.count_hand_code(a)[0] != W.count_hand_code(c)[0]  # distinct cells differ


# --- view mapping + not-applicable views (D29) -----------------------------------------

def test_view_max_depth_mapping():
    assert W.VarOrderMarkov("C").max_depth == 0
    assert W.VarOrderMarkov("L1").max_depth == 1
    assert W.VarOrderMarkov("O", k_max=4).max_depth == 4
    assert W.VarOrderMarkov("O", k_max=2).max_depth == 2  # K_MAX cap is configurable


def test_U_and_OM_raise_notimplemented_with_note():
    table = _rows_to_table(_pa(["FF", "SL"]))
    for view in ("U", "OM"):
        with pytest.raises(NotImplementedError):
            W.fit(table, view)
        note = W.not_applicable_note(view)
        assert note["applicable"] is False and "not applicable" in note["message"]


# --- predict_proba validity ------------------------------------------------------------

def test_predict_proba_rows_sum_to_one_all_views():
    rows = []
    for i in range(40):
        rows += _pa(["FF", "SL", "CH", "FF"], pa=i + 1, pitcher=(i % 2))
    train = _rows_to_table(rows)
    for view in W.GRAMMAR_VIEWS:
        proba = W.fit(train, view).predict_proba(train)
        assert proba.shape == (len(train), len(FAMILIES))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(proba >= 0)


def test_C_view_ignores_ordered_history():
    """The C view is the base rate; it must not depend on the ordered context."""
    rows = []
    for i in range(30):
        rows += _pa(["FF", "SL", "CH"], pa=i + 1, pitcher=1)
    train = _rows_to_table(rows)
    model = W.fit(train, "C")
    base = model.predict_proba(train)
    # Permuting the within-PA histories cannot change a C prediction (it uses no lags).
    lag = W.lag_family_codes(train, 1)
    ch = W.count_hand_code(train)
    pn = train["pitch_number"].to_numpy()
    permuted = W.permute_contexts_within_strata(lag, ch, pn, seed=3)
    assert np.allclose(base, model.predict_proba(train, lag_codes=permuted))


# --- backoff / order structure ---------------------------------------------------------

def test_large_count_context_recovers_empirical_transition():
    """A heavily observed depth-1 context concentrates on the family that actually follows."""
    rows = []
    for i in range(500):
        rows += _pa(["FF", "SL"], pa=i + 1, pitcher=1)  # FF is always followed by SL
    train = _rows_to_table(rows)
    model = W.fit(train, "O")
    depth1 = model._by_name["depth1"]
    cell = depth1.keys[(1, _FAMI["FF"])]  # (pitcher=1, prev1=FF)
    assert depth1.post_mean[cell][_FAMI["SL"]] > 0.95  # alpha negligible vs N -> ~empirical


def test_order_sensitivity_AB_differs_from_BA():
    """(A then B) and (B then A) are different depth-2 contexts with different predictions."""
    rows = []
    # After (FF, SL) [prev2=FF, prev1=SL] the next is CH; after (SL, FF) the next is CU.
    for i in range(300):
        rows += _pa(["FF", "SL", "CH"], pa=2 * i + 1, pitcher=1)
        rows += _pa(["SL", "FF", "CU"], pa=2 * i + 2, pitcher=1)
    train = _rows_to_table(rows)
    model = W.fit(train, "O")
    q_ab = model.predict_proba(_rows_to_table(_pa(["FF", "SL", "CH"], pa=99, pitcher=1)))[2]
    q_ba = model.predict_proba(_rows_to_table(_pa(["SL", "FF", "CU"], pa=98, pitcher=1)))[2]
    assert not np.allclose(q_ab, q_ba)
    # Each ordered context lifts its own follower above the other's.
    assert q_ab[_FAMI["CH"]] > q_ba[_FAMI["CH"]]
    assert q_ba[_FAMI["CU"]] > q_ab[_FAMI["CU"]]


def test_unseen_deep_context_backs_off_to_longest_seen_suffix():
    """An unseen depth-3 context resolves to its seen depth-2 suffix, not to the base."""
    rows = []
    # Depth-2 context (prev2=FF, prev1=SL) is well seen; depth-3 (CH, FF, SL) never trained.
    for i in range(200):
        rows += _pa(["FF", "SL", "CU"], pa=i + 1, pitcher=1)
    train = _rows_to_table(rows)
    model = W.fit(train, "O")
    probe = _rows_to_table(_pa(["CH", "FF", "SL", "CU"], pa=1, pitcher=1))  # pitch 4 has ctx (SL,FF,CH)
    stats = model.posterior_stats(probe)
    # Pitch 4 (index 3): depth-3 context (SL, FF, CH) unseen -> longest seen suffix is (SL, FF)
    # at depth 2, so the resolved ordered depth is 2 (not 0/base).
    assert stats["resolved_depth"][3] == 2


def test_k_max_cap_limits_grammar_depth():
    rows = []
    for i in range(60):
        rows += _pa(["FF", "SL", "CH", "CU", "FS"], pa=i + 1, pitcher=1)
    train = _rows_to_table(rows)
    model = W.fit(train, "O", k_max=2)
    assert model.max_depth == 2
    assert "depth3" not in model._by_name and "depth2" in model._by_name
    # A 4-prior row can resolve at most to depth 2.
    stats = model.posterior_stats(train)
    assert stats["resolved_depth"].max() <= 2


# --- effective order + motifs on a pure order-2 rule -----------------------------------

def _order2_world(n=400, seed=0):
    """A pure order-2 rule: after a same-family run (X, X) the family X is suppressed.

    Each PA is length 3. The first two pitches are a random family; the third is X again with
    low probability (suppression) or another family, so ``(X, X) -> X`` drops sharply while
    ``(X, Y) -> `` is neutral. Only depth 2 resolves the rule.
    """
    rng = np.random.default_rng(seed)
    fams = ["FF", "SL", "CH", "CU"]
    rows = []
    for i in range(n):
        a = fams[rng.integers(4)]
        b = a if rng.random() < 0.5 else fams[rng.integers(4)]
        if a == b:  # same-family run -> strongly avoid a third
            third = a if rng.random() < 0.05 else fams[rng.integers(4)]
        else:
            third = fams[rng.integers(4)]
        rows += _pa([a, b, third], pa=i + 1, pitcher=1)
    return _rows_to_table(rows)


def test_effective_order_concentrates_at_two_on_order2_rule():
    train = _order2_world(n=600, seed=1)
    model = W.fit(train, "O")
    eff = model.effective_order(train)
    # Rows with two priors (pitch 3) should overwhelmingly earn effective order 2; the
    # pitch-1/2 rows cannot (no depth-2 context), so the overall >=2 mass is material.
    pn3 = train["pitch_number"].to_numpy() == 3
    order3 = eff["per_row"][pn3]
    assert np.mean(order3 == 2) > 0.6
    assert eff["frac_ge2"] > 0.2


def test_top_motifs_surface_repeat_suppression_with_correct_direction():
    train = _order2_world(n=600, seed=2)
    model = W.fit(train, "O")
    motifs = model.top_motifs(min_support=20, top_n=10)
    same_run = [m for m in motifs if m["depth"] == 2 and len(set(m["context"])) == 1]
    assert same_run, "expected a same-family-run motif in the top motifs"
    top = same_run[0]
    # The repeated family is suppressed (its probability drops vs the suffix).
    assert top["direction"] == "suppress"
    assert top["top_family"] == top["context"][-1]
    assert top["p_ctx"] < top["p_suffix"]


# --- calibrated uncertainty ------------------------------------------------------------

def test_posterior_variance_smaller_for_better_supported_context():
    rows = []
    # A heavily supported depth-1 context (FF->) and a sparsely supported one (CU->).
    for i in range(400):
        rows += _pa(["FF", "SL"], pa=i + 1, pitcher=1)
    for i in range(3):
        rows += _pa(["CU", "CH"], pa=1000 + i, pitcher=1)
    train = _rows_to_table(rows)
    model = W.fit(train, "L1")
    stats = model.posterior_stats(train)
    lag1 = W.lag_family_codes(train, 1)[:, 0]
    heavy = stats["variance"][lag1 == _FAMI["FF"]].mean()
    sparse = stats["variance"][lag1 == _FAMI["CU"]].mean()
    assert heavy < sparse
    assert np.all(stats["ess"] > 0)
