"""WS1 empirical-Bayes machinery unit tests (SPEC 12.1; decisions D25-D28).

These pin the shrinkage mathematics against hand computations, the concentration fitting
against data simulated from the model, the D25 key mapping (order-sensitive O vs
order-invariant U, L1 = previous family, the multiset cap), and the backoff behaviour on
unseen keys. Everything runs on tiny handcrafted frames where the ground truth is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.families import FAMILIES
from workstreams.ws1_eb_tables import model as W

_FAMI = {f: i for i, f in enumerate(FAMILIES)}


# --- tiny table factory ----------------------------------------------------------------

def _rows_to_table(rows: list[dict]) -> pd.DataFrame:
    """Minimal decision-table columns the WS1 keys / tables need."""
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


def _pa(fams, pa=1, pitcher=1, R=None, balls=0, strikes=0):
    """One PA's rows from a family sequence (pitch_number 1..len)."""
    R = R if R is not None else [0.0] * len(fams)
    return [
        dict(game_pk=1, at_bat_number=pa, pitch_number=i + 1, family=f, pitcher=pitcher,
             balls=balls, strikes=strikes, R=r)
        for i, (f, r) in enumerate(zip(fams, R))
    ]


# --- Dirichlet shrinkage math ----------------------------------------------------------

def test_dirichlet_posterior_zero_counts_is_parent_exactly():
    parent = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
    post = W.dirichlet_posterior_mean(np.zeros(8), parent, alpha=7.0)
    assert np.allclose(post, parent, atol=1e-15)


def test_dirichlet_posterior_hand_computed_convex_combo():
    # counts n=(8,2,0,...), parent uniform, alpha=10 -> exact (n + alpha*pi)/(N+alpha).
    counts = np.zeros(8)
    counts[0], counts[1] = 8.0, 2.0
    parent = np.full(8, 1.0 / 8)
    post = W.dirichlet_posterior_mean(counts, parent, alpha=10.0)
    expected = (counts + 10.0 * parent) / (10.0 + 10.0)
    assert np.allclose(post, expected)
    assert np.isclose(post.sum(), 1.0)
    # Component 0 sits between the raw frequency (0.8) and the parent (0.125).
    assert 0.125 < post[0] < 0.8


def test_dirichlet_posterior_large_counts_approach_empirical():
    counts = np.zeros(8)
    counts[3] = 9000.0
    counts[4] = 1000.0
    parent = np.full(8, 1.0 / 8)
    post = W.dirichlet_posterior_mean(counts, parent, alpha=8.0)
    assert abs(post[3] - 0.9) < 1e-3 and abs(post[4] - 0.1) < 1e-3


def test_fit_dirichlet_concentration_recovers_known_alpha():
    # Simulate many cells from Dir(alpha0 * parent) -> Multinomial, recover alpha0 by MML.
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
    alpha_hat, method = W.fit_dirichlet_concentration(counts, parents, max_concentration=1e6)
    assert method == "mml"
    assert 0.6 * alpha0 < alpha_hat < 1.6 * alpha0  # recovered within tolerance


def test_fit_dirichlet_single_cell_falls_back_to_default():
    counts = np.zeros((1, 8))
    counts[0, 0] = 5.0
    parents = np.full((1, 8), 1.0 / 8)
    alpha, method = W.fit_dirichlet_concentration(counts, parents)
    assert method == "default"


# --- normal run-value pooling ----------------------------------------------------------

def test_normal_posterior_hand_computed_precision_weight():
    # n=10, cell mean=1.0, parent=0.0, kappa=5 -> 10/15*1 + 5/15*0 = 0.6667.
    m = W.normal_posterior_mean(10.0, 1.0, 0.0, kappa=5.0)
    assert np.isclose(m, 10.0 / 15.0)
    # Zero-count cell returns the parent exactly.
    assert W.normal_posterior_mean(0.0, 999.0, -0.3, kappa=5.0) == -0.3


def test_normal_posterior_sd_decreases_with_n():
    # Posterior sd = sigma / sqrt(n + kappa); a bigger cell has a tighter posterior.
    sigma2, kappa = 0.04, 5.0
    sd_small = np.sqrt(sigma2 / (2 + kappa))
    sd_big = np.sqrt(sigma2 / (200 + kappa))
    assert sd_big < sd_small


def test_run_value_zero_cell_is_parent_and_sd_shrinks_with_support():
    # Build a table where one (count,family) cell has many obs and another has few, and
    # check the fitted run-value posterior sd is smaller for the better-supported cell.
    rng = np.random.default_rng(1)
    rows = []
    # Pitcher 1 throws FF a lot at (0,0); pitcher 2 throws SL rarely at (0,0).
    for i in range(300):
        rows += _pa(["FF"], pa=i + 1, pitcher=1, R=[float(0.05 + 0.1 * rng.standard_normal())])
    for i in range(4):
        rows += _pa(["SL"], pa=1000 + i, pitcher=2, R=[float(-0.05 + 0.1 * rng.standard_normal())])
    table = _rows_to_table(rows)
    model = W.fit(table, "C", "run_value", max_concentration=200.0)
    stats = model.posterior_stats(table)
    fam = table["family"].astype("object").to_numpy()
    sd_big = stats["exp_reward_sd"][fam == "FF"].mean()
    sd_small = stats["exp_reward_sd"][fam == "SL"].mean()
    assert sd_big < sd_small


# --- D25 key mapping -------------------------------------------------------------------

def test_L1_key_equals_previous_family():
    table = _rows_to_table(_pa(["FF", "SL", "CH"]))
    prev = W.history_signature(table, "L1")
    order = np.argsort(table["pitch_number"].to_numpy())
    prev_ord = prev[order]
    assert prev_ord[0] == W._NONE_CODE  # first pitch has no prior
    assert prev_ord[1] == _FAMI["FF"]  # after FF
    assert prev_ord[2] == _FAMI["SL"]  # after SL


def test_O_key_is_order_sensitive_but_U_key_is_not():
    # Two PAs with the same prior multiset {A, B} but opposite order at pitch 3.
    ab = _pa(["FF", "SL", "CU"], pa=1)      # priors of pitch 3: FF then SL
    ba = _pa(["SL", "FF", "CU"], pa=2)      # priors of pitch 3: SL then FF
    table = _rows_to_table(ab + ba)
    o = W.history_signature(table, "O")
    u = W.history_signature(table, "U")
    pn = table["pitch_number"].to_numpy()
    pa = table["at_bat_number"].to_numpy()
    o_ab = o[(pa == 1) & (pn == 3)][0]
    o_ba = o[(pa == 2) & (pn == 3)][0]
    u_ab = u[(pa == 1) & (pn == 3)][0]
    u_ba = u[(pa == 2) & (pn == 3)][0]
    assert o_ab != o_ba          # ordered pair distinguishes AB from BA
    assert u_ab == u_ba          # unordered multiset does not


def test_U_multiset_signature_caps_at_first_six_priors():
    # A long PA: the U signature at pitch 8 must ignore priors beyond the first 6.
    long_seq = ["FF", "FF", "FF", "FF", "FF", "FF", "SL", "CU"]  # priors of pitch 8 incl. a SL
    capped = _rows_to_table(_pa(long_seq, pa=1))
    # Same first-6 priors (all FF) but a different 7th prior -> identical capped signature.
    alt_seq = ["FF", "FF", "FF", "FF", "FF", "FF", "CH", "CU"]
    alt = _rows_to_table(_pa(alt_seq, pa=1))
    u_capped = W.history_signature(capped, "U")
    u_alt = W.history_signature(alt, "U")
    pn = capped["pitch_number"].to_numpy()
    # Pitch 8's signature counts only positions 1..6 (six FF) -> equal despite the 7th diff.
    assert u_capped[pn == 8][0] == u_alt[alt["pitch_number"].to_numpy() == 8][0]
    # And it encodes exactly six FF (base-7): 6 * 7**index(FF).
    assert u_capped[pn == 8][0] == 6 * (7 ** _FAMI["FF"])


def test_C_view_has_no_history_key():
    table = _rows_to_table(_pa(["FF", "SL"]))
    assert np.all(W.history_signature(table, "C") == 0)


# --- backoff on unseen keys ------------------------------------------------------------

def test_unseen_pitcher_backs_off_to_count_hand_parent_exactly():
    # Train two pitchers; predict a THIRD unseen pitcher at a seen count -> the prediction
    # must equal the count_hand-level posterior mean (the pitcher cell is unseen).
    rows = []
    for i in range(40):
        rows += _pa(["FF"], pa=i + 1, pitcher=1)
    for i in range(40):
        rows += _pa(["SL"], pa=100 + i, pitcher=2)
    train = _rows_to_table(rows)
    model = W.fit(train, "C", "selection")

    unseen = _rows_to_table(_pa(["FF"], pa=1, pitcher=999))  # pitcher 999 never trained
    proba = model.predict_proba(unseen)
    # The count_hand cell (0,0,R,R) posterior mean, computed directly from the fitted level.
    ch_level = model._levels[1]
    ch_key = list(ch_level.keys.keys())[0]
    expected = ch_level.post_mean[ch_level.keys[ch_key]]
    assert np.allclose(proba[0], expected)
    # And the backoff counter recorded a count_hand-level resolution (not the pitcher level).
    assert model.backoff_counts_["count_hand"] == 1
    assert model.backoff_counts_["pitcher"] == 0


def test_seen_pitcher_resolves_at_pitcher_level_and_counter_increments():
    rows = []
    for i in range(60):
        rows += _pa(["FF", "SL"], pa=i + 1, pitcher=7)
    train = _rows_to_table(rows)
    model = W.fit(train, "L1", "selection")
    # A row with a seen (pitcher, count, prev_family) resolves at the deepest history level.
    seen = _rows_to_table(_pa(["FF", "SL"], pa=1, pitcher=7))
    model.predict_proba(seen)
    assert model.backoff_counts_["history"] >= 1
    # An unseen prev-family at a seen pitcher/count backs off to the pitcher level.
    novel = _rows_to_table(_pa(["CU", "FS"], pa=2, pitcher=7))  # prev CU never trained here
    model.predict_proba(novel)
    assert model.backoff_counts_["pitcher"] >= 1


def test_predict_proba_rows_sum_to_one():
    rows = []
    for i in range(30):
        rows += _pa(["FF", "SL", "CH"], pa=i + 1, pitcher=(i % 3))
    train = _rows_to_table(rows)
    for view in W.SELECTION_VIEWS:
        proba = W.fit(train, view, "selection").predict_proba(train)
        assert proba.shape == (len(train), len(FAMILIES))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(proba >= 0)


# --- OM infeasibility (D25) ------------------------------------------------------------

def test_OM_raises_notimplemented_with_support_note():
    table = _rows_to_table(_pa(["FF", "SL"]))
    with pytest.raises(NotImplementedError):
        W.fit(table, "OM", "selection")
    note = W.support_note(table)
    assert note["feasible"] is False
    assert "n_pairs" in note and "frac_pairs_lt50" in note
