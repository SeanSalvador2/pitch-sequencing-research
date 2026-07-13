"""Synthetic null / positive worlds -- the WS0 correctness oracle (SPEC 11; D18, D20)."""

from __future__ import annotations

import numpy as np
import pytest

from pitchseq.decision_table import build_decision_table
from pitchseq.reward import add_reward, sign_check
from pitchseq.sequences import build_sequences
from pitchseq.states import STATE_VIEWS, build_view
from pitchseq.synth import make_null_world, make_positive_world


def _reconstruct_velo_gap(table):
    """|velo_{t-1} - velo_{t-2}| per row from realised prior-pitch speeds."""
    t = table.sort_values(["pa_id", "pitch_number"], kind="stable")
    g = t.groupby("pa_id", sort=False)["exec_release_speed"]
    gap = (g.shift(1) - g.shift(2)).abs()
    return gap.reindex(table.index).to_numpy()


@pytest.mark.parametrize("world_fn", [make_null_world, make_positive_world])
def test_world_builds_through_full_0a_pipeline(world_fn):
    raw, truth = world_fn(n_games=20, seed=11, innings_per_game=5)
    # Decision table + all five views + sequences build without error.
    table = build_decision_table(raw)
    assert len(table) == truth["n_pitches"] == len(raw)
    for v in STATE_VIEWS:
        X, meta = build_view(table, v)  # leakage_audit runs inside build_view
        assert len(X) == len(table)
        assert meta["view"] == v
    seq = build_sequences(table, max_len=15)
    assert seq["family_idx"].shape == (len(table), 15)
    assert np.array_equal(seq["lengths"], seq["mask"].sum(axis=1).astype(int))


@pytest.mark.parametrize("world_fn", [make_null_world, make_positive_world])
def test_reward_sign_check_passes(world_fn):
    raw, _ = world_fn(n_games=20, seed=5, innings_per_game=5)
    means = sign_check(add_reward(raw))
    # Strikes / strikeouts positive reward; ball / walk / home run negative.
    assert means["called_strike"] > 0
    assert means["swinging_strike"] > 0
    assert means["strikeout"] > 0
    assert means["ball"] < 0
    assert means["walk"] < 0
    assert means["home_run"] < 0


@pytest.mark.parametrize("world_fn", [make_null_world, make_positive_world])
def test_counts_evolve_legally_and_pa_terminates_once(world_fn):
    raw, _ = world_fn(n_games=15, seed=7, innings_per_game=5)
    table = build_decision_table(raw)
    # Pre-pitch counts are always legal.
    assert int(table["balls"].max()) <= 3
    assert int(table["strikes"].max()) <= 2
    assert int(table["balls"].min()) >= 0
    assert int(table["strikes"].min()) >= 0

    # Exactly one terminal pitch per PA (events non-null), and it is the last pitch.
    raw_sorted = raw.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable")
    grp = raw_sorted.groupby(["game_pk", "at_bat_number"], sort=False)
    n_terminal = grp["events"].apply(lambda s: s.notna().sum())
    assert (n_terminal == 1).all()
    # The terminal pitch is the max pitch_number in its PA.
    last_pn = grp["pitch_number"].max()
    term_pn = raw_sorted.loc[raw_sorted["events"].notna()].set_index(["game_pk", "at_bat_number"])["pitch_number"]
    assert (term_pn.sort_index() == last_pn.sort_index()).all()


def test_null_world_truth_has_no_effect():
    _, truth = make_null_world(n_games=10, seed=1, innings_per_game=5)
    assert truth["world"] == "null"
    assert truth["effect"] == 0.0
    assert truth["affected_rate"] == 0.0


def test_positive_world_truth_matches_data():
    thr = 5.0
    raw, truth = make_positive_world(
        n_games=40, seed=9, innings_per_game=6, effect_size=0.22, velo_gap_threshold=thr
    )
    assert truth["world"] == "positive"
    assert truth["effect"] == 0.22
    assert truth["threshold"] == thr

    table = build_decision_table(raw)
    gap = _reconstruct_velo_gap(table)
    trig = np.nan_to_num(gap) >= thr  # NaN (t<3) -> not triggered
    # affected_rate is recomputable from the built table.
    assert abs(trig.mean() - truth["affected_rate"]) < 0.02
    assert truth["affected_rate"] > 0.10  # a non-trivial fraction is affected

    # The whiff rate is genuinely higher on triggered pitches (planted finding #2).
    whiff = (table["outcome1"].astype("object").to_numpy() == "whiff")
    lift = whiff[trig].mean() - whiff[~trig].mean()
    assert lift > 0.10
    assert abs(lift - truth["empirical_whiff_lift"]) < 0.03
    # The realised boost is close to the nominal effect size.
    assert abs(truth["realized_boost"] - 0.22) < 0.05
