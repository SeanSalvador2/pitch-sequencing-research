"""Reward sign convention and episode-return indexing (SPEC 5)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_raw
from pitchseq.decision_table import build_decision_table
from pitchseq.reward import add_reward, episode_returns, sign_check


def test_sign_check_passes_on_sample(sample_raw):
    r = add_reward(sample_raw)
    means = sign_check(r)
    assert means["called_strike"] > 0
    assert means["swinging_strike"] > 0
    assert means["strikeout"] > 0
    assert means["ball"] < 0
    assert means["walk"] < 0
    assert means["home_run"] < 0


def test_sign_check_raises_on_flipped_reward(sample_raw):
    r = add_reward(sample_raw)
    r["R"] = -r["R"]  # deliberately invert the convention
    with pytest.raises(ValueError, match="Reward sign check failed"):
        sign_check(r)


def test_episode_returns_and_agreement():
    rows = [
        # PA 1: three pitches.
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 1, "delta_run_exp": -0.02},
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 2, "delta_run_exp": -0.03},
        {"game_pk": 1, "at_bat_number": 1, "pitch_number": 3, "delta_run_exp": 0.15},
        # PA 2: two pitches.
        {"game_pk": 1, "at_bat_number": 2, "pitch_number": 1, "delta_run_exp": -0.05},
        {"game_pk": 1, "at_bat_number": 2, "pitch_number": 2, "delta_run_exp": 0.20},
    ]
    table = build_decision_table(make_raw(rows))
    g = episode_returns(table, assert_agreement=True)
    assert g.loc[1001] == pytest.approx(-(-0.02 - 0.03 + 0.15))  # = -0.10
    assert g.loc[1002] == pytest.approx(-(-0.05 + 0.20))  # = -0.15


def test_episode_returns_order_independent():
    rows = [
        {"game_pk": 7, "at_bat_number": 1, "pitch_number": 1, "delta_run_exp": -0.02},
        {"game_pk": 7, "at_bat_number": 1, "pitch_number": 2, "delta_run_exp": 0.10},
        {"game_pk": 7, "at_bat_number": 1, "pitch_number": 3, "delta_run_exp": -0.30},
    ]
    table = build_decision_table(make_raw(rows))
    # Shuffle rows; the agreement guard sorts by pitch_number internally.
    shuffled = table.sample(frac=1.0, random_state=1)
    g = episode_returns(shuffled, assert_agreement=True)
    assert g.loc[7001] == pytest.approx(-(-0.02 + 0.10 - 0.30))


def test_episode_returns_nan_last_pitch_no_false_alarm():
    # delta_run_exp is null on ~0.3% of pitches. A PA whose LAST pitch has a missing
    # reward must not trip the agreement guard (the guard checks indexing, not NaN
    # policy): the return keeps the partial sum, and an all-NaN PA stays NaN (min_count).
    rows = [
        {"game_pk": 9, "at_bat_number": 1, "pitch_number": 1, "delta_run_exp": -0.10},
        {"game_pk": 9, "at_bat_number": 1, "pitch_number": 2, "delta_run_exp": -0.20},
        {"game_pk": 9, "at_bat_number": 1, "pitch_number": 3, "delta_run_exp": None},
        {"game_pk": 9, "at_bat_number": 2, "pitch_number": 1, "delta_run_exp": None},
    ]
    table = build_decision_table(make_raw(rows))
    g = episode_returns(table, assert_agreement=True)  # must not raise
    assert g.loc[9001] == pytest.approx(0.30)  # -(-0.10 - 0.20), NaN skipped
    assert np.isnan(g.loc[9002])  # all-NaN PA has no defined return
