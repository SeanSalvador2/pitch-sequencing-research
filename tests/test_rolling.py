"""Leakage-safe rolling repertoire / tendency features (SPEC 3.2 / 0)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_raw
from pitchseq.decision_table import build_decision_table
from pitchseq.families import FAMILIES
from pitchseq.rolling import repertoire_mix_columns


def _three_game_frame():
    rows = []
    # Game 1 (2023-04-01): 3 FF @95 + 1 SL @85, thrown by pitcher 100 to batter 200.
    g1 = [
        # FF#1: swinging strike, in-zone -> swing + whiff
        {"pitch_type": "FF", "release_speed": 95.0, "pfx_x": 1.0, "pfx_z": 1.5, "zone": 5, "description": "swinging_strike"},
        # FF#2: ball, out-of-zone -> no swing (an out-of-zone take)
        {"pitch_type": "FF", "release_speed": 95.0, "pfx_x": 1.0, "pfx_z": 1.5, "zone": 13, "description": "ball"},
        # FF#3: foul, in-zone -> swing, no whiff
        {"pitch_type": "FF", "release_speed": 95.0, "pfx_x": 1.0, "pfx_z": 1.5, "zone": 6, "description": "foul"},
        # SL#1: in play (single), in-zone -> swing, in-play
        {"pitch_type": "SL", "release_speed": 85.0, "pfx_x": -1.0, "pfx_z": 0.5, "zone": 4, "description": "hit_into_play", "events": "single"},
    ]
    for i, r in enumerate(g1):
        r.update({"game_pk": 1, "game_date": "2023-04-01", "at_bat_number": i + 1, "pitch_number": 1})
        rows.append(r)
    # Game 2 (2023-04-08): 1 FF @97 (chase whiff, out-of-zone) + 1 CH @88 (called strike).
    g2 = [
        {"pitch_type": "FF", "release_speed": 97.0, "pfx_x": 1.2, "pfx_z": 1.6, "zone": 12, "description": "swinging_strike"},
        {"pitch_type": "CH", "release_speed": 88.0, "pfx_x": 0.8, "pfx_z": 0.9, "zone": 5, "description": "called_strike"},
    ]
    for i, r in enumerate(g2):
        r.update({"game_pk": 2, "game_date": "2023-04-08", "at_bat_number": i + 1, "pitch_number": 1})
        rows.append(r)
    # Game 2b (2023-04-15, doubleheader, SAME date as the decision game): 1 SI. Must be
    # excluded from the decision game's trailing window (strictly-earlier game_date).
    rows.append(
        {"pitch_type": "SI", "release_speed": 90.0, "pfx_x": 1.5, "pfx_z": 1.0, "zone": 5, "description": "ball",
         "game_pk": 4, "game_date": "2023-04-15", "at_bat_number": 1, "pitch_number": 1}
    )
    # Game 3 (2023-04-15): the decision pitch.
    rows.append(
        {"pitch_type": "FF", "release_speed": 96.0, "pfx_x": 1.1, "pfx_z": 1.4, "zone": 5, "description": "ball",
         "game_pk": 3, "game_date": "2023-04-15", "at_bat_number": 1, "pitch_number": 1}
    )
    return make_raw(rows)


def test_pitcher_repertoire_and_baselines_exact():
    table = build_decision_table(_three_game_frame())
    g3 = table[table["game_pk"] == 3].iloc[0]

    # Trailing window = games 1 (04-01) + 2 (04-08) only; the 04-15 doubleheader is excluded.
    assert g3["pitcher_n_trailing_pitches"] == pytest.approx(6.0)
    assert g3["repertoire_mix_FF"] == pytest.approx(4 / 6)
    assert g3["repertoire_mix_SL"] == pytest.approx(1 / 6)
    assert g3["repertoire_mix_CH"] == pytest.approx(1 / 6)
    assert g3["repertoire_mix_SI"] == pytest.approx(0.0)  # same-day SI excluded
    # Repertoire mix sums to 1 when history exists.
    assert sum(g3[c] for c in repertoire_mix_columns()) == pytest.approx(1.0)

    # Per-family execution baselines are means over the trailing family pitches.
    assert g3["pitcher_base_FF_release_speed"] == pytest.approx((95 + 95 + 95 + 97) / 4)
    assert g3["pitcher_base_FF_pfx_x"] == pytest.approx((1.0 + 1.0 + 1.0 + 1.2) / 4)
    assert g3["pitcher_base_FF_pfx_z"] == pytest.approx((1.5 + 1.5 + 1.5 + 1.6) / 4)
    assert g3["pitcher_base_SL_release_speed"] == pytest.approx(85.0)
    assert g3["pitcher_base_CH_release_speed"] == pytest.approx(88.0)
    assert np.isnan(g3["pitcher_base_SI_release_speed"])  # no SI in the window
    assert bool(g3["pitcher_low_history"]) is True  # 6 < 100


def test_batter_tendencies_exact():
    table = build_decision_table(_three_game_frame())
    g3 = table[table["game_pk"] == 3].iloc[0]

    # Trailing = 6 pitches (games 1 + 2). swings: FF#1, FF#3, SL#1, FF(g2) = 4.
    assert g3["batter_n_trailing_pitches"] == pytest.approx(6.0)
    assert g3["batter_tend_swing_rate"] == pytest.approx(4 / 6)
    # whiffs among swings: FF#1, FF(g2) = 2 of 4.
    assert g3["batter_tend_whiff_rate"] == pytest.approx(2 / 4)
    # out-of-zone pitches: FF#2 (take) + FF(g2) (swing) = 2; chase = the one swing.
    assert g3["batter_tend_chase_rate"] == pytest.approx(1 / 2)
    # in play: SL#1 = 1 of 6.
    assert g3["batter_tend_inplay_rate"] == pytest.approx(1 / 6)


def test_first_game_has_no_history():
    table = build_decision_table(_three_game_frame())
    g1 = table[table["game_pk"] == 1].iloc[0]
    assert g1["pitcher_n_trailing_pitches"] == pytest.approx(0.0)
    assert np.isnan(g1["repertoire_mix_FF"])  # NaN-safe when no history exists
    assert bool(g1["pitcher_low_history"]) is True
    assert np.isnan(g1["batter_tend_swing_rate"])
