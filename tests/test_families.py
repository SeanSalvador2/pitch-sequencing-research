"""Family mapping and leakage-safe feasible-action masks (SPEC 4)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from conftest import make_raw
from pitchseq.config import load_config
from pitchseq.families import (
    FAMILIES,
    add_family,
    family_map_from_config,
    feasible_action_mask,
)


def test_families_constant_matches_config():
    cfg = load_config()
    assert tuple(cfg["pitch_families"].keys()) == FAMILIES


def test_every_sample_pitch_type_maps(sample_raw):
    fam = add_family(sample_raw)
    # No pitch is left unmapped (unknown/null collapse to XX, never NaN).
    assert fam["family"].notna().all()
    reverse = family_map_from_config()
    for pt in sample_raw["pitch_type"].dropna().unique():
        expected = reverse.get(str(pt), "XX")
        got = fam.loc[sample_raw["pitch_type"] == pt, "family"].astype("object").unique()
        assert list(got) == [expected], f"{pt} -> {got}, expected {expected}"


def test_unknown_and_null_map_to_xx():
    raw = make_raw([{"pitch_type": "ZZ"}, {"pitch_type": None}, {"pitch_type": "FF"}])
    fam = add_family(raw)["family"].astype("object").tolist()
    assert fam == ["XX", "XX", "FF"]


def _pitches(pitch_type, n, game_pk, game_date, start_ab=1):
    return [
        {
            "pitch_type": pitch_type,
            "game_pk": game_pk,
            "game_date": game_date,
            "at_bat_number": start_ab + i,
            "pitch_number": 1,
            "pitcher": 100,
        }
        for i in range(n)
    ]


def test_feasible_mask_thresholds_and_strictly_pre_game():
    # Custom thresholds so tiny synthetic data exercises both count and share gates.
    cfg = copy.deepcopy(load_config())
    cfg["action"]["feasible_mask"]["min_count"] = 3
    cfg["action"]["feasible_mask"]["min_share"] = 0.3
    cfg["rolling"]["min_history_pitches"] = 5

    rows = []
    # Game A (2023-04-01): FF x4, SL x3, CH x3, SI x1 -> 11 pitches.
    rows += _pitches("FF", 4, game_pk=1, game_date="2023-04-01", start_ab=1)
    rows += _pitches("SL", 3, game_pk=1, game_date="2023-04-01", start_ab=5)
    rows += _pitches("CH", 3, game_pk=1, game_date="2023-04-01", start_ab=8)
    rows += _pitches("SI", 1, game_pk=1, game_date="2023-04-01", start_ab=11)
    # Game B (2023-04-10): the decision game.
    rows += _pitches("FF", 2, game_pk=2, game_date="2023-04-10", start_ab=1)

    raw = make_raw(rows)
    fam = add_family(raw, cfg)
    mask = feasible_action_mask(fam, cfg)

    is_game_a = raw["game_pk"] == 1
    is_game_b = raw["game_pk"] == 2

    # Strictly-pre-game: the first game has no prior history -> nothing feasible, flagged.
    assert not mask.loc[is_game_a, [f"feasible_{f}" for f in FAMILIES]].to_numpy().any()
    assert mask.loc[is_game_a, "low_history"].all()

    # Game B sees only Game A (11 pitches >= 5 -> established).
    b = mask.loc[is_game_b]
    assert b["feasible_FF"].all()  # count 4>=3, share 4/11=0.36>=0.3
    assert not b["feasible_SL"].any()  # count 3>=3 but share 3/11=0.27<0.3 (share gate)
    assert not b["feasible_CH"].any()  # same share failure
    assert not b["feasible_SI"].any()  # count 1<3 (count gate)
    assert not b["feasible_XX"].any()  # XX never feasible
    assert not b["low_history"].any()


def test_xx_never_feasible_on_sample(sample_raw):
    fam = add_family(sample_raw)
    mask = feasible_action_mask(fam)
    assert not mask["feasible_XX"].any()
    # Mask is aligned and complete.
    assert len(mask) == len(sample_raw)
    assert list(mask.columns) == [f"feasible_{f}" for f in FAMILIES] + ["low_history"]
