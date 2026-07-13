"""Event-tree outcome labels (SPEC 3.5 / 6)."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import make_raw
from pitchseq.outcomes import (
    DESCRIPTION_TO_OUTCOME1,
    OUTCOME2,
    add_outcomes,
    assert_outcome_coverage,
)


def test_sample_description_coverage(sample_raw):
    # Every description value present in the real sample is in the locked vocabulary.
    assert_outcome_coverage(sample_raw)
    labelled = add_outcomes(sample_raw)
    assert labelled["outcome1"].notna().all()
    # outcome2 is defined exactly on in-play rows.
    in_play = labelled["outcome1"].astype("object") == "in_play"
    assert labelled.loc[in_play, "outcome2"].notna().all()
    assert labelled.loc[~in_play, "outcome2"].isna().all()
    # Every in-play event lands in a valid level-2 bucket.
    assert set(labelled.loc[in_play, "outcome2"].astype("object").unique()).issubset(set(OUTCOME2))


def test_raises_on_bogus_description():
    raw = make_raw([{"description": "not_a_real_description"}])
    with pytest.raises(ValueError, match="Unrecognised description"):
        add_outcomes(raw)


def test_level1_and_swing_flags_exact():
    rows = [
        {"description": "called_strike"},
        {"description": "swinging_strike"},
        {"description": "foul_tip"},
        {"description": "foul"},
        {"description": "ball"},
        {"description": "automatic_ball"},
        {"description": "automatic_strike"},
        {"description": "hit_by_pitch"},
        {"description": "missed_bunt"},
        {"description": "foul_bunt"},
    ]
    out = add_outcomes(make_raw(rows))
    assert out["outcome1"].astype("object").tolist() == [
        "called_strike",
        "whiff",
        "whiff",
        "foul",
        "ball",
        "ball",  # automatic_ball
        "called_strike",  # automatic_strike
        "hbp",
        "whiff",  # missed_bunt
        "foul",  # foul_bunt
    ]
    assert out["is_swing"].tolist() == [False, True, True, True, False, False, False, False, True, True]
    assert out["is_whiff"].tolist() == [False, True, True, False, False, False, False, False, True, False]


def test_level2_from_events():
    rows = [
        {"description": "hit_into_play", "events": "single"},
        {"description": "hit_into_play", "events": "double"},
        {"description": "hit_into_play", "events": "triple"},
        {"description": "hit_into_play", "events": "home_run"},
        {"description": "hit_into_play", "events": "field_out"},
        {"description": "hit_into_play", "events": "grounded_into_double_play"},
        {"description": "hit_into_play", "events": None},
        {"description": "ball", "events": None},
    ]
    out = add_outcomes(make_raw(rows))
    assert out["outcome2"].astype("object").tolist()[:6] == [
        "single",
        "double",
        "triple",
        "home_run",
        "out_or_other",
        "out_or_other",
    ]
    # in-play with a missing event still resolves to out_or_other; non-in-play stays NA.
    assert out["outcome2"].astype("object").tolist()[6] == "out_or_other"
    assert pd.isna(out["outcome2"].astype("object").tolist()[7])
