"""Variable-length ordered history tensors (SPEC 3.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import make_raw
from pitchseq.decision_table import build_decision_table
from pitchseq.families import FAMILIES
from pitchseq.outcomes import OUTCOME1
from pitchseq.sequences import build_sequences


def _pa_table():
    rows = [
        dict(game_pk=1, at_bat_number=1, pitch_number=1, pitch_type="FF", release_speed=95.0,
             plate_x=0.5, plate_z=2.5, sz_top=3.5, sz_bot=1.5, stand="R", description="ball"),
        dict(game_pk=1, at_bat_number=1, pitch_number=2, pitch_type="SL", release_speed=85.0,
             plate_x=-0.5, plate_z=2.0, sz_top=3.5, sz_bot=1.5, stand="R", description="called_strike"),
        dict(game_pk=1, at_bat_number=1, pitch_number=3, pitch_type="FF", release_speed=96.0,
             plate_x=0.0, plate_z=2.5, sz_top=3.5, sz_bot=1.5, stand="R", description="swinging_strike"),
    ]
    return build_decision_table(make_raw(rows))


def _pos(table, ab, pn):
    return int(np.flatnonzero((table["at_bat_number"].to_numpy() == ab) & (table["pitch_number"].to_numpy() == pn))[0])


def test_history_tensor_matches_expectation():
    table = _pa_table()
    seq = build_sequences(table, max_len=15)
    p3 = _pos(table, 1, 3)  # decision at pitch 3: history = [FF@95, SL@85]

    assert seq["lengths"][p3] == 2
    assert seq["mask"][p3].tolist() == [1, 1] + [0] * 13
    assert seq["family_idx"][p3, 0] == FAMILIES.index("FF")
    assert seq["family_idx"][p3, 1] == FAMILIES.index("SL")
    assert seq["family_idx"][p3, 2] == len(FAMILIES)  # pad index
    assert seq["outcome1_idx"][p3, 0] == OUTCOME1.index("ball")
    assert seq["outcome1_idx"][p3, 1] == OUTCOME1.index("called_strike")

    assert seq["release_speed"][p3, :2].tolist() == [95.0, 85.0]
    assert seq["plate_x_br"][p3, :2].tolist() == [0.5, -0.5]
    assert seq["plate_z_norm"][p3, :2].tolist() == [0.5, 0.25]
    # consecutive-diff channels: 0 at the first position, then the step into the next pitch.
    assert seq["dvelo"][p3, 0] == 0.0
    assert seq["dvelo"][p3, 1] == pytest.approx(-10.0)
    assert seq["dloc"][p3, 1] == pytest.approx(math.hypot(-1.0, -0.25))

    # row_id alignment with the decision table.
    assert seq["row_id"][p3] == table.iloc[p3]["row_id"]


def test_first_pitch_has_empty_history():
    table = _pa_table()
    seq = build_sequences(table, max_len=15)
    p1 = _pos(table, 1, 1)
    assert seq["lengths"][p1] == 0
    assert seq["mask"][p1].sum() == 0
    assert (seq["family_idx"][p1] == len(FAMILIES)).all()


def test_intermediate_pitch_history_length():
    table = _pa_table()
    seq = build_sequences(table, max_len=15)
    p2 = _pos(table, 1, 2)  # history = [FF@95]
    assert seq["lengths"][p2] == 1
    assert seq["mask"][p2].tolist() == [1] + [0] * 14
    assert seq["family_idx"][p2, 0] == FAMILIES.index("FF")
    assert seq["release_speed"][p2, 0] == 95.0


def test_truncation_to_max_len():
    # A long PA: 6 pitches, max_len=3 -> the decision keeps only the most recent 3 priors.
    rows = [
        dict(game_pk=1, at_bat_number=1, pitch_number=i + 1, pitch_type=pt, release_speed=90.0 + i,
             description="foul" if i < 5 else "swinging_strike")
        for i, pt in enumerate(["FF", "SL", "CH", "FC", "CU", "FF"])
    ]
    table = build_decision_table(make_raw(rows))
    seq = build_sequences(table, max_len=3)
    p6 = _pos(table, 1, 6)  # 5 priors, truncated to last 3: CH, FC, CU
    assert seq["lengths"][p6] == 3
    assert [seq["family_idx"][p6, k] for k in range(3)] == [
        FAMILIES.index("CH"),
        FAMILIES.index("FC"),
        FAMILIES.index("CU"),
    ]


def test_sample_sequences_build(sample_table):
    seq = build_sequences(sample_table, max_len=15)
    assert seq["family_idx"].shape == (len(sample_table), 15)
    assert len(seq["row_id"]) == len(sample_table)
    # lengths never exceed max_len and match the mask row sums.
    assert seq["lengths"].max() <= 15
    assert np.array_equal(seq["lengths"], seq["mask"].sum(axis=1).astype(int))
