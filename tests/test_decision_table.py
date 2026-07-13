"""Canonical decision table structure and leakage discipline (SPEC 3)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_raw
from pitchseq.decision_table import (
    EXECUTION_COLS,
    LABEL_COLS,
    STATE_ELIGIBLE_COLS,
    build_decision_table,
    decision_table_column_groups,
    leakage_audit,
)


def test_row_id_unique_and_groups_present(sample_table):
    assert sample_table["row_id"].is_unique
    groups = decision_table_column_groups()
    for group, cols in groups.items():
        for c in cols:
            assert c in sample_table.columns, f"missing {group} column {c}"


def test_leakage_audit_passes_on_state_eligible_and_fails_on_exec_label():
    assert leakage_audit(STATE_ELIGIBLE_COLS) is True
    with pytest.raises(AssertionError, match="Leakage audit failed"):
        leakage_audit(STATE_ELIGIBLE_COLS + ["exec_release_speed"])
    with pytest.raises(AssertionError, match="Leakage audit failed"):
        leakage_audit(["balls", "R"])
    # No execution or label column is ever declared state-eligible.
    assert not (set(STATE_ELIGIBLE_COLS) & (set(EXECUTION_COLS) | set(LABEL_COLS)))


def test_base_state_bit_encoding():
    def occ(one, two, three):
        return {"on_1b": 111 if one else None, "on_2b": 222 if two else None, "on_3b": 333 if three else None}

    rows = [
        {**occ(0, 0, 0), "at_bat_number": 1},
        {**occ(1, 0, 0), "at_bat_number": 2},
        {**occ(0, 1, 0), "at_bat_number": 3},
        {**occ(1, 1, 0), "at_bat_number": 4},
        {**occ(0, 0, 1), "at_bat_number": 5},
        {**occ(1, 0, 1), "at_bat_number": 6},
        {**occ(0, 1, 1), "at_bat_number": 7},
        {**occ(1, 1, 1), "at_bat_number": 8},
    ]
    table = build_decision_table(make_raw(rows))
    assert table.sort_values("at_bat_number")["base_state"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_base_state_range_on_sample(sample_table):
    assert sample_table["base_state"].min() >= 0
    assert sample_table["base_state"].max() <= 7


def test_pitcher_pitch_count_strictly_increasing_in_game():
    rows = [
        {"game_pk": 1, "pitcher": 100, "at_bat_number": 1, "pitch_number": 1},
        {"game_pk": 1, "pitcher": 100, "at_bat_number": 1, "pitch_number": 2},
        {"game_pk": 1, "pitcher": 100, "at_bat_number": 1, "pitch_number": 3},
        {"game_pk": 1, "pitcher": 100, "at_bat_number": 2, "pitch_number": 1},
        {"game_pk": 1, "pitcher": 100, "at_bat_number": 2, "pitch_number": 2},
    ]
    table = build_decision_table(make_raw(rows)).sort_values(["at_bat_number", "pitch_number"])
    counts = table["pitcher_pitch_count"].tolist()
    assert counts == [0, 1, 2, 3, 4]
    assert all(b > a for a, b in zip(counts, counts[1:]))


def test_derived_keys_and_flags():
    rows = [
        {"game_pk": 5, "at_bat_number": 7, "pitch_number": 1, "stand": "L", "p_throws": "R"},
        {"game_pk": 5, "at_bat_number": 7, "pitch_number": 2, "stand": "L", "p_throws": "L"},
    ]
    table = build_decision_table(make_raw(rows)).sort_values("pitch_number")
    assert table["pa_id"].tolist() == [5007, 5007]
    assert table["row_id"].tolist() == ["5_7_1", "5_7_2"]
    assert table["is_seq_eligible"].tolist() == [False, True]
    assert table["platoon"].tolist() == [False, True]  # stand==p_throws


def test_sample_build_smoke(sample_table, sample_raw):
    # One row per input pitch, and the audit holds on the whole table's state-eligible set.
    assert len(sample_table) == len(sample_raw)
    assert leakage_audit(STATE_ELIGIBLE_COLS) is True
    # score_diff_pitcher is fld_score - bat_score.
    assert sample_table["score_diff_pitcher"].notna().all()
