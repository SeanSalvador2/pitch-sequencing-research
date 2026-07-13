"""The standard prediction-output contract (SPEC 8.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.eval.predictions import (
    ACTION_PROB_COLS,
    OUTCOME1_PROB_COLS,
    OUTCOME2_PROB_COLS,
    load_predictions,
    prediction_groups_present,
    save_predictions,
    validate_predictions,
)
from pitchseq.families import FAMILIES
from pitchseq.outcomes import OUTCOME1, OUTCOME2


def _valid_predictions(n=12, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"row_id": [f"1_{i}_1" for i in range(n)]})
    a = rng.random((n, len(FAMILIES)))
    a /= a.sum(axis=1, keepdims=True)
    for j, c in enumerate(ACTION_PROB_COLS):
        df[c] = a[:, j]
    o1 = rng.random((n, len(OUTCOME1)))
    o1 /= o1.sum(axis=1, keepdims=True)
    for j, c in enumerate(OUTCOME1_PROB_COLS):
        df[c] = o1[:, j]
    df["exp_reward"] = rng.normal(0, 0.1, n)
    df["model_id"] = "unit"
    df["state_view"] = "O"
    df["seconds"] = 1.5
    df["peak_mem_mb"] = 100.0
    df["n_params"] = 10
    return df


def test_valid_predictions_pass():
    df = _valid_predictions()
    assert validate_predictions(df) is True
    assert set(prediction_groups_present(df)) == {"action_prob", "outcome1_prob"}


def test_duplicate_row_id_rejected():
    df = _valid_predictions()
    df.loc[1, "row_id"] = df.loc[0, "row_id"]
    with pytest.raises(ValueError, match="not unique"):
        validate_predictions(df)


def test_bad_prob_sum_rejected():
    df = _valid_predictions()
    df.loc[3, ACTION_PROB_COLS[0]] += 0.5  # break the simplex on one row
    with pytest.raises(ValueError, match="sum to 1"):
        validate_predictions(df)


def test_negative_prob_rejected():
    df = _valid_predictions()
    df.loc[2, ACTION_PROB_COLS[0]] = -0.2
    df.loc[2, ACTION_PROB_COLS[1]] += 0.2  # keep the row summing to 1
    with pytest.raises(ValueError, match="negative"):
        validate_predictions(df)


def test_missing_metadata_rejected():
    df = _valid_predictions().drop(columns=["n_params"])
    with pytest.raises(ValueError, match="n_params"):
        validate_predictions(df)


def test_illegal_state_view_rejected():
    df = _valid_predictions()
    df.loc[0, "state_view"] = "Z"
    with pytest.raises(ValueError, match="state_view"):
        validate_predictions(df)


def test_partial_nan_group_rejected():
    df = _valid_predictions()
    df.loc[4, ACTION_PROB_COLS[0]] = np.nan  # partial NaN in an otherwise-present group
    with pytest.raises(ValueError, match="partially-NaN"):
        validate_predictions(df)


def test_all_nan_outcome2_rows_allowed():
    # outcome2 applies only to in-play pitches; all-NaN rows are "not applicable".
    df = _valid_predictions(n=6)
    for c in OUTCOME2_PROB_COLS:
        df[c] = np.nan
    # Populate outcome2 on two rows only; the rest stay all-NaN and must be tolerated.
    o2 = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    for j, c in enumerate(OUTCOME2_PROB_COLS):
        df.loc[0, c] = o2[j]
        df.loc[1, c] = o2[j]
    assert validate_predictions(df) is True


def test_parquet_roundtrip_preserves_dtypes(tmp_path):
    df = _valid_predictions()
    df["model_id"] = df["model_id"].astype("category")
    df["n_params"] = df["n_params"].astype("int64")
    path = tmp_path / "preds.parquet"
    save_predictions(df, path)
    back = load_predictions(path)
    assert list(back.columns) == list(df.columns)
    for c in df.columns:
        assert str(back[c].dtype) == str(df[c].dtype), c
    pd.testing.assert_frame_equal(back, df)
