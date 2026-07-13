"""Bits of predictability (SPEC 10)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pitchseq.eval.predictability import bits_of_predictability, bits_summary
from pitchseq.families import FAMILIES


def test_bits_hand_computed():
    n_fam = len(FAMILIES)
    q_c = np.full((2, n_fam), 1.0 / n_fam)  # uniform context predictor
    q_o = np.full((2, n_fam), (1.0 - 0.5) / (n_fam - 1))
    q_o[:, 0] = 0.5  # ordered predictor concentrates on family 0 (FF)
    actions = np.array([0, 0])  # both actions are FF
    bits = bits_of_predictability(q_o, q_c, actions)
    expected = math.log2(0.5 / (1.0 / n_fam))
    assert bits[0] == pytest.approx(expected)
    assert bits[1] == pytest.approx(expected)
    # When the ordered predictor is worse on the realised action, bits go negative.
    actions2 = np.array([1, 1])
    bits2 = bits_of_predictability(q_o, q_c, actions2)
    assert bits2[0] < 0


def test_bits_accepts_family_labels():
    n_fam = len(FAMILIES)
    q_c = np.full((1, n_fam), 1.0 / n_fam)
    q_o = np.full((1, n_fam), 0.01)
    q_o[0, FAMILIES.index("SL")] = 1.0 - 0.01 * (n_fam - 1)
    bits = bits_of_predictability(q_o, q_c, ["SL"])
    assert bits[0] > 0  # ordered predictor is more confident on the realised SL


def test_bits_summary_slices():
    df = pd.DataFrame(
        {
            "pitcher": [1, 1, 2, 2],
            "balls": [0, 0, 1, 1],
            "strikes": [0, 1, 0, 1],
        }
    )
    bits = np.array([1.0, 3.0, -1.0, 1.0])
    by_pitcher = bits_summary(df, bits, by=["pitcher"])
    assert list(by_pitcher["pitcher"]) == [1, 2]
    assert by_pitcher.loc[by_pitcher["pitcher"] == 1, "mean_bits"].iloc[0] == pytest.approx(2.0)
    assert by_pitcher.loc[by_pitcher["pitcher"] == 2, "mean_bits"].iloc[0] == pytest.approx(0.0)
    assert by_pitcher["count"].sum() == 4

    by_count = bits_summary(df, bits, by=["balls", "strikes"])
    assert len(by_count) == 4  # four distinct (balls, strikes) cells
