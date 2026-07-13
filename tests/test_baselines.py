"""Count-based reference baselines (decision D16; SPEC 8.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.eval import metrics as M
from pitchseq.eval.baselines import (
    GlobalCountHandBaseline,
    PitcherCountBaseline,
    PitcherCountPrevBaseline,
    TransitionBaseline,
    build_baselines,
)
from pitchseq.families import FAMILIES


def _markov_table(n_pa=400, seed=0, p_stay=0.15):
    """PAs whose family sequence is first-order Markov over {FF, SL}."""
    rng = np.random.default_rng(seed)
    fams = ["FF", "SL"]
    rows = []
    for pa in range(n_pa):
        length = int(rng.integers(3, 7))
        cur = fams[int(rng.integers(2))]
        for pn in range(1, length + 1):
            rows.append(
                {
                    "pa_id": pa,
                    "pitch_number": pn,
                    "family": cur,
                    "balls": (pn - 1) % 4,
                    "strikes": (pn - 1) % 3,
                    "stand": "R",
                    "pitcher": pa % 5,
                    "game_pk": pa % 10,
                }
            )
            # Transition: stay with prob p_stay, else switch.
            cur = cur if rng.random() < p_stay else (fams[1] if cur == fams[0] else fams[0])
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], categories=list(FAMILIES))
    return df


def _split(df):
    train = df[df["pa_id"] % 2 == 0]
    test = df[df["pa_id"] % 2 == 1]
    return train, test


def test_all_baselines_valid_distributions():
    df = _markov_table(seed=1)
    train, test = _split(df)
    for model in build_baselines(alpha=8.0).values():
        model.fit(train)
        proba = model.predict_proba(test)
        assert proba.shape == (len(test), len(FAMILIES))
        assert np.all(proba >= 0)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_unseen_key_falls_back_to_parent():
    df = _markov_table(seed=2)
    model = GlobalCountHandBaseline(alpha=8.0).fit(df)
    # A (balls, strikes, stand) key that never occurs -> exactly the global marginal.
    unseen = pd.DataFrame({"balls": [3], "strikes": [2], "stand": ["L"]})
    assert ("L" not in df["stand"].unique())
    proba = model.predict_proba(unseen)[0]
    assert np.allclose(proba, model.global_, atol=1e-12)


def test_shrinkage_pulls_low_support_toward_parent():
    # A pitcher-count cell with a single observation shrinks heavily toward the parent (a);
    # a cell with many observations sits close to its empirical frequency.
    alpha = 8.0
    df = _markov_table(seed=3)
    model = PitcherCountBaseline(alpha=alpha).fit(df)
    parent = model.parent._row_proba(0, 0, "R")

    # A (pitcher=0, balls=0, strikes=0) cell: compare its shrunk estimate to raw empirical.
    key_counts = model.cells_.get((0, 0, 0))
    assert key_counts is not None and key_counts.sum() > 0
    n = key_counts.sum()
    raw = key_counts / n
    shrunk = model._row_proba(0, 0, 0, "R")

    # Exact Dirichlet-smoothing identity: a convex combination of raw and parent.
    expected = (key_counts + alpha * parent) / (n + alpha)
    assert np.allclose(shrunk, expected)
    w_raw = n / (n + alpha)
    assert np.allclose(shrunk, w_raw * raw + (1.0 - w_raw) * parent)
    # Every component lies between the parent and the raw empirical (shrinkage toward parent).
    for c in range(len(FAMILIES)):
        lo, hi = sorted((parent[c], raw[c]))
        assert lo - 1e-9 <= shrunk[c] <= hi + 1e-9


def test_transition_baseline_uses_prev_family():
    # On first-order-Markov data the transition baseline (c) must beat the count-only
    # baseline (a) in held-out log loss.
    df = _markov_table(n_pa=600, seed=4, p_stay=0.12)
    train, test = _split(df)
    labels = list(FAMILIES)
    y = test["family"].astype("object").to_numpy()

    a = GlobalCountHandBaseline(alpha=8.0).fit(train)
    c = TransitionBaseline(alpha=8.0).fit(train)
    loss_a = M.log_loss(y, a.predict_proba(test), labels=labels)
    loss_c = M.log_loss(y, c.predict_proba(test), labels=labels)
    assert loss_c < loss_a - 0.05  # a clear margin from using prev_family

    # And the full hierarchy (d) is at least competitive with the transition model.
    d = PitcherCountPrevBaseline(alpha=8.0).fit(train)
    loss_d = M.log_loss(y, d.predict_proba(test), labels=labels)
    assert loss_d < loss_a
