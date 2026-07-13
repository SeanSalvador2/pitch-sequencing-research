"""Temporal splits and cluster bootstrap (SPEC 7 / 0)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchseq.splits import cluster_bootstrap_indices, make_splits, season_masks


def _season_frame():
    # 3 rows per season 2021..2025.
    seasons = np.repeat([2021, 2022, 2023, 2024, 2025], 3)
    return pd.DataFrame(
        {
            "season": seasons,
            "pitcher": np.arange(len(seasons)) % 4,
            "game_pk": np.arange(len(seasons)) // 2,
        }
    )


def test_primary_split_partitions_by_season():
    df = _season_frame()
    splits = make_splits(df)
    prim = splits["primary"]
    assert df.loc[prim["train"], "season"].isin([2021, 2022, 2023]).all()
    assert df.loc[prim["val"], "season"].isin([2024]).all()
    assert df.loc[prim["test"], "season"].isin([2025]).all()
    # Disjoint and covering.
    stacked = np.vstack([prim[k].to_numpy() for k in ("train", "val", "test")]).sum(axis=0)
    assert (stacked == 1).all()


def test_rolling_replications_present_and_ordered():
    df = _season_frame()
    splits = make_splits(df)
    assert len(splits["rolling"]) == 2
    r0 = splits["rolling"][0]
    assert df.loc[r0["train"], "season"].isin([2021, 2022]).all()
    assert df.loc[r0["val"], "season"].isin([2023]).all()
    assert df.loc[r0["test"], "season"].isin([2024]).all()


def test_non_temporal_split_raises():
    df = _season_frame()
    with pytest.raises(ValueError, match="not strictly before"):
        season_masks(df, {"train": [2024], "val": [2023], "test": [2025]})
    with pytest.raises(ValueError, match="overlap"):
        season_masks(df, {"train": [2021, 2022], "val": [2022], "test": [2023]})


def _cluster_frame():
    # Clusters (pitcher, game_pk) with known sizes: (100,1)=2, (100,2)=3, (200,1)=1.
    recs = (
        [{"pitcher": 100, "game_pk": 1}] * 2
        + [{"pitcher": 100, "game_pk": 2}] * 3
        + [{"pitcher": 200, "game_pk": 1}] * 1
    )
    return pd.DataFrame(recs)


def test_cluster_bootstrap_reproducible_and_whole_clusters():
    df = _cluster_frame()
    cid = (df["pitcher"].astype(str) + "_" + df["game_pk"].astype(str)).to_numpy()
    sizes = pd.Series(cid).value_counts().to_dict()

    a = list(cluster_bootstrap_indices(df, "pitcher_game", n_boot=5, seed=123))
    b = list(cluster_bootstrap_indices(df, "pitcher_game", n_boot=5, seed=123))
    # Reproducible under a fixed seed.
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    # A different seed gives a different draw (at least once).
    c = list(cluster_bootstrap_indices(df, "pitcher_game", n_boot=5, seed=999))
    assert any(not np.array_equal(x, z) for x, z in zip(a, c))

    for rep in a:
        counts = pd.Series(cid[rep]).value_counts().to_dict()
        # Every present cluster appears as a whole multiple of its size (no partial cluster).
        for cluster, cnt in counts.items():
            assert cnt % sizes[cluster] == 0
        # There are as many clusters drawn as the data has clusters.
        assert sum(counts[cl] // sizes[cl] for cl in counts) == len(sizes)


def test_bootstrap_indices_are_valid_positions():
    df = _cluster_frame()
    for rep in cluster_bootstrap_indices(df, "pitcher_game", n_boot=3, seed=1):
        assert rep.min() >= 0 and rep.max() < len(df)
