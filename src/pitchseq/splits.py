"""Temporal splits and clustered resampling (SPEC ``7`` / ``0``).

Splits are strictly by season -- never random rows, because pitches in the same PA / game /
pitcher / season are highly dependent (SPEC ``0``). Confidence intervals are bootstrapped by
resampling whole **pitcher-game** clusters, the dependence unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config

__all__ = ["make_splits", "season_masks", "cluster_bootstrap_indices"]


def season_masks(df: pd.DataFrame, spec: dict, season_col: str = "season") -> dict[str, pd.Series]:
    """Boolean row masks for one split spec ``{train:[...], val:[...], test:[...]}``."""
    seasons = df[season_col]
    masks = {name: seasons.isin(spec[name]) for name in ("train", "val", "test")}
    _assert_split(masks, spec)
    return masks


def _assert_split(masks: dict[str, pd.Series], spec: dict) -> None:
    """Assert the three folds are disjoint and temporally ordered."""
    tr, va, te = spec["train"], spec["val"], spec["test"]
    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
        raise ValueError(f"Split seasons overlap: train={tr} val={va} test={te}")
    # Temporal ordering: max(train) < min(val) < ... and max(val) < min(test).
    if tr and va and not (max(tr) < min(va)):
        raise ValueError(f"train seasons {tr} not strictly before val {va}")
    if va and te and not (max(va) < min(te)):
        raise ValueError(f"val seasons {va} not strictly before test {te}")
    # Row-level disjointness (a row lands in at most one fold).
    stacked = np.vstack([m.to_numpy() for m in masks.values()]).sum(axis=0)
    if (stacked > 1).any():
        raise ValueError("A row was assigned to more than one split fold.")


def make_splits(df: pd.DataFrame, config: dict | None = None, season_col: str = "season") -> dict:
    """Build the primary split and the rolling replications from config (SPEC ``7``).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``season_col``.
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.
    season_col : str, optional
        Season column name (default ``"season"``).

    Returns
    -------
    dict
        ``{"primary": {train,val,test: mask}, "rolling": [ {train,val,test: mask}, ... ]}``.
        Each mask is a boolean :class:`pandas.Series` aligned to ``df``.
    """
    if config is None:
        config = load_config()
    split_cfg = config["split"]

    primary_spec = {name: split_cfg[name] for name in ("train", "val", "test")}
    out = {"primary": season_masks(df, primary_spec, season_col)}

    rolling = []
    for spec in split_cfg.get("rolling_replications", []):
        rolling.append(season_masks(df, spec, season_col))
    out["rolling"] = rolling
    return out


def _cluster_ids(df: pd.DataFrame, cluster: str) -> np.ndarray:
    """Build the cluster id array for the requested cluster unit."""
    if cluster == "pitcher_game":
        return (
            df["pitcher"].astype("Int64").astype("str")
            + "_"
            + df["game_pk"].astype("Int64").astype("str")
        ).to_numpy()
    if cluster in df.columns:
        return df[cluster].to_numpy()
    raise ValueError(f"Unknown cluster spec {cluster!r}")


def cluster_bootstrap_indices(
    df: pd.DataFrame,
    cluster: str = "pitcher_game",
    n_boot: int = 200,
    seed: int | None = None,
):
    """Yield cluster-bootstrap resamples as positional row-index arrays.

    Whole clusters are drawn with replacement (as many clusters as the data has), and every
    row of a drawn cluster is included -- so within-cluster dependence is preserved (SPEC
    ``0`` / ``7``). Reproducible for a fixed ``seed``.

    Parameters
    ----------
    df : pandas.DataFrame
        Rows to resample. For ``cluster="pitcher_game"`` needs ``pitcher`` and ``game_pk``.
    cluster : str, optional
        Cluster unit: ``"pitcher_game"`` (default) or any column name in ``df``.
    n_boot : int, optional
        Number of bootstrap replicates (default 200).
    seed : int, optional
        Seed for reproducibility. When ``None`` the config global seed is used.

    Yields
    ------
    numpy.ndarray
        Positional indices (into ``df``) for one bootstrap replicate; concatenated rows of
        the drawn clusters (order grouped by draw).
    """
    if seed is None:
        seed = load_config().get("seeds", {}).get("global", 0)
    rng = np.random.default_rng(seed)

    ids = _cluster_ids(df, cluster)
    unique, inverse = np.unique(ids, return_inverse=True)
    n_clusters = len(unique)
    # Row positions for each cluster (grouped once, reused across replicates).
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    boundaries = np.flatnonzero(np.diff(sorted_inv)) + 1
    cluster_rows = np.split(order, boundaries)

    for _ in range(n_boot):
        drawn = rng.integers(0, n_clusters, size=n_clusters)
        yield np.concatenate([cluster_rows[c] for c in drawn])
