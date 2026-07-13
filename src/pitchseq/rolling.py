"""Leakage-safe rolling pitcher / batter features (SPEC ``3.2`` / ``0``).

Every feature here is a trailing aggregate over a pitcher's or batter's pitches dated
strictly *before the current game*, within the locked look-back window
(:data:`~pitchseq._trailing.TRAILING_WINDOW_DAYS` days). Because the value is computed per
``(entity, game_date)`` and broadcast to every pitch in that game, the current game can
never contribute to its own features -- the core leakage guarantee of SPEC ``0``.

Provided:

* pitcher repertoire mix per family (``repertoire_mix_<FAM>``, sums to 1 when history
  exists);
* pitcher per-family execution baselines (``pitcher_base_<FAM>_{release_speed,pfx_x,pfx_z}``)
  used later for history diffs-from-baseline;
* batter response tendencies (``batter_tend_{swing_rate,whiff_rate,chase_rate,inplay_rate}``);
* trailing pitch counts and ``low_history`` flags for both entities;
* (for the OM view) batter-vs-family trailing swing / whiff rates.

All aggregation goes through the same cumulative-sum primitive, so it scales to the full
dataset without per-row rescans.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._trailing import TRAILING_WINDOW_DAYS, trailing_window_sums
from .config import load_config
from .families import FAMILIES

__all__ = [
    "repertoire_mix_columns",
    "batter_tend_columns",
    "pitcher_rolling",
    "batter_rolling",
    "batter_family_rolling",
    "add_rolling",
]

_IN_ZONE = frozenset(range(1, 10))  # Statcast zones 1..9 are inside the strike zone.


def repertoire_mix_columns() -> list[str]:
    """Return the 8 ``repertoire_mix_<FAM>`` column names in family order."""
    return [f"repertoire_mix_{fam}" for fam in FAMILIES]


def batter_tend_columns() -> list[str]:
    """Return the batter-tendency column names."""
    return [
        "batter_tend_swing_rate",
        "batter_tend_whiff_rate",
        "batter_tend_chase_rate",
        "batter_tend_inplay_rate",
    ]


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den``, returning NaN where ``den <= 0`` (no history)."""
    out = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _float_array(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract a column as a plain float64 numpy array (nullable NA -> NaN)."""
    return df[col].astype("Float64").astype("float64").to_numpy()


def _merge_per_game_to_rows(
    df: pd.DataFrame, per_game_out: pd.DataFrame, entity_col: str
) -> pd.DataFrame:
    """Broadcast per-``(entity, game_date)`` values back onto every pitch row.

    Uses a MultiIndex reindex so duplicate (entity, date) targets fan out correctly and
    the original row index / order is preserved.
    """
    keyed = per_game_out.set_index([entity_col, "game_date"])
    target = pd.MultiIndex.from_arrays(
        [df[entity_col].to_numpy(), pd.to_datetime(df["game_date"]).to_numpy()]
    )
    result = keyed.reindex(target)
    result.index = df.index
    return result


def pitcher_rolling(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Trailing pitcher repertoire mix and per-family execution baselines.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``pitcher``, ``game_date`` (datetime64), ``family`` and the physical
        columns ``release_speed``, ``pfx_x``, ``pfx_z``.
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``df`` with ``repertoire_mix_<FAM>`` (8), ``pitcher_base_<FAM>_*``
        (24), ``pitcher_n_trailing_pitches`` and ``pitcher_low_history``.
    """
    if config is None:
        config = load_config()
    min_history = float(config["rolling"]["min_history_pitches"])

    fam = df["family"].astype("object").to_numpy()
    n = len(df)
    rs = _float_array(df, "release_speed")
    px = _float_array(df, "pfx_x")
    pz = _float_array(df, "pfx_z")
    valid = (~np.isnan(rs)).astype(np.float64)  # release_speed/pfx share a null pattern
    rs0 = np.nan_to_num(rs)
    px0 = np.nan_to_num(px)
    pz0 = np.nan_to_num(pz)

    cols: dict[str, np.ndarray] = {"p_total": np.ones(n, dtype=np.float64)}
    for fam_name in FAMILIES:
        d = (fam == fam_name).astype(np.float64)
        cols[f"cnt_{fam_name}"] = d
        cols[f"nval_{fam_name}"] = d * valid
        cols[f"sspd_{fam_name}"] = d * rs0
        cols[f"spfxx_{fam_name}"] = d * px0
        cols[f"spfxz_{fam_name}"] = d * pz0
    value_cols = list(cols)

    work = pd.DataFrame(cols, index=df.index)
    work["pitcher"] = df["pitcher"].to_numpy()
    work["game_date"] = pd.to_datetime(df["game_date"]).to_numpy()

    per_game = work.groupby(["pitcher", "game_date"], observed=True)[value_cols].sum().reset_index()
    trailing = trailing_window_sums(per_game, "pitcher", "game_date", value_cols, TRAILING_WINDOW_DAYS)
    pg = pd.concat([per_game[["pitcher", "game_date"]], trailing], axis=1)

    total = pg["p_total"].to_numpy()
    out: dict[str, np.ndarray] = {
        "pitcher_n_trailing_pitches": total,
        "pitcher_low_history": total < min_history,
    }
    for fam_name in FAMILIES:
        cnt = pg[f"cnt_{fam_name}"].to_numpy()
        nval = pg[f"nval_{fam_name}"].to_numpy()
        out[f"repertoire_mix_{fam_name}"] = _safe_div(cnt, total)
        out[f"pitcher_base_{fam_name}_release_speed"] = _safe_div(pg[f"sspd_{fam_name}"].to_numpy(), nval)
        out[f"pitcher_base_{fam_name}_pfx_x"] = _safe_div(pg[f"spfxx_{fam_name}"].to_numpy(), nval)
        out[f"pitcher_base_{fam_name}_pfx_z"] = _safe_div(pg[f"spfxz_{fam_name}"].to_numpy(), nval)

    pg_out = pd.concat([pg[["pitcher", "game_date"]], pd.DataFrame(out, index=pg.index)], axis=1)
    return _merge_per_game_to_rows(df, pg_out, "pitcher")


def batter_rolling(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Trailing batter response tendencies.

    ``swing_rate`` and ``inplay_rate`` are per-pitch; ``whiff_rate`` is per swing;
    ``chase_rate`` is swings at out-of-zone pitches over out-of-zone pitches (SPEC ``3.2``;
    chase = swing on a pitch whose ``zone`` is not 1..9).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``batter``, ``game_date`` (datetime64), ``zone``, ``is_swing``,
        ``is_whiff`` and ``outcome1`` (run :func:`~pitchseq.outcomes.add_outcomes` first).
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``df`` with the four ``batter_tend_*`` columns,
        ``batter_n_trailing_pitches`` and ``batter_low_history``.
    """
    if config is None:
        config = load_config()
    min_history = float(config["rolling"]["min_history_pitches"])

    n = len(df)
    is_swing = df["is_swing"].to_numpy().astype(np.float64)
    is_whiff = df["is_whiff"].to_numpy().astype(np.float64)
    zone = _float_array(df, "zone")
    in_zone = np.isin(zone, list(_IN_ZONE))
    ooz = (~in_zone) & (~np.isnan(zone))  # out of zone, zone known
    is_inplay = (df["outcome1"].astype("object").to_numpy() == "in_play").astype(np.float64)

    cols = {
        "b_total": np.ones(n, dtype=np.float64),
        "n_swing": is_swing,
        "n_whiff": is_whiff,
        "n_ooz": ooz.astype(np.float64),
        "n_chase": (is_swing.astype(bool) & ooz).astype(np.float64),
        "n_inplay": is_inplay,
    }
    value_cols = list(cols)

    work = pd.DataFrame(cols, index=df.index)
    work["batter"] = df["batter"].to_numpy()
    work["game_date"] = pd.to_datetime(df["game_date"]).to_numpy()

    per_game = work.groupby(["batter", "game_date"], observed=True)[value_cols].sum().reset_index()
    trailing = trailing_window_sums(per_game, "batter", "game_date", value_cols, TRAILING_WINDOW_DAYS)
    pg = pd.concat([per_game[["batter", "game_date"]], trailing], axis=1)

    total = pg["b_total"].to_numpy()
    swings = pg["n_swing"].to_numpy()
    out = {
        "batter_tend_swing_rate": _safe_div(swings, total),
        "batter_tend_whiff_rate": _safe_div(pg["n_whiff"].to_numpy(), swings),
        "batter_tend_chase_rate": _safe_div(pg["n_chase"].to_numpy(), pg["n_ooz"].to_numpy()),
        "batter_tend_inplay_rate": _safe_div(pg["n_inplay"].to_numpy(), total),
        "batter_n_trailing_pitches": total,
        "batter_low_history": total < min_history,
    }
    pg_out = pd.concat([pg[["batter", "game_date"]], pd.DataFrame(out, index=pg.index)], axis=1)
    return _merge_per_game_to_rows(df, pg_out, "batter")


def batter_family_rolling(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Trailing batter-vs-family swing / whiff rates (feeds the OM view, SPEC ``3.4``).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``batter``, ``game_date``, ``family``, ``is_swing``, ``is_whiff``.
    config : dict, optional
        Unused hook for symmetry / future thresholds.

    Returns
    -------
    pandas.DataFrame
        Indexed like ``df`` with ``batter_vs_fam_swing_rate_<FAM>`` and
        ``batter_vs_fam_whiff_rate_<FAM>`` (NaN where no trailing history of that family).
    """
    fam = df["family"].astype("object").to_numpy()
    is_swing = df["is_swing"].to_numpy().astype(np.float64)
    is_whiff = df["is_whiff"].to_numpy().astype(np.float64)

    cols: dict[str, np.ndarray] = {}
    for fam_name in FAMILIES:
        d = (fam == fam_name).astype(np.float64)
        cols[f"bf_total_{fam_name}"] = d
        cols[f"bf_swing_{fam_name}"] = d * is_swing
        cols[f"bf_whiff_{fam_name}"] = d * is_whiff
    value_cols = list(cols)

    work = pd.DataFrame(cols, index=df.index)
    work["batter"] = df["batter"].to_numpy()
    work["game_date"] = pd.to_datetime(df["game_date"]).to_numpy()

    per_game = work.groupby(["batter", "game_date"], observed=True)[value_cols].sum().reset_index()
    trailing = trailing_window_sums(per_game, "batter", "game_date", value_cols, TRAILING_WINDOW_DAYS)
    pg = pd.concat([per_game[["batter", "game_date"]], trailing], axis=1)

    out: dict[str, np.ndarray] = {}
    for fam_name in FAMILIES:
        tot = pg[f"bf_total_{fam_name}"].to_numpy()
        sw = pg[f"bf_swing_{fam_name}"].to_numpy()
        out[f"batter_vs_fam_swing_rate_{fam_name}"] = _safe_div(sw, tot)
        out[f"batter_vs_fam_whiff_rate_{fam_name}"] = _safe_div(pg[f"bf_whiff_{fam_name}"].to_numpy(), sw)
    pg_out = pd.concat([pg[["batter", "game_date"]], pd.DataFrame(out, index=pg.index)], axis=1)
    return _merge_per_game_to_rows(df, pg_out, "batter")


def add_rolling(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Attach pitcher and batter rolling features to ``df``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the pitcher and batter rolling columns concatenated.
    """
    if config is None:
        config = load_config()
    p = pitcher_rolling(df, config)
    b = batter_rolling(df, config)
    return pd.concat([df, p, b], axis=1)
