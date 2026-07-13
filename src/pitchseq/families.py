"""Pitch-family action space and leakage-safe feasible-action masks (SPEC ``4``).

The primary action is the **pitch family** (8 classes) mapped from the Statcast
``pitch_type`` classifier output. Family is preferred over exact ``pitch_type`` because
it reduces classifier drift, keeps enough support for policy evaluation, and bounds the
RL action space. ``XX`` is the descriptive-only bucket (rare/unknown) and is never a
recommendable action.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._trailing import TRAILING_WINDOW_DAYS, trailing_window_sums
from .config import load_config

__all__ = [
    "FAMILIES",
    "OTHER_FAMILY",
    "family_map_from_config",
    "add_family",
    "feasible_action_mask",
]

#: Canonical family ordering (SPEC ``4`` table order). ``XX`` is last and is the
#: descriptive-only "other/rare" bucket, excluded from recommendations.
FAMILIES: tuple[str, ...] = ("FF", "SI", "FC", "SL", "CU", "CH", "FS", "XX")

#: Bucket that unknown / null / rare pitch types collapse into.
OTHER_FAMILY = "XX"


def family_map_from_config(config: dict | None = None) -> dict[str, str]:
    """Build the ``pitch_type -> family`` lookup from configuration.

    Parameters
    ----------
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.

    Returns
    -------
    dict
        Maps each concrete ``pitch_type`` code to its family. Null / unknown codes are
        *not* keys here — they are handled by :func:`add_family` mapping to ``XX``.
    """
    if config is None:
        config = load_config()
    families_cfg = config["pitch_families"]

    if tuple(families_cfg.keys()) != FAMILIES:
        raise ValueError(
            f"config pitch_families keys {tuple(families_cfg.keys())} do not match the "
            f"locked FAMILIES ordering {FAMILIES}"
        )

    reverse: dict[str, str] = {}
    for family, codes in families_cfg.items():
        for code in codes:
            # YAML ``null`` in the XX list denotes "null pitch_type"; add_family handles
            # missing values directly, so a None entry is skipped here.
            if code is None:
                continue
            if code in reverse:
                raise ValueError(f"pitch_type {code!r} mapped to multiple families")
            reverse[str(code)] = family
    return reverse


def add_family(df: pd.DataFrame, config: dict | None = None, column: str = "family") -> pd.DataFrame:
    """Add the pitch-family column derived from ``pitch_type``.

    Unknown or null ``pitch_type`` values map to ``XX`` (SPEC ``4``).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``pitch_type``.
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.
    column : str, optional
        Output column name (default ``"family"``).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the family column added, dtype ``category`` over
        :data:`FAMILIES`.
    """
    reverse = family_map_from_config(config)
    out = df.copy()
    pt = out["pitch_type"].astype("object")
    fam = pt.map(reverse)
    fam = fam.where(fam.notna(), OTHER_FAMILY)  # null / unmapped -> XX
    out[column] = pd.Categorical(fam, categories=list(FAMILIES))
    return out


def feasible_action_mask(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Per-pitch, leakage-safe feasible-action mask over the 8 families (SPEC ``4``).

    A family is feasible for a decision iff, over the trailing window ending strictly
    before the current game (:data:`~pitchseq._trailing.TRAILING_WINDOW_DAYS` days), the
    pitcher threw it **>= min_count times AND >= min_share of pitches**. ``XX`` is never
    feasible. The mask is constant within a pitcher-game (it depends only on pre-game
    history), so it can never leak the current game.

    Low-history pitchers (fewer than ``rolling.min_history_pitches`` trailing pitches)
    are flagged; their mask may be empty. Consumers fall back to the observed action for
    those decisions rather than restricting to an empty feasible set.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``pitcher``, ``game_date`` (datetime64) and ``family`` (add it with
        :func:`add_family` first).
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.

    Returns
    -------
    pandas.DataFrame
        Same index as ``df`` with boolean columns ``feasible_<FAM>`` for each family and
        a boolean ``low_history`` column.
    """
    if config is None:
        config = load_config()
    if "family" not in df.columns:
        raise KeyError("feasible_action_mask requires a 'family' column; call add_family first")

    mask_cfg = config["action"]["feasible_mask"]
    min_count = float(mask_cfg["min_count"])
    min_share = float(mask_cfg["min_share"])
    min_history = float(config["rolling"]["min_history_pitches"])

    game_date = pd.to_datetime(df["game_date"])
    work = pd.DataFrame(
        {
            "pitcher": df["pitcher"].to_numpy(),
            "game_date": game_date.to_numpy(),
            "family": df["family"].astype("object").to_numpy(),
        },
        index=df.index,
    )

    # One-hot the family, then per-(pitcher, game) counts.
    fam_dummies = pd.get_dummies(work["family"]).reindex(columns=list(FAMILIES), fill_value=0)
    count_cols = [f"cnt_{fam}" for fam in FAMILIES]
    fam_dummies.columns = count_cols
    work = pd.concat([work[["pitcher", "game_date"]], fam_dummies.astype(np.float64)], axis=1)

    per_game = (
        work.groupby(["pitcher", "game_date"], observed=True)[count_cols].sum().reset_index()
    )
    per_game["total"] = per_game[count_cols].sum(axis=1)

    trailing = trailing_window_sums(
        per_game,
        entity_col="pitcher",
        date_col="game_date",
        value_cols=count_cols + ["total"],
        window_days=TRAILING_WINDOW_DAYS,
    )
    per_game_trailing = pd.concat([per_game[["pitcher", "game_date"]], trailing], axis=1)

    total = per_game_trailing["total"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for fam in FAMILIES:
            cnt = per_game_trailing[f"cnt_{fam}"].to_numpy()
            share = np.divide(cnt, total, out=np.zeros_like(cnt), where=total > 0)
            feasible = (cnt >= min_count) & (share >= min_share)
            if fam == OTHER_FAMILY:
                feasible = np.zeros_like(feasible, dtype=bool)  # XX never recommendable
            per_game_trailing[f"feasible_{fam}"] = feasible
    per_game_trailing["low_history"] = total < min_history

    out_cols = [f"feasible_{fam}" for fam in FAMILIES] + ["low_history"]
    merged = pd.merge(
        df[["pitcher"]].assign(game_date=game_date.to_numpy()).reset_index(),
        per_game_trailing[["pitcher", "game_date"] + out_cols],
        on=["pitcher", "game_date"],
        how="left",
    ).set_index("index")
    merged.index.name = df.index.name
    return merged[out_cols]
