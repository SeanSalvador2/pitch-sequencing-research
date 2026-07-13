"""Leakage-safe trailing-window aggregation (internal).

Both the feasible-action mask (``families``) and the rolling repertoire / tendency
features (``rolling``) need the same primitive: for every ``(entity, game_date)`` pair,
the sum of some per-game quantity over that entity's games whose date lies strictly
*before* the current game and within a fixed look-back window.

Locked window (SPEC ``0`` / ``4`` / ``7``): a prior game counts iff

    game_date - window_days  <=  prior_game_date  <  game_date

i.e. strictly earlier than the current game and no older than ``window_days``. Computing
this "per game_date" guarantees every pitch in a game shares one pre-game value, so the
current game can never leak into its own features.

The computation is two cumulative look-ups per game (an exclusive prefix sum plus a
``searchsorted`` for the window start), so it is ``O(n log n)`` and never rescans rows —
it scales to the full ~3.85M-row dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["TRAILING_WINDOW_DAYS", "trailing_window_sums"]

#: Locked trailing look-back, in days (SPEC ``4``: "trailing window ending before the
#: current game"; fixed to one calendar year to define ``trailing_season``).
TRAILING_WINDOW_DAYS = 365


def _entity_trailing(dates_ns: np.ndarray, values: np.ndarray, window_ns: int) -> np.ndarray:
    """Trailing-window sums for one entity.

    Parameters
    ----------
    dates_ns : numpy.ndarray
        Unique game dates for this entity as ``int64`` nanoseconds, sorted ascending.
    values : numpy.ndarray
        ``(n_games, k)`` per-game sums aligned to ``dates_ns``.
    window_ns : int
        Window length in nanoseconds.

    Returns
    -------
    numpy.ndarray
        ``(n_games, k)`` trailing sums; row ``i`` sums games ``j`` with
        ``dates_ns[i] - window_ns <= dates_ns[j] < dates_ns[i]``.
    """
    n, k = values.shape
    # Exclusive prefix sum: prefix[i] = sum of games 0..i-1 (all strictly earlier dates,
    # because dates are unique and ascending). prefix[0] = 0.
    prefix = np.zeros((n + 1, k), dtype=np.float64)
    np.cumsum(values, axis=0, out=prefix[1:])

    window_start = dates_ns - window_ns
    # First index whose date >= window_start; games below it are older than the window.
    lo = np.searchsorted(dates_ns, window_start, side="left")
    idx = np.arange(n)
    # prefix[idx] = everything strictly before game i; prefix[lo] = everything older
    # than the window. Their difference is exactly the in-window, strictly-prior sum.
    return prefix[idx] - prefix[lo]


def trailing_window_sums(
    per_game: pd.DataFrame,
    entity_col: str,
    date_col: str,
    value_cols: list[str],
    window_days: int = TRAILING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Trailing-window sums for every ``(entity, date)`` row.

    Parameters
    ----------
    per_game : pandas.DataFrame
        One row per ``(entity, date)`` with per-game sums in ``value_cols``. Duplicate
        ``(entity, date)`` pairs must already be aggregated away.
    entity_col, date_col : str
        Grouping and ordering columns. ``date_col`` must be datetime64.
    value_cols : list of str
        Columns to accumulate.
    window_days : int, optional
        Look-back length in days (default :data:`TRAILING_WINDOW_DAYS`).

    Returns
    -------
    pandas.DataFrame
        Same index as ``per_game``, columns ``value_cols`` holding the strictly-prior,
        in-window sums.
    """
    if per_game.empty:
        return per_game[value_cols].astype(np.float64).copy()

    window_ns = int(np.timedelta64(window_days, "D") / np.timedelta64(1, "ns"))
    dates_all = per_game[date_col].to_numpy(dtype="datetime64[ns]").astype("int64")
    values_all = per_game[value_cols].to_numpy(dtype=np.float64, na_value=np.nan)
    entities = per_game[entity_col].to_numpy()

    out = np.empty_like(values_all)

    # Stable sort by (entity, date) via lexsort; entity blocks become contiguous.
    order = np.lexsort((dates_all, entities))
    ent_sorted = entities[order]
    dates_sorted = dates_all[order]
    values_sorted = values_all[order]

    # Boundaries between contiguous entity blocks.
    if len(ent_sorted) > 0:
        change = np.empty(len(ent_sorted), dtype=bool)
        change[0] = True
        change[1:] = ent_sorted[1:] != ent_sorted[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(ent_sorted))
        for s, e in zip(starts, ends):
            out_sorted_block = _entity_trailing(dates_sorted[s:e], values_sorted[s:e], window_ns)
            # Scatter back to the original row positions.
            out[order[s:e]] = out_sorted_block

    result = pd.DataFrame(out, index=per_game.index, columns=value_cols)
    return result
