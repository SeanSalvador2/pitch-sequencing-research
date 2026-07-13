"""Raw Statcast loading (SPEC ``1`` / ``2``).

Accepts the on-disk forms the project produces: a single parquet file, a glob of
per-season parquet files, or the combined SQLite database (table ``statcast``). Applies
the game-type filter, parses ``game_date``, and returns rows in canonical pitch order.
"""

from __future__ import annotations

import glob
import os
import sqlite3
from pathlib import Path

import pandas as pd

from .config import load_config

__all__ = ["load_statcast", "SORT_KEYS"]

#: Canonical ordering key: chronological, then within game by at-bat and pitch.
SORT_KEYS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

_PARQUET_SUFFIXES = {".parquet", ".pq"}
_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _looks_like_glob(source: str) -> bool:
    return any(ch in source for ch in "*?[")


def _read_parquet_paths(paths: list[str]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No parquet files matched the given source.")
    frames = [pd.read_parquet(p) for p in sorted(paths)]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _read_sqlite(path: str, table: str = "statcast") -> pd.DataFrame:
    con = sqlite3.connect(path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)  # noqa: S608 - fixed table name
    finally:
        con.close()


def load_statcast(
    source: str | os.PathLike,
    game_types: list[str] | tuple[str, ...] | None = None,
    config: dict | None = None,
    table: str = "statcast",
) -> pd.DataFrame:
    """Load raw Statcast pitches from parquet or SQLite.

    Parameters
    ----------
    source : str or os.PathLike
        One of: a single parquet path; a glob pattern (contains ``*``/``?``/``[``)
        matching per-season parquet files; or a SQLite ``.db``/``.sqlite`` path holding
        the ``statcast`` table.
    game_types : sequence of str, optional
        Game types to keep. Defaults to ``config['data']['game_types']`` (``["R"]`` for
        the main analysis). Pass an empty sequence to keep all game types.
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.
    table : str, optional
        SQLite table name (default ``"statcast"``).

    Returns
    -------
    pandas.DataFrame
        Filtered, ``game_date``-parsed, and sorted by :data:`SORT_KEYS`.

    Raises
    ------
    FileNotFoundError
        If a plain path does not exist or a glob matches nothing.
    ValueError
        If the source type cannot be determined.
    """
    if config is None:
        config = load_config()
    if game_types is None:
        game_types = config.get("data", {}).get("game_types", ["R"])

    source_str = str(source)
    suffix = Path(source_str).suffix.lower()

    if _looks_like_glob(source_str):
        df = _read_parquet_paths(glob.glob(source_str))
    elif suffix in _SQLITE_SUFFIXES:
        if not Path(source_str).is_file():
            raise FileNotFoundError(f"SQLite database not found: {source_str}")
        df = _read_sqlite(source_str, table=table)
    elif suffix in _PARQUET_SUFFIXES:
        if not Path(source_str).is_file():
            raise FileNotFoundError(f"Parquet file not found: {source_str}")
        df = _read_parquet_paths([source_str])
    elif Path(source_str).is_dir():
        df = _read_parquet_paths(glob.glob(os.path.join(source_str, "*.parquet")))
    else:
        raise ValueError(
            f"Could not determine source type for {source_str!r}; expected a parquet "
            "file/glob/dir or a SQLite .db path."
        )

    df["game_date"] = pd.to_datetime(df["game_date"])

    if len(game_types) > 0:
        df = df[df["game_type"].isin(list(game_types))].copy()

    df = df.sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)
    return df
