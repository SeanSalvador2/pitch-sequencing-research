"""Build the canonical decision table from raw Statcast -- the first Phase-2 step (D15).

Runnable as ``python -m pitchseq.build_table``. It loads raw Statcast (SQLite or parquet)
via :mod:`pitchseq.io`, builds the SPEC ``3`` decision table one season at a time, writes a
``decision_table_<season>.parquet`` checkpoint per season next to the target, concatenates
them into the final table, runs the mandatory reward sign check (SPEC ``5``), records run
metadata (D13) to a JSON sidecar, and prints a compact headline block to paste back for
review.

Per-season **checkpointing with a look-back context window.** The leakage-safe rolling
features (SPEC ``3.2`` / ``0``) use a trailing 365-day window that crosses season
boundaries, so a season built in isolation would lose the prior season's trailing history at
its opening games. Each season ``Y`` is therefore built from raw spanning
``[min(date in Y) - 365d, max(date in Y)]`` and then filtered to season ``Y`` -- which makes
every checkpoint byte-for-byte identical to the corresponding rows of a whole-dataset build,
while still letting ``--resume`` skip the expensive per-season build after a crash.

Windows-safe by construction (D12): ``pathlib`` paths, ``argparse``, an
``if __name__ == "__main__"`` guard and no POSIX-only calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .decision_table import build_decision_table
from .io import load_statcast
from .reward import add_reward, sign_check
from .runmeta import track_run, write_runmeta

__all__ = ["main", "run_build", "build_season", "checkpoint_path"]

_TRAILING_CONTEXT_DAYS = 365  # matches pitchseq._trailing.TRAILING_WINDOW_DAYS


def checkpoint_path(out: Path, season: int) -> Path:
    """Return the per-season checkpoint path next to the final output ``out``."""
    return out.parent / f"decision_table_{season}.parquet"


def _season_series(raw: pd.DataFrame) -> pd.Series:
    """Season (calendar year) per raw row from ``game_date``."""
    return pd.to_datetime(raw["game_date"]).dt.year


def build_season(raw_all: pd.DataFrame, season: int, config: dict) -> pd.DataFrame:
    """Build the decision-table rows for one season with the trailing look-back context.

    Parameters
    ----------
    raw_all : pandas.DataFrame
        Raw Statcast for at least season ``season`` and its preceding 365 days.
    season : int
        Target season (calendar year).
    config : dict
        Study config.

    Returns
    -------
    pandas.DataFrame
        The decision table restricted to ``season`` (identical to the corresponding rows of
        a whole-dataset build).
    """
    dates = pd.to_datetime(raw_all["game_date"])
    seasons = dates.dt.year
    in_season = seasons == season
    if not in_season.any():
        return build_decision_table(raw_all.iloc[:0], config)  # empty, correct schema

    min_date = dates[in_season].min()
    max_date = dates[in_season].max()
    lookback_start = min_date - pd.Timedelta(days=_TRAILING_CONTEXT_DAYS)
    context_mask = (dates >= lookback_start) & (dates <= max_date)
    context = raw_all.loc[context_mask].copy()

    table = build_decision_table(context, config)
    return table.loc[table["season"] == season].reset_index(drop=True)


def _run_sign_check(raw_all: pd.DataFrame) -> dict:
    """Run the SPEC ``5`` reward sign check on raw rows; return a structured verdict."""
    try:
        means = sign_check(add_reward(raw_all))
        return {"status": "PASS", "means": {k: float(v) for k, v in means.items()}}
    except ValueError as exc:
        return {"status": "FAIL", "error": str(exc)}


def run_build(
    source: str,
    out: str,
    seasons: list[int] | None = None,
    resume: bool = False,
    config: dict | None = None,
) -> dict:
    """Build the decision table end to end and return a summary dict (no printing).

    Parameters
    ----------
    source : str
        Raw data source: a SQLite ``.db`` path, a parquet file, or a parquet glob (see
        :func:`pitchseq.io.load_statcast`).
    out : str
        Final decision-table parquet path. Per-season checkpoints are written alongside.
    seasons : list of int, optional
        Seasons to build. When ``None``, every season present in the source is built.
    resume : bool, optional
        Skip the build of any season whose checkpoint already exists (D12).
    config : dict, optional
        Study config; loaded from default when ``None``.

    Returns
    -------
    dict
        Summary with ``rows_per_season``, ``total_rows``, ``n_columns``, ``columns``,
        ``sign_check``, ``built_seasons``, ``resumed_seasons``, ``checkpoints``, ``out`` and
        the run-metadata fields (filled by :func:`pitchseq.runmeta.track_run`).
    """
    if config is None:
        config = load_config()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: every requested season already checkpointed -> concat only, no raw load.
    can_skip_load = (
        resume
        and seasons is not None
        and all(checkpoint_path(out_path, s).is_file() for s in seasons)
    )

    with track_run(step="build_table", source=str(source), out=str(out_path)) as meta:
        if can_skip_load:
            target_seasons = sorted(seasons)
            sign = {"status": "skipped", "reason": "all requested seasons resumed"}
        else:
            raw_all = load_statcast(source, config=config)
            sign = _run_sign_check(raw_all)
            if seasons is not None:
                target_seasons = sorted(seasons)
            else:
                target_seasons = sorted(_season_series(raw_all).unique().tolist())

        built, resumed = [], []
        checkpoints: list[Path] = []
        for season in target_seasons:
            cp = checkpoint_path(out_path, season)
            if resume and cp.is_file():
                resumed.append(int(season))
            else:
                table_season = build_season(raw_all, int(season), config)
                table_season.to_parquet(cp, engine="pyarrow", index=False)
                built.append(int(season))
            checkpoints.append(cp)

        # Concatenate season checkpoints into the final table.
        rows_per_season: dict[int, int] = {}
        frames = []
        for season, cp in zip(target_seasons, checkpoints):
            frame = pd.read_parquet(cp, engine="pyarrow")
            rows_per_season[int(season)] = int(len(frame))
            frames.append(frame)
        final = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        final.to_parquet(out_path, engine="pyarrow", index=False)

        meta["rows_per_season"] = rows_per_season
        meta["total_rows"] = int(len(final))
        meta["n_columns"] = int(final.shape[1])
        meta["columns"] = list(final.columns)
        meta["sign_check"] = sign
        meta["built_seasons"] = built
        meta["resumed_seasons"] = resumed
        meta["checkpoints"] = [str(c) for c in checkpoints]

    sidecar = out_path.parent / (out_path.stem + ".runmeta.json")
    write_runmeta(sidecar, meta)
    meta["runmeta_path"] = str(sidecar)
    return meta


def _format_headline(meta: dict) -> str:
    """Compact, copy-paste-friendly headline block."""
    rps = meta.get("rows_per_season", {})
    rows_line = ", ".join(f"{s}={rps[s]:,}" for s in sorted(rps))
    sign = meta.get("sign_check", {})
    if sign.get("status") == "PASS":
        m = sign.get("means", {})
        detail = "  ".join(f"{k}={v:+.4f}" for k, v in sorted(m.items()))
        sign_line = f"PASS   ({detail})"
    elif sign.get("status") == "FAIL":
        sign_line = f"FAIL   ({sign.get('error', '')[:120]})"
    else:
        sign_line = f"SKIPPED ({sign.get('reason', '')})"
    width = 68
    lines = [
        "=" * width,
        " decision-table build - headline",
        "=" * width,
        f" source         : {meta.get('source')}",
        f" output         : {meta.get('out')}",
        f" seasons        : {', '.join(str(s) for s in sorted(rps))}",
        f" rows / season  : {rows_line}",
        f" total rows     : {meta.get('total_rows', 0):,}",
        f" columns        : {meta.get('n_columns', 0)}",
        f" built / resumed: {meta.get('built_seasons')} / {meta.get('resumed_seasons')}",
        f" sign check     : {sign_line}",
        f" elapsed (s)    : {meta.get('seconds', float('nan')):.1f}",
        f" peak mem (MB)  : {meta.get('peak_mem_mb', float('nan')):.1f}",
        f" runmeta        : {meta.get('runmeta_path')}",
        "=" * width,
    ]
    return "\n".join(lines)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pitchseq.build_table",
        description="Build the canonical decision table (SPEC 3) from raw Statcast.",
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Raw source: SQLite .db, parquet file, or parquet glob.")
    parser.add_argument("--out", type=str, required=True,
                        help="Final decision-table parquet path.")
    parser.add_argument("--seasons", type=int, nargs="*", default=None,
                        help="Seasons to build (default: all seasons present).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip seasons whose checkpoint already exists.")
    parser.add_argument("--sample", type=str, default=None,
                        help="Use this parquet sample as the source (overrides --source).")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point. Returns the run summary dict and prints the headline block."""
    args = _parse_args(argv)
    source = args.sample if args.sample else args.source
    if source is None:
        print("error: one of --source or --sample is required", file=sys.stderr)
        raise SystemExit(2)

    meta = run_build(source, args.out, seasons=args.seasons, resume=args.resume)
    print(_format_headline(meta))
    if meta.get("sign_check", {}).get("status") == "FAIL":
        raise SystemExit(1)
    return meta


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m pitchseq.build_table`
    main()
