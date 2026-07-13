#!/usr/bin/env python3
"""
Pull full pitch-level Statcast data for a range of seasons and save it locally.

Data source: Baseball Savant, via pybaseball. One row per pitch, all columns.
Runs on your machine (Savant must be reachable) — NOT in a locked-down cloud box.

Usage
-----
    pip install -r requirements.txt
    python pull_statcast.py                       # 2021-2025, parquet per season
    python pull_statcast.py --start 2015 --end 2025
    python pull_statcast.py --sqlite              # also build a combined statcast.db
    python pull_statcast.py --outdir data/raw

What you get
------------
    data/raw/statcast_2021.parquet ... statcast_2025.parquet   (per season)
    data/raw/statcast.db  (optional, --sqlite: one `statcast` table, all seasons)

Notes
-----
* Resumable: a season whose .parquet already exists is skipped. Delete the file to re-pull.
* pybaseball caches daily chunks (~/.pybaseball) so re-runs are fast.
* Each season is ~700-750k pitches; the pull makes many small daily requests and
  can take ~10-25 min/season cold (much faster on cache). Be patient.
* We pull a wide window (Mar 15 - Nov 30) to capture regular + postseason and skip
  most spring training / the offseason. `game_type` is kept so you can filter:
      'R' regular, 'F/D/L/W' postseason rounds, 'S' spring, 'E' exhibition.
"""

from __future__ import annotations
import argparse
import os
import sys
import time

REQUIRED_FIELDS = [
    # sequencing & state
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number",
    "inning", "inning_topbot", "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "stand", "p_throws", "pitcher", "batter",
    "bat_score", "fld_score",
    # pitch identity
    "pitch_type", "pitch_name",
    # value / outcome (delta_run_exp is the key reward signal)
    "description", "type", "events", "delta_run_exp",
]


def human_mb(path: str) -> str:
    try:
        return f"{os.path.getsize(path) / 1e6:,.0f} MB"
    except OSError:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull Statcast pitch data by season.")
    ap.add_argument("--start", type=int, default=2021, help="first season (default 2021)")
    ap.add_argument("--end", type=int, default=2025, help="last season, inclusive (default 2025)")
    ap.add_argument("--outdir", default="data/raw", help="output directory (default data/raw)")
    ap.add_argument("--sqlite", action="store_true",
                    help="also build a combined SQLite db (statcast.db, table `statcast`)")
    ap.add_argument("--start-md", default="03-15", help="season start month-day (default 03-15)")
    ap.add_argument("--end-md", default="11-30", help="season end month-day (default 11-30)")
    args = ap.parse_args()

    try:
        import pandas as pd
        from pybaseball import statcast, cache
    except ImportError as e:
        print(f"Missing dependency: {e}. Run `pip install -r requirements.txt` first.",
              file=sys.stderr)
        return 1

    cache.enable()  # resumable daily-chunk cache
    os.makedirs(args.outdir, exist_ok=True)
    seasons = list(range(args.start, args.end + 1))
    print(f"Pulling Statcast for seasons {seasons[0]}-{seasons[-1]} -> {args.outdir}/")

    written = []
    for yr in seasons:
        out = os.path.join(args.outdir, f"statcast_{yr}.parquet")
        if os.path.exists(out):
            print(f"  [{yr}] already exists ({human_mb(out)}) — skipping. Delete to re-pull.")
            written.append(out)
            continue

        start_dt, end_dt = f"{yr}-{args.start_md}", f"{yr}-{args.end_md}"
        print(f"  [{yr}] pulling {start_dt} .. {end_dt} — this makes many daily requests, be patient...")
        t0 = time.time()
        try:
            df = statcast(start_dt=start_dt, end_dt=end_dt, verbose=True)
        except Exception as e:  # network hiccup etc. — keep going, re-run later
            print(f"  [{yr}] FAILED: {e}. Re-run the script to retry this season.", file=sys.stderr)
            continue

        if df is None or len(df) == 0:
            print(f"  [{yr}] returned 0 rows — skipping.", file=sys.stderr)
            continue

        # sanity: verify the load-bearing fields are present
        missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
        if missing:
            print(f"  [{yr}] WARNING: missing expected columns: {missing}")
        if "delta_run_exp" in df.columns:
            nn = df["delta_run_exp"].notna().mean()
            print(f"  [{yr}] delta_run_exp populated on {nn:5.1%} of pitches")

        df.to_parquet(out, index=False)
        mins = (time.time() - t0) / 60
        print(f"  [{yr}] {len(df):>8,} pitches, {df.shape[1]} cols -> {out} "
              f"({human_mb(out)}, {mins:.1f} min)")
        written.append(out)

    if not written:
        print("No data written.", file=sys.stderr)
        return 1

    # optional combined SQLite matching the pitch-sequencing repo's ingest schema
    if args.sqlite:
        import pandas as pd
        import sqlite3
        db = os.path.join(args.outdir, "statcast.db")
        print(f"Building combined SQLite -> {db} (table `statcast`)")
        if os.path.exists(db):
            os.remove(db)
        con = sqlite3.connect(db)
        total = 0
        for p in written:
            d = pd.read_parquet(p)
            d.to_sql("statcast", con, if_exists="append", index=False)
            total += len(d)
        con.execute("CREATE INDEX IF NOT EXISTS idx_ab "
                    "ON statcast(game_pk, at_bat_number, pitch_number)")
        con.commit()
        con.close()
        print(f"  wrote {total:,} pitches to {db} ({human_mb(db)})")

    print("\nDone. Per-season parquet files:")
    grand = 0
    for p in written:
        print(f"  {p}  ({human_mb(p)})")
        try:
            grand += os.path.getsize(p)
        except OSError:
            pass
    print(f"Total parquet: ~{grand/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
