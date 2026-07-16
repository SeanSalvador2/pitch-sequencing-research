"""First-pitch corroboration check for the WS3 matchup finding (M+ branch).

On the first pitch of a PA there is no within-PA history, so O and OM differ only by
matchup memory -- an OM edge HERE is the designed corroboration that Delta_matchup
reflects genuine cross-PA adaptation rather than anything else (SPEC 6: "first pitch of
a PA (only matchup memory can matter there)").

Reads the WS3 per-view outcome predictions from results/ws3/, joins them to the decision
table, restricts to pitch_number == 1 on the validation seasons, and reports the
per-view outcome1 log loss plus the O-OM gap with a pitcher-game clustered CI.

Usage:
    python scripts/first_pitch_check.py [--table data/processed/decision_table.parquet]
                                        [--ws3-dir results/ws3]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from pitchseq.config import load_config
from pitchseq.eval.metrics import clustered_ci, log_loss_per_row
from pitchseq.outcomes import OUTCOME1


def _find_prediction_files(ws3_dir: Path) -> list[Path]:
    pats = ["*pred*", "*.parquet"]
    seen: dict[Path, None] = {}
    for pat in pats:
        for p in sorted(ws3_dir.glob(pat)):
            if p.suffix == ".parquet":
                seen[p] = None
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/processed/decision_table.parquet")
    ap.add_argument("--ws3-dir", default="results/ws3")
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    ws3_dir = Path(args.ws3_dir)
    files = _find_prediction_files(ws3_dir)
    if not files:
        print(f"No parquet files found under {ws3_dir}. Directory contents:")
        for p in sorted(ws3_dir.iterdir()):
            print("  ", p.name)
        return

    prob_cols = [f"outcome1_prob_{c}" for c in OUTCOME1]

    # Collect outcome predictions per view from whatever files carry the standard schema.
    per_view: dict[str, pd.DataFrame] = {}
    for p in files:
        try:
            df = pd.read_parquet(p)
        except Exception as exc:  # unreadable/foreign parquet: skip, say so
            print(f"  (skipping {p.name}: {exc})")
            continue
        if "row_id" not in df.columns or not all(c in df.columns for c in prob_cols):
            continue
        views = df["state_view"].unique() if "state_view" in df.columns else ["?"]
        for v in views:
            sub = df[df["state_view"] == v] if "state_view" in df.columns else df
            per_view.setdefault(str(v), sub[["row_id"] + prob_cols])

    if not per_view:
        print("No files matched the standard outcome-prediction schema. Files seen:")
        for p in files:
            print("  ", p.name)
        return
    print("views with outcome predictions:", sorted(per_view))

    table = pd.read_parquet(
        args.table,
        columns=["row_id", "pitch_number", "outcome1", "pitcher", "game_pk", "season"],
    )
    first = table[table["pitch_number"] == 1]

    losses: dict[str, pd.Series] = {}
    for v, preds in sorted(per_view.items()):
        j = first.merge(preds, on="row_id", how="inner")
        if j.empty:
            print(f"  {v}: no first-pitch rows joined (check row_id alignment)")
            continue
        pl = log_loss_per_row(
            j["outcome1"].astype("object").to_numpy(),
            j[prob_cols].to_numpy(dtype=np.float64),
            list(OUTCOME1),
            1e-12,
        )
        losses[v] = pd.Series(pl, index=pd.Index(j["row_id"], name="row_id"))
        print(f"  {v}: first-pitch outcome1 log loss = {pl.mean():.5f}  (n={len(j):,})")

    if "O" in losses and "OM" in losses:
        merged = pd.concat({"O": losses["O"], "OM": losses["OM"]}, axis=1, join="inner")
        meta = first.set_index("row_id").loc[merged.index, ["pitcher", "game_pk"]].reset_index()
        gap = (merged["O"] - merged["OM"]).to_numpy()  # >0: OM better on pitch 1
        ci = clustered_ci(
            lambda idx: float(gap[np.asarray(idx)].mean()),
            meta,
            cluster="pitcher_game",
            n_boot=args.n_boot,
            seed=load_config().get("seeds", {}).get("global", 0),
        )
        print(
            f"\nFIRST-PITCH Delta_matchup (O-OM log loss, >0 means matchup memory helps "
            f"where ONLY it can): {ci['point']:+.5f}  CI[{ci['lo']:+.5f}, {ci['hi']:+.5f}]"
        )
        verdict = "CORROBORATED" if ci["lo"] > 0 else (
            "NOT CORROBORATED (CI includes 0)" if ci["hi"] > 0 else "CONTRADICTED"
        )
        print(f"M+ first-pitch corroboration: {verdict}")
    else:
        print("\nNeed both O and OM predictions to compute the first-pitch gap.")


if __name__ == "__main__":
    main()
