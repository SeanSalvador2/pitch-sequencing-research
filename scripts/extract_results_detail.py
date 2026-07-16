"""Compact digest of the real-data run reports, for the documentation fill-in.

Reads the WS1/WS2/WS3 report JSONs (and the WS3 per-view model sidecars) and prints a
bounded, paste-friendly digest of the details the papers need beyond the run headlines:
fitted concentrations (what the data decided about each hierarchy level), the WS2 motif
table and depth usage, WS3's per-view top gain features (which features carry the
measured effects), the held-out test central table, and the chosen hyperparameters.

Read-only; safe to run alongside a live WS4 run.

Usage:
    python scripts/extract_results_detail.py [--results results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    if not path.is_file():
        print(f"  !! missing: {path}")
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _walk_numeric(d, want, path="", depth=0, out=None):
    """Collect (path, value) leaves whose path mentions any `want` token."""
    if out is None:
        out = []
    if depth > 6:
        return out
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else str(k)
            if isinstance(v, (int, float)) and any(w in p.lower() for w in want):
                out.append((p, v))
            else:
                _walk_numeric(v, want, p, depth + 1, out)
    elif isinstance(d, list) and d and isinstance(d[0], dict):
        for i, v in enumerate(d[:8]):
            _walk_numeric(v, want, f"{path}[{i}]", depth + 1, out)
    return out


def _section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)

    # ---------------- WS1 ----------------
    _section("WS1 detail (fitted concentrations, backoff)")
    r = _load(root / "ws1" / "ws1_report_real.json")
    if r:
        print("top-level keys:", sorted(r.keys())[:20])
        conc = _walk_numeric(r, ("concentration", "alpha", "kappa"))
        for p, v in conc[:40]:
            print(f"  {p} = {v:.6g}")
        back = _walk_numeric(r, ("backoff",))
        for p, v in back[:20]:
            print(f"  {p} = {v:.6g}")

    # ---------------- WS2 ----------------
    _section("WS2 detail (per-depth concentrations, motif table, depth usage)")
    r = _load(root / "ws2" / "ws2_report_real.json")
    if r:
        print("top-level keys:", sorted(r.keys())[:20])
        conc = _walk_numeric(r, ("concentration", "alpha"))
        for p, v in conc[:30]:
            print(f"  {p} = {v:.6g}")
        motifs = None
        for key in ("motifs", "top_motifs", "grammar_motifs"):
            if isinstance(r.get(key), list):
                motifs = r[key]
                break
        if motifs is None:  # search one level down
            for v in r.values():
                if isinstance(v, dict):
                    for key in ("motifs", "top_motifs"):
                        if isinstance(v.get(key), list):
                            motifs = v[key]
                            break
        if motifs:
            print(f"  motif rows: {len(motifs)}; first 20:")
            for m in motifs[:20]:
                print("   ", json.dumps(m)[:160])
        eff = _walk_numeric(r, ("effective", "order_mass", "k0", "k1", "k2", "k3", "k4"))
        for p, v in eff[:15]:
            print(f"  {p} = {v:.6g}")

    # ---------------- WS3 ----------------
    _section("WS3 detail (per-view top gains, TEST central table, chosen params)")
    r = _load(root / "ws3" / "ws3_report_real.json")
    if r:
        fg = r.get("feature_gain", {})
        for view in ("U", "L1", "O", "OM"):
            gains = fg.get(view, {})
            if isinstance(gains, dict) and gains:
                top = sorted(gains.items(), key=lambda kv: -float(kv[1]))[:10]
                print(f"  top-10 gain [{view}]: " + ", ".join(f"{k}={float(v):,.0f}" for k, v in top))
        for split in ("val", "test"):
            blk = r.get(split, {})
            print(f"\n  {split.upper()} central: {json.dumps(blk.get('central', {}))[:600]}")
            print(f"  {split.upper()} deltas : {json.dumps(blk.get('deltas', {}))[:600]}")

    # WS3 sidecars: chosen hyperparameters per view.
    _section("WS3 chosen hyperparameters (model sidecars)")
    ws3 = root / "ws3"
    for p in sorted(ws3.glob("behavior_*.json")) + sorted(ws3.glob("outcome_*.json")):
        s = _load(p)
        if not s:
            continue
        chosen = s.get("chosen_params") or s.get("params") or {}
        n_par = s.get("n_params", "?")
        keep = {k: chosen[k] for k in ("num_leaves", "min_child_samples", "learning_rate",
                                       "n_estimators", "best_iteration") if k in chosen}
        print(f"  {p.name}: n_params={n_par} {keep if keep else list(chosen)[:6]}")

    print("\n(done - paste this whole digest back)")


if __name__ == "__main__":
    main()
