"""Runnable WS1 step: fit the empirical-Bayes tables and score them through the harness.

``python workstreams/ws1_eb_tables/run_ws1.py --table data/processed/decision_table.parquet
--out results/ws1/ [--views C U L1 O] [--target both] [--synth null|positive|off]``

Behaviour (decision D12 conventions -- argparse, ``pathlib``, a ``__main__`` guard, run
metadata, checkpoint-friendly per-view outputs):

1. Load the decision table (``--table``) or generate a synthetic world (``--synth``); the
   synthetic path is what Phase-1 CI exercises, the real table is Phase 2.
2. Temporal split from config via :func:`pitchseq.splits.make_splits` -- train on the train
   seasons, evaluate on the validation season.
3. Fit the selection (Dirichlet-multinomial) and/or run-value (normal partial-pooling)
   tables per view (:mod:`.model`).
4. Emit standard-schema predictions (:mod:`pitchseq.eval.predictions`) per view and score
   them **only** through the shared harness (:func:`pitchseq.eval.harness`): per-view
   selection log loss, run-value MAE, and the SPEC ``6`` ``Delta_order`` ablation with a
   clustered CI. The four count-based references (:mod:`pitchseq.eval.baselines`) are scored
   the same way, so WS1 -- the serious hierarchical version -- is compared like-for-like to
   the quick references.
5. Write per-view predictions parquet, a report JSON, a support-diagnostics CSV and a run
   metadata sidecar under ``--out``.
6. Print a compact headline block (per-view losses vs references, ``Delta_order`` annotated
   per **decision D21**, ``Delta_matchup`` = N/A because OM tables are infeasible per D25,
   the support summary, positive-world recovery where applicable, elapsed, peak RAM).

Import strategy (Windows-safe): the repository root is inserted on ``sys.path`` so the
``from workstreams.ws1_eb_tables.model import ...`` package import resolves whether this file
is launched as a script (``python workstreams/ws1_eb_tables/run_ws1.py``) or imported as a
module (the test suite adds the same root via the repository-root ``conftest.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pitchseq.config import load_config  # noqa: E402
from pitchseq.eval.baselines import build_baselines  # noqa: E402
from pitchseq.eval.falsification import velo_gap_prev_transition  # noqa: E402
from pitchseq.eval.harness import compare_views, evaluate_predictions, write_report  # noqa: E402
from pitchseq.eval.predictions import ACTION_PROB_COLS, save_predictions  # noqa: E402
from pitchseq.families import FAMILIES  # noqa: E402
from pitchseq.runmeta import track_run, write_runmeta  # noqa: E402
from pitchseq.splits import make_splits  # noqa: E402
from workstreams.ws1_eb_tables.model import (  # noqa: E402
    SELECTION_VIEWS,
    fit,
    support_note,
    support_table,
)

__all__ = ["run_ws1", "main"]

_DEFAULT_VIEWS = ("C", "U", "L1", "O")  # SPEC 6 view ladder order
_REFERENCE_FOR = {"C": "pitcher_count", "L1": "pitcher_count_prev", "U": "pitcher_count_prev", "O": "pitcher_count_prev"}
_VELO_GAP_THRESHOLD = 5.0


# --- data ------------------------------------------------------------------------------

def _load_table(source: str, config: dict) -> pd.DataFrame:
    """Load a decision-table parquet (or a directory of season checkpoints)."""
    path = Path(source)
    if path.is_dir():
        frames = [pd.read_parquet(p, engine="pyarrow") for p in sorted(path.glob("*.parquet"))]
        if not frames:
            raise FileNotFoundError(f"no parquet files under {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(path, engine="pyarrow")


def _synth_table(world: str, n_games: int, seed: int, config: dict) -> tuple[pd.DataFrame, dict]:
    """Generate a synthetic world and build its decision table (Phase-1 CI path)."""
    from pitchseq.decision_table import build_decision_table
    from pitchseq.synth import make_null_world, make_positive_world

    if world == "null":
        raw, truth = make_null_world(n_games=n_games, seed=seed, innings_per_game=6)
    elif world == "positive":
        raw, truth = make_positive_world(
            n_games=n_games, seed=seed, innings_per_game=6,
            effect_size=0.30, velo_gap_threshold=_VELO_GAP_THRESHOLD,
        )
    else:
        raise ValueError(f"--synth must be null|positive|off, got {world!r}")
    return build_decision_table(raw), truth


# --- predictions -----------------------------------------------------------------------

def _prediction_df(
    table: pd.DataFrame, view: str, targets: list[str], seconds: float, peak_mem_mb: float,
    n_params: int, sel_proba=None, rv_mean=None, rv_sd=None,
) -> pd.DataFrame:
    """Assemble a standard-schema prediction table for one view (SPEC ``8.1``)."""
    df = pd.DataFrame({"row_id": table["row_id"].to_numpy()})
    if "selection" in targets and sel_proba is not None:
        for j, col in enumerate(ACTION_PROB_COLS):
            df[col] = sel_proba[:, j]
    if "run_value" in targets and rv_mean is not None:
        df["exp_reward"] = rv_mean
        df["exp_reward_sd"] = rv_sd
    df["model_id"] = "ws1_eb_tables"
    df["state_view"] = view
    df["seconds"] = float(seconds)
    df["peak_mem_mb"] = float(peak_mem_mb)
    df["n_params"] = int(n_params)
    return df


def _baseline_reports(train: pd.DataFrame, val: pd.DataFrame, config: dict, n_boot: int, seed: int) -> dict:
    """Selection log loss of the four count-based references, scored through the harness."""
    out = {}
    for name, mdl in build_baselines(alpha=8.0).items():
        mdl.fit(train)
        proba = mdl.predict_proba(val)
        df = pd.DataFrame({"row_id": val["row_id"].to_numpy()})
        for j, col in enumerate(ACTION_PROB_COLS):
            df[col] = proba[:, j]
        df["model_id"] = f"baseline_{name}"
        df["state_view"] = "NA"
        df["seconds"] = 0.0
        df["peak_mem_mb"] = 0.0
        df["n_params"] = 0
        rep = evaluate_predictions(df, val, config=config, n_boot=n_boot, seed=seed)
        out[name] = float(rep["slices"]["all"]["action_prob"]["log_loss"])
    return out


def _positive_recovery(train: pd.DataFrame, val: pd.DataFrame, views, threshold: float) -> dict:
    """Per-view predicted E[R] lift on the triggered (velo-transition) rows (D27).

    Restricted to ``pitch_number >= 3`` rows where the ordered transition
    ``|velo_{t-1} - velo_{t-2}|`` is defined -- the region the planted effect lives in. The
    edge is the predicted-reward difference between rows whose transition clears the
    threshold and those that do not; ``attenuation`` is that edge over the empirical
    reward lift on the same rows.
    """
    gap = velo_gap_prev_transition(val)
    long_pa = val["pitch_number"].to_numpy() >= 3
    trig = long_pa & (np.nan_to_num(gap) >= threshold)
    base = long_pa & ~trig
    R = pd.to_numeric(val["R"], errors="coerce").to_numpy(dtype=float)
    ok_t = trig & np.isfinite(R)
    ok_b = base & np.isfinite(R)
    if not ok_t.any() or not ok_b.any():
        return {}
    true_lift = float(np.mean(R[ok_t]) - np.mean(R[ok_b]))
    raw_edge = {}
    for view in views:
        mean, _ = fit(train, view, "run_value").predict_run_value(val)
        raw_edge[view] = float(np.mean(mean[trig]) - np.mean(mean[base]))
    c_edge = raw_edge.get("C", 0.0)
    per_view = {}
    for view in views:
        edge = raw_edge[view]
        per_view[view] = {
            "edge": edge,
            # The ordered contribution beyond the no-history baseline: only a view that can
            # represent the ordered transition (O) recovers reward the context view C cannot.
            "edge_vs_C": float(edge - c_edge),
            "attenuation": float(edge / true_lift) if true_lift != 0 else float("nan"),
        }
    return {"true_lift": true_lift, "n_triggered": int(trig.sum()), "n_base": int(base.sum()), "per_view": per_view}


# --- the pipeline ----------------------------------------------------------------------

def run_ws1(
    source: str | None = None,
    synth: str = "off",
    out: str = "results/ws1",
    views=_DEFAULT_VIEWS,
    target: str = "both",
    n_games: int = 90,
    seed: int = 7,
    n_boot: int = 200,
    config: dict | None = None,
    write_outputs: bool = True,
) -> dict:
    """Fit and score the WS1 tables end to end; return a report dict (no printing).

    Parameters
    ----------
    source : str, optional
        Decision-table parquet (or directory of season checkpoints). Required when
        ``synth == "off"``.
    synth : {'off', 'null', 'positive'}, optional
        Run on a generated synthetic world instead of a real table (Phase-1 CI path).
    out : str, optional
        Output directory for predictions / report / support diagnostics / run metadata.
    views : sequence of str, optional
        Feasible views to fit (default ``C, L1, U, O``; ``OM`` is infeasible per D25).
    target : {'both', 'selection', 'run_value'}, optional
        Which table families to fit and score.
    n_games, seed : int, optional
        Synthetic-world size / seed (ignored for a real table).
    n_boot : int, optional
        Cluster-bootstrap replicates for the CIs.
    config : dict, optional
        Study config; loaded from default when ``None``.
    write_outputs : bool, optional
        When ``True`` (default) write the parquet / JSON / CSV artefacts under ``out``.

    Returns
    -------
    dict
        The full report: per-view selection / run-value blocks, baseline references,
        ``delta_order`` (annotated per D21), the support table, positive-world recovery
        (when applicable), and run metadata.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))
    targets = ["selection", "run_value"] if target == "both" else [target]
    views = list(views)
    world_tag = synth if synth != "off" else "real"
    out_dir = Path(out)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    with track_run(step="ws1_eb_tables", world=world_tag, target=target) as meta:
        # --- data + temporal split ---
        truth: dict = {}
        if synth != "off":
            table, truth = _synth_table(synth, n_games, seed, config)
        else:
            if source is None:
                raise ValueError("--table is required when --synth is off")
            table = _load_table(source, config)
        splits = make_splits(table, config)["primary"]
        train = table.loc[splits["train"].to_numpy()].reset_index(drop=True)
        val = table.loc[splits["val"].to_numpy()].reset_index(drop=True)
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"empty split (train={len(train)}, val={len(val)}); check seasons")

        # --- fit + predict per view ---
        pred_by_view: dict[str, pd.DataFrame] = {}
        selection_block: dict[str, dict] = {}
        run_value_block: dict[str, dict] = {}
        fitted_meta: dict[str, dict] = {}
        for view in views:
            sel_proba = rv_mean = rv_sd = None
            n_params = 0
            info: dict = {}
            with track_run(step="ws1_fit", view=view) as vmeta:
                if "selection" in targets:
                    sm = fit(train, view, "selection")
                    sel_proba = sm.predict_proba(val)
                    info["selection"] = {
                        "alpha": sm.concentrations_, "alpha_method": sm.concentration_methods_,
                        "n_cells": sm.n_cells_, "backoff": dict(sm.backoff_counts_),
                    }
                    n_params += sm.n_params
                if "run_value" in targets:
                    rm = fit(train, view, "run_value")
                    rv_mean, rv_sd = rm.predict_run_value(val)
                    info["run_value"] = {
                        "kappa": rm.concentrations_, "kappa_method": rm.concentration_methods_,
                        "sigma2": rm.sigma2_, "n_cells": rm.n_cells_, "backoff": dict(rm.backoff_counts_),
                    }
                    n_params += rm.n_params
            df = _prediction_df(
                val, view, targets, vmeta["seconds"], vmeta["peak_mem_mb"], n_params,
                sel_proba=sel_proba, rv_mean=rv_mean, rv_sd=rv_sd,
            )
            pred_by_view[view] = df
            fitted_meta[view] = info
            if write_outputs:
                save_predictions(df, out_dir / f"predictions_{world_tag}_{view}.parquet")

            report = evaluate_predictions(df, val, config=config, n_boot=n_boot, seed=seed)
            allslice = report["slices"]["all"]
            if "selection" in targets:
                selection_block[view] = allslice["action_prob"]
            if "run_value" in targets:
                run_value_block[view] = allslice["exp_reward"]

        # --- baselines + ablation deltas (scored through the harness) ---
        baseline_losses = {}
        delta_order = None
        if "selection" in targets:
            baseline_losses = _baseline_reports(train, val, config, n_boot, seed)
            if {"U", "L1", "O"} <= set(views):
                cv = compare_views(pred_by_view, val, config=config, target="family", n_boot=n_boot, seed=seed)
                delta_order = {
                    "point": cv["deltas"]["delta_order"],
                    "ci": cv["delta_order_ci"],
                    "losses": cv["losses"],
                    # D21: Delta_order is negatively biased under the null (min of two noisy
                    # losses is optimistic); the criterion is "not significantly positive",
                    # and a small negative value reads as consistent with no ordering effect.
                    "significantly_positive": bool(cv["delta_order_ci"]["lo"] > 0),
                    "interpretation": _annotate_delta_order(cv["deltas"]["delta_order"], cv["delta_order_ci"]),
                }

        # --- support diagnostics (the WS1 exhibit) ---
        support = support_table(train, val, views=views)
        support_records = support.to_dict(orient="records")

        # --- positive-world recovery (D27) ---
        recovery = {}
        if synth == "positive" and "run_value" in targets:
            recovery = _positive_recovery(train, val, views, _VELO_GAP_THRESHOLD)

        meta["n_train"] = int(len(train))
        meta["n_val"] = int(len(val))
        meta["views"] = views

    result = {
        "world": world_tag,
        "target": target,
        "views": views,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "selection": selection_block,
        "run_value": run_value_block,
        "baseline_selection_loss": baseline_losses,
        "reference_for": {v: _REFERENCE_FOR.get(v) for v in views},
        "delta_order": delta_order,
        "delta_matchup": None,  # OM tables are infeasible for pure tables (D25).
        "om_support_note": support_note(train),
        "support": support_records,
        "fitted": fitted_meta,
        "positive_recovery": recovery,
        "truth": truth,
        "seconds": float(meta.get("seconds", float("nan"))),
        "peak_mem_mb": float(meta.get("peak_mem_mb", float("nan"))),
    }

    if write_outputs:
        write_report(result, out_dir / f"ws1_report_{world_tag}.json")
        support.to_csv(out_dir / f"support_{world_tag}.csv", index=False)
        write_runmeta(out_dir / f"ws1_{world_tag}.runmeta.json", dict(meta))
        result["outputs"] = {
            "report": str(out_dir / f"ws1_report_{world_tag}.json"),
            "support_csv": str(out_dir / f"support_{world_tag}.csv"),
            "runmeta": str(out_dir / f"ws1_{world_tag}.runmeta.json"),
            "predictions_dir": str(out_dir),
        }
    return result


def _annotate_delta_order(delta: float, ci: dict) -> str:
    """One-line D21 reading of a Delta_order estimate."""
    if ci.get("lo", 0.0) > 0:
        return "significantly positive: ordered history improves selection out-of-sample"
    if delta < 0:
        return "small negative: consistent with no ordering effect (D21; min[U,L1] is optimistic)"
    return "not significantly positive: no evidence ordered selection helps beyond U/L1 (D21)"


# --- headline --------------------------------------------------------------------------

def _fmt(x, spec="+.4f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "  n/a "


def _format_headline(result: dict) -> str:
    width = 76
    lines = ["=" * width, " WS1 empirical-Bayes tables - headline", "=" * width]
    lines.append(f" world / target : {result['world']} / {result['target']}")
    lines.append(f" rows           : train={result['n_train']:,}  val={result['n_val']:,}")

    sel = result.get("selection", {})
    if sel:
        lines.append(" selection log loss (val) vs count-based references:")
        bl = result.get("baseline_selection_loss", {})
        for view in result["views"]:
            if view not in sel:
                continue
            ll = sel[view]["log_loss"]
            ref_name = result["reference_for"].get(view)
            ref = bl.get(ref_name)
            mark = ""
            if ref is not None:
                mark = f"  ref[{ref_name}]={_fmt(ref, '.4f')}  ({'<=' if ll <= ref + 1e-9 else '>'} ref)"
            lines.append(f"   {view:<3}: {_fmt(ll, '.4f')}{mark}")
        if bl:
            lines.append("   references     : " + "  ".join(f"{k}={_fmt(v, '.4f')}" for k, v in bl.items()))

    rv = result.get("run_value", {})
    if rv:
        lines.append(" run-value MAE (val):")
        for view in result["views"]:
            if view in rv and "mae" in rv[view]:
                lines.append(f"   {view:<3}: MAE={_fmt(rv[view]['mae'], '.4f')}  RMSE={_fmt(rv[view].get('rmse'), '.4f')}")

    do = result.get("delta_order")
    if do is not None:
        ci = do["ci"]
        lines.append(
            f" Delta_order    : {_fmt(do['point'])}  CI[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}]"
        )
        lines.append(f"                  {do['interpretation']}")
    lines.append(" Delta_matchup  : N/A - OM tables infeasible for pure tables (D25 support problem)")

    rec = result.get("positive_recovery")
    if rec and rec.get("per_view"):
        lines.append(
            f" positive recovery (true reward lift={_fmt(rec['true_lift'])}, "
            f"n_trig={rec['n_triggered']}):"
        )
        for view in result["views"]:
            if view in rec["per_view"]:
                pv = rec["per_view"][view]
                lines.append(
                    f"   {view:<3}: E[R] edge={_fmt(pv['edge'], '+.5f')}  vs-C={_fmt(pv['edge_vs_C'], '+.5f')}  "
                    f"attenuation={_fmt(pv['attenuation'], '.3f')}"
                )

    lines.append(" support (distinct cells / eval frac n<20 / deepest-level backoff rate):")
    for rec_s in result.get("support", []):
        deep = None
        for key in ("backoff_history", "backoff_pitcher"):
            if key in rec_s and not (isinstance(rec_s[key], float) and np.isnan(rec_s[key])):
                deep = rec_s[key]
                break
        lines.append(
            f"   {rec_s['view']:<3}: cells={int(rec_s['n_cells']):>6}  "
            f"n<20={_fmt(rec_s['frac_eval_lt20'], '.3f')}  deepest-hit={_fmt(deep, '.3f')}"
        )

    lines.append(f" elapsed (s)    : {_fmt(result.get('seconds'), '.1f')}")
    lines.append(f" peak mem (MB)  : {_fmt(result.get('peak_mem_mb'), '.1f')}")
    if result.get("outputs"):
        lines.append(f" outputs        : {result['outputs']['report']}")
    lines.append("=" * width)
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python workstreams/ws1_eb_tables/run_ws1.py",
        description="Fit and score the WS1 empirical-Bayes tables (SPEC 12.1).",
    )
    parser.add_argument("--table", type=str, default=None,
                        help="Decision-table parquet (or directory of season checkpoints).")
    parser.add_argument("--out", type=str, default="results/ws1",
                        help="Output directory for predictions / report / support / runmeta.")
    parser.add_argument("--views", type=str, nargs="*", default=list(_DEFAULT_VIEWS),
                        help=f"Views to fit (default {list(_DEFAULT_VIEWS)}; OM is infeasible).")
    parser.add_argument("--target", type=str, default="both",
                        choices=["both", "selection", "run_value"],
                        help="Which table families to fit / score.")
    parser.add_argument("--synth", type=str, default="off", choices=["off", "null", "positive"],
                        help="Run on a generated synthetic world instead of --table.")
    parser.add_argument("--n-games", type=int, default=200, help="Synthetic-world size.")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic-world seed.")
    parser.add_argument("--n-boot", type=int, default=200, help="Cluster-bootstrap replicates for CIs.")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point: run the pipeline and print the headline block."""
    args = _parse_args(argv)
    bad = [v for v in args.views if v not in SELECTION_VIEWS]
    if bad:
        print(f"error: unsupported view(s) {bad}; choose from {list(SELECTION_VIEWS)} (OM is infeasible)",
              file=sys.stderr)
        raise SystemExit(2)
    if args.synth == "off" and args.table is None:
        print("error: one of --table or --synth is required", file=sys.stderr)
        raise SystemExit(2)

    result = run_ws1(
        source=args.table, synth=args.synth, out=args.out, views=args.views,
        target=args.target, n_games=args.n_games, seed=args.seed, n_boot=args.n_boot,
    )
    print(_format_headline(result))
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    main()
