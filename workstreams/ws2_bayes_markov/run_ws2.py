"""Runnable WS2 step: fit the Bayesian variable-order Markov grammar and score it.

``python workstreams/ws2_bayes_markov/run_ws2.py --table data/processed/decision_table.parquet
--out results/ws2/ [--views C L1 O] [--k-max 4] [--synth null|positive|off]``

Behaviour (decision D12 conventions -- argparse, ``pathlib``, a ``__main__`` guard, run
metadata, checkpoint-friendly per-view outputs; decisions D29-D31):

1. Load the decision table (``--table``) or generate a synthetic world (``--synth``); the
   synthetic path is what Phase-1 CI exercises, the real table is Phase 2.
2. Temporal split from config via :func:`pitchseq.splits.make_splits` -- train on the train
   seasons, evaluate on the validation season.
3. Fit the grammar per view (:mod:`.model`): ``C`` (order 0), ``L1`` (order 1), ``O``
   (variable order ``<= K_MAX``). ``U`` / ``OM`` are not-applicable for a Markov grammar
   (D29) and are reported as such, never fitted.
4. Emit standard-schema selection predictions (:mod:`pitchseq.eval.predictions`) per view and
   score them **only** through the shared harness (:func:`pitchseq.eval.harness`): per-view
   selection log loss over every reporting slice, against the count-based references
   (:mod:`pitchseq.eval.baselines`). ``compare_views`` needs a ``U`` view, which WS2 does not
   have, so the order edge is computed directly as ``delta_order_L1 = Loss(L1) - Loss(O)``
   with a clustered CI using the shared metric functions (never abusing the ``U`` slot).
5. Report the SPEC ``10`` bits-of-predictability ``B_seq`` (decision D31) from the O-view and
   C-view predictions via :func:`pitchseq.eval.predictability.bits_of_predictability`, sliced
   overall / by pitch-number / by count-bucket / two-strike / three-ball, plus the grammar
   exhibits (effective-order distribution, top motifs, order usage).
6. Run the D30 acceptance wiring: **detect** the grammar (effective order ``>= 2`` mass
   material, repeat-suppression motif present, O beats L1 on selection log loss) and the
   internal negative control -- refit / re-evaluate on stratified-permuted histories, where
   the O-vs-L1 edge must **collapse** and effective order concentrate at ``<= 1``. Both
   verdicts (``GRAMMAR_DETECTED`` / ``COLLAPSES_UNDER_PERMUTATION``) print in the headline.
7. Write per-view predictions parquet, a report JSON, a motifs CSV and a run-metadata sidecar
   under ``--out``; print a compact headline block.

Import strategy (Windows-safe): the repository root is inserted on ``sys.path`` so the
``from workstreams.ws2_bayes_markov.model import ...`` package import resolves whether this
file is launched as a script or imported as a module.
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
from pitchseq.eval.harness import evaluate_predictions, write_report  # noqa: E402
from pitchseq.eval.metrics import clustered_ci, log_loss_per_row  # noqa: E402
from pitchseq.eval.predictability import bits_of_predictability, bits_summary  # noqa: E402
from pitchseq.eval.predictions import ACTION_PROB_COLS, save_predictions  # noqa: E402
from pitchseq.families import FAMILIES  # noqa: E402
from pitchseq.runmeta import track_run, write_runmeta  # noqa: E402
from pitchseq.splits import make_splits  # noqa: E402
from workstreams.ws2_bayes_markov.model import (  # noqa: E402
    GRAMMAR_VIEWS,
    K_MAX_DEFAULT,
    count_hand_code,
    fit,
    lag_family_codes,
    not_applicable_note,
    permute_contexts_within_strata,
)

__all__ = ["run_ws2", "main"]

_DEFAULT_VIEWS = ("C", "L1", "O")
_REFERENCE_FOR = {"C": "pitcher_count", "L1": "pitcher_count_prev", "O": "pitcher_count_prev"}
_LABELS = list(FAMILIES)

# D30 detection / collapse thresholds (documented; calibrated on the null world).
_EFF_ORDER_MASS_MIN = 0.03   # effective order >=2 mass this large counts as "material"
_COLLAPSE_MASS_MAX = 0.03    # effective order >=2 mass this small counts as "collapsed"
_MOTIF_MIN_SUPPORT = 30
_MOTIF_TOP_N = 20


# --- data ------------------------------------------------------------------------------

def _load_table(source: str) -> pd.DataFrame:
    """Load a decision-table parquet (or a directory of season checkpoints)."""
    path = Path(source)
    if path.is_dir():
        frames = [pd.read_parquet(p, engine="pyarrow") for p in sorted(path.glob("*.parquet"))]
        if not frames:
            raise FileNotFoundError(f"no parquet files under {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(path, engine="pyarrow")


def _synth_table(world: str, n_games: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Generate a synthetic world and build its decision table (Phase-1 CI path)."""
    from pitchseq.decision_table import build_decision_table
    from pitchseq.synth import make_null_world, make_positive_world

    if world == "null":
        raw, truth = make_null_world(n_games=n_games, seed=seed, innings_per_game=6)
    elif world == "positive":
        raw, truth = make_positive_world(n_games=n_games, seed=seed, innings_per_game=6)
    else:
        raise ValueError(f"--synth must be null|positive|off, got {world!r}")
    return build_decision_table(raw), truth


# --- predictions -----------------------------------------------------------------------

def _prediction_df(table: pd.DataFrame, view: str, proba: np.ndarray, seconds: float,
                   peak_mem_mb: float, n_params: int) -> pd.DataFrame:
    """Assemble a standard-schema selection prediction table for one view (SPEC ``8.1``)."""
    df = pd.DataFrame({"row_id": table["row_id"].to_numpy()})
    for j, col in enumerate(ACTION_PROB_COLS):
        df[col] = proba[:, j]
    df["model_id"] = "ws2_bayes_markov"
    df["state_view"] = view
    df["seconds"] = float(seconds)
    df["peak_mem_mb"] = float(peak_mem_mb)
    df["n_params"] = int(n_params)
    return df


def _baseline_losses(train: pd.DataFrame, val: pd.DataFrame, config: dict, n_boot: int, seed: int) -> dict:
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


# --- order edge (delta_order_L1) -------------------------------------------------------

def _order_edge(y: np.ndarray, proba_l1: np.ndarray, proba_o: np.ndarray, clusters: pd.DataFrame,
                n_boot: int, seed: int) -> dict:
    r"""``delta_order_L1 = Loss(L1) - Loss(O)`` with a clustered CI (D21-adapted).

    WS2 has no ``U`` view, so the shared ``compare_views`` (which forms ``min[U, L1]``) does
    not apply; the WS2 order edge is the direct L1-vs-O selection log-loss difference. Unlike
    the SPEC ``6`` ``Delta_order`` it carries no ``min`` bias, so its clustered CI is read
    directly: a positive lower bound is genuine ordered selection structure beyond the
    previous pitch.
    """
    loss_l1 = log_loss_per_row(y, proba_l1, _LABELS)
    loss_o = log_loss_per_row(y, proba_o, _LABELS)

    def metric(idx):
        return float(loss_l1[idx].mean() - loss_o[idx].mean())

    ci = clustered_ci(metric, clusters, n_boot=n_boot, seed=seed)
    return {"point": float(loss_l1.mean() - loss_o.mean()), "ci": ci,
            "loss_L1": float(loss_l1.mean()), "loss_O": float(loss_o.mean()),
            "significantly_positive": bool(ci["lo"] > 0)}


# --- bits of predictability (D31) ------------------------------------------------------

def _count_bucket(val: pd.DataFrame) -> np.ndarray:
    """Coarse count bucket per row: ``ahead`` / ``even`` / ``behind`` (pitcher's view)."""
    balls = pd.to_numeric(val["balls"], errors="coerce").fillna(0).to_numpy()
    strikes = pd.to_numeric(val["strikes"], errors="coerce").fillna(0).to_numpy()
    out = np.full(len(val), "even", dtype=object)
    out[strikes > balls] = "ahead"
    out[balls > strikes] = "behind"
    return out


def _bits_report(val: pd.DataFrame, q_o: np.ndarray, q_c: np.ndarray) -> dict:
    """``B_seq`` overall and sliced per D31 (pitch-number, count-bucket, two-strike, three-ball)."""
    y = val["family"].astype("object").to_numpy()
    bits = bits_of_predictability(q_o, q_c, y)
    work = pd.DataFrame({
        "pitch_number": pd.to_numeric(val["pitch_number"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "count_bucket": _count_bucket(val),
    })
    strikes = pd.to_numeric(val["strikes"], errors="coerce").fillna(0).to_numpy()
    balls = pd.to_numeric(val["balls"], errors="coerce").fillna(0).to_numpy()
    two_strike = strikes >= 2
    three_ball = balls >= 3

    by_pn = bits_summary(work, bits, by=["pitch_number"])
    by_cb = bits_summary(work, bits, by=["count_bucket"])
    return {
        "overall": float(bits.mean()),
        "by_pitch_number": {int(r.pitch_number): float(r.mean_bits) for r in by_pn.itertuples()},
        "by_count_bucket": {str(r.count_bucket): float(r.mean_bits) for r in by_cb.itertuples()},
        "two_strike": float(bits[two_strike].mean()) if two_strike.any() else float("nan"),
        "three_ball": float(bits[three_ball].mean()) if three_ball.any() else float("nan"),
    }


# --- grammar exhibits + permutation control (D30) --------------------------------------

def _has_repeat_suppression(motifs: list[dict]) -> bool:
    """A same-family run whose repeated family is suppressed vs its suffix (no-three-in-a-row)."""
    for m in motifs:
        ctx = m["context"]
        if m["depth"] >= 2 and len(set(ctx)) == 1 and m["direction"] == "suppress" and m["top_family"] == ctx[-1]:
            return True
    return False


def _permutation_control(train: pd.DataFrame, val: pd.DataFrame, k_max: int, seed: int,
                        clusters: pd.DataFrame, n_boot: int, proba_c: np.ndarray | None) -> dict:
    """Refit L1 / O on stratified-permuted histories; the ordered edge must collapse (D30).

    Histories are permuted among rows sharing ``(count x hand cell, pitch_number)`` -- removing
    all within-PA ordered dependence while preserving the count x hand mix and PA depth -- and
    the grammar is refit and re-evaluated on the permuted contexts. A genuine ordered edge, a
    genuine ``B_seq`` and a genuine effective order all collapse toward the base.
    """
    lag_tr = lag_family_codes(train, k_max)
    lag_va = lag_family_codes(val, k_max)
    ch_tr, ch_va = count_hand_code(train), count_hand_code(val)
    pn_tr = pd.to_numeric(train["pitch_number"], errors="coerce").fillna(0).to_numpy()
    pn_va = pd.to_numeric(val["pitch_number"], errors="coerce").fillna(0).to_numpy()
    lp_tr = permute_contexts_within_strata(lag_tr, ch_tr, pn_tr, seed=seed)
    lp_va = permute_contexts_within_strata(lag_va, ch_va, pn_va, seed=seed)

    m_l1 = fit(train, "L1", k_max=k_max, lag_codes=lp_tr)
    m_o = fit(train, "O", k_max=k_max, lag_codes=lp_tr)
    proba_l1 = m_l1.predict_proba(val, lag_codes=lp_va)
    proba_o = m_o.predict_proba(val, lag_codes=lp_va)
    y = val["family"].astype("object").to_numpy()
    edge = _order_edge(y, proba_l1, proba_o, clusters, n_boot, seed)
    eff = m_o.effective_order(val, lag_codes=lp_va)
    out = {"delta_order_L1": edge, "effective_order": {k: v for k, v in eff.items() if k != "per_row"}}
    if proba_c is not None:  # B_seq against the (context-only) C base collapses too.
        out["b_seq"] = float(bits_of_predictability(proba_o, proba_c, y).mean())
    return out


# --- the pipeline ----------------------------------------------------------------------

def run_ws2(
    source: str | None = None,
    synth: str = "off",
    out: str = "results/ws2",
    views=_DEFAULT_VIEWS,
    k_max: int = K_MAX_DEFAULT,
    n_games: int = 300,
    seed: int = 7,
    n_boot: int = 200,
    config: dict | None = None,
    write_outputs: bool = True,
) -> dict:
    """Fit and score the WS2 grammar end to end; return a report dict (no printing).

    Parameters
    ----------
    source : str, optional
        Decision-table parquet (or directory of season checkpoints). Required when
        ``synth == "off"``.
    synth : {'off', 'null', 'positive'}, optional
        Run on a generated synthetic world instead of a real table (Phase-1 CI path). The null
        world is WS2's positive control (D30): it plants an order-2 selection habit.
    out : str, optional
        Output directory for predictions / report / motifs / run metadata.
    views : sequence of str, optional
        Feasible views to fit (default ``C, L1, O``; ``U`` / ``OM`` are not-applicable per D29).
    k_max : int, optional
        Grammar-order cap for the ``O`` view (decision D29; default 4).
    n_games, seed : int, optional
        Synthetic-world size / seed (ignored for a real table); ``seed`` also drives the CI
        bootstrap and the permutation control.
    n_boot : int, optional
        Cluster-bootstrap replicates for the CIs.
    config : dict, optional
        Study config; loaded from default when ``None``.
    write_outputs : bool, optional
        When ``True`` (default) write the parquet / JSON / CSV artefacts under ``out``.

    Returns
    -------
    dict
        The full report: per-view selection blocks, baseline references, ``delta_order_L1`` +
        CI, ``B_seq`` (sliced), grammar exhibits, the D30 detect / collapse verdicts, and run
        metadata.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))
    views = [v for v in views]
    world_tag = synth if synth != "off" else "real"
    out_dir = Path(out)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    with track_run(step="ws2_bayes_markov", world=world_tag, k_max=k_max) as meta:
        # --- data + temporal split ---
        truth: dict = {}
        if synth != "off":
            table, truth = _synth_table(synth, n_games, seed)
        else:
            if source is None:
                raise ValueError("--table is required when --synth is off")
            table = _load_table(source)
        splits = make_splits(table, config)["primary"]
        train = table.loc[splits["train"].to_numpy()].reset_index(drop=True)
        val = table.loc[splits["val"].to_numpy()].reset_index(drop=True)
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"empty split (train={len(train)}, val={len(val)}); check seasons")
        clusters = val[["pitcher", "game_pk"]].reset_index(drop=True)
        y_val = val["family"].astype("object").to_numpy()

        # --- fit + predict per view ---
        pred_by_view: dict[str, pd.DataFrame] = {}
        proba_by_view: dict[str, np.ndarray] = {}
        selection_block: dict[str, dict] = {}
        fitted_meta: dict[str, dict] = {}
        models: dict[str, object] = {}
        for view in views:
            with track_run(step="ws2_fit", view=view) as vmeta:
                model = fit(train, view, k_max=k_max)
                proba = model.predict_proba(val)
            models[view] = model
            proba_by_view[view] = proba
            df = _prediction_df(val, view, proba, vmeta["seconds"], vmeta["peak_mem_mb"], model.n_params)
            pred_by_view[view] = df
            fitted_meta[view] = {"concentrations": model.concentrations_,
                                 "methods": model.concentration_methods_, "n_cells": model.n_cells_}
            if write_outputs:
                save_predictions(df, out_dir / f"predictions_{world_tag}_{view}.parquet")
            report = evaluate_predictions(df, val, config=config, n_boot=n_boot, seed=seed)
            selection_block[view] = report["slices"]["all"]["action_prob"]

        # --- baselines ---
        baseline_losses = _baseline_losses(train, val, config, n_boot, seed)

        # --- order edge (delta_order_L1) ---
        delta_order = None
        if {"L1", "O"} <= set(views):
            delta_order = _order_edge(y_val, proba_by_view["L1"], proba_by_view["O"], clusters, n_boot, seed)

        # --- bits of predictability (D31) ---
        bits = None
        if {"C", "O"} <= set(views):
            bits = _bits_report(val, proba_by_view["O"], proba_by_view["C"])

        # --- grammar exhibits (from the O grammar) ---
        exhibits, motifs = {}, []
        perm = None
        detect = None
        if "O" in views:
            o_model = models["O"]
            o_model.predict_proba(val)  # populate grammar_depth_counts_
            eff = o_model.effective_order(val)
            motifs = o_model.top_motifs(min_support=_MOTIF_MIN_SUPPORT, top_n=_MOTIF_TOP_N)
            exhibits = {
                "effective_order": {k: v for k, v in eff.items() if k != "per_row"},
                "order_usage": o_model.order_usage_summary(),
                "n_motifs": len(motifs),
            }
            # --- D30 negative control: stratified history permutation ---
            if {"L1", "O"} <= set(views):
                proba_c = proba_by_view.get("C")
                perm = _permutation_control(train, val, k_max, seed, clusters, n_boot, proba_c)
                detect = _verdicts(delta_order, eff, motifs, perm)

        meta["n_train"] = int(len(train))
        meta["n_val"] = int(len(val))
        meta["views"] = views

    result = {
        "world": world_tag,
        "views": views,
        "k_max": int(k_max),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "selection": selection_block,
        "baseline_selection_loss": baseline_losses,
        "reference_for": {v: _REFERENCE_FOR.get(v) for v in views},
        "delta_order_L1": delta_order,
        "delta_matchup": None,  # OM is not-applicable for a Markov grammar (D29).
        "not_applicable": {v: not_applicable_note(v) for v in ("U", "OM")},
        "bits_of_predictability": bits,
        "grammar_exhibits": exhibits,
        "top_motifs": motifs,
        "permutation_control": perm,
        "verdicts": detect,
        "fitted": fitted_meta,
        "truth": truth,
        "seconds": float(meta.get("seconds", float("nan"))),
        "peak_mem_mb": float(meta.get("peak_mem_mb", float("nan"))),
    }

    if write_outputs:
        write_report(result, out_dir / f"ws2_report_{world_tag}.json")
        if motifs:
            pd.DataFrame(motifs).to_csv(out_dir / f"motifs_{world_tag}.csv", index=False)
        write_runmeta(out_dir / f"ws2_{world_tag}.runmeta.json", dict(meta))
        result["outputs"] = {
            "report": str(out_dir / f"ws2_report_{world_tag}.json"),
            "motifs_csv": str(out_dir / f"motifs_{world_tag}.csv"),
            "runmeta": str(out_dir / f"ws2_{world_tag}.runmeta.json"),
            "predictions_dir": str(out_dir),
        }
    return result


def _verdicts(delta_order: dict, eff: dict, motifs: list[dict], perm: dict) -> dict:
    """The two D30 acceptance verdicts (grammar detected; collapses under permutation)."""
    o_beats_l1 = bool(delta_order["point"] > 0)
    order_mass = float(eff["frac_ge2"])
    repeat = _has_repeat_suppression(motifs)
    detected = bool(o_beats_l1 and order_mass > _EFF_ORDER_MASS_MIN and repeat)

    perm_edge = float(perm["delta_order_L1"]["point"])
    perm_mass = float(perm["effective_order"]["frac_ge2"])
    real_edge = float(delta_order["point"])
    edge_collapsed = bool(perm_edge < max(0.5 * real_edge, 1e-9) or perm_edge < 0.002)
    collapsed = bool(edge_collapsed and perm_mass < _COLLAPSE_MASS_MAX)
    return {
        "grammar_detected": detected,
        "grammar_detected_verdict": "GRAMMAR_DETECTED" if detected else "GRAMMAR_NOT_DETECTED",
        "detected_components": {"o_beats_l1": o_beats_l1, "order_ge2_mass": order_mass,
                                "repeat_suppression_motif": repeat},
        "collapses_under_permutation": collapsed,
        "collapse_verdict": "COLLAPSES_UNDER_PERMUTATION" if collapsed else "PERMUTATION_EDGE_PERSISTS",
        "collapse_components": {"perm_edge": perm_edge, "real_edge": real_edge, "perm_order_ge2_mass": perm_mass},
    }


# --- headline --------------------------------------------------------------------------

def _fmt(x, spec="+.4f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "  n/a "


def _format_headline(result: dict) -> str:
    width = 76
    lines = ["=" * width, " WS2 Bayesian variable-order Markov grammar - headline", "=" * width]
    lines.append(f" world / K_MAX  : {result['world']} / {result['k_max']}")
    lines.append(f" rows           : train={result['n_train']:,}  val={result['n_val']:,}")

    sel = result.get("selection", {})
    bl = result.get("baseline_selection_loss", {})
    if sel:
        lines.append(" selection log loss (val) vs count-based references:")
        for view in result["views"]:
            if view not in sel:
                continue
            ll = sel[view]["log_loss"]
            ref_name = result["reference_for"].get(view)
            ref = bl.get(ref_name)
            mark = f"  ref[{ref_name}]={_fmt(ref, '.4f')}  ({'<=' if ref is not None and ll <= ref + 1e-9 else '>'} ref)" if ref is not None else ""
            lines.append(f"   {view:<3}: {_fmt(ll, '.4f')}{mark}")
        if bl:
            lines.append("   references     : " + "  ".join(f"{k}={_fmt(v, '.4f')}" for k, v in bl.items()))

    do = result.get("delta_order_L1")
    if do is not None:
        ci = do["ci"]
        lines.append(f" delta_order_L1 : {_fmt(do['point'])}  CI[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}]  "
                     f"(Loss(L1)-Loss(O); {'significant' if do['significantly_positive'] else 'not significant'})")
    lines.append(" delta_matchup  : N/A - OM is not-applicable for a Markov grammar (D29)")

    bits = result.get("bits_of_predictability")
    if bits is not None:
        lines.append(f" B_seq (bits)   : overall={_fmt(bits['overall'])}  "
                     f"two_strike={_fmt(bits['two_strike'])}  three_ball={_fmt(bits['three_ball'])}")
        cb = bits.get("by_count_bucket", {})
        if cb:
            lines.append("   by count-bucket: " + "  ".join(f"{k}={_fmt(v)}" for k, v in cb.items()))
        pn = bits.get("by_pitch_number", {})
        if pn:
            top = sorted(pn.items())[:5]
            lines.append("   by pitch_number: " + "  ".join(f"t{k}={_fmt(v, '+.3f')}" for k, v in top))

    ex = result.get("grammar_exhibits", {})
    eo = ex.get("effective_order", {})
    if eo:
        dist = eo.get("distribution", {})
        lines.append(f" effective order: mean={_fmt(eo.get('mean'), '.3f')}  "
                     f">=2 mass={_fmt(eo.get('frac_ge2'), '.3f')}  (tau={_fmt(eo.get('tau'), '.2f')})")
        lines.append("   distribution   : " + "  ".join(f"k{k}={_fmt(v, '.3f')}" for k, v in sorted(dist.items())))

    motifs = result.get("top_motifs", [])
    if motifs:
        lines.append(" top motifs (ordered context -> next-token lift vs suffix):")
        for m in motifs[:5]:
            lines.append(f"   {m['text']}")

    v = result.get("verdicts")
    if v is not None:
        lines.append(f" D30 detect     : {v['grammar_detected_verdict']}  "
                     f"(O<L1={v['detected_components']['o_beats_l1']}, "
                     f">=2 mass={_fmt(v['detected_components']['order_ge2_mass'], '.3f')}, "
                     f"repeat-motif={v['detected_components']['repeat_suppression_motif']})")
        cc = v["collapse_components"]
        lines.append(f" D30 control    : {v['collapse_verdict']}  "
                     f"(edge {_fmt(cc['real_edge'])} -> {_fmt(cc['perm_edge'])}, "
                     f">=2 mass -> {_fmt(cc['perm_order_ge2_mass'], '.3f')})")

    lines.append(f" elapsed (s)    : {_fmt(result.get('seconds'), '.1f')}")
    lines.append(f" peak mem (MB)  : {_fmt(result.get('peak_mem_mb'), '.1f')}")
    if result.get("outputs"):
        lines.append(f" outputs        : {result['outputs']['report']}")
    lines.append("=" * width)
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python workstreams/ws2_bayes_markov/run_ws2.py",
        description="Fit and score the WS2 Bayesian variable-order Markov grammar (SPEC 12.2).",
    )
    parser.add_argument("--table", type=str, default=None,
                        help="Decision-table parquet (or directory of season checkpoints).")
    parser.add_argument("--out", type=str, default="results/ws2",
                        help="Output directory for predictions / report / motifs / runmeta.")
    parser.add_argument("--views", type=str, nargs="*", default=list(_DEFAULT_VIEWS),
                        help=f"Views to fit (default {list(_DEFAULT_VIEWS)}; U/OM are not-applicable).")
    parser.add_argument("--k-max", type=int, default=K_MAX_DEFAULT, help="Grammar-order cap for the O view.")
    parser.add_argument("--synth", type=str, default="off", choices=["off", "null", "positive"],
                        help="Run on a generated synthetic world instead of --table.")
    parser.add_argument("--n-games", type=int, default=300, help="Synthetic-world size.")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic-world / bootstrap / permutation seed.")
    parser.add_argument("--n-boot", type=int, default=200, help="Cluster-bootstrap replicates for CIs.")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point: run the pipeline and print the headline block."""
    args = _parse_args(argv)
    bad = [v for v in args.views if v not in GRAMMAR_VIEWS]
    if bad:
        print(f"error: unsupported view(s) {bad}; choose from {list(GRAMMAR_VIEWS)} (U/OM are not-applicable)",
              file=sys.stderr)
        raise SystemExit(2)
    if args.synth == "off" and args.table is None:
        print("error: one of --table or --synth is required", file=sys.stderr)
        raise SystemExit(2)

    result = run_ws2(
        source=args.table, synth=args.synth, out=args.out, views=args.views,
        k_max=args.k_max, n_games=args.n_games, seed=args.seed, n_boot=args.n_boot,
    )
    print(_format_headline(result))
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    main()
