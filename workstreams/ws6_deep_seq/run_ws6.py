"""WS6 runnable pipeline -- capacity-matched GRU views over the SPEC 6 ladder (SPEC 12.4).

House conventions (decision D12): argparse, ``pathlib`` paths, a ``__main__`` guard (Windows
spawn), per-(view, target) checkpoint/resume, outputs under ``results/``/``artifacts/``, and a
compact **headline block** + wall-clock / peak RAM printed at the end (runmeta, D13). The
pipeline:

1. builds the decision table (``--table`` parquet or a generated ``--synth`` world) and the
   temporal split (train 2021-2023 / val 2024 / locked test 2025);
2. fits one capacity-matched :class:`~workstreams.ws6_deep_seq.model.SeqModel` per view
   (C/U/L1/O/OM) per target (``selection`` / ``outcome1``), early-stopping on the val fold;
3. writes standard-schema predictions and scores them **only** through the shared harness
   (``evaluate_predictions`` + ``compare_views`` -> Delta_order / Delta_matchup with clustered
   CIs) on val and the locked test;
4. runs the falsification battery on synthetic worlds (order ablation reuses the per-view
   fits; a token-order permutation control refits O via the D17 factory adapter) and derives
   the D47 verdicts, plus the lightweight **motif-rediscovery probe**;
5. prints the headline: per-view losses vs the count references (+ WS2/WS3 comparison
   placeholders), both deltas with D21 annotations, and the SPEC 7 **Pareto row**
   (params / epochs / wall-clock per view).

Torch lives in the ``[deep]`` extra (``pip install -e ".[deep]"``); ``--device auto`` runs on
CPU locally and a Colab T4 unchanged (D46).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pitchseq.config import load_config  # noqa: E402
from pitchseq.decision_table import build_decision_table  # noqa: E402
from pitchseq.eval.baselines import build_baselines  # noqa: E402
from pitchseq.eval.harness import compare_views, evaluate_predictions, write_report  # noqa: E402
from pitchseq.eval.predictions import ACTION_PROB_COLS, OUTCOME1_PROB_COLS, save_predictions  # noqa: E402
from pitchseq.runmeta import track_run, write_runmeta  # noqa: E402
from workstreams.ws6_deep_seq.model import (  # noqa: E402
    STATE_VIEWS,
    FeatureEncoders,
    SeqModel,
    _l1_twin_delta_ci,
    _repeat_context_mask,
    motif_repeat_probe,
    temporal_split_masks,
    ws6_permutation_test,
)
from pitchseq.eval.falsification import velo_gap_prev_transition  # noqa: E402

__all__ = ["run_ws6", "main"]

_DEFAULT_VIEWS = ("C", "U", "L1", "O", "OM")
_TARGETS = ("selection", "outcome1")
_VELO_GAP_THRESHOLD = 5.0
_EFFECT_SIZE = 0.30  # positive-world planted whiff boost (matches WS1/WS3's contrast fixture)
_MODEL_ID = "ws6_gru"

# Compact default GRU hyperparameters (D46: embed ~16, hidden ~64, <=2 layers).
_DEFAULT_HP = dict(embed_dim=16, hidden=64, num_layers=1, dropout=0.1, max_len=15,
                   lr=3e-3, weight_decay=1e-4, batch_size=512, patience=8, internal_val_frac=0.2)


# --- data ------------------------------------------------------------------------------

def _load_table(source: str) -> pd.DataFrame:
    """Load a decision-table parquet (or a directory of per-season checkpoints)."""
    p = Path(source)
    if p.is_dir():
        parts = sorted(p.glob("decision_table*.parquet"))
        if not parts:
            raise FileNotFoundError(f"no decision_table*.parquet under {p}")
        return pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
    return pd.read_parquet(p)


def _synth_table(world: str, n_games: int, seed: int, effect_size: float) -> tuple[pd.DataFrame, dict]:
    from pitchseq.synth import make_null_world, make_positive_world

    if world == "null":
        raw, truth = make_null_world(n_games=n_games, seed=seed)
    else:
        raw, truth = make_positive_world(n_games=n_games, seed=seed, effect_size=effect_size,
                                         velo_gap_threshold=_VELO_GAP_THRESHOLD)
    return build_decision_table(raw), truth


# --- checkpoints -----------------------------------------------------------------------

def _ckpt_stem(out_dir: Path, world_tag: str, target: str, view: str) -> Path:
    return out_dir / f"model_{world_tag}_{target}_{view}"


def _done_path(out_dir: Path, world_tag: str, target: str, view: str) -> Path:
    return out_dir / f".{world_tag}_{target}_{view}.done"


# --- predictions -----------------------------------------------------------------------

def _pred_cols(target: str) -> list[str]:
    return ACTION_PROB_COLS if target == "selection" else OUTCOME1_PROB_COLS


def _pred_frame(rows: pd.DataFrame, view: str, target: str, proba: np.ndarray, meta: dict) -> pd.DataFrame:
    df = pd.DataFrame({"row_id": rows["row_id"].to_numpy()})
    for j, col in enumerate(_pred_cols(target)):
        df[col] = proba[:, j]
    df["model_id"] = f"{_MODEL_ID}_{target}"
    df["state_view"] = view
    df["seconds"] = float(meta.get("seconds", 0.0))
    df["peak_mem_mb"] = float(meta.get("peak_mem_mb", 0.0))
    df["n_params"] = int(meta.get("n_params", 0))
    return df


# --- fit one target across views -------------------------------------------------------

def _fit_views(table, target, views, masks, hp, out_dir, world_tag, seed, device, force, write_outputs):
    """Fit one SeqModel per view for ``target``; return models, per-split predictions, losses.

    Early-stops on the val fold (SPEC 7: 2024 is the model-selection fold). Reuses a single
    fitted :class:`FeatureEncoders` per (view) so the static / matchup / sequence scaling is
    identical across splits. Checkpoints per (view) with a ``.done`` marker.
    """
    train = table.loc[masks["train"]]
    val = table.loc[masks["val"]]
    test = table.loc[masks["test"]] if masks["test"].any() else None

    models: dict = {}
    preds_val: dict = {}
    preds_test: dict = {}
    per_row_val: dict = {}
    pareto: dict = {}
    loaded: list = []
    fitted: list = []
    from pitchseq.eval.metrics import log_loss_per_row
    from workstreams.ws6_deep_seq.model import _target_labels, _target_values

    labels = _target_labels(target)
    y_val = _target_values(val, target)

    for view in views:
        stem = _ckpt_stem(out_dir, world_tag, target, view)
        done = _done_path(out_dir, world_tag, target, view)
        if done.exists() and not force and stem.with_suffix(".pt").exists():
            model = SeqModel.load(stem, device=device)
            loaded.append(view)
        else:
            fitted.append(view)
            with track_run(step="ws6_fit", view=view, target=target) as meta:
                enc = FeatureEncoders(max_len=hp["max_len"]).fit(train, fit_match=(view == "OM"))
                model = SeqModel(
                    view, target, seed=seed, device=device,
                    embed_dim=hp["embed_dim"], hidden=hp["hidden"], num_layers=hp["num_layers"],
                    dropout=hp["dropout"], max_len=hp["max_len"], lr=hp["lr"],
                    weight_decay=hp["weight_decay"], batch_size=hp["batch_size"],
                    max_epochs=hp["max_epochs"], patience=hp["patience"],
                    internal_val_frac=hp["internal_val_frac"],
                )
                model.fit_table(train, val, encoders=enc)
            model.meta_["seconds"] = float(meta["seconds"])
            model.meta_["peak_mem_mb"] = float(meta["peak_mem_mb"])
            if write_outputs:
                model.save(stem)
                done.write_text("ok", encoding="utf-8")
        models[view] = model
        pareto[view] = {"n_params": int(model.meta_.get("n_params", 0)),
                        "epochs": int(model.meta_.get("epochs", 0)),
                        "seconds": float(model.meta_.get("seconds", 0.0)),
                        "peak_mem_mb": float(model.meta_.get("peak_mem_mb", 0.0))}

        pv = model.predict_proba_table(val)
        preds_val[view] = _pred_frame(val, view, target, pv, model.meta_)
        per_row_val[view] = log_loss_per_row(y_val, pv, labels)
        if test is not None:
            pt = model.predict_proba_table(test)
            preds_test[view] = _pred_frame(test, view, target, pt, model.meta_)
        if write_outputs:
            save_predictions(preds_val[view], out_dir / f"pred_{target}_{world_tag}_{view}_val.parquet")
            if test is not None:
                save_predictions(preds_test[view], out_dir / f"pred_{target}_{world_tag}_{view}_test.parquet")

    return {"models": models, "preds_val": preds_val, "preds_test": preds_test,
            "per_row_val": per_row_val, "pareto": pareto, "loaded": loaded, "fitted": fitted}


# --- harness scoring -------------------------------------------------------------------

_HARNESS_TARGET = {"selection": "family", "outcome1": "outcome1"}


def _score(fit, table, masks, target, config, n_boot, seed) -> dict:
    """Central per-view losses + Delta_order / Delta_matchup on val and locked test."""
    ht = _HARNESS_TARGET[target]
    val = table.loc[masks["val"]]
    central = {}
    for view, pdf in fit["preds_val"].items():
        rep = evaluate_predictions(pdf, val, config=config, n_boot=n_boot, seed=seed)
        grp = "action_prob" if target == "selection" else "outcome1_prob"
        central[view] = rep["slices"]["all"].get(grp, {})
    out = {"central": central}
    if {"U", "L1", "O"} <= set(fit["preds_val"]):
        out["val_deltas"] = compare_views(fit["preds_val"], val, config=config, target=ht,
                                          n_boot=n_boot, seed=seed)
    if fit["preds_test"] and {"U", "L1", "O"} <= set(fit["preds_test"]):
        test = table.loc[masks["test"]]
        out["test_deltas"] = compare_views(fit["preds_test"], test, config=config, target=ht,
                                           n_boot=n_boot, seed=seed)
    return out


def _baseline_selection_losses(table, masks, config, n_boot, seed) -> dict:
    """Selection log loss of the count-based references (scored through the harness)."""
    train, val = table.loc[masks["train"]], table.loc[masks["val"]]
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


# --- falsification + D47 verdicts (synthetic worlds only) ------------------------------

def _falsification(table, world, truth, sel_fit, out_fit, masks, hp, seed, n_perm, ci_boot):
    """D47 verdicts: null -> GRAMMAR_DETECTED + OUTCOME_QUIET; positive -> MECHANISM_RECOVERED.

    Order ablation reuses the per-view prediction fits (``sel_fit`` / ``out_fit``); only the
    token-order permutation control refits O (via the D17 factory), kept to ``n_perm``.
    """
    fal_params = dict(hp)
    fal_params.update(seed=seed, device=sel_fit["models"]["O"].device.type)

    # order-ablation losses reuse the fitted per-view models' per-row val losses
    sel_losses = {v: float(sel_fit["per_row_val"][v].mean()) for v in sel_fit["per_row_val"]}
    out_losses = {v: float(out_fit["per_row_val"][v].mean()) for v in out_fit["per_row_val"]}

    eval_pos = np.flatnonzero(np.asarray(masks["val"], bool))
    ab_sel = {"per_row": sel_fit["per_row_val"], "eval_pos": eval_pos}
    ab_out = {"per_row": out_fit["per_row_val"], "eval_pos": eval_pos}

    result: dict = {"world": world, "selection_losses": sel_losses, "outcome_losses": out_losses}

    perm = ws6_permutation_test(table, "outcome1", fal_params, n_permutations=n_perm, seed=seed,
                                train_mask=np.asarray(masks["train"], bool),
                                eval_mask=np.asarray(masks["val"], bool), alpha=0.05)
    result["permutation"] = perm

    from workstreams.ws6_deep_seq.model import _delta_order_ci
    out_ci = _delta_order_ci(ab_out, table, seed=seed, n_boot=ci_boot)
    result["outcome_delta_order_ci"] = out_ci
    result["outcome_delta_order"] = float(min(out_losses["U"], out_losses["L1"]) - out_losses["O"])

    if world == "null":
        rc = _repeat_context_mask(table)
        gram = _l1_twin_delta_ci(ab_sel, table, seed=seed, n_boot=ci_boot, row_mask=rc)
        grammar_detected = bool(gram["lo"] > 0)
        outcome_quiet = bool(out_ci["lo"] <= 0 and not perm["fired"])
        result.update({
            "grammar_repeat_delta": gram["point"], "grammar_repeat_ci": gram, "grammar_repeat_n": gram.get("n", 0),
            "grammar_verdict": "GRU_GRAMMAR_DETECTED" if grammar_detected else "GRU_GRAMMAR_MISSED",
            "outcome_verdict": "GRU_OUTCOME_QUIET" if outcome_quiet else "GRU_OUTCOME_NOISY",
            "grammar_detected": grammar_detected, "outcome_quiet": outcome_quiet,
            "d47_pass": bool(grammar_detected and outcome_quiet),
            "d47_verdict": "GRU_GRAMMAR_DETECTED+GRU_OUTCOME_QUIET" if (grammar_detected and outcome_quiet)
            else "GRU_NULL_INCOMPLETE",
        })
    else:  # positive
        detected = bool(out_ci["lo"] > 0 and perm["fired"])
        o_model = out_fit["models"]["O"]
        val = table.loc[masks["val"]]
        proba = o_model.predict_proba_table(val)
        from workstreams.ws6_deep_seq.model import _O1_INDEX
        whiff = proba[:, _O1_INDEX["whiff"]]
        gap = velo_gap_prev_transition(table)[eval_pos]
        trig = gap >= _VELO_GAP_THRESHOLD
        recovered = float("nan")
        if trig.any() and (~trig).any():
            recovered = float(whiff[trig].mean() - whiff[~trig].mean())
        planted = float(truth.get("empirical_whiff_lift")) if truth else None
        ratio = float(recovered / planted) if (planted and np.isfinite(recovered) and planted != 0) else float("nan")
        sign_ok = bool(np.isfinite(recovered) and recovered > 0)
        result.update({
            "order_effect_detected": detected,
            "recovered_whiff_lift": recovered, "planted_whiff_lift": planted, "recovery_ratio": ratio,
            "recovered_sign_ok": sign_ok,
            "d47_pass": bool(detected and sign_ok),
            "d47_verdict": "GRU_MECHANISM_RECOVERED" if (detected and sign_ok) else "GRU_MECHANISM_DIRECTIONAL"
            if sign_ok else "GRU_MECHANISM_MISSED",
        })
    return result


# --- orchestrator ----------------------------------------------------------------------

def run_ws6(
    source: str | None = None,
    synth: str = "off",
    out: str = "results/ws6",
    views=_DEFAULT_VIEWS,
    targets=_TARGETS,
    n_games: int = 200,
    seed: int = 0,
    epochs: int = 30,
    hidden: int = 64,
    embed: int = 16,
    batch: int = 512,
    max_len: int = 15,
    device: str = "auto",
    n_boot: int = 200,
    n_perm: int = 10,
    ci_boot: int = 60,
    effect_size: float = _EFFECT_SIZE,
    force: bool = False,
    write_outputs: bool = True,
    config: dict | None = None,
) -> dict:
    """Fit, score and falsify the WS6 GRU ladder; return the full result dict.

    Parameters mirror the CLI. ``synth`` in ``{'off','null','positive'}`` selects a generated
    world (else ``source`` is the decision table). Returns per-view central losses, both
    deltas (val + locked test), the falsification / D47 verdicts, the motif-probe table, the
    Pareto rows and run metadata.
    """
    import torch

    if config is None:
        config = load_config()
    views = [v for v in views if v in STATE_VIEWS]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    world = synth if synth != "off" else "real"
    world_tag = world

    hp = dict(_DEFAULT_HP)
    hp.update(hidden=hidden, embed_dim=embed, batch_size=batch, max_len=max_len, max_epochs=epochs)

    with track_run(step="ws6_deep_seq", world=world) as meta:
        if synth != "off":
            table, truth = _synth_table(synth, n_games, seed, effect_size)
        else:
            if source is None:
                raise ValueError("one of source (--table) or synth is required")
            table, truth = _load_table(source), {}
        masks = temporal_split_masks(table, config)
        n_train, n_val, n_test = int(masks["train"].sum()), int(masks["val"].sum()), int(masks["test"].sum())

        fits: dict = {}
        for target in targets:
            fits[target] = _fit_views(table, target, views, masks, hp, out_dir, world_tag,
                                      seed, device, force, write_outputs)

        scores = {t: _score(fits[t], table, masks, t, config, n_boot, seed) for t in targets}
        baseline = _baseline_selection_losses(table, masks, config, n_boot, seed) if "selection" in targets else {}

        falsification = None
        motif = None
        if synth != "off" and {"U", "L1", "O"} <= set(views) and "selection" in targets and "outcome1" in targets:
            falsification = _falsification(table, world, truth, fits["selection"], fits["outcome1"],
                                           masks, hp, seed, n_perm, ci_boot)
            val = table.loc[masks["val"]]
            motif = motif_repeat_probe(fits["selection"]["models"]["O"], val)

    result = {
        "world": world, "views": list(views), "targets": list(targets),
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "scores": scores, "baseline_selection_loss": baseline,
        "pareto": {t: fits[t]["pareto"] for t in targets},
        "falsification": falsification, "motif": motif, "truth": truth,
        "checkpoints": {t: {"loaded": fits[t]["loaded"], "fitted": fits[t]["fitted"]} for t in targets},
        "seconds": float(meta["seconds"]), "peak_mem_mb": float(meta["peak_mem_mb"]),
        "hp": {**hp, "device": device},
    }
    if falsification is not None:
        result["d47_verdict"] = falsification.get("d47_verdict")
        result["d47_pass"] = falsification.get("d47_pass")

    if write_outputs:
        rep_path = out_dir / f"ws6_report_{world_tag}.json"
        write_report(result, rep_path)
        write_runmeta(out_dir / f"ws6_{world_tag}.runmeta.json", meta)
        result["outputs"] = {"report": str(rep_path), "artifacts_dir": str(out_dir)}
    return result


# --- headline --------------------------------------------------------------------------

def _fmt(x, spec="+.4f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "  n/a "


def _annotate(delta, ci) -> str:
    if ci is None:
        return ""
    if ci.get("lo", 0.0) > 0:
        return "significantly positive: ordered history helps out-of-sample"
    if delta is not None and delta < 0:
        return "small negative: consistent with no ordering effect (D21; min[U,L1] optimistic)"
    return "not significantly positive: no evidence order helps beyond U/L1 (D21)"


def _delta_line(label, deltas_block) -> str:
    if not deltas_block:
        return f"   {label:<10}: n/a"
    pt = deltas_block.get("deltas", {}).get("delta_order")
    ci = deltas_block.get("delta_order_ci")
    line = f"   {label:<10}: {_fmt(pt)}  CI[{_fmt(ci['lo']) if ci else ' n/a '}, {_fmt(ci['hi']) if ci else ' n/a '}]"
    note = _annotate(pt, ci)
    return line + (f"  {note}" if note else "")


def _matchup_line(label, deltas_block) -> str:
    if not deltas_block or "delta_matchup_ci" not in deltas_block:
        return f"   {label:<10}: n/a"
    pt = deltas_block.get("deltas", {}).get("delta_matchup")
    ci = deltas_block.get("delta_matchup_ci")
    return f"   {label:<10}: {_fmt(pt)}  CI[{_fmt(ci['lo']) if ci else ' n/a '}, {_fmt(ci['hi']) if ci else ' n/a '}]"


def _format_headline(result: dict) -> str:
    width = 92
    L = ["=" * width, " WS6 deep sequence: capacity-matched GRU ladder - headline", "=" * width]
    L.append(f" world          : {result['world']}   device: {result['hp'].get('device','auto')}"
             f"   embed={result['hp']['embed_dim']} hidden={result['hp']['hidden']} layers={result['hp']['num_layers']}")
    L.append(f" rows           : train={result['n_train']:,}  val={result['n_val']:,}  test={result['n_test']:,}")

    sc = result["scores"]
    L.append(" CENTRAL TABLE (validation log loss, lower is better):")
    L.append(f"   {'view':<4} {'sel_ll':>9} {'out1_ll':>9}")
    for v in result["views"]:
        sel = sc.get("selection", {}).get("central", {}).get(v, {}).get("log_loss")
        o1 = sc.get("outcome1", {}).get("central", {}).get(v, {}).get("log_loss")
        L.append(f"   {v:<4} {_fmt(sel, '.4f'):>9} {_fmt(o1, '.4f'):>9}")
    bl = result.get("baseline_selection_loss", {})
    if bl:
        L.append("   selection references: " + "  ".join(f"{k}={_fmt(v, '.4f')}" for k, v in bl.items()))
    L.append("   WS2/WS3 comparison (fill from their reports in Phase 2): "
             "selection O-loss vs WS2 grammar & WS3 GBDT-O; outcome1 O-loss vs WS3 GBDT-O")

    L.append(" Delta_order (min[U,L1]-O), clustered CI:")
    L.append(_delta_line("selection", sc.get("selection", {}).get("val_deltas")))
    L.append(_delta_line("outcome1", sc.get("outcome1", {}).get("val_deltas")))
    L.append(" Delta_matchup (O-OM), clustered CI:")
    L.append(_matchup_line("selection", sc.get("selection", {}).get("val_deltas")))
    L.append(_matchup_line("outcome1", sc.get("outcome1", {}).get("val_deltas")))

    td = sc.get("outcome1", {}).get("test_deltas")
    if td:
        ci = td.get("delta_order_ci")
        L.append(f" LOCKED TEST outcome1 Delta_order: {_fmt(td['deltas']['delta_order'])}  "
                 f"CI[{_fmt(ci['lo']) if ci else 'n/a'}, {_fmt(ci['hi']) if ci else 'n/a'}]")

    L.append(" PARETO ROW (SPEC 7: params / epochs / wall-clock per view; selection target):")
    L.append(f"   {'view':<4} {'params':>9} {'epochs':>7} {'sec':>7} {'peakMB':>8}")
    for v in result["views"]:
        pr = result["pareto"].get("selection", {}).get(v, {})
        L.append(f"   {v:<4} {pr.get('n_params', 0):>9,} {pr.get('epochs', 0):>7} "
                 f"{_fmt(pr.get('seconds'), '.1f'):>7} {_fmt(pr.get('peak_mem_mb'), '.0f'):>8}")

    fal = result.get("falsification")
    if fal:
        L.append(" --- D47 falsification (synthetic) ---")
        if fal["world"] == "null":
            gci = fal.get("grammar_repeat_ci", {})
            L.append(f"   GRAMMAR: {fal['grammar_verdict']}  repeat-context L1-O delta="
                     f"{_fmt(fal.get('grammar_repeat_delta'))} CI[{_fmt(gci.get('lo'))},{_fmt(gci.get('hi'))}]"
                     f"  (n_repeat_rows={fal.get('grammar_repeat_n')})")
            oci = fal.get("outcome_delta_order_ci", {})
            L.append(f"   OUTCOME: {fal['outcome_verdict']}  Delta_order={_fmt(fal.get('outcome_delta_order'))}"
                     f" CI[{_fmt(oci.get('lo'))},{_fmt(oci.get('hi'))}]  perm p={_fmt(fal['permutation']['p_value'], '.3f')}"
                     f" fired={fal['permutation']['fired']}")
        else:
            oci = fal.get("outcome_delta_order_ci", {})
            L.append(f"   outcome1 Delta_order={_fmt(fal.get('outcome_delta_order'))}"
                     f" CI[{_fmt(oci.get('lo'))},{_fmt(oci.get('hi'))}]  perm p={_fmt(fal['permutation']['p_value'], '.3f')}"
                     f" fired={fal['permutation']['fired']}  detected={fal.get('order_effect_detected')}")
            L.append(f"   recovered whiff-lift={_fmt(fal.get('recovered_whiff_lift'))}  "
                     f"planted(empirical)={_fmt(fal.get('planted_whiff_lift'))}  "
                     f"recovery ratio={_fmt(fal.get('recovery_ratio'), '.3f')}")
        L.append(f"   D47 verdict: {fal.get('d47_verdict')}  (pass={fal.get('d47_pass')})")

    motif = result.get("motif")
    if motif:
        L.append(" --- motif rediscovery probe (P(same family 3rd) after [X,X] vs [Y,X]) ---")
        L.append(f"   {'fam':<4} {'P(X|XX)':>9} {'P(X|YX)':>9} {'suppress':>9}")
        for r in motif["per_family"]:
            L.append(f"   {r['family']:<4} {_fmt(r['p_same_after_repeat'], '.4f'):>9} "
                     f"{_fmt(r['p_same_after_mixed'], '.4f'):>9} {_fmt(r['suppression']):>9}")
        L.append(f"   mean suppression={_fmt(motif['mean_suppression'])}  "
                 f"frac_suppressed={_fmt(motif['frac_suppressed'], '.2f')}  motif_present={motif['motif_present']}")

    L.append(f" elapsed (s)    : {_fmt(result.get('seconds'), '.1f')}    peak mem (MB): {_fmt(result.get('peak_mem_mb'), '.1f')}")
    if result.get("outputs"):
        L.append(f" outputs        : {result['outputs']['report']}")
    L.append("=" * width)
    return "\n".join(L)


# --- CLI -------------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python workstreams/ws6_deep_seq/run_ws6.py",
        description="Fit and score the WS6 capacity-matched GRU ladder (SPEC 12.4).",
    )
    p.add_argument("--table", type=str, default=None, help="Decision-table parquet (or season-checkpoint dir).")
    p.add_argument("--out", type=str, default="results/ws6", help="Output directory.")
    p.add_argument("--views", type=str, nargs="*", default=list(_DEFAULT_VIEWS), help="Views to run.")
    p.add_argument("--targets", type=str, nargs="*", default=list(_TARGETS), help="Targets to run.")
    p.add_argument("--synth", type=str, default="off", choices=["off", "null", "positive"],
                   help="Run on a generated synthetic world instead of --table.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device.")
    p.add_argument("--n-games", type=int, default=200, help="Synthetic-world size.")
    p.add_argument("--epochs", type=int, default=30, help="Max training epochs (early stopping on val).")
    p.add_argument("--hidden", type=int, default=64, help="GRU / head hidden width.")
    p.add_argument("--embed", type=int, default=16, help="Token embedding dim.")
    p.add_argument("--batch", type=int, default=512, help="Minibatch size.")
    p.add_argument("--max-len", type=int, default=15, help="Max ordered-history length.")
    p.add_argument("--seed", type=int, default=0, help="Global seed (fits / bootstrap / permutation).")
    p.add_argument("--n-boot", type=int, default=200, help="Cluster-bootstrap replicates for CIs.")
    p.add_argument("--n-perm", type=int, default=10, help="Token-order permutation refits (synthetic).")
    p.add_argument("--ci-boot", type=int, default=60, help="Falsification CI replicates.")
    p.add_argument("--effect-size", type=float, default=_EFFECT_SIZE, help="Positive-world whiff boost.")
    p.add_argument("--force", action="store_true", help="Ignore .done markers and refit.")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point: run the pipeline and print the headline block."""
    args = _parse_args(argv)
    bad = [v for v in args.views if v not in STATE_VIEWS]
    if bad:
        print(f"error: unsupported view(s) {bad}; choose from {list(STATE_VIEWS)}", file=sys.stderr)
        raise SystemExit(2)
    if args.synth == "off" and args.table is None:
        print("error: one of --table or --synth is required", file=sys.stderr)
        raise SystemExit(2)

    result = run_ws6(
        source=args.table, synth=args.synth, out=args.out, views=args.views, targets=args.targets,
        n_games=args.n_games, seed=args.seed, epochs=args.epochs, hidden=args.hidden, embed=args.embed,
        batch=args.batch, max_len=args.max_len, device=args.device, n_boot=args.n_boot,
        n_perm=args.n_perm, ci_boot=args.ci_boot, effect_size=args.effect_size, force=args.force,
    )
    print(_format_headline(result))
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    main()
