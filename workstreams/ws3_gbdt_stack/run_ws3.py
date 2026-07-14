r"""Runnable WS3 step: fit the GBDT behavior + decomposed outcome stack and score it.

``python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet
--out results/ws3/ [--views C U L1 O OM] [--stage behavior|outcome|assemble|eval|all]
[--synth null|positive|off]``

Behaviour (decision D12 conventions -- argparse, ``pathlib``, a ``__main__`` guard, run
metadata; decisions D32-D35). WS3 is the heaviest CPU step so far, so the pipeline is split
into **stages** with **per-view checkpointing**: a completed ``(view, stage)`` writes its
model file(s) and a small ``.done`` marker, and a re-run skips it (``--force`` rebuilds).

Stages
------
``behavior``
    Fit the multiclass behavior model ``mu(a|s)`` per view, save it, and write standard-schema
    ``action_probs`` predictions for the validation + test rows.
``outcome``
    Fit the decomposed outcome stack per view (stage A ``P(o1|s,a)``, stage B ``P(o2|s,a)``,
    stage C count-conditional node values, and the direct ``E[R|s,a]`` regressor cross-check),
    save it, and write ``outcome1``/``outcome2`` probs + assembled ``exp_reward`` /
    ``exp_reward_sd``.
``assemble``
    Publish the counterfactual value grid ``qhat(s,a)`` (all 8 families) and the behavior
    propensities ``mu(a|s)`` per view for the val + test rows (decision D33 artifacts).
``eval``
    Score everything **only** through the shared harness: per-view selection / outcome log
    loss and run-value MAE, and the SPEC ``6`` ``Delta_order`` / ``Delta_matchup`` with
    clustered CIs on (a) the selection target, (b) the outcome1 target and (c) the run-value
    MAE -- the project's central ablation table. On synthetic worlds it also runs the
    falsification battery through the shared :mod:`pitchseq.eval.falsification` with a WS3
    model factory and prints the D35 verdicts (``NULL_QUIET`` / ``MECHANISM_RECOVERED``).
``all``
    Run all four in order.

Import strategy (Windows-safe): the repository root is inserted on ``sys.path`` so the
``from workstreams.ws3_gbdt_stack.model import ...`` package import resolves whether this file
is launched as a script or imported as a module.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pitchseq.config import load_config  # noqa: E402
from pitchseq.eval.baselines import build_baselines  # noqa: E402
from pitchseq.eval.harness import compare_views, evaluate_predictions, write_report  # noqa: E402
from pitchseq.eval.metrics import clustered_ci  # noqa: E402
from pitchseq.eval.predictions import (  # noqa: E402
    ACTION_PROB_COLS,
    OUTCOME1_PROB_COLS,
    OUTCOME2_PROB_COLS,
    POLICY_PROB_COLS,
    save_predictions,
)
from pitchseq.families import FAMILIES  # noqa: E402
from pitchseq.runmeta import track_run, write_runmeta  # noqa: E402
from pitchseq.splits import make_splits  # noqa: E402
from workstreams.ws3_gbdt_stack.model import (  # noqa: E402
    BEHAVIOR_VIEWS,
    BehaviorModel,
    OutcomeStack,
    load_ws3_artifacts,
    make_ws3_model_factory,
    propensities,
    save_behavior,
    save_outcome,
)

__all__ = ["run_ws3", "main"]

_DEFAULT_VIEWS = ("C", "U", "L1", "O", "OM")  # SPEC 6 view ladder (WS3 runs all five, D34)
_STAGES = ("behavior", "outcome", "assemble", "eval", "all")
_REFERENCE_FOR = {"C": "pitcher_count", "U": "pitcher_count_prev", "L1": "pitcher_count_prev",
                  "O": "pitcher_count_prev", "OM": "pitcher_count_prev"}
_VELO_GAP_THRESHOLD = 5.0
_EFFECT_SIZE = 0.30  # positive-world planted whiff boost (matches WS1's contrast fixture)

# --- D35 thresholds (documented; calibrated on the synthetic worlds) --------------------
_RECOVERY_RATIO_FLOOR = 0.30  # WS3 recovers the whiff lift ~directly (O carries the transition);
#                               far above WS1's 0.03 family-proxy attenuation (contrast exhibit).
_MECH_TOP_GROUP = "velo_diff"

# Compact deterministic LightGBM for the falsification battery's repeated permutation refits
# (the same model family as the WS3 models, sized for the many refits).
_FALS_PARAMS = {"n_estimators": 80, "num_leaves": 15, "min_child_samples": 20, "learning_rate": 0.1}


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


def _synth_table(world: str, n_games: int, seed: int, effect_size: float,
                 velo_gap: float) -> tuple[pd.DataFrame, dict]:
    """Generate a synthetic world and build its decision table (Phase-1 CI path)."""
    from pitchseq.decision_table import build_decision_table
    from pitchseq.synth import make_null_world, make_positive_world

    if world == "null":
        raw, truth = make_null_world(n_games=n_games, seed=seed, innings_per_game=6)
    elif world == "positive":
        raw, truth = make_positive_world(n_games=n_games, seed=seed, innings_per_game=6,
                                         effect_size=effect_size, velo_gap_threshold=velo_gap)
    else:
        raise ValueError(f"--synth must be null|positive|off, got {world!r}")
    return build_decision_table(raw), truth


# --- checkpoint markers ----------------------------------------------------------------

def _done_path(out_dir: Path, stage: str, world_tag: str, view: str) -> Path:
    return out_dir / f".{stage}_{world_tag}_{view}.done"


def _is_done(out_dir: Path, stage: str, world_tag: str, view: str, force: bool) -> bool:
    return (not force) and _done_path(out_dir, stage, world_tag, view).is_file()


def _mark_done(out_dir: Path, stage: str, world_tag: str, view: str, payload: dict) -> None:
    _done_path(out_dir, stage, world_tag, view).write_text(
        json.dumps({"stage": stage, "view": view, **payload}, default=str), encoding="utf-8"
    )


# --- prediction assembly ---------------------------------------------------------------

def _base_pred(rows: pd.DataFrame, view: str, seconds: float, peak_mem_mb: float, n_params: int) -> pd.DataFrame:
    df = pd.DataFrame({"row_id": rows["row_id"].to_numpy()})
    df["model_id"] = "ws3_gbdt_stack"
    df["state_view"] = view
    df["seconds"] = float(seconds)
    df["peak_mem_mb"] = float(peak_mem_mb)
    df["n_params"] = int(n_params)
    return df


def _behavior_pred(rows: pd.DataFrame, view: str, proba: np.ndarray, meta: dict, n_params: int) -> pd.DataFrame:
    df = _base_pred(rows, view, meta["seconds"], meta["peak_mem_mb"], n_params)
    for j, col in enumerate(ACTION_PROB_COLS):
        df[col] = proba[:, j]
    return df


def _outcome_pred(rows: pd.DataFrame, view: str, p1: np.ndarray, p2: np.ndarray, er: np.ndarray,
                  sd: np.ndarray, meta: dict, n_params: int) -> pd.DataFrame:
    df = _base_pred(rows, view, meta["seconds"], meta["peak_mem_mb"], n_params)
    for j, col in enumerate(OUTCOME1_PROB_COLS):
        df[col] = p1[:, j]
    for j, col in enumerate(OUTCOME2_PROB_COLS):
        df[col] = p2[:, j]
    df["exp_reward"] = er
    df["exp_reward_sd"] = sd
    return df


# --- stages ----------------------------------------------------------------------------

def _stage_behavior(views, train, eval_rows, out_dir, world_tag, tune, params, seed, force, state):
    """Fit + persist the behavior model per view and write action-prob predictions."""
    built, skipped = [], []
    for view in views:
        if _is_done(out_dir, "behavior", world_tag, view, force):
            skipped.append(view)
            continue
        with track_run(step="ws3_behavior", view=view) as meta:
            model = BehaviorModel(view).fit(train, tune=tune, params=params, seed=seed)
            proba = model.predict_proba(eval_rows)
        save_behavior(out_dir, model)
        df = _behavior_pred(eval_rows, view, proba, meta, model.n_params)
        save_predictions(df, out_dir / f"pred_behavior_{world_tag}_{view}.parquet")
        state.setdefault("behavior", {})[view] = model
        _mark_done(out_dir, "behavior", world_tag, view,
                   {"seconds": meta["seconds"], "peak_mem_mb": meta["peak_mem_mb"], "params": model.params_})
        built.append(view)
    return {"built": built, "skipped": skipped}


def _stage_outcome(views, train, eval_rows, out_dir, world_tag, tune, params, seed, force, state):
    """Fit + persist the outcome stack per view and write outcome / exp-reward predictions."""
    built, skipped = [], []
    disagreement = {}
    for view in views:
        if _is_done(out_dir, "outcome", world_tag, view, force):
            skipped.append(view)
            continue
        with track_run(step="ws3_outcome", view=view) as meta:
            model = OutcomeStack(view).fit(train, tune=tune, params=params, seed=seed)
            p1 = model.predict_outcome1(eval_rows)
            p2 = model.predict_outcome2(eval_rows)
            er, sd = model.exp_reward(eval_rows, with_sd=True)
        dis = model.disagreement(eval_rows)
        disagreement[view] = dis
        save_outcome(out_dir, model)
        df = _outcome_pred(eval_rows, view, p1, p2, er, sd, meta, model.n_params)
        save_predictions(df, out_dir / f"pred_outcome_{world_tag}_{view}.parquet")
        state.setdefault("outcome", {})[view] = model
        _mark_done(out_dir, "outcome", world_tag, view,
                   {"seconds": meta["seconds"], "peak_mem_mb": meta["peak_mem_mb"],
                    "params": model.params_, "disagreement": dis})
        built.append(view)
    return {"built": built, "skipped": skipped, "disagreement": disagreement}


def _stage_assemble(views, eval_rows, out_dir, world_tag, force, state):
    """Write the counterfactual q-grid and behavior propensities per view (decision D33)."""
    built, skipped = [], []
    artifacts = None
    for view in views:
        if _is_done(out_dir, "assemble", world_tag, view, force):
            skipped.append(view)
            continue
        behavior = state.get("behavior", {}).get(view)
        outcome = state.get("outcome", {}).get(view)
        if behavior is None or outcome is None:
            if artifacts is None:
                artifacts = load_ws3_artifacts(out_dir)
            behavior = behavior or artifacts.behavior[view]
            outcome = outcome or artifacts.outcome[view]
        with track_run(step="ws3_assemble", view=view) as meta:
            qg = outcome.q_grid(eval_rows)
            mu = propensities(behavior, eval_rows)
        _save_grid(eval_rows, qg, out_dir / f"qgrid_{world_tag}_{view}.parquet", view, "qhat")
        _save_grid(eval_rows, mu, out_dir / f"propensity_{world_tag}_{view}.parquet", view, POLICY_PROB_COLS)
        _mark_done(out_dir, "assemble", world_tag, view,
                   {"seconds": meta["seconds"], "peak_mem_mb": meta["peak_mem_mb"]})
        built.append(view)
    return {"built": built, "skipped": skipped}


def _save_grid(rows: pd.DataFrame, grid: np.ndarray, path: Path, view: str, cols) -> None:
    """Persist an ``(n, 8)`` grid keyed by ``row_id`` (q-grid or propensity artifact)."""
    df = pd.DataFrame({"row_id": rows["row_id"].to_numpy()})
    if cols == "qhat":
        names = [f"qhat_{f}" for f in FAMILIES]
    else:
        names = list(cols)
    for j, col in enumerate(names):
        df[col] = grid[:, j]
    df["state_view"] = view
    df.to_parquet(path, engine="pyarrow", index=False)


# --- run-value ablation (Delta on MAE, mirroring compare_views) ------------------------

def _runvalue_compare(outcome_preds: dict, table: pd.DataFrame, config: dict, n_boot: int, seed: int) -> dict:
    r"""``Delta_order`` / ``Delta_matchup`` on run-value **MAE** (mirrors :func:`compare_views`).

    Aligns the per-row absolute reward error ``|R - E[R|s,a]|`` across views on the common
    ``row_id`` set, then forms ``Delta_order = min(MAE_U, MAE_L1) - MAE_O`` and
    ``Delta_matchup = MAE_O - MAE_OM`` (positive = the ordered / matchup view has lower error),
    with clustered CIs. Lower MAE is better, so the sign convention matches the log-loss deltas.
    """
    merged = {}
    for v, pdf in outcome_preds.items():
        m = pdf[["row_id", "exp_reward"]].merge(
            table[["row_id", "R", "pitcher", "game_pk"]], on="row_id", how="inner"
        )
        m = m.sort_values("row_id", kind="stable").reset_index(drop=True)
        ok = np.isfinite(pd.to_numeric(m["R"], errors="coerce").to_numpy()) & np.isfinite(m["exp_reward"].to_numpy())
        merged[v] = m.loc[ok].reset_index(drop=True)

    common = None
    for m in merged.values():
        ids = set(m["row_id"].tolist())
        common = ids if common is None else (common & ids)
    common = sorted(common) if common else []
    cset = set(common)

    abs_err = {}
    ref_keys = None
    for v, m in merged.items():
        keep = m["row_id"].isin(cset).to_numpy()
        sub = m.loc[keep]
        abs_err[v] = np.abs(sub["R"].to_numpy(dtype=float) - sub["exp_reward"].to_numpy(dtype=float))
        if ref_keys is None:
            ref_keys = sub[["pitcher", "game_pk"]].reset_index(drop=True)

    mae = {v: float(e.mean()) for v, e in abs_err.items()}
    out = {"mae": mae, "n_common": len(common)}
    if {"U", "L1", "O"} <= set(abs_err):
        out["delta_order"] = float(min(mae["U"], mae["L1"]) - mae["O"])

        def order_metric(idx):
            return float(min(abs_err["U"][idx].mean(), abs_err["L1"][idx].mean()) - abs_err["O"][idx].mean())

        out["delta_order_ci"] = clustered_ci(order_metric, ref_keys, n_boot=n_boot, seed=seed)
    if {"O", "OM"} <= set(abs_err):
        out["delta_matchup"] = float(mae["O"] - mae["OM"])

        def matchup_metric(idx):
            return float(abs_err["O"][idx].mean() - abs_err["OM"][idx].mean())

        out["delta_matchup_ci"] = clustered_ci(matchup_metric, ref_keys, n_boot=n_boot, seed=seed)
    return out


# --- scoring one split (val or test) ---------------------------------------------------

def _load_preds(out_dir: Path, kind: str, world_tag: str, views) -> dict:
    """Load per-view prediction parquets written by the behavior / outcome stages."""
    preds = {}
    for v in views:
        p = out_dir / f"pred_{kind}_{world_tag}_{v}.parquet"
        if p.is_file():
            preds[v] = pd.read_parquet(p, engine="pyarrow")
    return preds


def _score_split(behavior_preds: dict, outcome_preds: dict, split_table: pd.DataFrame,
                 config: dict, n_boot: int, seed: int, views) -> dict:
    """Per-view losses + the three Delta blocks for one split (val or test)."""
    central = {"selection": {}, "outcome1": {}, "outcome2": {}, "run_value": {}}
    for v in views:
        if v in behavior_preds:
            rep = evaluate_predictions(behavior_preds[v], split_table, config=config, n_boot=n_boot, seed=seed)
            central["selection"][v] = rep["slices"]["all"].get("action_prob", {})
        if v in outcome_preds:
            rep = evaluate_predictions(outcome_preds[v], split_table, config=config, n_boot=n_boot, seed=seed)
            allsl = rep["slices"]["all"]
            central["outcome1"][v] = allsl.get("outcome1_prob", {})
            central["outcome2"][v] = allsl.get("outcome2_prob", {})
            central["run_value"][v] = allsl.get("exp_reward", {})

    deltas = {}
    if {"U", "L1", "O"} <= set(behavior_preds):
        deltas["selection"] = compare_views(behavior_preds, split_table, config=config,
                                            target="family", n_boot=n_boot, seed=seed)
    if {"U", "L1", "O"} <= set(outcome_preds):
        deltas["outcome1"] = compare_views(outcome_preds, split_table, config=config,
                                           target="outcome1", n_boot=n_boot, seed=seed)
        deltas["run_value"] = _runvalue_compare(outcome_preds, split_table, config, n_boot, seed)
    return {"central": central, "deltas": deltas}


def _baseline_selection_losses(train, val, config, n_boot, seed) -> dict:
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


# --- falsification battery + D35 verdicts (synthetic worlds only) -----------------------

def _falsification(table: pd.DataFrame, world: str, truth: dict, seed: int, n_perm: int,
                   ci_boot: int, fals_params: dict) -> dict:
    """Run the shared falsification battery with a WS3 factory and derive the D35 verdict."""
    from pitchseq.eval.falsification import (
        mechanism_ablation,
        pseudo_history_control,
        run_null_world_acceptance,
        run_positive_world_acceptance,
    )

    factory = make_ws3_model_factory(fals_params)
    mech = mechanism_ablation(table, factory, target="outcome1")
    mech_ranked = sorted(mech["deltas"].items(), key=lambda kv: kv[1], reverse=True)
    mech_top = mech_ranked[0][0] if mech_ranked else None
    pseudo = pseudo_history_control(table, factory, target="outcome1", seed=seed)

    out = {
        "world": world,
        "mechanism_ablation": {"deltas": mech["deltas"], "ranking": [g for g, _ in mech_ranked], "top_group": mech_top},
        "pseudo_history": pseudo,
    }
    if world == "null":
        acc = run_null_world_acceptance(table, factory, n_permutations=n_perm, seed=seed,
                                        target="outcome1", ci_boot=ci_boot)
        quiet = not acc["order_effect_detected"]
        out.update({
            "acceptance": acc,
            "d35_verdict": "NULL_QUIET" if quiet else "NULL_NOISY",
            "d35_pass": bool(quiet),
        })
    else:  # positive
        acc = run_positive_world_acceptance(table, factory, n_permutations=n_perm, seed=seed,
                                            target="outcome1", velo_gap_threshold=_VELO_GAP_THRESHOLD,
                                            ci_boot=ci_boot)
        planted = float(truth.get("empirical_whiff_lift", float("nan")))
        recovered = float(acc.get("recovered_whiff_lift", float("nan")))
        ratio = recovered / planted if planted not in (0.0, None) and np.isfinite(planted) else float("nan")
        recovered_ok = bool(np.isfinite(ratio) and ratio > _RECOVERY_RATIO_FLOOR)
        mech_ok = bool(mech_top == _MECH_TOP_GROUP)
        detected = bool(acc["order_effect_detected"])
        out.update({
            "acceptance": acc,
            "planted_whiff_lift": planted,
            "recovered_whiff_lift": recovered,
            "recovery_ratio": ratio,
            "recovery_ratio_floor": _RECOVERY_RATIO_FLOOR,
            "ws1_attenuation_reference": 0.03,
            "mechanism_top_is_velo": mech_ok,
            "d35_verdict": "MECHANISM_RECOVERED" if (detected and mech_ok and recovered_ok) else "MECHANISM_MISSED",
            "d35_pass": bool(detected and mech_ok and recovered_ok),
        })
    return out


# --- the pipeline ----------------------------------------------------------------------

def run_ws3(
    source: str | None = None,
    synth: str = "off",
    out: str = "results/ws3",
    views=_DEFAULT_VIEWS,
    stage: str = "all",
    tune: bool = False,
    params: dict | None = None,
    n_games: int = 200,
    seed: int = 7,
    n_boot: int = 200,
    threads: int = 1,
    n_perm: int = 24,
    ci_boot: int = 60,
    effect_size: float = _EFFECT_SIZE,
    velo_gap: float = _VELO_GAP_THRESHOLD,
    fals_params: dict | None = None,
    config: dict | None = None,
    write_outputs: bool = True,
    force: bool = False,
) -> dict:
    """Fit and score the WS3 stack end to end; return a report dict (no printing).

    Parameters
    ----------
    source : str, optional
        Decision-table parquet (or directory of season checkpoints). Required when
        ``synth == "off"``.
    synth : {'off', 'null', 'positive'}, optional
        Run on a generated synthetic world (Phase-1 CI path) instead of ``source``.
    out : str, optional
        Output directory for models / predictions / artifacts / report / run metadata.
    views : sequence of str, optional
        Views to run (default all five; WS3 is the first workstream that runs every view).
    stage : {'behavior', 'outcome', 'assemble', 'eval', 'all'}, optional
        Which stage(s) to run (per-view checkpointed).
    tune : bool, optional
        Grid-select LightGBM params by internal-holdout validation log loss (decision D34).
    params : dict, optional
        Explicit LightGBM params (overrides defaults; e.g. small trees for tests).
    n_games, seed : int, optional
        Synthetic-world size / seed (ignored for a real table); ``seed`` also drives the CIs
        and the falsification battery.
    n_boot : int, optional
        Cluster-bootstrap replicates for the central-table CIs.
    threads : int, optional
        LightGBM ``num_threads`` (1 keeps runs reproducible; raise for speed on real data).
    n_perm, ci_boot : int, optional
        Permutations / bootstrap replicates for the synthetic falsification battery.
    effect_size, velo_gap : float, optional
        Positive-world planted whiff boost and velo-transition threshold.
    fals_params : dict, optional
        LightGBM params for the falsification factory (compact / fast by default).
    config : dict, optional
        Study config; loaded from default when ``None``.
    write_outputs : bool, optional
        When ``True`` (default) persist models / predictions / report under ``out``.
    force : bool, optional
        Ignore ``.done`` markers and rebuild every requested ``(view, stage)``.

    Returns
    -------
    dict
        The full report (central table on validation + locked test, the three Delta blocks,
        decomposed-vs-direct disagreement, the falsification battery + D35 verdict on synthetic
        worlds, per-stage timing / RAM, checkpoint bookkeeping, and run metadata).
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))
    views = [v for v in views]
    if params is None and threads != 1:
        params = {"num_threads": int(threads)}
    elif params is not None and "num_threads" not in params:
        params = {**params, "num_threads": int(threads)}
    fals_params = {**_FALS_PARAMS, "num_threads": int(threads), **(fals_params or {})}
    world_tag = synth if synth != "off" else "real"
    out_dir = Path(out)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    stages = ["behavior", "outcome", "assemble", "eval"] if stage == "all" else [stage]
    state: dict = {}

    with track_run(step="ws3_gbdt_stack", world=world_tag, stage=stage) as meta:
        # --- data + temporal split ---
        truth: dict = {}
        if synth != "off":
            table, truth = _synth_table(synth, n_games, seed, effect_size, velo_gap)
        else:
            if source is None:
                raise ValueError("--table is required when --synth is off")
            table = _load_table(source)
        splits = make_splits(table, config)["primary"]
        train = table.loc[splits["train"].to_numpy()].reset_index(drop=True)
        val = table.loc[splits["val"].to_numpy()].reset_index(drop=True)
        test = table.loc[splits["test"].to_numpy()].reset_index(drop=True)
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"empty split (train={len(train)}, val={len(val)}); check seasons")
        eval_rows = pd.concat([val, test], ignore_index=True) if len(test) else val.copy()

        checkpoints: dict = {}
        stage_timing: dict = {}

        if "behavior" in stages:
            t0 = time.perf_counter()
            checkpoints["behavior"] = _stage_behavior(
                views, train, eval_rows, out_dir, world_tag, tune, params, seed, force, state)
            stage_timing["behavior"] = time.perf_counter() - t0
        if "outcome" in stages:
            t0 = time.perf_counter()
            checkpoints["outcome"] = _stage_outcome(
                views, train, eval_rows, out_dir, world_tag, tune, params, seed, force, state)
            stage_timing["outcome"] = time.perf_counter() - t0
        if "assemble" in stages:
            t0 = time.perf_counter()
            checkpoints["assemble"] = _stage_assemble(views, eval_rows, out_dir, world_tag, force, state)
            stage_timing["assemble"] = time.perf_counter() - t0

        result: dict = {
            "world": world_tag, "stage": stage, "views": views,
            "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
            "checkpoints": checkpoints, "stage_seconds": stage_timing,
            "reference_for": {v: _REFERENCE_FOR.get(v) for v in views},
        }
        # disagreement (from the outcome stage's markers or freshly computed state)
        result["disagreement"] = checkpoints.get("outcome", {}).get("disagreement", {})

        if "eval" in stages:
            t0 = time.perf_counter()
            behavior_preds = _load_preds(out_dir, "behavior", world_tag, views)
            outcome_preds = _load_preds(out_dir, "outcome", world_tag, views)
            # Backfill the decomposed-vs-direct disagreement from the outcome sidecars for any
            # view whose outcome stage ran in an earlier process (a standalone --stage eval).
            result["disagreement"] = _collect_disagreement(out_dir, views, result.get("disagreement", {}))
            result["baseline_selection_loss"] = _baseline_selection_losses(train, val, config, n_boot, seed)
            result["val"] = _score_split(behavior_preds, outcome_preds, val, config, n_boot, seed, views)
            if len(test):
                result["test"] = _score_split(behavior_preds, outcome_preds, test, config, n_boot, seed, views)
            # feature-importance exhibit (gain) per view, from the fitted / loaded outcome stacks
            result["feature_gain"] = _collect_feature_gain(out_dir, views, state)
            if synth != "off":
                result["falsification"] = _falsification(
                    table, synth, truth, seed, n_perm, ci_boot, fals_params)
                result["d35_verdict"] = result["falsification"]["d35_verdict"]
            result["truth"] = truth
            stage_timing["eval"] = time.perf_counter() - t0
            if write_outputs:
                _mark_done(out_dir, "eval", world_tag, "all", {"seconds": stage_timing["eval"]})

        meta["n_train"] = int(len(train))
        meta["n_val"] = int(len(val))
        meta["views"] = views

    result["seconds"] = float(meta.get("seconds", float("nan")))
    result["peak_mem_mb"] = float(meta.get("peak_mem_mb", float("nan")))

    if write_outputs:
        write_report(result, out_dir / f"ws3_report_{world_tag}.json")
        write_runmeta(out_dir / f"ws3_{world_tag}.runmeta.json", dict(meta))
        result["outputs"] = {
            "report": str(out_dir / f"ws3_report_{world_tag}.json"),
            "runmeta": str(out_dir / f"ws3_{world_tag}.runmeta.json"),
            "artifacts_dir": str(out_dir),
        }
    return result


def _collect_disagreement(out_dir: Path, views, current: dict) -> dict:
    """Merge the run's disagreement with what the outcome-stage ``.done`` markers recorded.

    Lets a standalone ``--stage eval`` (whose outcome stage ran in an earlier process) still
    report the decomposed-vs-direct cross-check per view.
    """
    out = dict(current)
    for v in views:
        if v in out:
            continue
        for cand in out_dir.glob(f".outcome_*_{v}.done"):
            try:
                payload = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if "disagreement" in payload:
                out[v] = payload["disagreement"]
                break
    return out


def _collect_feature_gain(out_dir: Path, views, state) -> dict:
    """Top gain-importance features per view from the outcome-stage JSON sidecars."""
    gains = {}
    for v in views:
        model = state.get("outcome", {}).get(v)
        if model is not None:
            gains[v] = dict(list(model.feature_gain().items())[:12])
            continue
        meta_path = out_dir / f"outcome_{v}.json"
        if meta_path.is_file():
            gains[v] = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_gain_top", {})
    return gains


# --- headline --------------------------------------------------------------------------

def _fmt(x, spec="+.4f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "  n/a "


def _annotate_delta(delta, ci, kind="loss") -> str:
    """One-line D21 reading of a Delta_order estimate (loss or MAE; lower is better)."""
    if ci is None:
        return ""
    if ci.get("lo", 0.0) > 0:
        return "significantly positive: ordered history helps out-of-sample"
    if delta is not None and delta < 0:
        return "small negative: consistent with no ordering effect (D21; min[U,L1] optimistic)"
    return "not significantly positive: no evidence order helps beyond U/L1 (D21)"


def _delta_line(label, block, key="delta_order", ci_key="delta_order_ci", annotate=True) -> str:
    if not block:
        return f"   {label:<10}: n/a"
    pt = block.get("deltas", {}).get(key) if "deltas" in block else block.get(key)
    ci = block.get(ci_key)
    line = f"   {label:<10}: {_fmt(pt)}  CI[{_fmt(ci['lo']) if ci else ' n/a '}, {_fmt(ci['hi']) if ci else ' n/a '}]"
    if annotate:
        note = _annotate_delta(pt, ci)
        if note:
            line += f"  {note}"
    return line


def _format_headline(result: dict) -> str:
    width = 78
    L = ["=" * width, " WS3 GBDT behavior + decomposed outcome stack - headline", "=" * width]
    L.append(f" world / stage  : {result['world']} / {result['stage']}")
    L.append(f" rows           : train={result['n_train']:,}  val={result['n_val']:,}  test={result['n_test']:,}")

    val = result.get("val")
    if val:
        cen = val["central"]
        L.append(" CENTRAL TABLE (validation; log loss / MAE, lower is better):")
        L.append(f"   {'view':<4} {'sel_ll':>8} {'out1_ll':>8} {'out2_ll':>8} {'rv_mae':>8}")
        for v in result["views"]:
            sel = cen["selection"].get(v, {}).get("log_loss")
            o1 = cen["outcome1"].get(v, {}).get("log_loss")
            o2 = cen["outcome2"].get(v, {}).get("log_loss")
            mae = cen["run_value"].get(v, {}).get("mae")
            L.append(f"   {v:<4} {_fmt(sel,'.4f'):>8} {_fmt(o1,'.4f'):>8} {_fmt(o2,'.4f'):>8} {_fmt(mae,'.4f'):>8}")
        bl = result.get("baseline_selection_loss", {})
        if bl:
            L.append("   selection references: " + "  ".join(f"{k}={_fmt(v2,'.4f')}" for k, v2 in bl.items()))

        d = val["deltas"]
        L.append(" Delta_order (min[U,L1]-O), clustered CI:")
        L.append(_delta_line("selection", d.get("selection")))
        L.append(_delta_line("outcome1", d.get("outcome1")))
        L.append(_delta_line("run-value", d.get("run_value"), annotate=False) + "  (MAE; O lower error when >0)")
        L.append(" Delta_matchup (O-OM), clustered CI:")
        L.append(_delta_line("selection", d.get("selection"), key="delta_matchup", ci_key="delta_matchup_ci", annotate=False))
        L.append(_delta_line("outcome1", d.get("outcome1"), key="delta_matchup", ci_key="delta_matchup_ci", annotate=False))
        L.append(_delta_line("run-value", d.get("run_value"), key="delta_matchup", ci_key="delta_matchup_ci", annotate=False))

    test = result.get("test")
    if test and test["deltas"].get("outcome1"):
        to = test["deltas"]["outcome1"]
        ci = to.get("delta_order_ci")
        L.append(f" LOCKED TEST outcome1 Delta_order: {_fmt(to['deltas']['delta_order'])}  "
                 f"CI[{_fmt(ci['lo']) if ci else 'n/a'}, {_fmt(ci['hi']) if ci else 'n/a'}]")

    dis = result.get("disagreement", {})
    if dis:
        parts = "  ".join(f"{v}={_fmt(dis[v]['mean_abs_diff'],'.4f')}" for v in result["views"] if v in dis)
        flagged = [v for v in dis if dis[v].get("flag")]
        L.append(f" decomposed vs direct (D32) mean|dec-direct|: {parts}"
                 + (f"  FLAG:{flagged}" if flagged else "  (all within tolerance)"))

    fg = result.get("feature_gain", {}).get("C")
    if fg:
        top = list(fg.items())[:4]
        L.append(" top gain features (C view): " + ", ".join(f"{k}={_fmt(v,'.0f')}" for k, v in top))

    fal = result.get("falsification")
    if fal:
        L.append(" --- D35 falsification (synthetic) ---")
        acc = fal.get("acceptance", {})
        losses = acc.get("losses", {})
        if losses:
            L.append("   order_ablation (outcome1) losses: " + "  ".join(f"{k}={_fmt(v,'.4f')}" for k, v in losses.items()))
        ci = acc.get("delta_order_ci", {})
        L.append(f"   delta_order={_fmt(acc.get('delta_order'))}  CI[{_fmt(ci.get('lo'))}, {_fmt(ci.get('hi'))}]  "
                 f"perm p={_fmt(acc.get('permutation_p'),'.3f')}  fired={acc.get('permutation_fired')}")
        mech = fal.get("mechanism_ablation", {})
        L.append("   mechanism ablation (loss increase): "
                 + "  ".join(f"{g}={_fmt(mech['deltas'][g],'.4f')}" for g in mech.get("ranking", [])))
        L.append(f"   mechanism top group: {mech.get('top_group')}")
        if fal["world"] == "positive":
            L.append(f"   recovered whiff-lift={_fmt(fal['recovered_whiff_lift'],'.4f')}  "
                     f"planted(empirical)={_fmt(fal['planted_whiff_lift'],'.4f')}  "
                     f"recovery ratio={_fmt(fal['recovery_ratio'],'.3f')}")
            L.append(f"   (contrast: WS1 family-proxy attenuation ~0.03; floor {fal['recovery_ratio_floor']})")
        L.append(f" D35 verdict    : {fal['d35_verdict']}")

    L.append(f" elapsed (s)    : {_fmt(result.get('seconds'), '.1f')}"
             + ("  per-stage: " + "  ".join(f"{k}={_fmt(v,'.1f')}" for k, v in result.get("stage_seconds", {}).items())
                if result.get("stage_seconds") else ""))
    L.append(f" peak mem (MB)  : {_fmt(result.get('peak_mem_mb'), '.1f')}")
    if result.get("outputs"):
        L.append(f" outputs        : {result['outputs']['report']}")
    L.append("=" * width)
    return "\n".join(L)


# --- CLI -------------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python workstreams/ws3_gbdt_stack/run_ws3.py",
        description="Fit and score the WS3 GBDT behavior + decomposed outcome stack (SPEC 12.3).",
    )
    parser.add_argument("--table", type=str, default=None,
                        help="Decision-table parquet (or directory of season checkpoints).")
    parser.add_argument("--out", type=str, default="results/ws3",
                        help="Output directory for models / predictions / artifacts / report.")
    parser.add_argument("--views", type=str, nargs="*", default=list(_DEFAULT_VIEWS),
                        help=f"Views to run (default all five {list(_DEFAULT_VIEWS)}).")
    parser.add_argument("--stage", type=str, default="all", choices=list(_STAGES),
                        help="Pipeline stage (per-view checkpointed).")
    parser.add_argument("--synth", type=str, default="off", choices=["off", "null", "positive"],
                        help="Run on a generated synthetic world instead of --table.")
    parser.add_argument("--tune", action="store_true",
                        help="Grid-select LightGBM params by internal-holdout val log loss (D34).")
    parser.add_argument("--force", action="store_true", help="Ignore .done markers and rebuild.")
    parser.add_argument("--threads", type=int, default=1,
                        help="LightGBM num_threads (1 = reproducible; raise for speed on real data).")
    parser.add_argument("--n-games", type=int, default=300, help="Synthetic-world size.")
    parser.add_argument("--seed", type=int, default=7, help="Synthetic-world / bootstrap / falsification seed.")
    parser.add_argument("--n-boot", type=int, default=200, help="Cluster-bootstrap replicates for CIs.")
    parser.add_argument("--n-perm", type=int, default=24, help="Falsification permutations (synthetic).")
    parser.add_argument("--ci-boot", type=int, default=60, help="Falsification acceptance CI replicates.")
    parser.add_argument("--effect-size", type=float, default=_EFFECT_SIZE, help="Positive-world whiff boost.")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point: run the requested stage(s) and print the headline block."""
    args = _parse_args(argv)
    bad = [v for v in args.views if v not in BEHAVIOR_VIEWS]
    if bad:
        print(f"error: unsupported view(s) {bad}; choose from {list(BEHAVIOR_VIEWS)}", file=sys.stderr)
        raise SystemExit(2)
    if args.synth == "off" and args.table is None:
        print("error: one of --table or --synth is required", file=sys.stderr)
        raise SystemExit(2)

    result = run_ws3(
        source=args.table, synth=args.synth, out=args.out, views=args.views, stage=args.stage,
        tune=args.tune, n_games=args.n_games, seed=args.seed, n_boot=args.n_boot,
        threads=args.threads, n_perm=args.n_perm, ci_boot=args.ci_boot,
        effect_size=args.effect_size, force=args.force,
    )
    print(_format_headline(result))
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    main()
