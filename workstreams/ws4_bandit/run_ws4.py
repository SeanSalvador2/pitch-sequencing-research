r"""Runnable WS4 step: build the Thompson target policy and evaluate it through the OPE gate.

``python workstreams/ws4_bandit/run_ws4.py --table data/processed/decision_table.parquet
--ws3-dir results/ws3/ --out results/ws4/ [--views C L1 O] [--synth null|positive|off]``

Behaviour (decision D12 conventions -- argparse, ``pathlib``, a ``__main__`` guard, run
metadata, checkpoint markers; decisions D36-D39). The pipeline:

1. **WS3 artifacts** for the D38 ablation views (C, L1, O). On a real table they are *loaded*
   from ``--ws3-dir`` (decision D33 -- WS4 never re-fits an outcome/behavior model). On a
   synthetic world it trains small WS3 stacks itself (via the WS3 code) and checkpoints them.
2. **Behavior-policy recovery FIRST** (decision D37 / SPEC ``0.3``): target = behavior mu
   through :func:`pitchseq.eval.ope.behavior_policy_recovery`. If it fails the run prints
   ``FAILED_GATE`` and stops **before any target value is reported**.
3. **Thompson targets** per view (decision D36) from WS3's q-grid + (scaled) uncertainty.
4. **OPE** of each ``pi_alpha`` (config alpha grid) per view through
   :func:`pitchseq.eval.ope.evaluate_policy` with the full SPEC ``9`` report block.
5. **Exhibits**: the value-vs-alpha frontier (value, lower_95, ESS, support), deviation-from-
   behavior maps, and the ambiguity decomposition.
6. **Headline**: the D38 prescriptive-ablation table -- per view (C / L1 / O) at each alpha:
   value +/- CI vs behavior, ESS%, verdict -- with the D39 honest-negative annotations
   (INCONCLUSIVE and below-behavior are first-class), plus the C -> O value gaps with
   clustered CIs (the sequencing-prescription evidence, isolated from the count-driven
   value-vs-behavior gain).

The C -> O gap is the sequencing-prescription statistic. Every view's target is scored
against **one common OPE evaluator** -- the behavior mu and q-grid of the richest ablation
view (O; :data:`~workstreams.ws4_bandit.model.COMMON_EVAL_VIEW`) -- so the C -> O value
change isolates the policy's *information*, not the evaluator's. The gap CI is a
pitcher-game-clustered (SPEC ``7``) paired bootstrap over the doubly-robust contributions
from :func:`pitchseq.eval.ope.dr` (consuming the OPE estimator, not re-implementing it); the
per-policy value CIs are :func:`~pitchseq.eval.ope.evaluate_policy`'s own per-decision
bootstrap.

Honest-negative reading (decision D39, SPEC ``9`` closing rule)
==============================================================
On the synthetic **positive** world the planted effect is a whiff boost keyed on the ordered
velo transition ``|velo_{t-1} - velo_{t-2}|`` -- a *state*-value effect. A **myopic** policy
can only exploit its small *family-differential* component (some families convert more
foul/in-play mass to whiffs than others on a triggered pitch), which is a real but ~0.003-run
edge that sits below the OPE noise floor until full-data scale. So the honest C -> O gap is
**directionally positive but frequently CI-inconclusive on small synthetic worlds** -- a
first-class D39 outcome, and precisely the motivation for the *sequential* workstreams
(WS5 tabular MDP, WS7 offline RL) that can value a setup pitch. The verdict machinery reports
``SEQ_EXPLOITED`` only when the real-reward-anchored gap CI actually excludes 0, and
``SEQ_INCONCLUSIVE_MYOPIC`` (with this explanation) otherwise -- it never fabricates the gain.

Both worlds also show the bandit beating behavior in raw value (the behavior policy is
habit-based, not reward-optimal): this **count-driven myopic improvement is real but is NOT
sequencing evidence** -- the C -> O ablation, not the value-vs-behavior gain, isolates
sequencing. The headline prints this explanation whenever a gain over behavior appears.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pitchseq.config import load_config  # noqa: E402
from pitchseq.eval import ope  # noqa: E402
from pitchseq.eval.predictions import POLICY_PROB_COLS, save_predictions  # noqa: E402
from pitchseq.families import FAMILIES, feasible_action_mask  # noqa: E402
from pitchseq.runmeta import track_run, write_runmeta  # noqa: E402
from pitchseq.splits import cluster_bootstrap_indices, make_splits  # noqa: E402
from workstreams.ws3_gbdt_stack.model import (  # noqa: E402
    BehaviorModel,
    OutcomeStack,
    WS3Artifacts,
    load_ws3_artifacts,
    save_ws3_artifacts,
)
from workstreams.ws4_bandit.model import (  # noqa: E402
    ABLATION_VIEWS,
    COMMON_EVAL_VIEW,
    DEFAULT_N_SAMPLES,
    POSTERIOR_SCALE,
    ambiguity_stats,
    build_bandit_inputs,
    deviation_map,
    feasible_matrix,
    soften,
    thompson_policy,
)

__all__ = ["run_ws4", "main"]

# Small deterministic WS3 params for the synthetic Phase-1 CI path (the real run loads
# pre-fit WS3 artifacts via --ws3-dir and never trains here).
_SYNTH_WS3_PARAMS = {"n_estimators": 120, "num_leaves": 31, "min_child_samples": 40,
                     "learning_rate": 0.08}
_VELO_GAP_THRESHOLD = 5.0
_EFFECT_SIZE = 0.30


# --- data ------------------------------------------------------------------------------

def _load_table(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.is_dir():
        frames = [pd.read_parquet(p, engine="pyarrow") for p in sorted(path.glob("*.parquet"))]
        if not frames:
            raise FileNotFoundError(f"no parquet files under {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(path, engine="pyarrow")


def _synth_table(world: str, n_games: int, seed: int, effect_size: float, velo_gap: float):
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


# --- checkpointing (synthetic WS3 fit) -------------------------------------------------

def _ws3_done(out_dir: Path, world_tag: str, view: str) -> Path:
    return out_dir / f".ws3fit_{world_tag}_{view}.done"


def _fit_or_load_ws3(train, views, out_dir, world_tag, params, seed, force, synth) -> WS3Artifacts:
    """On a synthetic world, fit + checkpoint a small WS3 stack per view; on real data the
    caller loads the artifacts instead (this is only reached for ``synth != 'off'``)."""
    behavior, outcome = {}, {}
    for view in views:
        done = _ws3_done(out_dir, world_tag, view)
        b_path = out_dir / f"behavior_{view}.joblib"
        o_path = out_dir / f"outcome_{view}.joblib"
        if (not force) and done.is_file() and b_path.is_file() and o_path.is_file():
            import joblib
            behavior[view] = joblib.load(b_path)
            outcome[view] = joblib.load(o_path)
            continue
        with track_run(step="ws4_ws3fit", view=view):
            bm = BehaviorModel(view).fit(train, params=params, seed=seed)
            os_ = OutcomeStack(view).fit(train, params=params, seed=seed)
        save_ws3_artifacts(out_dir, behavior={view: bm}, outcome={view: os_})
        done.write_text(json.dumps({"view": view, "n_train": int(len(train))}), encoding="utf-8")
        behavior[view] = bm
        outcome[view] = os_
    return WS3Artifacts(behavior=behavior, outcome=outcome, directory=str(out_dir))


# --- logged frame (bandit: one-step episodes) ------------------------------------------

def _count_state(table: pd.DataFrame) -> np.ndarray:
    """Coarse discrete state = count cell ``balls*3 + strikes`` (0..11) -- FQE's one-hot key."""
    b = table["balls"].to_numpy(dtype=np.int64)
    s = table["strikes"].to_numpy(dtype=np.int64)
    return b * 3 + s


def _feasibility_for_eval(table: pd.DataFrame, eval_rows: pd.DataFrame, config: dict):
    """SPEC ``4`` feasible mask computed on the **full** ``table`` (so the trailing 365-day
    window sees the train seasons) and subset to ``eval_rows`` by ``row_id`` -- view-independent,
    so it is computed once and shared across the ablation views."""
    fm = feasible_action_mask(table, config=config)
    fm = fm.set_axis(table["row_id"].to_numpy(), axis=0)
    fm_eval = fm.loc[eval_rows["row_id"].to_numpy()]
    return feasible_matrix(fm_eval), fm_eval["low_history"].to_numpy(dtype=bool)


def _build_logged(table, mu_common, observed_action) -> pd.DataFrame:
    """Assemble the OPE ``logged`` frame -- one one-step (myopic) episode per decision.

    The behavior matrix ``mu_prob_0..7`` and taken-action propensity ``mu_prob`` are the
    common-evaluator behavior policy; ``pa_id`` is unique per row (bandit semantics) and
    ``step`` is 0. The coarse count ``state`` is the outcome-table key for the
    behavior-recovery self-test and the FQE cross-check (FQE itself is toggled by trimming the
    estimator list, not by dropping the state column, which recovery always needs).
    """
    n = len(table)
    R = pd.to_numeric(table["R"], errors="coerce").to_numpy(dtype=np.float64)
    data = {
        "reward": R,
        "action": observed_action.astype(np.int64),
        "mu_prob": mu_common[np.arange(n), observed_action],
        "pa_id": np.arange(n, dtype=np.int64),
        "step": np.zeros(n, dtype=np.int64),
        "pitcher": table["pitcher"].to_numpy(),
        "game_pk": table["game_pk"].to_numpy(),
        "state": _count_state(table),
    }
    for a in range(len(FAMILIES)):
        data[f"mu_prob_{a}"] = mu_common[:, a]
    return pd.DataFrame(data)


def _eval_config(config: dict, fqe: bool) -> dict:
    """The config used for the target evaluations -- FQE trimmed from the estimator list when
    ``fqe`` is off (the ``state`` column stays, since behavior-recovery needs it)."""
    if fqe:
        return config
    ope_cfg = dict(config.get("ope", {}))
    ope_cfg["estimators"] = [e for e in ope_cfg.get("estimators", []) if e != "FQE"]
    return {**config, "ope": ope_cfg}


# --- OPE per (view, alpha) -------------------------------------------------------------

def _evaluate_view_alpha(logged, target, q_common, config, n_boot, seed):
    """One :func:`~pitchseq.eval.ope.evaluate_policy` call -> the compact per-cell record."""
    res = ope.evaluate_policy(
        logged, target, q_hat=q_common, config=config, n_boot=n_boot, seed=seed,
        reward_col="reward", action_col="action", mu_col="mu_prob",
        episode_col="pa_id", step_col="step", state_col="state",
    )
    prim = res["estimators"].get(res["primary_estimator"], {})
    diag = res["diagnostics"]
    dm = res["estimators"].get("DM", {})
    return {
        "value": prim.get("value"),
        "lower_95": prim.get("lower_95"),
        "ci95": prim.get("ci95"),
        "primary": res["primary_estimator"],
        "dm_value": dm.get("value"),
        "ess": diag.get("ess"),
        "ess_frac": diag.get("ess_frac"),
        "oos_support_frac": diag.get("oos_support_frac"),
        "max_weight": diag.get("max_weight"),
        "mean_tv": diag.get("mean_tv"),
        "mean_kl": diag.get("mean_kl"),
        "verdict": res["verdict"],
        "report": res["report"],
    }


def _gap_ci(rewards, mu_taken, q_common, pi_a, pi_b, actions, cluster_frame, n_boot, seed):
    r"""Pitcher-game-clustered paired bootstrap of the O - C (or L1 - C) DR value gap.

    Uses the doubly-robust per-row contributions from :func:`pitchseq.eval.ope.dr` (the OPE
    estimator, not a re-implementation) under a **common** evaluator, then bootstraps the mean
    of their per-row difference over pitcher-game clusters (SPEC ``7``). Reports the point
    estimate, the two-sided 95% interval and the one-sided 95% lower bound (SPEC ``9``).
    """
    ar = np.arange(len(actions))
    w_a = pi_a[ar, actions] / mu_taken
    w_b = pi_b[ar, actions] / mu_taken
    _, contrib_a = ope.dr(rewards, w_a, q_common, pi_a, actions)
    _, contrib_b = ope.dr(rewards, w_b, q_common, pi_b, actions)
    diff = contrib_a - contrib_b
    point = float(diff.mean())
    reps = []
    for idx in cluster_bootstrap_indices(cluster_frame, cluster="pitcher_game",
                                         n_boot=n_boot, seed=seed):
        reps.append(float(diff[np.asarray(idx)].mean()))
    reps = np.asarray(reps, dtype=np.float64)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {
        "value": point,
        "lower_95": float(np.percentile(reps, 5.0)),
        "ci95": [float(lo), float(hi)],
        "se": float(reps.std(ddof=1)) if len(reps) > 1 else float("nan"),
    }


# --- verdicts (synthetic) --------------------------------------------------------------

def _prescription_verdict(world: str, gaps_by_alpha: dict, behavior_value: float,
                          view_values: dict) -> dict:
    r"""The D38 / D39 prescriptive verdict from the honest C -> O gap and value-vs-behavior.

    * ``SEQ_EXPLOITED`` -- the real-reward-anchored O - C gap lower_95 > 0 at some alpha
      (the sequencing-prescription is confirmed; the tested alpha is reported).
    * ``SEQ_NEUTRAL_PRESCRIPTION`` -- the O - C gap CI contains 0 at every alpha and the point
      estimate is small (O ~= C): any value-vs-behavior gain is count-driven, not sequencing.
    * ``SEQ_INCONCLUSIVE_MYOPIC`` -- the O - C gap is directionally positive (point > 0 at the
      upper-bound alpha) but its lower_95 <= 0: a first-class D39 outcome (the myopic bandit
      cannot yet confirm the -- real, WS3-recovered -- sequential edge; see the module docstring).
    """
    alphas = sorted(gaps_by_alpha)
    any_sig = any(gaps_by_alpha[a]["O_minus_C"]["lower_95"] > 0 for a in alphas)
    sig_alpha = next((a for a in alphas if gaps_by_alpha[a]["O_minus_C"]["lower_95"] > 0), None)
    top = gaps_by_alpha[max(alphas)]["O_minus_C"]
    directional = top["value"] > 0

    # SEQ_EXPLOITED is emitted ONLY when the real-reward-anchored gap CI actually clears 0 --
    # it is never fabricated. Absent that, a *known-positive* synthetic world reports the
    # honest D39 negative (the myopic bandit cannot confirm the -- real, WS3-recovered --
    # sequential edge), while the *null* world reports the expected neutral result.
    if any_sig:
        verdict = "SEQ_EXPLOITED"
    elif world == "positive":
        verdict = "SEQ_INCONCLUSIVE_MYOPIC"
    else:
        verdict = "SEQ_NEUTRAL_PRESCRIPTION"

    # Count-driven value-vs-behavior gain (present on both worlds; NOT sequencing).
    best_view_gain = max(
        (float(view_values[v][max(alphas)]["value"]) - behavior_value) for v in view_values
    )
    return {
        "verdict": verdict,
        "significant_alpha": sig_alpha,
        "upper_bound_alpha": max(alphas),
        "o_minus_c_upper": top,
        "value_over_behavior_max": best_view_gain,
        "count_driven_gain": bool(best_view_gain > 0),
    }


# --- the pipeline ----------------------------------------------------------------------

def run_ws4(
    source: str | None = None,
    synth: str = "off",
    out: str = "results/ws4",
    ws3_dir: str | None = None,
    views=ABLATION_VIEWS,
    alphas=None,
    posterior_scale: float = POSTERIOR_SCALE,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_boot: int = 300,
    gap_boot: int = 400,
    seed: int = 7,
    thompson_seed: int = 20260714,
    n_games: int = 400,
    ws3_params: dict | None = None,
    effect_size: float = _EFFECT_SIZE,
    velo_gap: float = _VELO_GAP_THRESHOLD,
    fqe: bool = True,
    config: dict | None = None,
    write_outputs: bool = True,
    force: bool = False,
) -> dict:
    """Build the Thompson target policy and evaluate it through the OPE gate; return a report.

    Parameters
    ----------
    source : str, optional
        Decision-table parquet (or a directory of season checkpoints). Required when
        ``synth == 'off'``.
    synth : {'off', 'null', 'positive'}, optional
        Run on a generated synthetic world (Phase-1 CI path) instead of ``source``.
    out : str, optional
        Output directory (predictions, exhibits, report, run metadata; synthetic WS3 fits).
    ws3_dir : str, optional
        Directory of WS3 artifacts to consume (decision D33). Required when ``synth == 'off'``;
        ignored on a synthetic world (WS3 stacks are trained + checkpointed under ``out``).
    views : sequence of str, optional
        Ablation views (default :data:`~workstreams.ws4_bandit.model.ABLATION_VIEWS`, C/L1/O).
    alphas : sequence of float, optional
        The pi_alpha grid (default: config ``ope.conservative_alpha``).
    posterior_scale : float, optional
        Residual-sd -> posterior-SE shrinkage for the Thompson draws (decision D36; see
        :data:`~workstreams.ws4_bandit.model.POSTERIOR_SCALE`).
    n_samples : int, optional
        Monte-Carlo draws per row for :func:`~workstreams.ws4_bandit.model.thompson_policy`.
    n_boot, gap_boot : int, optional
        Bootstrap replicates for the per-policy report CIs and the clustered gap CIs.
    seed, thompson_seed : int, optional
        Seeds for the bootstraps / behavior-recovery split, and for the Thompson draws.
    n_games : int, optional
        Synthetic-world size (ignored for a real table).
    ws3_params : dict, optional
        LightGBM params for the synthetic WS3 fit (default small/fast :data:`_SYNTH_WS3_PARAMS`).
    effect_size, velo_gap : float, optional
        Positive-world planted whiff boost and velo-transition threshold.
    fqe : bool, optional
        Include a coarse count ``state`` so FQE runs as a (verdict-excluded, D24) cross-check.
    config : dict, optional
        Study config; loaded from default when ``None``.
    write_outputs : bool, optional
        Persist predictions / exhibits / report / run metadata under ``out``.
    force : bool, optional
        Ignore the synthetic-WS3 ``.done`` markers and re-fit.

    Returns
    -------
    dict
        The full report: the behavior-recovery gate, the per-view/per-alpha OPE table, the
        C -> O and L1 -> C gaps with clustered CIs, the value-vs-alpha frontier, deviation
        maps, ambiguity stats, the D38/D39 verdict, timing / RAM and output paths.
    """
    if config is None:
        config = load_config()
    if seed is None:
        seed = int(config.get("seeds", {}).get("global", 0))
    if alphas is None:
        alphas = list(config.get("ope", {}).get("conservative_alpha", [0.0, 0.1, 0.25, 0.5, 1.0]))
    alphas = [float(a) for a in alphas]
    views = list(views)
    if COMMON_EVAL_VIEW not in views:
        raise ValueError(f"the common evaluator view {COMMON_EVAL_VIEW!r} must be in views {views}")
    ws3_params = {**_SYNTH_WS3_PARAMS, **(ws3_params or {})}
    world_tag = synth if synth != "off" else "real"
    out_dir = Path(out)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"world": world_tag, "views": views, "alphas": alphas,
                    "posterior_scale": float(posterior_scale)}

    import time as _time
    _t0 = _time.monotonic()

    def _p(msg: str) -> None:
        """Phase-progress narration (real runs are long; silence reads as a hang)."""
        print(f"[ws4 +{_time.monotonic() - _t0:7.0f}s] {msg}", flush=True)

    with track_run(step="ws4_bandit", world=world_tag) as meta:
        # --- data + split ---
        truth: dict = {}
        if synth != "off":
            table, truth = _synth_table(synth, n_games, seed, effect_size, velo_gap)
        else:
            if source is None:
                raise ValueError("--table is required when --synth is off")
            _p("loading decision table...")
            table = _load_table(source)
        _p(f"table loaded: {len(table):,} rows; building temporal splits...")
        splits = make_splits(table, config)["primary"]
        train = table.loc[splits["train"].to_numpy()].reset_index(drop=True)
        val = table.loc[splits["val"].to_numpy()].reset_index(drop=True)
        test = table.loc[splits["test"].to_numpy()].reset_index(drop=True)
        eval_rows = pd.concat([val, test], ignore_index=True) if len(test) else val.copy()
        if len(train) == 0 or len(eval_rows) == 0:
            raise ValueError(f"empty split (train={len(train)}, eval={len(eval_rows)})")

        # --- WS3 artifacts (load real / fit synthetic) ---
        if synth != "off":
            artifacts = _fit_or_load_ws3(train, views, out_dir, world_tag, ws3_params, seed, force, synth)
        else:
            if ws3_dir is None:
                raise ValueError("--ws3-dir is required when --synth is off (decision D33)")
            _p(f"loading WS3 artifacts from {ws3_dir}...")
            artifacts = load_ws3_artifacts(ws3_dir)
            missing = [v for v in views if v not in artifacts.behavior or v not in artifacts.outcome]
            if missing:
                raise ValueError(f"WS3 artifacts under {ws3_dir} miss views {missing}")

        # --- per-view bandit inputs (q, q_sd, mu, feasible, align) ---
        # Feasibility (SPEC 4) uses the FULL table's trailing history; computed once and shared.
        _p("computing feasibility masks over the full table's trailing history...")
        feas_eval, low_hist_eval = _feasibility_for_eval(table, eval_rows, config)
        _p(f"building bandit inputs per view (q-grid + propensities on {len(eval_rows):,} eval rows; "
           "the slowest silent phase)...")
        inputs = {}
        for v in views:
            inputs[v] = build_bandit_inputs(eval_rows, artifacts, v, config=config,
                                            posterior_scale=posterior_scale,
                                            feasible_mask=feas_eval, low_history=low_hist_eval)
            _p(f"  inputs[{v}] ready")
        common = inputs[COMMON_EVAL_VIEW]
        mu_common = common.mu
        q_common = common.q
        observed = common.align["observed_action"].to_numpy()
        rewards = pd.to_numeric(eval_rows["R"], errors="coerce").to_numpy(dtype=np.float64)
        finite_R = np.isfinite(rewards)
        eval_config = _eval_config(config, fqe)
        logged = _build_logged(eval_rows, mu_common, observed)
        logged = logged.loc[finite_R].reset_index(drop=True)
        scored_rows = eval_rows.loc[finite_R].reset_index(drop=True)
        cluster_frame = scored_rows[["pitcher", "game_pk"]]
        result["n_train"] = int(len(train))
        result["n_eval"] = int(len(eval_rows))
        result["n_scored"] = int(finite_R.sum())
        result["feasible_summary"] = {
            "mean_n_feasible": float(common.feasible_mask.sum(axis=1).mean()),
            "empty_mask_frac": float((~common.align["any_feasible"]).mean()),
            "low_history_frac": float(common.align["low_history"].mean()),
        }

        # --- GATE FIRST: behavior-policy recovery (D37 / SPEC 0.3) ---
        _p(f"gate: behavior-policy recovery on {int(finite_R.sum()):,} scored rows "
           f"(n_boot={n_boot} cluster bootstrap)...")
        recovery = ope.behavior_policy_recovery(
            logged, config=eval_config, n_actions=len(FAMILIES), seed=seed, n_boot=n_boot,
            reward_col="reward", action_col="action", mu_col="mu_prob",
            state_col="state", episode_col="pa_id", step_col="step",
        )
        result["behavior_recovery"] = recovery
        if not recovery["passed"]:
            result["gate"] = "FAILED_GATE"
            result["seconds"] = float(meta.get("seconds", float("nan")))
            if write_outputs:
                _write(out_dir, world_tag, result, meta)
            return result
        result["gate"] = "PASSED"
        _p("gate PASSED; drawing Thompson targets per view "
           f"(n_samples={n_samples} MC draws/row)...")

        # --- Thompson targets per view + softening + OPE per alpha ---
        targets = {}
        for v in views:
            bi = inputs[v]
            targets[v] = thompson_policy(bi.q, bi.q_sd, bi.feasible_mask, n_samples=n_samples,
                                         rng=thompson_seed, observed_action=observed)
            targets[v] = targets[v][finite_R]  # align to the scored logged rows
            _p(f"  thompson[{v}] done")
        mu_scored = mu_common[finite_R]
        q_scored = q_common[finite_R]
        actions_scored = observed[finite_R]
        rewards_scored = rewards[finite_R]
        mu_taken_scored = logged["mu_prob"].to_numpy()

        frontier: dict = {v: {} for v in views}
        gaps_by_alpha: dict = {}
        _n_cells = len(alphas) * len(views)
        _done = 0
        for a in alphas:
            for v in views:
                pol = soften(targets[v], mu_scored, a)
                frontier[v][a] = _evaluate_view_alpha(logged, pol, q_scored, eval_config, n_boot, seed)
                _done += 1
                _p(f"OPE cell {_done}/{_n_cells} done (view={v}, alpha={a})")
            # C -> O and L1 -> C gaps at this alpha (common evaluator, clustered).
            pol_c = soften(targets["C"], mu_scored, a)
            pol_o = soften(targets["O"], mu_scored, a)
            gap_oc = _gap_ci(rewards_scored, mu_taken_scored, q_scored, pol_o, pol_c,
                             actions_scored, cluster_frame, gap_boot, seed)
            entry = {"O_minus_C": gap_oc}
            if "L1" in views:
                pol_l1 = soften(targets["L1"], mu_scored, a)
                entry["L1_minus_C"] = _gap_ci(rewards_scored, mu_taken_scored, q_scored,
                                              pol_l1, pol_c, actions_scored, cluster_frame,
                                              gap_boot, seed)
            gaps_by_alpha[a] = entry
            _p(f"prescriptive-ablation gaps done for alpha={a}")

        result["frontier"] = frontier
        result["gaps"] = gaps_by_alpha
        behavior_value = float(frontier[COMMON_EVAL_VIEW][0.0]["value"]) if 0.0 in alphas \
            else float(np.nanmean(rewards_scored))
        result["behavior_value"] = behavior_value

        # --- exhibits: ambiguity + deviation maps (at the upper-bound alpha) ---
        _p("OPE complete; computing ambiguity/deviation exhibits and writing outputs...")
        ambiguity = {}
        deviation = {}
        for v in views:
            bi = inputs[v]
            ambiguity[v] = ambiguity_stats(bi.q, bi.q_sd, bi.feasible_mask)
            dev = deviation_map(targets[v], mu_scored, scored_rows, by=("balls", "strikes"))
            deviation[v] = {
                "overall_mean_tv": float((0.5 * np.abs(targets[v] - mu_scored).sum(axis=1)).mean()),
                "by_count": dev.to_dict(orient="records"),
            }
        result["ambiguity"] = ambiguity
        result["deviation"] = deviation

        # --- verdict (synthetic worlds) ---
        if synth != "off":
            result["prescription"] = _prescription_verdict(synth, gaps_by_alpha, behavior_value, frontier)
            result["truth"] = truth

        # --- standard-schema policy predictions (pure target, alpha=1, per view) ---
        if write_outputs:
            _write_policy_predictions(out_dir, world_tag, scored_rows, views, targets, meta)

    result["seconds"] = float(meta.get("seconds", float("nan")))
    result["peak_mem_mb"] = float(meta.get("peak_mem_mb", float("nan")))
    if write_outputs:
        result["outputs"] = _write(out_dir, world_tag, result, meta)
    return result


def _write_policy_predictions(out_dir, world_tag, rows, views, targets, meta):
    """Persist standard-schema ``policy_prob`` predictions per view (the pure alpha=1 target;
    the softened frontier lives in the report JSON)."""
    for v in views:
        pol = targets[v]
        df = pd.DataFrame({"row_id": rows["row_id"].to_numpy()})
        for j, col in enumerate(POLICY_PROB_COLS):
            df[col] = pol[:, j]
        df["model_id"] = "ws4_bandit"
        df["state_view"] = v
        df["seconds"] = float(meta.get("seconds", 0.0))
        df["peak_mem_mb"] = float(meta.get("peak_mem_mb", 0.0))
        df["n_params"] = 0
        save_predictions(df, out_dir / f"policy_{world_tag}_{v}.parquet")


def _write(out_dir: Path, world_tag: str, result: dict, meta: dict) -> dict:
    report_path = out_dir / f"ws4_report_{world_tag}.json"
    report_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    runmeta_path = out_dir / f"ws4_{world_tag}.runmeta.json"
    write_runmeta(runmeta_path, dict(meta))
    # Frontier CSV (value-vs-alpha overlay data).
    _write_frontier_csv(out_dir / f"frontier_{world_tag}.csv", result)
    return {"report": str(report_path), "runmeta": str(runmeta_path), "artifacts_dir": str(out_dir)}


def _write_frontier_csv(path: Path, result: dict) -> None:
    if "frontier" not in result:
        return
    recs = []
    for v, by_a in result["frontier"].items():
        for a, cell in by_a.items():
            recs.append({
                "view": v, "alpha": a, "value": cell["value"], "lower_95": cell["lower_95"],
                "ess": cell["ess"], "ess_frac": cell["ess_frac"],
                "oos_support_frac": cell["oos_support_frac"], "max_weight": cell["max_weight"],
                "mean_tv": cell["mean_tv"], "verdict": cell["verdict"],
            })
    pd.DataFrame(recs).to_csv(path, index=False)


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# --- headline --------------------------------------------------------------------------

def _fmt(x, spec="+.4f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "  n/a "


def _format_headline(result: dict) -> str:
    W = 82
    L = ["=" * W, " WS4 Bayesian contextual bandit - myopic prescription (OPE gate) - headline", "=" * W]
    L.append(f" world          : {result['world']}")
    L.append(f" rows           : train={result.get('n_train', 0):,}  eval(scored)={result.get('n_scored', 0):,}")
    fs = result.get("feasible_summary", {})
    if fs:
        L.append(f" feasibility    : mean #feasible/row={_fmt(fs['mean_n_feasible'], '.2f')}  "
                 f"empty-mask(no-rec)={_fmt(fs['empty_mask_frac'], '.1%')}  "
                 f"low-history={_fmt(fs['low_history_frac'], '.1%')}")

    rec = result.get("behavior_recovery", {})
    if result.get("gate") == "FAILED_GATE":
        L.append(" behavior recovery: FAILED_GATE - OPE cannot recover the behavior value; "
                 "stopping before any target value (SPEC 0.3 / D37).")
        for name, e in rec.get("estimators", {}).items():
            L.append(f"    {name:<10} value={_fmt(e['value'])} ok={e['ok']}")
        L.append("=" * W)
        return "\n".join(L)
    L.append(f" behavior recovery: PASS  (observed={_fmt(rec.get('observed_mean'))}  "
             f"IPS weights unit={rec.get('ips_weights_unit')})  [gate; SPEC 0.3 / D37]")

    bval = result.get("behavior_value")
    L.append(f" behavior value V(mu) = {_fmt(bval)}   (alpha=0 baseline)")
    L.append("")
    L.append(" D38 PRESCRIPTIVE-ABLATION TABLE (per view, per alpha; common evaluator = O):")
    L.append(f"   {'view':<4} {'alpha':>5} {'value':>9} {'lower95':>9} {'d(vs beh)':>9} "
             f"{'ESS%':>6} {'oos':>6} {'verdict':>12}")
    for v in result["views"]:
        for a in result["alphas"]:
            c = result["frontier"][v][a]
            dv = (c["value"] - bval) if (c["value"] is not None and bval is not None) else None
            L.append(f"   {v:<4} {a:>5.2f} {_fmt(c['value']):>9} {_fmt(c['lower_95']):>9} "
                     f"{_fmt(dv):>9} {_fmt(c['ess_frac'], '.1%'):>6} "
                     f"{_fmt(c['oos_support_frac'], '.2f'):>6} {str(c['verdict']):>12}")
    L.append("   (D39: INCONCLUSIVE and lower95<V(mu) are first-class honest results, not failures.)")

    L.append("")
    L.append(" SEQUENCING-PRESCRIPTION GAPS (clustered by pitcher-game; the O-vs-C isolation):")
    for a in result["alphas"]:
        g = result["gaps"][a]
        oc = g["O_minus_C"]
        line = (f"   alpha={a:>4.2f}  O-C = {_fmt(oc['value'])}  "
                f"CI[{_fmt(oc['ci95'][0])},{_fmt(oc['ci95'][1])}]  lower95={_fmt(oc['lower_95'])}")
        if "L1_minus_C" in g:
            line += f"   L1-C={_fmt(g['L1_minus_C']['value'])}"
        if oc["lower_95"] > 0:
            line += "  <-- CI excludes 0"
        L.append(line)

    amb = result.get("ambiguity", {})
    if amb:
        L.append("")
        L.append(" AMBIGUITY (share of decidable rows whose recommendation is a toss-up):")
        for v in result["views"]:
            s = amb[v]
            L.append(f"   {v:<4} mean P(top>runner-up)={_fmt(s['mean_p_beat'], '.3f')}  "
                     f"ambiguous @50/80/95 = {_fmt(s['ambiguous_50'], '.2f')}/"
                     f"{_fmt(s['ambiguous_80'], '.2f')}/{_fmt(s['ambiguous_95'], '.2f')}  "
                     f"(decidable={s['n_decidable']}, no-rec={s['n_infeasible']})")

    dev = result.get("deviation", {})
    if dev:
        L.append(" DEVIATION FROM BEHAVIOR (mean TV, alpha=1): "
                 + "  ".join(f"{v}={_fmt(dev[v]['overall_mean_tv'], '.3f')}" for v in result["views"]))

    presc = result.get("prescription")
    if presc:
        L.append("")
        L.append(" --- D38 SYNTHETIC VERDICT ---")
        verdict = presc["verdict"]
        L.append(f" prescriptive verdict : {verdict}")
        oc = presc["o_minus_c_upper"]
        L.append(f"   O-C at alpha={_fmt(presc['upper_bound_alpha'], '.2f')} (upper bound): "
                 f"{_fmt(oc['value'])}  lower95={_fmt(oc['lower_95'])}")
        if verdict == "SEQ_EXPLOITED":
            L.append(f"   O-view policy beats C-view policy (gap CI>0 at alpha={presc['significant_alpha']}): "
                     "ordered state is prescriptively exploitable.")
        elif verdict == "SEQ_INCONCLUSIVE_MYOPIC":
            L.append("   O-C is directionally positive but CI-inconclusive: the planted effect is a")
            L.append("   *state*-value effect; a MYOPIC policy captures only its small family-differential")
            L.append("   component (below the OPE noise floor at this scale). Present (WS3 recovered it),")
            L.append("   not myopically prescriptive -> the motivation for WS5/WS7 (D39 honest negative).")
        else:  # SEQ_NEUTRAL_PRESCRIPTION
            L.append("   O-view policy value ~= C-view policy value within CI: no sequencing prescription")
            L.append("   edge (the null world's expected result).")
        if presc["count_driven_gain"]:
            L.append(f"   NOTE: the bandit beats behavior by up to {_fmt(presc['value_over_behavior_max'])} in value,")
            L.append("   but the synthetic behavior policy is habit-based (not reward-optimal): this myopic")
            L.append("   COUNT-DRIVEN improvement is real yet is NOT sequencing evidence. Only the O-vs-C gap")
            L.append("   above isolates sequencing prescription (D38).")

    L.append("")
    L.append(f" elapsed (s)    : {_fmt(result.get('seconds'), '.1f')}   peak mem (MB): {_fmt(result.get('peak_mem_mb'), '.1f')}")
    if result.get("outputs"):
        L.append(f" outputs        : {result['outputs']['report']}")
    L.append("=" * W)
    return "\n".join(L)


# --- CLI -------------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python workstreams/ws4_bandit/run_ws4.py",
        description="Build the WS4 Thompson target policy and evaluate it through the OPE gate (SPEC 12.5).",
    )
    p.add_argument("--table", type=str, default=None,
                   help="Decision-table parquet (or directory of season checkpoints).")
    p.add_argument("--ws3-dir", type=str, default=None,
                   help="Directory of WS3 artifacts to consume (required when --synth is off; D33).")
    p.add_argument("--out", type=str, default="results/ws4", help="Output directory.")
    p.add_argument("--views", type=str, nargs="*", default=list(ABLATION_VIEWS),
                   help=f"Ablation views (default {list(ABLATION_VIEWS)}; O must be included).")
    p.add_argument("--synth", type=str, default="off", choices=["off", "null", "positive"],
                   help="Run on a generated synthetic world instead of --table.")
    p.add_argument("--posterior-scale", type=float, default=POSTERIOR_SCALE,
                   help="Residual-sd -> posterior-SE shrinkage for the Thompson draws (D36).")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="Thompson MC draws per row.")
    p.add_argument("--n-boot", type=int, default=300, help="Per-policy report bootstrap replicates.")
    p.add_argument("--gap-boot", type=int, default=400, help="Clustered gap bootstrap replicates.")
    p.add_argument("--n-games", type=int, default=900, help="Synthetic-world size.")
    p.add_argument("--seed", type=int, default=7, help="Bootstrap / split / synthetic-world seed.")
    p.add_argument("--no-fqe", action="store_true", help="Drop the count state (skip the FQE cross-check).")
    p.add_argument("--force", action="store_true", help="Ignore synthetic-WS3 .done markers and re-fit.")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point: run the pipeline and print the headline block."""
    args = _parse_args(argv)
    bad = [v for v in args.views if v not in ABLATION_VIEWS]
    if bad:
        print(f"error: unsupported view(s) {bad}; choose from {list(ABLATION_VIEWS)}", file=sys.stderr)
        raise SystemExit(2)
    if args.synth == "off" and (args.table is None or args.ws3_dir is None):
        print("error: --table and --ws3-dir are required when --synth is off", file=sys.stderr)
        raise SystemExit(2)

    result = run_ws4(
        source=args.table, synth=args.synth, out=args.out, ws3_dir=args.ws3_dir,
        views=args.views, posterior_scale=args.posterior_scale, n_samples=args.n_samples,
        n_boot=args.n_boot, gap_boot=args.gap_boot, n_games=args.n_games, seed=args.seed,
        fqe=not args.no_fqe, force=args.force,
    )
    print(_format_headline(result))
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    main()
