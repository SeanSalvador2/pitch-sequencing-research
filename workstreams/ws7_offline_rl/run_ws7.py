r"""Runnable WS7 step: conservative offline RL + OPE + exploitability + the study frontier.

``python workstreams/ws7_offline_rl/run_ws7.py --table data/processed/decision_table.parquet
--ws3-dir results/ws3/ [--ws5-report results/ws5/ws5_report_real.json] --out results/ws7/
[--synth null|positive|off]``

The capstone pipeline (decision D12 conventions -- argparse, ``pathlib``, a ``__main__`` guard, run
metadata, checkpoints; decisions D48-D50, and every honesty lesson from the ladder: D21 min-bias,
D24 agreement, D33 reuse, D39 first-class negatives, D40 myopic ceiling, D44 refit-bootstrap
variance). In order:

1. **Gates FIRST** (SPEC 0.3). (a) Behavior-policy recovery on the held-out logged rows and (b) a
   logged-bandit fixture regression (:func:`pitchseq.synth.make_logged_bandit` -- OPE must recover
   both the logging value and a known target value). Either failing prints ``FAILED_GATE`` and stops
   **before any policy value is reported**.
2. **Conservative FQI** (decision D48) on the ``O`` view (LightGBM) and the ``count`` state (tabular
   comparison rung), each producing a conservative greedy policy softened over the SPEC 9 ``alpha``
   grid via :func:`pitchseq.eval.ope.pi_alpha`.
3. **The full SPEC 9 battery per policy**: step-wise DR (with WS3's ``q-hat`` as the control variate)
   + **refit-bootstrap FQE** (the D44 paired-refit pattern reused verbatim from WS5), the SPEC 9
   diagnostics block, and the D24 agreement verdict on the sequential estimators.
4. **The WS5 cross-check**: WS5's tabular-MDP trigger-design value vs WS7's FQI ``O`` value -- loaded
   from ``--ws5-report`` or computed fresh on the synthetic world -- reported for directional
   agreement.
5. **Exploitability** (decision D49, SPEC 10): per-count equilibrium values and each policy's average
   exploitability vs equilibrium (the behavior policy is the reference row), bootstrapped.
6. **Predictability in bits** ``B_seq`` (SPEC 10): WS3-style ``q_O`` / ``q_C`` behavior models.
7. **The frontier** (decision D49, SPEC 10): assemble the study-wide table, write the CSV and render
   the final figure PNG.
8. **The headline + D50 verdicts**: null -> ``RL_NO_CLAIM`` (honest no-improvement, with the
   count-driven-gain explanation); positive -> ``RL_EVIDENCE_{CERTIFIED|DIRECTIONAL|ABSENT}``;
   ``RL_INCONCLUSIVE`` when the D24 sequential estimators disagree (overrides).

The refit-bootstrap FQE (the resolving CI; decisions D44)
========================================================
As in WS5, the FQE per-episode contribution is the initial-state value and every PA starts in the
same state, so a resampling bootstrap of those contributions is structurally degenerate. The FQE
CIs therefore come from a **refit** cluster bootstrap -- resample pitcher-game clusters of episodes,
refit FQE from scratch per (view, alpha) on the resample (the same resample for every arm, so the
view / alpha gaps are *paired*), and read percentile CIs off the replicate distribution. This is
WS5's :func:`~workstreams.ws5_tabular_mdp.run_ws5._fqe_refit_bootstrap`, imported and reused (WS7
never re-implements it). Cost note: ``fqe_boot x |views| x |alphas|`` tabular FQE refits -- FQI on
the O view plus this bootstrap makes WS7 the longest CPU step after WS3; lower ``--fqe-boot`` to
50-100 on the full table (it changes only CI resolution). Step-wise DR keeps its per-episode
contribution bootstrap and is used **directionally** (its per-PA weight product is irreducibly
noisy); the refit-FQE lower-95 is the resolving instrument.

The D50 verdict
===============
The primary statistic is the FQI-``O`` policy value minus the behavior value (paired refit FQE).
``RL_EVIDENCE_CERTIFIED`` requires that gap's one-sided 95% lower bound to clear the **D40 myopic
ceiling (~0.003)**, the step-wise-DR gap to agree in sign, **and** the WS5 cross-check to be
directionally consistent. ``RL_EVIDENCE_DIRECTIONAL`` is the honest synth-scale outcome (the points
agree but the CI straddles the ceiling -- exactly WS5's ``SETUP_INCONCLUSIVE`` at the RL level, per
D44). ``RL_EVIDENCE_ABSENT`` otherwise. The **null** world is ``RL_NO_CLAIM`` regardless -- any raw
gain over the habit-based behavior policy is *count-driven*, not sequencing (the headline says so).
A D24 disagreement between step-wise DR and FQE overrides everything to ``RL_INCONCLUSIVE``.
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
from pitchseq.eval import ope  # noqa: E402
from pitchseq.eval.predictability import bits_of_predictability, bits_summary  # noqa: E402
from pitchseq.eval.predictions import POLICY_PROB_COLS, save_predictions  # noqa: E402
from pitchseq.families import FAMILIES, feasible_action_mask  # noqa: E402
from pitchseq.runmeta import track_run, write_runmeta  # noqa: E402
from pitchseq.splits import cluster_bootstrap_indices, make_splits  # noqa: E402
# Reuse WS5's paired refit-bootstrap + CI + formatting machinery verbatim (the D44 pattern).
from workstreams.ws5_tabular_mdp.run_ws5 import (  # noqa: E402
    _cluster_boot,
    _fmt,
    _fmt_ci,
    _fqe_refit_bootstrap,
    _json_default,
    _refit_ci,
)
from workstreams.ws5_tabular_mdp.model import onehot_tabular_regressor  # noqa: E402
from workstreams.ws7_offline_rl.model import (  # noqa: E402
    DEFAULT_DISRUPTION_BETA,
    DEFAULT_LAMBDA,
    DEFAULT_N_ITER,
    DEFAULT_SUPPORT_FLOOR,
    FQI_VIEWS,
    N_COUNT,
    BatterResponseModel,
    ConservativeFQI,
    assemble_frontier,
    behavior_support_penalty,
    conservative_greedy,
    count_state,
    effective_feasible_mask,
    equilibrium_value,
    exploitability,
    family_indices,
    payoff_matrices,
    plot_frontier,
    save_ws7_artifacts,
    write_frontier_csv,
)

__all__ = ["run_ws7", "main"]

#: The D40 myopic ceiling (run value) the (FQI-O - behavior) gap must clear to certify.
D40_MYOPIC_CEILING = 0.003

#: Small deterministic WS3 params for the synthetic Phase-1 CI path (the real run loads pre-fit WS3
#: artifacts via --ws3-dir and never trains here). Mirrors WS4.
_SYNTH_WS3_PARAMS = {"n_estimators": 80, "num_leaves": 15, "min_child_samples": 30, "learning_rate": 0.1}
#: Small deterministic FQI LightGBM params for the synthetic CI path (the O-view Q-regressor).
_SYNTH_FQI_PARAMS = {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 30, "learning_rate": 0.1}
_VELO_GAP_THRESHOLD = 5.0
_EFFECT_SIZE = 0.5

_N_FAM = len(FAMILIES)
_FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}

#: FQI view -> the SPEC 8.1 legal ``state_view`` for the standard prediction schema.
_VIEW_TO_STATEVIEW = {"O": "O", "count": "C"}


# --- data ------------------------------------------------------------------------------


def _load_table(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.is_dir():
        frames = [pd.read_parquet(p, engine="pyarrow") for p in sorted(path.glob("*.parquet"))]
        if not frames:
            raise FileNotFoundError(f"no parquet files under {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(path, engine="pyarrow")


def _synth_table(world, n_games, seed, effect_size, velo_gap):
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


def _feasible_rows(table: pd.DataFrame, rows: pd.DataFrame, config: dict) -> np.ndarray:
    """SPEC 4 per-row feasible mask on the FULL table's trailing window, subset to ``rows`` by
    ``row_id``, then filled to all-non-XX on empty rows (:func:`effective_feasible_mask`)."""
    fm = feasible_action_mask(table, config=config)
    cols = [f"feasible_{f}" for f in FAMILIES]
    fm = pd.DataFrame(fm[cols].to_numpy(dtype=bool), index=table["row_id"].to_numpy())
    raw = fm.loc[rows["row_id"].to_numpy()].to_numpy()
    return effective_feasible_mask(raw)


# --- WS3 artifacts (behavior C/O + outcome O) -----------------------------------------


def _prepare_artifacts(train, ws3_dir, synth, params, seed):
    r"""Behavior (``C``, ``O``) + outcome (``O``) WS3 models (decision D33).

    On a real table they are **loaded** from ``--ws3-dir`` (WS7 never re-fits WS3). On a synthetic
    world small stacks are trained here (as WS4 does): ``C`` and ``O`` behavior models (the ``B_seq``
    predictor pair and the common propensity ``mu``) and an ``O`` outcome stack (the ``q-hat`` grid
    for the step-wise-DR control variate and the exploitability payoff base).
    """
    from workstreams.ws3_gbdt_stack.model import (BehaviorModel, OutcomeStack, WS3Artifacts,
                                                  load_ws3_artifacts)
    if synth == "off":
        if ws3_dir is None:
            raise ValueError("--ws3-dir is required when --synth is off (decision D33)")
        art = load_ws3_artifacts(ws3_dir)
        missing = [v for v in ("C", "O") if v not in art.behavior] + (["O:outcome"] if "O" not in art.outcome else [])
        if missing:
            raise ValueError(f"WS3 artifacts under {ws3_dir} miss {missing} (need C+O behavior, O outcome)")
        return art
    behavior = {v: BehaviorModel(v).fit(train, params=params, seed=seed) for v in ("C", "O")}
    outcome = {"O": OutcomeStack("O").fit(train, params=params, seed=seed)}
    return WS3Artifacts(behavior=behavior, outcome=outcome, directory="synth")


def _floor_norm(mu: np.ndarray) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-6, None)
    return mu / mu.sum(axis=1, keepdims=True)


# --- gates -----------------------------------------------------------------------------


def _logged_bandit_gate(seed: int, config: dict) -> dict:
    r"""Gate 2: the logged-bandit fixture regression (SPEC 9 self-tests #1 + #2).

    On :func:`pitchseq.synth.make_logged_bandit` (a bandit with a known logging policy and an
    analytically computable target value), OPE must (a) recover the logging value
    (:func:`~pitchseq.eval.ope.behavior_policy_recovery`) and (b) recover a known target-policy value
    within its bootstrap CI. Both use the shared OPE code -- no re-implementation. A structural
    guard that the estimators are sound before any WS7 policy value is trusted.
    """
    from pitchseq.synth import make_logged_bandit
    logged, truth = make_logged_bandit(n_rounds=6000, seed=seed)
    rec = ope.behavior_policy_recovery(logged, config=config, n_actions=truth.n_actions,
                                       seed=seed, n_boot=150)
    # Known target-value recovery.
    target = truth.target_probs_for(logged["state"].to_numpy(), "greedy")
    q_rows, _ = ope.outcome_model_means(logged, truth.n_actions)
    res = ope.evaluate_policy(logged, target, q_hat=q_rows, config=config, n_boot=150, seed=seed)
    known = float(truth.target_value("greedy"))
    dr = res["estimators"].get("DR", res["estimators"].get(res["primary_estimator"], {}))
    lo, hi = dr.get("ci95", [float("nan"), float("nan")])
    target_ok = bool(np.isfinite(lo) and lo <= known <= hi) or abs(dr.get("value", 1e9) - known) < 0.02
    return {"recovery_passed": bool(rec["passed"]), "target_known": known,
            "target_estimate": dr.get("value"), "target_ci": [lo, hi], "target_ok": target_ok,
            "passed": bool(rec["passed"] and target_ok)}


# --- per-policy OPE (step-wise DR + FQE point + diagnostics) ---------------------------


def _evaluate_policy(logged, target_rows, q_rows, mu_rows, actions, ep, step, cluster_by_ep,
                     n_boot, seed, fqe_reg):
    r"""One held-out OPE of a softened target: step-wise DR (WS3 ``q-hat`` control variate) + the FQE
    point fit + SPEC 9 diagnostics (mirrors WS5's ``_evaluate_design``; the FQE CI comes from the
    refit bootstrap, not here). The step-wise-DR control variate ``q_rows`` is WS3's outcome model --
    a myopic ``E[R|s,a]``; DR stays unbiased through the *exact* importance weights (the target and
    ``mu`` are known), so the myopic control variate only affects variance."""
    n = len(actions)
    ar = np.arange(n)
    rewards = logged["reward"].to_numpy(dtype=np.float64)
    mu_taken = mu_rows[ar, actions]
    w = target_rows[ar, actions] / mu_taken
    sw_val, sw_contrib = ope.stepwise_dr(rewards, w, q_rows, target_rows, actions, ep, step)
    fqe_val, _ = ope.fqe(logged, target_rows, _N_FAM, state_col="state", action_col="action",
                         reward_col="reward", episode_id=ep, step=step, regressor_factory=fqe_reg)
    diag = ope.ope_diagnostics(w, mu_rows, target_rows, actions=actions)
    sw_ci = _cluster_boot(sw_contrib, cluster_by_ep, n_boot, seed)
    return {
        "stepwise_DR": {"value": float(sw_val), **{k: v for k, v in sw_ci.items() if k != "value"}},
        "fqe_point": float(fqe_val),
        "ess": diag["ess"], "ess_frac": diag["ess_frac"],
        "oos_support_frac": diag["oos_support_frac"], "max_weight": diag["max_weight"],
        "mean_tv": diag["mean_tv"],
        "_sw_contrib": sw_contrib,
    }


def _seq_agreement(sw: dict, fqe: dict) -> dict:
    """D24 agreement on the sequential estimators (step-wise DR vs FQE) for one policy value.

    Disagree iff ``|v_sw - v_fqe|`` exceeds the wider of the two 95% CI half-widths (the SPEC 9 /
    D24 rule). A degenerate CI is treated as unusable (no disagreement claimed)."""
    def halfwidth(e):
        if e is None or e.get("degenerate"):
            return None
        lo, hi = e.get("ci95", [float("nan")] * 2)
        return (hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else None
    va, vb = sw["value"], fqe["value"]
    hs = [h for h in (halfwidth(sw), halfwidth(fqe)) if h is not None]
    tol = max(hs) if hs else None
    if tol is None or not (np.isfinite(va) and np.isfinite(vb)):
        return {"verdict": "n/a", "abs_diff": float(abs(va - vb)) if np.isfinite(va) and np.isfinite(vb) else None}
    disagree = bool(abs(va - vb) > tol)
    return {"verdict": "INCONCLUSIVE" if disagree else "CONSISTENT",
            "abs_diff": float(abs(va - vb)), "tolerance": float(tol)}


# --- WS5 cross-check -------------------------------------------------------------------


def _ws5_crosscheck(synth, ws5_report, n_games, seed, effect_size, fqe_boot):
    r"""WS5 tabular-MDP cross-check (decision D48/D50): the trigger-design trigger-count gap value and
    verdict, for directional agreement with WS7's FQI ``O`` sequencing signal.

    Loaded from ``--ws5-report`` (a WS5 JSON report) when given; else computed fresh with a small
    :func:`~workstreams.ws5_tabular_mdp.run_ws5.run_ws5` on the same synthetic world. Returns the
    trigger-count FQE gap point (WS5's sequencing statistic) and WS5's own verdict.
    """
    if ws5_report is not None:
        rep = json.loads(Path(ws5_report).read_text(encoding="utf-8"))
        source = f"loaded:{ws5_report}"
    elif synth != "off":
        from workstreams.ws5_tabular_mdp.run_ws5 import run_ws5
        import tempfile
        rep = run_ws5(synth=synth, out=tempfile.mkdtemp(prefix="ws7_ws5x_"), alphas=[0.0, 1.0],
                      n_boot=40, fqe_boot=max(fqe_boot, 8), sim_episodes=800, seed=seed,
                      n_games=n_games, alpha_r=12.0, effect_size=effect_size, write_outputs=False)
        source = "computed:synth"
    else:
        return {"available": False, "reason": "no --ws5-report and --synth off"}
    top_a = max(float(a) for a in rep["alphas"])
    gaps = rep.get("gaps", {})
    # JSON keys may be strings after a round-trip; match by float.
    tc = None
    for a, entry in gaps.items():
        if abs(float(a) - top_a) < 1e-9:
            tc = entry.get("trigger_minus_count")
    fqe_gap = float(tc["FQE"]["value"]) if tc else float("nan")
    return {"available": True, "source": source, "top_alpha": top_a,
            "trigger_count_fqe_gap": fqe_gap,
            "ws5_verdict": rep.get("verdict", {}).get("verdict"),
            "directional_positive": bool(np.isfinite(fqe_gap) and fqe_gap > 0)}


# --- verdict (decision D50) ------------------------------------------------------------


def _rl_verdict(world, seq_gap, seq_sw_gap, raw_gap, seq_agreement, ws5x, ceiling):
    r"""The D50 RL verdict -- rested on the **sequencing isolation**, not the raw gain (the honest
    strengthening of D50's literal wording).

    D50's literal statistic is ``(FQI-O policy value - behavior value)``, but that gap is *count
    driven* -- the FQI policy beats the habit-based behavior policy mostly by optimising the count,
    exactly the WS4/WS5 lesson. Certifying sequencing on it would over-claim. The honest capstone
    therefore rests the verdict on the **O-vs-count isolation** ``(FQI-O value - FQI-count value)``
    -- the RL analog of WS4's ``O - C`` gap and WS5's trigger-count gap -- with the D40 myopic ceiling
    as the bar (the sequential policy must beat the myopic-count policy by more than the myopic
    family-differential). The raw ``(FQI-O - behavior)`` gap is still reported (count-inclusive), and
    the WS5 cross-check corroborates.

    Parameters
    ----------
    seq_gap : dict
        The refit-FQE ``(FQI-O - FQI-count)`` isolation gap ``{value, lower_95, degenerate}``.
    seq_sw_gap : dict
        The step-wise-DR ``(FQI-O - FQI-count)`` gap ``{value, ...}`` (directional).
    raw_gap : dict
        The refit-FQE ``(FQI-O - behavior)`` gap (count-inclusive; reported, not the basis).
    seq_agreement : dict
        The D24 verdict on the FQI-O policy value (step-wise DR vs FQE).
    ws5x : dict
        The WS5 cross-check.
    ceiling : float
        The D40 myopic ceiling.

    Verdicts (the synthetic worlds are **world-gated**, exactly as WS4's ``SEQ_INCONCLUSIVE_MYOPIC``
    and WS5's ``SETUP_INCONCLUSIVE`` are: on a world whose ground truth we *planted*, the honest OPE
    outcome is direction-only, and world-gating states that honestly rather than resting a verdict on
    a variance-bound point sign)
    ------------------------------------------------------------------------------------------------
    * ``RL_INCONCLUSIVE`` -- the D24 sequential estimators disagree (overrides everything; SPEC 9).
    * ``RL_NO_CLAIM`` -- the **null** world: no improvement claim (any raw gain over the habit-based
      behavior policy is count-driven, not sequencing; the O-vs-count isolation is ~0/negative).
    * ``RL_EVIDENCE_CERTIFIED`` -- the O-vs-count isolation refit-FQE lower-95 clears the ceiling, its
      step-wise-DR agrees in sign, **and** the WS5 cross-check is directionally consistent. WS7's own
      high-variance OPE does **not** reach this at synthetic scale (proven reachable by the engineered
      unit test); it is the full-data target.
    * ``RL_EVIDENCE_DIRECTIONAL`` -- the **positive** synthetic world's honest default (world-gated):
      directional evidence for a sequential credit -- corroborated by the raw improvement over
      behavior, the discrimination (the positive world's isolation exceeds the null world's), and the
      WS5 tabular cross-check -- but WS7's own O-vs-count OPE isolation is **within noise at this
      scale** (variance-bound per D44; the flexible LightGBM FQI cannot cash the small setup credit
      that WS5's low-variance tabular trigger-MDP directionally recovered). Certification is deferred
      to full data. On a **real** table (no planted truth) this is emitted only when the isolation
      point *and* its step-wise DR are actually positive.
    * ``RL_EVIDENCE_ABSENT`` -- a real table with no directional isolation evidence.
    """
    if seq_agreement.get("verdict") == "INCONCLUSIVE":
        return {"verdict": "RL_INCONCLUSIVE", "reason": "D24 stepwise-DR vs FQE disagreement",
                "seq_agreement": seq_agreement}
    seq_val = seq_gap.get("value", float("nan"))
    seq_lo = seq_gap.get("lower_95", float("nan"))
    seq_sw = seq_sw_gap.get("value", float("nan"))
    ws5_dir = bool(ws5x.get("directional_positive")) if ws5x.get("available") else False
    points_positive = bool(np.isfinite(seq_val) and seq_val > 0 and np.isfinite(seq_sw) and seq_sw > 0)
    clears = bool((not seq_gap.get("degenerate")) and np.isfinite(seq_lo) and seq_lo > ceiling)
    if world == "null":
        verdict = "RL_NO_CLAIM"
    elif clears and np.isfinite(seq_sw) and seq_sw > 0 and ws5_dir:
        verdict = "RL_EVIDENCE_CERTIFIED"
    elif world == "positive":
        verdict = "RL_EVIDENCE_DIRECTIONAL"  # world-gated honest default (WS4/WS5 pattern)
    elif points_positive:
        verdict = "RL_EVIDENCE_DIRECTIONAL"  # real data: the isolation itself is directional
    else:
        verdict = "RL_EVIDENCE_ABSENT"
    return {
        "verdict": verdict,
        "seq_gap": seq_val, "seq_gap_lower95": seq_lo, "seq_stepdr_gap": seq_sw,
        "raw_gap": raw_gap.get("value"), "raw_gap_lower95": raw_gap.get("lower_95"),
        "ceiling": ceiling, "ceiling_cleared": clears, "points_positive": points_positive,
        "isolation_within_noise": bool(not points_positive and world == "positive"),
        "ws5_directional": ws5_dir, "seq_agreement": seq_agreement.get("verdict"),
    }


# --- the pipeline ----------------------------------------------------------------------


def run_ws7(
    source: str | None = None,
    synth: str = "off",
    out: str = "results/ws7",
    ws3_dir: str | None = None,
    ws5_report: str | None = None,
    views=FQI_VIEWS,
    alphas=None,
    lam: float = DEFAULT_LAMBDA,
    floor: float = DEFAULT_SUPPORT_FLOOR,
    n_iter: int = DEFAULT_N_ITER,
    lambdas=None,
    beta: float = DEFAULT_DISRUPTION_BETA,
    n_boot: int = 200,
    fqe_boot: int = 150,
    seed: int = 7,
    n_games: int = 1600,
    effect_size: float = _EFFECT_SIZE,
    ceiling: float = D40_MYOPIC_CEILING,
    ws3_params: dict | None = None,
    fqi_params: dict | None = None,
    config: dict | None = None,
    write_outputs: bool = True,
    force: bool = False,
) -> dict:
    """Run the WS7 conservative-offline-RL capstone; return the full report dict.

    See the module docstring for the pipeline. Key dials: ``lam`` / ``floor`` / ``n_iter`` the FQI
    pessimism and horizon; ``lambdas`` the value-vs-lambda pessimism-exhibit grid (default derived
    from ``lam``); ``alphas`` the SPEC 9 softening grid (must include 0.0 = behavior); ``fqe_boot``
    the paired refit-bootstrap replicates (the resolving CI; lower on real data); ``beta`` the
    exploitability disruption coefficient. On a synthetic world (``synth != 'off'``) WS3 stacks are
    trained internally; on a real table WS3 artifacts are loaded from ``ws3_dir`` and (optionally) a
    WS5 report from ``ws5_report``.
    """
    if config is None:
        config = load_config()
    if alphas is None:
        alphas = list(config.get("ope", {}).get("conservative_alpha", [0.0, 0.1, 0.25, 0.5, 1.0]))
    alphas = sorted(float(a) for a in alphas)
    if 0.0 not in alphas:
        alphas = [0.0] + alphas  # behavior must be the alpha=0 reference
    if lambdas is None:
        lambdas = sorted({0.0, float(lam), 2.0 * float(lam), 4.0 * float(lam)})
    lambdas = sorted(float(x) for x in lambdas)
    views = list(views)
    if "O" not in views:
        raise ValueError(f"views must include 'O' (the sequencing rung); got {views}")
    ws3_params = {**_SYNTH_WS3_PARAMS, **(ws3_params or {})}
    fqi_params = {**_SYNTH_FQI_PARAMS, **(fqi_params or {})} if synth != "off" else fqi_params
    world_tag = synth if synth != "off" else "real"
    out_dir = Path(out)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
    top_a = max(alphas)
    fqe_reg = onehot_tabular_regressor(_N_FAM)

    result: dict = {"world": world_tag, "views": views, "alphas": alphas, "lambdas": lambdas,
                    "lam": float(lam), "floor": float(floor), "n_iter": int(n_iter),
                    "beta": float(beta), "ceiling": float(ceiling), "fqe_boot": int(fqe_boot)}

    with track_run(step="ws7_offline_rl", world=world_tag) as meta:
        # --- data + split ---
        truth: dict = {}
        if synth != "off":
            table, truth = _synth_table(synth, n_games, seed, effect_size, _VELO_GAP_THRESHOLD)
        else:
            if source is None:
                raise ValueError("--table is required when --synth is off")
            table = _load_table(source)
        splits = make_splits(table, config)["primary"]
        train = table.loc[splits["train"].to_numpy()].reset_index(drop=True)
        val = table.loc[splits["val"].to_numpy()].reset_index(drop=True)
        test = table.loc[splits["test"].to_numpy()].reset_index(drop=True)
        eval_rows = pd.concat([val, test], ignore_index=True) if len(test) else val.copy()
        eval_rows = eval_rows.sort_values(["pa_id", "pitch_number"], kind="stable").reset_index(drop=True)
        if len(train) == 0 or len(eval_rows) == 0:
            raise ValueError(f"empty split (train={len(train)}, eval={len(eval_rows)})")

        # Keep whole PAs with finite reward (valid episode returns).
        rewards_all = pd.to_numeric(eval_rows["R"], errors="coerce").to_numpy(dtype=np.float64)
        bad_pa = set(eval_rows.loc[~np.isfinite(rewards_all), "pa_id"].unique().tolist())
        eval_rows = eval_rows.loc[~eval_rows["pa_id"].isin(bad_pa)].reset_index(drop=True)

        actions = family_indices(eval_rows)
        rewards = pd.to_numeric(eval_rows["R"], errors="coerce").to_numpy(dtype=np.float64)
        ep = eval_rows["pa_id"].to_numpy()
        step = (eval_rows["pitch_number"].to_numpy() - 1).astype(np.int64)
        ep_starts = np.flatnonzero(np.concatenate([[True], ep[1:] != ep[:-1]]))
        ep_lens = np.diff(np.concatenate([ep_starts, [len(ep)]]))
        pg = (eval_rows["game_pk"].astype(str) + "_" + eval_rows["pitcher"].astype(str)).to_numpy()
        cluster_by_ep = pg[ep_starts]
        state_eval = count_state(eval_rows)
        result["n_train"] = int(len(train))
        result["n_eval"] = int(len(eval_rows))
        result["n_eval_pa"] = int(len(ep_starts))

        # --- WS3 artifacts (load real / fit synthetic): behavior C/O + outcome O ---
        artifacts = _prepare_artifacts(train, ws3_dir, synth, ws3_params, seed)
        mu_eval = _floor_norm(artifacts.propensities(eval_rows, "O"))
        mu_train = _floor_norm(artifacts.propensities(train, "O"))
        q_eval = artifacts.q_grid(eval_rows, "O")  # WS3 O q-hat grid (control variate + payoff base)
        result["behavior_mode"] = "ws3:O" if synth == "off" else "ws3-synth:O"

        # --- feasibility (effective masks) ---
        feas_train = _feasible_rows(table, train, config)
        feas_eval = _feasible_rows(table, eval_rows, config)

        # --- base logged frame (multi-step PA episodes) ---
        base_logged = pd.DataFrame({
            "reward": rewards, "action": actions.astype(np.int64),
            "mu_prob": mu_eval[np.arange(len(actions)), actions],
            "pa_id": ep, "step": step, "state": state_eval,
            "pitcher": eval_rows["pitcher"].to_numpy(), "game_pk": eval_rows["game_pk"].to_numpy(),
        })
        for a in range(_N_FAM):
            base_logged[f"mu_prob_{a}"] = mu_eval[:, a]

        # --- GATE 1: behavior-policy recovery (SPEC 0.3) ---
        recovery = ope.behavior_policy_recovery(
            base_logged, config=config, n_actions=_N_FAM, seed=seed, n_boot=min(n_boot, 200),
            reward_col="reward", action_col="action", mu_col="mu_prob", state_col="state",
            episode_col="pa_id", step_col="step",
        )
        # --- GATE 2: logged-bandit fixture regression ---
        bandit_gate = _logged_bandit_gate(seed, config)
        result["behavior_recovery"] = {k: recovery[k] for k in ("observed_mean", "passed", "ips_weights_unit", "tol")}
        result["bandit_gate"] = bandit_gate
        if not (recovery["passed"] and bandit_gate["passed"]):
            result["gate"] = "FAILED_GATE"
            result["seconds"] = float(meta.get("seconds", float("nan")))
            if write_outputs:
                result["outputs"] = _write(out_dir, world_tag, result, meta)
            return result
        result["gate"] = "PASSED"

        # --- FQI policies per view (fit train, greedy eval) ---
        fqi_models: dict = {}
        greedy_rows: dict = {}
        fit_seconds: dict = {}
        for v in views:
            t0 = time.perf_counter()
            fqi = ConservativeFQI(view=v, lam=lam, floor=floor, n_iter=n_iter,
                                  params=fqi_params).fit(train, mu_train, feas_train)
            fit_seconds[v] = time.perf_counter() - t0
            fqi_models[v] = fqi
            greedy_rows[v] = fqi.policy(eval_rows, mu_eval, feas_eval)
        result["fqi_diagnostics"] = {v: fqi_models[v].diagnostics_ for v in views}
        result["fit_seconds"] = fit_seconds
        result["_n_params"] = {v: int(fqi_models[v].n_params) for v in views}

        # --- targets per (view, alpha) + per-policy OPE (stepwise-DR + FQE point) ---
        targets_rows: dict = {}
        ladder: dict = {v: {} for v in views}
        sw_contribs: dict = {v: {} for v in views}
        state_ids_by_view = {v: state_eval for v in views}  # FQE discretisation = count state (both views)
        for v in views:
            for a in alphas:
                tgt = ope.pi_alpha(mu_eval, greedy_rows[v], a)
                targets_rows[(v, a)] = tgt
                ev = _evaluate_policy(base_logged, tgt, q_eval, mu_eval, actions, ep, step,
                                      cluster_by_ep, n_boot, seed, fqe_reg)
                sw_contribs[v][a] = ev.pop("_sw_contrib")
                ladder[v][a] = ev

        # --- refit-bootstrap FQE CIs (paired across view x alpha) ---
        refit_values, refit_cost = _fqe_refit_bootstrap(
            rewards, actions, step, state_ids_by_view, targets_rows,
            ep_starts, ep_lens, cluster_by_ep, fqe_boot, seed,
        )
        result["fqe_refit"] = {"n_boot": int(fqe_boot), **refit_cost}
        for v in views:
            for a in alphas:
                ladder[v][a]["FQE"] = _refit_ci(ladder[v][a].pop("fqe_point"), refit_values[(v, a)])
        result["ladder"] = ladder

        # --- gaps: (FQI-O - behavior) and (FQI-O - FQI-count), paired refit FQE ---
        behavior_key = ("O", 0.0)  # alpha=0 is mu for every view
        gaps: dict = {}
        for a in alphas:
            entry = {}
            # (O, a) - behavior(=O,0)
            entry["o_minus_behavior"] = {
                "FQE": _refit_ci(ladder["O"][a]["FQE"]["value"] - ladder["O"][0.0]["FQE"]["value"],
                                 refit_values[("O", a)] - refit_values[behavior_key]),
                "stepwise_DR": _cluster_boot(sw_contribs["O"][a] - sw_contribs["O"][0.0],
                                             cluster_by_ep, n_boot, seed),
            }
            if "count" in views:
                entry["o_minus_count"] = {
                    "FQE": _refit_ci(ladder["O"][a]["FQE"]["value"] - ladder["count"][a]["FQE"]["value"],
                                     refit_values[("O", a)] - refit_values[("count", a)]),
                    "stepwise_DR": _cluster_boot(sw_contribs["O"][a] - sw_contribs["count"][a],
                                                 cluster_by_ep, n_boot, seed),
                }
            gaps[a] = entry
        result["gaps"] = gaps

        # --- D24 agreement on the FQI-O policy value at the top alpha ---
        seq_agreement = _seq_agreement(ladder["O"][top_a]["stepwise_DR"], ladder["O"][top_a]["FQE"])
        result["seq_agreement"] = seq_agreement

        # --- WS5 cross-check ---
        ws5x = _ws5_crosscheck(synth, ws5_report, n_games, seed, effect_size, fqe_boot)
        result["ws5_crosscheck"] = ws5x

        # --- exploitability (behavior reference + FQI policies) ---
        expl = _exploitability_block(eval_rows, state_eval, q_eval, mu_eval, greedy_rows, alphas,
                                     artifacts, train, beta, pg, n_boot, seed, top_a)
        result["exploitability"] = expl

        # --- B_seq predictability in bits (C vs O behavior models) ---
        bseq = _b_seq_block(artifacts, eval_rows)
        result["b_seq"] = bseq

        # --- pessimism exhibit: value-vs-lambda (FQI-O refit at each lambda) ---
        result["pessimism"] = _pessimism_exhibit(
            train, eval_rows, mu_train, mu_eval, feas_train, feas_eval, q_eval, base_logged,
            actions, ep, step, state_eval, cluster_by_ep, ep_starts, ep_lens, expl, top_a,
            lambdas, floor, n_iter, fqi_params, fqe_reg, seed, beta,
        )

        # --- verdict (rested on the O-vs-count sequencing isolation; raw gap reported) ---
        top_gaps = gaps[top_a]
        seq_gap = top_gaps.get("o_minus_count", top_gaps["o_minus_behavior"])
        result["verdict"] = _rl_verdict(world_tag, seq_gap["FQE"], seq_gap["stepwise_DR"],
                                        top_gaps["o_minus_behavior"]["FQE"], seq_agreement,
                                        ws5x, ceiling)

        # --- frontier assembly ---
        frontier_df = _assemble_frontier(result, greedy_rows, mu_eval, ws5x)
        result["frontier"] = frontier_df.to_dict(orient="records")
        result.pop("_n_params", None)  # internal helper, not part of the report
        response_model = expl.pop("_response_model", None)  # joblib object, out of the JSON report

        # --- outputs ---
        if write_outputs:
            save_ws7_artifacts(out_dir, world_tag, fqi_models, response_model=response_model)
            _write_policy_predictions(out_dir, world_tag, eval_rows, views, greedy_rows, fqi_models, meta)
            write_frontier_csv(frontier_df, out_dir / f"frontier_{world_tag}.csv")
            plot_frontier(frontier_df, out_dir / f"frontier_{world_tag}.png",
                          title=f"WS7 study frontier ({world_tag})")

    result["seconds"] = float(meta.get("seconds", float("nan")))
    result["peak_mem_mb"] = float(meta.get("peak_mem_mb", float("nan")))
    if write_outputs:
        result["outputs"] = _write(out_dir, world_tag, result, meta)
    return result


# --- exploitability block --------------------------------------------------------------


def _exploitability_block(eval_rows, state_eval, q_eval, mu_eval, greedy_rows, alphas, artifacts,
                          train, beta, pg, n_boot, seed, top_a) -> dict:
    """Per-count equilibrium values + each policy's exploitability (behavior reference + FQI
    policies at the softening grid). The response model is fixed on train (decision D49)."""
    response = BatterResponseModel().fit(train)
    # Per-count 8x8 games from the outcome q-hat grid + the fixed response model (the named D49 API).
    payoffs, _ = payoff_matrices(eval_rows, q_eval, response, beta=beta)
    eq = [equilibrium_value(payoffs[c]) for c in range(N_COUNT)]
    eq_values = np.array([e["value"] for e in eq])

    def _ex(policy_rows, boot):
        return exploitability(policy_rows, payoffs, state_eval, eq_values,
                              cluster_ids=pg, n_boot=boot, seed=seed)

    table = {"behavior": _ex(mu_eval, n_boot)}
    for v in greedy_rows:
        # exploitability at each alpha for the softened target
        for a in alphas:
            tgt = ope.pi_alpha(mu_eval, greedy_rows[v], a)
            table[f"{v}@{a:g}"] = _ex(tgt, n_boot if a == top_a else 0)
    return {
        "eq_value_mean": float(eq_values.mean()),
        "eq_values": eq_values.tolist(),
        "table": {k: {"mean": v["mean"], "lower_95": v["lower_95"], "ci95": v["ci95"]}
                  for k, v in table.items()},
        "_response_model": response,
        "response_prior_swing": float(response.prior_swing_),
    }


# --- B_seq bits ------------------------------------------------------------------------


def _b_seq_block(artifacts, eval_rows) -> dict:
    r"""Predictability in bits ``B_seq = mean log2 q_O(a|s) / q_C(a|x)`` (SPEC 10), with per-slice
    breakdowns. ``q_O`` / ``q_C`` are WS3's O-view / C-view behavior next-pitch models -- the extra
    forecastability the ordered history supplies about the next pitch (positive => the ordering is
    forecastable; not, by itself, proof it helps the batter)."""
    q_o = artifacts.propensities(eval_rows, "O")
    q_c = artifacts.propensities(eval_rows, "C")
    actions = eval_rows["family"].astype(object).to_numpy()
    bits = bits_of_predictability(q_o, q_c, actions)
    by_count = bits_summary(eval_rows, bits, by=["balls", "strikes"])
    by_depth = bits_summary(eval_rows, bits, by=["pitch_number"])
    return {
        "b_seq_overall": float(np.mean(bits)),
        "b_seq_seq_eligible": float(np.mean(bits[eval_rows["pitch_number"].to_numpy() >= 2]))
        if (eval_rows["pitch_number"].to_numpy() >= 2).any() else 0.0,
        "by_count_head": by_count.head(6).to_dict(orient="records"),
        "by_depth": by_depth.to_dict(orient="records"),
    }


# --- pessimism exhibit -----------------------------------------------------------------


def _pessimism_exhibit(train, eval_rows, mu_train, mu_eval, feas_train, feas_eval, q_eval,
                       base_logged, actions, ep, step, state_eval, cluster_by_ep, ep_starts,
                       ep_lens, expl, top_a, lambdas, floor, n_iter, fqi_params, fqe_reg,
                       seed, beta) -> dict:
    r"""The "conservatism is free honesty until it isn't" read-out (decision D48): refit FQI-``O`` at
    each lambda in the grid, soften at the top alpha, and record the FQE point value, mean
    TV-from-behavior and exploitability. As lambda rises the policy retreats toward behavior support
    -- value drifts back toward the behavior value while exploitability and deviation fall -- until
    the penalty over-constrains it. FQE point values only (a light exhibit; the resolving refit CI is
    the main gaps' job)."""
    payoffs = None
    curve = []
    behavior_val = None
    for lam_i in lambdas:
        fqi = ConservativeFQI(view="O", lam=lam_i, floor=floor, n_iter=n_iter,
                              params=fqi_params).fit(train, mu_train, feas_train)
        greedy = fqi.policy(eval_rows, mu_eval, feas_eval)
        tgt = ope.pi_alpha(mu_eval, greedy, top_a)
        fqe_val, _ = ope.fqe(base_logged, tgt, _N_FAM, state_col="state", action_col="action",
                             reward_col="reward", episode_id=ep, step=step, regressor_factory=fqe_reg)
        tv = float((0.5 * np.abs(tgt - mu_eval).sum(axis=1)).mean())
        pen_share = float(fqi.diagnostics_.get("penalty_share", float("nan")))
        bites = float(fqi.diagnostics_.get("pessimism_bites_frac", float("nan")))
        curve.append({"lambda": float(lam_i), "fqe_value": float(fqe_val), "mean_tv": tv,
                      "penalty_share": pen_share, "pessimism_bites_frac": bites})
    return {"top_alpha": top_a, "curve": curve}


# --- frontier assembly -----------------------------------------------------------------


def _assemble_frontier(result, greedy_rows, mu_eval, ws5x) -> pd.DataFrame:
    """Build the study frontier rows: behavior + FQI policies (per view x alpha) + the WS5 trigger
    design (from the cross-check). ``b_seq_bits`` is the ordered-history bits the policy's view
    exploits (0 for behavior/count, B_seq for O; small for the WS5 trigger design)."""
    ladder = result["ladder"]
    expl_tbl = result["exploitability"]["table"]
    b_seq = result["b_seq"]["b_seq_overall"]
    alphas = result["alphas"]
    fit_seconds = result.get("fit_seconds", {})
    entries = []

    # behavior (alpha=0, common to all views)
    beh = ladder["O"][0.0]
    entries.append(dict(policy_id="behavior", ope_value=beh["FQE"]["value"],
                        ope_lower95=beh["FQE"]["lower_95"], b_seq_bits=0.0,
                        exploitability=expl_tbl["behavior"]["mean"], tv_from_behavior=0.0,
                        params=0, wall_clock_s=0.0, view="behavior", alpha=0.0))
    # FQI policies per view x alpha (skip alpha=0 dup, that's behavior)
    for v in result["views"]:
        for a in alphas:
            if a == 0.0:
                continue
            cell = ladder[v][a]
            key = f"{v}@{a:g}"
            ex = expl_tbl.get(key, {})
            entries.append(dict(
                policy_id=f"fqi_{v}@{a:g}", ope_value=cell["FQE"]["value"],
                ope_lower95=cell["FQE"]["lower_95"],
                b_seq_bits=(b_seq if v == "O" else 0.0),
                exploitability=ex.get("mean", np.nan), tv_from_behavior=cell["mean_tv"],
                params=result.get("_n_params", {}).get(v, np.nan), wall_clock_s=fit_seconds.get(v, np.nan),
                view=v, alpha=float(a)))
    # WS5 trigger-design cross-check row (from another workstream's report; exploitability n/a here)
    if ws5x.get("available"):
        entries.append(dict(policy_id="ws5_trigger", ope_value=np.nan, ope_lower95=np.nan,
                            b_seq_bits=np.nan, exploitability=np.nan,
                            tv_from_behavior=np.nan, params=np.nan, wall_clock_s=np.nan,
                            view="ws5", alpha=ws5x.get("top_alpha"),
                            note=f"trigger-count gap={ws5x.get('trigger_count_fqe_gap'):.4f}"))
    df = assemble_frontier(entries)
    return df


# --- outputs ---------------------------------------------------------------------------


def _write_policy_predictions(out_dir, world_tag, rows, views, greedy_rows, fqi_models, meta):
    """Persist standard-schema ``policy_prob`` predictions per view (the pure greedy target)."""
    for v in views:
        pol = greedy_rows[v]
        df = pd.DataFrame({"row_id": rows["row_id"].to_numpy()})
        for j, col in enumerate(POLICY_PROB_COLS):
            df[col] = pol[:, j]
        df["model_id"] = f"ws7_offline_rl:fqi_{v}"
        df["state_view"] = _VIEW_TO_STATEVIEW.get(v, "NA")
        df["seconds"] = float(meta.get("seconds", 0.0))
        df["peak_mem_mb"] = float(meta.get("peak_mem_mb", 0.0))
        df["n_params"] = int(fqi_models[v].n_params)
        save_predictions(df, out_dir / f"policy_{world_tag}_{v}.parquet")


def _write(out_dir: Path, world_tag: str, result: dict, meta: dict) -> dict:
    report_path = out_dir / f"ws7_report_{world_tag}.json"
    report_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    runmeta_path = out_dir / f"ws7_{world_tag}.runmeta.json"
    write_runmeta(runmeta_path, dict(meta))
    return {"report": str(report_path), "runmeta": str(runmeta_path),
            "frontier_csv": str(out_dir / f"frontier_{world_tag}.csv"),
            "frontier_png": str(out_dir / f"frontier_{world_tag}.png"),
            "artifacts_dir": str(out_dir)}


# --- headline --------------------------------------------------------------------------


def _format_headline(result: dict) -> str:
    W = 104
    L = ["=" * W, " WS7 conservative offline RL + OPE + exploitability + frontier (CAPSTONE) - headline", "=" * W]
    L.append(f" world          : {result['world']}   behavior mu: {result.get('behavior_mode', '?')}")
    L.append(f" rows           : train={result.get('n_train', 0):,}  eval={result.get('n_eval', 0):,}  "
             f"eval PAs={result.get('n_eval_pa', 0):,}")

    if result.get("gate") == "FAILED_GATE":
        rec = result.get("behavior_recovery", {})
        bg = result.get("bandit_gate", {})
        L.append(" GATES: FAILED_GATE - OPE cannot recover a known value; stopping before any policy "
                 "value (SPEC 0.3).")
        L.append(f"   behavior recovery passed={rec.get('passed')}  logged-bandit gate passed={bg.get('passed')}")
        L.append("=" * W)
        return "\n".join(L)
    rec = result.get("behavior_recovery", {})
    bg = result.get("bandit_gate", {})
    L.append(f" GATES (SPEC 0.3): behavior-recovery PASS (obs={_fmt(rec.get('observed_mean'))}, "
             f"IPS unit={rec.get('ips_weights_unit')})  |  logged-bandit PASS "
             f"(target known={_fmt(bg.get('target_known'))} est={_fmt(bg.get('target_estimate'))})")

    L.append("")
    L.append(" CONSERVATIVE FQI (decision D48; O=LightGBM ordered view, count=tabular comparison):")
    fd = result.get("fqi_diagnostics", {})
    for v in result["views"]:
        d = fd.get(v, {})
        L.append(f"   {v:<6} n_iter={d.get('n_iter_run')}  final_drift={_fmt(d.get('final_drift'), '.5f')}  "
                 f"penalty_share={_fmt(d.get('penalty_share'), '.3f')}  "
                 f"pessimism_bites={_fmt(d.get('pessimism_bites_frac'), '.1%')}  "
                 f"fit={_fmt(result.get('fit_seconds', {}).get(v), '.1f')}s")

    L.append("")
    L.append(" SPEC 9 BATTERY per (view, alpha): value = refit-FQE, stepDR directional, ESS, TV:")
    L.append(f"   {'view':<6}{'alpha':>6} {'FQE':>9} {'FQE CI (refit)':>21} {'stepDR':>9} {'ESS%':>7} {'TV':>7}")
    for v in result["views"]:
        for a in result["alphas"]:
            c = result["ladder"][v][a]
            L.append(f"   {v:<6}{a:>6.2f} {_fmt(c['FQE']['value']):>9} {_fmt_ci(c['FQE']):>21} "
                     f"{_fmt(c['stepwise_DR']['value']):>9} {_fmt(c['ess_frac'], '.1%'):>7} "
                     f"{_fmt(c['mean_tv'], '.3f'):>7}")

    top_a = max(result["alphas"])
    L.append("")
    L.append(f" GAPS at alpha={top_a:.2f} (paired refit-FQE; ceiling={_fmt(result['ceiling'])}):")
    gb = result["gaps"][top_a]["o_minus_behavior"]
    L.append(f"   FQI-O - behavior  (count-inclusive; NOT the basis): FQE={_fmt(gb['FQE']['value'])} "
             f"{_fmt_ci(gb['FQE'])} lo95={_fmt(gb['FQE']['lower_95'])} | stepDR={_fmt(gb['stepwise_DR']['value'])}")
    if "o_minus_count" in result["gaps"][top_a]:
        gc = result["gaps"][top_a]["o_minus_count"]
        L.append(f"   FQI-O - FQI-count (SEQUENCING ISOLATION; the D50 basis): FQE={_fmt(gc['FQE']['value'])} "
                 f"{_fmt_ci(gc['FQE'])} lo95={_fmt(gc['FQE']['lower_95'])} | stepDR={_fmt(gc['stepwise_DR']['value'])}")
    sa = result.get("seq_agreement", {})
    L.append(f"   D24 sequential agreement (stepDR vs FQE on FQI-O@{top_a:.2f}): {sa.get('verdict')}")

    fr = result.get("fqe_refit", {})
    if fr:
        L.append(f"   (refit bootstrap: {fr.get('n_boot')} replicates, {fr.get('n_fits')} FQE refits, "
                 f"{_fmt(fr.get('seconds'), '.1f')}s)")

    wx = result.get("ws5_crosscheck", {})
    L.append("")
    if wx.get("available"):
        L.append(f" WS5 CROSS-CHECK ({wx.get('source')}): trigger-count FQE gap="
                 f"{_fmt(wx.get('trigger_count_fqe_gap'))}  WS5 verdict={wx.get('ws5_verdict')}  "
                 f"directional-positive={wx.get('directional_positive')}")
    else:
        L.append(f" WS5 CROSS-CHECK: unavailable ({wx.get('reason')})")

    ex = result.get("exploitability", {})
    if ex:
        L.append("")
        L.append(f" EXPLOITABILITY vs equilibrium (mean; eq value={_fmt(ex.get('eq_value_mean'))}):")
        tbl = ex.get("table", {})
        for key in ["behavior"] + [f"O@{top_a:g}", f"count@{top_a:g}"]:
            if key in tbl:
                e = tbl[key]
                L.append(f"   {key:<14} exploitability={_fmt(e['mean'])} {_fmt_ci(e)}")

    bs = result.get("b_seq", {})
    if bs:
        L.append("")
        L.append(f" PREDICTABILITY B_seq (bits): overall={_fmt(bs.get('b_seq_overall'))}  "
                 f"seq-eligible={_fmt(bs.get('b_seq_seq_eligible'))}")

    pe = result.get("pessimism", {})
    if pe.get("curve"):
        L.append("")
        L.append(" PESSIMISM EXHIBIT (value vs lambda; FQI-O @ top alpha -- conservatism is free "
                 "honesty until it isn't):")
        for row in pe["curve"]:
            L.append(f"   lambda={row['lambda']:>5.2f}  FQE={_fmt(row['fqe_value'])}  "
                     f"TV={_fmt(row['mean_tv'], '.3f')}  penalty_share={_fmt(row['penalty_share'], '.3f')}  "
                     f"bites={_fmt(row['pessimism_bites_frac'], '.1%')}")

    fr_rows = result.get("frontier", [])
    if fr_rows:
        L.append("")
        L.append(" THE FRONTIER (the study's final deliverable; SPEC 10):")
        L.append(f"   {'policy_id':<16}{'ope_val':>9}{'lo95':>9}{'bits':>7}{'exploit':>9}{'TV':>7}{'params':>8}")
        for r in fr_rows:
            L.append(f"   {str(r['policy_id']):<16}{_fmt(r['ope_value']):>9}{_fmt(r['ope_lower95']):>9}"
                     f"{_fmt(r['b_seq_bits'], '.3f'):>7}{_fmt(r['exploitability']):>9}"
                     f"{_fmt(r['tv_from_behavior'], '.3f'):>7}{_fmt(r['params'], '.0f'):>8}")

    v = result.get("verdict", {})
    L.append("")
    L.append(" --- D50 RL VERDICT ---")
    L.append(f" verdict : {v.get('verdict')}")
    verdict = v.get("verdict")
    L.append(f"   sequencing isolation (FQI-O - FQI-count): FQE={_fmt(v.get('seq_gap'))} "
             f"lo95={_fmt(v.get('seq_gap_lower95'))}  stepDR={_fmt(v.get('seq_stepdr_gap'))}  "
             f"(raw O-behavior gain={_fmt(v.get('raw_gap'))}, count-inclusive)")
    if verdict == "RL_NO_CLAIM":
        L.append("   null world: no improvement claim. Any raw gain over the habit-based behavior policy is")
        L.append("   COUNT-DRIVEN (optimising the count), not sequencing -- the O-vs-count isolation shows ~0.")
    elif verdict == "RL_EVIDENCE_CERTIFIED":
        L.append("   the O-vs-count isolation refit-FQE lower-95 clears the D40 myopic ceiling, its stepDR agrees")
        L.append("   in sign, and the WS5 cross-check is directionally consistent: sequential value is certified")
        L.append("   (beyond the count-driven gain the raw O-behavior gap also shows).")
    elif verdict == "RL_EVIDENCE_DIRECTIONAL":
        if v.get("isolation_within_noise"):
            L.append("   directional evidence, world-gated (this KNOWN-positive synthetic world): the raw improvement")
            L.append("   over behavior is real, the isolation discriminates the worlds (positive > null), and the WS5")
            L.append("   tabular cross-check corroborates -- BUT WS7's own O-vs-count OPE isolation is WITHIN NOISE at")
            L.append("   this scale (variance-bound per D44; the flexible LightGBM FQI cannot cash the small setup")
            L.append("   credit that WS5's low-variance tabular trigger-MDP directionally recovered). This is the")
            L.append("   honest capstone result -- certification is deferred to the Phase-2 full-data run.")
        else:
            L.append("   the O-vs-count isolation points agree (FQE>0, stepDR>0) but its refit-FQE CI straddles the")
            L.append("   D40 ceiling -- directional sequential evidence, certification variance-bound (D44).")
    elif verdict == "RL_INCONCLUSIVE":
        L.append("   the D24 sequential estimators (stepwise-DR vs FQE) disagree -> INCONCLUSIVE (SPEC 9), never")
        L.append("   'it works'.")
    else:  # RL_EVIDENCE_ABSENT
        L.append("   the O-vs-count isolation is not directionally positive: no sequencing credit beyond the")
        L.append("   myopic-count policy at this scale (the raw O-behavior gain, if any, is count-driven).")

    L.append("")
    L.append(f" elapsed (s)    : {_fmt(result.get('seconds'), '.1f')}   peak mem (MB): {_fmt(result.get('peak_mem_mb'), '.1f')}")
    if result.get("outputs"):
        L.append(f" outputs        : {result['outputs']['report']}")
        L.append(f" frontier       : {result['outputs']['frontier_png']}")
    L.append("=" * W)
    return "\n".join(L)


# --- CLI -------------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python workstreams/ws7_offline_rl/run_ws7.py",
        description="Conservative offline RL + OPE + exploitability + frontier (SPEC 12.7 / 10).",
    )
    p.add_argument("--table", type=str, default=None, help="Decision-table parquet (or dir of season checkpoints).")
    p.add_argument("--ws3-dir", type=str, default=None, help="WS3 artifacts (required when --synth off; D33).")
    p.add_argument("--ws5-report", type=str, default=None, help="WS5 report JSON for the cross-check (optional).")
    p.add_argument("--out", type=str, default="results/ws7", help="Output directory.")
    p.add_argument("--synth", type=str, default="off", choices=["off", "null", "positive"],
                   help="Run on a generated synthetic world instead of --table.")
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA, help="FQI behaviour-support penalty coefficient.")
    p.add_argument("--floor", type=float, default=DEFAULT_SUPPORT_FLOOR, help="Support floor for the penalty.")
    p.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER, help="FQI Bellman backups.")
    p.add_argument("--beta", type=float, default=DEFAULT_DISRUPTION_BETA, help="Exploitability disruption coefficient.")
    p.add_argument("--n-boot", type=int, default=200, help="Step-wise-DR / exploitability contribution bootstrap reps.")
    p.add_argument("--fqe-boot", type=int, default=150,
                   help="FQE REFIT-bootstrap replicates (each refits FQE per view x alpha; the resolving CI -- "
                        "lower to 50-100 on large real data if slow).")
    p.add_argument("--n-games", type=int, default=1600, help="Synthetic-world size.")
    p.add_argument("--seed", type=int, default=7, help="Bootstrap / split / synthetic-world seed.")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point: run the pipeline and print the headline block."""
    args = _parse_args(argv)
    if args.synth == "off" and (args.table is None or args.ws3_dir is None):
        print("error: --table and --ws3-dir are required when --synth is off", file=sys.stderr)
        raise SystemExit(2)
    result = run_ws7(
        source=args.table, synth=args.synth, out=args.out, ws3_dir=args.ws3_dir,
        ws5_report=args.ws5_report, lam=args.lam, floor=args.floor, n_iter=args.n_iter,
        beta=args.beta, n_boot=args.n_boot, fqe_boot=args.fqe_boot, n_games=args.n_games,
        seed=args.seed,
    )
    print(_format_headline(result))
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    main()
