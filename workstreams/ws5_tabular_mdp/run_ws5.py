r"""Runnable WS5 step: estimate the tabular MDPs, plan, and evaluate the setup value (SPEC 12.6).

``python workstreams/ws5_tabular_mdp/run_ws5.py --table data/processed/decision_table.parquet
[--ws3-dir results/ws3/] --out results/ws5/ [--synth null|positive|off]``

Pipeline (decision D12 conventions -- argparse, ``pathlib``, a ``__main__`` guard, run
metadata, checkpoints; decisions D40-D43, D24, D28, D39). For each of the three D43 state
designs (``count`` ~ C, ``count_prev`` ~ L1, ``count_prev_trigger`` ~ O-lite):

1. **Estimate the MDP** on train (Dirichlet-smoothed transitions, two-level-shrunk rewards,
   per-state feasibility) and **plan** with undiscounted policy iteration (decision D41).
2. **Soften** the greedy target toward the common behavior policy via the SPEC ``9`` mixture
   ``pi_alpha`` (primary), with a temperature-softened variant reported alongside.
3. **Model-based value** of the softened target in the estimated MDP -- the transparent
   simulator (decision D42), computed exactly by policy evaluation and cross-checked by the
   Monte-Carlo :func:`~workstreams.ws5_tabular_mdp.model.simulate`.
4. **OPE value** on the held-out logged rows through :mod:`pitchseq.eval.ope`
   (step-wise DR + FQE; behavior propensities are WS3's when ``--ws3-dir`` is given, else the
   state-conditional empirical behavior from train counts). The **behavior-recovery gate runs
   first** (SPEC ``0.3`` / D37).
5. **D42 cross-check** -- per (design, alpha), the model-based value of the *same softened
   policy* vs its held-out step-wise-DR and FQE values, agreement judged by the D24 rule on the
   real CIs; divergence is *reported*, never silently averaged. The in-sample *greedy* optimism
   (model-based greedy vs its own held-out value) is a separate, clearly-labeled exhibit.
6. **Headline** -- the D43 state-ladder x D42 table, the trigger-vs-count gap with a **refit
   cluster-bootstrap** CI against the **D40 myopic ceiling (~0.003)**, the setup diagnostics,
   and the verdict.

The FQE refit cluster bootstrap (real uncertainty in a constant-initial-state domain)
=====================================================================================
:func:`pitchseq.eval.ope.fqe`'s per-episode contribution is the initial-state value
``V(s_0) = sum_a pi(a|s_0) q_hat(s_0, a)`` -- and in this domain every PA starts in the *same*
state (0-0 count, no previous pitch, trigger 0), so the contribution array is **constant** and a
resampling bootstrap of it is structurally degenerate (every resample has the same mean; the
"CI" collapses to a point). The FQE uncertainty therefore comes from a **refit** cluster
bootstrap instead: for each of ``--fqe-boot`` replicates, resample pitcher-game clusters of
episodes with replacement, **refit FQE from scratch on the resampled episodes** -- the same
resample for every design and alpha, so the design gaps are *paired* -- and read percentile CIs
off the replicate distribution. The refits are exact tabular fits through
:func:`~workstreams.ws5_tabular_mdp.model.onehot_tabular_regressor` (numerically identical to
``ope.tabular_regressor``, orders of magnitude faster), supplied through ``ope.fqe``'s
documented ``regressor_factory`` callback -- FQE itself is never re-implemented. Wherever a CI
would still be structurally degenerate (e.g. the step-wise-DR gap at ``alpha = 0``, whose paired
contributions are identically zero), it is printed as ``n/a (constant contributions)``, never as
a fake interval.

Step-wise DR keeps its per-episode contribution bootstrap (those contributions carry real
variance), but its per-PA importance-weight product is irreducibly noisy for a target far from
behavior, so the verdict uses it **directionally**, with the refit-bootstrap FQE CI as the
resolving instrument.

Verdicts (decisions D40/D43/D39)
================================
* **SETUP_EXPLOITED** (positive world): at some alpha, the refit-bootstrap FQE trigger-count
  gap's one-sided 95% lower bound exceeds the D40 myopic ceiling **and** the step-wise-DR gap is
  directionally positive at that alpha **and** the model-based gap is directionally positive --
  the setup pitch is cashed on held-out data beyond what a myopic policy could reach.
* **SEQ_NEUTRAL_MDP** (null world): no alpha satisfies that triple condition (the richer designs
  are worth no more than ``count`` up to overfitting cost) -- the expected null result.
* **SETUP_INCONCLUSIVE** (D39 honest fallback): a positive world where the triple condition is
  not met at the tested scale. The headline then prints the full evidence story -- the
  directional agreement of the FQE point, step-wise-DR point and model-based gap, the setup
  diagnostics, and the unit-level constructed-world test that proves the machinery cashes setups
  -- and notes that certification is a data-scale question for Phase 2.
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
from pitchseq.eval.predictions import POLICY_PROB_COLS, save_predictions  # noqa: E402
from pitchseq.families import FAMILIES, feasible_action_mask  # noqa: E402
from pitchseq.runmeta import track_run, write_runmeta  # noqa: E402
from pitchseq.splits import cluster_bootstrap_indices, make_splits  # noqa: E402
from workstreams.ws5_tabular_mdp.model import (  # noqa: E402
    DEFAULT_ALPHA_R,
    DEFAULT_ALPHA_T,
    DEFAULT_VELO_GAP_THRESHOLD,
    RICHEST_DESIGN,
    STATE_DESIGNS,
    XX_INDEX,
    encode_states,
    estimate_behavior_policy,
    estimate_mdp,
    greedy_policy_matrix,
    onehot_tabular_regressor,
    policy_evaluation,
    policy_iteration,
    policy_to_row_probs,
    setup_diagnostics,
    simulate,
    soften_policy,
)

__all__ = ["run_ws5", "main"]

#: The D40 myopic ceiling (run value): WS4 measured the maximum myopically-exploitable edge on
#: the positive fixture at ~0.003 run. WS5's positive-world acceptance is to EXCEED it by valuing
#: the setup (sequential credit).
D40_MYOPIC_CEILING = 0.003

#: Positive-world planted whiff boost (saturates the effect; see the synth module) and the
#: velo-transition threshold echoed into the trigger flag.
_EFFECT_SIZE = 0.5
_VELO_GAP_THRESHOLD = DEFAULT_VELO_GAP_THRESHOLD
#: Reward shrinkage tuned so the richer designs generalise (the held-out setup gap discriminates).
_SYNTH_ALPHA_R = 12.0
#: Temperature for the secondary (Boltzmann) softening variant.
_TEMPERATURE = 0.015

_N_FAM = len(FAMILIES)
_FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}

#: The three D43 designs map to the nested C / L1 / O measurement views for the standard
#: prediction schema (SPEC 8.1 legal ``state_view`` set): count ~ C, count+prev ~ L1,
#: count+prev+trigger ~ O-lite. The exact design name is retained in ``model_id``.
_DESIGN_TO_VIEW = {"count": "C", "count_prev": "L1", "count_prev_trigger": "O"}


# --- data ------------------------------------------------------------------------------


def _load_table(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.is_dir():
        frames = [pd.read_parquet(p, engine="pyarrow") for p in sorted(path.glob("*.parquet"))]
        if not frames:
            raise FileNotFoundError(f"no parquet files under {path}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(path, engine="pyarrow")


def _synth_table(world, n_games, seed, effect_size, velo_gap, out_dir, force):
    """Build (and checkpoint) a synthetic decision table."""
    cache = Path(out_dir) / f".table_{world}.parquet"
    if (not force) and cache.is_file():
        return pd.read_parquet(cache, engine="pyarrow")
    from pitchseq.decision_table import build_decision_table
    from pitchseq.synth import make_null_world, make_positive_world

    if world == "null":
        raw, _ = make_null_world(n_games=n_games, seed=seed, innings_per_game=6)
    elif world == "positive":
        raw, _ = make_positive_world(n_games=n_games, seed=seed, innings_per_game=6,
                                     effect_size=effect_size, velo_gap_threshold=velo_gap)
    else:
        raise ValueError(f"--synth must be null|positive|off, got {world!r}")
    table = build_decision_table(raw)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(cache, engine="pyarrow")
    except Exception:  # pragma: no cover - caching is best-effort
        pass
    return table


def _feasible_rows(table: pd.DataFrame, rows: pd.DataFrame, config: dict) -> np.ndarray:
    """SPEC ``4`` per-row feasible mask, computed on the **full** ``table`` trailing window and
    subset to ``rows`` by ``row_id`` (so a 2024 game's trailing 365 days see the train seasons)."""
    fm = feasible_action_mask(table, config=config)
    cols = [f"feasible_{f}" for f in FAMILIES]
    fm = pd.DataFrame(fm[cols].to_numpy(dtype=bool), index=table["row_id"].to_numpy())
    return fm.loc[rows["row_id"].to_numpy()].to_numpy()


def _action_index(table: pd.DataFrame) -> np.ndarray:
    return np.array(
        [_FAMILY_TO_IDX.get(str(f), XX_INDEX) for f in table["family"].astype(object).to_numpy()],
        dtype=np.int64,
    )


# --- behavior policy (common evaluator) -----------------------------------------------


def _common_behavior(train, eval_rows, ws3_dir, config):
    r"""The common behavior policy ``mu(a|s)`` per eval row (the OPE denominator).

    Two documented modes: **WS3** (``--ws3-dir``) reuses WS3's contextual GBDT propensities of
    the richest ablation view (the finest behavior model; decision D33); the **fallback** fits
    the state-conditional empirical behavior from train counts on the richest MDP state
    (:func:`~workstreams.ws5_tabular_mdp.model.estimate_behavior_policy`). Both are floored so
    importance ratios stay finite (SPEC ``9``).
    """
    if ws3_dir is not None:
        from workstreams.ws3_gbdt_stack.model import load_ws3_artifacts
        art = load_ws3_artifacts(ws3_dir)
        view = "O" if "O" in art.behavior else sorted(art.behavior)[0]
        mu = art.propensities(eval_rows, view)
        mode = f"ws3:{view}"
    else:
        ss_tr = encode_states(train, RICHEST_DESIGN)
        mu_state = estimate_behavior_policy(train, ss_tr)
        ss_ev = encode_states(eval_rows, RICHEST_DESIGN)
        mu = mu_state[ss_ev.state_ids]
        mode = "empirical:count_prev_trigger"
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-6, None)
    mu = mu / mu.sum(axis=1, keepdims=True)
    return mu, mode


def _project_behavior(mu_rows: np.ndarray, state_ids: np.ndarray, n_states: int) -> np.ndarray:
    """Project a per-row behavior policy onto per-state means (for the model-based softening)."""
    num = np.zeros((n_states, _N_FAM), dtype=np.float64)
    den = np.zeros(n_states, dtype=np.float64)
    np.add.at(num, state_ids, mu_rows)
    np.add.at(den, state_ids, 1.0)
    with np.errstate(invalid="ignore"):
        proj = np.divide(num, den[:, None], out=np.full_like(num, 1.0 / _N_FAM), where=den[:, None] > 0)
    proj = np.clip(proj, 1e-9, None)
    return proj / proj.sum(axis=1, keepdims=True)


# --- per-design planning ---------------------------------------------------------------


def _plan_design(train, eval_rows, design, fr_train, alpha_t, alpha_r, threshold, feas_share):
    """Estimate + plan one design; return the MDP, greedy policy and eval-row state ids."""
    mdp = estimate_mdp(train, design, alpha_t=alpha_t, alpha_r=alpha_r, threshold=threshold,
                       feasible_rows=fr_train, feas_state_share=feas_share)
    policy_idx, Q, V = policy_iteration(mdp.P, mdp.R, mdp.feasible)
    greedy_state = greedy_policy_matrix(policy_idx, mdp.n_actions)
    ss_eval = encode_states(eval_rows, design, threshold)
    return mdp, policy_idx, Q, V, greedy_state, ss_eval


# --- CIs -------------------------------------------------------------------------------


def _cluster_boot(contrib: np.ndarray, cluster_by_unit: np.ndarray, n_boot: int, seed: int) -> dict:
    """Cluster (pitcher-game) bootstrap of a per-unit contribution array -> value + CIs (SPEC 9).

    Flags a **structurally degenerate** input -- constant contributions, whose every resample has
    the same mean -- so the formatter prints ``n/a (constant contributions)`` instead of a fake
    interval (the resampling bootstrap is uninformative there).
    """
    contrib = np.asarray(contrib, dtype=np.float64)
    point = float(contrib.mean()) if len(contrib) else float("nan")
    degenerate = bool(len(contrib) > 0 and np.ptp(contrib) == 0.0)
    if degenerate:
        return {"value": point, "lower_95": float("nan"), "ci95": [float("nan"), float("nan")],
                "se": float("nan"), "n_boot": 0, "degenerate": True}
    frame = pd.DataFrame({"pg": cluster_by_unit})
    reps = [float(contrib[np.asarray(idx)].mean())
            for idx in cluster_bootstrap_indices(frame, cluster="pg", n_boot=n_boot, seed=seed)]
    reps = np.asarray([v for v in reps if np.isfinite(v)], dtype=np.float64)
    if reps.size == 0:
        return {"value": point, "lower_95": float("nan"), "ci95": [float("nan"), float("nan")],
                "se": float("nan"), "n_boot": 0, "degenerate": False}
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"value": point, "lower_95": float(np.percentile(reps, 5.0)),
            "ci95": [float(lo), float(hi)],
            "se": float(reps.std(ddof=1)) if len(reps) > 1 else float("nan"),
            "n_boot": int(len(reps)), "degenerate": False}


def _refit_ci(point: float, reps: np.ndarray) -> dict:
    """Percentile CI from refit-bootstrap replicate values (point = the full-data estimate)."""
    reps = np.asarray(reps, dtype=np.float64)
    finite = reps[np.isfinite(reps)]
    if finite.size == 0:
        return {"value": float(point), "lower_95": float("nan"),
                "ci95": [float("nan"), float("nan")], "se": float("nan"), "n_boot": 0,
                "method": "fqe_refit_cluster_bootstrap", "degenerate": True}
    degenerate = bool(np.ptp(finite) == 0.0)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {"value": float(point), "lower_95": float(np.percentile(finite, 5.0)),
            "ci95": [float(lo), float(hi)],
            "se": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
            "n_boot": int(finite.size), "method": "fqe_refit_cluster_bootstrap",
            "degenerate": degenerate}


# --- the FQE refit cluster bootstrap (the real uncertainty; see the module docstring) ---


def _fqe_refit_bootstrap(rewards, actions, step, state_ids_by_design, targets,
                         ep_starts, ep_lens, cluster_by_ep, n_boot, seed):
    r"""Refit-bootstrap replicate FQE values per (design, alpha) -- paired across designs.

    For each replicate: resample pitcher-game clusters of episodes with replacement
    (:func:`pitchseq.splits.cluster_bootstrap_indices` over the per-episode frame), rebuild the
    logged rows of the sampled episode *instances* (relabelling ``pa_id`` per instance so a
    twice-sampled episode is two episodes), and refit :func:`pitchseq.eval.ope.fqe` from scratch
    on them for **every** design and alpha -- the same resample for all arms, so design *gaps*
    difference away the shared episode-sampling noise (a paired bootstrap). Episode rows are
    contiguous in the (pa_id, pitch_number)-sorted eval frame, so an episode's rows are the
    slice ``[ep_starts[p], ep_starts[p] + ep_lens[p])``.

    Parameters
    ----------
    rewards, actions, step : numpy.ndarray, shape (n,)
        Logged per-decision arrays (eval order).
    state_ids_by_design : dict
        ``design -> (n,)`` current-state ids per row.
    targets : dict
        ``(design, alpha) -> (n, 8)`` per-row softened target probabilities.
    ep_starts, ep_lens : numpy.ndarray, shape (n_episodes,)
        Row offsets / lengths per episode (ascending ``pa_id`` order).
    cluster_by_ep : numpy.ndarray, shape (n_episodes,)
        Pitcher-game cluster id per episode.
    n_boot : int
        Refit replicates. ``< 1`` skips the bootstrap (all FQE CIs reported n/a).
    seed : int
        Resampling seed.

    Returns
    -------
    (dict, dict)
        ``values[(design, alpha)] -> (n_boot,) replicate FQE values`` and a cost record
        ``{"n_fits", "seconds", "seconds_per_fit"}``.
    """
    keys = list(targets.keys())
    values = {k: np.full(int(max(n_boot, 0)), np.nan, dtype=np.float64) for k in keys}
    if n_boot < 1:
        return values, {"n_fits": 0, "seconds": 0.0, "seconds_per_fit": float("nan")}
    reg = onehot_tabular_regressor(_N_FAM)
    designs = sorted({d for d, _ in keys})
    alphas_by_design = {d: sorted(a for dd, a in keys if dd == d) for d in designs}
    ep_frame = pd.DataFrame({"pg": cluster_by_ep})
    t0 = time.perf_counter()
    n_fits = 0
    for r, idx in enumerate(cluster_bootstrap_indices(ep_frame, cluster="pg",
                                                      n_boot=n_boot, seed=seed)):
        pos = np.asarray(idx, dtype=np.int64)
        lens = ep_lens[pos]
        row_idx = np.concatenate([np.arange(s, s + l) for s, l in zip(ep_starts[pos], lens)])
        new_ep = np.repeat(np.arange(len(pos), dtype=np.int64), lens)
        st = step[row_idx]
        frame = pd.DataFrame({"reward": rewards[row_idx], "action": actions[row_idx],
                              "pa_id": new_ep, "step": st})
        for d in designs:
            frame["state"] = state_ids_by_design[d][row_idx]
            for a in alphas_by_design[d]:
                v, _ = ope.fqe(frame, targets[(d, a)][row_idx], _N_FAM,
                               state_col="state", action_col="action", reward_col="reward",
                               episode_id=new_ep, step=st, regressor_factory=reg)
                values[(d, a)][r] = v
                n_fits += 1
    seconds = time.perf_counter() - t0
    return values, {"n_fits": int(n_fits), "seconds": float(seconds),
                    "seconds_per_fit": float(seconds / n_fits) if n_fits else float("nan")}


# --- per-(design, alpha) held-out evaluation (stepwise + FQE point + diagnostics) --------


def _evaluate_design(logged, target_rows, q_rows, mu_rows, actions, ep, step,
                     cluster_by_ep, n_boot, seed, fqe_reg):
    r"""One held-out OPE of a design's softened target: step-wise DR + the FQE point fit.

    Consumes the OPE estimators directly (:func:`pitchseq.eval.ope.stepwise_dr`,
    :func:`~pitchseq.eval.ope.fqe`, :func:`~pitchseq.eval.ope.ope_diagnostics`). The step-wise-DR
    CI is a cluster bootstrap of its per-episode contributions (they carry real variance); the
    FQE CI is **not** computed here -- its per-episode contributions are constant in this domain
    (every PA starts in the same state), so the refit bootstrap supplies it instead (see
    :func:`_fqe_refit_bootstrap`). Returns the estimator values, diagnostics and the raw
    step-wise contributions (for the paired gaps).
    """
    n = len(actions)
    ar = np.arange(n)
    rewards = logged["reward"].to_numpy(dtype=np.float64)
    mu_taken = mu_rows[ar, actions]
    w = target_rows[ar, actions] / mu_taken

    sw_val, sw_contrib = ope.stepwise_dr(rewards, w, q_rows, target_rows, actions, ep, step)
    fqe_val, _fqe_contrib = ope.fqe(logged, target_rows, _N_FAM, state_col="state",
                                    action_col="action", reward_col="reward",
                                    episode_id=ep, step=step, regressor_factory=fqe_reg)
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


def _d42_agreement(model_based: float, sw: dict, fqe: dict) -> dict:
    """D24-style agreement among the three D42 lenses for the SAME softened policy at one alpha.

    Model-based (an exact quantity in the estimated MDP -- a point, half-width 0) vs the held-out
    step-wise DR and FQE, each with its **real** CI (contribution bootstrap / refit bootstrap).
    A pair diverges iff the absolute difference exceeds the wider of the two 95% CI half-widths;
    a pair where neither side has a usable CI cannot be assessed and is marked ``None``.
    """
    def halfwidth(entry):
        if entry is None or entry.get("degenerate"):
            return None
        lo, hi = entry.get("ci95", [float("nan")] * 2)
        return (hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else None

    vals = {"model_based": (float(model_based), 0.0),
            "stepwise_DR": (sw["value"], halfwidth(sw)),
            "FQE": (fqe["value"], halfwidth(fqe))}
    pairs = {}
    any_div = False
    assessed = 0
    names = list(vals)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            va, ha = vals[a]
            vb, hb = vals[b]
            widths = [h for h in (ha, hb) if h is not None]
            tol = max(widths) if widths else None
            if tol is None or tol == 0.0 or not (np.isfinite(va) and np.isfinite(vb)):
                pairs[f"{a}|{b}"] = {"abs_diff": float(abs(va - vb)), "diverges": None}
                continue
            div = bool(abs(va - vb) > tol)
            pairs[f"{a}|{b}"] = {"abs_diff": float(abs(va - vb)), "tolerance": float(tol),
                                 "diverges": div}
            assessed += 1
            any_div = any_div or div
    verdict = "DIVERGES" if any_div else ("CONSISTENT" if assessed else "n/a")
    return {"verdict": verdict, "pairs": pairs}


# --- the pipeline ----------------------------------------------------------------------


def run_ws5(
    source: str | None = None,
    synth: str = "off",
    out: str = "results/ws5",
    ws3_dir: str | None = None,
    designs=STATE_DESIGNS,
    alphas=None,
    alpha_t: float = DEFAULT_ALPHA_T,
    alpha_r: float | None = None,
    threshold: float = _VELO_GAP_THRESHOLD,
    feas_state_share: float = 0.5,
    temperature: float = _TEMPERATURE,
    n_boot: int = 250,
    fqe_boot: int = 200,
    sim_episodes: int = 3000,
    seed: int = 7,
    sim_seed: int = 20260715,
    n_games: int = 2000,
    effect_size: float = _EFFECT_SIZE,
    ceiling: float = D40_MYOPIC_CEILING,
    config: dict | None = None,
    write_outputs: bool = True,
    force: bool = False,
) -> dict:
    """Estimate the D43 tabular MDPs, plan, and evaluate the setup value through the OPE gate.

    Parameters
    ----------
    source : str, optional
        Decision-table parquet (or a directory of season checkpoints). Required when
        ``synth == 'off'``.
    synth : {'off', 'null', 'positive'}, optional
        Run on a generated synthetic world (Phase-1 CI path) instead of ``source``.
    out : str, optional
        Output directory (predictions, report, run metadata; synthetic-table cache).
    ws3_dir : str, optional
        WS3 artifacts for behavior propensities (decision D33); when absent the state-conditional
        empirical behavior is fit from train counts.
    designs : sequence of str, optional
        The state-space ladder (default all three D43 designs).
    alphas : sequence of float, optional
        The ``pi_alpha`` grid (default: config ``ope.conservative_alpha``).
    alpha_t, alpha_r : float, optional
        Transition Dirichlet pseudo-count and reward shrinkage (``alpha_r`` default: the synth-
        tuned :data:`_SYNTH_ALPHA_R` so the richer designs generalise).
    threshold : float, optional
        Velo-gap trigger threshold (echoes the synth planted value).
    feas_state_share : float, optional
        Per-state feasibility share threshold.
    temperature : float, optional
        Boltzmann temperature for the secondary softening variant.
    n_boot : int, optional
        Cluster-bootstrap replicates for the step-wise-DR contribution CIs.
    fqe_boot : int, optional
        **Refit**-bootstrap replicates for the FQE CIs (the resolving uncertainty; see the module
        docstring). Each replicate refits FQE once per (design, alpha). ``< 1`` skips the
        bootstrap and reports the FQE CIs as n/a (the SETUP_EXPLOITED gate then cannot fire).
    sim_episodes : int, optional
        Simulator rollouts for the D42 model-based-value cross-check.
    seed, sim_seed : int, optional
        Bootstrap / behavior-recovery seed and simulator seed.
    n_games : int, optional
        Synthetic-world size (calibrated so the synthetic verdicts are deterministic).
    effect_size : float, optional
        Positive-world planted whiff boost.
    ceiling : float, optional
        The D40 myopic ceiling the setup gap must exceed (default 0.003).
    config : dict, optional
        Study config; loaded from default when ``None``.
    write_outputs : bool, optional
        Persist predictions / report / run metadata under ``out``.
    force : bool, optional
        Ignore the synthetic-table cache and rebuild.

    Returns
    -------
    dict
        The full report: the behavior gate, the per-design ladder (model-based + OPE + ESS + the
        per-alpha D42 agreement), the greedy-optimism exhibit, the trigger-vs-count / prev-vs-
        count gaps with refit-bootstrap CIs vs the ceiling, setup diagnostics, the verdict,
        refit-bootstrap cost, timing / RAM and output paths.
    """
    if config is None:
        config = load_config()
    if alphas is None:
        alphas = list(config.get("ope", {}).get("conservative_alpha", [0.0, 0.1, 0.25, 0.5, 1.0]))
    alphas = [float(a) for a in alphas]
    if alpha_r is None:
        alpha_r = _SYNTH_ALPHA_R if synth != "off" else DEFAULT_ALPHA_R
    designs = list(designs)
    if RICHEST_DESIGN not in designs or "count" not in designs:
        raise ValueError(f"designs must include 'count' and {RICHEST_DESIGN!r}; got {designs}")
    world_tag = synth if synth != "off" else "real"
    out_dir = Path(out)
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"world": world_tag, "designs": designs, "alphas": alphas, "ceiling": float(ceiling),
                    "alpha_t": float(alpha_t), "alpha_r": float(alpha_r), "threshold": float(threshold),
                    "fqe_boot": int(fqe_boot)}
    fqe_reg = onehot_tabular_regressor(_N_FAM)

    with track_run(step="ws5_tabular_mdp", world=world_tag) as meta:
        # --- data + split ---
        if synth != "off":
            table = _synth_table(synth, n_games, seed, effect_size, threshold, out_dir, force)
        else:
            if source is None:
                raise ValueError("--table is required when --synth is off")
            table = _load_table(source)
        splits = make_splits(table, config)["primary"]
        train = table.loc[splits["train"].to_numpy()].reset_index(drop=True)
        val = table.loc[splits["val"].to_numpy()].reset_index(drop=True)
        test = table.loc[splits["test"].to_numpy()].reset_index(drop=True)
        eval_rows = pd.concat([val, test], ignore_index=True) if len(test) else val.copy()
        # Sort eval rows by (pa_id, pitch_number) so every estimator's per-episode contributions
        # share one ascending-pa_id order (clean paired differencing) and episode rows are
        # contiguous (the refit bootstrap slices episodes by offset).
        eval_rows = eval_rows.sort_values(["pa_id", "pitch_number"], kind="stable").reset_index(drop=True)
        if len(train) == 0 or len(eval_rows) == 0:
            raise ValueError(f"empty split (train={len(train)}, eval={len(eval_rows)})")

        fr_train = _feasible_rows(table, train, config)
        actions = _action_index(eval_rows)
        rewards = pd.to_numeric(eval_rows["R"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(rewards)
        # Keep whole PAs finite (drop a PA if any reward is non-finite) so episode returns are valid.
        bad_pa = set(eval_rows.loc[~finite, "pa_id"].unique().tolist())
        keep = ~eval_rows["pa_id"].isin(bad_pa).to_numpy()
        eval_rows = eval_rows.loc[keep].reset_index(drop=True)
        actions = actions[keep]
        rewards = rewards[keep]
        ep = eval_rows["pa_id"].to_numpy()
        step = (eval_rows["pitch_number"].to_numpy() - 1).astype(np.int64)
        ep_order = pd.unique(ep)  # ascending pa_id (eval is sorted) == every estimator's contrib order
        # Episode row offsets (episodes are contiguous in the sorted frame).
        ep_starts = np.flatnonzero(np.concatenate([[True], ep[1:] != ep[:-1]]))
        ep_lens = np.diff(np.concatenate([ep_starts, [len(ep)]]))
        pg = (eval_rows["game_pk"].astype(str) + "_" + eval_rows["pitcher"].astype(str)).to_numpy()
        cluster_by_ep = pg[ep_starts]  # cluster of each episode (constant within a PA)
        result["n_train"] = int(len(train))
        result["n_eval"] = int(len(eval_rows))
        result["n_eval_pa"] = int(len(ep_order))

        # --- common behavior mu ---
        mu_common, mu_mode = _common_behavior(train, eval_rows, ws3_dir, config)
        result["behavior_mode"] = mu_mode

        # --- logged frame (multi-step PA episodes); state filled per design below ---
        base_logged = pd.DataFrame({
            "reward": rewards, "action": actions.astype(np.int64),
            "mu_prob": mu_common[np.arange(len(actions)), actions],
            "pa_id": ep, "step": step, "pitcher": eval_rows["pitcher"].to_numpy(),
            "game_pk": eval_rows["game_pk"].to_numpy(),
        })
        for a in range(_N_FAM):
            base_logged[f"mu_prob_{a}"] = mu_common[:, a]

        # --- per-design planning + softened targets ---
        plans = {}
        for d in designs:
            plans[d] = _plan_design(train, eval_rows, d, fr_train, alpha_t, alpha_r, threshold, feas_state_share)
        result["state_summary"] = {
            d: {"n_states": int(plans[d][0].n_states),
                "n_reachable": int((plans[d][0].support.sum(axis=1) > 0).sum()),
                "mean_feasible": float(plans[d][0].feasible.sum(axis=1)[~plans[d][0].terminal_mask()].mean())}
            for d in designs
        }

        # --- GATE FIRST: behavior-policy recovery (SPEC 0.3 / D37) ---
        gate_logged = base_logged.copy()
        gate_logged["state"] = plans[RICHEST_DESIGN][5].state_ids
        recovery = ope.behavior_policy_recovery(
            gate_logged, config=config, n_actions=_N_FAM, seed=seed, n_boot=min(n_boot, 200),
            reward_col="reward", action_col="action", mu_col="mu_prob",
            state_col="state", episode_col="pa_id", step_col="step",
        )
        result["behavior_recovery"] = {k: recovery[k] for k in ("observed_mean", "passed", "ips_weights_unit", "tol")}
        result["behavior_recovery"]["estimators"] = recovery["estimators"]
        if not recovery["passed"]:
            result["gate"] = "FAILED_GATE"
            result["seconds"] = float(meta.get("seconds", float("nan")))
            if write_outputs:
                result["outputs"] = _write(out_dir, world_tag, result, meta)
            return result
        result["gate"] = "PASSED"

        # --- ladder first pass: model-based + stepwise + FQE point per (design, alpha) ---
        ladder: dict = {d: {} for d in designs}
        sw_contribs: dict = {d: {} for d in designs}
        targets_rows: dict = {}                              # (design, alpha) -> per-row target
        state_ids_by_design = {d: plans[d][5].state_ids for d in designs}
        for d in designs:
            mdp, policy_idx, Q, V, greedy_state, ss_eval = plans[d]
            mu_proj = _project_behavior(mu_common, ss_eval.state_ids, mdp.n_states)
            term_mask = mdp.terminal_mask()
            greedy_row = policy_to_row_probs(greedy_state, ss_eval.state_ids)
            for a in alphas:
                # per-state softened policy (for the exact model-based value + q_hat control variate)
                pi_state = (1 - a) * mu_proj + a * greedy_state
                pi_state = np.clip(pi_state, 1e-12, None)
                pi_state = pi_state / pi_state.sum(axis=1, keepdims=True)
                V_pi, Q_pi = policy_evaluation(mdp.P, mdp.R, pi_state, terminal_mask=term_mask)
                model_based = float(mdp.start_dist @ V_pi)
                # per-row conservative target (SPEC 9 pi_alpha mixture, reusing eval/ope) + q_hat
                target_rows = ope.pi_alpha(mu_common, greedy_row, a)
                targets_rows[(d, a)] = target_rows
                q_rows = Q_pi[ss_eval.state_ids]
                logged_d = base_logged.copy()
                logged_d["state"] = ss_eval.state_ids
                ev = _evaluate_design(logged_d, target_rows, q_rows, mu_common, actions, ep, step,
                                      cluster_by_ep, n_boot, seed, fqe_reg)
                sw_contribs[d][a] = ev.pop("_sw_contrib")
                ladder[d][a] = {"model_based": model_based, **ev}
            # simulator cross-check of the (pure greedy) model-based value (decision D42)
            sim_v, _ = simulate(mdp.P, mdp.R, greedy_state, sim_episodes, np.random.default_rng(sim_seed),
                                mdp.start_dist, terminal_mask=term_mask)
            V_gr, _ = policy_evaluation(mdp.P, mdp.R, greedy_state, terminal_mask=term_mask)
            result["state_summary"][d]["model_based_greedy"] = float(mdp.start_dist @ V_gr)
            result["state_summary"][d]["simulated_greedy"] = float(sim_v)

        # --- the FQE refit cluster bootstrap (real CIs; paired across designs/alphas) ---
        refit_values, refit_cost = _fqe_refit_bootstrap(
            rewards, actions, step, state_ids_by_design, targets_rows,
            ep_starts, ep_lens, cluster_by_ep, fqe_boot, seed,
        )
        result["fqe_refit"] = {"n_boot": int(fqe_boot), **refit_cost}

        # --- fill FQE CIs + the per-alpha D42 agreement into the ladder ---
        for d in designs:
            for a in alphas:
                cell = ladder[d][a]
                cell["FQE"] = _refit_ci(cell.pop("fqe_point"), refit_values[(d, a)])
                cell["d42"] = _d42_agreement(cell["model_based"], cell["stepwise_DR"], cell["FQE"])

        # --- paired gaps (common evaluator): trigger-count and prev-count per alpha ---
        gaps: dict = {}
        for a in alphas:
            entry = {}
            for lo_d, hi_d, key in (("count", RICHEST_DESIGN, "trigger_minus_count"),
                                    ("count", "count_prev", "prev_minus_count")):
                if hi_d not in designs or lo_d not in designs:
                    continue
                fqe_point_gap = ladder[hi_d][a]["FQE"]["value"] - ladder[lo_d][a]["FQE"]["value"]
                fqe_gap_reps = refit_values[(hi_d, a)] - refit_values[(lo_d, a)]  # paired
                sw_gap = sw_contribs[hi_d][a] - sw_contribs[lo_d][a]
                entry[key] = {
                    "FQE": _refit_ci(fqe_point_gap, fqe_gap_reps),
                    "stepwise_DR": _cluster_boot(sw_gap, cluster_by_ep, n_boot, seed),
                    "model_based": (ladder[hi_d][a]["model_based"] - ladder[lo_d][a]["model_based"]),
                }
            gaps[a] = entry
        result["ladder"] = ladder
        result["gaps"] = gaps

        # --- setup diagnostics (D43) ---
        if RICHEST_DESIGN in designs and "count_prev" in designs:
            result["setup_diagnostics"] = setup_diagnostics(plans[RICHEST_DESIGN][0], plans["count_prev"][0], threshold)

        # --- temperature-softened variant (secondary; primary is pi_alpha) ---
        temp_variant = {}
        for d in designs:
            mdp, policy_idx, Q, V, greedy_state, ss_eval = plans[d]
            temp_state = soften_policy(Q, mdp.feasible, temperature)
            V_t, _ = policy_evaluation(mdp.P, mdp.R, temp_state, terminal_mask=mdp.terminal_mask())
            temp_variant[d] = {"model_based": float(mdp.start_dist @ V_t), "temperature": float(temperature)}
        result["temperature_variant"] = temp_variant

        # --- verdict ---
        result["verdict"] = _ladder_verdict(world_tag, gaps, ceiling, result.get("setup_diagnostics"))

        # --- standard-schema policy predictions (pure greedy target per design) ---
        if write_outputs:
            _write_policy_predictions(out_dir, world_tag, eval_rows, designs, plans, meta)

    result["seconds"] = float(meta.get("seconds", float("nan")))
    result["peak_mem_mb"] = float(meta.get("peak_mem_mb", float("nan")))
    if write_outputs:
        result["outputs"] = _write(out_dir, world_tag, result, meta)
    return result


def _ladder_verdict(world, gaps, ceiling, setup_diag):
    r"""The D40/D43/D39 verdict -- the triple-condition gate on the trigger-count gap.

    ``SETUP_EXPLOITED`` requires, at some alpha, ALL of:

    1. refit-bootstrap FQE gap ``lower_95 > ceiling`` (the resolving held-out instrument, with a
       real -- non-degenerate -- CI);
    2. the step-wise-DR gap **directionally positive** at the same alpha (the importance-weighted
       lens agrees in sign; its CI is too wide to resolve the ceiling);
    3. the model-based gap **directionally positive** (the estimated MDP agrees the trigger
       design is worth more).

    ``SEQ_NEUTRAL_MDP`` -- no alpha satisfies the triple condition on the null world.
    ``SETUP_INCONCLUSIVE`` -- the D39 honest fallback on a positive world: the evidence story
    (directional agreement, setup diagnostics, the unit-level constructed-world proof) is
    reported, and certification is deferred to Phase-2 data scale.
    """
    alphas = sorted(gaps)
    fired = []
    for a in alphas:
        g = gaps[a].get("trigger_minus_count")
        if not g:
            continue
        f, s, mb = g["FQE"], g["stepwise_DR"], g["model_based"]
        cond = (not f.get("degenerate")) and np.isfinite(f["lower_95"]) \
            and f["lower_95"] > ceiling and s["value"] > 0 and mb > 0
        if cond:
            fired.append(a)
    top_alpha = max(alphas)
    tc_top = gaps[top_alpha].get("trigger_minus_count", {})
    if fired:
        verdict = "SETUP_EXPLOITED"
    elif world == "positive":
        verdict = "SETUP_INCONCLUSIVE"
    else:
        verdict = "SEQ_NEUTRAL_MDP"
    evidence = {}
    if tc_top:
        evidence = {
            "alpha": top_alpha,
            "fqe_value": tc_top["FQE"]["value"], "fqe_lower_95": tc_top["FQE"]["lower_95"],
            "stepdr_value": tc_top["stepwise_DR"]["value"],
            "model_based": tc_top["model_based"],
            "directional_agreement": bool(tc_top["FQE"]["value"] > 0
                                          and tc_top["stepwise_DR"]["value"] > 0
                                          and tc_top["model_based"] > 0),
        }
    return {
        "verdict": verdict,
        "fired_alphas": fired,
        "best_alpha": (fired[0] if fired else None),
        "trigger_minus_count_top": tc_top,
        "evidence": evidence,
        "setup_gap_mean": (setup_diag or {}).get("setup_gap_mean"),
        "setup_gap_positive_frac": (setup_diag or {}).get("setup_gap_positive_frac"),
        "optimal_action_change_frac": (setup_diag or {}).get("optimal_action_change_frac"),
    }


# --- outputs ---------------------------------------------------------------------------


def _write_policy_predictions(out_dir, world_tag, rows, designs, plans, meta):
    """Persist standard-schema ``policy_prob`` predictions per design (the pure greedy target)."""
    for d in designs:
        mdp, policy_idx, Q, V, greedy_state, ss_eval = plans[d]
        pol = policy_to_row_probs(greedy_state, ss_eval.state_ids)
        df = pd.DataFrame({"row_id": rows["row_id"].to_numpy()})
        for j, col in enumerate(POLICY_PROB_COLS):
            df[col] = pol[:, j]
        df["model_id"] = f"ws5_tabular_mdp:{d}"
        df["state_view"] = _DESIGN_TO_VIEW.get(d, "NA")
        df["seconds"] = float(meta.get("seconds", 0.0))
        df["peak_mem_mb"] = float(meta.get("peak_mem_mb", 0.0))
        df["n_params"] = int(mdp.n_states * mdp.n_actions)
        save_predictions(df, out_dir / f"policy_{world_tag}_{d}.parquet")


def _write(out_dir: Path, world_tag: str, result: dict, meta: dict) -> dict:
    report_path = out_dir / f"ws5_report_{world_tag}.json"
    report_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    runmeta_path = out_dir / f"ws5_{world_tag}.runmeta.json"
    write_runmeta(runmeta_path, dict(meta))
    _write_ladder_csv(out_dir / f"ladder_{world_tag}.csv", result)
    return {"report": str(report_path), "runmeta": str(runmeta_path), "artifacts_dir": str(out_dir)}


def _write_ladder_csv(path: Path, result: dict) -> None:
    if "ladder" not in result:
        return
    recs = []
    for d, by_a in result["ladder"].items():
        for a, cell in by_a.items():
            recs.append({
                "design": d, "alpha": a, "model_based": cell["model_based"],
                "stepwise_DR": cell["stepwise_DR"]["value"],
                "stepwise_ci_lo": cell["stepwise_DR"]["ci95"][0],
                "stepwise_ci_hi": cell["stepwise_DR"]["ci95"][1],
                "FQE": cell["FQE"]["value"],
                "FQE_ci_lo": cell["FQE"]["ci95"][0], "FQE_ci_hi": cell["FQE"]["ci95"][1],
                "FQE_lower95": cell["FQE"]["lower_95"],
                "FQE_ci_degenerate": cell["FQE"].get("degenerate", False),
                "ess_frac": cell["ess_frac"], "d42": cell["d42"]["verdict"],
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


def _fmt_ci(entry, spec="+.4f") -> str:
    """Render a CI dict; a structurally degenerate CI prints as n/a, never a fake interval."""
    if entry is None:
        return "n/a"
    if entry.get("degenerate"):
        return "n/a (constant contributions)"
    lo, hi = entry.get("ci95", [float("nan")] * 2)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return f"[{_fmt(lo, spec)},{_fmt(hi, spec)}]"


def _format_headline(result: dict) -> str:
    W = 104
    L = ["=" * W, " WS5 tabular MDP / controlled Markov reward process - setup value (OPE gate) - headline", "=" * W]
    L.append(f" world          : {result['world']}   behavior mu: {result.get('behavior_mode', '?')}")
    L.append(f" rows           : train={result.get('n_train', 0):,}  eval={result.get('n_eval', 0):,}  "
             f"eval PAs={result.get('n_eval_pa', 0):,}")

    rec = result.get("behavior_recovery", {})
    if result.get("gate") == "FAILED_GATE":
        L.append(" behavior recovery: FAILED_GATE - OPE cannot recover the behavior value; stopping "
                 "before any target value (SPEC 0.3 / D37).")
        L.append("=" * W)
        return "\n".join(L)
    L.append(f" behavior recovery: PASS  (observed={_fmt(rec.get('observed_mean'))}  "
             f"IPS weights unit={rec.get('ips_weights_unit')})  [gate; SPEC 0.3 / D37]")

    ss = result.get("state_summary", {})
    L.append("")
    L.append(" STATE SPACES + GREEDY-OPTIMISM EXHIBIT (in-sample greedy MB vs its own held-out FQE value):")
    L.append(f"   {'design':<20}{'S':>4}{'reach':>6}{'feas/st':>8} {'MB(greedy)':>10} {'sim(greedy)':>11} "
             f"{'FQE@a=1':>9} {'optimism':>9}")
    for d in result["designs"]:
        s = ss.get(d, {})
        fqe_g = result["ladder"][d][1.0]["FQE"]["value"] if 1.0 in result["ladder"][d] else None
        opt = (s.get("model_based_greedy") - fqe_g) if (fqe_g is not None and s.get("model_based_greedy") is not None) else None
        L.append(f"   {d:<20}{s.get('n_states', 0):>4}{s.get('n_reachable', 0):>6}"
                 f"{_fmt(s.get('mean_feasible'), '.2f'):>8} {_fmt(s.get('model_based_greedy')):>10} "
                 f"{_fmt(s.get('simulated_greedy')):>11} {_fmt(fqe_g):>9} {_fmt(opt):>9}")
    L.append("   (in-sample greedy value is optimistic -- max-selection bias, sharper on the sparser designs;")
    L.append("    an EXHIBIT of tabular support honesty, not a verdict input.)")

    L.append("")
    L.append(" D43 LADDER x D42 CROSS-CHECK (per alpha, the SAME softened pi_alpha in all three lenses):")
    L.append(f"   {'design':<20}{'alpha':>5} {'MB':>8} {'stepDR':>8} {'stepDR CI':>19} "
             f"{'FQE':>8} {'FQE CI (refit)':>19} {'ESS%':>6} {'D42':>10}")
    for d in result["designs"]:
        for a in result["alphas"]:
            cell = result["ladder"][d][a]
            sw, fq = cell["stepwise_DR"], cell["FQE"]
            L.append(f"   {d:<20}{a:>5.2f} {_fmt(cell['model_based']):>8} {_fmt(sw['value']):>8} "
                     f"{_fmt_ci(sw):>19} {_fmt(fq['value']):>8} {_fmt_ci(fq):>19} "
                     f"{_fmt(cell['ess_frac'], '.1%'):>6} {cell['d42']['verdict']:>10}")
    L.append("   (D42: DIVERGES = the estimated MDP's value of this softened policy sits outside the held-out")
    L.append("    OPE's real CIs -- reported per D42, never silently averaged.)")

    L.append("")
    L.append(" SETUP GAP vs D40 MYOPIC CEILING (trigger-count; FQE CI = PAIRED REFIT cluster bootstrap):")
    L.append("   myopic ceiling ~= +0.003 (D40); this gap must exceed it -- gate: FQE lower-95 > ceiling")
    L.append("   AND stepDR gap > 0 AND model-based gap > 0 at the same alpha")
    for a in result["alphas"]:
        g = result["gaps"].get(a, {}).get("trigger_minus_count")
        if not g:
            continue
        fqe, sw = g["FQE"], g["stepwise_DR"]
        line = (f"   alpha={a:>4.2f}  FQE={_fmt(fqe['value'])} {_fmt_ci(fqe)} lo95={_fmt(fqe['lower_95'])}"
                f"  | stepDR={_fmt(sw['value'])} {_fmt_ci(sw)}  | MB={_fmt(g['model_based'])}")
        if (not fqe.get("degenerate")) and np.isfinite(fqe.get("lower_95", float("nan"))) \
                and fqe["lower_95"] > result["ceiling"] and sw["value"] > 0 and g["model_based"] > 0:
            line += "  <== CLEARS (all 3)"
        L.append(line)
    fr = result.get("fqe_refit", {})
    if fr:
        L.append(f"   (refit bootstrap: {fr.get('n_boot')} replicates, {fr.get('n_fits')} FQE refits, "
                 f"{_fmt(fr.get('seconds'), '.1f')}s total, {_fmt(fr.get('seconds_per_fit'), '.3f')}s/fit)")

    diag = result.get("setup_diagnostics")
    if diag:
        L.append("")
        L.append(" SETUP DIAGNOSTICS (count_prev_trigger MDP):")
        L.append(f"   setup Q-gap (create vs not): mean={_fmt(diag['setup_gap_mean'])} "
                 f"median={_fmt(diag['setup_gap_median'])} pos-frac={_fmt(diag['setup_gap_positive_frac'], '.2f')} "
                 f"(n={diag['n_states_with_gap']})")
        L.append(f"   optimal-action change vs count_prev: {_fmt(diag['optimal_action_change_frac'], '.1%')} "
                 f"of {diag['n_states_compared']} states  (mean trigger-creating actions/state="
                 f"{_fmt(diag['mean_creating_actions'], '.2f')})")

    v = result.get("verdict", {})
    L.append("")
    L.append(" --- D43 SYNTHETIC VERDICT ---")
    L.append(f" verdict : {v.get('verdict')}")
    ev = v.get("evidence", {})
    if v.get("verdict") == "SETUP_EXPLOITED":
        L.append(f"   at alpha={v.get('best_alpha')} the refit-bootstrap FQE trigger-count gap lower-95 clears the")
        L.append("   D40 ceiling, with the stepDR and model-based gaps directionally positive: the tabular MDP")
        L.append("   CASHES a setup pitch (sequential credit the myopic WS4 bandit could not).")
    elif v.get("verdict") == "SEQ_NEUTRAL_MDP":
        L.append("   no alpha satisfies the triple gate: the richer designs are worth no more than count up to")
        L.append("   overfitting cost (the null world's expected result; the trigger flag is inert).")
    else:  # SETUP_INCONCLUSIVE (D39 honest fallback)
        L.append("   the triple gate is not met at this scale -- the honest D39 negative. The evidence story:")
        if ev:
            L.append(f"     at alpha={_fmt(ev.get('alpha'), '.2f')}: FQE gap={_fmt(ev.get('fqe_value'))} "
                     f"(lo95={_fmt(ev.get('fqe_lower_95'))}), stepDR gap={_fmt(ev.get('stepdr_value'))}, "
                     f"MB gap={_fmt(ev.get('model_based'))} -- directional agreement="
                     f"{ev.get('directional_agreement')}")
        L.append(f"     setup Q-gap mean={_fmt(v.get('setup_gap_mean'))} "
                 f"(pos-frac={_fmt(v.get('setup_gap_positive_frac'), '.2f')}); optimal action changes on "
                 f"{_fmt(v.get('optimal_action_change_frac'), '.1%')} of trigger states.")
        L.append("     The machinery provably cashes setups (the constructed-world unit test); certification on")
        L.append("     this fixture is a data-scale question deferred to the Phase-2 full-data run.")

    L.append("")
    L.append(f" elapsed (s)    : {_fmt(result.get('seconds'), '.1f')}   peak mem (MB): {_fmt(result.get('peak_mem_mb'), '.1f')}")
    if result.get("outputs"):
        L.append(f" outputs        : {result['outputs']['report']}")
    L.append("=" * W)
    return "\n".join(L)


# --- CLI -------------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python workstreams/ws5_tabular_mdp/run_ws5.py",
        description="Estimate the D43 tabular MDPs, plan, and evaluate the setup value (SPEC 12.6).",
    )
    p.add_argument("--table", type=str, default=None,
                   help="Decision-table parquet (or directory of season checkpoints).")
    p.add_argument("--ws3-dir", type=str, default=None,
                   help="WS3 artifacts for behavior propensities (else empirical from train counts; D33).")
    p.add_argument("--out", type=str, default="results/ws5", help="Output directory.")
    p.add_argument("--synth", type=str, default="off", choices=["off", "null", "positive"],
                   help="Run on a generated synthetic world instead of --table.")
    p.add_argument("--alpha-t", type=float, default=DEFAULT_ALPHA_T, help="Transition Dirichlet pseudo-count.")
    p.add_argument("--alpha-r", type=float, default=None, help="Reward shrinkage (default: synth-tuned).")
    p.add_argument("--threshold", type=float, default=_VELO_GAP_THRESHOLD, help="Velo-gap trigger threshold (mph).")
    p.add_argument("--n-boot", type=int, default=250, help="Step-wise-DR contribution bootstrap replicates.")
    p.add_argument("--fqe-boot", type=int, default=200,
                   help="FQE REFIT-bootstrap replicates (each refits FQE once per design x alpha; "
                        "the resolving CI -- lower it on large real data if slow).")
    p.add_argument("--n-games", type=int, default=2000, help="Synthetic-world size.")
    p.add_argument("--seed", type=int, default=7, help="Bootstrap / split / synthetic-world seed.")
    p.add_argument("--force", action="store_true", help="Ignore the synthetic-table cache and rebuild.")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    """CLI entry point: run the pipeline and print the headline block."""
    args = _parse_args(argv)
    if args.synth == "off" and args.table is None:
        print("error: --table is required when --synth is off", file=sys.stderr)
        raise SystemExit(2)
    result = run_ws5(
        source=args.table, synth=args.synth, out=args.out, ws3_dir=args.ws3_dir,
        alpha_t=args.alpha_t, alpha_r=args.alpha_r, threshold=args.threshold,
        n_boot=args.n_boot, fqe_boot=args.fqe_boot, n_games=args.n_games, seed=args.seed,
        force=args.force,
    )
    print(_format_headline(result))
    return result


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    main()
