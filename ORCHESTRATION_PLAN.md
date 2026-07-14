# Orchestration Plan

Running log for the pitch-sequencing comparative study. One implementation unit is
dispatched at a time, reviewed against SPEC.md, then committed. This file records the
plan, every dispatch, the review verdict, and design decisions made along the way.

## Mission recap

Build the rigor ladder described in SPEC.md and README.md: a shared foundation
(decision table, five nested state views C/U/L1/O/OM, reward, temporal splits,
evaluation harness, OPE) plus seven workstreams, each delivering code, a
research-grade notebook scaffold, PAPER.md, SEAN-README.md, and THEORY.md.

**Phase 0 (now):** the `pitchseq` package per SPEC §2–§11, genuinely running and
passing its correctness oracle offline on CPU against synthetic fixtures and the
committed 20k sample.

**Phase 1 (now):** all seven workstreams scaffolded and code-complete against
sample/fixtures — unit-tested where feasible, but **no full-scale training and no
notebook execution** here (the full data is not in this environment).

**Phase 2 (later):** real runs on the full data — **on Sean's desktop, because that
is where `statcast.db` lives** (data locality, not hardware, is the gate; nearly
everything is CPU work). The orchestrator plans each run and hands Sean exact,
copy-pasteable PowerShell steps; Sean runs them locally and pastes back the printed
headline numbers; the orchestrator reviews against SPEC acceptance checks and then
has the subagent fill real numbers + the chosen interpretation branch into the
pre-written papers/notebooks/READMEs.

## Phase-2 execution contract (locked)

- Sean's machine: Windows desktop, PowerShell, conda env `statcast`, repo at
  `~\pitch-sequencing-research`, data at `data/raw/statcast.db` and
  `data/raw/statcast_*.parquet`, AMD GPU, free Colab available.
- Every runnable step handed to Sean includes: (a) one line on what/why, (b) exact
  command(s) prefixed `conda activate statcast; cd ~\pitch-sequencing-research`,
  (c) rough expected runtime + what it outputs, (d) exactly what to paste back.
- All heavy scripts must be **resumable and checkpointed**, write outputs to
  `results/` or `artifacts/` (gitignored), and print headline numbers (log loss,
  Δ_order, Δ_matchup, OPE value + CI, ESS, …) plus wall-clock and peak RAM — the
  raw material for the SPEC §7 performance-vs-compute Pareto plot.
- CPU is the default everywhere. WS1–WS5 and WS7 are CPU steps on the full 3.85M
  pitches. Only WS6 is GPU-optional: ship both a CPU/local path and a
  self-contained free-Colab (T4) path; note the ROCm/DirectML caveat for the AMD
  GPU and recommend Colab as the simpler route. Only WS6's optional Transformer
  variant is genuinely GPU-preferred.
- Steps are kept small so Sean can run a few at a time and re-run failures cheaply.
- **RUNBOOK.md** at repo root accumulates the exact Phase-2 steps per workstream,
  in build order, as each workstream unit lands. It is the single place Sean works
  from in Phase 2.

## Hard constraints

- Temporal splits only; leakage is fatal; OPE before policy; falsifiable by
  construction; report the honest finding (SPEC §0).
- No full-scale training or notebook execution in this phase (no GPU; full data
  absent — `data/raw/statcast.db` exists only on the local machine).
- Workstreams plug into the shared harness; none re-implements evaluation.
- Plain commit messages, author Sean Salvador. Big data and model blobs stay out
  of git; only the 20k sample and tiny fixtures are committed.

## Build order

WS0 foundation (3 units) → WS1 EB tables → WS2 Bayes Markov → WS3 GBDT stack
(centerpiece) → OPE self-tests already in WS0 → WS4 bandit → WS5 tabular MDP →
WS6 GRU → WS7 offline RL capstone.

WS0 units:

- **0a — data layer:** pyproject + `io / families / reward / outcomes / rolling /
  decision_table / states / sequences / splits` + unit tests on the 20k sample and
  tiny synthetic inputs.
- **0b — fixtures + evaluation:** `synth.py` null/positive worlds, `eval/predictions`,
  `eval/metrics`, `eval/falsification`, `eval/predictability`, `eval/harness` + the
  null/positive acceptance tests of SPEC §11.
- **0c — OPE:** `eval/ope.py` (DM/IPS/SNIPS/DR/stepwise-DR/FQE + diagnostics),
  logged-bandit fixture, behavior-policy-recovery and known-value self-tests,
  leakage-guard audit test; whole suite green.

Each Phase-1 workstream then ships as one reviewed unit (WS3 may split into two:
code+tests, then docs+notebook).

## Locked design decisions

| # | decision | rationale |
|---|---|---|
| D1 | `configs/default.yaml` is canonical; duplicate root `default.yaml` removed | two copies of a contract drift |
| D2 | `.gitignore` covers `CLAUDE.md`, egg-info, pytest cache, model/artifact dirs; sample parquet + SCHEMA.csv explicitly un-ignored | keep the repo clean; the two committed data files stay visible to fresh clones |
| D3 | The 20k sample is a **random row sample** (19,050 PAs / 20,000 pitches): within-PA history is mostly absent. Sequence logic is validated on synthetic fixtures; the sample validates schema conformance, dtypes, family mapping, reward sign, and end-to-end runnability | discovered on inspection; affects what each test can honestly claim |
| D4 | Reward sign confirmed on sample: `delta_run_exp` < 0 on strikes/fouls, > 0 on balls, so `R = -delta_run_exp` is pitcher-positive as SPEC §5 assumes | verified empirically before building |
| D5 | Outcome event tree fixed at two levels. Level 1 (pitch result): `ball / called_strike / whiff / foul / hit_by_pitch / in_play` from `description`. Level 2 (conditional on `in_play`): `out / single / double / triple / home_run` from `events`. Node-level and final-event losses per SPEC §8.2 | SPEC names the event tree but leaves node granularity open; this is the minimal tree covering §8.2 |
| D6 | Batter-relative location: `plate_x_br = plate_x` for RHB, `−plate_x` for LHB (positive = outer half for either hand); `plate_z_norm = (plate_z − sz_bot)/(sz_top − sz_bot)` | removes handedness ambiguity in SPEC §3.3 |
| D7 | `row_id = "{game_pk}_{at_bat_number}_{pitch_number}"` (string) is the prediction-table key | deterministic, human-readable join key |
| D8 | Subagents never commit; the orchestrator reviews then commits | single point of review |
| D9 | PyTorch install deferred until WS6 | disk allowance; nothing earlier needs it |
| D10 | `base_state` bit-encodes occupancy: `on_1b→1, on_2b→2, on_3b→4` (0–7) | matches SPEC §3.2 derivation |
| D11 | Tabular **O** view = C + last-3 ordered pitch-token slots + consecutive-diff and run-length features; deep/Markov **O** = full ordered sequence via `sequences.py`. Same information set, representation differs by model family | SPEC §6 defines the information ladder, not the encoding |
| D12 | Every workstream ships CLI entry points designed for Sean's Windows box: argparse, `pathlib` paths, `if __name__ == "__main__"` guards (Windows spawn), checkpoint/resume, outputs under `results/`/`artifacts/`, headline numbers + wall-clock + peak RAM printed at the end | Phase-2 execution contract |
| D13 | Shared `pitchseq/runmeta.py` utility (psutil-based) records wall-clock, peak RSS, and run metadata to JSON next to each output — feeds the §7 Pareto plot | one implementation, used by every runnable step |
| D14 | `results/` added to `.gitignore`; only tiny summary JSON/CSVs may be committed deliberately after review | keep heavy artifacts out of git |
| D15 | WS0 gains a runnable full-data build step: `python -m pitchseq.build_table` (per-season checkpointed decision-table build, `data/processed/decision_table.parquet`) — first RUNBOOK step of Phase 2 | the decision table is the first thing Sean must build locally |
| D16 | §8.2 count-based reference baselines (global family-by-count×hand, shrunk pitcher-by-count, first-order transition, pitcher×count×prev) live in `eval/baselines.py` — a deliberate one-file addition to the SPEC §2 tree | they are scoring references used by the harness, not a workstream |
| D17 | Falsification tests operate through a **model-callback interface** (caller supplies `fit(X,y) → predict_proba`); falsification builds the permuted/pseudo-history datasets and compares losses. Unit tests exercise it with small LightGBM/logistic learners on the synthetic worlds | permutation/pseudo-history tests require retraining; the harness must not know model internals |
| D18 | Synthetic worlds emit **raw-Statcast-schema-compatible** frames (the minimal column subset the builders need), so fixtures flow through `build_decision_table → states → eval` and exercise the leakage-safe path end-to-end | the strongest form of the SPEC §11 oracle |
| D19 | `src/pitchseq/runmeta.py` (psutil) lands in unit 0b — wall-clock + peak-RSS context manager feeding `seconds`/`peak_mem_mb` in the prediction schema and all Phase-2 run logs | prediction schema and the Pareto plot need it from the start |
| D20 | Null world allows **selection** order-dependence but forbids **outcome** order-dependence; positive world injects a known previous-pitch outcome effect (velo-differential whiff boost) with stored ground-truth magnitude | separates finding #1 from #2 (SPEC's three conflated findings); fixtures carry their own truth for acceptance tests |
| D21 | Interpretation rule for Δ_order, binding on all workstream papers/notebooks: Δ_order = Loss(min[U,L1]) − Loss(O) is **negatively biased under the null** (min of two noisy losses is optimistic), so the null criterion is "not significantly positive", and a small negative Δ_order on real data reads as *consistent with no ordering effect*, never as "order hurts". The planted positive-world effect keys on the transition **into** the previous pitch (\|velo_{t−1} − velo_{t−2}\|), because a \|velo_t − velo_{t−1}\| effect is already representable by L1 and cannot separate O from L1 | discovered while building the §11 oracle; prevents misreading Phase-2 results |
| D22 | Outcome-target models condition on the current action (`action_family` appended to every view); selection-target models never do | SPEC §0's "after conditioning on the current pitch" — keeps finding #1 out of finding #2 |
| D23 | `build_table` builds each season from raw spanning `[min(date in season) − 365d, max(date in season)]`, then filters to the season — per-season checkpoints are byte-identical to a whole-data build while staying resumable | reconciles D15 checkpointing with the cross-season trailing window |
| D24 | OPE agreement verdict operationalized: a pair of estimators disagrees iff \|v_i − v_j\| exceeds the wider of their 95% CI half-widths; the verdict set is {DM, SNIPS, DR} on one-step data and {stepwise-DR, FQE} on sequential data. Raw IPS is excluded (variance would trigger vacuously); FQE-on-bandit is excluded (identically DM). Any disagreement ⇒ INCONCLUSIVE per SPEC §9 | SPEC names the principle; this pins the test |
| D25 | WS1 view→key mapping (tabular EB): C → balls×strikes×stand×p_throws with a pitcher hierarchy level; L1 → C + prev family; U → C + unordered prior-family multiset signature (capped); O → C + ordered last-2 prior families; **OM declared infeasible for pure tables** and documented as the support-problem exhibit | tables cannot pool matchup memory; saying so honestly is part of WS1's job |
| D26 | WS1 shrinkage: selection = hierarchical Dirichlet-multinomial, concentration fitted per level (child ← parent backoff); run value = hierarchical normal partial pooling (precision-weighted toward parent). Posterior uncertainty reported, not just point estimates | the "empirical Bayes" in WS1, and the uncertainty WS4 will want later |
| D27 | WS1 positive-world control: the planted velo-transition effect is only *indirectly* visible to family tables (family archetypes ⇒ velo bands ⇒ the O-keyed last-2-family tables proxy the trigger). WS1 must detect the direction with attenuated magnitude and REPORT the attenuation; failure to reach the full planted size is expected, not a bug | tests the tables honestly without changing the fixture |
| D28 | Every workstream ships: standard-schema predictions per feasible view scored ONLY by the shared harness; a support-diagnostics exhibit; a RUNBOOK section; tests in the top-level suite (`tests/test_wsN_*.py`); docs (PAPER/SEAN-README/THEORY) inside `workstreams/wsN_*/`; notebooks under `notebooks/` | uniformity is what makes the comparison a measurement |
| D29 | WS2 = per-context variable-order Markov over family tokens with hierarchical Dirichlet backoff (depth-k context → depth-(k−1) suffix), K_MAX=4, concentrations fitted per depth (MML, MoM fallback), conditioned on the count×hand cell with a pitcher hierarchy level. View mapping: C = order-0 (count cell), L1 = order-1, O = variable order ≤ K_MAX; **U and OM are declared not-applicable for a Markov grammar** (inherently ordered; no cross-PA pooling) and documented | SPEC §12.2's "fuses the Markov and Bayesian-pooling ideas", made concrete |
| D30 | WS2's null-world role inverts: the null world plants an order-2 **selection** habit (no-three-in-a-row), so it is a **positive control for WS2's grammar** (must detect effective order ≥ 2 and the repeat-suppression motif); the negative control is stratified history permutation (grammar must collapse toward order ≤ 1). Outcome-nullness is irrelevant to a selection grammar | the fixture's selection/outcome separation (D20) pays off |
| D31 | WS2 is the first consumer of `eval/predictability.py`: report B_seq (bits of predictability from ordered history, q_O vs q_C) with per-slice breakdowns, plus the grammar exhibits (effective-order distribution, top motifs by lift) | SPEC §10 wired in early, on the model built to show it |

## Dispatch log

| unit | dispatched | verdict | commit |
|---|---|---|---|
| housekeeping (this file, .gitignore, dedupe config) | — (orchestrator) | — | 811db47 |
| Phase-2 execution contract | — (orchestrator) | — | a62b168 |
| 0a data layer | 2026-07-13 | **ACCEPTED after one fix round.** Round 1: high quality, 42 tests green, SPEC-conformant. Review found (1) OM cross-matchup contamination — groupwise cumsum followed by a global `.shift(1)` leaked the previous matchup's totals into the first PA of the next (confirmed by probe); (2) `episode_returns` agreement guard falsely fired when a PA's last pitch has missing `delta_run_exp` (would have broken on full data); (3) `prev_pitch_type` categorical vocabulary was data-dependent. Fix round: all three fixed and re-verified by orchestrator probes; implementer session was cut off by an account spend limit after writing 2 of 3 regression tests — the final one (reward-NaN) was codified by the orchestrator from its own review probe. 45 tests green; sample end-to-end demo unchanged (C=29 U=46 L1=41 O=82 OM=103). | this commit |
| 0b fixtures + eval | 2026-07-13 | **ACCEPTED, no fix round.** runmeta + synth null/positive worlds + eval/{predictions,metrics,baselines,falsification,predictability,harness}. 90/90 tests green (~45s), re-verified by orchestrator. §11 oracle passes deterministically: null world Δ_order=−0.0145 (CI [−0.026,−0.006], perm p=0.80, quiet) → NULL_CONFIRMED; positive world Δ_order=+0.0687 (CI [+0.051,+0.085], perm p=0.040, fires; planted effect recovered at 0.71× magnitude; mechanism ablation isolates velo_diff) → POSITIVE_CONFIRMED. Reviewed in depth: permutation permutes only order-carrying columns within strata×prior-family-multiset with per-permutation refits; synth outcome vector provably context+current-family-only in the null world. Three sharp ambiguity resolutions adopted as D21/D22. | this commit |
| 0c OPE | 2026-07-13 | **ACCEPTED, no fix round** (agent ran its own correctness review mid-build and fixed three robustness gaps before reporting). eval/ope.py (DM/IPS/SNIPS/DR/stepwise-DR/FQE, diagnostics, π_α, evaluate_policy, behavior-policy recovery), logged-bandit + two-step-MDP fixtures with analytic values, build_table CLI, RUNBOOK steps 0–1. 119/119 tests green, re-verified. §9 self-tests pass against analytic truth: behavior recovery exact for IPS/SNIPS (w≡1); known-value recovery within CI for both canonical targets; double-robustness demonstrated both ways (corrupt q̂ → DM biased/DR fine; corrupt w → IPS biased/DR fine); INCONCLUSIVE verdict fires on engineered disagreement. π_α interpolates exactly between V(μ) and V(π̃) with the expected ESS/KL/TV degradation. One sanctioned deviation (+3-line eval/__init__.py registration) accepted. **WS0 IS COMPLETE.** | this commit |
| WS1a code | 2026-07-13 | **ACCEPTED, no fix round.** Hierarchical Dirichlet-multinomial selection + hierarchical-normal run-value tables over the D25 ladder, MML-fitted concentrations (MoM fallback), posterior uncertainty, backoff counters, support exhibit, harness-scored run_ws1.py, RUNBOOK WS1.1. 141 tests green (22 new), demos re-run by orchestrator and matching. Emergent validation: on the null world the fitted history-level α→1e6 pools order away and Δ_order = 0 exactly; on the positive world the recovery signature is O>U>L1≈C≈0 with attenuation ~0.03 (D27's mechanism-blindness lesson, quantified). Two honest world-specific artifacts documented rather than hidden: WS1-C loses ~0.009 to the quick pitcher_count reference on synthetic worlds (stand/p_throws uninformative there); U vs O cell-count order is world-dependent. | this commit |
| WS1b docs+notebook | 2026-07-13 | **ACCEPTED, no fix round.** 63-cell notebook scaffold (validates, all code cells compile, zero saved outputs; agent additionally ran a throwaway copy end-to-end and reproduced the null signature), PAPER.md (~4.1k words, 15 placeholder slots, real references only), THEORY.md (10 named-derivation sections), SEAN-README.md. S1–S2 × R1–R3 branch grid identical across all four artifacts. Agent's reconciliation accepted: WS1's Δ_order is on the *selection* target in the shipped code, so docs frame it that way and route the outcome question to the positive-world run-value recovery. **WS1 COMPLETE.** | this commit |
| WS2a code | 2026-07-13 | **ACCEPTED (salvaged).** Implementer was killed by the account spend limit during its final post-edit confirmation run; all artifacts had landed (model.py with exact suffix-chain hierarchical-Dirichlet backoff, run_ws2.py, 31 tests, RUNBOOK WS2.1). Orchestrator completed the verification the agent could not: full suite exit 0 (172 tests), null-world demo re-run — GRAMMAR_DETECTED (O=1.3322 < L1=1.3385, delta_order_L1=+0.0063 CI[+0.0041,+0.0080], order-2 mass 0.296, all top-5 motifs = the planted repeat-suppression habit, correctly signed) and COLLAPSES_UNDER_PERMUTATION (edge → +0.0001, order-2 mass → 0). B_seq=+0.012 bits overall, 0 at t≤2, rising with PA depth and count leverage — textbook. | this commit |
| WS2b docs+notebook | 2026-07-14 | **ACCEPTED, no fix round** (dispatched after the spend limit was raised). 51-cell notebook scaffold (validates, compiles, 0 outputs; throwaway copy ran end-to-end reproducing both D30 verdicts), PAPER.md (13 real references), THEORY.md (9 named-step sections incl. the interpolation-unroll ↔ effective-order tie), SEAN-README.md. G×B×M branch grid shared across artifacts. Five honest reconciliations accepted, notably: WS2's permutation control stratifies on count×hand×pitch_number only — coarser than SPEC §8.3's finest — documented as a limitation, with the narrow order-beyond-L1 claim resting on the delta_order_L1 CI instead. **WS2 COMPLETE.** | this commit |
| D32 | WS3 outcome model is the decomposed event tree: P(outcome1\|s,a) × count-conditional node run values (+ P(outcome2\|·) for in-play), assembled into E[R\|s,a]; a direct E[R\|s,a] GBDT regressor ships as a cross-check with bounded-disagreement reporting | interpretable, reusable pieces; SPEC §8.2 wants node probs AND exp_reward |
| D33 | WS3 publishes the counterfactual q̂ grid — E[R\|s,a] for all 8 families per decision row, plus behavior propensities μ(a\|s) per view — as saved artifacts with a loader; WS4/WS5/WS7 consume these and never refit their own outcome models | SPEC §12.3: WS3 "supplies the behavior propensities + outcome model that WS4/5/7 consume" |
| D34 | WS3 runs ALL FIVE views (first workstream that can); headline Δ_order and Δ_matchup with clustered CIs are THE project's central ablation. Hyperparameter budget predeclared: fixed sensible LightGBM defaults + one small validation grid (num_leaves × min_child_samples × learning_rate, ≤ 12 combos), identical per view; all runs runmeta-logged for the §7 Pareto plot | SPEC §7 equal-budget discipline |
| D35 | WS3 reruns the §11 oracle as a real-model test: null world quiet (D21 criterion); positive world must recover the planted effect nearly directly (O view carries o_velo_delta_last), with mechanism ablation isolating the velo channel — contrast exhibit vs WS1's ~0.03 family-proxy attenuation | the centerpiece must pass the same gate the harness did |
| WS3a code | 2026-07-14 | **ACCEPTED, no fix round.** Behavior μ(a\|s) + decomposed event-tree outcome stack across ALL FIVE views; q̂ grid + propensities + joblib persistence with `load_ws3_artifacts` (the D33 contract WS4/5/7 consume); per-view/per-stage checkpointed run_ws3.py; 31 new tests (203 total, exit 0, re-verified). D35 gates pass: null world NULL_QUIET with the D20 separation displayed live (Δ_order selection +0.0088 CI>0 — planted habit; Δ_order outcome −0.003 n.s.); positive world MECHANISM_RECOVERED — Δ_order outcome +0.028 val / +0.038 locked test, permutation p=0.04, mechanism ablation top=velo_diff, **recovery ratio 0.955 vs WS1's 0.03**, and `o_velo_delta_last` is the top gain feature (verified in the saved report). Two honest flags carried forward: OM decomposed-vs-direct disagreement slightly over threshold on untuned demo models; **Δ_matchup significantly negative on both synth worlds** (real overfit cost of matchup features where no matchup effect exists — must be a pre-written branch (M−) in WS3b and Phase-2 reading). Falsification battery is synth-only by design (24× refits infeasible at 3.85M rows). | this commit |
| WS3b docs+notebook | 2026-07-14 | **ACCEPTED, no fix round.** 52-cell centerpiece notebook (validates, compiles, 0 outputs; tiny-world throwaway ran end-to-end and fired H1×M− live with `o_velo_delta_last` rank-1), PAPER.md (~5.6k words), THEORY.md (9 sections incl. the Δ_matchup no-min-bias derivation and the q̂ ≠ do() firewall), SEAN-README.md. SPEC §6 H×M branch grid as two independent axes with cross-reading; both WS3a honest flags carried (M− pre-written; OM cross-check flag story). All formulas verified verbatim against model.py docstrings. **WS3 COMPLETE.** | this commit |
| D36 | WS4 target policy = Thompson probability over WS3's posterior q̂ (per-action normals from exp_reward ± sd, P(argmax) by sampling), feasibility-masked, XX excluded; softened via the SPEC §9 π_α mixture across the config alpha grid. No online learning — a fixed uncertainty-aware target from logged data | SPEC §12's "uncertainty-aware target policy, evaluated offline"; reuses WS3 per D33 |
| D37 | WS4 evaluates STRICTLY through eval/ope.evaluate_policy (μ = WS3 propensities, q̂ = WS3 grid); behavior-policy recovery re-run in-pipeline before any target value is reported; the full §9 diagnostics block per α; INCONCLUSIVE honored verbatim | OPE-before-policy (SPEC §0.3) enforced structurally |
| D38 | WS4's core exhibit is the **prescriptive ablation**: V(π̃ built from view-v q̂) for v ∈ {C, L1, O} on both synthetic worlds — null world: all views' policies improve equally (any gain is count-driven, not sequencing); positive world: O-view policy must beat C-view policy (the planted mechanism is exploitable only with ordered state). Plus deviation-from-behavior maps and the value-vs-α frontier with ESS/support overlays | extends the C/U/L1/O/OM ladder into prescription — the bandit's own falsification |
| D39 | Pre-written honest-negative branches are mandatory: INCONCLUSIVE verdict or lower_95 below behavior value are first-class outcomes with their own interpations, not failure states | SPEC §9's closing rule |
| WS4a code | 2026-07-14 | **ACCEPTED, no fix round — with a sanctioned deviation that is the unit's best result.** Thompson target policy over WS3's q̂ (feasibility-masked, observed-action fallback for empty masks), OPE-gated pipeline (behavior-recovery runs FIRST), prescriptive-ablation machinery, 23 new tests (226 total, exit 0, re-verified). Deviation: the dispatched positive-world verdict SEQ_EXPLOITED is not honestly achievable by a MYOPIC policy on our fixture — the planted effect is a state-value effect (trigger fixed by the prior two pitches; every current family gets the boost), leaving only a ~0.003-run family-differential, below the OPE noise floor. Agent proved this with ground-truth probes, shipped SEQ_INCONCLUSIVE_MYOPIC as a D39 first-class verdict, and added a self-test proving the SEQ_EXPLOITED path fires on a constructed genuine advantage. Null world: SEQ_NEUTRAL_PRESCRIPTION with the O−C gap significantly negative (O-view q̂ overfitting) and count-driven gains correctly explained as non-sequencing. Also fixed a real feasibility-mask bug (mask must be computed on the full table's trailing window; eval-rows-only spuriously flagged 27% no-recommendation). Ambiguity exhibit: ~99% of recommendations are toss-ups at 95% posterior confidence — an honest statement about myopic pitch prescription. | this commit |
| WS4b docs+notebook | 2026-07-14 | **ACCEPTED, no fix round.** 45-cell notebook (validates, compiles, 0 outputs; throwaway ran end-to-end → G-PASS/V0/P0/INCONCLUSIVE), PAPER.md (~6.8k words, 8 real references incl. Thompson 1933 and Dudík et al. 2011), THEORY.md (9 sections; the state-value/action-differential ceiling decomposition; SNIPS-nonlinearity honesty), SEAN-README.md. Four-axis branch grid (Gate × Value × Sequencing × Verdict); the myopic ceiling is the chapter's stated contribution. **WS4 COMPLETE.** | this commit |
| D41 | WS5 tabular MDP: state = (balls, strikes) × prev-family(9) × velo-gap-trigger flag(2) + absorbing terminals; action = family (feasibility-masked); transitions/rewards Dirichlet-smoothed train counts; policy iteration; PA = episode. The trigger flag makes the setup effect *representable*: P(trigger′\|s,a) depends on the chosen family's velo band vs prev | the simplest sequential model that can value a setup pitch (SPEC §12.6) |
| D42 | WS5 doubles as the OPE cross-validator (SPEC's "transparent simulator"): model-based value of the softened target from the estimated MDP vs eval/ope stepwise-DR/FQE on logged data — agreement within CI on synthetic worlds validates both; divergence is a reported diagnostic, not silently averaged | two independent value estimates or nothing |
| D43 | WS5's ladder-in-state-space ablation: three state designs — count-only (≈C), count+prev (≈L1), count+prev+trigger (≈O-lite). Null world: equal policy values (SEQ_NEUTRAL); positive world: trigger design must win AND exceed the D40 myopic ceiling (~0.003) → SETUP_EXPLOITED — the first rung that can cash the setup | converts D40's target into WS5's acceptance gate |
| D44 | FQE per-episode contributions are V(s₀), and every PA starts in the identical state — so cluster-bootstrapping those contributions yields a **structurally degenerate CI**. FQE uncertainty must come from refit-based bootstrap (refit FQE per cluster resample, paired across arms); degenerate CIs are printed "n/a (constant contributions)", never as intervals. Verdict gates require REAL CIs; at α=1 the honest stepwise-DR gap CI is ±~0.07 wide at synth scale — sequential-OPE variance is the binding constraint | found in WS5a review: the shipped SETUP_EXPLOITED rested on a degenerate CI (false-confidence gate) |
| WS5a code | 2026-07-14 | **ACCEPTED after one fix round.** Round 1 (salvaged through a container restart): tabular MDP (three D43 state designs, Dirichlet-smoothed estimation, policy iteration, simulator, setup diagnostics), but review found the SETUP_EXPLOITED gate resting on a structurally degenerate FQE CI (D44) and the D42 verdict comparing mismatched policies/regimes. Fix round delivered: paired-refit FQE cluster bootstrap (200 reps ≈ 3,000 refits ≈ 1.1 s/fit on synth), degenerate CIs printed n/a, triple-condition gate (refit-FQE lo95 > D40 ceiling AND stepDR > 0 AND MB > 0), per-(design,α) D42 agreement on the same softened policy. Final honest verdicts, orchestrator-verified: null SEQ_NEUTRAL_MDP; positive **SETUP_INCONCLUSIVE** — all three estimators directionally positive (FQE +0.0079, stepDR +0.0107, MB +0.0277 at α=1) with the lower-95 not clearing at synth scale; representability proven by the constructed-world unit test; certification deferred to Phase-2 data scale. Suite exit 0 (31 new tests). Echoes WS4's lesson at the sequential level: OPE variance, not point estimates, binds prescriptive claims. | this commit |
| WS5b docs+notebook | | | |
| D40 | **The myopic ceiling is now measured**: on the positive fixture, the maximum myopically-exploitable value is ~0.003 run (family-differential only), below OPE resolution; the full planted effect is a *setup* (state-value) effect. WS5/WS7's positive-world acceptance is therefore to exceed the myopic ceiling by valuing the setup (sequential credit), turning WS4's honest INCONCLUSIVE into the ladder's motivating contrast | converts a null into a falsifiable target for the sequential rungs |
| WS5 | | | |
| WS6 | | | |
| WS7 | | | |
