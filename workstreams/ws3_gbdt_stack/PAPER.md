# A Gradient-Boosted Ablation of Five Nested State Views for Pitch Sequencing

*Workstream 3 of a comparative pitch-sequencing study — the **centerpiece**.* This draft is
written to be completed in place: every quantity that depends on the real Statcast data is a
`{PLACEHOLDER}`, and the Results and Discussion are **branched** so that the correct
interpretation is already written for whichever numbers arrive. The synthetic-world numbers
quoted in §5 are *completed validation*, not placeholders.

---

## Abstract

This workstream is the study's central measurement instrument. It is the first to run **all
five** nested state views (`C`, `U`, `L1`, `O`, `OM`) through a single strong model family —
gradient-boosted decision trees — and the first to test whether ordered pitch history sharpens
not only *selection* (what is thrown next) but *outcomes* (the run value of the current pitch,
after conditioning on it). We fit, per view, a multiclass LightGBM **behavior** model
`μ(a | s)` over the 8 pitch families and a **decomposed event-tree outcome** stack (decision
D32): a stage-A classifier `P(o₁ | s, a)` over 6 pitch results, a stage-B classifier
`P(o₂ | s, a)` over 5 in-play refinements, and count-conditional node run values `V(node, c)`,
assembled into `E[R | s, a]` with a residual-based uncertainty, plus a direct `E[R | s, a]`
regressor as a cross-check. The central ablation reports `Δ_order = Loss(min[U, L1]) − Loss(O)`
and `Δ_matchup = Loss(O) − Loss(OM)` with pitcher-game clustered CIs on **three** targets —
selection log loss, outcome-1 log loss, and run-value MAE — on validation (2024) and the locked
test (2025). On the real data (2021–2025 Statcast, ~3.85M pitches) the outcome-1 `Δ_order` is
{DELTA_ORDER_OUTCOME1} (95% clustered CI {DELTA_ORDER_OUTCOME1_CI}) on validation and
{DELTA_ORDER_OUTCOME1_TEST} ({DELTA_ORDER_OUTCOME1_TEST_CI}) on the locked test, and
`Δ_matchup` is {DELTA_MATCHUP_OUTCOME1} ({DELTA_MATCHUP_OUTCOME1_CI}). We validate the method on
the correctness oracle as a real-model test (decision D35): on the null world the planted
*selection* habit is detected (`Δ_order` selection +0.0088, CI [+0.0021, +0.0143]) while the
*outcome* `Δ_order` is null (−0.0033, CI [−0.0081, +0.0008]) and the permutation test does not
fire (`NULL_QUIET`); on the positive world with a planted velocity-transition whiff effect the
outcome `Δ_order` is +0.0277 (CI [+0.0211, +0.0339]) on validation and +0.0382 (CI [+0.0313,
+0.0449]) on the locked test, the permutation test fires (p = 0.040), the mechanism ablation
isolates the velocity channel, and the model recovers the planted whiff lift at **ratio 0.955**
— against WS1's ~0.03 name-keyed-table attenuation, the study's headline **contrast exhibit**
(`o_velo_delta_last` is the top gain feature of the `O` and `OM` outcome models). A methodological
finding carried forward: on **both** synthetic worlds `Δ_matchup` is significantly **negative**
(null −0.0078, positive −0.0129), a genuine out-of-sample cost of the matchup feature block where
no matchup effect exists — pre-written here as the live `M−` branch. Beyond the ablation, WS3
publishes the artifacts every prescriptive workstream consumes (decision D33): the counterfactual
value grid `q̂(s, a)` for all 8 families and the behavior propensities `μ(a | s)`. The headline is
a two-axis result grid — does *order* carry outcome value (H1/H2/H3), and does *matchup memory*
(M+/M0/M−) — read under the study's finding #2 / finding #3 firewall: a boosted ablation is
*predictive*, not causal.

---

## 1. Introduction

The study this workstream belongs to is organized as a **rigor ladder**, not a horse race
(SPEC §0):

> How much evidence for sequencing survives progressively harder tests — from descriptive
> order patterns, to out-of-sample **outcome** dependence, to **counterfactual policy** value?

The ladder exists because three findings are routinely conflated in the applied literature and
must be kept separate (SPEC §0, verbatim):

> 1. **Selection structure** — prior pitches help predict *what is thrown next*.
> 2. **Predictive sequencing value** — prior pitches help predict the *outcome* of the current
>    pitch, after conditioning on the current pitch and game state.
> 3. **Prescriptive/causal value** — *changing* the sequence would improve outcomes.
>
> The first is easy; the second is hard; the third needs assumptions that public data cannot
> fully satisfy (we never see the pitch that wasn't thrown; `pitch_type` is a classifier
> output, not the battery's intent; scouting/target info is unobserved).

Workstream 3 (WS3) is the **centerpiece**. It is where the ladder's central question — does
ordered history carry *out-of-sample outcome* value (finding #2), not merely selection structure
(finding #1) — is measured with a single strong, uniform model. Three properties make it the
study's measurement instrument rather than one more model on the pile.

**First, it runs all five views through one model family (decision D34).** WS1's tables declared
`OM` infeasible and `U`/`O` world-dependent; WS2's grammar declared `U` and `OM` not-applicable.
WS3 is the first workstream that fits every one of `C`, `U`, `L1`, `O`, `OM` with the same
gradient-boosted machinery, so the `Δ_order` and `Δ_matchup` ablations are finally computed
like-for-like across the whole ladder. The differences between the five views' losses *are* the
sequencing evidence (SPEC §6), and WS3 is the first to read all four rungs of it.

**Second, it measures outcomes, not just selection.** WS1 probed run value with flat tables and
WS2 was scoped to selection alone (its firewall). WS3 fits the decomposed event-tree outcome
model (decision D32) that those workstreams could not, so it is the first to put a clustered CI
on the outcome-1 `Δ_order` — the project's central number — and on the run-value MAE ablation.
Because the tabular `O` view carries pitch *physics* as features (velocity and location
differences, decision D11), WS3 can see mechanisms a name-keyed table is blind to; the positive
synthetic world quantifies the gap as a recovery ratio of 0.955 against WS1's ~0.03 (§5).

**Third, it supplies the prescriptive phase.** WS4 (bandit), WS5 (MDP), and WS7 (offline RL)
never refit their own outcome or behavior model — they **import** WS3's saved counterfactual q̂
grid and propensities through `load_ws3_artifacts` (decision D33). WS3's outputs are therefore
load-bearing for half the project, which is the operational reason SPEC §12.3 names it the
centerpiece: "If O doesn't beat U/L1 here, be skeptical of everything fancier."

**Contributions.**
1. The first uniform five-view `Δ_order` / `Δ_matchup` ablation on **three** targets (selection,
   outcome, run-value) with pitcher-game clustered CIs on validation and a locked 2025 test — the
   project's central results table.
2. An interpretable **decomposed outcome stack** (stage-A/stage-B classifiers × count-conditional
   node values) assembled into `E[R | s, a]` with a residual-based uncertainty, cross-checked
   against a direct regressor (decision D32).
3. The **contrast exhibit**: a real-model falsification (decision D35) that recovers a planted
   velocity-transition effect at ratio 0.955, versus WS1's ~0.03 name-proxy attenuation —
   concrete evidence that the ladder needs more than one rung.
4. A carried-forward **methodological finding** — a significantly negative `Δ_matchup` on both
   synthetic worlds — established as a real fragmentation/overfit cost and pre-written as a live
   real-data branch.
5. The **D33 artifact contract**: q̂ grid + propensities + fitted models, saved with a loader that
   WS4/WS5/WS7 consume.

---

## 2. Related work

**Gradient-boosted decision trees.** WS3's model family is gradient boosting, introduced as
greedy stagewise function approximation by Friedman (2001): fit each new tree to the functional
gradient (pseudo-residuals) of the loss, and shrink it by a learning rate. The two systems that
made boosted trees the default for tabular problems refine the leaf-value step with second-order
information — Chen and Guestrin (2016), XGBoost — and the histogram/leaf-wise engineering that
makes it scale to millions of rows — Ke et al. (2017), LightGBM, the library WS3 uses. WS3's
multiclass models are softmax cross-entropy ensembles with one tree per class per round (§4;
`THEORY.md` §2).

**Calibration of boosted classifiers.** Boosted trees are accurate rankers but not automatically
well-calibrated probability estimators; Niculescu-Mizil and Caruana (2005) document the
characteristic sigmoidal distortion and the post-hoc corrections for it. Because WS3's outcome
probabilities are consumed as *levels* by the assembly and by the downstream prescriptive
workstreams (not merely as a ranking), we report reliability curves and a run-value calibration
slope/intercept (SPEC §8.2), and read them with this literature in mind.

**Baseball analytics and sequencing.** The applied grounding is Tango, Lichtman, and Dolphin
(2007), whose count- and matchup-based percentage analysis motivates the count × handedness ×
pitcher context that defines the `C` view, and Marchi and Albert (2013) for the treatment of
Statcast-style pitch data. The empirical-Bayes pooling philosophy the study leads with is Efron
and Morris (1975), the canonical baseball shrinkage demonstration; WS3 inherits it indirectly
through the count-conditional node values (train-fold means with a global fallback) and through
the reference baselines it is scored against (decision D16).

**Internal cross-references.** WS3 is read against its two predecessors on the ladder. WS1
(empirical-Bayes tables) established the *support problem* and quantified name-keyed
*mechanism-blindness* at ~3% recovery of a planted velocity effect — the number WS3's 0.955 is
the contrast to. WS2 (variable-order Markov grammar) resolved the *selection* depth question
(finding #1) with a calibrated effective-order read and a firewall to outcomes; WS3's selection
`Δ_order` should be at least competitive with WS2's grammar, and its outcome ablation is the
finding-#2 test WS2 deliberately did not attempt. The shared ablation-bias rule (decision D21)
and its absence for straight-difference statistics are derived in `../ws1_eb_tables/THEORY.md`
§10 and `../ws2_bayes_markov/THEORY.md` §8, and reused here (§5; `THEORY.md` §6).

---

## 3. Data

**Source.** Statcast pitch-level data via pybaseball, seasons 2021–2025, a single tracking era
(Hawk-Eye), ~3.85M pitches over 119 raw columns (SPEC §1). The reward signal `delta_run_exp` is
populated on ~99.7% of pitches. `pitch_type` is a Statcast *classifier output* and is treated
throughout as an observed proxy for the pitch thrown, not as the battery's intent (SPEC §1, §8).

**The canonical decision table and the five views.** Every workstream reads one shared table
(SPEC §3): one row per pitch, the decision made immediately *before* that pitch is released, keyed
by `(game_pk, at_bat_number, pitch_number)`. WS3 consumes the five nested state views built by
`states.py` (SPEC §6): `C` (non-sequence context — count, base-out, inning, score, handedness,
rolling repertoire and tendencies), `U` (C + order-invariant aggregates of the prior pitches this
PA), `L1` (C + the immediately preceding pitch), `O` (C + U + L1 + the ordered current-PA
sequence: last-3 positional slots, consecutive-difference features, run length — decision D11),
and `OM` (O + matchup memory: within-game batter-vs-pitcher and trailing batter-vs-family). The
views are information-nested by construction, and their feature widths on the committed demo are
`C = 29`, `U = 46`, `L1 = 41`, `O = 82`, `OM = 103`.

**The action and the outcome tree.** The action is the pitch family (8 classes, SPEC §4).
Outcome models append the current `action_family` to the view (decision D22) — SPEC §0's
"after conditioning on the current pitch," which keeps finding #1 out of finding #2. The outcome
event tree (decision D5) has level 1 `o₁ ∈ {ball, called_strike, whiff, foul, hbp, in_play}` and
level 2 (on in-play rows) `o₂ ∈ {single, double, triple, home_run, out_or_other}`.

**Reward.** `R = −delta_run_exp`, larger better for the pitcher (SPEC §5); the mandatory sign
check (R > 0 on called strikes / whiffs / outs, R < 0 on balls / walks / home runs) is verified
before any run-value node is computed.

**Leakage discipline.** A feature for the decision before pitch `t` must be computable strictly
before pitch `t` is thrown (SPEC §0). The current pitch's own execution (its velocity, movement,
location) is downstream of the decision and never a state feature; prior pitches' realized
measurements are fully observed before the current decision and are legitimate. The behavior view
is leakage-audited directly; the outcome view is audited on every column except the appended
`action_family` (which is the decision, not execution). WS3 inherits the shared column-name audit
by reading only the audited views.

**Temporal splits.** Train on 2021–2023, select on 2024, lock 2025 for test (SPEC §7). Splits are
by season, never by random row, and all confidence intervals are bootstrapped over pitcher-game
clusters.

---

## 4. Methods

### 4.1 The two stacks

WS3 fits two things per view. **The behavior model** `μ(a | s)` (`BehaviorModel`) is a multiclass
LightGBM over the 8 families, trained on the state view alone — no action, no execution — and
answers *selection* (finding #1). **The decomposed outcome stack** (`OutcomeStack`, decision D32)
is three separately fitted pieces, each conditioned on the current action (decision D22):

- **Stage A** `P(o₁ | s, a)` — a multiclass LightGBM over the 6 level-1 outcomes.
- **Stage B** `P(o₂ | s, a)` — a multiclass LightGBM over the 5 level-2 outcomes, trained on the
  in-play rows only but evaluable for any row (the counterfactual "if put in play, what happens").
- **Stage C** `V(node, c)` — count-conditional node run values: the train-fold mean reward for
  each event-tree node at each count `c = (balls, strikes)`, with a global-node fallback
  (`NodeValueTable`).

### 4.2 The assembly and its uncertainty (the exact formulas)

With the count `c`, the ten event-tree leaves are the 5 terminal level-1 outcomes plus the 5
level-2 refinements of `in_play`. The expected reward is the leaf-probability-weighted sum of node
values,

$$\mathbb{E}[R \mid s, a, c] =
  \sum_{k \in \{\text{ball},\text{called\_strike},\text{whiff},\text{foul},\text{hbp}\}}
       P_A(o_1 = k \mid s, a)\; V_1(k, c)
  \;+\; P_A(o_1 = \text{in\_play} \mid s, a)
       \sum_{j \in \mathcal{O}_2} P_B(o_2 = j \mid s, a)\; V_2(j, c),$$

and the residual-based predictive variance follows from the law of total variance over the leaves,

$$\mathrm{Var}[R \mid s, a, c] = \sum_{\ell} p_\ell\big(S_\ell(c) + V_\ell(c)^2\big) - \mathbb{E}[R \mid s, a, c]^2,
\qquad \texttt{exp\_reward\_sd} = \sqrt{\max(\mathrm{Var}, 0)},$$

with `p_ℓ` the leaf probability, `V_ℓ` its node value, and `S_ℓ` its train-fold within-node
reward variance — no bootstrap. Both formulas match `model.py` and `THEORY.md` §4 exactly.

### 4.3 The direct-regressor cross-check (D32)

A single LightGBM regression `Ê_dir[R | s, a]` on the same `(view + action)` features provides a
consistency check: `mean |E_decomposed − E_direct|`, flagged when it exceeds 0.03 on the
`|R| ~ 0.05–0.3` reward scale (`OutcomeStack.disagreement`). It diagnoses either a mis-specified
node-value lookup or a reward dependence on `(s, a)` beyond `(node, count)` (`THEORY.md` §5); it is
an internal consistency check, not a ground-truth gate.

### 4.4 The D33 artifact contract

Two derived artifacts feed the prescriptive workstreams. The **counterfactual value grid**
`q̂(s, a)` (`q_grid`) sweeps the action over all 8 families for every decision row — build the view
once, swap `action_family`, re-run stages A/B, re-assemble — giving `E[R | s, A = f]` for every
family. The **propensities** `μ(a | s)` (`propensities`) are the behavior model's per-row family
distribution. Both, plus the fitted boosters and node lookups, are persisted per view and reloaded
by `load_ws3_artifacts` into a `WS3Artifacts` object with `.q_grid`, `.propensities`, and
`.exp_reward` helpers. WS4/WS5/WS7 import these and never refit their own outcome/behavior model.
The q̂ grid is a *conditional* expectation, not a causal effect (§7; `THEORY.md` §8).

### 4.5 The hyperparameter budget (D34)

Every view gets the identical predeclared `DEFAULT_PARAMS` (num_leaves 31, min_child_samples 50,
learning_rate 0.05, 300 estimators, `reg_lambda` 1.0, deterministic single-thread) and, under
`--tune`, the identical 12-combo grid
(`num_leaves ∈ {15,31,63} × min_child_samples ∈ {20,100} × learning_rate ∈ {0.03,0.1}`) selected
by validation log loss on an internal holdout **carved from the train seasons** — never the
reported validation/test folds. Stage A tunes; stage B and the direct regressor reuse stage A's
chosen params to bound the grid cost. Equal budgets make the deltas comparable: the only thing
that varies view-to-view is the feature set, so a loss gap isolates the *information* contribution,
not the search budget (SPEC §7; `THEORY.md` §9). Every run is runmeta-logged (wall-clock, peak RAM,
param count) for the SPEC §7 Pareto plot.

---

## 5. Experimental setup

**Scoring.** Every view's predictions are written in the standard schema (SPEC §8.1) and scored
*only* through the shared harness (SPEC §8): multiclass **log loss** (primary) for selection and
outcome nodes, run-value **MAE/RMSE** and a **calibration** slope/intercept, and the ablations
`Δ_order = Loss(min[U, L1]) − Loss(O)` and `Δ_matchup = Loss(O) − Loss(OM)` with pitcher-game
clustered bootstrap CIs, computed on **three** targets: selection log loss, outcome-1 log loss,
and run-value MAE. Every metric is reported across the six SPEC §6 slices (all, `seq_eligible`,
`long_pa`, `two_strike`, `three_ball`, `first_pitch`).

**Reference baselines.** Selection is scored like-for-like against the four count-based references
(`eval/baselines.py`): global family frequency by count × hand, pitcher × count, first-order
transition, and pitcher × count × prev-family. WS3's `O` view should beat the `pitcher_count_prev`
reference on selection.

**The `Δ_order` reading rule (D21).** Because `Δ_order` uses `min[U, L1]` and the minimum of two
noisy losses is optimistically biased low, `Δ_order` is *negatively biased under the null*: the
criterion is "**significantly positive**" (clustered CI lower bound > 0), a small negative value
reads as *consistent with no ordering effect*, never "order hurts," and this binds on all three
targets. The `Δ_matchup` statistic takes **no minimum** — a straight `Loss(O) − Loss(OM)` — so it
carries **no** such optimism bias, and its CI is read directly (§6.2; `THEORY.md` §6).

**Falsification protocol (completed validation).** WS3 reruns the SPEC §11 oracle as a real-model
test (decision D35). The following are *observed results* on the committed WS3a run, not
aspirations:

- **Null world** (an ordered *selection* habit — a no-three-in-a-row tendency — but outcomes
  depending only on context + current family, by construction). The D20 finding-#1/finding-#2
  separation is displayed live: the *selection* `Δ_order` is **+0.0088** (CI [+0.0021, +0.0143],
  significantly positive — the planted habit *is* detected) while the *outcome* `Δ_order` is
  **−0.0033** (CI [−0.0081, +0.0008], not significant — no outcome order effect). The permutation
  test does not fire (p = 0.880), and the mechanism ablation's top group is `family_slots` (there
  is no velocity mechanism to find). Verdict: **`NULL_QUIET`**. On the null world's selection
  target the `O` behavior model beats every count-based reference.
- **Positive world** (a planted whiff boost when the velocity transition into the previous pitch,
  `|velo_{t−1} − velo_{t−2}|`, exceeds 5 mph — keyed on the transition *into* the prior pitch so it
  can separate `O` from `L1`, per D21). The outcome `Δ_order` is **+0.0277** (CI [+0.0211,
  +0.0339]) on validation and **+0.0382** (CI [+0.0313, +0.0449]) on the locked test; the
  permutation test fires (p = 0.040); the mechanism ablation isolates the velocity channel
  (`velo_diff` top, Δ ≈ 0.0144); and the model recovers the planted whiff lift at **0.294 against a
  planted 0.308 — recovery ratio 0.955**. Verdict: **`MECHANISM_RECOVERED`**. The feature-gain
  evidence corroborates it: `o_velo_delta_last` is the **top gain feature** of the `O` and `OM`
  stage-A models (≈28,700 gain).

**The contrast exhibit (decision D35).** WS3's 0.955 recovery against WS1's ~0.03 name-proxy
attenuation is the study's concrete argument that the ladder needs more than one rung: when the
model can split on the *actual mechanism* (`o_velo_delta_last` is a native `O` feature), it
recovers ~95% of a real planted effect; when it can only key on pitch *names* (WS1's tables), it
recovers ~3%. Same effect, same data-generating process — the difference is representation.

**The negative-`Δ_matchup` observation (completed methodological finding).** On **both** synthetic
worlds — where no matchup effect is planted — the outcome `Δ_matchup` is significantly
**negative**: **−0.0078** (CI [−0.0145, −0.0018]) on the null world and **−0.0129** (CI [−0.0193,
−0.0080]) on the positive world. Because `Δ_matchup` has no `min`-bias, this is not a statistical
artifact: it is a genuine out-of-sample **fragmentation/overfit cost** of the ~21-column matchup
block, which adds estimation variance without adding signal where no matchup effect exists
(`THEORY.md` §6.3). It is pre-written below as the live `M−` branch, and it means a negative
`Δ_matchup` on real data must be read as "no detectable matchup outcome signal **and** a real cost
of carrying matchup features," checked with the support diagnostics — not as "matchup hurts
baseball."

**Two honest flags carried forward.** The `OM` decomposed-vs-direct disagreement lands slightly
over the 0.03 threshold on the untuned demo models (other views 0.023–0.026, unflagged), the same
fragmentation story from the assembly side (§4.3). The falsification battery is **synthetic-only by
design**: its 24 permutation refits are infeasible on 3.85M rows, so on real data the central table
carries the ablation and the synthetic worlds carry the recovery/permutation checks. The demo runs
took ~11 minutes per world at peak ~650 MB.

---

## 6. Results

### 6.1 The central table (real data)

The project's headline table — the outcome that every prior workstream was building toward.

| view | width | selection log loss | outcome-1 log loss | outcome-2 log loss | run-value MAE | reference (selection) |
|---|---|---|---|---|---|---|
| `C`  | 29  | {C_SEL_LL}  | {C_O1_LL}  | {C_O2_LL}  | {C_MAE}  | `pitcher_count` {REF_PITCHER_COUNT} |
| `U`  | 46  | {U_SEL_LL}  | {U_O1_LL}  | {U_O2_LL}  | {U_MAE}  | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |
| `L1` | 41  | {L1_SEL_LL} | {L1_O1_LL} | {L1_O2_LL} | {L1_MAE} | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |
| `O`  | 82  | {O_SEL_LL}  | {O_O1_LL}  | {O_O2_LL}  | {O_MAE}  | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |
| `OM` | 103 | {OM_SEL_LL} | {OM_O1_LL} | {OM_O2_LL} | {OM_MAE} | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |

**Ablation, validation (clustered 95% CI):**

| target | `Δ_order` = min[U,L1] − O | `Δ_matchup` = O − OM |
|---|---|---|
| selection | {DELTA_ORDER_SEL} {DELTA_ORDER_SEL_CI} | {DELTA_MATCHUP_SEL} {DELTA_MATCHUP_SEL_CI} |
| outcome-1 | {DELTA_ORDER_OUTCOME1} {DELTA_ORDER_OUTCOME1_CI} | {DELTA_MATCHUP_OUTCOME1} {DELTA_MATCHUP_OUTCOME1_CI} |
| run-value MAE | {DELTA_ORDER_MAE} {DELTA_ORDER_MAE_CI} | {DELTA_MATCHUP_MAE} {DELTA_MATCHUP_MAE_CI} |

**Locked-test confirmation (2025):** outcome-1 `Δ_order` = {DELTA_ORDER_OUTCOME1_TEST}
({DELTA_ORDER_OUTCOME1_TEST_CI}). Decomposed-vs-direct disagreement per view:
{DISAGREEMENT_BY_VIEW}. Run-value calibration (O view): slope {O_CAL_SLOPE}, intercept
{O_CAL_INTERCEPT}. Per-slice outcome-1 `Δ_order`: {DELTA_ORDER_BY_SLICE}.

### 6.2 Branched interpretation — the two axes

The result is read on two independent axes, exactly the SPEC §6 interpretation grid. The
**order** axis (H1/H2/H3) asks whether the fully ordered view `O` beats the history-lite views on
*outcomes*, read under D21. The **matchup** axis (M+/M0/M−) asks whether the matchup view `OM`
beats `O`, read directly (no `min`-bias). Exactly one branch on each axis applies; the notebook's
§9 branch selector prints which, and the write-ups below stand alone once the numbers are filled.
Because the two axes are independent, any H×M pair is a coherent reading; a short cross-reading
follows.

#### Order axis (outcome-1 `Δ_order`, read under D21)

**H1 — `O` beats both `U` and `L1` (CI lower bound > 0): genuine ordered outcome dependence.**
The fully ordered view lowers outcome log loss below the better of `U`/`L1` by more than the
clustered CI — out-of-sample *sequencing value* (finding #2), the strong result. Read the effect
size against the +0.0277 synthetic benchmark: a real-data `Δ_order` an order of magnitude smaller
is still meaningful but modest. Before any headline claim, check three things: per-slice
consistency (a real ordered effect should concentrate in `long_pa` / `two_strike`, not appear only
in aggregate), the mechanism ablation (which feature group carries it), and the locked-test row
(does 2025 confirm 2024). For the study this is the live finding-#2 signal WS4's bandit and WS5's
MDP will try to *use* and WS7's OPE will try to *prove* — and the burden then shifts to whether the
predictable order is also *exploitable* (finding #3), which WS3 cannot answer.

**H2 — `O ≈ U/L1` but both beat `C`: history matters, order does not.** The history views lower
outcome loss below context-only, but the fully ordered `O` does not refine on the better of
`U`/`L1` (`Δ_order` not significantly positive). Within-PA history carries outcome value, but its
*order* beyond the previous pitch / unordered bag does not — the result SPEC §13 explicitly
anticipates ("O barely beats L1"), consistent with the motif literature. It is reported without
embarrassment (SPEC §0): it tells the prescriptive workstreams they need carry only `L1`/`U`
history, and it sets the bar WS6's learned representation must clear to justify going deeper. The
q̂ grid and propensities are still fully valid at the `L1`/`U`/`O` level; the prescriptive phase
simply inherits a "history helps, order doesn't" outcome model.

**H3 — nothing beats `C`: no sequencing signal in outcomes at all.** No history view lowers outcome
loss below context-only. Within-PA history carries no *outcome* value a strong tabular model can
resolve — the cleanest possible finding-#2 null. This does **not** deny finding #1: selection
structure may still exist (WS2's grammar and WS3's own selection ablation can be positive while the
outcome ablation is null — exactly the D20 separation the null world displays). Audit it before
accepting: confirm the outcome models are well-calibrated (an uncalibrated model can mask a real
effect) and that `O`'s selection edge, if any, simply does not translate to outcomes. Under H3 the
prescriptive phase inherits a context-only outcome model, and the study's honest headline is "pitch
sequences are forecastable but not, at this data scale, outcome-predictive."

#### Matchup axis (outcome-1 `Δ_matchup`, read directly)

**M+ — `OM` beats `O` (CI lower bound > 0): real batter–pitcher adaptation.** The matchup view
lowers outcome loss below `O` by more than the clustered CI — the first evidence in the study of
longer-term batter–pitcher adaptation carrying outcome value (WS1/WS2 could not fit `OM`). It must
corroborate on the `first_pitch` slice, which isolates matchup memory by construction (no within-PA
history exists on the first pitch, so any `OM`-over-`O` edge there is *purely* cross-PA). If it
holds, WS3 has found a real matchup outcome signal for WS4/WS5/WS7 to exploit and WS6's cross-PA
representation to extend.

**M0 — `OM ≈ O`: no detectable matchup memory.** The matchup view neither helps nor hurts beyond
`O` (CI spans 0). No detectable longer-term batter–pitcher adaptation in outcomes at this data
scale — a clean read, and the expected one if within-game/season rematch counts are too thin to
resolve. The prescriptive phase can use `O` and `OM` interchangeably for outcomes.

**M− — `OM` significantly below `O`: matchup features actively cost out-of-sample.** The matchup
view has significantly *higher* outcome loss than `O` (CI upper bound < 0). Because `Δ_matchup` has
no `min`-bias, this is **not** a statistical artifact — it is a genuine out-of-sample cost: the
~21-column matchup block adds estimation variance without adding signal, so a strong model
generalizes *worse* with it (`THEORY.md` §6.3). This is **observed on both synthetic worlds**
(−0.0078 null, −0.0129 positive), where no matchup effect exists by construction, so it is a live
real-data possibility, not a hypothetical. The correct reading is "**no matchup outcome signal
plus a fragmentation cost**," confirmed with the support-diagnostics checklist: (i) are the OM
features mostly missing/low-support (within-game rematches are rare early; batter-vs-family
rolling is sparse for low-history batters)? (ii) does the `first_pitch` slice — where OM's *only*
information is cross-PA — show the same or worse loss? (iii) does the decomposed-vs-direct
disagreement flag `OM` (it does on the demo)? If all three point to fragmentation, `M−` is a
statement about *estimation under sparse features*, not about baseball: matchup memory might still
matter, but carrying it as features costs more than it returns here, and the honest move is to hand
cross-PA memory to WS6's pooled/embedding representation rather than force it into a tree.

#### Cross-reading the grid

The two axes are independent, so the honest headline is a pair. The study's *most anticipated*
cell is **H2 × M−** or **H3 × M−**: within-PA history carries a little outcome value but its fine
order does not, and the matchup block costs out-of-sample — a modest, defensible finding-#2 result
with a clean methodological note on why more features are not more information. The *strongest*
cell is **H1 × M+**: genuine ordered dependence *and* real matchup adaptation, a live signal for
the entire prescriptive phase. The *cleanest null* is **H3 × M0**. Whichever pair fires, the
finding is read under the firewall (§7): a boosted ablation measures *prediction*, and only the
OPE/RL workstreams can test whether any of it is *prescriptive*.

---

## 7. Discussion

The interpretation mirrors the §6 grid, so the discussion is written per axis and completed by the
same numbers.

**On the order axis.** Under **H1**, WS3 resolves genuine ordered outcome dependence, and the
burden shifts to the prescriptive workstreams: WS4's bandit and WS5's MDP inherit a q̂ grid that
*encodes* the ordered effect, and WS7's OPE must test whether acting on it beats the observed
policy within support — the finding-#2-to-finding-#3 step the firewall (below) forbids WS3 from
taking itself. Under **H2** (the SPEC §13 expectation), the prescriptive phase inherits a
"history-helps-order-doesn't" outcome model and can carry only `L1`/`U`; the honest contribution is
a *bounded* finding-#2 result with the synthetic worlds proving the model *could* have seen deeper
order had it been there (recovery ratio 0.955). Under **H3**, the outcome channel is null and the
study's finding-#2 verdict is negative — a clean, publishable result that redirects the prescriptive
phase to context-only value and hands the "can a learned representation see what trees cannot"
question to WS6.

**On the matchup axis, and the negative reading specifically.** Under **M+**, WS3 supplies the
first real matchup outcome signal and the `first_pitch` slice is its cleanest witness. Under
**M−** — observed on both synthetic worlds and therefore a live real-data outcome — the key
discipline is to *not over-read it*. A negative `Δ_matchup` is a statement that adding the matchup
feature block hurts out-of-sample outcome prediction *in this data*, driven by estimation variance
over sparse features, not evidence that matchup memory is absent from the game. For the prescriptive
phase this means the outcome model handed to WS4/5/7 should be the `O` model, not `OM`, whenever
`M−` fires; and the cross-PA adaptation question is best routed to WS6's pooled representation, which
can borrow strength across matchups where a tree fragments. The methodological lesson generalizes:
in this study, "more features" is not "more information," and the ablation's negative arm is the
instrument that says so honestly.

**What each branch means for the prescriptive phase.** The q̂ grid and propensities are published
regardless of branch (decision D33), but the *view* the prescriptive workstreams should read from
is branch-dependent: `O` under H1/M0/M−, `OM` under M+, `L1`/`U`-competitive-with-`O` under H2, and
`C` under H3. WS3's job is to make that choice on evidence, and to hand downstream a q̂ grid whose
level is calibrated (the run-value slope/intercept) and whose disagreement flag is clear.

---

## 8. Limitations

1. **Predictive, not causal (the firewall).** A boosted ablation measures whether ordered history
   *predicts* outcomes out-of-sample (finding #2). It does **not** establish that *changing* the
   sequence would change outcomes (finding #3). The q̂ grid is `E_model[R | s, A = f]`, a conditional
   expectation, not `E[R | s, do(A = f)]`; identifying the two requires ignorability and overlap that
   public data cannot satisfy (SPEC §0), which is exactly what the OPE/RL workstreams (WS4/5/7) test
   rather than assume (`THEORY.md` §8).
2. **`pitch_type` is a classifier output**, not the battery's intent; a mislabeled pitch is a
   mislabeled action and a mislabeled outcome condition. Family (8 classes) blunts this but cannot
   remove it (SPEC §1).
3. **Family granularity.** The action and the conditioning are at family resolution; within-family
   variation in velocity/location is carried as *features* (the D11 physics columns) but not as
   *actions*, so a mechanism that lives below the family level is only partially represented.
4. **Classifier-output outcome labels.** The event-tree labels derive from `description`/`events`;
   the two-level tree (decision D5) is the minimal tree covering SPEC §8.2 and does not model, e.g.,
   launch-angle-conditioned run value beyond the in-play refinement.
5. **The D34 budget bounds "GBDT couldn't find it."** A null `Δ_order` under H2/H3 means a strong
   tabular model *within a predeclared 12-combo budget* did not resolve ordered outcome value — not
   that no model could. The budget is fixed for fairness (equal across views); a much larger search
   or a different model family (WS6) could in principle differ, which is why the ladder continues.
6. **Matchup fragmentation.** As §6.2's `M−` branch and the synthetic worlds show, the `OM` feature
   block can cost out-of-sample where matchup signal is thin; WS3 reports this honestly rather than
   hiding it, but it means WS3's matchup read is a *feature-based* one, and a pooled/embedding model
   (WS6) may resolve cross-PA memory that trees fragment.
7. **No catcher/umpire effects** (absent from base Statcast, SPEC §3.2) and **single-metric reward**
   (`−delta_run_exp`); a different reward could reweight the node values.
8. **Falsification is synthetic-only** (decision D35): the permutation battery's 24 refits are
   infeasible at 3.85M rows, so the real-data run carries the ablation and the synthetic worlds carry
   the recovery/permutation checks.

---

## 9. Conclusion

WS3 is the study's centerpiece: the first uniform gradient-boosted ablation of all five nested state
views, the first to measure outcome (not just selection) sequencing value, and the supplier of the
q̂ grid and propensities the entire prescriptive phase consumes. Its conclusion is branch-conditional
and complete once the real numbers arrive, on two independent axes:

- **Order axis.** Under **H1**, WS3 finds genuine out-of-sample ordered *outcome* dependence
  (finding #2) and hands a live signal to WS4/5/7. Under **H2** (the SPEC §13 expectation), it finds
  that within-PA history helps outcomes but its fine order does not — a clean, bounded result whose
  synthetic validation (recovery ratio 0.955 vs WS1's 0.03) proves the model would have seen deeper
  order had it existed. Under **H3**, it finds no outcome sequencing signal at all, redirecting the
  prescriptive phase to context-only value while leaving finding #1 (selection) intact.
- **Matchup axis.** Under **M+**, WS3 supplies the study's first real batter–pitcher adaptation
  signal, witnessed on the `first_pitch` slice. Under **M0**, no detectable matchup memory. Under
  **M−** — observed on both synthetic worlds and therefore a live real-data outcome — the matchup
  feature block *costs* out-of-sample, a genuine fragmentation finding (no `min`-bias to explain it
  away) read with the support-diagnostics checklist and handed to WS6.

Across every cell of the grid the durable contributions are the same: a like-for-like five-view
ablation on three targets with a locked-test confirmation, an interpretable decomposed outcome model
cross-checked against a black-box regressor, the 0.955-vs-0.03 contrast exhibit that justifies the
ladder, and the D33 artifact contract that makes WS3 the load-bearing supply for the prescriptive
phase — all read under the finding-#2 / finding-#3 firewall that keeps a predictive ablation from
being mistaken for a causal one.

---

## References

Chen, T., and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proceedings of the
22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*.

Efron, B., and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations.
*Journal of the American Statistical Association*.

Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of
Statistics*.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. (2017). LightGBM:
a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing
Systems (NeurIPS)*.

Marchi, M., and Albert, J. (2013). *Analyzing Baseball Data with R*. Chapman and Hall/CRC.

Niculescu-Mizil, A., and Caruana, R. (2005). Predicting good probabilities with supervised
learning. *Proceedings of the 22nd International Conference on Machine Learning (ICML)*.

Tango, T. M., Lichtman, M. G., and Dolphin, A. E. (2007). *The Book: Playing the Percentages in
Baseball*.
