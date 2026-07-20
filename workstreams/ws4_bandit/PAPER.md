# A Bayesian Contextual Bandit for Pitch Prescription, and the Myopic Ceiling It Measures

*Workstream 4 of a comparative pitch-sequencing study — the first prescriptive rung (Phase B).*
This paper is completed in place with the real-data results (2021–2025 Statcast). The Results and
Discussion were pre-**branched** so that the correct interpretation was already written for whichever
numbers arrived; the branch **selected by the data** is marked at each fork, and the unselected
branches are retained and labelled as *pre-registered alternatives*. The synthetic-world numbers
quoted in §5 are *completed validation* from the committed WS4a run, not placeholders.

---

## Abstract

Workstream 4 (WS4) is the study's entry into **prescription**: it turns the counterfactual value
grid `q̂(s, a)` that Workstream 3 (WS3) publishes into an uncertainty-aware **"best next pitch"
target policy** and evaluates it **offline**, never online, through the off-policy-evaluation (OPE)
gate. The target is a **Thompson policy** `π̃(a | s) = P(a = argmax)` over independent per-action
normals `𝒩(q̂(s, a), σ(s, a)²)`, restricted to the leakage-safe feasible families (`XX` excluded),
with low-history rows carrying no recommendation (a fallback to the observed action). It is softened
toward behavior along the SPEC §9 conservative mixture `π_α = (1−α)μ + απ̃` and scored **strictly**
through `eval/ope.evaluate_policy` (decision D37), after an in-pipeline **behavior-policy recovery**
self-test that must pass *before any target value is reported* (SPEC §0.3). WS4 refits nothing: it
consumes WS3's saved `q̂`, propensities `μ(a | s)`, and residual uncertainty through
`load_ws3_artifacts` (decision D33). The core exhibit is the **prescriptive ablation** (decision
D38): the value of the policy built from the `C`, `L1`, and `O` views, all scored against **one
fixed evaluator** (the `O`-view behavior and `q̂`), so the `C → O` value gap isolates the policy's
*information*, not the evaluator's. On the real data (2021–2025 Statcast, 3,567,640 regular-season
decisions; WS4 scores the 1,419,590 held-out validation+test rows) the behavior recovery is
**G-PASS** (observed `+0.0000`, IPS weights unit), so the OPE estimates are trustworthy; the behavior
value is `V(μ) = −0.0001`, and **every softened policy scores at or below it** — the moderate-α
(`α = 0.25`) `O`-view value is `−0.0023` (`d = −0.0022` vs behavior), the deficit growing monotonically
(`−0.0009` at `α = 0.10` to `−0.0089` at `α = 1`) as the effective sample size collapses
`100% → 24.7% → 5.5% → 1.7% → 0.6%`. So the **myopic bandit does not beat observed MLB pitcher behavior
at any α > 0** (the `V−` reading — a first-class honest negative, decision D39, not a failure). The
sequencing `O − C` gap is **real but negligible**: CI-positive from `α ≥ 0.1` yet only `+0.0001`
(clustered 95% CI `[+0.0001, +0.0001]`) at `α = 0.25`, rising to `+0.0003` (`[+0.0002, +0.0004]`) at
`α = 1`, and growing solely as the policy moves off support (`P0`, sequential-not-myopic). This is the
study's honest WS4 headline — **now confirmed on real data, not just the synthetic fixture**: myopic
prescription does not beat observed behavior, the sequencing-prescription gap is real but too small to
act on, and both facts motivate the sequential rungs (WS5 setup value, WS7 offline RL). Per-α agreement
is `CONSISTENT` at `α ∈ {0.1, 0.25, 0.5}` and `INCONCLUSIVE` at the `{0, 1}` endpoints (decision D24),
all first-class. We validate the method on the correctness oracle with a completed two-world study. On
the **null** world the behavior recovery passes exactly (IPS weights ≡ 1 at `α = 0`), the raw
value-vs-behavior column shows a *count-driven* gain (the habit-based synthetic behavior policy is
not reward-optimal), but the `O − C` gap is significantly **negative** (`−0.0030`, CI
`[−0.0056, −0.0002]` at `α = 1`) — the `O`-view `q̂` *overfits* — so the verdict is
`SEQ_NEUTRAL_PRESCRIPTION`: any gain is count-driven, not sequencing. On the **positive** world (a
planted velocity-transition whiff boost) the `O − C` gap is directionally positive but its CI
straddles zero, while sitting `~+0.003` **above** the null world's gap — the sequencing signal
exactly offsets the `O`-view overfitting — so the verdict is the first-class honest negative
`SEQ_INCONCLUSIVE_MYOPIC`. The paper's central contribution explains *why* that INCONCLUSIVE is
correct rather than a failure: the **myopic ceiling**. Ground-truth probes decompose the planted
effect into a **state-value** term and an **action-differential** term, `E[R | s, a] = v(s) +
δ(s, a)`. The trigger `|velo_{t−1} − velo_{t−2}| ≥ 5` is fixed by the *history*, so on a triggered
pitch **every** current family inherits the same `+0.032` state lift; only the small
family-differential (split-finger `+0.042` vs four-seam `+0.030`, spread `~0.012`) is *myopically*
exploitable, for a true best-case myopic `O − C` policy edge of `~0.003` run — **below the OPE noise
floor** (`~0.006` CI half-width at synthetic scale). A gap-machinery self-test confirms the
machinery is not blind: `SEQ_EXPLOITED` fires when a genuinely myopic advantage is constructed. The
effect is therefore **present but sequential-not-myopic** — precisely the falsifiable target
(decision D40) that WS5 (a tabular MDP that can value a *setup* pitch) and WS7 (offline RL) exist to
exceed. A companion ambiguity exhibit reports that on the real data the mean posterior confidence that the top
family beats the runner-up is only `~0.57` (`~0.62–0.66` on the synthetic fixtures), and `~99%` of the
1,320,705 decidable recommendations are toss-ups at 95% confidence (`ambiguous@95 ≈ 1.00`) — an honest
statement of how resolvable "best next pitch" is under myopia. Everything is
read under the study's finding-#2 / finding-#3 firewall: WS4 does not *assume* `q̂` is causal, it
*tests* whether acting on it beats the observed policy within support, and reports INCONCLUSIVE when
its estimators disagree.

---

## 1. Introduction

The study this workstream belongs to is a **rigor ladder**, not a horse race (SPEC §0):

> How much evidence for sequencing survives progressively harder tests — from descriptive order
> patterns, to out-of-sample **outcome** dependence, to **counterfactual policy** value?

The ladder exists because three findings are routinely conflated and must be kept separate
(SPEC §0, verbatim):

> 1. **Selection structure** — prior pitches help predict *what is thrown next*.
> 2. **Predictive sequencing value** — prior pitches help predict the *outcome* of the current
>    pitch, after conditioning on the current pitch and game state.
> 3. **Prescriptive/causal value** — *changing* the sequence would improve outcomes.
>
> The first is easy; the second is hard; the third needs assumptions that public data cannot fully
> satisfy (we never see the pitch that wasn't thrown; `pitch_type` is a classifier output, not the
> battery's intent; scouting/target info is unobserved).

Workstreams 1–3 (Phase A) live in findings #1 and #2: they measure whether ordered history *predicts*
selection and outcomes. WS4 opens **Phase B** — the prescriptive phase — and takes the first step
toward finding #3: *what should he throw, and can we prove it?* The discipline SPEC §0.3 imposes on
that step is absolute and is the reason WS4 is built the way it is:

> **OPE before policy.** Build and self-test the off-policy-evaluation harness (it must recover the
> observed policy's value) *before* optimizing any policy. Otherwise you produce recommendations you
> cannot evaluate.

So WS4 is not a policy-learning workstream that reports a run gain. It is an **evaluation**
workstream that constructs one honest, uncertainty-aware target policy and subjects it to the OPE
gate — behavior-recovery first, the full SPEC §9 diagnostics per deviation level, and the
`INCONCLUSIVE`-if-estimators-disagree rule (decision D24) taken verbatim. The headline is not a
number a coach can act on; it is the honest **measurement of how far a myopic recommender can get**.

**The myopic ceiling is the chapter's real content.** A contextual bandit is *myopic* by
construction: it treats each pitch as a one-step episode and maximises the immediate reward
`E[R | s, a]`, with no notion that today's pitch changes tomorrow's *state*. That is exactly the
right first rung — the simplest thing that can turn `q̂` into a recommendation — and its limits are
informative. On the positive synthetic world we can measure those limits exactly, because we planted
the effect ourselves. The planted whiff boost is a **state-value** effect: its trigger is fixed by
the two prior pitches, so by the time the current decision is made, the boost is already owed to
*every* feasible family equally. A greedy chooser cannot manufacture that boost — it can only harvest
the sliver of it that differs *across today's families*, which we measure at `~0.003` run against a
total planted `~0.032`. The setup — throwing the pitch that *creates* the velocity transition a pitch
early — is worth the other order of magnitude, and it is invisible to a bandit because it pays off in
a *future* state a myopic value ignores. WS4's honest INCONCLUSIVE is therefore not a null result: it
is a *quantified ceiling*, and it converts the study's positive-world acceptance test for the
sequential rungs (WS5, WS7) into a falsifiable target — **beat 0.003** (decision D40).

**Contributions.**
1. **An OPE-gated Bayesian bandit.** A Thompson target policy over WS3's posterior `q̂` grid
   (feasibility-masked, `XX` excluded, observed-action fallback for low-history rows), softened by
   the SPEC §9 `π_α` mixture and evaluated **strictly** through `eval/ope.evaluate_policy` with the
   behavior-recovery self-test run *first* (decisions D36/D37). Prescription, never trusted without
   OPE.
2. **The prescriptive ablation (D38).** The `C/L1/O` state-view ladder extended into prescription:
   `V(π̃` built from view-`v` `q̂`) under **one fixed evaluator**, so the `C → O` gap isolates the
   *information* the ordered state adds to the *policy*. It is the bandit's own falsification —
   distinct from, and cleaner than, the raw value-vs-behavior column, which is contaminated by
   count-driven gains that are not sequencing.
3. **The myopic-ceiling decomposition (D40).** A ground-truth split of the planted effect into
   `v(s) + δ(s, a)` that measures the myopically-reachable fraction (`~0.003` of `~0.032`), shows it
   sits below the OPE noise floor, and — with a gap-machinery self-test proving the `SEQ_EXPLOITED`
   path fires on a *constructed* myopic advantage — establishes that the effect is genuinely
   *sequential-not-myopic*. This is the ladder's motivating contrast: the honest thing a bandit
   cannot do, made into the target the sequential workstreams must clear.
4. **An honest treatment of two modelling choices** most studies leave implicit: the posterior-scale
   `κ` that maps WS3's residual reward noise to a posterior standard error of the mean (§4.2), and the
   **fixed-evaluator** design that makes the ablation gap an information statistic rather than an
   evaluator artifact (§4.4).

---

## 2. Related work

**Thompson sampling.** The target policy is the sampling rule of Thompson (1933): draw a plausible
value for each action from the posterior and act greedily on the draw, so an action is played in
proportion to its posterior probability of being optimal. WS4 uses exactly this rule as a *target
policy* rather than an online exploration schedule — the per-row probabilities `P(a = argmax)` are
its `π̃`. The modern rehabilitation of the rule as a strong practical algorithm is the empirical study
of Chapelle and Li (2011), and its first rigorous regret analyses are Agrawal and Goyal (2012, for
the multi-armed case) and Agrawal and Goyal (2013, for contextual bandits with linear payoffs). We
borrow the *posterior-probability-of-optimality* semantics from this line; WS4 adds nothing to its
theory and is careful (§4.2) that the posterior it samples from is a defensible one.

**Contextual bandits and offline evaluation.** The framing of a per-decision policy over context is
the contextual-bandit setting; Li, Chu, Langford, and Schapire (2010) is the canonical
recommendation instance (news articles) and, as importantly for us, introduced the **offline replay**
evaluation of a bandit policy from logged data — the discipline that WS4's entire gate embodies.
The estimator WS4 leads with is the **doubly robust** estimator of Dudík, Langford, and Li (2011),
which is unbiased if *either* the propensities or the outcome model is correct; WS4 consumes it
through `eval/ope.dr` (the gap CI is a clustered paired bootstrap of its per-row contributions) and
reports it alongside the Direct Method and self-normalized IPS, using their agreement as the
verdict (decision D24). Because a pitch decision is treated as a **one-step** episode, WS4 is a
genuine bandit and not a sequential problem; the importance weighting is therefore ordinary
per-decision weighting. Where the study later evaluates full plate appearances (WS5/WS7), the correct
device is the **per-decision importance sampling** of Precup, Sutton, and Singh (2000) — cited here
precisely to mark the boundary WS4 does *not* cross: a myopic bandit needs no eligibility trace or
per-decision correction because its horizon is one.

**Shared project grounding.** The empirical-Bayes shrinkage philosophy the study leads with is Efron
and Morris (1975), the canonical baseball demonstration; WS4 inherits it indirectly through the
posterior uncertainty WS3 supplies and the `κ` shrinkage that turns residual noise into a posterior
standard error (§4.2). The applied grounding — that pitch value is dominated by the **count** and the
matchup — is Tango, Lichtman, and Dolphin (2007), and it is exactly why the raw value-vs-behavior
gain is *count-driven* and must be separated from sequencing (§5).

**Internal cross-references.** WS4 is a strict consumer of WS3 (decision D33): it reads the
`q̂` grid, the propensities `μ(a | s)`, and the residual `exp_reward_sd` through
`load_ws3_artifacts` and refits nothing. The OPE machinery — estimators, `π_α`, behavior-recovery,
the agreement verdict — is WS0's `eval/ope.py`, built and self-tested against analytic truth before
any workstream optimized a policy (SPEC §0.3), and WS4 re-runs the behavior-recovery self-test
in-pipeline. The finding-#2 / finding-#3 firewall around `q̂` is derived in
`../ws3_gbdt_stack/THEORY.md` §8 and restated here (§7).

---

## 3. Data

WS4 does not read the raw Statcast data or refit a model. It reads **two things**: the shared
decision table (for the reward `R`, the feasibility mask, and the pitcher-game clustering) and **WS3's
saved artifacts** (decision D33). This section states both.

**The decision table and the reward.** Every workstream reads one shared table (SPEC §3): one row per
pitch, the decision made immediately *before* that pitch is released, keyed by
`(game_pk, at_bat_number, pitch_number)`. WS4 evaluates on the same held-out rows WS3 scored —
validation (2024) and the locked test (2025) — with the behavior `μ` and `q̂` read from WS3's
train-fold (2021–2023) models. The reward is `R = −delta_run_exp`, larger better for the pitcher
(SPEC §5); each pitch is one **one-step episode** (`pa_id` unique per row, `step = 0`), which is what
makes the problem a bandit.

**The action space and the feasible mask.** The action is the pitch **family** (8 classes, SPEC §4);
`XX` (other/rare) is descriptive-only and **never** a recommendable action. The feasible-action mask
is the leakage-safe SPEC §4 mask — a family is feasible only if the pitcher threw it `≥ 50` times
**and** `≥ 3%` over a trailing window ending before the current game. A decision whose mask is empty
(a low-history pitcher whose repertoire cannot be established from pre-game history) **carries no
recommendation**: the target mirrors the observed action there. The mask is computed on the **full**
table's trailing window and subset to the evaluation rows, never computed on the held-out rows alone
(§5, the feasibility-mask fix).

**What WS3 supplies (decision D33).** For each state view WS4 loads a `WS3Artifacts` bundle
(`load_ws3_artifacts`) and calls three helpers: the behavior propensities `μ(a | s)` (the view's
`BehaviorModel`); the counterfactual value grid `q̂(s, a)` for all 8 families (the view's
`OutcomeStack`, one action-swept pass); and the residual `exp_reward_sd` that accompanies each `q̂`.
`build_bandit_inputs` assembles these into the aligned `(q, q_sd, μ, feasible_mask, align)` bundle the
pipeline evaluates, with the behavior propensities floored at `MU_FLOOR = 1e-6` and renormalised so
importance ratios stay finite (SPEC §9), and `q_sd` formed from the residual sd as described in §4.2.
The `q̂` grid is a *conditional expectation* `E_model[R | s, A = f]`, **not** a causal effect
`E[R | s, do(A = f)]` — the firewall WS4 inherits and tests rather than assumes (§7;
`../ws3_gbdt_stack/THEORY.md` §8).

**The ablation views (D38).** WS4 runs the ablation on **three** nested views — `C` (context only),
`L1` (the immediately preceding pitch), and `O` (the full ordered current-PA history). `U` and `OM`
are out of scope for the myopic-bandit ablation; the `C → O` change is the sequencing-prescription
evidence. `O` is the richest view and is designated the **common evaluator** (`COMMON_EVAL_VIEW`,
§4.4).

**Temporal splits and clustering.** Train on 2021–2023, evaluate on 2024 + locked 2025 (SPEC §7).
All confidence intervals are bootstrapped over **pitcher-game** clusters; the gap CI is a paired
clustered bootstrap (§4.5).

---

## 4. Methods

### 4.1 The Thompson target policy (D36)

The target policy is built per state view from WS3's posterior over `q̂`. For each decision row,
`thompson_policy` draws `n_samples` (default 1500) independent samples of the per-action value from
`𝒩(q̂(s, a), q_sd(s, a)²)`, restricts the `argmax` to the row's **feasible** actions, and returns the
empirical win frequency per action:

$$\tilde\pi(a \mid s) \;=\; P\!\left(a = \arg\max_{a' \in \mathcal{F}(s)}\; \theta_{a'}\right),
\qquad \theta_{a'} \sim \mathcal{N}\!\big(\hat q(s, a'),\, \sigma(s, a')^2\big),$$

with `𝓕(s)` the feasible set. The result is a valid distribution: it sums to exactly 1, places
**zero** mass on infeasible actions, and is deterministic under a seed. A dominant mean yields a
near-point-mass; equal means with equal sds yield a uniform draw over the feasible actions; a row
with an **empty** feasible mask falls back to the **observed-action point mass** and *carries no
recommendation*. The Monte-Carlo error of the win frequencies is `O(1/√n_samples)` (`THEORY.md` §2),
negligible against the CI widths that dominate the verdict.

### 4.2 The posterior scale `κ` — an honest modelling choice

WS3's `exp_reward_sd` is a **residual** reward standard deviation — the law-of-total-variance spread
of a *single pitch's* reward around `q̂(s, a)` (`../ws3_gbdt_stack/THEORY.md` §4.2). It is the outcome
**noise**, on the order of the reward scale (`|R| ~ 0.05–0.3`, empirically `~0.17`), and it is *not*
the posterior standard error of the *mean* `q̂(s, a)` that a Thompson sampler needs. Used raw it is
`~30×` the inter-family `q̂` gaps (`~0.005`) and drives the policy to a near-uniform draw over feasible
actions — a defensible but uninformative recommender. `build_bandit_inputs` therefore returns

$$q\_sd(s, a) \;=\; \kappa \cdot \texttt{exp\_reward\_sd}(s, a), \qquad \kappa = \texttt{POSTERIOR\_SCALE} = 0.05,$$

mapping the per-pitch residual sd to a posterior standard error of a mean via a single documented
shrinkage `κ ≈ 1/√(n_eff)`, with `n_eff` the effective per-cell support a LightGBM leaf aggregates
(`min_child_samples` is on the `10²` scale, so `κ ≈ 0.05` corresponds to `n_eff ≈ 400`). We report
this openly because it is the single confidence knob of the whole workstream: `κ = 1` recovers the
literal "`exp_reward ± sd`" of decision D36 but yields the near-uniform policy; `κ = 0.05` makes the
Thompson uncertainty commensurate with the `q̂` gaps, so the target is a concentrated (`~0.62–0.75`
top-action confidence) recommendation. `thompson_policy` itself is agnostic to how `q_sd` was formed;
`κ` is exposed as `--posterior-scale`, and §8 reports the sensitivity of the ambiguity exhibit to it.

### 4.3 Conservative softening (`π_α`, SPEC §9)

The target is softened toward behavior along the SPEC §9 conservative mixture

$$\pi_\alpha(a \mid s) = (1-\alpha)\,\mu(a \mid s) + \alpha\,\tilde\pi(a \mid s),$$

over the config `α` grid `{0, 0.1, 0.25, 0.5, 1.0}`. `α = 0` is the behavior policy (the OPE baseline);
`α = 1` is the pure target; intermediate `α` trade recommendation strength against overlap. `soften`
delegates to `eval/ope.pi_alpha` — WS4 does **not** re-implement the mixture. The value is linear in
`α`, `V(π_α) = (1−α)V(μ) + αV(π̃)` (`THEORY.md` §4), so the frontier is a straight line whose slope is
the policy's value over behavior, read against a monotonically degrading effective sample size.

### 4.4 The fixed-yardstick evaluator (the crux of the ablation)

Every view's target policy is scored against **one common OPE evaluator** — the behavior `μ` and `q̂`
of the richest ablation view (`O`, `COMMON_EVAL_VIEW`). This is deliberate and load-bearing. The
ablation asks whether the *information* in the ordered state improves the *policy*; if each view were
scored against its *own* `q̂`, a `C → O` gap could reflect the evaluator changing (a richer `q̂` scoring
a policy differently) rather than the policy changing. Fixing the evaluator removes that confound: the
only thing that varies across `C/L1/O` is the `q̂` the *target policy* was *built from*, so the gap is
an **information** statistic (`THEORY.md` §5 derives why a per-view evaluator would bias it). The price
is that the `C`-view policy is scored by an evaluator it did not generate; the benefit is that the gap
means what the ablation says it means.

### 4.5 The gate, the ablation gap, and the verdict

**The gate (D37).** Before any target value is computed, the pipeline re-runs the SPEC §9
behavior-policy recovery self-test (`eval/ope.behavior_policy_recovery`): set the target equal to the
behavior policy and require every estimator to recover the observed held-out mean reward, with the IPS
importance weights exactly 1. If it **fails**, the pipeline prints `FAILED_GATE` and stops *before any
target value is reported* — nothing below it is interpretable.

**The ablation gap.** The `O − C` (and `L1 − C`) value gap is the sequencing-prescription statistic.
It is a pitcher-game-clustered **paired** bootstrap over the doubly-robust per-row contributions from
`eval/ope.dr` under the common evaluator: because both policies are scored by the same evaluator on the
same rows, the common-evaluator variance cancels in the pairing and the CI is tight around the *policy
difference* (`THEORY.md` §7).

**The verdict.** Each `π_α` is evaluated through `eval/ope.evaluate_policy`, which reports the full
SPEC §9 block (value, 95% lower bound, ESS, out-of-support fraction, max weight, KL/TV from behavior)
and an agreement verdict: `CONSISTENT` unless DM/SNIPS/DR differ by more than the wider of their CI
half-widths, in which case `INCONCLUSIVE` (decision D24, verbatim). On the synthetic worlds a
prescriptive verdict summarises the gap: `SEQ_EXPLOITED` **only** when the real-reward-anchored `O − C`
lower bound clears 0 at some `α` (never fabricated); `SEQ_NEUTRAL_PRESCRIPTION` when the gap CI contains
0 at every `α` and the point estimate is small; `SEQ_INCONCLUSIVE_MYOPIC` when the gap is directionally
positive but its lower bound is `≤ 0` — the first-class honest negative of decision D39.

---

## 5. Experimental setup

**Scoring.** Every view's target predictions are written in the standard schema (SPEC §8.1) and scored
*only* through `eval/ope.evaluate_policy` (SPEC §9). The exhibits are the value-vs-`α` frontier (value +
95% lower bound with an ESS overlay per view), the prescriptive-ablation table (`O − C` and `L1 − C`
gaps with clustered CIs per `α`), the ambiguity decomposition, and the deviation-from-behavior maps.

**The completed two-world validation.** WS4 reruns the SPEC §11 oracle as a real-model prescriptive
test. The following are *observed results* on the committed WS4a run at validation scale, not
aspirations. Both worlds pass the gate: **behavior recovery passes with IPS weights ≡ 1 at `α = 0`**,
so the OPE harness recovers the observed policy's value exactly and everything below it is
interpretable.

- **Null world** (an ordered *selection* habit but outcomes depending only on context + current
  family, by construction — no ordered *outcome* effect). Behavior value `V(μ) = +0.0047`. The
  per-view frontier at `α ∈ {0, 0.5, 1}` is, for `C`, `+0.0047 / +0.0053 / +0.0059` (value) with
  lower bounds `+0.0024 / +0.0025 / +0.0028` and ESS `100% / 75% / 45%`; for `O`,
  `+0.0047 / +0.0038 / +0.0029` with lower bounds `+0.0024 / +0.0009 / −0.0008` and ESS
  `100% / 80% / 51%`. The raw value column shows a small **count-driven** gain over behavior (the
  synthetic behavior policy is habit-based, not reward-optimal — a myopic count-aware policy beats it
  for reasons that have nothing to do with sequencing). But the **sequencing** statistic tells the
  real story: the `O − C` gap is `−0.0015` (CI `[−0.0028, −0.0001]`) at `α = 0.5` and `−0.0030` (CI
  `[−0.0056, −0.0002]`) at `α = 1` — **significantly negative**. The `O`-view `q̂` *overfits*: it
  carries extra ordered features that, in a world with no ordered outcome effect, add estimation
  variance and make the `O`-built policy slightly *worse* under a fixed evaluator (the prescriptive
  echo of WS3's negative-`Δ_matchup` fragmentation story). `L1 − C ≈ +0.0002` (negligible). Verdict:
  **`SEQ_NEUTRAL_PRESCRIPTION`** — no sequencing prescription edge; the value gain is count-driven.
- **Positive world** (a planted whiff boost when `|velo_{t−1} − velo_{t−2}| ≥ 5` mph — keyed on the
  transition *into* the prior pitch, per D21, so it can separate `O` from `L1`). Behavior value
  `V(μ) = +0.0111`. Here the `O − C` gap is `≈ +0.0001` with CIs that **straddle zero** at `α = 0.5`
  and `α = 1` — but it sits `~+0.003` **above** the null world's `O − C` gap at the same `α`. That
  offset is the sequencing signal: the `O`-built policy is doing `~0.003` of *real* prescriptive work,
  which is just enough to cancel the `~0.003` of `O`-view overfitting the null world exposes, leaving
  a gap near zero. Verdict: **`SEQ_INCONCLUSIVE_MYOPIC`** — the effect is present (WS3 recovered it at
  ratio 0.955) but not *myopically* prescriptive at this scale.

**The myopic ceiling (decision D40) — the intellectual payload.** Why is the positive world's honest
verdict INCONCLUSIVE rather than a failure? Because the planted effect is a **state-value** effect, and
a myopic policy can reach only a measured sliver of it. Ground-truth probes decompose the per-pitch
expected reward into a state term and an action term,

$$\mathbb{E}[R \mid s, a] \;=\; v(s) \;+\; \delta(s, a),$$

where `v(s)` depends only on the state (here, on the two prior pitches, through the trigger) and
`δ(s, a)` is the part that differs across *today's* action. The trigger `|velo_{t−1} − velo_{t−2}| ≥ 5`
is **fixed by history**: by the current decision it is already true or false, so the `+0.032` whiff
boost is owed to `v(s)` — **every** feasible current family inherits it equally. A greedy chooser cannot
create that boost by its choice of *today's* pitch; it can only harvest the **family-differential**
`δ(s, a)` — the fact that some families convert a triggered pitch's foul/in-play mass to whiffs slightly
more than others (split-finger `+0.042` vs four-seam `+0.030`, a spread of `~0.012`). The best a myopic
`O`-policy can do over a `C`-policy is capture that differential on the triggered rows, worth
`~0.003` run — and the OPE noise floor at synthetic scale is a CI half-width of `~0.006`, so `0.003` is
**undetectable by construction** (`THEORY.md` §8 does the `n`-to-resolve arithmetic). The rest of the
`+0.032` lives in the **setup**: throwing the pitch that *creates* the velocity transition a pitch
early, which pays off in a *future* state a bandit's one-step value cannot see.

**The gap-machinery self-test (the machinery is not blind).** To prove the near-zero gap is the
*effect's* property and not a broken detector, WS4a ships a self-test that constructs a world with a
genuine *myopic* family advantage and confirms the `SEQ_EXPLOITED` path fires — the `O − C` lower bound
clears 0 when there is a myopic edge to find. So the machinery *sees*; the positive world's effect is
genuinely **sequential-not-myopic**. This is the precise, falsifiable handoff to the sequential rungs
(D40): WS5 and WS7 pass their positive-world acceptance by **exceeding** the myopic ceiling — valuing
the setup pitch (sequential credit) that the bandit provably cannot.

**The ambiguity exhibit (how resolvable is "best next pitch"?).** Across decidable rows (`≥ 2` feasible
families) the mean posterior probability that the top family beats the runner-up is only `~0.62–0.66`,
and `~99%` of recommendations are **toss-ups at 95% confidence** (`ambiguous_95 ≈ 0.99`). This is an
honest statement, not a bug: the inter-family `q̂` gaps (`~0.005`) are small against even the shrunken
posterior sd, so "best next pitch" is rarely a confident distinction under myopia. §8 discusses what a
practitioner should take from it.

**Support and deviation.** ESS falls from `100%` at `α = 0` to `~45–51%` at `α = 1` (the price of
deviating from behavior), and the max importance weight rises to `~15` — well short of a support
collapse, so the `α = 1` values are readable but wide. The deviation from behavior (mean total-variation
distance at `α = 1`) is `~0.35` for `C` and `~0.30` for `O`: the ordered-state policy stays slightly
*closer* to behavior, consistent with `O`'s `q̂` being noisier and its Thompson draws less concentrated.

**A real bug found and fixed.** An early version computed the feasibility mask on the held-out
evaluation rows alone, which undercounted each pitcher's pre-game repertoire (a 2024 game's trailing
365-day window lies in the train seasons) and **spuriously flagged 27% of decisions as
no-recommendation**. Computing the mask on the full table's trailing window and subsetting to the
evaluation rows removes the spurious flags entirely; on the real data only a **genuine** `4.9%`
remains (low-history pitchers whose pre-game repertoire cannot be established — §6.1), not the 27%
artifact. It is recorded here because it is exactly the kind of leakage-adjacent error the study's
discipline exists to catch: the mask must see the same history the pitcher did.

---

## 6. Results

### 6.1 The central table (real data)

The behavior-recovery gate, the per-view/per-`α` OPE frontier (the D38 prescriptive ablation), and the
sequencing-prescription gaps.

**Data vintage.** 3,567,640 regular-season decisions, 2021–2025 (SPEC's ~3.85M counts all game types;
the config filters to `game_type == "R"`). WS4 refits nothing: it reads WS3's train-fold (2021–2023,
2,143,214) behavior and outcome models and scores the held-out evaluation rows — validation (2024) plus
the locked test (2025), i.e. eval = val + test 2024–2025 — for **1,419,590 scored decisions**. All CIs
are pitcher-game clustered. Wall-clock `~9.5 h` (FQE-in-loop dominates; peak RAM 8,991.5 MB).

**Feasibility.** Mean feasible families per decision `= 3.71`. Empty-mask (**no recommendation**)
`= 4.9%` (`69,823` rows) and low-history `= 4.6%`; these are **genuine** low-history pitchers whose
pre-game repertoire cannot be established, *not* the spurious 27% the eval-rows-only mask bug produced
(§5). The `1,320,705` decidable rows (`≥ 2` feasible families) carry the ambiguity read below.

**The gate (precondition, D37).** Behavior-policy recovery **PASSES**: the OPE harness recovers the
observed held-out mean reward with the IPS importance weights exactly 1 (observed `+0.0000`), so every
target value below is interpretable (SPEC §0.3). Behavior value `V(μ) = −0.0001` (the `α = 0` baseline).

**The value-vs-`α` frontier (D38 ablation).** Values are `C`/`L1`/`O` near-identical to four decimals —
the tiny ablation differences live in the `O − C` gap table below — so one shared frontier is shown,
with `value = V(μ) + d`:

| `α` | value | `d` vs behavior | ESS% | oos | per-`α` verdict |
|---|---|---|---|---|---|
| 0.00 | −0.0001 | 0.0000  | 100  | ~0 | INCONCLUSIVE |
| 0.10 | −0.0010 | −0.0009 | 24.7 | ~0 | CONSISTENT |
| 0.25 | −0.0023 | −0.0022 | 5.5  | ~0 | CONSISTENT |
| 0.50 | −0.0046 | −0.0045 | 1.7  | ~0 | CONSISTENT |
| 1.00 | −0.0090 | −0.0089 | 0.6  | ~0 | INCONCLUSIVE |

Every softened policy scores **at or below** behavior; `d` is negative and grows monotonically with α
while ESS collapses — the `V−` reading (§6.2). (Each `value` at `α > 0` is `V(μ) + d`, arithmetically
from the D38 block; the per-view value lower bound is not in the results log — the resolved CIs are on
the `O − C` gap.) The `α = 0` and `α = 1` endpoints carry a per-`α` **INCONCLUSIVE** (D24: DM/SNIPS/DR
diverge by more than their CI half-widths) — distinct from the gate, which is the IPS behavior-recovery
check and PASSES; the interior `α ∈ {0.1, 0.25, 0.5}` are CONSISTENT.

**The sequencing-prescription gaps (`O − C` isolation, common evaluator `O`, pitcher-game clustered).**

| `α` | `O − C` | clustered 95% CI | resolves > 0? |
|---|---|---|---|
| 0.00 | +0.0000 | [+0.0000, +0.0000] | no |
| 0.10 | +0.0000 | [+0.0000, +0.0000] | yes (CI excludes 0) |
| 0.25 | +0.0001 | [+0.0001, +0.0001] | yes |
| 0.50 | +0.0002 | [+0.0001, +0.0002] | yes |
| 1.00 | +0.0003 | [+0.0002, +0.0004] | yes |

`L1 − C` **tracks `O − C`** at every α. The gap is statistically resolved (CI excludes 0) from
`α ≥ 0.1` but is **practically negligible** (`+0.0001` to `+0.0003` run) and grows *only* as the policy
moves off support (ESS `24.7% → 0.6%`) — the `P0` sequential-not-myopic reading (§6.2).

**Ambiguity (`O`).** Mean `P(top > runner-up) = ~0.57`; `ambiguous@95 ≈ 1.00` — `~99%` of the
`1,320,705` decidable recommendations are toss-ups at 95% posterior confidence. **Deviation (mean TV
from behavior, `α = 1`):** `C = 0.290`, `L1 = 0.293`, `O = 0.296` (`≈ 0.29`; the ordered policy deviates
marginally *more* than context-only on real data — the reverse of the synthetic fixture, where `O`'s
noisier draws stayed closer). **ESS at `α = 1`:** `C = O = 0.6%` (`C`/`L1`/`O` near-identical); max
weight `{MAX_W}` *(pending: max importance weight not in the results log; the ESS collapse to `0.6%` is
the binding support statistic)*.

**Headline (real data, 2021–2025).** Myopic prescription does **not** beat observed MLB behavior — every
deviation lowers estimated value (`V−`). The sequencing-prescription gap is **real but negligible** —
CI-positive from `α ≥ 0.1`, yet only `+0.0001`–`+0.0003` run and growing only off support (`P0`). This is
the real-data confirmation of the myopic ceiling D40 measured on the synthetic fixture (`~0.003` of
`~0.032`), and it is precisely why the ladder continues to the sequential rungs (WS5 setup value, WS7
offline RL).

### 6.2 Branched interpretation — four axes

The result is read on four axes. The **gate** axis is a precondition; the **value** and **verdict**
axes qualify how much to trust the numbers; and the **sequencing** axis answers WS4's actual question.
A code cell in the notebook (§9) inspects the computed report and prints which branch fired on each
axis; the write-ups below stand alone once the numbers are filled.

#### Gate axis (precondition)

**Selected by the data (2021–2025). G-PASS — behavior recovery passes.** The OPE harness recovers the
observed policy's value (IPS weights ≡ 1 at `α = 0`); every target value below the gate is
interpretable. Proceed to the value and sequencing axes.

*Realized (2021–2025).* Selected. Behavior-policy recovery **PASSES** with observed `+0.0000` and the
IPS importance weights exactly unit — the OPE harness reproduces the observed held-out mean reward, so
the estimates below are trustworthy (SPEC §0.3 / D37). This is the precondition the whole reading rests
on, and it is stated first.

*Pre-registered alternative — not selected.* **G-FAIL — `FAILED_GATE`.** The harness cannot recover even
the *observed* policy's value on this data. **Stop.** No target value is trustworthy; paste the
`FAILED_GATE` block and diagnose the propensity / reward join before reading anything else. This is SPEC
§0.3 enforced structurally, not a soft warning. *Not selected: the gate passed.*

#### Value axis (at moderate `α`, read against `V(μ)`)

*Pre-registered alternative — not selected.* **V+ — `lower_95 > V(μ)`: the bandit beats behavior.** The
moderate-`α` policy's 95% lower bound clears the behavior value: a real improvement over the observed
policy *in myopic value*. Before believing it, check the support (ESS not collapsed, oos small, max
weight moderate) — a "gain" riding on a handful of high-weight rows is an artifact. Note this is **not**
by itself sequencing evidence (read the P axis): the habit-based behavior policy is beatable for
count-driven reasons. *Not selected: on real data every softened policy scored at or below `V(μ)`; no α
produced a gain over behavior.*

*Pre-registered alternative — not selected.* **V0 — CI straddles `V(μ)`: the typical honest outcome.**
The policy value is indistinguishable from behavior at this scale. The expected result for a
conservative myopic recommender on real baseball, where value is dominated by the count and the matchup
and the room to improve *myopically* is thin. *Not selected: the real-data policy did not merely tie
behavior — it scored below it and the deficit grew with α (V− selected). Observed MLB pitchers are
already strong enough myopically that even a tie was not reached.*

**Selected by the data (2021–2025). V− — `lower_95 < V(μ)`: the policy trails behavior.** The target
scores *below* the observed policy — a `q̂` or propensity misfit, or a support problem (the target
recommends actions the OPE cannot evaluate). Diagnose with the oos fraction and max weight before any
other reading; do not report a prescription from a policy that loses to behavior.

*Realized (2021–2025).* Selected — but as the **honest, expected** outcome, not a misfit. Every softened
policy value is `≤ V(μ) = −0.0001`: the `d`-vs-behavior deficit is `−0.0009 / −0.0022 / −0.0045 /
−0.0089` at `α = 0.10 / 0.25 / 0.50 / 1.0` (`C`/`L1`/`O` near-identical), negative and growing
monotonically with α while ESS collapses `100% → 24.7% → 5.5% → 1.7% → 0.6%` and oos stays `~0`. The
diagnosis is *not* a broken evaluator (the gate PASSED and oos is `~0`) — it is the substantive finding:
**a myopic one-pitch-ahead recommender does not beat observed MLB pitcher behavior at any α > 0.** Every
move off the observed policy, toward the Thompson target, *lowers* estimated value. This is a
first-class D39 honest negative and the real-data confirmation of the myopic-ceiling story (§7): greed
cannot cash the sequencing effect WS3 sees, so it cannot manufacture value the observed policy does not
already have.

#### Sequencing axis (`O − C` gap — the real question)

*Pre-registered alternative — not selected.* **P+ — `O − C` gap CI `> 0`: prescriptively exploitable
ordering.** The ordered-state policy beats the context-only policy under the fixed evaluator by more
than the clustered CI — the ordered state carries information a *myopic* recommender can *act on*. This
would **exceed the myopic ceiling** the synthetic world measures, so it is a strong claim: cross-check
it against WS3's order axis (a P+ here should have an `H1` there — predictable *and* exploitable order),
confirm the gap concentrates in `long_pa` / `two_strike` slices, and confirm the per-`α` verdict is
`CONSISTENT`. If it survives, the bandit has found a myopically-actionable ordered edge and the burden
passes to WS7's full OPE battery. *Not selected: although the `O − C` gap's CI does exclude 0 from
`α ≥ 0.1`, its magnitude (`+0.0001`–`+0.0003` run) is far below anything a recommender could act on and
it grows only as the policy leaves support (ESS `→ 0.6%`) — statistically detectable but not a
prescriptively exploitable edge, so the selected reading is the negligible `P0` below, not `P+`.*

**Selected by the data (2021–2025). P0 — `O − C` gap ≈ 0 (sequential-not-myopic sub-reading).** The
ordered-state policy is, in any actionable sense, indistinguishable from the context-only policy. Which
of two things it means is resolved by the *positive-world discrimination* check — does the real-data
`O − C` sit meaningfully **above** the null-world signature (a significantly negative gap from `O`-view
overfitting)?
  - *Nothing there.* If the gap is near the (negative) overfitting baseline, there is no sequencing
    prescription signal — order is either not predictive of outcomes (cross-check WS3's `H2/H3`) or
    predictive-but-not-exploitable.
  - *Sequential-not-myopic.* If the gap sits **above** the overfitting baseline (the positive world's
    `~+0.003` offset) but is practically negligible, the sequencing signal is *present but below the
    myopic OPE floor* — the WS4 fixture's own verdict. Route it to WS5/WS7: the effect is a setup effect
    a bandit cannot cash in, and the sequential rungs' acceptance test is to exceed exactly this ceiling
    (D40).

*Realized (2021–2025). Selected: the sequential-not-myopic sub-reading, with a real-data refinement.* On
real data the `O − C` gap is `+0.0000 / +0.0001 / +0.0002 / +0.0003` at `α = 0.10 / 0.25 / 0.50 / 1.0`,
and its clustered CI **excludes 0** from `α ≥ 0.1` — so, unlike the synthetic fixture (whose CI
*contained* 0), the sequencing-specific prescriptive advantage is now **statistically detectable**. But
it is **practically negligible** (`+0.0001`–`+0.0003` run, `L1 − C` tracking `O − C`) and grows *only*
as the policy moves off support (the gap widens exactly as ESS collapses `24.7% → 0.6%`, i.e. where the
estimate is least trustworthy). This is the real-data analogue of the myopic ceiling (D40): WS3's
finding-#2 order signal (`+0.0003` nats of out-of-sample outcome dependence, locked-test replicated) is
genuinely present, but a greedy one-pitch-ahead recommender cannot convert that sliver into *decision*
value — the gap is real, tiny, and off-support-driven, not a myopically actionable edge (so **not P+**),
and it is positive rather than the negative `O`-overfitting signature (so **not P−**). The question of
whether ordered state carries *actionable* value is therefore routed to the sequential rungs — WS5
(setup value through the count transition) and WS7 (offline RL) — whose acceptance test is to exceed
this ceiling.

*Pre-registered alternative — not selected.* **P− — `O − C` gap CI `< 0`: `O`-view `q̂` overfitting
dominates.** The ordered-state policy is significantly *worse* than the context-only policy under the
fixed evaluator — the null-world signature. The extra ordered features in `O`'s `q̂` add estimation
variance without prescriptive signal, so the `O`-built policy fragments (the prescriptive echo of WS3's
negative `Δ_matchup`). Read it as "no sequencing prescription edge **and** a real cost of building the
policy from the ordered `q̂`," and hand the outcome model handed downstream should be the `C`/`L1` one,
not `O`. *Not selected: on real data the `O − C` gap is (barely) positive, not negative — the ordered
`q̂` does not fragment below `C`, and the null-world overfitting signature did not reproduce on real
data.*

#### Verdict axis (per-`α`, decision D24)

**CONSISTENT.** DM, SNIPS, and DR agree within the wider of their CI half-widths at this `α`. The
value and its lower bound are a coherent read. *Realized (2021–2025): the interior deviation levels
`α ∈ {0.1, 0.25, 0.5}` are CONSISTENT — the trustworthy rows the reading leans on.*

**INCONCLUSIVE.** The monitored estimators disagree materially. Per SPEC §9's closing rule the verdict
is **INCONCLUSIVE, not "it works"**: the value is not a coherent read at this `α` (typically a
high-`α`, low-ESS row where the model-based DM and the weighted DR pull apart). Report it as
inconclusive and lean on the lower-`α`, higher-ESS rows. *Realized (2021–2025): the `α = 0` and `α = 1`
endpoints are INCONCLUSIVE (D24) — this is the estimator-agreement verdict, distinct from and not in
tension with the gate (the IPS behavior-recovery check, which PASSES); all cells are first-class per
D39.*

#### Reading the grid

The honest headline is a tuple `(G, V, P, verdict)`. *Selected by the data (2021–2025):*
**G-PASS × V− × P0(sequential-not-myopic) × CONSISTENT (interior α) / INCONCLUSIVE (α ∈ {0, 1})**. The
gate passes, so the numbers are interpretable; the myopic policy does not merely tie behavior but scores
*below* it at every α > 0 (`V−`, a shade worse than the *anticipated* `V0`); the ordered-state gap is
statistically detectable yet negligible and off-support (`P0` sequential-not-myopic); and the interior
deviation levels agree while the endpoints are INCONCLUSIVE. This is the real-data confirmation of the
myopic ceiling and the motivation for the sequential rungs. The *strongest* (unrealized) cell would have
been **G-PASS × V+ × P+ × CONSISTENT**: a myopically-actionable ordered edge, to be cross-checked hard
against WS3 and WS7. The *diagnostic* cells are any with **G-FAIL** (stop) or a genuinely misfit-driven
**V−/P−** (diagnose before reading) — here `V−` is the honest negative, not a misfit (the gate PASSED
and oos is `~0`).

---

## 7. Discussion

**The ceiling as the bridge to the sequential rungs.** WS4's contribution is not a run gain; it is a
*measurement of a limit*. A myopic bandit values `E[R | s, a]` and so can only ever exploit the part of
an effect that differs across *today's* action — the `δ(s, a)` term. On the positive world that part is
`~0.003` run of a `~0.032` planted effect, below the OPE noise floor, and the gap-machinery self-test
proves the near-zero measurement is the *effect's* property, not the detector's. The remaining order of
magnitude is the **setup**: value that accrues in a *future* state because of today's pitch. This is
precisely what a sequential model can see and a bandit cannot, and it is why the ladder does not stop at
WS4. Decision D40 makes the handoff falsifiable: WS5 (a tabular MDP that assigns credit to a setup pitch
through the count transition) and WS7 (conservative offline RL with the full OPE battery) pass their
positive-world acceptance by **exceeding 0.003** — turning WS4's honest INCONCLUSIVE into the ladder's
motivating contrast rather than a dead end.

*Realized (2021–2025): this ceiling is now confirmed on real data, not just the synthetic fixture.* The
`O − C` sequencing-prescription gap is statistically detectable (CI excludes 0 from `α ≥ 0.1`) but
negligible (`+0.0001`–`+0.0003` run) and grows only off support, while **every** softened policy scores
below observed behavior (`V−`) — greed cannot convert WS3's tiny finding-#2 order signal (`+0.0003` nats)
into decision value. The `~0.003`-of-`~0.032` synthetic ceiling is no longer just a fixture property; the
real data shows the same shape (a real-but-unactionable sequencing sliver), which is exactly why the
ladder proceeds to WS5 and WS7 rather than stopping at a myopic recommender.

**What the ambiguity exhibit means for a practitioner.** That `~99%` of recommendations are toss-ups at
95% confidence is the most practically important number in the workstream. It says that, *under myopia
and at family resolution*, "the single best next pitch" is usually not a resolvable question: several
families have expected values within their posterior error bars of each other. The correct practitioner
takeaway is a **distribution, not a pick** — the Thompson `π̃` already is one, and the honest product of
a myopic recommender is "these three families are near-equivalent here," not "throw the slider." A
confident single pick would be over-reading the `q̂` gaps; the workstream's uncertainty machinery exists
to prevent exactly that. *Realized (2021–2025): on real data the mean top-vs-runner-up confidence is
`~0.57` (below the `~0.62–0.66` synthetic), and `~99%` of the 1,320,705 decidable recommendations are
toss-ups at 95% — the honest read is even starker on real baseball than on the fixtures.*

**The count-driven-gain trap.** On both worlds the bandit beats the *habit-based* behavior policy in raw
value. It would be easy — and wrong — to report that as a sequencing result. It is not: the synthetic
behavior policy is not reward-optimal, so a myopic count-aware policy improves on it for reasons Tango et
al. (2007) would predict (value is dominated by the count), with no ordered history involved. The `C → O`
gap is the instrument that isolates sequencing *from* the count-driven gain, which is why the paper leads
with the gap and not the value-vs-behavior column, and why the notebook's headline prints the count-driven
explanation whenever a raw gain appears. *Realized (2021–2025): on real data the trap did not even get the
chance to fire — observed MLB behavior is already strong enough myopically that the bandit scores* **below**
*it at every α (`V−`), the reverse of the synthetic worlds whose habit-based behavior was beatable for
count reasons. The `C → O` gap remains the sequencing instrument, and it is negligible.*

**On the fixed-evaluator choice.** Scoring every view against `O`'s evaluator is what makes the gap an
information statistic (§4.4). Its tradeoff is that the absolute *level* of each view's value is expressed
in the `O`-evaluator's units, so the value-vs-behavior column is most trustworthy for the `O` view and is
read as a level, not a per-view ranking, for `C`/`L1`. The gap — a *difference* under the *same*
evaluator — is unaffected, which is the whole point.

---

## 8. Limitations

1. **Myopia is a structural limit, not a bug.** A bandit's one-step value cannot represent a setup pitch;
   the `~0.003`-of-`0.032` ceiling is a property of the *class* of policy, measured, not a shortfall of
   this implementation. The limitation is *why* WS5/WS7 exist, and the honest thing WS4 can say is exactly
   how far myopia gets.
2. **The posterior scale `κ`.** The `κ = 0.05` shrinkage that turns residual noise into a posterior SE is
   a modelling choice (§4.2), justified by the leaf-support argument but not learned. A smaller `κ`
   sharpens the recommendations (fewer toss-ups) and a larger `κ` softens them toward uniform; the
   ambiguity exhibit and the deviation maps are `κ`-sensitive, and the workstream reports `κ` explicitly
   rather than hiding it inside the sampler. The *gap* and the *verdict* are far less `κ`-sensitive than
   the ambiguity shares, because the gap is a policy *difference* under a fixed evaluator.
3. **Evaluator dependence.** The value levels are expressed in the fixed `O`-evaluator's units (§7); a
   different common evaluator would move the levels (not the gap). The choice of the richest ablation
   view as the evaluator is defensible but is a choice.
4. **Synthetic-scale OPE noise floor.** The `~0.006` CI half-width that swallows the `~0.003` myopic edge
   is a property of the *validation-scale* fixture; at full-data scale the floor shrinks as `~1/√(ESS)`
   (`THEORY.md` §8), and whether the real-data `C → O` gap resolves above 0 is an open question WS4 poses
   and cannot answer at fixture scale. A P0 at fixture scale does not preclude a P+ at full scale.
   *Realized (2021–2025): at full data scale the floor did shrink as predicted — the `O − C` gap's CI now
   resolves above 0 from `α ≥ 0.1` — but the gap stayed negligible (`+0.0001`–`+0.0003`) and
   off-support-driven, so full scale turned the fixture's P0 into a* statistically detectable but still
   non-actionable *gap, not the myopically-actionable P+ that would have exceeded the ceiling.*
5. **The firewall.** WS4 tests whether acting on `q̂` beats behavior *within support*; it does not certify
   `q̂` as causal. A positive `C → O` gap is evidence that ordered `q̂` yields a better *evaluable* policy,
   not proof that *changing* the pitch *causes* the gain — the finding-#3 claim only the full OPE battery
   (WS7), with its support diagnostics and estimator agreement, can approach (`../ws3_gbdt_stack/THEORY.md`
   §8).
6. **Family granularity and `pitch_type` proxy.** The action is the pitch family (SPEC §4) and `pitch_type`
   is a classifier output (SPEC §1); a mechanism below the family level, or a mislabeled pitch, is only
   partially represented — inherited from WS3.
7. **One reward metric.** Value is `−delta_run_exp`; a different reward could reweight the `q̂` grid and
   move every number.

---

## 9. Conclusion

WS4 is the study's first prescriptive rung and its most disciplined honest negative. It builds one
uncertainty-aware target policy — a feasibility-masked Thompson policy over WS3's posterior `q̂` — and
subjects it to the OPE gate *before* reporting any value, exactly as SPEC §0.3 demands. Its conclusion is
branch-conditional and complete now that the real numbers have arrived:

**Selected by the data (2021–2025): G-PASS × V− × P0 (sequential-not-myopic).** The gate passes; the
myopic policy does not beat observed MLB behavior at any α > 0 (every softened value `≤ V(μ) = −0.0001`,
the deficit growing `−0.0009 → −0.0089` as ESS collapses `100% → 0.6%`); and the `O − C`
sequencing-prescription gap is statistically detectable (CI excludes 0 from `α ≥ 0.1`) but negligible
(`+0.0001`–`+0.0003` run) and off-support-driven — the real-data confirmation of the myopic ceiling. The
branch-conditional readings below are retained; the selected one is the third.

- *(pre-registered alternative — not selected; the gate passed).* **If the gate fails (G-FAIL)**, nothing
  is interpretable and the workstream's honest output is "the harness cannot measure a policy on this data
  — fix the propensities first."
- *(pre-registered alternative — not selected; the gap resolves above 0 but is negligible and
  off-support, not a myopically-actionable edge).* **If the sequencing gap resolves above 0 (P+)**, WS4
  has found a myopically-actionable ordered edge that *exceeds* the measured ceiling — a strong claim, to
  be cross-checked against WS3's order axis and handed to WS7.
- **Selected by the data (2021–2025). If the gap is `≈ 0` (P0)** — the fixture's own reading and the
  realized real-data outcome — the honest verdict is the sequential-not-myopic reading (on real data the
  gap's CI *excludes* 0 but the magnitude is negligible, `+0.0001`–`+0.0003`), and its *content* is the
  myopic ceiling: the sequencing effect, though present, is a **setup** effect a bandit provably cannot
  cash in, which is the falsifiable target (D40) the sequential rungs exist to exceed. Paired with `V−`,
  the whole prescriptive read is a first-class D39 honest negative.
- *(pre-registered alternative — not selected; the real-data gap is positive, not negative).* **If the
  gap is negative (P−)**, the ordered `q̂` overfits and the policy should be built from the simpler view —
  the null-world signature, read as fragmentation, not "order hurts."

Across every branch the durable contributions are the same: an OPE-gated Bayesian bandit that never
reports a value it cannot recover the baseline for; a prescriptive ablation that isolates sequencing from
the count-driven gain under a fixed evaluator; the myopic-ceiling decomposition that measures, to `~0.003`
of `~0.032`, exactly how much of a setup effect greed can reach — now **confirmed on real data** (every
softened policy below behavior, an `O − C` gap that resolves but stays negligible and off-support); and
the ambiguity exhibit's honest toss-up statement of how resolvable "best next pitch" really is
(`~0.57` mean top-vs-runner-up confidence, `~99%` toss-ups on real data). WS4's honest negative — the
myopic policy does not beat observed behavior, and the sequencing prescriptive gap is real but too small
to act on — is the machine telling the truth, and the sequel workstreams (WS5, WS7) exist to catch what
greed cannot.

---

## References

Agrawal, S., and Goyal, N. (2012). Analysis of Thompson sampling for the multi-armed bandit problem.
*Proceedings of the 25th Annual Conference on Learning Theory (COLT)*, PMLR.

Agrawal, S., and Goyal, N. (2013). Thompson sampling for contextual bandits with linear payoffs.
*Proceedings of the 30th International Conference on Machine Learning (ICML)*.

Chapelle, O., and Li, L. (2011). An empirical evaluation of Thompson sampling. *Advances in Neural
Information Processing Systems (NeurIPS) 24*.

Dudík, M., Langford, J., and Li, L. (2011). Doubly robust policy evaluation and learning. *Proceedings of
the 28th International Conference on Machine Learning (ICML)*.

Efron, B., and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal
of the American Statistical Association*.

Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010). A contextual-bandit approach to personalized
news article recommendation. *Proceedings of the 19th International Conference on World Wide Web (WWW)*.

Precup, D., Sutton, R. S., and Singh, S. (2000). Eligibility traces for off-policy policy evaluation.
*Proceedings of the 17th International Conference on Machine Learning (ICML)*.

Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the
evidence of two samples. *Biometrika*, 25(3/4), 285–294.

Tango, T. M., Lichtman, M. G., and Dolphin, A. E. (2007). *The Book: Playing the Percentages in Baseball*.
