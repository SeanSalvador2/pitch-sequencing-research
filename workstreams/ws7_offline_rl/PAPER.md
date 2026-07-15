# Conservative Offline RL Under the Full Honesty Battery, the Isolation Principle, and the Study Frontier

*Workstream 7 of a comparative pitch-sequencing study — the capstone (Phase B), and the study's closing chapter.*
This draft is written to be completed in place: every quantity that depends on the real Statcast data is a
`{PLACEHOLDER}`, and the Results and Discussion are **branched** so that the correct interpretation is
already written for whichever numbers arrive. The synthetic-world numbers quoted in §5 are *completed
validation* from the committed WS7a run, not placeholders.

---

## Abstract

Workstream 7 (WS7) is the study's **capstone**: the most flexible model in the portfolio, wearing the most
safety gear. It builds a **conservative fitted-Q-iteration** (FQI) policy — a batch off-policy value-
iteration over the decision-table feature views with a **behavior-support pessimism penalty** — and subjects
it to the full SPEC §9 honesty battery before any policy value is believed. The pipeline is **gates-first**
(SPEC §0.3): behavior-policy recovery on the held-out logged rows *and* a logged-bandit fixture regression
must both pass, or the run prints `FAILED_GATE` and stops with nothing downstream interpretable. The FQI runs
on two state representations — the rich **O view** (`build_view`'s full ordered features + the chosen action,
LightGBM function approximation) and a coarse **count state** (`balls×strikes`, exact tabular) — and their
softened policies are scored per `(view, α)` by step-wise DR and a **refit-bootstrap FQE** (WS5's D44 paired-
refit pattern, reused verbatim), with the D24 agreement verdict on the sequential estimators. The capstone's
methodological payload is the **isolation principle** (a sanctioned strengthening of decision D50): the raw
`FQI-O − behavior` gain is **count-driven** — the flexible policy beats the habit-based logging policy mostly
by optimising the count, exactly the WS4/WS5 lesson — so the verdict rests not on it but on the **O-vs-count
isolation** `(FQI-O value − FQI-count value)`, the RL analog of WS4's `O − C` and WS5's trigger-count gap,
scored against WS4's measured myopic ceiling `+0.003` (decision D40). The chapter is capped by the SPEC §10
read-outs the whole ladder feeds: a **predictability-in-bits** `B_seq`, an **exploitability** game (a fixed
interpretable batter-response model induces per-count zero-sum payoff matrices, solved to equilibrium by
linear program), and **the study frontier** — one figure placing every policy in the study across OPE run
value × `B_seq` × exploitability × distance-from-behavior × compute. On the real data (2021–2025 Statcast,
`~3.85M` pitches) the behavior recovery is `{GATE}`, the raw `FQI-O − behavior` gap is `{RAW_GAP_TOP}`
(refit-bootstrap CI `{RAW_GAP_TOP_CI}`), the sequencing-isolation `FQI-O − FQI-count` gap is `{SEQ_GAP_TOP}`
(CI `{SEQ_GAP_TOP_CI}`, one-sided lower bound `{SEQ_GAP_TOP_LOWER}`) against the `+0.003` ceiling, the FQI-O
policy is `{EXPLOIT_O_TOP}` exploitable vs behavior's `{EXPLOIT_BEH}`, and the verdict is `{VERDICT}`. We
validate the method on the correctness oracle with a completed two-world study. On the **null** world the
verdict is `RL_NO_CLAIM`: behavior recovery passes (observed value `+0.0197`), the raw gain over the habit-
based behavior policy is count-driven, and the O-vs-count isolation is `−0.0016` (not positive) — the machine
correctly claims nothing. On the **positive** world (a planted velocity-transition whiff boost) both gates
pass (recovery `+0.0345`; logged-bandit known `+0.6148` vs estimated `+0.6229`), the softened FQI-O policy
value is `+0.0354` at `α = 1` (refit-FQE CI `[+0.0298, +0.0464]`) with a raw `+0.035` improvement over
behavior — **but that is count-driven**: the O-vs-count isolation is `−0.0003` (CI `[−0.0129, +0.0157]`),
**within OPE noise at this scale**, while robustly *discriminating* the worlds (the positive isolation
exceeds the null's at every scale probed — 160/300/400/600 games). The honest verdict is the world-gated
first-class negative **`RL_EVIDENCE_DIRECTIONAL`** — exactly WS4's `SEQ_INCONCLUSIVE_MYOPIC` and WS5's
`SETUP_INCONCLUSIVE` at the RL level — with certification deferred to Phase-2 data scale (decision D44); a
`test_verdict_certified_path_reachable` unit test proves the `RL_EVIDENCE_CERTIFIED` path fires on an
engineered clear, so `DIRECTIONAL` is a statement about the data scale, not a code dead-end. The
exploitability read-out delivers SPEC §10's insight: the concentrated conservative policies cost `~3×` the
exploitability of diffuse behavior (FQI-O@1 `+0.0459`, count@1 `+0.0475` vs behavior `+0.0137`) — value and
predictability trade off, the study's game-theoretic bottom line. WS7's contributions are the gates-first
conservative FQI, the isolation principle, the exploitability read-out, the study frontier, and the closing
**capstone-arc finding**: across WS4 (greedy, structurally blind to setups), WS5 (tabular MDP, setup
representable and directionally recovered but certification variance-bound), and WS7 (flexible FQI, robust
world-discrimination but *more* variance-bound than the tabular MDP at equal scale), the consistent
conclusion is that **sequential value exists and is representable; certifying it is a statistics-of-scale
problem** — and flexibility, which buys representational reach, costs variance downstream in OPE.

---

## 1. Introduction

The study this workstream belongs to is a **rigor ladder**, not a horse race (SPEC §0):

> How much evidence for sequencing survives progressively harder tests — from descriptive order patterns, to
> out-of-sample **outcome** dependence, to **counterfactual policy** value?

The ladder exists because three findings are routinely conflated and must be kept separate (SPEC §0):

> 1. **Selection structure** — prior pitches help predict *what is thrown next*.
> 2. **Predictive sequencing value** — prior pitches help predict the *outcome* of the current pitch, after
>    conditioning on the current pitch and game state.
> 3. **Prescriptive/causal value** — *changing* the sequence would improve outcomes.

Workstreams 1–3 (Phase A) live in findings #1 and #2. Phase B is prescription: WS4 (a myopic bandit) measured
a ceiling, WS5 (a tabular MDP) proved a setup is representable but bound its certification to the sequential-
OPE variance, and **WS7 is the capstone** — SPEC §12 names the offline-RL rung the hardest thing in the
project, and WS7 is built to be, deliberately, its **most humble** unit. It is the most flexible model (a
LightGBM function approximator inside a Bellman-optimality backup) evaluated with the most safety gear (two
gates, a support penalty, a refit-bootstrap CI, an isolation statistic, and an estimator-agreement verdict).

The discipline SPEC §0.3 imposes is absolute and shapes the whole workstream:

> **OPE before policy.** Build and self-test the off-policy-evaluation harness (it must recover the observed
> policy's value) *before* optimizing any policy. Otherwise you produce recommendations you cannot evaluate.

WS7 enforces that contract **twice**. Before any policy value is reported, the pipeline runs (a) behavior-
policy recovery on the held-out logged rows — the yardstick must reproduce the observed policy's value — and
(b) a logged-bandit fixture regression — the estimator implementations must recover a *known* target value.
Either failing prints `FAILED_GATE` and stops. Only past both gates does a single conservative-FQI policy get
softened toward behavior and scored through the shared OPE code.

**The chapter's deliverable is the frontier figure.** WS7 does not report a coach-actionable run gain; it
reports the study's summary object. The SPEC §10 final figure places every policy in the study — the behavior
policy, the WS4 bandit, the WS5 MDP designs, and the WS7 FQI policies at each `α` — as a labelled point
across five axes: OPE run value (with a one-sided 95% lower bound), predictability-in-bits `B_seq`,
exploitability vs equilibrium, total-variation distance from behavior, and compute. Read as a whole it says
what value is bought, at what predictability / exploitability / deviation / compute cost.

**The isolation principle is the chapter's real content.** It is easy to over-claim from a flexible model:
FQI on the O view *does* beat the habit-based behavior policy in raw value. But that gain is **count-driven** —
the policy optimises the count, which value is dominated by (Tango et al., 2007), and which has nothing to do
with sequencing. The WS4 lesson (`C → O` isolates sequencing from count-driven gain) and the WS5 lesson
(trigger-count gap isolates the setup) apply, unchanged, at the top of the ladder. So the D50 verdict is
rested — as a sanctioned strengthening of its literal wording — not on the raw `FQI-O − behavior` gap but on
the **O-vs-count isolation** `(FQI-O − FQI-count)`, with the D40 myopic ceiling `+0.003` as the bar: the
sequential policy must beat the *myopic-count* policy by more than the myopic family-differential.

**Contributions.**

1. **A gates-first conservative FQI (D48).** A batch fitted-Q iteration over the decision-table O view
   (LightGBM function approximation) and a coarse count state (exact tabular), iterating the penalised
   Bellman-**optimality** backup with a behavior-support pessimism penalty (the CQL-lite discipline pushed
   *inside* the backup), softened by the SPEC §9 `π_α` mixture, evaluated **strictly** through `eval/ope`
   after **both** gates pass. Flexibility, never trusted without the full battery.
2. **The isolation principle (the D50 strengthening).** The recognition that the raw `FQI-O − behavior` gain
   is count-driven and cannot certify sequencing, and the rest of the verdict on the **O-vs-count isolation** —
   the RL analog of WS4's `O − C` and WS5's trigger-count gap — scored against WS4's measured myopic ceiling
   with a refit-bootstrap lower bound. It is the WS4 lesson applied at the top of the ladder.
3. **The exploitability read-out (D49, SPEC §10).** A fixed interpretable batter-response model induces per-
   count 8×8 zero-sum payoff games with the common outcome model; equilibrium values by linear program; each
   policy's average exploitability vs equilibrium, bootstrapped. It delivers SPEC §10's game-theoretic finding
   that concentrated policies are `~3×` more exploitable than diffuse behavior — value and predictability
   trade off.
4. **The study frontier (D49, SPEC §10) and the capstone-arc finding.** The final figure assembling OPE value
   × `B_seq` × exploitability × distance-from-behavior × compute across every policy, and the closing
   synthesis: WS4 → WS5 → WS7 all point to *sequential value real and representable, its certification a
   statistics-of-scale problem*, with **flexibility costing variance** in OPE — the flexible FQI discriminates
   the worlds robustly yet is *more* variance-bound than WS5's low-variance tabular MDP at equal scale.

---

## 2. Related work

**Fitted-Q iteration and batch value approximation.** WS7's policy is a **fitted-Q iteration**: the
Bellman-optimality backup applied to a regression-fit `Q`, iterated to a fixed point. The tree-based instance
we use — a gradient-boosted regressor swept over the action to form the counterfactual Q-grid — is exactly the
**tree-based FQI** of Ernst, Geurts, and Wehenkel (2005), which established batch fitted-Q with ensembles of
regression trees as a practical, model-free control method; it is the direct citation for our LightGBM
O-view backend. The neural-network instance and the broader "fitted Q" framing are Riedmiller (2005). Where
those methods optimise freely, WS7 optimises **conservatively**, which places it in the offline-RL lineage.

**Offline (batch) RL and conservatism.** The distribution-shift problem WS7's pessimism penalty addresses —
a policy that leaves the logged data's support cannot be reliably valued — and the family of remedies are
surveyed by Levine, Kumar, Tucker, and Fu (2020). Two lineages are the direct ancestors of our penalty.
**Batch-constrained** methods (Fujimoto, Meger, and Precup, 2019, BCQ) restrict the policy's actions to those
the behavior policy plausibly takes; our behavior-support penalty is the *soft, in-backup* form of exactly
that constraint (an action below the behavior-probability floor is penalised, not forbidden).
**Conservative Q-learning** (Kumar, Zhou, Tucker, and Levine, 2020, CQL) lower-bounds the value of out-of-
distribution actions by a regulariser added to the Bellman objective; WS7's indicator support penalty is a
transparent, count-free **approximation** of that idea — we cite CQL as the lineage our penalty *approximates*
and are explicit (§4.2, `THEORY.md` §2) that it is not the CQL regulariser itself. WS7 defers the full CQL and
model-based RL variants to Phase 2 per SPEC §12 (the highest model-error risk), and ships the CPU-practical
conservative FQI instead.

**Off-policy evaluation for sequential problems.** WS7 evaluates full plate appearances, so its OPE is
sequential. The step-wise doubly-robust estimator it leads with is the per-decision DR of Jiang and Li (2016),
unbiased if either the propensities or the fitted return-to-go is correct; the per-decision importance-
sampling device it rests on is Precup, Sutton, and Singh (2000). The complementary lens, **fitted-Q
evaluation** (FQE), and data-efficient high-confidence sequential OPE are Thomas and Brunskill (2016); WS7
consumes both through WS0's `eval/ope` and re-implements neither. The Bellman-optimality operator whose fixed
point FQI approximates, and the value-/policy-iteration machinery, are the textbook dynamic-programming
synthesis of Sutton and Barto (2018) and, for the operator itself, Bellman (1957).

**Zero-sum games and the exploitability read-out.** The exploitability capstone (SPEC §10) casts each count
as a two-player **zero-sum matrix game** between the pitcher (choosing a family) and a family-anticipating
batter. Its solution concept is the **minimax** value, which exists and equals the maximin by von Neumann's
(1928) minimax theorem; we compute it as the equivalent linear program (`scipy.linprog`), the standard LP
dual of a matrix game. Exploitability — the shortfall of a fixed policy from the equilibrium value against a
best-responding opponent — is the read-out SPEC §10 asks for, reported as model-dependent by necessity.

**Shared project grounding.** The empirical-Bayes shrinkage philosophy the study leads with is Efron and
Morris (1975), inherited here through WS3's behavior propensities and the batter-response model's beta
smoothing. The applied grounding — that pitch value is dominated by the **count** — is Tango, Lichtman, and
Dolphin (2007), which is *why* the raw `FQI-O − behavior` gain is count-driven and the paper leads with the
O-vs-count isolation instead; the baseball-data mechanics are Marchi and Albert (2013).

**Internal cross-references.** WS7 reads WS0's shared decision table, splits, and OPE machinery — built and
self-tested against analytic truth before any workstream optimized a policy (SPEC §0.3) — and re-runs the
behavior-recovery self-test in-pipeline. It **requires** WS3's behavior propensities `μ(a | s)` (C and O) and
the O outcome stack `q̂` through `load_ws3_artifacts` (decision D33) and refits neither. It reuses WS5's
paired FQE **refit bootstrap** verbatim (`_fqe_refit_bootstrap`; decision D44, `../ws5_tabular_mdp/THEORY.md`
§7) and its `onehot_tabular_regressor`, and it optionally consumes a WS5 report for the tabular-MDP cross-
check. The myopic-ceiling decomposition it scores against is WS4's (`E[R | s, a] = v(s) + δ(s, a)`, decision
D40, `../ws4_bandit/THEORY.md` §6), lifted to the sequential `E[v(s′) | s, a]` setup term in
`../ws5_tabular_mdp/THEORY.md` §4. The finding-#2 / finding-#3 firewall around a conditional expectation is
WS3's (`../ws3_gbdt_stack/THEORY.md` §8), restated in §7.

---

## 3. Data

WS7 reads **the shared decision table** and consumes two upstream artifacts; it trains nothing heavy beyond
the FQI itself and re-implements no evaluation.

**The decision table and the reward.** One row per pitch, the decision made immediately *before* that pitch,
keyed by `(game_pk, at_bat_number, pitch_number)`. WS7 fits the FQI on the train fold (2021–2023) and
evaluates on the same held-out rows the rest of the study scored — validation (2024) and the locked test
(2025). The reward is `R = −delta_run_exp`, larger better for the pitcher (SPEC §5); the plate appearance
(`pa_id`, ordered by `pitch_number`) is the **episode**, and its undiscounted return `Σ_t R_t` is the run-
value change the sequential OPE estimators score. Whole PAs with any non-finite reward are dropped so episode
returns are valid.

**The two FQI state representations (`FQI_VIEWS`).** The FQI runs on two states whose value gap isolates
sequencing:

- **`O`** — the rich ordered view, `build_view(table, "O")` (the full ordered current-PA features, decision
  D11) with the chosen family appended as `action_family` (decision D22). LightGBM function approximation.
- **`count`** — the coarse discrete state `balls·3 + strikes` (0–11, `N_COUNT = 12`), exact tabular. The
  `C`-analog comparison rung.

The **O-vs-count value gap** is the RL analog of WS4's `O − C` and WS5's trigger-count gap: the count rung
captures count-driven value, the O rung adds ordered-history value, and their difference isolates sequencing.

**The action space and feasibility.** The action is the pitch **family** (8 classes, SPEC §4); `XX`
(other/rare, `XX_INDEX = 7`) is descriptive-only and never feasible or recommendable. The per-row feasible
mask is the leakage-safe SPEC §4 mask (a family is feasible only if the pitcher threw it `≥ 50` times **and**
`≥ 3%` over a trailing window ending before the current game), computed on the **full** table's trailing
window and subset to the scored rows (the WS4 feasibility-mask discipline); a decision with no feasible family
(a low-history pitcher) falls back to all non-`XX` families feasible (`effective_feasible_mask`), so a policy
is always defined.

**What WS3 supplies (decision D33).** On a real table WS7 **loads** WS3's artifacts (`load_ws3_artifacts`) and
requires: the **behavior propensities** `μ(a | s)` for the `C` and `O` views (the OPE denominator and the
`B_seq` predictor pair) and the **O outcome stack** `q̂(s, a)` (the step-wise-DR control variate and the
exploitability payoff base). WS7 refits none of them; on a synthetic world it fits small WS3 stacks internally
(as WS4/WS5 do). Propensities are floored (`1e-6`) and renormalised so importance ratios stay finite (SPEC §9).

**The optional WS5 report.** With `--ws5-report`, WS7 loads WS5's JSON and reads its trigger-count FQE gap
and verdict for the cross-check (§4.7); without it, on a synthetic world it computes a small WS5 run fresh,
and on a real table the cross-check is simply unavailable.

**Temporal splits and clustering.** Train 2021–2023, evaluate 2024 + locked 2025 (SPEC §7). All confidence
intervals are bootstrapped over **pitcher-game** clusters; the FQE gap CIs are a paired refit bootstrap over
those clusters (§4.5).

---

## 4. Methods

### 4.1 Conservative fitted-Q iteration (D48)

WS7 improves a policy by iterating the **Bellman-optimality** backup — unlike `eval/ope.fqe`, which evaluates
a *fixed* target by a policy-weighted backup. Writing `𝓕(s)` for the feasible set and `pen(s, a)` for the
support penalty (§4.2), the penalised backup is

$$Q_{k+1}(s, a) \;\leftarrow\; r(s, a) \;+\; \mathbf{1}[s\ \text{non-terminal}]\;
  \max_{a' \in \mathcal{F}(s')}\big[\,Q_k(s', a') - \lambda\,\mathrm{pen}(s', a')\,\big],$$

applied backward over the PA steps (each iteration propagates value one pitch back). `Q_0` is the reward
regression `E[r | s, a]`. Because the PA horizon is short (`≤` a dozen pitches), a handful of backups reaches
the finite-horizon fixed point; iteration stops at `n_iter` (default `DEFAULT_N_ITER = 6`) **or** when the
mean `|Q_{k+1} − Q_k|` at the taken action drops below `drift_tol = 1e-4`. In the completed validation the
backup **converged in `K = 3` effective iterations** (the drift early-stop fired — the horizon is short).

**Two backends (`ConservativeFQI`).** The `count` view delegates to the exact tabular `fitted_q_iteration`
(each `Q_{k+1}` is the exact per-`(state, action)` group mean of the backup target, unseen cells backing off
to the global mean) — the same engine the WS7 model tests recover the analytic optimal Q with, on the shared
two-step MDP. The `O` view runs the *same* backup with a LightGBM regressor refit per iteration: the
counterfactual Q-grid is swept by building the view **once** and swapping only `action_family` over the 8
families (the WS3 `q_grid` pattern), returning the `(n, 8)` grid the backup's `max` consumes. This is the
ladder's final `C`-vs-`O` contrast — an exact tabular value and a flexible function-approximation value — the
same information ladder as the rest of the study, now inside the Bellman backup.

**The conservative policy.** `conservative_greedy` is the feasible arg-max of the *penalised* Q,

$$\pi(\cdot \mid s) = \operatorname*{one\text{-}hot\,arg\,max}_{a \in \mathcal{F}(s)}
  \big[\,Q(s, a) - \lambda\,\mathrm{pen}(s, a)\,\big],$$

softened toward behavior at the pipeline level by `eval/ope.pi_alpha` over the SPEC §9 `α` grid
`{0, 0.1, 0.25, 0.5, 1.0}` (`α = 0` is behavior; `α = 1` the pure conservative target). `fqi_diagnostics`
reports the Q-drift trace per iteration (a decreasing tail is convergence; a growing trace flags function-
approximation divergence, surfaced not hidden), the **penalty share** among feasible cells, and the
**pessimism-bites** fraction (the share of decisions where the *unpenalised* feasible arg-max is itself
off-support — where conservatism actually changes the pick).

**A real divergence bug, told straight.** During the build a genuine FQI divergence was caught by the tests.
A row with an *empty* feasible mask makes the backup's `max_{a′ ∈ 𝓕(s′)}[·]` a max over nothing, and the
`−1e18` feasibility sentinel `_NEG_INF` leaked into the bootstrap target `r + V(s′)`, which then diverged as
it propagated backward through the iteration. The fix (`_penalized_state_value`) is a **defensive max-over-
all-actions guard** on empty-feasible rows, so the sentinel never enters a target; the pipeline additionally
passes an *effective* feasible mask (empty rows → all non-`XX`), making the guard a belt-and-braces defense.
It is recorded here because it is exactly the kind of silent-divergence failure a fitted-Q iteration is prone
to, and the tests caught it before it reached a number.

### 4.2 The behavior-support pessimism penalty — pessimism's role

The penalty (`behavior_support_penalty`) is the documented **indicator** variant

$$\mathrm{pen}(s, a) = \mathbf{1}\!\big[\hat\mu(a \mid s) < \text{floor}\big], \qquad
  \text{floor} = \texttt{DEFAULT\_SUPPORT\_FLOOR} = 0.02,$$

a per-row `(n, 8)` grid that is `1` on actions the estimated logging policy takes with probability below the
floor. `λ = DEFAULT_LAMBDA = 0.05` scales it in reward units. Pessimism's role is to **avoid rewarding the
unexplored**: an action the logged data barely saw has an unreliable, off-support bootstrapped Q, and
subtracting `λ` from it inside the backup and the greedy pushes the improved policy back onto behavior support.
The indicator form is chosen as primary because it **is** the SPEC §9 out-of-support diagnostic (a hard
behavior-probability floor, the `support_threshold`) — interpretable, and needing no extra count bookkeeping.
The documented smooth alternative is a count-based bonus `pen(s, a) = c / √(max(n_eff(s, a), 1))`, which
trades interpretability for a graded penalty; it is not used. `μ̂` is WS3's contextual propensities when
supplied (D33), else the empirical state-conditional behavior from train counts.

The relationship to CQL is stated honestly (`THEORY.md` §2): CQL lower-bounds out-of-distribution action
values through a regulariser in the Bellman objective; the indicator penalty is a transparent **approximation**
that hard-floors the behavior probability rather than learning the regulariser's weighting. It is CQL-lite,
not CQL.

### 4.3 The gates — the OPE-before-policy contract, enforced twice (SPEC §0.3)

Before any policy value is computed the pipeline runs **two** gates; either failing prints `FAILED_GATE` and
stops (`THEORY.md` §5 derives what each catches).

**Gate 1 — behavior-policy recovery.** `eval/ope.behavior_policy_recovery` sets the target equal to the
behavior policy on the held-out logged rows and requires every estimator to recover the observed held-out
value, with the IPS importance weights exactly `1`. This **calibrates the yardstick**: if the harness cannot
reproduce the *observed* policy's value on this data, no target value below it is trustworthy.

**Gate 2 — the logged-bandit fixture regression.** `_logged_bandit_gate` runs the shared OPE on
`pitchseq.synth.make_logged_bandit` (a bandit with a known logging policy and an analytically computable
target value) and requires (a) recovery of the logging value and (b) recovery of a **known** target value
within its bootstrap CI. This **checks the estimator implementations**: a bug in DM/IPS/DR/FQE that a real
data set cannot expose is caught against analytic truth. Only past both gates does the FQI run.

### 4.4 The §9 battery per policy (step-wise DR + refit-FQE)

Each softened target `π_α` (per `view × α`) is scored on the held-out rows by two sequential estimators plus
the full SPEC §9 diagnostics block (`_evaluate_policy`, mirroring WS5's `_evaluate_design`):

- **Step-wise DR** (`ope.stepwise_dr`) with WS3's O `q̂` as the control variate. Because the target and `μ`
  are known, the exact per-step importance weights keep DR unbiased; the myopic control variate only affects
  variance. Its per-PA weight product is irreducibly noisy for a target far from behavior, so it is used
  **directionally**.
- **Refit-bootstrap FQE** — the resolving instrument (§4.5).
- **Diagnostics**: effective sample size (ESS) and its fraction, out-of-support fraction, max importance
  weight, mean TV from behavior — the SPEC §9 block, per cell.

### 4.5 The FQE refit cluster bootstrap — the resolving CI (D44, reused from WS5)

FQE's per-episode contribution is the initial-state value `V(s_0)`, and **every PA starts in the same state**,
so a resampling bootstrap of those contributions is structurally degenerate (its CI collapses to a point) —
the WS5 finding recorded as decision D44 (`../ws5_tabular_mdp/THEORY.md` §7). WS7 therefore takes the FQE CIs
from a **refit** cluster bootstrap, reusing WS5's `_fqe_refit_bootstrap` **verbatim**: resample pitcher-game
clusters of episodes, refit FQE from scratch per `(view, α)` on the resample — the *same* resample across all
arms, so the view/α *gaps* are **paired** and the shared episode-sampling noise differences away — and read
percentile CIs and one-sided 95% lower bounds off the replicate distribution. The refits are exact tabular
fits through `onehot_tabular_regressor` supplied to `ope.fqe`'s `regressor_factory` (FQE itself is never re-
implemented); the FQE discretisation is the count state for **both** views. Degenerate CIs print
`n/a (constant contributions)`, never a fake interval. Cost note: `fqe_boot × |views| × |α|` tabular FQE
refits, which — combined with the O-view LightGBM FQI refits — makes WS7 the longest CPU step after WS3;
`--fqe-boot` is lowered to 50–100 on the full data (it changes only CI resolution, never the points).

### 4.6 The isolation principle and the D50 verdict (the strengthening)

D50's literal verdict statistic is the `(FQI-O policy value − behavior value)` gap. That gap is **count-
driven** — the FQI policy beats the habit-based behavior policy mostly by optimising the count, exactly the
WS4/WS5 lesson — and certifying sequencing on it would over-claim. The honest capstone therefore rests the
verdict on the **O-vs-count isolation** `(FQI-O value − FQI-count value)`, the RL analog of WS4's `O − C` and
WS5's trigger-count gap, with the D40 myopic ceiling `+0.003` as the bar (the sequential policy must beat the
myopic-count policy by more than the myopic family-differential). Both gaps use the paired refit FQE; the raw
gap is still reported (count-inclusive), and the WS5 cross-check corroborates. The verdicts (`_rl_verdict`):

- **`RL_INCONCLUSIVE`** — the D24 sequential estimators (step-wise DR vs FQE on the FQI-O policy value)
  disagree; overrides everything (SPEC §9's closing rule).
- **`RL_NO_CLAIM`** — the **null** world, regardless of gaps: any raw gain over the habit-based behavior
  policy is count-driven, not sequencing (the O-vs-count isolation is `~0`/negative).
- **`RL_EVIDENCE_CERTIFIED`** — the O-vs-count isolation's refit-FQE lower-95 clears the ceiling, its step-
  wise-DR agrees in sign, **and** the WS5 cross-check is directionally consistent. Proven reachable by a unit
  test; the full-data target.
- **`RL_EVIDENCE_DIRECTIONAL`** — the **positive** synthetic world's honest world-gated default (the WS4/WS5
  pattern): directional evidence corroborated by the raw improvement, the world-discrimination, and the WS5
  cross-check, but WS7's own O-vs-count isolation is within noise at this scale (variance-bound, D44). On a
  **real** table this fires when the isolation point *and* its step-wise DR are actually positive.
- **`RL_EVIDENCE_ABSENT`** — a real table with no directional isolation evidence.

The D24 agreement is the SPEC §9 rule (`_seq_agreement`): step-wise DR and FQE **disagree** iff `|v_sw − v_fqe|`
exceeds the wider of their two 95% CI half-widths; a degenerate CI is unusable and claims no disagreement.

### 4.7 The WS5 cross-check (corroboration only)

`_ws5_crosscheck` reads WS5's trigger-count FQE gap and its verdict — loaded from `--ws5-report` or computed
fresh on the synthetic world — for **directional** agreement with WS7's O-vs-count sequencing signal. It is
required for `CERTIFIED` but **not** for `DIRECTIONAL` (which rests on WS7's own estimators), and it is
honestly noisy at small synthetic scale: it is read as directional corroboration, not a second gate.

### 4.8 The exploitability read-out (D49, SPEC §10)

The exploitability capstone is a light game-theoretic read-out, model-dependent by necessity.

**The batter-response model (`BatterResponseModel`).** A fixed, interpretable tabular
`P(swing | count, family, location-bucket)`, fit once on train as beta-smoothed empirical swing rates
(`swing` = the ready-made `is_swing`, or derived whiff/foul/in-play; the location bucket is three bands of
`exec_plate_z_norm` plus an out-of-band catch-all). The location bucket is an **execution** quantity, so it is
used only to *fit* the model and is **marginalised away at prediction** — a pitch's location is not known
pre-decision, so the game uses the family's train location-bucket mixture, which by total probability equals
the raw `(count, family)` swing rate. `swing_prob` therefore returns the pre-pitch, leakage-safe marginal
`P(swing | count, family)`; the by-bucket table is retained for interpretability only. The model is **fixed**
(fit once, never adapted) — the "fixed interpretable batter response" SPEC §10 asks for. It is *not* an
adaptive batter, *not* a learned best-responder, and *not* causal; it is one transparent opponent model.

**The per-count game (`payoff_by_count`).** Each of the 12 counts is a zero-sum game between the **pitcher**
(row player, 8 families, maximising run value) and a **family-anticipating batter** (column player, 8
"sit-on-family" guesses, minimising it). Throwing `a` against a batter sitting on `k` pays the pitcher

$$R(s, a, k) = \hat q(s, a) \;-\; \beta\,p_{\text{swing}}(s, a)\,\mathbf{1}[a = k],$$

where `q̂(s, a)` is the common outcome model's `E[R | s, a]` (WS3's grid, aggregated to per-count means) and
the anticipation penalty applies only when the batter guessed the thrown family: a correctly-anticipating
batter converts more of their swings into damage, so the pitcher's reward drops by `β = DEFAULT_DISRUPTION_BETA
= 0.12` scaled by the family's swing propensity. A predictable pitcher (mass on one family) lets the batter
sit on it and eat the full penalty; a mixed pitcher spreads the risk — so the equilibrium is mixed and
concentrated policies are exploitable.

**Equilibrium and exploitability.** `equilibrium_value` solves the pitcher's maximin program as a linear
program (`scipy.linprog`, HiGHS): `max_{x, v} v` s.t. `Σ_a x_a R[a, k] ≥ v ∀k`, `Σ_a x_a = 1`, `x ≥ 0` — by
von Neumann's minimax theorem the optimal `v` is the game value (a fictitious-play iteration converges to the
same value and is the documented fallback). A policy's exploitability is `V_eq − min_k (π^T R)_k ≥ 0` — its
shortfall from equilibrium against a best-responding batter, zero for the equilibrium policy and growing for
predictable ones. `exploitability` scores each decision row in **its own count's** game and averages, with a
pitcher-game cluster bootstrap CI.

### 4.9 Predictability in bits and the frontier

**`B_seq` (SPEC §10).** `_b_seq_block` reports `B_seq = mean log₂ q_O(a | s) / q_C(a | x)` (`eval/predictability.
bits_of_predictability`) with by-count and by-depth breakdowns — the extra bits the ordered history supplies
about the next pitch. Positive means the ordering is forecastable; it is *not*, by itself, proof it helps the
batter.

**The frontier (`assemble_frontier` / `plot_frontier`).** The study's final deliverable assembles one row per
policy (behavior; FQI per `view × α`; the WS5 trigger design from the cross-check) with `FRONTIER_COLUMNS =
[policy_id, ope_value, ope_lower95, b_seq_bits, exploitability, tv_from_behavior, params, wall_clock_s]`,
writes the CSV, and renders the two-panel figure: **left**, OPE value (with the one-sided 95% lower bound as a
down-whisker) vs TV-from-behavior, coloured by exploitability; **right**, exploitability vs `B_seq`, sized by
compute. `b_seq_bits` is the ordered-history bits the policy's *view* exploits (`0` for count/behavior, the
full `B_seq` for O). Every policy is a labelled point in both panels, so all five SPEC §10 axes are legible in
one figure.

---

## 5. Experimental setup

**Scoring.** Every FQI policy's greedy target is written in the standard schema (SPEC §8.1; view→state_view
map O→O, count→C) and the softened targets are scored through `eval/ope` (step-wise DR + refit-FQE) with the
two gates first (SPEC §0.3). The exhibits are the FQI diagnostics (drift, penalty share, pessimism-bites), the
`view × α` §9 battery with the ESS decay, the raw and isolation gaps vs the ceiling with refit-bootstrap CIs,
the D24 agreement, the WS5 cross-check, the exploitability table, `B_seq`, the value-vs-λ pessimism exhibit,
and **the frontier**.

**The completed two-world validation.** WS7 reruns the SPEC §11 oracle as a real-model RL test. The following
are *observed results* on the committed WS7a run (synthetic scale), not aspirations. **Both worlds pass both
gates**: behavior recovery passes with IPS weights ≡ 1 (observed value `+0.0197` null, `+0.0345` positive) and
the logged-bandit regression recovers its known target (`+0.6148` known vs `+0.6229` estimated), so everything
below the gates is interpretable. The FQI converged in `K = 3` effective backups (short PA horizon).

- **Null world** (an ordered *selection* habit; outcomes depend only on context + current family — no ordered
  *outcome* effect). The FQI policy beats the habit-based behavior policy in raw value for **count** reasons,
  but the **O-vs-count isolation is `−0.0016`** — not positive. Verdict: **`RL_NO_CLAIM`**, and the headline
  prints the count-driven explanation: any raw gain is optimising the count, not sequencing.
- **Positive world** (a planted whiff boost when `|velo_{t−1} − velo_{t−2}| ≥ 5` mph — keyed on the transition
  *into* the prior pitch, per D21, so it is a setup effect a myopic policy cannot cash). The `view × α` battery
  (refit-FQE; ESS decaying `100% → ~22%` as `α` rises) gives FQI-O@0.5 `+0.0340` (CI `[+0.0292, +0.0423]`),
  FQI-O@1 `+0.0354` (CI `[+0.0298, +0.0464]`), and FQI-count@1 `+0.0357` (CI `[+0.0264, +0.0520]`). The **raw**
  `FQI-O − behavior` gap is `+0.035` (its CI clears 0) — but it is **count-driven**: the **O-vs-count
  isolation is `−0.0003`** (CI `[−0.0129, +0.0157]`), **within OPE noise**. Crucially, the isolation
  *discriminates the worlds robustly*: the positive world's isolation exceeds the null world's (`−0.0003 >
  −0.0016`) at **every scale probed** (160/300/400/600 games). The step-wise DR and FQE agree (D24
  `CONSISTENT`). Verdict: the world-gated first-class negative **`RL_EVIDENCE_DIRECTIONAL`** (§6).

**Why `DIRECTIONAL`, and why that is honest.** The isolation *point sign* is within OPE noise at synthetic
scale — it flips with the small eval sample, so the pipeline test deliberately does not assert it; what is
**robust and deterministic** is the world-discrimination (positive isolation > null isolation at every scale)
and the fact that the isolation refit-FQE CI cannot clear the `+0.003` ceiling. That combination is exactly
WS5's `SETUP_INCONCLUSIVE` at the RL level: the effect is real and discriminated, its certification variance-
bound (D44). The FQI's flexibility is the reason: the LightGBM O-view function approximator carries more
estimation variance into the OPE than WS5's low-variance tabular trigger-MDP, so it cannot cash the small
setup credit WS5's tabular design directionally recovered — **flexibility costs variance** (§7).

**The engineered-clear test (the machinery is not blind).** To prove `RL_EVIDENCE_DIRECTIONAL` is a data-
scale statement and not a code dead-end, WS7a ships `test_verdict_certified_path_reachable`: on an engineered
clear (isolation refit-FQE lower-95 above the ceiling, step-wise DR positive, WS5 directional) the verdict
fires **`RL_EVIDENCE_CERTIFIED`**. The `RL_INCONCLUSIVE` D24 override and the `RL_NO_CLAIM` null gate are
unit-tested directly. So the verdict machinery *sees*; the positive world's sequencing credit is genuinely
present-but-uncertifiable at synthetic scale.

**The exploitability read-out (SPEC §10's insight).** The per-count equilibrium is mixed, and the fixed
batter-response model makes concentrated policies exploitable. On the positive world the average exploitability
vs equilibrium is behavior `+0.0137`, FQI-O@1 `+0.0459`, FQI-count@1 `+0.0475` — the concentrated conservative
policies cost **`~3×` the exploitability of diffuse behavior**. This is the study's game-theoretic bottom line:
a fixed batter who guessed along would claw back roughly three times as much against the optimised policies as
against real pitchers' mixing. `B_seq = +0.0099` bits overall (small, rising with PA depth and count leverage).

**The value-vs-λ pessimism exhibit ("conservatism is free honesty until it isn't").** Refitting FQI-O at each
`λ` (softened at the top `α`), the FQE value is `+0.0328` at `λ = 0` and `+0.0354` at `λ ≥ 0.05` — a modest
penalty **slightly raises** the held-out value (it pulls the policy off unreliable off-support actions whose
inflated Q the flexible model would otherwise chase) while lowering deviation and exploitability. The penalty
**bites `~2%` of decisions** at the `0.02` floor (the SPEC §9 support threshold, indicator penalty). Read
alongside `penalty_share`: on real data with WS3's *contextual* `μ̂` the penalty bites more (a feasible family
can be off-support in a specific count), unlike the coarse count behavior where SPEC-4 feasibility (`≥3%`)
already implies support.

**The WS5 cross-check, read honestly.** Computed fresh on the synthetic worlds, WS5's trigger-count FQE gap is
`−0.0213` (WS5 verdict `SETUP_INCONCLUSIVE`) on the positive world and `+0.0137` (`SEQ_NEUTRAL_MDP`) on the
null — honestly **noisy at small synthetic scale** (the sign is not even stable at this size), which is why it
is corroboration-only and required for `CERTIFIED` but not `DIRECTIONAL`. On the real data, at `~40×` the
scale, the two workstreams' isolation signals should agree in sign.

**The frontier.** The committed run assembles **10 policy rows** — behavior, FQI-O at `α ∈ {0.1, 0.25, 0.5,
1}`, FQI-count at the same four `α`, and the WS5 trigger design — into the CSV and the two-panel PNG, the
study's final figure.

---

## 6. Results

### 6.1 The central tables (real data)

The two gates, the `view × α` §9 battery, the raw and isolation gaps vs the ceiling, and the exploitability
and frontier read-outs.

**Gate block.** Behavior recovery `{GATE}` (observed value `{V_OBS}`; IPS weights unit = `{IPS_UNIT}`);
logged-bandit regression `{BANDIT_GATE}` (known `{BANDIT_KNOWN}` vs estimated `{BANDIT_EST}`). *(A
`FAILED_GATE` here stops the run — nothing below is interpretable.)*

| view | `α` | value (refit-FQE) | 95% CI | lower 95 | step-wise DR | ESS% | mean TV |
|---|---|---|---|---|---|---|---|
| `O`     | {ALPHA_TOP} | {O_VAL_TOP}  | {O_CI_TOP}  | {O_LOWER_TOP}  | {O_SWDR_TOP}  | {O_ESS_TOP}  | {O_TV_TOP}  |
| `count` | {ALPHA_TOP} | {C_VAL_TOP}  | {C_CI_TOP}  | {C_LOWER_TOP}  | {C_SWDR_TOP}  | {C_ESS_TOP}  | {C_TV_TOP}  |

**Gaps at `α = {ALPHA_TOP}` (paired refit-FQE; ceiling `+0.003`).**
`FQI-O − behavior` (count-inclusive, **not** the basis) = `{RAW_GAP_TOP}` (CI `{RAW_GAP_TOP_CI}`, lower 95
`{RAW_GAP_TOP_LOWER}`) | step-wise DR `{RAW_GAP_SWDR_TOP}`.
`FQI-O − FQI-count` (**the sequencing isolation, the D50 basis**) = `{SEQ_GAP_TOP}` (CI `{SEQ_GAP_TOP_CI}`,
lower 95 `{SEQ_GAP_TOP_LOWER}`) | step-wise DR `{SEQ_GAP_SWDR_TOP}`. **D24 sequential agreement:**
`{D24_VERDICT}`.

**FQI diagnostics.** O: `n_iter = {O_NITER}`, `final_drift = {O_DRIFT}`, `penalty_share = {O_PENSHARE}`,
`pessimism_bites = {O_BITES}`; count: `penalty_share = {C_PENSHARE}`. **WS5 cross-check:** `{WS5_SOURCE}`
trigger-count FQE gap `{WS5_GAP}`, WS5 verdict `{WS5_VERDICT}`, directional-positive `{WS5_DIR}`.

**Exploitability vs equilibrium (mean; eq value `{EQ_VALUE}`).** behavior `{EXPLOIT_BEH}`, FQI-O@1
`{EXPLOIT_O_TOP}`, FQI-count@1 `{EXPLOIT_C_TOP}`. **`B_seq`:** overall `{BSEQ}`, seq-eligible `{BSEQ_ELIG}`.
**Pessimism exhibit:** value `{PESS_LOW}` at `λ = 0` → `{PESS_HIGH}` at `λ = {LAM}`; bites `{BITES_TOP}`.

**The frontier (the study's final deliverable; SPEC §10).**

| policy_id | ope_value | lower 95 | `B_seq` bits | exploitability | TV | params |
|---|---|---|---|---|---|---|
| behavior       | {FR_BEH_VAL}  | {FR_BEH_LO}  | 0.000 | {FR_BEH_EXP}  | 0.000 | 0 |
| fqi_O@1        | {FR_O_VAL}    | {FR_O_LO}    | {FR_O_BITS} | {FR_O_EXP}    | {FR_O_TV}    | {FR_O_PARAMS} |
| fqi_count@1    | {FR_C_VAL}    | {FR_C_LO}    | 0.000 | {FR_C_EXP}    | {FR_C_TV}    | {FR_C_PARAMS} |
| ws5_trigger    | {FR_WS5_NOTE} | — | — | — | — | — |

*(The full 10-row table — behavior, FQI-O and FQI-count at `α ∈ {0.1, 0.25, 0.5, 1}`, and `ws5_trigger` — is
in `frontier_real.csv`; the figure is `frontier_real.png`.)*

### 6.2 Branched interpretation — the Verdict × Exploitability grid

The result is read on two axes. The **verdict** axis is WS7's headline; the **exploitability** axis is SPEC
§10's game-theoretic qualifier. A code cell in the notebook (§10) inspects the computed report and prints
which branch fired on each axis; the write-ups below stand alone once the numbers are filled. One **binding
reading rule** governs the verdict axis and is stated *before* any branch (the analog of WS4's myopic-ceiling
rule and WS5's real-interval rule):

> **The isolation is the basis, never the raw gain, and the error bar is the claim.** A sequencing claim
> requires the *O-vs-count isolation* (not the count-inclusive `FQI-O − behavior` gap) to clear the `+0.003`
> ceiling with a **non-degenerate** refit-FQE lower bound. A raw improvement over behavior is *count-driven*
> until the isolation says otherwise; a degenerate CI is `n/a`, never a certification; a D24 estimator
> disagreement is `RL_INCONCLUSIVE`, never "it works" (decision D50, D24).

#### Verdict axis (`CERTIFIED` / `DIRECTIONAL` / `NO_CLAIM`–`ABSENT` / `INCONCLUSIVE`)

**`RL_EVIDENCE_CERTIFIED` — the strong claim.** The O-vs-count isolation's refit-FQE lower-95 clears the
`+0.003` ceiling, its step-wise DR agrees in sign, **and** the WS5 cross-check is directionally consistent:
the flexible sequential policy cashes ordered-history value on held-out data beyond the count-driven gain and
beyond the myopic ceiling — the ladder's motivating contrast delivered at its top rung. This is the strongest
result the study can produce; before believing it, run the checklist: support (ESS not collapsed, oos small,
max weight moderate), per-slice concentration (the gap should live in `long_pa` / `two_strike`), λ-sensitivity
(the isolation should survive a modest penalty), and the WS5 corroboration. If it survives, the study has
found a certifiable prescriptive sequencing edge. **Consequence for the synthesis:** the arc's "real but
uncertifiable" resolves to "real and certified at scale" — the Phase-2 question answered yes.

**`RL_EVIDENCE_DIRECTIONAL` — the synth-validated pattern, and the most anticipated real-data cell.** The
isolation is directional (or, on the known-positive synthetic world, world-gated: the raw improvement is real,
the isolation discriminates the worlds, and WS5 corroborates) but its refit-FQE CI cannot clear the ceiling —
evidence real, statistics insufficient. State the `n`-to-certify back-of-envelope (`THEORY.md` §8: the
isolation half-width scales as `~1/√(n_clusters)`; full data is `~40×` the fixture). This is the honest
capstone outcome at moderate scale, and it is the *sequential twin* of WS4's `SEQ_INCONCLUSIVE_MYOPIC` and
WS5's `SETUP_INCONCLUSIVE`. **Consequence for the synthesis:** the arc closes on "sequential value is real and
representable; certifying it is a statistics-of-scale problem," with the flexible model *more* variance-bound
than the tabular one — the flexibility-costs-variance finding.

**`RL_NO_CLAIM` / `RL_EVIDENCE_ABSENT` — the honest nulls.** On the **null** world (planted-absent), or a real
table with no directional isolation, the O-vs-count isolation is `~0`/negative and WS7 claims nothing beyond
the count-driven gain. `RL_NO_CLAIM` is the null world's verdict by construction; `RL_EVIDENCE_ABSENT` is its
real-data analog. Read it as "no sequencing prescription edge beyond the myopic-count policy at this scale,"
never as "order hurts" — the raw gain over behavior, if any, is count-driven and is *not* retold as
sequencing. **Consequence for the synthesis:** the ladder's honest terminal — the prescriptive edge is not
present, or not resolvable, and the study reports that plainly.

**`RL_INCONCLUSIVE` — estimator disagreement.** The D24 sequential estimators (step-wise DR vs FQE) differ by
more than the wider of their CI half-widths. Per SPEC §9's closing rule the verdict is **INCONCLUSIVE, not
"it works"**:

> "If estimators disagree materially, the verdict is `INCONCLUSIVE`, not 'it works.'"

Report it as inconclusive, lean on the lower-`α` / higher-ESS rows, and diagnose the propensity/`q̂`/support
before any reading. **Consequence for the synthesis:** the capstone's value estimate is not a coherent read at
this scale; the frontier's FQI rows carry the caveat.

#### Exploitability axis (`E-cheap` / `E-costly`)

**`E-cheap` — value gained without exploitability cost.** The FQI policy's exploitability is at or below
behavior's while its OPE value is above — a policy that is both better *and* no more predictable. This is the
happy case and would be a genuinely strong prescriptive result; it is **not** the pattern the synthetic
validation shows, so treat an `E-cheap` real-data reading with the same support checklist as `CERTIFIED`.

**`E-costly` — value bought with predictability (the verified pattern).** The FQI policy's exploitability is
above behavior's (the synthetic worlds show `~3×`), because a concentrated policy lets the fixed batter-
response model sit on it. Value is bought with predictability. The cross-reading with the batter's-side story
is the study's game-theoretic bottom line: an optimised pitcher who always throws the model's pick is more
forecastable, and a batter who guesses along claws back a measurable share — model-dependent by necessity
(§8), but the *direction* (concentration costs exploitability) is robust to the model. **Consequence for the
synthesis:** the frontier is a genuine multi-objective trade-off, not a single-number leaderboard — the
capstone's summary is a Pareto surface, and the "best" policy depends on how a user weights value against
predictability.

#### Reading the grid

The honest headline is a pair `(verdict, exploitability)`. The study's *most anticipated* real-data cell is
**`RL_EVIDENCE_DIRECTIONAL` × `E-costly`** at moderate scale — a real, world-discriminating sequencing signal
below the OPE certification floor, bought at a predictability cost — the capstone's own synthetic reading,
lifted to real data. The *strongest* cell is **`RL_EVIDENCE_CERTIFIED` × `E-cheap`**: a certified sequencing
edge with no exploitability penalty, cross-checked against WS3/WS4/WS5. The *null* cell is
**`RL_NO_CLAIM`/`RL_EVIDENCE_ABSENT`**. The *stop* cell is **`RL_INCONCLUSIVE`** (diagnose before reading).

---

## 7. Discussion

**The capstone-arc finding (WS4 → WS5 → WS7).** Each prescriptive rung's honest negative is the next rung's
motivating contrast, and together they make one coherent finding. WS4's greedy bandit was **structurally
blind** to setups — a one-step value cannot represent an action whose payoff is a future state — and it
*measured* the ceiling (`~0.003` of `~0.032`, decision D40). WS5's tabular MDP proved the setup is
**representable** (a constructed-world test cashes it) and directionally recovered it, but bound its
*certification* to the sequential-OPE variance (decision D44). WS7's flexible FQI **discriminates the worlds
robustly** (the O-vs-count isolation separates positive from null at every scale) yet is **more variance-bound
than the tabular MDP at equal scale**: its LightGBM function approximator, which buys representational reach
(no hand-designed trigger flag), carries more estimation variance into the OPE, so its isolation CI is *wider*
than WS5's and it cannot cash the small setup credit WS5's low-variance tabular design recovered. The
consistent conclusion across all three rungs is: **sequential value exists and is representable; certifying it
is a statistics-of-scale problem** — and that is the study's Phase-2 question, stated precisely.

**Flexibility costs variance — a general finding.** The arc's mechanism is worth stating as a portfolio
result. Moving up the model-complexity ladder (bandit → tabular MDP → flexible FQI) monotonically increases
*representational* reach and, in a finite-sample OPE, *estimation variance*. WS5's tabular trigger-MDP has the
lowest variance because its state is coarse and its FQE is an exact averager; WS7's FQI has the highest because
its state is rich and its `Q` is a fit. So the paradox that the *fanciest* model is the *least* certain at a
fixed data scale is not a bug — it is the bias-variance trade-off appearing on the OPE side of the ledger, and
it is the reason the study's certification bottleneck is data scale, not model class. The right model for a
*certification* at moderate scale is often the *simplest one that can represent the effect*; the flexible model
earns its place at full scale.

**The frontier as the study's summary object.** WS7's deliverable is not a run gain but the frontier figure,
and that is deliberate. The whole ladder — descriptive grammar (WS1/WS2), predictive ablation (WS3/WS6),
myopic prescription (WS4), tabular sequential prescription (WS5), flexible sequential prescription (WS7) —
resolves into a single multi-objective picture: what OPE value each policy buys, at what predictability
(`B_seq`), what exploitability, what deviation from behavior, and what compute. No scalarisation is imposed
(`THEORY.md` §7): the frontier is a Pareto surface, and the study's honest bottom line is not "throw the
slider" but "here is the trade-off, and here is how far each rung of rigor gets you before the error bars
close in."

**The count-driven baseline, one last time.** As in WS4 and WS5, a flexible policy beats the habit-based
behavior policy in raw value for count reasons (value is dominated by the count; Tango et al., 2007), which is
*not* sequencing. The O-vs-count isolation is the instrument that separates the setup value from the count-
driven gain, which is why the capstone leads with the isolation and prints the count-driven explanation
whenever a raw gain appears. It would have been easy — and wrong — to headline the `+0.035` raw improvement;
the isolation principle is precisely the discipline that prevents it.

**The firewall.** WS7 tests whether *acting* on the FQI value beats behavior *within support*; it does not
certify the ordered state as *causal*. `RL_EVIDENCE_CERTIFIED` is evidence the ordered state yields a better
*evaluable* policy, not proof that *changing* the sequence *causes* the gain — the finding-#3 claim public data
cannot fully satisfy (`../ws3_gbdt_stack/THEORY.md` §8). The capstone approaches finding #3 more closely than
any other rung, with its support diagnostics and estimator agreement, but it does not cross the causal line,
and it says so.

---

## 8. Limitations

1. **The exploitability model is model-dependent by necessity (SPEC §10).** The batter-response model is one
   fixed, interpretable opponent — a beta-smoothed `P(swing | count, family)` marginalised over location. It
   is *not* adaptive, *not* a learned best-responder, and *not* causal; the anticipation penalty `β = 0.12`
   and the disruption mechanism are modelling choices. The exploitability *levels* would move under a different
   response model; SPEC §10 asks for robust/worst-case reporting, and the robust claim is the *direction* —
   concentration costs exploitability — not the exact `~3×`.
2. **The penalty form.** The support penalty is the indicator `1[μ̂(a|s) < 0.02]`, a hard behavior-probability
   floor, chosen because it *is* the SPEC §9 diagnostic. It is a CQL-lite *approximation*, not the CQL
   regulariser; the smooth count-based alternative is documented but not used. The floor and `λ = 0.05` are
   config choices, reported openly through the value-vs-λ exhibit.
3. **`γ = 1` and the finite horizon.** The return is the undiscounted PA sum (SPEC §5), which makes `γ = 1`
   correct, not convenient (`../ws5_tabular_mdp/THEORY.md` §1); the short PA horizon is why `K = 3` backups
   reach the fixed point. A domain with long or non-terminating episodes would need discounting and a
   contraction argument the PA structure makes unnecessary here.
4. **Synthetic-scale OPE floors bind certification, not representability.** The `~0.003` ceiling is swallowed
   by the isolation's refit-bootstrap CI at synthetic scale; whether the real-data isolation resolves above it
   is the open Phase-2 question, and the `n`-to-certify arithmetic (`THEORY.md` §8) is a back-of-envelope, not
   a guarantee. A `DIRECTIONAL` at fixture scale does not preclude a `CERTIFIED` at full scale.
5. **Flexibility's variance is the binding constraint.** The FQI's LightGBM function approximation makes its
   OPE noisier than WS5's tabular MDP, so at equal scale WS7 is *less* able to certify the setup than the
   simpler model. This is a property of the model class in finite-sample OPE, measured, not a shortfall of the
   implementation.
6. **Deferred variants (SPEC §12).** Full CQL and model-based RL are deferred to Phase 2 (the highest model-
   error risk); WS7 ships the CPU-practical conservative FQI. A full stochastic-game solver for the
   exploitability capstone is likewise deferred — the read-out is the light game-theoretic version.
7. **Family granularity and the `pitch_type` proxy.** The action is the pitch family (SPEC §4) and
   `pitch_type` is a classifier output (SPEC §1); a mechanism below the family level, or a mislabeled pitch,
   is only partially represented — inherited from WS3.
8. **One reward metric.** Value is `−delta_run_exp`; a different reward could reweight the `q̂` grid, the
   payoff games, and every number.

---

## 9. Conclusion

WS7 is the study's capstone and its most humble unit: the most flexible model in the portfolio, wearing the
most safety gear. It builds a conservative fitted-Q iteration with a behavior-support pessimism penalty,
subjects it to the OPE-before-policy contract *twice* (behavior recovery and a logged-bandit regression),
scores its softened policy through the full SPEC §9 battery with WS5's refit-bootstrap FQE, and rests its
verdict — as a sanctioned strengthening of decision D50 — on the **O-vs-count isolation** rather than the
count-driven raw gain. Its conclusion is branch-conditional and complete once the real numbers arrive:

- **If the isolation clears the ceiling with a real CI (`RL_EVIDENCE_CERTIFIED`)**, the flexible sequential
  policy has cashed ordered-history value beyond the count-driven gain and the myopic ceiling — a strong
  claim, cross-checked against WS3/WS4/WS5 and read against its exploitability cost.
- **If the isolation is directional but uncertifiable (`RL_EVIDENCE_DIRECTIONAL`)** — the fixture's own
  reading and the most anticipated real-data outcome — the honest content is that sequential value is real and
  world-discriminating but below the OPE certification floor, with the flexibility-costs-variance finding and
  the `n`-to-certify arithmetic as the falsifiable Phase-2 prediction.
- **If there is no directional isolation (`RL_NO_CLAIM` / `RL_EVIDENCE_ABSENT`)**, WS7 claims nothing beyond
  the count-driven gain — read as no prescriptive sequencing edge at this scale, never as "order hurts."
- **If the estimators disagree (`RL_INCONCLUSIVE`)**, the value is not a coherent read and the workstream says
  so, verbatim per SPEC §9.

Across every branch the durable contributions are the same: a gates-first conservative FQI that never reports
a value it cannot recover the baseline for; the isolation principle that separates sequencing from the count-
driven gain at the top of the ladder; the exploitability read-out that measures the `~3×` predictability cost
of concentration; the study frontier that resolves the whole ladder into one multi-objective figure; and the
capstone-arc finding that sequential value is real and representable, its certification a statistics-of-scale
problem, with flexibility costing variance in OPE. WS7's honest `DIRECTIONAL` is the capstone telling the
truth about its own confidence — greedy could not see the setup, the tabular model saw it but could not prove
it, the flexible model discriminates it but is noisier still — and everything points to the same place: real
but small, needs the full data.

---

## References

Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

Efron, B., and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal of
the American Statistical Association*, 70(350), 311–319.

Ernst, D., Geurts, P., and Wehenkel, L. (2005). Tree-based batch mode reinforcement learning. *Journal of
Machine Learning Research*, 6, 503–556.

Fujimoto, S., Meger, D., and Precup, D. (2019). Off-policy deep reinforcement learning without exploration.
*Proceedings of the 36th International Conference on Machine Learning (ICML)*.

Jiang, N., and Li, L. (2016). Doubly robust off-policy value evaluation for reinforcement learning.
*Proceedings of the 33rd International Conference on Machine Learning (ICML)*.

Kumar, A., Zhou, A., Tucker, G., and Levine, S. (2020). Conservative Q-learning for offline reinforcement
learning. *Advances in Neural Information Processing Systems (NeurIPS) 33*.

Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020). Offline reinforcement learning: tutorial, review, and
perspectives on open problems. *arXiv:2005.01643*.

Marchi, M., and Albert, J. (2013). *Analyzing Baseball Data with R*. Chapman and Hall/CRC.

Precup, D., Sutton, R. S., and Singh, S. (2000). Eligibility traces for off-policy policy evaluation.
*Proceedings of the 17th International Conference on Machine Learning (ICML)*.

Riedmiller, M. (2005). Neural fitted Q iteration — first experiences with a data efficient neural
reinforcement learning method. *Proceedings of the 16th European Conference on Machine Learning (ECML)*.

Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, second edition. MIT Press.

Tango, T. M., Lichtman, M. G., and Dolphin, A. E. (2007). *The Book: Playing the Percentages in Baseball*.
Potomac Books.

Thomas, P. S., and Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement
learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*.

von Neumann, J. (1928). Zur Theorie der Gesellschaftsspiele. *Mathematische Annalen*, 100, 295–320.
