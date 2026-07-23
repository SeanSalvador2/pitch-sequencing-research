# A Tabular MDP That Values the Setup Pitch, and the Sequential Gate That Keeps It Honest

*Workstream 5 of a comparative pitch-sequencing study — the first sequential prescriptive rung (Phase B).*
This paper is completed in place with the real-data results (2021–2025 Statcast). The Results and Discussion
were pre-**branched** so that the correct interpretation was already written for whichever numbers arrived;
the branch **selected by the data** is marked at each fork, and the unselected branches are retained and
labelled as *pre-registered alternatives*. The synthetic-world numbers quoted in §5 are *completed
validation* from the committed WS5a run, not placeholders.

---

## Abstract

Workstream 5 (WS5) is the study's first **sequential** prescriptive rung and its transparent simulator. It
builds a small **tabular Markov decision process** (MDP) over the count and a sliver of ordered history,
estimated from logged transition and reward counts, planned with undiscounted policy iteration, and used
two ways: as a **model-based** value source and as a **policy** whose softened target is scored through the
shared off-policy-evaluation (OPE) gate (`eval/ope`). It is the first rung that can value a **setup pitch** —
an action whose payoff is not its own reward but the *state it creates for the next pitch*. The state ladder
mirrors the study's `C / L1 / O` measurement views in state space (decision D43): `count` (`(balls,
strikes)`, 16 states), `count_prev` (`+` previous family, 112), and `count_prev_trigger` (`+` a leakage-safe
velocity-gap **trigger flag**, 220). The trigger flag makes the setup effect *representable*: from a state
with previous family `p`, choosing a current family whose velo band differs from `p`'s by `≥ τ = 5` mph
drives the next state's trigger to `1`, and a triggered state carries the planted whiff boost — so policy
iteration values actions for the trigger they *create*, which the coarser designs cannot express. The
softened target `π_α = (1−α)μ + απ_greedy` (SPEC §9) is scored three ways per `(design, α)` — the exact
model-based MDP value, held-out step-wise DR, and held-out fitted-Q evaluation (FQE) — with agreement judged
on real CIs (decision D42, the OPE cross-validation) and divergence **reported, never silently averaged**.
On the real data (2021–2025 Statcast; 3,567,640 regular-season decisions, of which WS5 estimates the MDP on
the 2,143,214 train-fold rows and scores 1,414,477 held-out rows across 364,077 plate appearances) the
behavior recovery is **PASS** (observed `+0.0002`, IPS weights unit), so the OPE gate is trustworthy; the
behavior value is `V(μ) = +0.0002`; and at the top `α = 1` the trigger-count value gap is FQE `+0.0130`
(refit-bootstrap 95% CI `[+0.0076, +0.0188]`, one-sided lower bound `+0.0085`) — which, *in the FQE lens
alone*, clears the D40 myopic ceiling `+0.003`. The verdict is nonetheless `SEQ_NEUTRAL_MDP`: the
setup-capable `count_prev_trigger` design earns **no certifiable advantage** over plain `count` once
estimator agreement and support are enforced, because the triple gate fails on two counts — the step-wise-DR
gap disagrees violently at `α = 1` (`−0.0989`, ESS `6.3%`, off-support) and the **setup Q-gap is negative**
(mean `−0.0120`, median `−0.0093`, positive on only `0.30` of the `n = 135` reachable trigger states). That
negative Q-gap is the interesting real-data finding: the estimated MDP concludes that deliberately creating a
large consecutive-pitch velocity differential is, on average, **not worth it** on real MLB data — the
opposite sign of the synthetic positive world (`+0.0087`, `0.68`). The D42 cross-check `DIVERGES` at mid-`α`
on every design (the tabular MDP is optimistic where cells are thin — greedy `count_prev_trigger` model-based
`+0.0206` vs held-out FQE `+0.0179`, optimism `+0.0027`), which is why the lone FQE-positive signal cannot be
trusted on its own. This is consistent with WS4 (a myopic bandit does not beat observed behavior) and WS3 (a
statistically-real but tiny out-of-sample outcome-order effect): even a model that *can* value a setup finds
no certifiable setup value for the one representable mechanism, and WS7's flexible offline RL is the final
word. *(Cosmetic caveat: the tool prints the `SEQ_NEUTRAL_MDP` message reusing the synthetic branch's "null
world … trigger flag inert" wording; on real data the substantive meaning is "no certifiable setup value,
with a negative setup mechanism," not a literally null world — the verdict class and the triple-gate logic
are correct.)* We validate the method on the correctness oracle with a completed two-world
study. On the **null** world the verdict is `SEQ_NEUTRAL_MDP`: no `α` clears the gate, the trigger flag is
inert (setup Q-gap mean `+0.0020`, positive fraction `0.49`), and the in-sample model-based values are
visibly optimistic against their held-out FQE values — the greedy-optimism exhibit. On the **positive**
world (a planted velocity-transition whiff boost) the honest verdict is the first-class negative
`SETUP_INCONCLUSIVE`: at `α = 1` the trigger-count gap is FQE `+0.0079` (CI `[−0.0081, +0.0218]`), step-wise
DR `+0.0107` (CI `[−0.061, +0.077]`), and model-based `+0.0277` — **all three directionally positive, none
certifiable** — while the setup diagnostics cleanly separate the worlds (Q-gap mean `+0.0087`, positive
fraction `0.68`; optimal action changes on `48.2%` of trigger states vs `42.6%` under the null). A
constructed-world unit test **proves** the machinery cashes a setup (the trigger design's optimal policy
takes the setup action and its simulated value is provably higher); certification on this fixture is deferred
to Phase-2 data scale. The paper's methodological payload is a hardening finding recorded as decision D44:
the first shipped gate rested on a **structurally degenerate** FQE confidence interval — every plate
appearance starts in the identical state, so FQE's per-episode contributions `V(s₀)` are constant and a
cluster-resampling bootstrap of them collapses to a point. The fix is a **paired FQE refit cluster
bootstrap** (resample episodes, refit FQE from scratch, pair across designs), and the lesson — echoing WS4's
myopic ceiling one rung up — is that in sequential OPE the **variance**, not the point estimate, binds a
prescriptive claim. WS5's contributions are the setup-capable state design, the paired-refit sequential gate,
the D42 triple-lens cross-validation, and the honest-INCONCLUSIVE finding at synthetic scale.

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

Workstreams 1–3 (Phase A) live in findings #1 and #2. WS4 opened Phase B — prescription — and delivered a
disciplined honest negative: a myopic bandit's best exploitable edge on the positive fixture is `~0.003`
run, below the OPE noise floor, because the planted effect is a **state-value** effect a one-step value
cannot manufacture. WS4 turned that into a falsifiable target for the sequential rungs (decision D40): *beat
`0.003` by valuing the setup a bandit provably cannot.* **WS5 is the first model that can hold that target.**

WS5 is the simplest *sequential* prescriptive model: a tabular MDP whose action-value keeps the term a
bandit drops,

$$Q(s, a) = \mathbb{E}[R \mid s, a] + \mathbb{E}\!\left[v(s') \mid s, a\right],$$

where the second piece — today's action shifting the *next* state's baseline value — is exactly the setup
credit. The MDP is made *setup-capable* by a single engineered state variable, a leakage-safe velocity-gap
**trigger flag** (D43): choosing a current family whose velocity band differs enough from the previous
pitch's drives the next state's trigger to `1`, and a triggered state carries higher reward, so planning
values the trigger a pitch *creates*. This is the same `C / L1 / O` ladder as the rest of the study, moved
from feature space into state space.

The discipline SPEC §0.3 imposes is absolute and shapes the whole workstream:

> **OPE before policy.** Build and self-test the off-policy-evaluation harness (it must recover the observed
> policy's value) *before* optimizing any policy. Otherwise you produce recommendations you cannot evaluate.

So WS5 does not report a run gain. It constructs one transparent MDP, softens its policy toward behavior,
and subjects it to the OPE gate — behavior recovery first, then the setup gap scored against the D40 ceiling
with a **real** confidence interval. The chapter has two payloads, and they are deliberately different in
kind. The first is that the setup is **representable and cashable in principle**: a constructed-world unit
test proves the trigger design's optimal policy *sets up* and wins. The second is that **certification is
variance-bound**: on the full fixture the held-out gap is directionally positive in all three lenses but its
refit-bootstrap interval does not clear the ceiling, an honest `SETUP_INCONCLUSIVE` (decision D39). The
binding constraint is the sequential-OPE variance (decision D44), and whether it can be cleared is a
data-scale question Phase 2 answers.

**Contributions.**

1. **A setup-capable tabular state design (D43).** The `count / count_prev / count_prev_trigger` ladder,
   with a leakage-safe velo-gap trigger flag computed from prior-pitch execution speeds only, that makes the
   setup effect *representable* — from previous family `p`, an action whose velo band differs by `≥ τ` drives
   the next state's trigger, and policy iteration values it. A constructed-world unit test proves the design
   cashes a setup the coarser designs structurally cannot.
2. **The paired-refit sequential gate (D44).** The recognition that FQE's per-episode contribution `V(s₀)`
   is constant in a constant-initial-state domain — so a resampling bootstrap of it is structurally
   degenerate — and the fix: a **paired FQE refit cluster bootstrap** that resamples episodes, refits FQE
   from scratch, and pairs across designs, plus a triple-condition gate (refit-FQE lower-95 above the ceiling
   **and** step-wise-DR gap directionally positive **and** model-based gap positive). Degenerate CIs print
   `n/a (constant contributions)`, never as intervals.
3. **The D42 triple-lens cross-validation.** Per `(design, α)`, the *same* softened policy scored by the
   estimated MDP, step-wise DR, and FQE, with pairwise agreement on real CIs (decision D24) — divergence
   reported as a misspecification diagnostic, never averaged. This is SPEC's "transparent simulator used to
   validate the OPE code," made into a per-cell audit.
4. **The honest-INCONCLUSIVE finding at synth scale.** A completed two-world validation whose positive-world
   verdict is `SETUP_INCONCLUSIVE`: real, representable, directionally agreed, and statistically
   uncertifiable at fixture scale — with the `n`-to-certify arithmetic that turns it into a falsifiable
   Phase-2 prediction rather than a null.

---

## 2. Related work

**Dynamic programming and the MDP.** The formal object is a finite Markov decision process (Bellman, 1957;
Puterman, 1994): a state space, an action space, a transition kernel, a reward, and a return. WS5's return
is the undiscounted sum of per-pitch rewards over a plate appearance, which by SPEC §5 is the total
run-value change — so `γ = 1` is the *correct* choice, not a convenience (discounting would distort the
run-value semantics), and it is finite because every plate appearance terminates with probability one. The
planning algorithm is **policy iteration** (Howard, 1960): alternate exact policy evaluation — here a linear
solve of `(I − P_π)V = R_π` on the non-terminal block — with greedy improvement over the feasible actions,
which converges to the optimal policy in finitely many steps for a finite MDP. The textbook synthesis of the
value/return, Bellman-equation, and policy-iteration machinery WS5 uses is Sutton and Barto (2018).

**Off-policy evaluation for sequential problems.** WS5 evaluates full plate appearances, so its OPE is
sequential, not bandit. The step-wise doubly-robust estimator it leads with is the per-decision DR of Jiang
and Li (2016), a backward recursion with per-step importance ratios that is unbiased if either the
propensities or the fitted return-to-go is correct; the per-decision importance-sampling device it rests on
is Precup, Sutton, and Singh (2000). The complementary lens, **fitted-Q evaluation** (FQE), and the broader
program of data-efficient, high-confidence sequential OPE are Thomas and Brunskill (2016); WS5 consumes both
through WS0's `eval/ope` and never re-implements them. The workstream's central caution — that a conservative
policy must stay near logged behavior and be evaluated with honest support diagnostics and confidence bounds
— is the batch-constrained framing of Le, Voloshin, and Yue (2019); we cite it precisely for that
constraint-and-confidence discipline, not for its policy-learning algorithm, which WS5 does not use (WS5's
policy is planned, not learned by constrained optimization).

**Shared project grounding.** The Dirichlet/empirical-Bayes shrinkage the study leads with (Efron and Morris,
1975) is inherited here in the smoothed transitions and the two-level-shrunk reward table (the WS1 lemmas,
`../ws1_eb_tables/THEORY.md`). The applied grounding — that pitch value is dominated by the **count** — is
Tango, Lichtman, and Dolphin (2007), which is why the `count` design is a serious baseline and the ladder
must beat it to claim sequencing value; the baseball-data mechanics are Marchi and Albert (2013).

**Internal cross-references.** WS5 reads WS0's shared decision table, splits, and OPE machinery — built and
self-tested against analytic truth before any workstream optimized a policy (SPEC §0.3) — and re-runs the
behavior-recovery self-test in-pipeline. It optionally consumes WS3's contextual behavior propensities
`μ(a | s)` as the OPE denominator (decision D33; without a `--ws3-dir`, it fits the state-conditional
empirical behavior from train counts). Its positive-world acceptance test is WS4's measured **myopic
ceiling** (`~0.003` run, decision D40); the `E[R | s, a] = v(s) + δ(s, a)` decomposition that establishes
the ceiling is derived in `../ws4_bandit/THEORY.md` §6 and extended here into the sequential `E[v(s') | s,
a]` setup term (§4.1, `THEORY.md` §4). The finding-#2 / finding-#3 firewall around a conditional expectation
is WS3's (`../ws3_gbdt_stack/THEORY.md` §8), restated in §7.

---

## 3. Data

WS5 reads **one thing** and optionally a second: the shared decision table (SPEC §3) and, optionally, WS3's
saved behavior propensities. It trains nothing heavy and re-implements no evaluation.

**The decision table and the reward.** One row per pitch, the decision made immediately *before* that pitch,
keyed by `(game_pk, at_bat_number, pitch_number)`. WS5 estimates the MDP on the train fold (2021–2023) and
evaluates on the same held-out rows the other workstreams scored — validation (2024) and the locked test
(2025). The reward is `R = −delta_run_exp`, larger better for the pitcher (SPEC §5); the plate appearance
(`pa_id`) is the **episode**, and its undiscounted return `Σ_t R_t` is the run-value change the OPE
estimators score.

**The state ladder (D43).** Three nested tabular designs encode each decision row's pre-pitch state:

- `count` — `(balls, strikes)`, 12 non-terminal cells (the `C` analogue);
- `count_prev` — `×` the previous family (8 families `+ NONE` = 9), 108 non-terminal cells (`L1`);
- `count_prev_trigger` — `×` a velo-gap trigger flag (2), 216 non-terminal cells (an `O`-lite analogue).

Each design appends four **absorbing terminals** — `walk`, `strikeout`, `hbp`, `in_play_end` — for 16, 112,
and 220 total states. The **trigger flag** for the current decision is
`1[ t ≥ 3 ∧ |exec_release_speed_{t−1} − exec_release_speed_{t−2}| ≥ τ ]` with `τ = 5` mph — computed from
the two prior pitches' execution speeds only, so it is leakage-safe (SPEC §0), and it is exactly the planted
positive-world mechanism.

**The action space.** The action is the pitch **family** (8 classes, SPEC §4); `XX` (other/rare) is
descriptive-only and never feasible. The per-state feasible action set aggregates the leakage-safe SPEC §4
per-row masks (a family is feasible only if the pitcher threw it `≥ 50` times and `≥ 3%` over a trailing
window ending before the current game) by a majority-share rule, computed on the **full** table's trailing
window and subset to the training rows.

**The behavior policy (the OPE denominator).** With `--ws3-dir`, WS5 reuses WS3's contextual GBDT
propensities of the richest ablation view (decision D33); without it, the state-conditional empirical
behavior `μ̂(a | s) = (N(s,a) + α_b)/(N(s) + α_b A)` from train counts. Both are floored so importance ratios
stay finite (SPEC §9).

**Temporal splits and clustering.** Train 2021–2023, evaluate 2024 + locked 2025 (SPEC §7). All confidence
intervals are bootstrapped over **pitcher-game** clusters; the FQE gap CIs are a paired refit bootstrap over
those clusters (§4.5).

---

## 4. Methods

### 4.1 The state designs and why the trigger makes a setup representable (D43)

Each decision row is encoded to a non-terminal state id by a fixed scheme: `count` → `balls·3 + strikes`;
`count_prev` → `count_id·9 + prev_idx`; `count_prev_trigger` → `(count_id·9 + prev_idx)·2 + trigger`
(`encode_states`). Successor states are the next pitch's encoded state within the PA, or the absorbing
terminal implied by the last pitch's outcome (`terminal_type`: an in-play ball → `in_play_end`, a hit-by-
pitch → `hbp`, ball four → `walk`, strike three → `strikeout`; fouls never terminate).

The setup is representable **only** in the trigger design. Write the sequential action-value
`Q(s, a) = E[R | s, a] + E[v(s') | s, a]`. A setup pitch is one whose *own* reward `E[R | s, a]` is
unremarkable (or slightly negative — the immediate cost of a show-me pitch) but whose choice raises the next
state's baseline value `v(s')`, because it drives the next trigger. The `count` design's state does not carry
the trigger, so `E[v(s') | s, a]` is constant across its actions and the setup term cancels in the arg-max —
it *cannot* value a setup. The trigger design's transition kernel `P̂(s' | s, a)` carries the exact next
previous family (`prev' = a`) and the resulting velo-gap `trigger'`, so choosing a family whose band differs
from `p`'s by `≥ τ` moves probability mass onto triggered next-states, raising `E[v(s') | s, a]`. This is the
sequential extension of WS4's `E[R | s, a] = v(s) + δ(s, a)` ceiling decomposition
(`../ws4_bandit/THEORY.md` §6): the `E[v(s') | s, a]` variation is exactly the value WS4's `γ = 0` dropped
(`THEORY.md` §4).

### 4.2 Estimation (Dirichlet-smoothed transitions, two-level-shrunk rewards)

**Transitions.** With counts `N(s, a, s')` and the per-source-state reachable set `succ(s) = {s' :
N(s, ·, s') > 0}` of size `K_s`,

$$\hat P(s' \mid s, a) = \frac{N(s, a, s') + \alpha_t}{N(s, a) + \alpha_t\,K_s}, \qquad s' \in succ(s),$$

zero elsewhere (Dirichlet smoothing over the reachable set, `α_t = 1`). The dominant mass is the observed
`N(s, a, s')`, which carry the exact `prev'` and `trigger'`, so the setup mechanism is preserved; `α_t` only
regularises thin cells. A never-taken `(s, a)` backs off to the state's action-marginal transition; an
unobserved state routes to `in_play_end`; terminals self-loop.

**Rewards (two-level shrinkage).** The state mean is shrunk toward the global mean `r̄`, then the cell mean
toward the state mean:

$$\hat r(s) = \frac{\sum_{i:s_i=s} R_i + \alpha_r\,\bar r}{n_s + \alpha_r}, \qquad
  \hat R(s, a) = \frac{\sum_{i:(s_i,a_i)=(s,a)} R_i + \alpha_r\,\hat r(s)}{n_{s,a} + \alpha_r}.$$

Thin cells fall back smoothly to the state mean, then the global mean (no NaNs); terminals carry `R = 0`.
`α_r` is tuned so the richer designs generalise (`_SYNTH_ALPHA_R = 12` on the synthetic worlds).

### 4.3 Planning (undiscounted policy iteration, γ = 1)

Policy evaluation solves the Bellman equation exactly. For a stochastic policy `π`, form `R_π = (π ⊙ R)·1`
and `P_π = Σ_a π(·, a) P(·, a, ·)`, pin terminals at `V = 0`, and solve the linear system on the
non-terminal block `nt`:

$$(I - \gamma\,P_\pi)\big|_{nt}\; V\big|_{nt} = R_\pi\big|_{nt}, \qquad Q = R + \gamma\,P V.$$

Because every PA terminates w.p. 1, `I − γP_π` is non-singular at `γ = 1` and the solve is exact (a singular
solve falls back to iterative fixed-point evaluation, unreachable in practice). Policy iteration
(`policy_iteration`) initialises with the myopic greedy-on-reward policy and alternates this exact evaluation
with greedy improvement restricted to each state's feasible actions until the policy is stable — guaranteed
in finitely many steps for a finite MDP (`THEORY.md` §3).

### 4.4 Softening and the transparent simulator (D42)

The greedy target is softened toward behavior along the SPEC §9 mixture `π_α = (1−α)μ + απ_greedy`, built at
the row level via `ope.pi_alpha` (WS5 does not re-implement it); `α = 0` is behavior, `α = 1` the pure
target. A temperature-softened Boltzmann variant `π(a | s) ∝ exp(Q(s,a)/T)·1[feasible]` is reported as a
secondary. The **transparent simulator** (`simulate`) rolls the policy forward in the estimated MDP; its
Monte-Carlo return converges to the exact `Σ_s start_dist[s] V(s)`, and that self-consistency is the D42
validation of the model-based value.

### 4.5 The FQE refit cluster bootstrap — a real CI in a constant-initial-state domain (D44)

FQE's per-episode contribution is the initial-state value `V(s₀) = Σ_a π(a | s₀) q̂(s₀, a)`. In this domain
**every plate appearance starts in the identical state** — `0-0` count, no previous pitch, trigger `0` — so
the per-episode contribution array is a **constant**, and a cluster-resampling bootstrap of it is
structurally degenerate: every resample has the same mean, and the CI collapses to a point. The randomness a
real CI must capture is instead **estimation** uncertainty — which held-out episodes were logged, and hence
the fitted `q̂`, not the start state. The fix (`_fqe_refit_bootstrap`) is a **paired refit cluster
bootstrap**:

> For each of `--fqe-boot` replicates, resample pitcher-game clusters of episodes with replacement (relabel
> `pa_id` per instance, so a twice-sampled episode is two episodes) and **refit FQE from scratch** on the
> resampled episodes — for **every** design and `α`, using the *same* resample across arms so the design
> *gaps* difference away the shared episode-sampling noise (a paired bootstrap). Read percentile CIs and the
> one-sided 95% lower bound off the replicate distribution.

The refits are exact tabular fits through `onehot_tabular_regressor`, numerically identical to
`ope.tabular_regressor` but orders of magnitude faster (decode the one-hot blocks to integer codes and use
`bincount`), supplied through `ope.fqe`'s documented `regressor_factory` — FQE itself is never
re-implemented. Any CI that is still structurally degenerate (e.g. the step-wise-DR gap at `α = 0`, whose
paired contributions are identically zero) prints `n/a (constant contributions)`, never a fake interval. The
step-wise-DR estimator keeps its per-episode contribution bootstrap (those contributions carry real
variance), but its per-PA importance-weight product is irreducibly noisy for a target far from behavior, so
the verdict uses it **directionally**, with the refit-bootstrap FQE CI as the resolving instrument.

### 4.6 The triple-condition gate (D40 / D43 / D39)

`SETUP_EXPLOITED` fires when, at some `α`, **all three** hold: (1) the refit-bootstrap FQE trigger-count
gap's one-sided 95% lower bound exceeds the D40 ceiling `+0.003`, with a **non-degenerate** CI; (2) the
step-wise-DR gap is directionally positive at that `α` (a sign check — its per-PA weight-product CI is too
wide to resolve the ceiling); and (3) the model-based gap is positive. `SEQ_NEUTRAL_MDP` is the null world's
result (no `α` clears the gate). `SETUP_INCONCLUSIVE` is the D39 first-class fallback on a positive world
where the triple condition is not met at the tested scale: the headline then prints the full evidence story
— the directional agreement of the FQE point, step-wise-DR point, and model-based gap, the setup
diagnostics, and the constructed-world proof — and certification is deferred to Phase-2 data scale. The
gate's logic is a **conjunction that controls the false-positive direction**: each lens fails differently (a
degenerate or wide CI; a sign flip; an unfavorable in-sample value), and requiring all three to point the
same way with a real interval makes an accidental clearance improbable (`THEORY.md` §8).

### 4.7 The D42 per-`(design, α)` cross-check

For each `(design, α)` the *same* softened policy `π_α` is scored by the model-based value (an exact point,
half-width 0), step-wise DR (contribution-bootstrap CI), and FQE (refit-bootstrap CI). A pair **diverges**
iff the absolute difference exceeds the wider of the two 95% CI half-widths (decision D24, on the real CIs);
a pair where neither side has a usable CI is unassessable and marked `None`; the cell verdict is `DIVERGES`
if any assessed pair diverges, else `CONSISTENT`. Divergence is **reported, never silently averaged** — it
is a misspecification diagnostic (the tabular state's in-sample optimism at thin cells), and it is why the
gate leans on the held-out FQE gap, not the in-sample model value. The in-sample **greedy-optimism** exhibit
(model-based greedy vs its own held-out FQE value) is the same phenomenon at its extreme, reported as a
separate, clearly-labeled figure, never a verdict input.

---

## 5. Experimental setup

**Scoring.** Every design's greedy target is written in the standard schema (SPEC §8.1) and the softened
targets are scored through `eval/ope` (step-wise DR + FQE) with the behavior-recovery gate first (SPEC §0.3).
The exhibits are the state-space summary (states, reachable, mean feasible actions), the D43 ladder × D42
cross-check table (model-based / step-wise DR / FQE / ESS per design × `α`), the greedy-optimism exhibit,
the setup gap vs the D40 ceiling with the refit-bootstrap CI, and the setup diagnostics.

**The completed two-world validation.** WS5 reruns the SPEC §11 oracle as a real-model sequential test. The
following are *observed results* on the committed WS5a run (synthetic scale, `n_games = 2000`), not
aspirations. Both worlds pass the gate: behavior recovery passes with IPS weights ≡ 1 at `α = 0` (observed
`+0.0352` positive, `+0.0127` null), so everything below the gate is interpretable.

- **Null world** (an ordered *selection* habit, outcomes depending only on context + current family — no
  ordered *outcome* effect). The trigger flag is inert. The setup diagnostics sit at their noise floor: setup
  Q-gap mean `+0.0020` (median `−0.0006`), positive fraction `0.49`; optimal action changes on `42.6%` of
  the 141 reachable trigger states. The held-out trigger-count FQE gap at `α = 1` is *negative*, `−0.0106`
  (CI `[−0.0299, +0.0095]`), while the in-sample model-based gap is `+0.0250` — the richer design overfits,
  and the model value is optimistic against the held-out lens. No `α` clears the triple gate. Verdict:
  **`SEQ_NEUTRAL_MDP`** — the richer designs are worth no more than `count` up to overfitting cost, the
  expected null result.
- **Positive world** (a planted whiff boost when `|velo_{t−1} − velo_{t−2}| ≥ 5` mph — keyed on the
  transition *into* the prior pitch, per D21, so it separates the trigger design from `count_prev`). The
  setup diagnostics discriminate: setup Q-gap mean `+0.0087` (median `+0.0079`), positive fraction `0.68`;
  optimal action changes on `48.2%` of trigger states. At `α = 1` the trigger-count gap is directionally
  positive in **all three** lenses — FQE `+0.0079` (CI `[−0.0081, +0.0218]`, one-sided lower bound `≈
  −0.004`), step-wise DR `+0.0107` (CI `[−0.061, +0.077]`), model-based `+0.0277` — and the held-out FQE gap
  (`+0.0079`) sits clearly above the null world's (`−0.0106`), so the ladder discriminates the worlds. But
  none of it clears the `+0.003` ceiling with a real CI: the FQE lower bound is negative and the step-wise-DR
  CI is `~±0.07` wide. Verdict: the first-class honest negative **`SETUP_INCONCLUSIVE`** (§6).

**The D44 methods-hardening finding.** The first WS5a build shipped `SETUP_EXPLOITED`. It was wrong — not in
arithmetic, but in **confidence**. FQE's per-episode contribution `V(s₀)` is constant in this
constant-initial-state domain, so the cluster-resampling bootstrap of it returned a zero-width interval `[x,
x]`, and the gate read a directional gap as *certified*. The review caught it, and the fix (the paired refit
bootstrap, §4.5) makes the FQE CI real; the corrected gate then reports `SETUP_INCONCLUSIVE`. The refit cost
is honest and reported: **200 replicates × 3 designs × 5 α = 3,000 FQE refits at ≈ 1.1 s/fit ≈ 57 minutes**,
inside a demo of `~3,850 s` at peak `~1.2 GB`. This is not an embarrassment tucked away; it is the chapter's
methodological content, and it generalises: in sequential OPE the **variance**, not the point estimate, binds
a prescriptive claim. *Realized on real data (2021–2025): the same paired refit bootstrap ran at production
scale — **100 replicates × 3 designs × 5 α = 1,500 FQE refits at ≈ 5.7 s/fit ≈ 8,578 s**, inside a 2.6 h run
at peak 11.3 GB (`--fqe-boot 100`, halved from the 200 default per the RUNBOOK to fit the wall-clock; the CI
resolution coarsens, the point estimates do not). It produced a genuinely non-degenerate FQE CI whose lower
bound even clears the ceiling at `α = 1` — and the honest gate still returned no certification, because the
step-wise-DR lens disagreed and the setup Q-gap was negative (§6). The variance-binds lesson holds one level
deeper on real data: a real interval is necessary but not sufficient; the lenses must also agree.*

**The D42 divergence, read honestly.** At `α = 0` the three lenses are `CONSISTENT` on every design (the
MDP's behavior value matches the held-out behavior value). As `α` rises and the design richens they
`DIVERGE`: at `α = 1` on the richest design the model-based value `+0.0798` sits `0.034` above the FQE value
`+0.0459`, against a tolerance of `0.017` → `DIVERGES`. This is the tabular state's in-sample optimism at
thin cells, reported per D42, and it is *informative*: it says the count-plus-trigger abstraction is not
Markov-sufficient, which is the explicit brief handed to WS7.

**The representability proof (the machinery is not blind).** To prove the near-zero certified gap is the
*fixture's* property and not a broken detector, WS5a ships a constructed-world unit test: a minimal world
where one family (`CU`, slow) at pitch 2 creates pitch 3's trigger while carrying a small immediate cost. The
`count` design's optimal policy avoids `CU` (it sees only the cost); the trigger design's optimal policy
**takes `CU`** (it sees the setup), its start value is provably higher (a clear margin,
`V_trigger − V_count > 0.1` on the fixture), and the transparent simulator reproduces the advantage. A
companion pipeline test confirms the `SETUP_EXPLOITED` path fires on an engineered clear. So the machinery *sees*; the positive world's effect is
genuinely present-but-uncertifiable at synthetic scale.

---

## 6. Results

### 6.1 The central table (real data)

The behavior-recovery gate, the state-space summary, the per-design three-lens table at the top `α`, the
setup gap vs the D40 ceiling across `α`, and the setup diagnostics.

**Data vintage.** 3,567,640 regular-season decisions, 2021–2025 (SPEC's ~3.85M counts all game types; the
config filters to `game_type == "R"`). WS5 estimates the MDP on the **train fold (2021–2023, 2,143,214**
rows), with the behavior policy `μ` taken from WS3's `O`-view propensities (`behavior μ: ws3:O`, decision
D33), and scores the held-out evaluation rows — validation (2024) plus the locked test (2025) — for
**1,414,477 scored decisions across 364,077 plate appearances (episodes)**. All CIs are pitcher-game
clustered; the FQE gap CIs are a paired refit bootstrap. Wall-clock `~2.6 h` (peak RAM 11,314.2 MB).

**Gate (precondition, D37).** Behavior-policy recovery **PASSES**: the OPE harness recovers the observed
held-out value with the IPS importance weights exactly unit (observed `+0.0002`), so every value below is
interpretable (SPEC §0.3). **Behavior value** `V(μ) = +0.0002` (the behavior-recovery observed held-out
value, the `α = 0` baseline).

**The state-space ladder at `α = 1` (per-design three-lens, top `α`).** At the pure target each design's
softened policy *is* its greedy policy, so the model-based and held-out FQE columns are the two ends of the
greedy-optimism exhibit:

| design | states | `α` | model-based | FQE (held-out) | greedy-optimism (MB − FQE) | D42 |
|---|---|---|---|---|---|---|
| `count` | 16 | 1.00 | +0.0091 | +0.0049 | +0.0043 | DIVERGES¹ |
| `count_prev` | 112 | 1.00 | +0.0192 | +0.0170 | +0.0022 | DIVERGES¹ |
| `count_prev_trigger` | 220 | 1.00 | +0.0206 | +0.0179 | +0.0027 | DIVERGES¹ |

¹ MB-vs-OPE `DIVERGES` at mid-`α` on **every** design — the tabular MDP value sits outside the held-out OPE
CIs — and the `α = 1` model-based-above-FQE optimism shown here (sharper on sparser designs: `count` `+0.0043`
vs `count_prev_trigger` `+0.0027`, a max-selection bias) is the same phenomenon at its extreme. *(pending:
per-design absolute step-wise DR, per-design refit-bootstrap CIs, per-design ESS, and reachable-state counts
are not in the results log; step-wise DR and a refit-bootstrap CI are logged for the trigger − count **gap**
only — see the next table — and the `α = 1` evaluation is off-support at ESS `≈ 6.3%`.)*

**Setup gap vs the D40 `+0.003` ceiling (trigger − count, across `α`; FQE CI = paired refit bootstrap; the
triple gate requires FQE lo95 > ceiling AND step-wise-DR gap > 0 AND model-based gap > 0 at the *same* `α`).**

| `α` | FQE gap | refit-bootstrap 95% CI | one-sided lo95 | step-wise DR gap | model-based gap | FQE lo95 clears +0.003? |
|---|---|---|---|---|---|---|
| 0.00 | +0.0001 | [−0.0001, +0.0003] | −0.0000 | −0.0001 | −0.0003 | no |
| 0.10 | +0.0005 | [+0.0001, +0.0008] | +0.0002 | +0.0008 | +0.0008 | no |
| 0.25 | +0.0013 | [+0.0003, +0.0020] | +0.0006 | +0.0045 | +0.0026 | no |
| 0.50 | +0.0036 | [+0.0017, +0.0054] | +0.0020 | +0.0164 | +0.0056 | no |
| 1.00 | +0.0130 | [+0.0076, +0.0188] | +0.0085 | **−0.0989** | +0.0114 | **yes** |

The FQE gap is CI-positive from `α ≥ 0.1`, and its one-sided lower bound clears `+0.003` at `α = 1`
(lo95 `+0.0085`). But **no `α` satisfies the triple gate**: at `α = 1`, where the FQE lens finally clears,
the step-wise-DR gap has flipped hard negative (`−0.0989`, ESS `6.3%` — the target is off-support and the
per-PA weight product is high-variance), so the two held-out estimators disagree exactly where the FQE lens
looks best. The `count_prev − count` FQE gap at `α = 1` is `+0.0121` *(arithmetically `+0.0170 − +0.0049` from
the greedy-optimism FQE points; its paired refit-bootstrap CI is not separately logged)*.

**Setup diagnostics (`count_prev_trigger` MDP).** The create-vs-not setup Q-gap has mean `−0.0120`, median
`−0.0093`, and is positive on only `0.30` of `n = 135` reachable trigger states; the trigger design's optimal
action differs from `count_prev`'s on `36.6%` of `161` states (mean trigger-creating actions per state
`1.86`). The Q-gap is **negative** — the estimated MDP finds that deliberately creating the velocity-gap
trigger is, on average, *not* worth it — the sign-flipped opposite of the synthetic positive world
(`+0.0087`, positive fraction `0.68`; §5). This, not the value column, is the substantive real-data content.

**Refit-bootstrap cost (realized on real data).** 100 replicates × 3 designs × 5 `α` = **1,500 FQE refits**
at `≈ 5.7 s/fit`, `8,578 s` total (`--fqe-boot 100`, halved from the 200 default per the RUNBOOK to fit the
wall-clock; this coarsens the CI resolution but never the point estimates).

**Cosmetic note (verdict wording), a results caveat.** The printed `SEQ_NEUTRAL_MDP` message reuses the
synthetic branch's wording ("null world … trigger flag inert"). On real data that phrasing is misleading: the
verdict **class** (`SEQ_NEUTRAL_MDP`) and the triple-gate logic are correct, but the substantive meaning is
**"the setup-capable design earns no certifiable advantage over `count`, and the one representable setup
mechanism (a large consecutive-pitch velocity gap) shows a negative in-model Q-gap,"** *not* "the world is
null." The real data is not inert — it actively leans against the setup.

**Headline (real data, 2021–2025).** The setup-capable `count_prev_trigger` design earns **no certifiable
setup value**. The interesting finding is the **negative setup Q-gap** (`−0.0120`, positive on only 30% of
trigger states): on real MLB data the estimated MDP judges the one representable setup mechanism slightly
*harmful*, the opposite of the synthetic positive world. The lone FQE-positive signal (lo95 `+0.0085 > +0.003`
at `α = 1`) does **not** survive, because the two held-out estimators disagree there (step-wise DR `−0.0989`,
off-support) and the D42 cross-check `DIVERGES` at mid-`α` (the tabular MDP is optimistic where cells are
thin). This is consistent with WS4 (a myopic bandit does not beat observed behavior) and WS3 (a
statistically-real but tiny out-of-sample order effect): even a model built to value a setup finds no
certifiable setup value for the one mechanism it can represent — and WS7's flexible offline RL is the final
word.

### 6.2 Branched interpretation — three axes

The result is read on three axes. The **design-ladder** axis is WS5's actual question; the **D42** axis
qualifies how much to trust the value; the **verdict** axis is the headline. A code cell in the notebook (§8)
inspects the computed report and prints which branch fired on each axis; the write-ups below stand alone once
the numbers are filled. One **binding reading rule** governs all three, and is stated *before* any branch is
read (the analogue of WS1's D21 and WS4's myopic-ceiling rule):

> **Gate on a real interval, never on a point.** A prescriptive claim requires a non-degenerate CI whose
> one-sided lower bound clears the target. A degenerate CI (constant contributions) is printed `n/a`, never
> rendered as an interval, and can never fire a positive verdict. Point estimates and in-sample model values
> are *evidence*, not *certification* (decision D44).

#### Design-ladder axis (`S+` / `S0` / `S−`)

*Selected by the data (2021–2025): a blend of **S0 and S−** — directional-but-uncertifiable in the FQE lens,
and leaning negative on the setup mechanism itself. The trigger − count FQE gap is directionally positive and
its lower bound even clears the ceiling at `α = 1`, but the other lenses do not corroborate it (S0), and the
setup Q-gap is negative (S−, in spirit).*

*Pre-registered alternative — not selected.* **S+ — the trigger/prev designs beat `count` with real-CI
clearance.** At some `α` the refit-bootstrap FQE
trigger-count gap's one-sided lower bound clears `+0.003` — **sequential prescription value found**, a setup
the myopic bandit provably could not cash. This is a strong claim: cross-check it against WS3's H-branch (was
the order predictive out of sample?) and WS4's P-branch (a myopic `SEQ_INCONCLUSIVE_MYOPIC` there, exceeded
here, is the D40 handoff working). Confirm the step-wise-DR and model-based gaps are directionally positive
at the same `α`, and that the D42 cell is not `DIVERGES`. If it survives, the burden passes to WS7's full
offline-RL battery. *Not selected: no `α` clears the triple gate. The FQE lens's one-sided lower bound does
clear `+0.003`, but only at `α = 1` (lo95 `+0.0085`), where the step-wise-DR gap has flipped to `−0.0989`
(off-support, ESS `6.3%`) and the D42 cell `DIVERGES` — so the certifiable-clearance S+ requires is not met.*

**Selected by the data (2021–2025). S0 — directional but uncertifiable (the synth-validated pattern).** The
gap is directionally positive in
all three lenses and above the null world's baseline, but its refit-bootstrap lower bound does not clear the
ceiling — the fixture's own `SETUP_INCONCLUSIVE`. Report the *evidence story*, not a null: directional
agreement, the world-discriminating setup diagnostics, and the constructed-world proof that the machinery
cashes setups. State the `n`-to-certify back-of-envelope (CI half-width `~ c/√(n_clusters)`; inverting for a
`+0.008` gap to clear `+0.003` needs several-fold more effective clusters, and the full data is `~40×` the
fixture) as a concrete Phase-2 prediction. This is the most anticipated real-data outcome at moderate scale.

*Realized (2021–2025).* Selected — with a real-data twist that the synthetic fixture did not have. The FQE
lens *is* directionally positive and CI-positive from `α ≥ 0.1`, and its lower bound even clears the ceiling
at `α = 1` (lo95 `+0.0085`) — but unlike the fixture, where all three lenses agreed in sign, here the other
two lenses do **not** corroborate it at the deciding `α` (step-wise DR `−0.0989`; D42 `DIVERGES`). So the
honest reading is directional-but-uncertifiable, not S+: the disagreement, not agreement, is the story, and
certification is denied by estimator conflict rather than by CI width alone.

**Selected by the data (2021–2025) — the negative lean (in spirit). S− — the richer designs lose (sparsity /
overfit).** The trigger-count gap's CI upper bound is below zero:
the richer state is worth *less* than `count` out of sample. Read it through the per-`(s, a)` support
histograms and the greedy-optimism exhibit — the 216-state design is estimated from far thinner cells than
the 16-state one, so its extra resolution adds estimation variance without prescriptive signal (the
state-space echo of WS3's negative `Δ_matchup` and WS4's `O`-view overfitting). The honest reading is "no
sequential prescription edge **and** a real estimation cost of the richer state," and the policy handed
downstream should be built from the coarser design.

*Realized (2021–2025).* Selected *in spirit*, as the leaning-negative half of the reading. The literal S−
condition did not fire — the trigger − count value gap's upper bound is positive, not below zero. What is
negative is the **setup Q-gap itself** (mean `−0.0120`, positive on only 30% of `n = 135` trigger states):
the estimated MDP judges the one representable setup mechanism slightly *harmful* on average, the sign-flipped
opposite of the synthetic positive world (`+0.0087`, 68%). The richer design's raw value is higher, but that
is count-driven and optimistic (the greedy-optimism exhibit and the D42 `DIVERGES`), not setup value. So the
honest downstream reading is "no certifiable setup edge, and the representable setup mechanism leans negative"
— the state-space echo of WS4's `V−` and WS3's tiny-but-real order story, one rung up.

#### D42 agreement axis (`A+` / `A−`)

*Pre-registered alternative — not selected.* **A+ — the three lenses agree (trust the value).** Model-based,
step-wise DR, and FQE sit inside each other's
95% CIs wherever both are assessable; the value is a coherent read and the ladder gap can be taken at face
value (subject to its own CI). Expect this at low `α`, where the softened policy stays near behavior. *Not
selected: on real data the lenses do not agree — MB-vs-OPE `DIVERGES` at mid-`α` on every design, and at
`α = 1` the step-wise-DR gap (`−0.0989`) is the opposite sign of the FQE gap (`+0.0130`).*

**Selected by the data (2021–2025). A− — `DIVERGES` (a misspecification diagnostic, not an averaging
problem).** The estimated MDP's value of
the softened policy sits outside the held-out OPE's real CIs — reported per D42, never silently averaged. On
the synthetic worlds this grows with `α` and design richness: it is the tabular state's in-sample optimism at
thin cells, and it says the count-plus-trigger abstraction is **not Markov-sufficient**. For WS7 this is the
explicit brief — a richer function class is the response to a `DIVERGES` here, and the same paired-refit gate
philosophy carries over. When A− fires, lean on the held-out FQE gap and its refit CI, never the in-sample
model value.

*Realized (2021–2025).* Selected. MB-vs-OPE `DIVERGES` at mid-`α` on **every** design. The greedy-optimism
exhibit is the same effect at `α = 1`: `count_prev_trigger` model-based `+0.0206` sits above its held-out FQE
`+0.0179` (optimism `+0.0027`; `count` `+0.0043`, `count_prev` `+0.0022` — sharper on sparser designs). So the
reading leans on the held-out FQE gap and its refit CI, never the in-sample model value — and the lone
FQE-positive signal, lacking model-based *and* step-wise-DR corroboration at the deciding `α`, cannot be
certified. The tabular count-plus-trigger state is not Markov-sufficient on real data; answering that is the
explicit brief handed to WS7.

#### Verdict axis (`SETUP_EXPLOITED` / `SETUP_INCONCLUSIVE` / `SEQ_NEUTRAL_MDP`), under D39

*Pre-registered alternative — not selected.* **`SETUP_EXPLOITED`.** The triple condition holds at some `α`:
the tabular MDP cashes a setup pitch on
held-out data beyond the myopic ceiling — the ladder's motivating contrast delivered. Never fabricated; it
fires only on a real, non-degenerate lower bound. *Not selected: no `α` meets the triple condition — the FQE
lo95 clears only at `α = 1`, where step-wise DR is negative and D42 `DIVERGES`.*

*Pre-registered alternative — not selected.* **`SETUP_INCONCLUSIVE` (D39 first-class, the fixture's
verdict).** No `α` meets the triple condition on a
positive world. The effect is present, representable, and directionally agreed, but the sequential-OPE
variance cannot certify it at this scale — the sequential twin of WS4's `SEQ_INCONCLUSIVE_MYOPIC`, with
certification deferred to Phase-2 data scale. *Not selected: this was the synthetic **positive** world's
verdict, where all three lenses agreed in sign and only the CI width blocked certification. On real data the
estimators disagree in sign at the deciding `α` and the setup Q-gap is negative, so the honest class is
`SEQ_NEUTRAL_MDP` (no certifiable setup value), not a directional-positive INCONCLUSIVE.*

**Selected by the data (2021–2025). `SEQ_NEUTRAL_MDP`.** No `α` meets the triple condition and the ladder
shows no held-out advantage — the richer designs worth no more than `count` up to
overfitting cost. *(On the null world this reads as "the trigger flag is inert, the setup diagnostics at
their noise floor, the null control passing." On real data the class is the same but the substance differs —
see below and the cosmetic note in §6.1.)*

*Realized (2021–2025).* Selected — but read it as **"no certifiable setup value,"** NOT the null world's
"trigger flag inert." The printed message reuses the synthetic branch's null-world wording (cosmetic note,
§6.1); the verdict **class** and triple-gate logic are correct, while the substance is different: the
setup-capable design earns no certifiable advantage over `count`, and the one representable setup mechanism
shows a **negative** in-model Q-gap (`−0.0120`, positive on 30% of trigger states). The real data is not
inert — it leans against the setup. This is the honest D39 first-class negative for a *real* world, not a
passing null control.

#### Reading the grid

The honest headline is a triple `(S, A, verdict)`. *Selected by the data (2021–2025):*
**S0/S− × A− × `SEQ_NEUTRAL_MDP`** — the FQE trigger − count gap is directionally positive and its lower
bound even clears the ceiling at `α = 1` (S0), but the lenses `DIVERGE` and disagree in sign there (A−), and
the setup Q-gap itself is negative (S−, in spirit), so no certifiable setup value is found. This landed *near*
the pre-registered *most anticipated* cell **S0 × A− × `SETUP_INCONCLUSIVE`** (a real setup present but below
the sequential-OPE floor, tabular state visibly imperfect) but the verdict class is the neutral one and the
setup mechanism leaned negative rather than merely uncertifiable-positive. The *strongest* (unrealized) cell
would have been **S+ × A+ × `SETUP_EXPLOITED`**: a certified setup with agreeing lenses, cross-checked against
WS3/WS4 and handed to WS7. The pre-registered *null* cell is **S0(inert) × (A±) × `SEQ_NEUTRAL_MDP`** — the
class realized, but on real data with a negative-leaning mechanism, not an inert flag. The *diagnostic* note
is any **S−** (richer-design overfit — build from the coarser state).

---

## 7. Discussion

**The variance-binds lesson.** WS5's durable methodological contribution is not a run gain; it is the
recognition, made concrete and fixed, that in sequential OPE the *variance* binds a prescriptive claim. The
first gate's error was not a wrong number but a fake confidence: a constant-initial-state domain makes FQE's
contribution array degenerate, and a naive bootstrap returned a zero-width interval that certified a
directional gap. The paired refit bootstrap restores a real interval, and the corrected verdict is honest.
This is the same lesson WS4 taught at the myopic level — its `0.003` edge was directionally right but
swallowed by the noise floor — one rung up and sharper, because here the failure was *silent*. The discipline
that generalises is the binding reading rule (§6.2): gate on a real interval, print degenerate CIs as `n/a`,
never promote a point to a certification.

*Realized (2021–2025): the lesson deepened on real data. The refit bootstrap produced a genuinely
non-degenerate FQE interval whose lower bound even cleared the `+0.003` ceiling at `α = 1` (lo95 `+0.0085`) —
so, unlike the synthetic fixture, CI width was **not** the binding constraint. Certification still failed,
because the honest gate requires the lenses to **agree**: the step-wise-DR gap was the opposite sign
(`−0.0989`, off-support) and the model-based lens `DIVERGED` from the held-out ones. A real interval is
necessary but not sufficient; agreement across independent estimators is the second half of the discipline,
and it is what turned a lone FQE-positive point into an honest `SEQ_NEUTRAL_MDP`.*

**The WS4 → WS5 → WS7 arc.** Each rung's honest negative is the next rung's motivating contrast. WS4 measured
the myopic ceiling and handed WS5 a falsifiable target; WS5 proved the setup is *representable* and cashable
in principle and bound its *certification* to the sequential-OPE variance; WS7 inherits both — the paired-
refit gate philosophy and a richer function class to answer the D42 `DIVERGES` diagnostic that WS5's tabular
state raises. The ladder is designed so that "the setup is real but I can't certify it at this scale" (WS5)
is the precise handoff to "here is a function class that can represent more of it, evaluated with the same
honesty" (WS7).

*Realized (2021–2025): the arc holds, and the real handoff is sharper than the synthetic one. The three rungs
tell one consistent story — WS3's tiny out-of-sample outcome-order effect (`+0.0003` nats) → WS4's
no-myopic-value (every softened policy below behavior) → WS5's no-certifiable-setup-value. WS5 additionally
finds the one representable setup mechanism leaning **negative** (Q-gap `−0.0120`), so the handoff to WS7 is
not "certify a setup a well-specified tabular MDP already sees" but "find setup value that a well-specified
tabular MDP could not, and resolve the D42 `DIVERGES` its abstraction raised." WS7's flexible offline RL gets
the last word.*

**The count-driven baseline.** As in WS4, a policy can beat the habit-based synthetic behavior in raw value
for count reasons (value is dominated by the count; Tango et al., 2007), which is *not* sequencing. The
trigger-count *gap* under a common held-out lens is the instrument that isolates the setup value from the
count-driven gain, which is why the paper leads with the gap and not the raw value column.

*Realized (2021–2025): the raw values climb monotonically with design richness (`count` FQE `+0.0049` →
`count_prev` `+0.0170` → `count_prev_trigger` `+0.0179` at `α = 1`), which would be a tempting but false
"richer state wins" story. It is count/matchup-driven and optimistic (the greedy-optimism gaps and the D42
`DIVERGES`), not setup value — exactly why the verdict rests on the trigger − count gap under a common
held-out lens, and why that gap, not the value column, decided `SEQ_NEUTRAL_MDP`.*

**The firewall.** WS5 tests whether *acting* on the estimated MDP beats behavior *within support*; it does
not certify the trigger as *causal*. `SETUP_EXPLOITED` is evidence the ordered state yields a better
*evaluable* policy, not proof that *changing* the sequence *causes* the gain — the finding-#3 claim only
WS7's full battery, with its support diagnostics and estimator agreement, can approach
(`../ws3_gbdt_stack/THEORY.md` §8).

---

## 8. Limitations

1. **Tabular Markov-sufficiency.** The state is `(count, prev-family, trigger)` plus terminals. If the true
   dynamics depend on more (exact velocities, location, the batter), the tabular state is not
   Markov-sufficient and the model-based value is biased — which is exactly what a D42 `DIVERGES` reports.
   WS5 measures this honestly; it does not fix it (WS7's job).
2. **One engineered mechanism.** The trigger flag encodes *one* setup mechanism — a large ordered velo
   transition — because that is the one the fixture plants. Real data has unknown mechanisms; the MDP tests
   only *representable* ones, and a real setup that does not project onto the trigger is invisible here. The
   claim is never "this is the setup mechanism," only "this representable one is (not) cashable."
3. **Certification is variance-bound, not point-bound.** The binding constraint is the refit-bootstrap CI
   width, not the gap's sign. At synthetic scale it cannot clear the ceiling; whether it does at full scale
   is the open question, and the `n`-to-certify arithmetic (§6.2) is a back-of-envelope, not a guarantee.
   *Realized (2021–2025): at full scale the CI did shrink as predicted — the FQE gap's refit-bootstrap lower
   bound cleared `+0.003` at `α = 1` (lo95 `+0.0085`) — so CI width was no longer the binding constraint.
   What denied certification instead was **estimator disagreement** (step-wise DR `−0.0989`, opposite sign)
   and the **negative setup Q-gap**; the honest gate needs the lenses to agree, not merely a narrow interval,
   so the fixture-scale prediction ("more clusters will certify it") was not the mechanism that mattered.*
4. **Refit-bootstrap cost on real data.** The default `--fqe-boot 200` is `~3,000` refits; at `~1.5M`
   held-out rows budget up to a few hours, and lower `--fqe-boot` to 50–100 if slow (RUNBOOK WS5.1) — it
   changes only the CI resolution, never the point estimates. *Realized (2021–2025): the run used
   `--fqe-boot 100` (1,500 refits, `≈ 5.7 s/fit`, `8,578 s`), within a 2.6 h total at peak 11.3 GB.*
5. **Behavior model.** Without `--ws3-dir`, the OPE denominator is the coarser state-conditional empirical
   behavior from train counts; the behavior-recovery gate still guards it, but WS3's contextual `μ` tightens
   the importance weights.
6. **Family granularity and the `pitch_type` proxy.** The action is the pitch family (SPEC §4) and
   `pitch_type` is a classifier output (SPEC §1); a mechanism below the family level, or a mislabeled pitch,
   is only partially represented.
7. **One reward metric.** Value is `−delta_run_exp`; a different reward could reweight the reward table and
   move every number.

---

## 9. Conclusion

WS5 is the study's first sequential prescriptive rung and its most instructive methods-hardening story. It
builds a transparent tabular MDP whose trigger flag makes a setup pitch *representable*, plans it with exact
undiscounted policy iteration, and subjects its softened policy to the OPE gate — behavior recovery first,
then the setup gap scored against WS4's measured myopic ceiling with a **real** confidence interval. Its
conclusion is branch-conditional and complete now that the real numbers have arrived:

**Selected by the data (2021–2025): S0/S− × A− × `SEQ_NEUTRAL_MDP` — no certifiable setup value, leaning
negative.** The gate passes (behavior recovery `+0.0002`, IPS weights unit). The trigger − count FQE gap is
directionally positive and its refit-bootstrap lower bound even clears `+0.003` at `α = 1` (lo95 `+0.0085`),
but the triple gate is not met: the step-wise-DR gap flips to `−0.0989` there (off-support, ESS `6.3%`), the
D42 cross-check `DIVERGES` at mid-`α`, and the setup Q-gap is **negative** (`−0.0120`, positive on 30% of
trigger states). So the setup-capable design earns no certifiable advantage over `count`, and the one
representable setup mechanism leans slightly harmful in-model — the sign-flipped opposite of the synthetic
positive world. The verdict class is `SEQ_NEUTRAL_MDP`, read as "no certifiable setup value" (not the printed
synthetic "null world" wording; §6.1 cosmetic note). This is consistent with WS4 (a myopic bandit does not
beat observed behavior) and WS3 (a real but tiny outcome-order effect), and WS7's flexible offline RL is the
final word. The branch-conditional readings below are retained; the selected one is the S0/S− blend under
`SEQ_NEUTRAL_MDP`.

- *(pre-registered alternative — not selected; no `α` cleared the triple gate).* **If the gap clears the
  ceiling with a real CI (S+ → `SETUP_EXPLOITED`)**, the tabular MDP has cashed a
  setup pitch beyond what a myopic policy could reach — a strong claim, cross-checked against WS3/WS4 and
  handed to WS7.
- *(partially realized — the directional-but-uncertifiable shape is half of the selected reading, but the
  verdict class is `SEQ_NEUTRAL_MDP`, not `SETUP_INCONCLUSIVE`).* **If the gap is directional but uncertifiable
  (S0 → `SETUP_INCONCLUSIVE`)** — the fixture's own reading and
  the most anticipated real-data outcome — the honest content is that the setup is real and representable but
  below the sequential-OPE floor, with the `n`-to-certify arithmetic as the falsifiable Phase-2 prediction.
  *On real data the FQE lens was directional-but-uncertifiable as anticipated, but the estimators disagreed in
  sign and the setup Q-gap was negative, so the class is the neutral one, not a directional-positive
  INCONCLUSIVE.*
- *(selected in spirit — the leaning-negative half).* **If the richer designs lose (S−)**, the tabular state
  overfits and the policy should be built from the
  coarser design — read as fragmentation, not "order hurts." *Realized via the **negative setup Q-gap** rather
  than a negative value-gap: the representable setup mechanism leans harmful, not merely absent.*
- *(selected verdict class — but reframed).* **On the null world (`SEQ_NEUTRAL_MDP`)**, the machinery
  correctly finds nothing where there is nothing. *On real data the same class means "no certifiable setup
  value with a negative-leaning mechanism," not an inert flag; the real world is not null.*

Across every branch the durable contributions are the same: a setup-capable tabular state design proven to
cash a setup by a constructed-world test; a paired-refit sequential gate that never certifies a claim on a
degenerate interval; a D42 triple-lens cross-validation that reports divergence rather than averaging it away;
and, on real data, the honest finding that the setup-capable design earns **no certifiable value** and the one
representable setup mechanism leans **negative** — the machine telling the truth about a real world that does
not reward the setup it can see. The variance it exposes, and the misspecification the D42 `DIVERGES` flags,
are the exact things the capstone exists to shrink and to answer.

---

## References

Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

Efron, B., and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations. *Journal of
the American Statistical Association*.

Howard, R. A. (1960). *Dynamic Programming and Markov Processes*. MIT Press.

Jiang, N., and Li, L. (2016). Doubly robust off-policy value evaluation for reinforcement learning.
*Proceedings of the 33rd International Conference on Machine Learning (ICML)*.

Le, H. M., Voloshin, C., and Yue, Y. (2019). Batch policy learning under constraints. *Proceedings of the
36th International Conference on Machine Learning (ICML)*.

Marchi, M., and Albert, J. (2013). *Analyzing Baseball Data with R*. Chapman and Hall/CRC.

Precup, D., Sutton, R. S., and Singh, S. (2000). Eligibility traces for off-policy policy evaluation.
*Proceedings of the 17th International Conference on Machine Learning (ICML)*.

Puterman, M. L. (1994). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.

Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, second edition. MIT Press.

Tango, T. M., Lichtman, M. G., and Dolphin, A. E. (2007). *The Book: Playing the Percentages in Baseball*.

Thomas, P. S., and Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement
learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*.
