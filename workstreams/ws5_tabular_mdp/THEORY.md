# WS5 Theory: The Setup-Valuing MDP and Its Honest Sequential Gate, Derived

This document derives every formula the WS5 tabular MDP uses, **one named step at a time**, with a
**plain-words explanation after each block**. Nothing is skipped. The boxed results match `PAPER.md` and the
code in `model.py` / `run_ws5.py` exactly; where the code names a function or a constant, it is noted. WS5
consumes WS0's OPE machinery (`eval/ope.py`) and the WS1 shrinkage lemmas
(`../ws1_eb_tables/THEORY.md`), so §2 recaps the posterior-mean derivation and §6–§9 recap and extend the
shared estimator/verdict pieces. Notation is introduced once in §1 and reused throughout. The intellectual
payload is §7 — the **degenerate-CI derivation and its paired-refit fix** — for which §1 (termination), §4
(the setup term), and §6 (FQE as exact DP) are the load-bearing preliminaries.

---

## 1. Setup and notation: the episodic MDP of a plate appearance, and why γ = 1

WS5 models a plate appearance (PA) as an **episodic Markov decision process**. The logged data is a set of
transitions `(s, a, r, s')` — the pre-pitch state, the family thrown, the reward, and the next state.

- `s ∈ 𝓢` — the **state** in a given design (`count ⊆ count_prev ⊆ count_prev_trigger`, D43), plus four
  **absorbing terminals** `𝓣 = {walk, strikeout, hbp, in_play_end}` (`TERMINALS`). State counts: 16 / 112 /
  220 (12 / 108 / 216 non-terminal `+ 4`).
- `a ∈ 𝓐` — the **action**, one of `A = 8` pitch families; `XX` is descriptive-only and never feasible
  (SPEC §4). `𝓕(s) ⊆ 𝓐` is the leakage-safe feasible set.
- `r = R = −delta_run_exp` — the reward, larger better (SPEC §5). Terminals carry `R = 0`.
- `P(s' | s, a)` — the transition kernel (§2); `μ(a | s)` — the behavior policy; `π(a | s)` — a target
  policy; `π_α = (1−α)μ + απ_greedy` — the softened target (§4).
- The **return** of a PA is the **undiscounted** sum `G = Σ_t R_t`, which by SPEC §5 is the total run-value
  change across the PA — the exact quantity the OPE estimators score. Hence `γ = 1`.

**Step — justify `γ = 1` as correct, not convenient.** A discount `γ < 1` would down-weight later pitches
(a strike-three worth less than a strike-one), distorting the run-value semantics the reward already carries.
The undiscounted return is the right object; the only question is whether it is *finite*.

**Step — prove termination via the count dynamics.** Order the non-terminal states by the count `(balls,
strikes)`. Across a pitch the count is **non-decreasing**: a ball increments `balls` (→ `walk` at 4), a
called strike or whiff increments `strikes` up to 2 (→ `strikeout` at 3), a foul increments `strikes` only
below 2 and is a self-transition at `strikes = 2`, and an in-play ball or hit-by-pitch terminates. The
*only* non-terminating trajectory is an infinite run of two-strike fouls. Under the smoothed kernel every
reachable terminal has strictly positive mass (§2), so from every non-terminal state the probability of
reaching a terminal within a bounded number of steps is bounded below by some `ε > 0`. Therefore the PA
terminates with probability one and the expected episode length is finite:

$$\boxed{\; \Pr[\text{PA terminates}] = 1 \ \Rightarrow\ G = \textstyle\sum_t R_t \text{ is finite for } \gamma = 1. \;}$$

Equivalently, the policy is **proper**: the sub-stochastic block of `P_π` on the non-terminal states has
spectral radius `< 1` (mass leaks to the terminals), which makes `I − P_π` non-singular in §3.

**In plain terms.** A plate appearance always ends — with a walk, a strikeout, a hit-by-pitch, or a ball in
play. The count only ratchets one way (more balls, more strikes), and the only way to stall is fouling off
two-strike pitches forever, which can't actually go on forever because every pitch has some chance of ending
the at-bat. Because the at-bat is guaranteed to end, we can just *add up* the run-value changes with no
discounting — and that undiscounted total is exactly the number SPEC §5 says we care about.

---

## 2. Estimating the model: Dirichlet-smoothed transitions and two-level-shrunk rewards

The transition kernel and reward table are estimated from logged counts on the train fold, with the same
shrinkage philosophy as WS1 (`../ws1_eb_tables/THEORY.md`).

**Step — write the transition posterior mean (Dirichlet).** For source state `s` let `succ(s) = {s' :
N(s, ·, s') > 0}` be the reachable set of size `K_s`, and place a symmetric Dirichlet prior with
pseudo-count `α_t` on the next-state distribution over `succ(s)`. With multinomial counts `N(s, a, s')`, the
posterior is Dirichlet and its mean is

$$\boxed{\; \hat P(s' \mid s, a) = \frac{N(s, a, s') + \alpha_t}{N(s, a) + \alpha_t\,K_s}, \qquad s' \in succ(s), \;}$$

zero off `succ(s)` (`estimate_mdp`, `alpha_t` default `1.0`). This is the Dirichlet-multinomial posterior
mean of WS1 §2, applied per `(s, a)` over the reachable support. The observed `N(s, a, s')` dominate the
mass, and they carry the exact next previous family `prev' = a` and the resulting velo-gap `trigger'`, so the
setup mechanism is preserved; `α_t` only regularises thin cells.

**Step — handle unobserved cells by back-off.** A never-taken `(s, a)` (`N(s, a) = 0`) backs off to the
state's action-marginal transition `N(s, ·, s')`; an unobserved state routes deterministically to
`in_play_end`; terminals self-loop `P(s | s, ·) = 1`. This keeps `P̂` a proper distribution for every
`(s, a)` (each row of §1's proof) with no zeros on the reachable support.

**Step — write the two-level reward shrinkage.** Let `r̄` be the global mean reward. Shrink the state mean
toward `r̄`, then the cell mean toward the state mean (a hierarchical-normal partial pooling, WS1 §5):

$$\hat r(s) = \frac{\sum_{i:s_i=s} R_i + \alpha_r\,\bar r}{n_s + \alpha_r}, \qquad
  \boxed{\; \hat R(s, a) = \frac{\sum_{i:(s_i,a_i)=(s,a)} R_i + \alpha_r\,\hat r(s)}{n_{s,a} + \alpha_r} \;}$$

(`estimate_mdp`, `alpha_r` synth-tuned to `_SYNTH_ALPHA_R = 12`). A thin `(s, a)` cell falls back smoothly
to `r̂(s)`, then to `r̄` — no NaNs. Terminals carry `R̂ = 0`.

**Step — the behavior policy (the OPE denominator).** The state-conditional empirical behavior is the same
Dirichlet posterior mean over actions,

$$\hat\mu(a \mid s) = \frac{N(s, a) + \alpha_b}{N(s) + \alpha_b\,A},$$

full-support so importance ratios stay finite (`estimate_behavior_policy`, `alpha_b` default `0.5`). With a
`--ws3-dir` the OPE uses WS3's contextual propensities instead (decision D33).

**In plain terms.** We build the model by counting. "From this situation, throwing this pitch, where did the
at-bat go next?" gives the transition table; "what reward followed?" gives the reward table. Both counts are
thin in the richer designs, so we borrow strength the WS1 way: a rarely seen cell is pulled toward the
average of its situation, which is pulled toward the overall average — a little smoothing that prevents a
one-pitch cell from claiming a wild value, while leaving well-observed cells essentially untouched.

---

## 3. Planning: policy evaluation as a linear solve, and policy-iteration convergence

**Step — write the Bellman evaluation equation.** For a fixed (possibly stochastic) policy `π`, define the
policy-averaged reward and kernel `R_π(s) = Σ_a π(a|s) R(s,a)` and `P_π(s, s') = Σ_a π(a|s) P(s'|s,a)`. The
value satisfies `V = R_π + γ P_π V`. Pinning terminals at `V = 0` and restricting to the non-terminal block
`nt`,

$$\boxed{\; (I - \gamma\,P_\pi)\big|_{nt}\, V\big|_{nt} = R_\pi\big|_{nt}, \qquad Q(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a)\,V(s') \;}$$

(`_values_for_policy` / `policy_evaluation`). By §1 the policy is proper, so `(I − γP_π)|_{nt}` is
non-singular at `γ = 1` and the solve is **exact** (a singular/non-finite solve falls back to iterative
fixed-point evaluation, unreachable in practice).

**Step — greedy improvement over feasible actions.** The improved policy is the feasible arg-max,
`π'(s) = argmax_{a ∈ 𝓕(s)} Q(s, a)` (`_greedy_indices`, infeasible actions masked to `−∞`).

**Step — sketch monotone improvement (the policy-improvement theorem).** If `π'` is greedy w.r.t. `Q^π`,
then for every `s`, `Q^π(s, π'(s)) = max_{a∈𝓕} Q^π(s, a) ≥ Q^π(s, π(s)) = V^π(s)`. Substituting this into the
Bellman equation and iterating the (monotone) operator `T^{π'}` gives `V^{π'} ≥ V^π` pointwise, with strict
improvement at some state unless `π` is already greedy w.r.t. its own value (Howard, 1960):

$$\boxed{\; \pi' \text{ greedy w.r.t. } Q^\pi \ \Rightarrow\ V^{\pi'} \ge V^\pi \text{ pointwise, with equality iff } \pi \text{ is optimal.} \;}$$

**Step — convergence in finitely many steps.** There are finitely many deterministic feasible policies, and
policy iteration (`policy_iteration`, initialised at the myopic greedy-on-`R` policy) produces a strictly
improving sequence until it repeats — so it stops at the optimal policy in a finite number of sweeps (a
`max_iter` guard raises if a numerical estimate fails to stabilise).

**In plain terms.** Evaluating a fixed plan is just solving a small system of linear equations — no
iteration, no approximation, because the at-bat always ends. Improving the plan is "in each situation, switch
to the pitch your own value function now likes best (among the ones this pitcher actually throws)." A
one-line argument shows each such switch can only raise the value, and since there are finitely many plans,
you can't improve forever — you land on the best plan in a handful of passes.

---

## 4. Why the trigger flag makes a setup representable: the discarded look-ahead term

This is the design's reason for existing. It shows, formally, why the `count` design *cannot* value a setup
and the trigger design *can* — the sequential extension of WS4's myopic-ceiling decomposition.

**Step — decompose the action-value into reward and look-ahead.** From §3, at `γ = 1`,

$$\boxed{\; Q(s, a) = \underbrace{\mathbb{E}[R \mid s, a]}_{\text{this pitch}} + \underbrace{\mathbb{E}\!\left[v(s') \mid s, a\right]}_{\text{the setup: next state's value}}, \qquad v(s') = V^\pi(s'). \;}$$

A **myopic** policy (WS4's bandit) uses `Q_{\text{myopic}}(s, a) = E[R | s, a]` — it sets the second term to
zero (`γ = 0`). WS4 §6 further split the first term as `E[R | s, a] = v(s) + δ(s, a)`, where `v(s)` is the
state baseline (owed to every action) and `δ(s, a)` the family-differential; the myopic arg-max can reach
only `δ` (`../ws4_bandit/THEORY.md` §6).

**Step — locate the planted effect.** The positive world plants a whiff boost on the current pitch when
`|velo_{t−1} − velo_{t−2}| ≥ τ = 5` mph — the transition **into the previous pitch**, fixed by history (D21).
So on a triggered row the boost is in `v(s)`: it is owed to every current family equally, and a myopic policy
cannot manufacture it (WS4's `~0.003`-of-`0.032` ceiling). The value lives in *creating* the trigger a pitch
earlier — i.e. in the look-ahead term `E[v(s') | s, a]`.

**Step — show the trigger design carries the look-ahead variation and `count` does not.** In the trigger
design, `P̂(s' | s, a)` carries the next state's velo-gap `trigger'`, which depends on whether the chosen
family `a`'s velo band `v_a` differs from the previous family `p`'s by `≥ τ`. Writing `s' = (count',
prev'=a, trigger')`,

$$\mathbb{E}\!\left[v(s') \mid s, a\right] = \sum_{s'} \hat P(s' \mid s, a)\, V^\pi(s'), \qquad
  \Pr[\text{trigger'} = 1 \mid s, a] \approx \mathbf{1}\!\left[\,|v_a - v_p| \ge \tau\,\right],$$

so choosing a trigger-creating `a` shifts mass onto higher-value triggered next-states — the term **varies
with `a`**, and policy iteration values it. In the `count` design, `s'` retains no previous family or
trigger, so `E[v(s') | s, a]` is (to smoothing) constant across `a` and **cancels in the arg-max**:

$$\boxed{\; \text{trigger design: } \partial_a\,\mathbb{E}[v(s')\mid s,a] \neq 0 \ \text{(setup representable)}; \qquad
  \text{count design: } \partial_a\,\mathbb{E}[v(s')\mid s,a] \approx 0 \ \text{(setup invisible).} \;}$$

`setup_diagnostics` measures exactly this: the **setup Q-gap** `max_{a\text{ creating}} Q(s,a) −
max_{a\text{ not}} Q(s,a)` per reachable state (positive-world mean `+0.0087`, positive fraction `0.68`; null
`+0.0020`, `0.49`), and the share of trigger states whose optimal action differs from the `count_prev` parent
(`48.2%` vs `42.6%`).

**In plain terms.** A pitch's value has two parts: what it does *now*, and what *situation* it leaves for the
next pitch. A greedy chooser only sees the first part. The trick we planted pays off through the *second*
part — throwing a pitch that makes the next state a "triggered" one worth more. The `count` design's state
forgets what it just threw, so it literally can't tell that one choice sets up a better next state than
another; the trigger design remembers, so its planner can prefer the setup pitch. That memory — one extra
bit, "did the last two pitches jump in speed" — is the entire difference between a model that can value a
setup and one that can't.

---

## 5. The simulator as exact model-based evaluation, and why greedy values are optimistic

**Step — state the simulator's target.** The transparent simulator (`simulate`) draws `s_0 ∼ start_dist`,
acts `a ∼ π(·|s)`, accrues `R(s,a)`, transitions `s' ∼ P(·|s,a)`, and stops at a terminal. Its Monte-Carlo
mean return is an unbiased estimator of the model-based value, and by the strong law it converges to the
exact policy-evaluation value:

$$\boxed{\; \frac{1}{M}\sum_{e=1}^{M} G_e \;\xrightarrow{M\to\infty}\; \sum_s start\_dist[s]\,V^\pi(s), \;}$$

so the simulator's self-consistency with `policy_evaluation` is the D42 validation of the model-based value
(the pipeline reports both `model_based_greedy` and `simulated_greedy` per design; they agree to Monte-Carlo
error).

**Step — derive the in-sample greedy optimism (max-selection bias).** Policy iteration selects the arg-max
action per state on *estimated* values `Q̂`. Even with unbiased per-cell estimates, the value of the *selected*
action is upward-biased, because selecting the maximiser correlates the choice with the estimation error:

$$\mathbb{E}\!\left[\max_a \hat Q(s,a)\right] \ge \max_a \mathbb{E}\!\left[\hat Q(s,a)\right],$$

by Jensen's inequality applied to the convex `max`. The gap grows with the estimation variance, which is
larger on the sparser designs — so the in-sample model-based greedy value overstates the held-out FQE value
more for `count_prev_trigger` (216 cells) than for `count` (12). This is the **greedy-optimism exhibit**
(committed positive-world optimism `+0.0140` / `+0.0360` / `+0.0339`), reported as a separate figure, never a
verdict input. It is the same "max of noisy estimates is optimistic" argument the study invokes for the
`min`-biased `Δ_order` under D21 (`../ws1_eb_tables/THEORY.md` §10), here on a `max`.

**In plain terms.** Rolling the plan forward by simulation gives the same number as solving the equations —
a nice consistency check that the model-based value is what we think it is. But there's a catch when we let
the planner *pick* the best action using estimated values: picking the winner of a noisy contest flatters the
winner, because whatever action got lucky in the sample is the one we chose. That optimism is bigger where
the data is thinner, which is exactly the richer designs — so the in-sample "best plan" value always looks
rosier than it holds up out of sample, and we show that gap openly rather than gate on it.

---

## 6. FQE on tabular states: backward recursion is exact dynamic programming here

**Step — state fitted-Q evaluation.** FQE (`ope.fqe`, Thomas and Brunskill 2016; the DR sibling is Jiang and
Li 2016) fits the target's Q-function by backward regression: at each step the regression target is `r_t` for
terminal decisions and `r_t + V_{t+1}(s_{t+1})` for non-terminal ones, with
`V_{t+1}(s') = Σ_a π(a|s') q̂_{t+1}(s', a)` from the already-fitted next-step model, and reports the mean over
episodes of the initial-state value `V(s_0) = Σ_a π(a|s_0) q̂_0(s_0, a)`.

**Step — show that on tabular states the regressor is exact.** WS5 supplies `onehot_tabular_regressor`, whose
`fit` is the exact per-`(state, action)` group mean (a `bincount` over the decoded one-hot codes, with the
global mean as the fallback for unseen cells) — numerically identical to `ope.tabular_regressor`. On discrete
states this regression has **no model bias**: the fitted `q̂(s, a)` is the empirical mean return-to-go of the
cell, so the backward recursion *is* the exact tabular dynamic-programming backup:

$$\boxed{\; \hat q(s, a) = \frac{1}{|\{i:(s_i,a_i)=(s,a)\}|}\sum_{i} \big(r_i + \hat V_{\text{next}}(s'_i)\big) \;=\; \text{exact tabular DP backup}. \;}$$

So the only error in the FQE value is **estimation** (finite counts), not approximation — which is precisely
what makes §7's degeneracy argument exact and its refit fix the right response.

**In plain terms.** FQE learns "what's the expected rest-of-at-bat value from here, under the recommended
plan" by working backwards from the end. On our discrete states the learner is just an averager — it fills
each cell with the average return-to-go it saw — so there's no modelling approximation at all; it's exact
dynamic programming done from data. That matters because it means any wobble in the FQE number comes purely
from *which at-bats we happened to log*, which is the thing §7's error bars must capture.

---

## 7. The D44 section: why the naive FQE CI is degenerate, and the paired-refit fix

This is the workstream's methodological payload. It derives why the first shipped gate's confidence interval
was fake, and why the refit bootstrap is the correct instrument.

**Step — identify the per-episode contribution.** FQE's reported value is the mean over episodes of the
initial-state value (§6), so the per-episode contribution is `c_e = V(s_0^{(e)}) = Σ_a π(a|s_0^{(e)})
q̂(s_0^{(e)}, a)`.

**Step — show the contributions are constant.** In this domain **every PA starts in the identical state** —
`0-0` count, no previous pitch, trigger `0` — so `s_0^{(e)} = s_0` for all `e`. The target row `π(·|s_0)` and
the fitted `q̂(s_0, ·)` are then the same for every episode, hence

$$\boxed{\; c_e = \sum_a \pi(a\mid s_0)\,\hat q(s_0, a) \equiv \bar c \quad\text{for all } e \ \Rightarrow\ \mathrm{ptp}(\{c_e\}) = 0. \;}$$

**Step — show the naive cluster bootstrap has zero variance.** A cluster-resampling bootstrap forms
replicate means `\bar c^{(b)} = \frac{1}{|B_b|}\sum_{e \in B_b} c_e`. Since every `c_e = \bar c`, every
replicate mean is `\bar c` regardless of which episodes are drawn:

$$\mathrm{Var}_{\text{boot}}\big[\bar c^{(b)}\big] = 0 \ \Rightarrow\ \text{CI} = [\bar c,\ \bar c] \ \text{(a point).}$$

The interval is **structurally degenerate** — it collapses to a point not because the estimate is certain but
because the resampling has nothing to vary. The first gate read that fake zero-width interval as a certified
lower bound, producing a false `SETUP_EXPLOITED`. WS5 flags any `ptp = 0` input as degenerate and prints
`n/a (constant contributions)`, never an interval (`_cluster_boot`, `_refit_ci`).

**Step — identify what a real CI must randomize.** The FQE value's uncertainty is **estimation** uncertainty:
which held-out episodes were logged, and hence the fitted `q̂(s_0, a)` — *not* the start state, which is fixed.
The correct bootstrap therefore resamples the *episodes that determine the fit*, and **refits**.

**Step — the paired refit cluster bootstrap.** For each replicate `b`, resample pitcher-game clusters of
episodes with replacement (relabelling `pa_id` per instance), and refit FQE from scratch on the resampled
episodes for **every** design and `α`, using the *same* resample across arms:

$$\boxed{\; \hat V^{(b)}_{d,\alpha} = \mathrm{FQE}\big(\text{episodes}^{(b)};\ \pi_{d,\alpha}\big), \qquad
  \widehat{\text{gap}}^{(b)} = \hat V^{(b)}_{\text{trigger},\alpha} - \hat V^{(b)}_{\text{count},\alpha}, \;}$$

read percentile CIs and the one-sided 95% lower bound off `{\hat V^{(b)}}` and `{\widehat{\text{gap}}^{(b)}}`
(`_fqe_refit_bootstrap`, `_refit_ci`). Because the *same* resample drives every arm, the shared
episode-sampling noise **differences away** in the gap — a paired bootstrap, so the gap CI is tighter than
either level's. The refits are exact tabular fits (§6) through `onehot_tabular_regressor`, orders of
magnitude faster than hashing, which is what makes `~3,000` refits feasible (`~57` min).

**Step — state honestly what it still misses.** The refit bootstrap captures sampling variability of the
*episodes*; it does **not** capture **model-class misspecification** — if the tabular state is not
Markov-sufficient, the fitted `q̂` is biased and the refit CI is centered on a biased point. That bias is
exactly what the D42 `DIVERGES` diagnostic (§8, `_d42_agreement`) surfaces, and answering it is WS7's brief.

**In plain terms.** Here is the bug, told straight. Every at-bat starts on the very same square, so FQE's
per-at-bat number is the *same number* every time. If you try to build error bars by reshuffling those
identical numbers, nothing moves — the "interval" is a single point, a fake certainty. The first version of
the gate believed that fake certainty and declared victory. The real uncertainty isn't about the starting
square (that never changes); it's about *which at-bats we logged*, which changes the model we fit. So we
rebuild the error bars the expensive, honest way: resample whole at-bats and **refit the model from
scratch**, hundreds of times, using the same resample for every design so the *comparison* between designs is
pinned down tightly. What this still can't catch is if the tabular state is simply too coarse to be right —
and that's what the divergence check in the next section is for.

---

## 8. The triple-condition gate: a conjunction that controls the false-positive direction

**Step — write the gate.** `SETUP_EXPLOITED` fires at some `α` iff **all three** hold (`_ladder_verdict`),
with the D40 ceiling `c = 0.003`:

$$\boxed{\; \exists\,\alpha:\ \ \underbrace{\text{lower}_{95}\big(\text{gap}^{\text{FQE}}_\alpha\big) > c \ \wedge\ \neg\,\text{degenerate}}_{\text{(1) resolving lens, real CI}} \ \wedge\ \underbrace{\text{gap}^{\text{stepDR}}_\alpha > 0}_{\text{(2) sign check}} \ \wedge\ \underbrace{\text{gap}^{\text{MB}}_\alpha > 0}_{\text{(3) model agrees}}. \;}$$

`SEQ_NEUTRAL_MDP` if no `α` satisfies it on the null world; `SETUP_INCONCLUSIVE` (D39) if none does on a
positive world.

**Step — derive why the conjunction controls false positives.** Each lens fails in a *different* direction,
so requiring agreement makes an accidental clearance improbable. (1) The refit-FQE lower bound is the only
lens with a *resolving* CI, but on a degenerate or wide interval it cannot clear `c` — it guards against fake
confidence (§7). (2) The step-wise-DR gap has real per-episode variance but its per-PA importance-weight
product is heavy-tailed, so its CI is too wide to bound the ceiling — it is used only as a **sign** check,
catching a gap whose weighted lens points the wrong way. (3) The model-based gap is an exact in-sample
quantity but optimistic (§5) — it is the *easiest* to satisfy, so it is a necessary-not-sufficient screen. A
spurious clear would need the refit-FQE lower bound (hard), the DR sign (independent noise), and the MB sign
to line up together — a conjunction whose components are driven by different error sources:

$$\boxed{\; \Pr[\text{false } \texttt{SETUP\_EXPLOITED}] \le \min_k \Pr[\text{lens }k\text{ clears spuriously}] \ \text{— the tightest lens (the real-CI FQE bound) dominates.} \;}$$

**Step — the D42 agreement rule (D24).** Separately, per `(design, α)`, the *same* `π_α` is scored by the
three lenses; a pair diverges iff `|V_i − V_j|` exceeds the wider of their 95% CI half-widths (`_d42_agreement`;
model-based has half-width 0). `DIVERGES` if any assessed pair diverges — reported, never averaged. It is a
misspecification diagnostic (the tabular optimism of §5/§7), and it is why the gate's resolving lens is the
held-out FQE, not the model-based value.

**In plain terms.** We only say "the setup paid off" when three different instruments agree, and we picked
three that go wrong in three different ways: the refit-FQE bound can be fooled by fake-narrow error bars (so
we demand a *real* one that clears the target); the importance-weighted estimate is too jumpy to trust for a
threshold (so we only ask it which *direction* it points); and the in-sample model value is always a bit
rosy (so it's just a sanity screen). For a false alarm, all three independent things would have to line up at
once — which is unlikely, and dominated by the one honest error bar. The separate divergence check is our
"is the tabular map even right?" alarm, and it's why the verdict leans on the held-out number rather than the
model's own optimistic value.

---

## 9. Sample size: how many clusters certify a +0.008 gap?

**Step — scale the CI half-width.** A refit-bootstrap gap estimate is a mean over pitcher-game clusters, so
its standard error, and hence the CI half-width `h`, scales as

$$h \;\approx\; \frac{c_0}{\sqrt{n_{\text{clusters}}}} \;=\; O\!\left(n_{\text{clusters}}^{-1/2}\right).$$

At synthetic scale the committed positive-world FQE gap CI at `α = 1` is `[−0.0081, +0.0218]`, a half-width
`h ≈ 0.015`, with a one-sided lower bound `≈ −0.004`.

**Step — recover the current standard error from the observed lower bound.** To fire `SETUP_EXPLOITED` the
one-sided 95% lower bound of a gap of size `Δ ≈ +0.008` must exceed the ceiling `c = 0.003`; that lower bound
is `\text{LB} = Δ − 1.645\,se`. The committed run has `\text{LB}_0 ≈ −0.004`, so

$$1.645\,se_0 \approx \Delta - \text{LB}_0 = 0.008 - (-0.004) = 0.012 \ \Rightarrow\ se_0 \approx 0.0073.$$

**Step — state the target and invert for the cluster count.** We need a standard error small enough that the
lower bound clears the ceiling, `Δ − 1.645\,se_{\text{target}} > c`, i.e.

$$se_{\text{target}} \lesssim \frac{\Delta - c}{1.645} = \frac{0.008 - 0.003}{1.645} \approx 0.0030.$$

Since `se ∝ n_{\text{clusters}}^{-1/2}`,

$$\frac{n_{\text{target}}}{n_0} = \left(\frac{se_0}{se_{\text{target}}}\right)^2 \approx \left(\frac{0.0073}{0.0030}\right)^2 \approx 6\times.$$

**Step — connect to Phase 2.** The synthetic fixture has `~36{,}000` eval PAs; the full data has `~1.5M`
held-out rows, roughly `~40×` the fixture, comfortably past the `~6×` needed if the gap size and overlap hold.
So a fixture-scale `SETUP_INCONCLUSIVE` does **not** preclude a full-scale `SETUP_EXPLOITED`:

$$\boxed{\; \text{certifying a } +0.008 \text{ gap above } +0.003 \text{ needs } \sim 6\times \text{ the effective clusters; full data is } \sim 40\times \text{ the fixture — a Phase-2 question.} \;}$$

**In plain terms.** How small an edge you can *prove* is set by your error bar, and the error bar shrinks like
one-over-the-square-root of how many independent at-bats (really, pitcher-games) you have. Our synthetic error
bar is about twice too wide to prove the `0.008` setup edge clears the `0.003` bar, and closing that gap takes
roughly six times the data. The real season is about forty times bigger than our fixture, so at full scale the
error bar could easily shrink under the bar — which is exactly why "can't tell yet" at synthetic scale is an
honest *deferral to Phase 2*, not a null. It's the same arithmetic WS4 did for its myopic edge, one rung up.
