# WS4 Theory: The Myopic Bandit and Its Ceiling, Derived

This document derives every formula the WS4 bandit uses, one named step at a time, with a plain-words
explanation after each block. Nothing is skipped. The boxed results match `PAPER.md` and the code in
`model.py` / `run_ws4.py` exactly; where the code names a function or a constant, it is noted. WS4
consumes WS0's OPE machinery (`eval/ope.py`) and WS3's artifacts, so §4 and §7–§9 recap the shared
estimator/verdict pieces and point to `../ws3_gbdt_stack/THEORY.md` §8 for the finding-#2/#3 firewall
around `q̂`. Notation is introduced once in §1 and reused throughout. The intellectual payload is §6 —
the **myopic ceiling** — for which §3 (the `κ` rescale), §5 (the fixed-evaluator ablation), and §8 (the
OPE noise floor) are the load-bearing preliminaries.

---

## 1. Setup and notation: the logged-bandit view of a pitch decision

WS4 treats each pitch decision as a **one-step contextual bandit**. The logged data is a set of tuples
`(s_i, a_i, r_i)` — the state before the pitch, the family actually thrown, and the reward — with a
known logging (behavior) policy.

- `s` — the **state**, the feature vector of one decision row in a given view (`C ⊆ L1 ⊆ O`, SPEC §6).
- `a ∈ 𝓐` — the **action**, one of the `A = 8` pitch families (`FAMILIES`); `XX` is descriptive-only
  and never feasible (SPEC §4). `𝓕(s) ⊆ 𝓐` is the leakage-safe **feasible** set for state `s`.
- `r = R = −delta_run_exp` — the reward, larger better for the pitcher (SPEC §5).
- `μ(a | s)` — the **behavior** (logging) policy, WS3's per-view propensities (`BehaviorModel`),
  floored at `MU_FLOOR = 1e-6` and renormalised so importance ratios are finite (`_floor_renormalise`).
- `q̂(s, a)` — WS3's **counterfactual value grid**, `E_model[R | s, A = a]` for every family
  (`OutcomeStack.q_grid`), with residual sd `exp_reward_sd(s, a)` (`OutcomeStack.exp_reward`).
- `π̃(a | s)` — WS4's **target** policy (the Thompson policy, §2). `π_α = (1−α)μ + απ̃` — the softened
  policy (§4).
- One decision = one **episode**: `pa_id` unique per row, `step = 0` (`_build_logged`). This is what
  makes the importance weighting ordinary per-decision weighting and the problem a genuine bandit.

**In plain terms.** Every row is a little one-shot gamble: the pitcher was in situation `s`, chose
pitch `a`, and got reward `r`. We know how likely he was to choose each pitch (`μ`, from WS3), and we
have a model's guess of the run value of *each* pitch he could have thrown (`q̂`, also from WS3). WS4
builds a *new* way to choose (`π̃`) and asks the OPE machinery: if he had chosen this way, would he
have done better — and can we even tell? The "one-step" part is the whole character of the workstream:
a bandit judges a pitch only by *this* pitch's reward, never by what it sets up next.

---

## 2. Thompson sampling: the posterior-argmax probability and its Monte-Carlo estimator

The target policy is the sampling rule of Thompson (1933): treat each action's value as uncertain, draw
a plausible value for each, and play the winner. Over many draws, each action is played with its
posterior probability of being the best.

**Step — write the per-action posterior.** WS4 models the value of action `a` in state `s` as an
independent normal centered on WS3's estimate,

$$\theta_a \mid \text{data} \;\sim\; \mathcal{N}\!\big(\hat q(s, a),\; \sigma(s, a)^2\big),
\qquad \sigma(s, a) = q\_sd(s, a),$$

with `q_sd` the *posterior standard error of the mean* (§3 derives why it is not the raw residual sd).

**Step — define the target as the argmax probability.** The Thompson probability of action `a` is the
probability its draw is the largest among the **feasible** actions,

$$\boxed{\;\tilde\pi(a \mid s) \;=\; P\!\left(\theta_a = \max_{a' \in \mathcal{F}(s)} \theta_{a'}\right),
\qquad \theta_{a'} \sim \mathcal{N}\!\big(\hat q(s, a'),\, \sigma(s, a')^2\big)\;}$$

restricted to `𝓕(s)` (infeasible actions get zero mass). This is a valid distribution over `𝓕(s)`.

**Step — the Monte-Carlo estimator (`thompson_policy`).** The argmax probability has no closed form for
`A > 2` unequal-variance normals, so it is estimated by sampling: draw `M = n_samples` independent
vectors `θ^{(m)}`, push infeasible actions to `−∞` so they never win, take the per-draw argmax, and
count wins:

$$\hat{\tilde\pi}(a \mid s) \;=\; \frac{1}{M}\sum_{m=1}^{M} \mathbf{1}\!\left[a = \arg\max_{a' \in \mathcal{F}(s)} \theta^{(m)}_{a'}\right].$$

In code this is the vectorised `draws = rng.standard_normal((m, M, A)) * sd + mu`, masked with
`np.where(feasible, draws, -inf)`, then `draws.argmax(axis=2)` counted per action (`thompson_policy`,
`model.py`). The result sums to exactly 1 and places zero mass off `𝓕(s)`.

**Step — the finite-sample error (state it).** For a fixed row, each `1[a wins]` is Bernoulli with mean
`π̃(a|s)`, so the estimator is an average of `M` i.i.d. indicators and its standard error is

$$\mathrm{SE}\!\left[\hat{\tilde\pi}(a\mid s)\right] \;=\; \sqrt{\frac{\tilde\pi(a\mid s)\,(1-\tilde\pi(a\mid s))}{M}} \;=\; O\!\left(\frac{1}{\sqrt{M}}\right)
\;\le\; \frac{1}{2\sqrt{M}}.$$

At the default `M = 1500` this is `≤ 0.013` per probability — negligible against the OPE CI half-widths
(`~0.006` on value, §8) that dominate the verdict, and deterministic under the seed.

**Step — the boundary cases.** If one feasible mean dominates relative to its sd, `π̃` is a
near-point-mass; if `σ → 0` it is the exact argmax point mass; equal means with equal sds give a uniform
draw over `𝓕(s)`; an **empty** `𝓕(s)` falls back to the observed-action point mass (the row *carries no
recommendation*, `thompson_policy(..., observed_action=...)`).

**In plain terms.** To turn "here's each pitch's expected value, give or take" into "here's how to
choose," Thompson sampling imagines many parallel worlds: in each, it draws a plausible value for every
feasible pitch from that pitch's error bar and throws the one that came out best. A pitch that wins in
70% of the imagined worlds gets 70% of the recommendation. There's no formula for that win rate with
eight unequal error bars, so we simulate 1500 worlds and count — and because 1500 is a lot, the count
is accurate to about a percentage point, far tighter than the downstream evaluation noise. If a pitcher
has no established repertoire (empty feasible set), we make no recommendation and just mirror what he
actually threw.

---

## 3. Why `q_sd` needs the `κ` rescale: residual noise vs the posterior SE of a mean

This is the single most consequential modelling choice in WS4, and it is derived, not asserted. The
question is: *which* standard deviation does the Thompson normal in §2 need?

**Step — name the two variances.** WS3 supplies `exp_reward_sd`, the **residual** reward sd — the
spread of a *single pitch's* reward around its conditional mean,

$$\mathrm{Var}[R \mid s, a] \;=\; \texttt{exp\_reward\_sd}(s, a)^2,$$

the law-of-total-variance quantity of `../ws3_gbdt_stack/THEORY.md` §4.2, empirically `~0.17` on the
`|R| ~ 0.05–0.3` scale. What Thompson sampling needs is a different thing: the **posterior standard
error of the estimated mean** `q̂(s, a)`,

$$\sigma(s, a)^2 \;=\; \mathrm{Var}\!\left[\,\mathbb{E}[R \mid s, a] \;\middle|\; \text{data}\,\right],$$

the uncertainty in *where the mean is*, not the scatter of individual pitches around it.

**Step — relate them by the effective support.** For a mean estimated from `n_eff` comparable
observations, the standard error of the mean is the residual sd shrunk by `√n_eff`,

$$\sigma(s, a) \;=\; \frac{\texttt{exp\_reward\_sd}(s, a)}{\sqrt{n_\text{eff}}} \;=\; \kappa \cdot \texttt{exp\_reward\_sd}(s, a),
\qquad \kappa = \frac{1}{\sqrt{n_\text{eff}}}.$$

A LightGBM leaf aggregates on the order of `min_child_samples ~ 10²` comparable pitches, so
`n_eff ≈ 400` and `κ ≈ 0.05` — the constant `POSTERIOR_SCALE = 0.05` (`build_bandit_inputs` returns
`q_sd = POSTERIOR_SCALE * exp_reward_sd`; `model.py`).

**Step — show what each choice does to the policy.** The Thompson policy's concentration is governed by
the ratio of the inter-family `q̂` gaps to `σ`. The gaps are `~0.005`.
- With `κ = 1` (use the residual sd raw): `σ ≈ 0.17`, so the gap-to-sd ratio is `~0.03` — the normals
  overlap almost completely, every feasible family wins about equally often, and `π̃` is **near-uniform**.
  Defensible (it honestly says "we can't tell"), but useless as a recommender.
- With `κ = 0.05`: `σ ≈ 0.0085`, so the gap-to-sd ratio is `~0.6` — the normals separate enough that the
  top family wins `~0.62–0.75` of the draws, and `π̃` is a **concentrated** recommendation.

$$\boxed{\; \kappa \text{ maps the per-pitch residual noise to the SE of the *mean*; } \kappa=1 \text{ gives a uniform policy, } \kappa\approx 0.05 \text{ a usable one.} \;}$$

`thompson_policy` is agnostic to how `q_sd` was formed (§2); `κ` is the single exposed confidence knob
(`--posterior-scale`), reported openly because it sets the ambiguity exhibit's toss-up share (§6 of
`PAPER.md`).

**In plain terms.** WS3 hands us "a pitch's reward is `q̂` give or take `0.17`." But `0.17` is how much
*one pitch's* result bounces around — a whiff here, a homer there — not how unsure we are about the
*average* value of throwing that pitch. Thompson sampling needs the second thing. Averages are far more
certain than individual outcomes: average a few hundred comparable pitches and the uncertainty in the
average shrinks by the square root of the count, roughly twentyfold, from `0.17` to `~0.0085`. If we
skipped that shrink and sampled with `0.17`, every pitch would look equally good and the recommender
would shrug at everything. The `κ = 0.05` factor is exactly that "√(a few hundred)" shrink, and we state
it out loud because it's the knob that decides how boldly the bandit distinguishes pitches.

---

## 4. The `π_α` conservative mixture and its value linearity

**Step — write the mixture (SPEC §9).** The target is softened toward behavior,

$$\pi_\alpha(a \mid s) = (1-\alpha)\,\mu(a \mid s) + \alpha\,\tilde\pi(a \mid s), \qquad \alpha \in [0, 1],$$

a convex combination of two distributions, so each row is a valid distribution (`soften` delegates to
`eval/ope.pi_alpha`; WS4 does not re-implement it). `α = 0` is behavior, `α = 1` the pure target.

**Step — derive the value linearity.** The **primary** estimator the frontier plots is DR (bandit
data; `_primary_estimator`), and DR's per-row contribution is affine in the target probabilities. Both
pieces are affine in `π`: the baseline `\bar q^\pi(s_i) = Σ_a π(a|s_i) q̂(s_i,a)` is linear in `π`, and
the weight `w_i(π) = π(a_i|s_i)/μ(a_i|s_i)` is linear in `π`. Since `π_α = (1−α)μ + απ̃` is affine in
`α`, each contribution — and hence the value — is affine in `α`:

$$\boxed{\; V_{\mathrm{DR}}(\pi_\alpha) = (1-\alpha)\,V_{\mathrm{DR}}(\mu) + \alpha\,V_{\mathrm{DR}}(\tilde\pi). \;}$$

The same holds for DM (`V_DM = (1/n)Σ_i Σ_a π(a|s_i) q̂(s_i,a)`) and IPS (linear in `w`, hence in `π`).
The only exception among the reported estimators is **SNIPS**, a self-normalized *ratio*
`Σ w_i r_i / Σ w_i` of two `π`-linear quantities, which is affine near the endpoints but not exactly
linear in between. So the DR frontier value-vs-`α` is a **straight line** from `V(μ)` at `α=0` to
`V(π̃)` at `α=1`; its slope is the policy's value over behavior. (WS0's `eval/ope` self-test confirms
`π_α` interpolates exactly between `V(μ)` and `V(π̃)`; dispatch log 0c.)

**Step — note what is *not* linear.** The **effective sample size** and the importance-weight tail are
*not* linear in `α`: ESS degrades and the max weight grows as `α → 1` (the price of deviating from
behavior). So the frontier is read as a straight value line *against a degrading support overlay* — a
higher `α` buys a stronger recommendation at the cost of a weaker evaluation.

**In plain terms.** `π_α` is a dial between "do what he already does" (`α=0`) and "do what the bandit
suggests" (`α=1`). Because every OPE estimator is a straightforward average that's linear in the policy,
turning the dial moves the *estimated value* in a straight line between the two ends — no surprises. What
*doesn't* move in a straight line is how trustworthy the estimate is: the further you dial toward the
bandit, the fewer logged pitches genuinely resemble what it would do, so the effective sample size drops
and a few pitches start carrying too much weight. That's why we always read the value next to the ESS.

---

## 5. The D38 prescriptive ablation and why the evaluator must be fixed

The ablation is WS4's central exhibit. This section states it formally and derives the crux: why every
view's policy must be scored against **one** evaluator.

**Step — define the two policies.** For view `v ∈ {C, L1, O}`, build the Thompson target from *that
view's* `q̂_v`: `π̃_v(a|s) = P(argmax over 𝓕(s) of 𝒩(q̂_v(s,a), σ_v²))`. The policies differ **only** in
the information set of the `q̂` they were built from — `C` sees context, `O` sees the ordered history.

**Step — score both under a common evaluator.** Fix a single evaluator `(μ_O, q̂_O)` — the behavior and
value grid of the richest view `O` (`COMMON_EVAL_VIEW`). The ablation statistic is the DR value gap

$$\Delta_{O-C} \;=\; V_{\text{DR}}^{(\mu_O, \hat q_O)}(\pi_{\alpha,O}) \;-\; V_{\text{DR}}^{(\mu_O, \hat q_O)}(\pi_{\alpha,C}),$$

and likewise `Δ_{L1−C}`. Both policies are evaluated on the same rows by the same evaluator; the only
thing that differs is the `q̂` each policy was *built from*.

**Step — derive why the yardstick must be fixed.** Suppose instead each view were scored by its *own*
evaluator, `V^{(μ_v, q̂_v)}`. Then

$$\Delta_{O-C}^{\text{per-view}} = V^{(\mu_O,\hat q_O)}(\pi_O) - V^{(\mu_C,\hat q_C)}(\pi_C)
= \underbrace{\big[V^{(\mu_O,\hat q_O)}(\pi_O) - V^{(\mu_O,\hat q_O)}(\pi_C)\big]}_{\text{policy-information difference (wanted)}}
+ \underbrace{\big[V^{(\mu_O,\hat q_O)}(\pi_C) - V^{(\mu_C,\hat q_C)}(\pi_C)\big]}_{\text{evaluator difference (confound)}}.$$

The second bracket is nonzero whenever `q̂_O ≠ q̂_C` — a richer evaluator scores the *same* policy `π_C`
differently — so a per-view ablation confounds "the ordered policy is better" with "the ordered
evaluator scores things differently." Fixing the evaluator kills the second bracket identically:

$$\boxed{\; \Delta_{O-C} = V^{(\mu_O,\hat q_O)}(\pi_O) - V^{(\mu_O,\hat q_O)}(\pi_C) \text{ is a pure *policy-information* statistic.} \;}$$

**Step — read the sign.** `Δ_{O−C} > 0` (CI clears 0): the ordered state builds a myopically-better
policy — *exploitable* sequencing (branch P+). `≈ 0`: no myopic sequencing edge (P0). `< 0`
(significantly): the ordered `q̂_O` **overfits** — its extra features add estimation variance without
prescriptive signal, so the `O`-built policy is worse under the fixed evaluator (branch P−, the
null-world signature — the prescriptive echo of WS3's negative `Δ_matchup`, `../ws3_gbdt_stack/THEORY.md`
§6.3).

**In plain terms.** The ablation gives the *same* decision-maker three qualities of information — context
only, plus-last-pitch, plus-full-order — and asks whether the richer information changes the *value* of
its choices. The trap is measuring each with its own ruler: if the "full-order" policy is graded by a
"full-order" grader, you can't tell whether the policy got better or the grader just grades differently.
So we grade all three with the *one* richest grader. Then the only difference between the three is what
the *policy* was allowed to know, which is exactly what we wanted to measure. A positive gap means order
helped the policy; a negative gap — which the null world shows — means the extra ordered features just
made the policy noisier.

---

## 6. The myopic ceiling: `E[R|s,a] = v(s) + δ(s,a)` and what greed can reach

This is the workstream's payload. It explains, from the ground-truth generative model, *why* a myopic
bandit's honest verdict on the positive world is INCONCLUSIVE — and turns that into a number the
sequential rungs must beat.

**Step — decompose the per-pitch reward.** Write the expected reward as a **state** term plus an
**action-differential** term,

$$\boxed{\; \mathbb{E}[R \mid s, a] \;=\; v(s) \;+\; \delta(s, a), \qquad \sum_a \bar\mu(a)\,\delta(s,a) = 0, \;}$$

where `v(s) = Σ_a \barμ(a) E[R|s,a]` is the state's baseline value (an action-average) and `δ(s,a)` is
how much action `a` beats that baseline. A **myopic argmax** policy chooses `a*(s) = argmax_a q̂(s,a)`,
so its per-state advantage over any action-blind policy depends **only on `δ`**: `v(s)` is added to
*every* action's value and cancels in the argmax.

**Step — locate the planted effect in the decomposition.** The positive world plants a whiff boost on the
current pitch when the velocity transition **into the previous pitch**, `|velo_{t−1} − velo_{t−2}|`,
exceeds 5 mph (`make_positive_world`, keyed on the transition into the *prior* pitch per D21). The trigger
is a function of the **history** `(velo_{t−1}, velo_{t−2})`, which is **fixed by the time the current
decision is made**. Therefore the boost is part of `v(s)`:

$$\text{on a triggered row, } v(s) \text{ rises by } \approx +0.032 \text{ for *every* feasible current family.}$$

Ground-truth probes confirm this uniform `+0.032` state lift. What differs across *today's* families is
only the small residual: some families convert a triggered pitch's foul/in-play mass into whiffs slightly
more than others (split-finger `δ ≈ +0.042`, four-seam `δ ≈ +0.030`), a **family-differential spread of
`~0.012`**.

**Step — compute the myopically-reachable edge.** A myopic `O`-policy can, on triggered rows, tilt toward
the higher-`δ` families; a `C`-policy cannot see the trigger and stays at the family baseline. The best
achievable `O − C` policy edge is the differential captured on the triggered fraction — arithmetic that
lands at

$$\boxed{\; \Delta_{O-C}^{\text{myopic, best}} \;\approx\; 0.003 \text{ run} \quad\text{(the family-differential, harvested on triggered rows)}. \;}$$

The full `+0.032` is **not** myopically reachable: it lives in `v(s)`, owed equally to every current
family, so no choice of *today's* pitch can manufacture it.

**Step — locate the rest of the effect (the setup term).** Where did the other order of magnitude go? Into
the **setup**. The velocity transition that triggers the boost is *created* by throwing the pitch that
makes `|velo_{t−1} − velo_{t−2}|` large — i.e. by *today's* action affecting *tomorrow's* state `s'`. In a
sequential value this is the term

$$Q(s, a) = \mathbb{E}[R \mid s, a] + \gamma\,\mathbb{E}\!\left[v(s') \mid s, a\right],$$

whose second piece — today's action shifting the *next* state's baseline value — is exactly what a myopic
bandit's `Q(s,a) = E[R|s,a]` **drops** (`γ = 0` for a bandit). A sequential planner (WS5's MDP, WS7's
offline RL) keeps it and can therefore value the setup; a bandit provably cannot.

$$\boxed{\; \text{Myopic ceiling: a bandit can reach } \delta(s,a) \text{ (}\sim 0.003\text{); the setup, } \gamma\,\mathbb{E}[v(s')\mid s,a] \text{, is invisible to it.} \;}$$

**Step — connect to the observed verdict.** `0.003` sits below the OPE noise floor (`~0.006`, §8), so the
positive world's `O − C` gap is directionally positive but CI-inconclusive — sitting `~0.003` above the
null world's *negative* overfitting baseline, which is the discrimination that proves the signal is
*present*. The gap-machinery self-test (constructing a genuine myopic `δ` edge and confirming
`SEQ_EXPLOITED` fires) proves the near-zero measurement is the *effect's* property, not the detector's.
This is decision D40: WS5/WS7's positive-world acceptance is to **exceed 0.003** by valuing the setup.

**In plain terms.** Split each pitch's expected value into two parts: a part that depends only on the
*situation* (which is the same no matter which pitch you pick now) and a part that depends on *which pitch
you pick*. A greedy chooser can only ever win the second part — the first part is baked in before it
chooses. The trick we planted rewards a *velocity jump that already happened*: by the time this pitch is
chosen, the jump is history, so the whole `0.032` reward-boost is in the "situation" part and is owed to
every pitch equally. Only a sliver — about `0.003`, the bit where some pitch types finish a triggered
count slightly better than others — is in the "which pitch" part a greedy chooser can grab. The other
`0.029` is the *setup*: the value of having thrown the pitch that *created* the jump one pitch earlier —
and that pays off in the *next* situation, which a one-pitch-at-a-time bandit literally cannot see. So the
bandit's "I can't find a sequencing edge" is exactly right: the edge is real but it's a setup, and setups
are the sequential workstreams' job. That's the ceiling, and beating `0.003` is their falsifiable target.

---

## 7. The paired clustered bootstrap for the gap CI

**Step — state the estimator.** The `O − C` gap CI is a pitcher-game-clustered **paired** bootstrap over
the per-row doubly-robust contributions under the common evaluator (`_gap_ci`, `run_ws4.py`). Let

$$d_i \;=\; \mathrm{dr}_i(\pi_O) - \mathrm{dr}_i(\pi_C), \qquad
\mathrm{dr}_i(\pi) = \bar q_O^{\pi}(s_i) + w_i^{\pi}\,(r_i - \hat q_O(s_i, a_i)),$$

the difference of the two policies' DR per-row contributions (`eval/ope.dr`, consuming the estimator, not
re-implementing it). The point estimate is `Δ = mean_i d_i`; the CI resamples **pitcher-game clusters**
with replacement (`cluster_bootstrap_indices`, SPEC §7) and recomputes `mean d_i`, reporting the 2.5/97.5
percentiles and the one-sided 95% lower bound.

**Step — derive why pairing tightens the CI.** Both `dr_i(π_O)` and `dr_i(π_C)` share the same evaluator
`q̂_O` and the same row `i`, so they are **positively correlated**. The variance of the paired difference,

$$\mathrm{Var}[d_i] = \mathrm{Var}[\mathrm{dr}_i(\pi_O)] + \mathrm{Var}[\mathrm{dr}_i(\pi_C)] - 2\,\mathrm{Cov}[\mathrm{dr}_i(\pi_O), \mathrm{dr}_i(\pi_C)],$$

is **reduced** by the `2 Cov` term — the common-evaluator variance (the `\bar q_O^π(s_i)` baseline and the
shared `q̂_O(s_i,a_i)` correction) largely cancels because it moves *together* for both policies. Bootstrapping
the *difference* directly (rather than differencing two independently-bootstrapped values) keeps that
cancellation, so the gap CI is tight around the *policy difference* even when each policy's own value CI is
wide.

$$\boxed{\; \text{Paired bootstrap of } d_i \text{ cancels the common-evaluator variance; the gap CI measures the *difference*, not two noisy levels.} \;}$$

**In plain terms.** We want the error bar on "how much better is the ordered policy than the context
policy," not on either one alone. Because both are graded by the *same* grader on the *same* pitches, most
of their noise is *shared* — when the grader is generous on a pitch, it's generous to both. Subtracting the
two, pitch by pitch, cancels that shared noise, so the difference is pinned down far more tightly than
either value is. That's why the gap can have a usefully narrow interval even when each policy's own value
is uncertain — and it's the same "compare on matched pairs" logic as a paired `t`-test.

---

## 8. The OPE noise floor: why `0.003 < 0.006` is undetectable at fixture scale

**Step — scale the CI half-width.** An importance-weighted OPE value is a mean of `n` per-row
contributions whose effective count is the **effective sample size** `ESS = (Σw)² / Σw²`. Its standard
error, and hence the CI half-width `h`, scales as

$$h \;\propto\; \frac{\hat\sigma_{\text{contrib}}}{\sqrt{\mathrm{ESS}}} \;=\; O\!\left(\frac{1}{\sqrt{\mathrm{ESS}}}\right).$$

At validation scale (`n_scored ~ 10^4`, `ESS ~ 0.5·n` at `α = 1`) the observed gap half-width is `h ≈
0.006`.

**Step — compare to the signal.** The myopically-reachable edge is `Δ ≈ 0.003` (§6). Since `Δ = 0.003 <
0.006 = h`, the gap CI **contains 0 by construction** — the signal is smaller than the noise, so no amount
of careful estimation resolves it at this scale. This is not a modelling failure; it is a *resolution*
statement.

**Step — the `n` required to resolve it (back-of-envelope).** To shrink `h` from `0.006` to below `Δ =
0.003` — a factor of `2` — needs `ESS` up by `2² = 4×`, i.e. roughly `4×` the effective data (with
comparable overlap), so `ESS ~ 2·10^4` and, at `~50%` ESS, `n_scored ~ 4·10^4`. To resolve it *comfortably*
(say `h = Δ/2 = 0.0015`, a `4×` shrink) needs `16×` the data. Full-data scale (`~1.5M` val+test rows) is
`~100×` the fixture, so at full scale the floor could in principle fall well below `0.003` — which is
exactly why *whether the real-data `C → O` gap resolves above 0 is the open question WS4 poses and cannot
answer at fixture scale* (a fixture-scale P0 does not preclude a full-scale P+).

**In plain terms.** How small an edge you can detect is set by your error bar, and the error bar shrinks
like one-over-root-of-the-effective-sample-size. At the small synthetic scale the error bar on the gap is
about `0.006`, and the myopic edge is about `0.003` — half the noise — so of course the interval covers
zero; you can't see a signal smaller than your noise. Halving the noise takes four times the data, and
seeing the edge *clearly* takes sixteen times. The real dataset is about a hundred times bigger, so at full
scale the noise floor could drop under `0.003` and the gap *might* resolve — but that's a question only the
full run answers, and a "can't tell" at fixture scale doesn't mean "nothing there."

---

## 9. The verdict rules, restated formally (D24 / D39)

**Step — the per-`α` agreement verdict (D24).** For the monitored estimator set `{DM, SNIPS, DR}` on
one-step (bandit) data, the verdict is

$$\text{verdict}(\alpha) = \begin{cases} \texttt{INCONCLUSIVE} & \exists\, (i,j):\; |V_i - V_j| > \max(h_i, h_j) \\ \texttt{CONSISTENT} & \text{otherwise,} \end{cases}$$

with `h_i` the 95% CI half-width of estimator `i` (`_agreement_verdict`, `eval/ope.py`). Raw IPS is
excluded (its variance would make the tolerance vacuous); FQE-on-bandit is excluded (identically DM). Any
material disagreement ⇒ INCONCLUSIVE, per SPEC §9's closing rule — *not* "it works."

**Step — the prescriptive verdict (D38/D39).** From the gap CIs across the `α` grid (`_prescription_verdict`):

$$\text{prescription} = \begin{cases}
\texttt{SEQ\_EXPLOITED} & \exists\, \alpha:\; \text{lower}_{95}\big(\Delta_{O-C}(\alpha)\big) > 0 \\
\texttt{SEQ\_INCONCLUSIVE\_MYOPIC} & \text{(above fails) and } \Delta_{O-C}(\alpha_{\max}) > 0 \text{ and world is positive} \\
\texttt{SEQ\_NEUTRAL\_PRESCRIPTION} & \text{otherwise (gap CI contains 0 at every } \alpha, \text{ point small).}
\end{cases}$$

`SEQ_EXPLOITED` is emitted **only** when the real-reward-anchored gap lower bound actually clears 0 — it is
never fabricated. On a *known-positive* world where the effect is sequential-not-myopic (§6), the honest
outcome is `SEQ_INCONCLUSIVE_MYOPIC` (a first-class D39 negative); on the *null* world it is
`SEQ_NEUTRAL_PRESCRIPTION`.

**Step — the honest-negative principle (D39).** An `INCONCLUSIVE` per-`α` verdict, a `lower_95` below
`V(μ)`, and a `SEQ_INCONCLUSIVE_MYOPIC` / `SEQ_NEUTRAL_PRESCRIPTION` overall verdict are **first-class
outcomes with their own interpretation**, not failures. They are pre-written as branches (G/V/P/verdict,
`PAPER.md` §6) so the correct reading is fixed *before* the real numbers arrive — the discipline that keeps
an honest null from being quietly reworked into a spurious positive.

**In plain terms.** Two verdicts run in parallel. The first asks, at each softening level, whether the
different OPE estimators *agree*; if they diverge by more than their own error bars, we refuse to call it —
"inconclusive," never "it works." The second summarizes the sequencing gap across all levels: we only say
"exploited" when the gap's lower bound genuinely clears zero on the real reward, and otherwise we say the
honest thing — "neutral" on the null world, "inconclusive-because-myopic" on the positive world where the
effect is a setup a bandit can't reach. Crucially, we wrote down what each of those verdicts *means* before
we ran anything, so a disappointing number can't be quietly retold as a win.
</content>
