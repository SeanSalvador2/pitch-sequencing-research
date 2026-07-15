# WS7 Theory: Conservative FQI, the Isolation Principle, the Exploitability Game, and the Honest Verdict

This document derives every formula the WS7 capstone uses, **one named step at a time**, with a **plain-words
explanation after each block**. Nothing is skipped. The boxed results match `PAPER.md` and the code in
`model.py` / `run_ws7.py` exactly; where the code names a function or a constant, it is noted. WS7 consumes
WS0's OPE machinery (`eval/ope.py`), WS3's artifacts (`../ws3_gbdt_stack/`), WS4's myopic-ceiling
decomposition (`../ws4_bandit/THEORY.md` §6), and WS5's refit-bootstrap and verdict machinery
(`../ws5_tabular_mdp/THEORY.md` §7–§8), so several sections recap and *extend* shared derivations rather than
re-deriving from scratch. Notation is introduced once in §1 and reused throughout. The intellectual payload is
§4 — the **isolation principle** — for which §1 (the backup), §2 (pessimism), and §3 (error propagation) are
the load-bearing preliminaries, and §8–§9 (the variance-bound certification and the honest verdict) are the
closing arithmetic.

---

## 1. Offline RL notation, the Bellman-optimality operator, and FQI as approximate value iteration

WS7 improves a policy from a fixed batch of logged plate-appearance (PA) transitions `(s, a, r, s′)`.

- `s ∈ 𝓢` — the pre-pitch **state** in a given representation (`O` = `build_view(·,"O")` + `action_family`,
  the rich ordered features; `count` = `balls·3 + strikes ∈ {0,…,11}`, exact tabular). Terminal = the last
  pitch of a PA (`succ = −1`).
- `a ∈ 𝓐` — the **action**, one of `A = 8` pitch families; `XX` (index 7) is descriptive-only and never
  feasible (SPEC §4). `𝓕(s) ⊆ 𝓐` is the leakage-safe feasible set (`effective_feasible_mask` fills empty rows
  to all non-`XX`).
- `r = R = −delta_run_exp` — the reward, larger better (SPEC §5).
- `μ(a | s)` — the behavior (logging) policy; `π(a | s)` — a target; `π_α = (1−α)μ + απ_greedy` — the softened
  target (`ope.pi_alpha`, SPEC §9).
- The PA **return** is the undiscounted `G = Σ_t R_t` (SPEC §5), so `γ = 1`; the PA terminates w.p. 1 (the
  count only ratchets; `../ws5_tabular_mdp/THEORY.md` §1), so the return is finite.

**Step — write the Bellman-optimality operator.** The optimal action-value `Q^*` is the fixed point of the
optimality operator `𝓣`,

$$(\mathcal{T} Q)(s, a) = \mathbb{E}\!\left[\,r(s,a) + \mathbf{1}[s\ \text{non-terminal}] \max_{a' \in \mathcal{F}(s')} Q(s', a')\,\right],$$

the max taken over the successor's *feasible* actions (infeasible actions are outside the arg-max).

**Step — insert the pessimism penalty (the conservative operator).** WS7 backs up the *penalised* successor
value (§2 derives the penalty). With `pen(s,a) = 1[μ̂(a|s) < floor]` and coefficient `λ`,

$$\boxed{\; Q_{k+1}(s, a) \;\leftarrow\; r(s, a) \;+\; \mathbf{1}[s\ \text{non-terminal}]\;
  \max_{a' \in \mathcal{F}(s')}\big[\,Q_k(s', a') - \lambda\,\mathrm{pen}(s', a')\,\big]. \;}$$

This is `fitted_q_iteration` (tabular `count`) and the LightGBM loop in `ConservativeFQI.fit` (`O`): `Q_0` is
the reward regression `Ê[r | s, a]`; each iteration forms the target `y = r + V_{\text{pen}}(s')` on the
non-terminal rows (`V_{\text{pen}}(s') = \max_{a'∈𝓕(s')}[Q_k(s',a') − λ\,\mathrm{pen}(s',a')]`,
`_penalized_state_value`) and refits `Q_{k+1}` — an exact per-`(s,a)` group mean for `count`, a LightGBM
regressor over `build_view(O)+action_family` for `O` (the counterfactual grid swept by swapping
`action_family`, the WS3 `q_grid` pattern).

**Step — contraction for the tabular backend.** On the tabular `count` state with an exact group-mean fit and
`γ = 1` on a *proper* (terminating) episodic MDP, `𝓣` is a contraction in the weighted sup-norm induced by the
sub-stochastic non-terminal block (mass leaks to terminals, spectral radius `< 1`;
`../ws5_tabular_mdp/THEORY.md` §1). So `Q_k → Q^*` geometrically, and the arg-max policy is optimal on the
reached cells — which is exactly what `test_fqi_recovers_analytic_optimal_q_two_step` verifies: unpenalised
FQI recovers the analytic `Q^*` on the synthetic two-step MDP (`Q_1^* = R_1`, `Q_0^* = R_0 + \max_a R_1`) to
`< 1e-9`.

**Step — why a small `K` suffices for short PAs.** Each backup propagates value **one pitch back**. A finite-
horizon MDP of depth `H` reaches its exact fixed point in `H` backups (there is no value beyond the terminal to
propagate). A PA is `≤` a dozen pitches, so `n_iter = DEFAULT_N_ITER = 6` requested, with an early stop when
the mean taken-action drift `|Q_{k+1} − Q_k|` drops below `drift_tol = 1e-4`, reaches the finite-horizon fixed
point; in the completed validation it converged in `K = 3` effective backups.

**In plain terms.** Fitted-Q iteration is "guess the value of each pitch-and-situation, then repeatedly improve
the guess by looking one pitch ahead and keeping the best feasible follow-up — minus a penalty for follow-ups
the data barely saw." On the simple count board the improvement is exact averaging and provably lands on the
true best-play value (a unit test confirms it to nine decimals). Because an at-bat is only a handful of pitches
long, you only need to look ahead a handful of times — three passes were enough — and the code stops as soon as
the numbers stop moving.

---

## 2. Pessimism: how the support penalty lower-bounds Q on the unexplored, and its effect on the arg-max

This is the conservatism that keeps the improved policy near behavior support — the SPEC §9 discipline pushed
inside the backup.

**Step — define the indicator support penalty.** With the behavior-probability floor `f = DEFAULT_SUPPORT_FLOOR
= 0.02` (the SPEC §9 `support_threshold`),

$$\boxed{\; \mathrm{pen}(s, a) = \mathbf{1}\!\big[\hat\mu(a \mid s) < f\big] \in \{0, 1\}, \;}$$

a per-row `(n, 8)` grid (`behavior_support_penalty`). `μ̂` is WS3's contextual propensities (D33) or the
empirical count-state behavior.

**Step — show it lower-bounds the effective Q on off-support actions.** The backup and the greedy score the
**penalised** value `Q̃(s, a) = Q(s, a) − λ\,\mathrm{pen}(s, a)`. On an off-support action (`μ̂ < f`) this is
`Q(s,a) − λ`; on an on-support action it is `Q(s,a)`. So for every `(s,a)`,

$$\tilde Q(s, a) \le Q(s, a), \qquad \text{with equality iff } \hat\mu(a \mid s) \ge f,$$

i.e. the penalty is a **one-sided** correction that pushes down *only* the actions the logged data cannot
support a reliable estimate for (whose bootstrapped `Q` is off-support and untrustworthy). It never inflates a
value; it can only demote a poorly-supported one.

**Step — its effect on the arg-max (why conservatism can *raise* the held-out value).** The greedy
(`conservative_greedy`) is `argmax_{a∈𝓕(s)} Q̃(s,a)`. An off-support action wins only if it clears the best
on-support action by more than `λ`:

$$\pi(s) = \arg\max_{a \in \mathcal{F}(s)} \big[Q(s,a) - \lambda\,\mathrm{pen}(s,a)\big]
  \;\Rightarrow\; \text{an off-support } a \text{ is chosen only if } Q(s,a) - \lambda > \max_{a'\ \text{on-support}} Q(s,a').$$

`test_harsh_penalty_makes_greedy_retreat_to_support` pins this: a high-`Q`, below-floor action is chosen at
`λ = 0` and *retreats* to the lower-`Q`, high-support action at `λ = 2`. Because an off-support `Q` is exactly
the one a flexible function approximator is most likely to have *inflated* (thin data, high variance; §3),
demoting it can raise the *held-out* FQE value — the value-vs-λ exhibit's `+0.0328 → +0.0354` from `λ = 0` to
`λ ≥ 0.05`: a modest penalty is "free honesty" here, slightly value-positive, and it bites `~2%` of decisions
at the floor (`pessimism_bites_frac`).

**Step — the honest relation to CQL.** Conservative Q-learning (Kumar et al., 2020) adds a regulariser
`α_{\text{CQL}}(\mathbb{E}_{a∼π}[Q(s,a)] − \mathbb{E}_{a∼μ}[Q(s,a)])` to the Bellman objective, learning a
value lower bound on out-of-distribution actions. WS7's indicator penalty is a transparent **approximation**:
it hard-floors the behavior probability rather than learning the regulariser's action-weighting, and it acts
per-`(s,a)` inside the max rather than as an objective term. It is *CQL-lite*, and `PAPER.md` §4.2 and this
section state that honestly — it is not CQL, and it does not claim CQL's lower-bound guarantee, only CQL's
*intent* (don't reward the unexplored).

**In plain terms.** The penalty is one bit — "did the real pitcher throw this pitch here at least 2% of the
time?" If not, we quietly subtract a little from that pitch's estimated value before the model is allowed to
recommend it. That only ever *lowers* the value of pitches the data barely saw, and it only changes the
recommendation when such a pitch was about to win by a hair. Since those barely-seen pitches are exactly the
ones a flexible model tends to *over*-rate on thin data, gently demoting them can actually make the real,
held-out value go *up* — "free honesty," until you crank the penalty so high it starts refusing good pitches
too. It is a plain-spoken stand-in for the fancier CQL idea, and we say so.

---

## 3. Function-approximation error propagation in FQI — why flexibility costs variance downstream

This section sketches, at the named-step level, why the flexible `O`-view FQI carries *more* estimation
variance into the OPE than the exact tabular `count` FQI — the mechanism behind the study's
flexibility-costs-variance finding.

**Step — write the one-step fit error.** At iteration `k` the regressor fits `Q_{k+1}` to the target
`y = r + V_{\text{pen}}(s')`, incurring a fit error `ε_k = Q_{k+1} − 𝓣_λ Q_k` (regression bias + estimation
variance). For the exact tabular fit on a well-sampled cell `ε_k` is pure estimation noise `∝ σ/√n_{s,a}`; for
the LightGBM fit it also carries approximation error and the extra variance of a flexible learner on thin
strata.

**Step — propagate through the backups (the compounding bound).** Value iteration composes `K` backups, and the
optimality operator's Lipschitz constant in sup-norm is `1` at `γ = 1` on the proper episodic MDP (§1), so
per-iteration errors accumulate additively rather than being damped:

$$\boxed{\; \big\|Q_K - Q^*\big\|_\infty \;\le\; \sum_{k=0}^{K-1} \big\|\varepsilon_k\big\|_\infty \;+\; \big\|Q_0 - Q^*\big\|\text{-transient}, \;}$$

the classic FQI error-propagation form (Ernst et al., 2005; the discounted version has a `1/(1−γ)` prefactor,
here replaced by the finite horizon `K`). The point is **directional**: the bound scales with the per-fit error
`\|ε_k\|`, which is larger for the flexible learner on thin cells.

**Step — carry it to the OPE (the downstream variance).** The FQI `Q` enters the pipeline only through the
*greedy policy* it induces; that policy is then scored by an *independent* held-out FQE (§8) and step-wise DR.
A noisier `Q` induces a *noisier greedy* (its arg-max flips on more cells across resamples), so the refit-
bootstrap FQE of the induced policy has a **wider** replicate distribution — the isolation gap's CI is wider
for the flexible `O` view than it would be for a coarse exact state. This is why WS7's O-vs-count isolation CI
(`[−0.0129, +0.0157]`) is wider than WS5's tabular trigger-count CI at comparable scale, and why WS7 is *more*
variance-bound than WS5 despite being *more* capable.

**In plain terms.** Every pass of the value-improver fits a model to a moving target, and each fit has some
wobble. A flexible model (the boosted trees on the rich ordered state) wobbles more than a simple averager on a
coarse board, especially where the data is thin — and because looking one pitch ahead stacks these fits, the
wobble adds up rather than washing out. That wobble doesn't stay inside the model: it makes the recommended
plan itself jitter from sample to sample, which makes the *honest error bar* on the plan's value wider. So the
fancier model, which can *represent* more, ends up with a *shakier* measurement at the same amount of data —
the exact reason the capstone is less able to certify the setup than the simpler tabular rung below it.

---

## 4. The isolation principle, formalized — differencing out the count term

This is the workstream's payload. It shows why the raw `FQI-O − behavior` gain cannot certify sequencing and
why the **O-vs-count** difference can — the RL lift of WS4's ceiling decomposition.

**Step — recap WS4's action-value decomposition.** WS4 split the per-decision expected reward into a state
baseline and a family-differential (`../ws4_bandit/THEORY.md` §6):

$$\mathbb{E}[R \mid s, a] = v(s) + \delta(s, a), \qquad \sum_a \mu(a\mid s)\,\delta(s,a) = 0,$$

where `v(s)` is owed to every action and `δ` is the part that differs across *today's* action. A myopic policy
reaches only `δ`; the planted setup lives in `v(s)` (the trigger is fixed by history), hence WS4's `~0.003`
ceiling.

**Step — lift to the sequential action-value.** At `γ = 1` the FQI action-value keeps the look-ahead term a
bandit drops (`../ws5_tabular_mdp/THEORY.md` §4):

$$Q(s, a) = \underbrace{\mathbb{E}[R \mid s, a]}_{\text{this pitch}} + \underbrace{\mathbb{E}[v(s') \mid s, a]}_{\text{the setup}}.$$

Now decompose a policy's *value* the same way. Write the improved policy's value as the behavior value plus a
count-improvement term plus a sequencing term. The **count** state carries only `(balls, strikes)`, so an
`FQI-count` policy can optimise the count-driven part of the value — the part `v(count)` dominated by the count
(Tango et al., 2007) — but its state cannot represent the ordered-history setup, so its `E[v(s')|s,a]` variation
is (to smoothing) constant across `a` and it cannot cash the setup. The **O** state carries the ordered history,
so an `FQI-O` policy optimises *both* the count part *and* the sequencing part.

**Step — show `FQI-O − behavior` confounds the two.** Let `V_π` denote the held-out value of policy `π`. Then

$$\underbrace{V_{\text{FQI-O}} - V_{\text{behavior}}}_{\text{raw gap}}
  = \underbrace{\big(V_{\text{FQI-count}} - V_{\text{behavior}}\big)}_{\text{count-driven improvement}}
  \;+\; \underbrace{\big(V_{\text{FQI-O}} - V_{\text{FQI-count}}\big)}_{\text{sequencing isolation}}.$$

The first term is large and positive whenever the logging policy is *habit-based* (not reward-optimal), because
any count-aware policy beats it on the count — which is exactly the synthetic behavior policy, and plausibly
the real one. So the **raw gap is count-driven**, and certifying sequencing on it over-claims.

**Step — the isolation differences out the count term.** Subtracting the count rung's value cancels the shared
count-improvement term and leaves the sequencing contribution:

$$\boxed{\; \text{isolation} \;=\; V_{\text{FQI-O}} - V_{\text{FQI-count}} \;=\; \big(V_{\text{FQI-O}} - V_{\text{behavior}}\big) - \big(V_{\text{FQI-count}} - V_{\text{behavior}}\big), \;}$$

the RL analog of WS4's `O − C` gap and WS5's trigger-count gap. This is `run_ws7._rl_verdict`'s basis: the
verified positive-world isolation is `−0.0003` (CI `[−0.0129, +0.0157]`, within noise) against a raw
`+0.035`, and the null-world isolation is `−0.0016`. The isolation **discriminates** (positive `−0.0003 >`
null `−0.0016` at every scale probed) even though its point sign is within noise — the world-gated
`RL_EVIDENCE_DIRECTIONAL`.

**In plain terms.** The fancy model easily beats the real pitchers' habits — but almost all of that win is just
*count management* that any competent model gets right, and has nothing to do with sequencing. To measure the
sequencing part alone, we run the *same* value-improver on a stripped-down "count-only" board and subtract:
whatever extra the ordered-history model earns *over* the count-only model is the sequencing-specific slice.
That slice is tiny and, at this data size, buried in noise — but it is reliably *larger* in the world where we
planted a real effect than in the world where we didn't, so the instrument is discriminating even when it can't
yet put a confident number on the slice. Leading with the raw win would have been the easy, wrong headline; the
isolation is the discipline that stops it.

---

## 5. The two-gate logic — what each gate catches

WS7 enforces OPE-before-policy (SPEC §0.3) with **two** gates; either failing prints `FAILED_GATE` and stops
before any policy value (`run_ws7`, before the FQI fit).

**Step — Gate 1: behavior-policy recovery calibrates the yardstick.** Set the target equal to the behavior
policy `π = μ` on the held-out logged rows. Then every importance weight is `w = μ(a|s)/μ(a|s) = 1`, so IPS,
SNIPS, and DR must all return the observed held-out mean reward exactly (`ope.behavior_policy_recovery`,
`ips_weights_unit = True`). What it catches: a **miscalibrated yardstick** — a propensity/reward join error, a
mis-indexed episode, a wrong denominator — anything that makes the harness misreport even the *observed*
policy's value. If the ruler cannot reproduce the length of the thing you already measured, no new measurement
below it is trustworthy. Verified: observed value `+0.0197` (null), `+0.0345` (positive), IPS weights ≡ 1.

**Step — Gate 2: the logged-bandit fixture regression checks the estimator implementations.** Run the shared
OPE on `pitchseq.synth.make_logged_bandit` — a bandit with a known logging policy *and* an analytically
computable target value — and require (a) recovery of the logging value and (b) recovery of a **known** target
value within its bootstrap CI (`_logged_bandit_gate`). What it catches: an **implementation bug in an
estimator** that real data cannot expose because real data has no ground-truth target value. Gate 1 checks the
*identity* target (`π = μ`, weights ≡ 1) on real data; Gate 2 checks a *non-trivial* target against analytic
truth on a fixture. The two are complementary — one calibrates the yardstick on the data of interest, the other
proves the yardstick's mechanism on a problem with a known answer. Verified: known `+0.6148` vs estimated
`+0.6229`, within CI.

**Step — the conjunction.** The FQI runs only if **both** pass:

$$\boxed{\; \text{proceed} \iff \big(\text{recovery.passed}\big) \wedge \big(\text{logged-bandit.passed}\big); \quad \text{else } \texttt{FAILED\_GATE}, \text{ stop.} \;}$$

**In plain terms.** Before trusting the scale to weigh a new policy, we check it two ways. First, can it read
back the weight of the pitcher's *own* policy — the thing we already know — on this exact data? That catches a
broken hookup (wrong propensities, a join gone sideways). Second, on a toy problem where we can compute the
right answer by hand, does each estimator hit that answer? That catches a bug in the estimator's math that real
data would hide, because real data never comes with an answer key. Only if the scale passes both do we let it
weigh anything new.

---

## 6. Zero-sum matrix games: minimax, the LP, and exploitability as the equilibrium shortfall

This section derives the SPEC §10 exploitability read-out and states the payoff model's assumptions explicitly.

**Step — write the per-count payoff.** For each count `s` build a zero-sum game between the **pitcher** (row
player, chooses family `a`, maximiser) and a **family-anticipating batter** (column player, sits on family
`k`, minimiser). The pitcher's payoff (`payoff_by_count`) is

$$\boxed{\; R(s, a, k) = \hat q(s, a) \;-\; \beta\,p_{\text{swing}}(s, a)\,\mathbf{1}[a = k], \qquad \beta = \texttt{DEFAULT\_DISRUPTION\_BETA} = 0.12, \;}$$

where `q̂(s,a) = E[R|s,a]` is the common outcome model's per-count mean (WS3's grid) and `p_{\text{swing}}(s,a)
= P(\text{swing} | s, a)` is the fixed `BatterResponseModel`'s marginal. Only the matched diagonal `a = k`
carries the anticipation penalty: a correctly-guessing batter converts more swings into damage.

**Step — state the payoff model's assumptions.** (i) The batter model is **fixed** (fit once on train, never
adapted) and **marginalised over location** (an execution quantity unknown pre-pitch, so by total probability
the game uses the raw `(count, family)` swing rate — `BatterResponseModel.swing_prob`). (ii) The anticipation
effect is a *single* multiplicative disruption `β·p_swing` on the matched family — a modelling choice, not a
learned response. (iii) The game is **per-count**, aggregating the row-level `q̂` to 12 count means, which keeps
the LP small and the game interpretable while every decision row is still scored in *its own* count's game.
(iv) It is **model-dependent by necessity** (SPEC §10) — the *levels* move under a different response model; the
robust claim is the *direction*.

**Step — the minimax value and its LP.** By von Neumann's (1928) minimax theorem a finite zero-sum game has a
value `V_{eq}` equal to both the maximin and the minimax. The pitcher's maximin program is
`max_{x∈Δ_A} min_k (x^T R)_k`, linearised (`equilibrium_value`) by introducing `v = min_k (x^T R)_k`:

$$\boxed{\; \max_{x,\,v}\ v \quad \text{s.t.}\quad \sum_a x_a\,R[a,k] \ge v\ \ \forall k,\ \ \sum_a x_a = 1,\ \ x \ge 0, \;}$$

cast for `scipy.linprog` (a minimiser) as `min −v` with variables `[x_0,…,x_{A−1}, v]`, column constraints
`−R[:,k]^T x + v ≤ 0`, and the simplex equality. The optimal `v` is `V_{eq}`; the fictitious-play iteration is
the documented fallback. `test_equilibrium_value_matching_pennies_exact` pins it (value `0`, strategy
`(.5,.5)`); `test_equilibrium_asymmetric_game` pins a non-symmetric value `1.5`.

**Step — define exploitability.** If the pitcher commits to mixed strategy `π`, the batter best-responds by the
column minimising the pitcher's value, so the pitcher's guaranteed value is `min_k (π^T R)_k`. Exploitability
is the shortfall from equilibrium (`policy_exploitability`):

$$\boxed{\; \mathrm{expl}(\pi; s) = V_{eq}(s) - \min_k (\pi^T R_s)_k \;\ge\; 0, \;}$$

zero for the equilibrium (maximin-optimal) policy and positive-and-growing for predictable/concentrated ones
(`test_exploitability_dominated_positive_equilibrium_zero`: a pure policy in matching pennies has
exploitability `1`, the mixed equilibrium `0`). The read-out (`exploitability`) scores each row in its own
count's game and averages, with a pitcher-game cluster bootstrap. Verified: behavior `+0.0137`, FQI-O@1
`+0.0459`, FQI-count@1 `+0.0475` — concentrated policies `~3×` behavior.

**In plain terms.** Picture each count as a guessing game: the pitcher picks a pitch, an imagined batter
"sits on" a family, and if the batter guessed right he does extra damage (scaled by how often that pitch gets
swung at). The best the pitcher can guarantee, against a batter who always guesses his most-exploitable option,
is the game's *equilibrium* value — found by a small linear program that mixes his pitches so no single guess
hurts too much. A policy's *exploitability* is how far short of that equilibrium it falls: zero if it mixes
perfectly, large if it is predictable. Our optimised policies concentrate on a favourite pitch, so a guessing
batter claws back about three times as much against them as against real pitchers' natural mixing — that is the
price of predictability, and it is by construction (the batter model is fixed and interpretable, not a learned
adversary).

---

## 7. The frontier as a multi-objective read-out — Pareto dominance, no scalarization

**Step — define the frontier point.** Each policy `π` is a point in a five-dimensional read-out
(`FRONTIER_COLUMNS`, `assemble_frontier`):

$$\pi \;\mapsto\; \big(\,V(\pi),\ \underline V_{95}(\pi),\ B_{\text{seq}}(\pi),\ \mathrm{expl}(\pi),\ \mathrm{TV}(\pi\|\mu),\ \text{compute}(\pi)\,\big),$$

the OPE value and its one-sided lower bound, the ordered-history bits the policy's view exploits (`0` for
count/behavior, `B_seq` for O), the exploitability, the total-variation distance from behavior, and the compute
(`params` / `wall_clock_s`).

**Step — define Pareto dominance.** Orient each axis so *larger is better* (value up; `−exploitability`,
`−TV`, `−compute` up; `B_seq` read as informative, not directly good/bad). Policy `π` **Pareto-dominates** `π′`
iff it is no worse on every axis and strictly better on at least one:

$$\boxed{\; \pi \succ \pi' \iff \big(\forall j:\ o_j(\pi) \ge o_j(\pi')\big) \ \wedge\ \big(\exists j:\ o_j(\pi) > o_j(\pi')\big). \;}$$

The **frontier** is the set of non-dominated policies. A policy with more value *and* less exploitability *and*
less deviation dominates; the interesting policies are the ones that trade (more value bought with more
exploitability or deviation), which no partial order ranks.

**Step — why no scalarization is imposed.** A scalarised score `Σ_j w_j o_j` would require weights `w_j`
trading a bit of run value against a bit of exploitability against a unit of compute — an *external* preference
the study does not own and should not fabricate. Imposing one would (i) hide the trade-off the frontier exists
to show and (ii) make the "winner" an artifact of the chosen weights. So `plot_frontier` renders the raw
axes — value vs deviation coloured by exploitability (left), exploitability vs `B_seq` sized by compute
(right) — and lets the reader supply the preference. The frontier is the study's summary object precisely
because it is *not* a leaderboard.

**In plain terms.** Every policy gets a little report card with five grades: how much value it earns, how sure
we are of the low end, how predictable it makes the pitcher, how far it strays from real behavior, and how much
compute it costs. One policy "beats" another only if it is at least as good on *all five* and better on one —
and the policies worth arguing about are exactly the ones that win some and lose some, which no single ranking
can settle. We deliberately refuse to mash the five grades into one number, because that would require deciding
how many runs a bit of predictability is worth — a judgment the study doesn't get to make for you. The chart
shows the trade-offs and lets you choose.

---

## 8. The refit-bootstrap CI and the certification sample-size arithmetic for the isolation gap

**Step — recap the degeneracy and the refit fix (D44).** FQE's per-episode contribution is the initial-state
value `V(s_0)`, and every PA starts in the same state, so the contribution array is constant and a resampling
bootstrap of it is structurally degenerate (its CI collapses to a point) — the WS5 derivation, verbatim
(`../ws5_tabular_mdp/THEORY.md` §7). WS7 reuses WS5's `_fqe_refit_bootstrap`: for each replicate, resample
pitcher-game clusters of episodes, **refit** FQE from scratch per `(view, α)` on the resample — the *same*
resample across arms — and read percentile CIs off the replicates. The isolation gap is **paired**:

$$\widehat{\text{iso}}^{(b)} = \hat V^{(b)}_{O,\alpha} - \hat V^{(b)}_{\text{count},\alpha},$$

so the shared episode-sampling noise differences away and the gap CI is tighter than either level's. Degenerate
CIs print `n/a (constant contributions)`, never an interval.

**Step — scale the half-width.** The isolation is a mean over pitcher-game clusters, so its standard error and
CI half-width scale as `h ≈ c_0 / √(n_clusters) = O(n_clusters^{-1/2})`. At synthetic scale the verified
positive-world isolation CI at the top `α` is `[−0.0129, +0.0157]`, a half-width `h ≈ 0.0143`.

**Step — recover the current standard error.** To certify, the one-sided 95% lower bound of the isolation of
size `Δ` must exceed the ceiling `c = 0.003`; `LB = Δ − 1.645\,se`. Taking the discriminating signal
`Δ ≈ +0.003`–`+0.006` (the isolation sits within noise but discriminates; use the WS5-corroborated setup scale
`~+0.006`) with `h ≈ 0.0143 ⇒ se_0 ≈ h/1.96 ≈ 0.0073`:

$$1.645\,se_0 \approx 0.012, \qquad \text{so } LB_0 \approx \Delta - 0.012 < 0 \ \Rightarrow\ \text{does not clear } c.$$

**Step — invert for the cluster count.** We need `se_{\text{target}} ≲ (Δ − c)/1.645`. For a setup-scale
`Δ ≈ +0.006` over `c = 0.003`, `se_{\text{target}} ≲ 0.0018`, and since `se ∝ n^{-1/2}`,

$$\boxed{\; \frac{n_{\text{target}}}{n_0} = \Big(\frac{se_0}{se_{\text{target}}}\Big)^2 \approx \Big(\frac{0.0073}{0.0018}\Big)^2 \approx 16\times. \;}$$

**Step — connect to Phase 2, honestly.** WS7 is *more* variance-bound than WS5 (§3), so its required multiple
(`~16×`) exceeds WS5's (`~6×`) at the same signal — the flexibility tax on certification. The full data is
`~40×` the fixture, comfortably past `~16×` **if** the isolation signal and overlap hold at scale. So a
fixture-scale `RL_EVIDENCE_DIRECTIONAL` does **not** preclude a full-scale `RL_EVIDENCE_CERTIFIED`; it is an
honest deferral, and the flexible model's larger `n`-to-certify is the quantified cost of its flexibility.

**In plain terms.** How small a sequencing edge we can *prove* is set by our error bar, and the error bar
shrinks like one-over-the-square-root of how many independent pitcher-games we have. At synthetic scale our
error bar on the sequencing slice is several times too wide to prove it clears the myopic bar, and closing that
gap takes very roughly sixteen times the data — more than the simpler tabular model needed, because the
flexible model is shakier per data point (§3). The real season is about forty times bigger than our fixture, so
at full scale the error bar could shrink under the bar — which is why "can't certify yet" is a *Phase-2
deferral*, not a null, and why the fanciest model needs the most data to make good on what it can represent.

---

## 9. The honest-verdict calculus — the conjunction gate's false-positive control, plus WS5 corroboration

**Step — write the verdict gate.** `RL_EVIDENCE_CERTIFIED` fires (positive/real world; `_rl_verdict`) iff, at
the top `α`, **all three** hold, with the D40 ceiling `c = 0.003`:

$$\boxed{\; \underbrace{\text{lower}_{95}\big(\text{iso}^{\text{FQE}}\big) > c \ \wedge\ \neg\,\text{degenerate}}_{\text{(1) resolving lens, real CI}} \ \wedge\ \underbrace{\text{iso}^{\text{stepDR}} > 0}_{\text{(2) sign check}} \ \wedge\ \underbrace{\text{ws5.directional\_positive}}_{\text{(3) cross-workstream corroboration}}. \;}$$

`RL_INCONCLUSIVE` overrides everything on a D24 step-wise-DR/FQE disagreement; `RL_NO_CLAIM` is the null world
regardless; `RL_EVIDENCE_DIRECTIONAL` is the world-gated positive default; `RL_EVIDENCE_ABSENT` the real-data
no-evidence case.

**Step — derive why the conjunction controls false positives (recap WS5 §8, extended).** Each leg fails in a
*different* direction, so requiring agreement makes an accidental clearance improbable. (1) The refit-FQE lower
bound is the only lens with a *resolving* CI, but a degenerate or wide interval cannot clear `c` — it guards
against fake confidence (§8). (2) The step-wise-DR gap has real per-episode variance but a heavy-tailed per-PA
importance-weight product, so its CI is too wide to bound the ceiling — used only as a **sign** check. (3) The
**WS5 cross-check** is the extension beyond WS5's own three-lens gate: an *independent* workstream (a low-
variance tabular MDP) must point the same way — a corroboration whose error source (tabular estimation) is
different again from WS7's (function-approximation OPE). A spurious clearance would need the refit-FQE lower
bound (hard), the DR sign (independent noise), *and* a separate model's sign to line up:

$$\Pr[\text{false } \texttt{CERTIFIED}] \le \min_k \Pr[\text{leg }k\text{ clears spuriously}] \ \text{— dominated by the real-CI FQE bound.}$$

**Step — why the WS5 term is corroboration, not a gate.** WS5's cross-check is required for `CERTIFIED` but
**not** for `DIRECTIONAL` (which rests on WS7's own estimators), because at synthetic scale it is honestly
noisy — its trigger-count sign is not even stable (`−0.0213` positive world, `+0.0137` null). It is read as
directional corroboration; on real data at `~40×` scale the two isolation signals should agree in sign, and
requiring that agreement for the strong claim is a cheap, independent false-positive control.

**Step — the world-gating, stated honestly.** On a synthetic world whose ground truth we *planted*, the honest
OPE outcome is direction-only, so the positive world is **world-gated** to `RL_EVIDENCE_DIRECTIONAL` when it
does not certify — exactly as WS4's `SEQ_INCONCLUSIVE_MYOPIC` and WS5's `SETUP_INCONCLUSIVE` are. What is
asserted deterministically (`test_positive_world_directional_and_discriminates`) is the *world-discrimination*
(positive isolation > null isolation) and the *non-clearance* (the CI does not clear `c`), never the isolation
*point sign* (which flips with the small eval sample). The `CERTIFIED` path is proven reachable on an
engineered clear (`test_verdict_certified_path_reachable`), so `DIRECTIONAL` is a data-scale statement.

**In plain terms.** We only stamp "certified" when three different instruments, that go wrong in three
different ways, all agree: an honest error bar that actually clears the bar (not a fake-narrow one), an
importance-weighted estimate pointing the right *direction*, and a *separate* simpler model from the previous
chapter pointing the same way. For a false alarm, all three independent things would have to line up at once —
unlikely, and pinned down by the one real error bar. The separate-model check is corroboration, not a gate,
because at small scale it is too jittery to lean on — but at full scale, demanding that WS5 and WS7 agree is a
cheap, independent guard. And on a world where we know the truth because we planted it, the honest verdict is
"the direction is right and the worlds are told apart, but the error bar can't certify it yet" — the same
first-class "not yet" WS4 and WS5 reported, now at the top of the ladder.
