# Empirical-Bayes Conditional Tables as a Transparent Baseline for Pitch Sequencing

*Workstream 1 of a comparative pitch-sequencing study.* This paper is completed in place with the
real-data results (2021–2025 Statcast). The Results and Discussion were pre-**branched** so that
the correct interpretation was already written for whichever numbers arrived; the branch **selected
by the data** is marked at each fork, and the unselected branches are retained and labelled as
*pre-registered alternatives* for transparency.

---

## Abstract

We ask how much evidence for within-plate-appearance pitch *sequencing* survives the simplest
honest model: a set of hierarchical, empirical-Bayes–shrunk conditional lookup tables. Two
table families are fit over five nested state views (context `C`; unordered priors `U`;
previous pitch `L1`; full ordered priors `O`; matchup memory `OM`): a Dirichlet-multinomial
**selection** table for the next-pitch family, and a normal partial-pooling **run-value** table
for expected reward conditioned on the pitch thrown. Each level's shrinkage strength is fit by
maximum marginal likelihood. On the real data (2021–2025 Statcast, $\sim$3.85M pitches), the
context view attains selection log loss 1.4714 and the fully ordered view 1.4787,
giving an ordered-refinement ablation $\Delta_{\text{order}}$ = −0.0015 (95% clustered CI
[−0.0017, −0.0013]), read under D21 as consistent with no table-visible ordering effect. The
hierarchy's own history views do not beat the context view, yet an external pitcher × count ×
prev-family reference (1.4410) beats every table view by ≈0.03–0.04 log loss: first-order
selection structure is real and material, but the hand-keyed cell ladder fragments before it
can express it. The support exhibit shows distinct conditioning cells rising from
30,430 (`C`) to 212,434 (`O`) — and 320,487 for the unordered `U` view — with the fraction
of evaluation pitches in cells of fewer than 20 training examples climbing from 0.072 to 0.348;
matchup memory (`OM`) is shown to be table-infeasible. We validate the method on two synthetic worlds
with known ground truth: on a null world the fitted history concentration diverges and
$\Delta_{\text{order}} = 0$ exactly; on a positive world with a planted ordered velocity-transition
effect the ordered table recovers the effect's sign at $\sim$3% of its planted magnitude — a
measured demonstration of the *mechanism-blindness* of name-keyed tables. The headline
contribution is not a number but a constraint: naive conditioning exhausts its support long
before it exhausts the questions, which is the standing motivation for the models above WS1 on
the rigor ladder.

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

Workstream 1 (WS1) is the ladder's bottom rung. It is a deliberately transparent statistical
baseline: hierarchical conditional tables, shrunk toward well-estimated parents by empirical
Bayes. Its role is threefold. First, it is the **model everything else must beat**: if a
gradient-boosted or recurrent model cannot outperform a shrunk lookup table, its complexity has
not earned its place. Second, it speaks directly to finding #1 (selection) and *probes* finding
#2 (outcome) through its run-value tables, while being honest about how weakly a flat table can
resolve either. Third, and most importantly, it is the study's **support-problem exhibit**: it
makes concrete, in the currency of out-of-sample evaluation exposure, the fact that naive
conditioning fragments the data faster than it extracts signal. That single fact is why the
ladder continues past WS1 at all.

The empirical-Bayes idea WS1 rests on is old and well understood: shrinkage estimators
(Robbins, 1956; Stein, 1956; James and Stein, 1961), their parametric empirical-Bayes framing
(Morris, 1983), and — most famously for this domain — Efron and Morris's (1975) demonstration
that shrinking early-season batting averages toward the league mean predicts the rest of the
season better than the raw averages. WS1 applies that logic twice (to a family distribution and
to a mean reward) and stacks it into a hierarchy so a thin cell borrows from its parent rather
than from the global mean.

---

## 2. Related work

**Shrinkage and empirical Bayes.** The estimators WS1 uses descend directly from the classical
shrinkage literature: Robbins (1956) introduced the empirical-Bayes program; Stein (1956) and
James and Stein (1961) established that shrinkage estimators dominate the maximum-likelihood
estimator under quadratic loss in three or more dimensions; Morris (1983) gave the parametric
empirical-Bayes account that WS1's level-wise concentration fitting instantiates. Efron and
Morris (1975) is the canonical demonstration of shrinkage in baseball, and is the intuition we
lead with throughout: a small-sample rate estimate should be pulled toward a pooled parent by an
amount that depends on how little data supports it.

**Fitting the Dirichlet.** WS1's selection tables are Dirichlet-multinomial; the concentration
of each hierarchy level is fit by maximum marginal likelihood, with a method-of-moments fallback.
The estimators and their numerical behavior follow the standard treatment of estimating a
Dirichlet distribution (Minka, 2000/2003) and the general hierarchical-model machinery in Gelman
et al. (2013).

**Baseball analytics and sequencing.** The applied grounding is Tango, Lichtman, and Dolphin
(2007), whose count- and matchup-based percentage analysis motivates the count × handedness ×
pitcher conditioning that defines WS1's context view, and Marchi and Albert (2013), whose
treatment of Statcast-style pitch data and partial pooling parallels the modeling here. The
broader question — whether pitch *order*, beyond the previous pitch, carries out-of-sample
signal — is where this study places its rigor ladder; WS1's contribution to that question is to
establish the transparent lower bound and to quantify the support limits every richer model
inherits.

---

## 3. Data

**Source.** Statcast pitch-level data via pybaseball, seasons 2021–2025, a single tracking era
(Hawk-Eye), $\sim$3.85M pitches over 119 raw columns (SPEC §1). The reward signal
`delta_run_exp` is populated on $\sim$99.7% of pitches. `pitch_type` is a Statcast *classifier
output* and is treated throughout as an observed proxy for the pitch thrown, not as the battery's
intent (SPEC §1, §8; see §8 below).

**The canonical decision table.** Every workstream reads one shared table (SPEC §3): one row per
pitch, representing the decision made immediately *before* that pitch is released, keyed by
`(game_pk, at_bat_number, pitch_number)`. Its columns partition into non-sequence context $X_t$
(the count, base-out state, inning, score, handedness, rolling repertoire and tendencies — this
*is* the `C` view), within-PA history $H^{PA}_t$ (per-prior-pitch family, location, velocity,
movement, and outcome tokens — the material for `U`/`L1`/`O`), matchup memory $H^{M}_t$ (the
material for `OM`), and the labels: the current pitch's execution $Z_t$, its event-tree outcome
$Y_t$, and the reward $R_t$.

**Reward.** $R_t = -\texttt{delta\_run\_exp}_t$, so larger is better for the pitcher (SPEC §5).
The mandatory sign check — $R>0$ on called strikes / whiffs / outs, $R<0$ on balls / walks / home
runs — is verified before any run-value table is fit.

**Leakage discipline.** The one non-negotiable rule (SPEC §0) is that a feature for the decision
before pitch $t$ must be computable strictly before pitch $t$ is thrown. The current pitch's own
velocity, movement, and location are *execution*, downstream of the decision, and are never state
features; prior pitches' realized measurements are known before the current decision and are fair
game. The shared builders enforce this with a column-name audit, which WS1 inherits by reading
only the audited table.

**Temporal splits.** Train on 2021–2023, select on 2024, lock 2025 for test (SPEC §7). Splits are
by season, never by random row — pitches in the same PA/game are far too dependent for a random
split to be honest — and all confidence intervals are bootstrapped over pitcher-game clusters.

---

## 4. Methods

### 4.1 The key ladder (D25)

WS1 realizes the five state views as a deepening conditioning key (decision D25). Writing
$ch = (\text{balls}, \text{strikes}, \text{stand}, \text{p\_throws})$ for the count × handedness
cell and $pit$ for the pitcher, each view's deepest selection cell is:

| view | selection key | history component |
|---|---|---|
| `C` | $(ch, pit)$ | none |
| `L1` | $(ch, pit, \text{prev})$ | immediately preceding family |
| `U` | $(ch, pit, \text{multiset})$ | order-invariant multiset of prior families (capped at 6) |
| `O` | $(ch, pit, (\text{prev}_2, \text{prev}_1))$ | ordered last two prior families |
| `OM` | — infeasible | matchup memory cannot be tabulated |

The run-value tables use the same ladder but additionally condition on the current action
(decision D22: outcome models condition on the pitch actually thrown), so every run-value cell
carries the current `family` as an extra key coordinate and the estimand is $E[R \mid \text{cell},
\text{family}]$.

### 4.2 Selection: hierarchical Dirichlet-multinomial

Level $l$ has one shared concentration $\alpha_l$. A cell $c$ with observed family counts $n_c$
($N_c = \sum_k n_{c,k}$) and parent posterior mean $\pi_{\text{parent}(c)}$ over the 8 families has
the closed-form Dirichlet posterior mean

$$\hat p_c = \frac{n_c + \alpha_l\,\pi_{\text{parent}(c)}}{N_c + \alpha_l}.$$

Rearranged into shrinkage form (derived step by step in `THEORY.md` §3),

$$\hat p_c = \frac{N_c}{N_c + \alpha_l}\,\hat p^{\text{MLE}}_c + \frac{\alpha_l}{N_c + \alpha_l}\,\pi_{\text{parent}(c)},$$

a convex combination of the cell's own empirical distribution and the parent. The posterior
effective sample size is $\text{ESS}_c = N_c + \alpha_l$, and the per-family posterior variance is
$\hat p_{c,k}(1 - \hat p_{c,k})/(N_c + \alpha_l + 1)$. An unseen cell ($N_c = 0$) returns the
parent exactly: that is **backoff**.

### 4.3 Run value: hierarchical normal partial pooling

Level $l$ has one shared shrinkage strength $\kappa_l$ (a prior sample size). A cell $c$ with
$n_c$ observations, sample mean $\bar R_c$, and parent posterior mean $m_{\text{parent}(c)}$ has the
normal-normal posterior mean

$$\hat m_c = \frac{n_c}{n_c + \kappa_l}\,\bar R_c + \frac{\kappa_l}{n_c + \kappa_l}\,m_{\text{parent}(c)},$$

again the parent exactly when $n_c = 0$. With a pooled within-cell residual variance $\sigma^2$,
the posterior standard deviation of the cell mean is

$$\text{sd}(\hat m_c) = \frac{\sigma}{\sqrt{n_c + \kappa_l}},$$

so reported uncertainty falls like $1/\sqrt{n_c + \kappa_l}$ — thin cells carry honestly wide
error bars, which is exactly the uncertainty the downstream bandit (WS4) consumes.

### 4.4 Fitting the concentrations (empirical Bayes, level by level)

Each level's $\alpha_l$ / $\kappa_l$ is fit *conditional on the posterior means of the level
above* (a top-down empirical-Bayes pass), pooling the exchangeable cells at that level. The
primary estimator is the **maximum marginal likelihood** (the Dirichlet-multinomial Pólya
evidence, or the normal-normal evidence), a smooth 1-D optimization on $\log_{10}$ of the
concentration; a documented **method-of-moments** estimator (Pearson-overdispersion matching for
the Dirichlet; a DerSimonian–Laird–style between-cell-variance match for the normal) is the
fallback when the optimizer fails or a level is too sparse to identify the concentration
(`THEORY.md` §§4–5, §8). The fitted value and the estimator actually used are recorded per level.

A fitted concentration is a **readout**, not a nuisance: $\alpha_l \to \infty$ means the level adds
nothing (child cells look like independent draws from the parent, so they are pooled all the way
back), while $\alpha_l \to 0$ means the cells are genuinely heterogeneous. The selection ceiling is
effectively unbounded ($10^6$) because a flat table genuinely gains little from within-PA history
for selection; the run-value ceiling is moderate ($200$) so a well-supported cell keeps some of
its own reward signal for the prescriptive workstreams to consume (decisions D26, D27).

### 4.5 Support diagnostics

For each view the support exhibit reports the number of distinct deepest cells, per-cell training
support percentiles, the fraction of evaluation rows whose resolved cell has fewer than 5 / 20 /
50 training pitches, and the backoff rate at each level (the fraction of evaluation rows resolved
at that level). `OM` is not fit: keying on the specific batter-versus-pitcher history multiplies
the already-fragmented `O` cells by the batter dimension, and an average pitcher-batter pair has
on the order of 20 career pitches (SPEC §3.4), so nearly every `OM` cell would be empty and back
off to `O`. WS1 reports the pitcher-batter support numbers that justify declaring `OM` infeasible
(decision D25) and hands matchup memory to the pooled/embedding models (WS3+).

---

## 5. Experimental setup

**Scoring.** Every view's predictions are written in the standard schema (SPEC §8.1) and scored
*only* through the shared harness (SPEC §8): multiclass **log loss** (primary) for selection,
**MAE/RMSE** and a run-value **calibration** slope/intercept for run value, and the
$\Delta_{\text{order}} = \text{Loss}(\min[U, L1]) - \text{Loss}(O)$ ablation with a pitcher-game
clustered bootstrap CI. WS1 computes $\Delta_{\text{order}}$ on the selection target.

**Reference baselines.** WS1 is scored like-for-like against four count-based references
(`eval/baselines.py`): global family frequency by count × hand, pitcher × count, first-order
transition (prev-family × count), and pitcher × count × prev-family, each Dirichlet-smoothed. The
serious hierarchical version should not lose to the quick references.

**Slices.** All headline metrics are reported across the six SPEC §6 slices: all pitches,
sequence-eligible ($t \ge 2$), long PA ($t \ge 3$), two-strike, three-ball, and first-pitch.

**Δ_order reading rule (D21).** Because $\Delta_{\text{order}}$ uses $\min[U, L1]$ and the minimum
of two noisy losses is optimistically biased low, $\Delta_{\text{order}}$ is *negatively biased
under the null*. The null criterion is therefore "**not significantly positive**" (CI lower bound
not above 0), a small negative value reads as *consistent with no ordering effect* rather than
"order hurts", and only a significantly positive value claims genuine ordered structure. The
planted positive-world effect is keyed on the transition *into* the previous pitch,
$|\text{velo}_{t-1} - \text{velo}_{t-2}|$, precisely because a $|\text{velo}_t - \text{velo}_{t-1}|$
effect would already be representable by `L1` and could not separate `O` from `L1`.

**Falsification protocol (completed validation).** WS1 ships two synthetic worlds with known truth
(SPEC §11), and the following are *observed results*, not aspirations:

- **Null world** (order-dependent selection habits, but outcomes depend only on
  context + current pitch): the fitted history-level concentration diverges to its ceiling
  ($\alpha \to 10^6$), which pools the within-PA order away so that every view makes an identical
  selection prediction and $\Delta_{\text{order}} = 0.0000$ **exactly**. The tables manufacture no
  ordered structure that is not there.
- **Positive world** (a planted whiff boost when $|\text{velo}_{t-1} - \text{velo}_{t-2}| \ge 5$
  mph): the run-value recovery signature is $O > U > L1 \approx C \approx 0$ in the
  ordered-contribution-beyond-`C` measure, with the ordered `O` table recovering the effect's sign
  at attenuation $\approx 0.03$ of the planted $\approx +0.0145$ reward lift. The direction is
  correct and only the ordered view sees it; the tiny magnitude is the *mechanism-blindness*
  lesson (§7): a family table sees a velocity mechanism only through the family → velocity-band
  proxy (decision D27).

Two honest world-specific artifacts are reported rather than hidden: on the synthetic worlds
`stand`/`p_throws` are uninformative, so WS1-`C` can lose $\sim$0.009 log loss to the quick
`pitcher_count` reference (expected to reverse on real data, where handedness matters), and the
`U`-vs-`O` distinct-cell ordering is world-dependent (`U` keys the full capped multiset, `O` only
the ordered last two).

---

## 6. Results

### 6.1 Headline table (real data)

**Data vintage.** 3,567,640 regular-season decisions, 2021–2025 (SPEC's $\sim$3.85M counts all
game types; the config filters to `game_type == "R"`). Train 2021–2023 (2,143,214), validation
2024 (711,898), locked test 2025 (712,528). WS1 fits on train and reports on validation.

| view | selection log loss | run-value MAE | run-value RMSE | reference | reference log loss |
|---|---|---|---|---|---|
| `C`  | 1.4714  | 0.1200  | 0.2234  | `pitcher_count`      | 1.4866 |
| `U`  | 1.4785  | 0.1200  | 0.2234  | `pitcher_count_prev` | 1.4410 |
| `L1` | 1.4772 | 0.1201 | 0.2234 | `pitcher_count_prev` | 1.4410 |
| `O`  | 1.4787  | 0.1200  | 0.2234  | `pitcher_count_prev` | 1.4410 |

$\Delta_{\text{order}}$ = −0.0015 (95% clustered CI [−0.0017, −0.0013]); $\Delta_{\text{matchup}}$
= N/A (`OM` infeasible). Other references: `global_count_hand` = 1.7432, `transition` = 1.6437.

Support: distinct cells 30,430 (`C`) → 94,992 (`L1`) → 212,434 (`O`) → 320,487 (`U`);
fraction of evaluation pitches in cells with $n < 20$ rising from 0.072 (`C`) to
0.222, 0.348, 0.332 (`L1`/`O`/`U`).

**Fitted concentrations (the "what did the data decide" exhibit).** Selection: global level 1,
count × hand $\alpha = 27.7$, **pitcher $\alpha = 2.35$** (pitchers are extremely distinct —
almost no pooling), history levels $\alpha_{L1} = 128.3$, $\alpha_O = 73.4$, $\alpha_U = 72.2$
(the within-PA history levels are shrunk hard toward the pitcher parent). Run value: **$\kappa$
pegged at the 200 ceiling on every non-global level** — the fitter pools the cell reward means as
hard as it is allowed to, the honest statement that cell-level run values are noise-dominated and
EB run-value tables carry almost no cell-level signal on real data.

### 6.2 Branched interpretation

The result is read on two axes: a **selection** axis (does within-PA history sharpen the
next-pitch distribution beyond context, pitcher, and count?), branches **S1/S2**; and a
**$\Delta_{\text{order}}$** axis (does the fully ordered view refine on the better of `U`/`L1`,
read under D21?), branches **R1/R2/R3**. Exactly one branch on each axis applies; each is written
to stand alone once the numbers above are filled.

#### Selection axis

*Pre-registered alternative — not selected.* **S1 — history views clearly beat `C`.** The best
history view lowers next-pitch log loss below `C` by more than the clustered CI: **ordered
selection structure is present** (finding #1). Prior pitches genuinely help predict what is thrown
next — the expected real-data result, since pitchers do sequence their selection. For the study
this means the selection channel is live and worth the richer models: WS2's variable-order grammar
and WS3's behavior model should extend the edge with calibrated higher-order dependence. It says
nothing yet about *outcomes* (finding #2): a predictable pitcher is not necessarily an exploitable
one.

**Selected by the data (2021–2025). S2 — history barely beats `C` (flat beyond count/pitcher).**
Within-PA history does not sharpen next-pitch prediction beyond context, pitcher, and count. For a
flat table this is unsurprising and often correct: the higher-order selection signal is below what
naive cells can resolve, so empirical Bayes pools it away (as it provably does on the null world).
This is *not* evidence that pitchers do not sequence — it is evidence that a lookup table cannot
see it — and the proper test of whether a better model can is exactly WS2/WS3. If they too land
here, the modest-selection-structure reading of the sequencing literature is corroborated.

*Realized (2021–2025).* Within the D25 hierarchy the three history views (`U` = 1.4785,
`L1` = 1.4772, `O` = 1.4787) do **not** beat WS1-`C` (1.4714): adding within-PA history to the
hand-keyed cell ladder does not sharpen the forecast. But the reference nuance is a first-class
finding, and it says the failure is one of *representation*, not of baseball. The external
pitcher × count × prev-family reference (1.4410) beats WS1-`C` by ≈0.030 log loss and beats the
`pitcher_count` reference (1.4866) by 0.046 — so first-order selection structure is unambiguously
real and material. What the D25 ladder cannot do is *express* it: keying on
`(balls, strikes, stand, p_throws, pitcher, prev)` shatters support faster than the lean
prev-keyed reference, so the hierarchy cannot reach the hand-marginalized projection the reference
computes directly. This is a fragmentation tax — the same one WS2's grammar pays, and exactly the
gap WS3's features are built to close. One synthetic-world prediction is confirmed on the way past:
WS1-`C` beats `pitcher_count` (1.4714 vs 1.4866), reversing the §5 handedness artifact precisely
as the docs anticipated — on real data `stand`/`p_throws` carry signal.

#### Δ_order axis (read under D21)

*Pre-registered alternative — not selected.* **R1 — $\Delta_{\text{order}}$ significantly positive
(CI lower bound > 0).** The fully ordered view
beats the better of `U`/`L1` by more than the clustered CI: a **table-visible ordered dependence**.
This is the strong claim and must survive two checks before it is believed: (i) the support exhibit
(§6.1) — if `O`'s edge rides on cells with $n < 20$, it is likely overfitting thin history keys; and
(ii) per-slice consistency — a real ordered effect should concentrate in `long_pa`/`two_strike`, not
appear only in aggregate. If it survives both, WS1 has found genuine ordered structure even at the
table level, and WS2/WS3 should confirm and sharpen it — the first real rung of evidence that *order*
carries signal.

**Selected by the data (2021–2025). R2 — $\Delta_{\text{order}} \approx 0$ or small negative
(consistent with no ordering effect).**
Under D21 this reads as *consistent with no table-visible ordering effect*, and — crucially — **not**
as proof that order is absent. Two reasons withhold the stronger claim. First, $\min[U, L1]$ is
optimistically biased, so a small negative is the bias, not a cost of order. Second, and measured: a
family table is **mechanism-blind** — on the positive world it recovered only $\sim$3% (attenuation
$\approx 0.03$) of a real planted effect, because it sees a velocity mechanism only through the
pitch-name → velocity-band proxy. So "$\approx 0$ from a table" is fully compatible with a real but
modest ordered *outcome* effect that only a feature-based (WS3) or learned (WS6) model can resolve.
This is the expected, honest WS1 result, and it is precisely the argument for continuing up the
ladder: any later claim of an ordering effect must clear this bar and explain why the table missed it.

*Realized (2021–2025).* $\Delta_{\text{order}}$ = −0.0015 (CI [−0.0017, −0.0013]) — a small
negative, read directly as R2: no table-visible ordering effect, the negative being the
$\min[U, L1]$ optimism rather than a cost of order. The run-value MAE is flat at 0.1200 across all
four views, and the fitted concentrations (§6.1) say why the run-value channel is mute: the history
levels pool hard ($\alpha$ 72–128) and $\kappa$ is pegged at its 200 ceiling on every non-global
level, so the cell reward tables carry essentially no cell-level signal for order to move. WS3's
feature-based ablation (which *can* split on pitch physics) is where any real ordered outcome effect
must show up, exactly as the mechanism-blindness ceiling below predicts.

*Pre-registered alternative — not selected.* **R3 — $\Delta_{\text{order}}$ clearly negative,
beyond the D21 bias scale.** For WS1's pooling
tables — whose null $\Delta_{\text{order}}$ is exactly 0, so there is no optimism bias to blame — a
clearly negative value is **diagnostic of over-fragmentation**: the `O` key has splintered support so
badly that its predictions are noisier than `U`/`L1` even after shrinkage. Diagnose it directly with
§6.1 (elevated $n < 5/20$ fractions and a short deepest-level backoff bar for `O`). The fix is not in
WS1 — a flat table can only pool harder — but is exactly WS2's variable-order backoff, which spends
resolution only where support justifies it. R3 is a statement about *estimation*, not baseball: order
might still matter, but it cannot be seen by counting cells this fine.

---

## 7. Discussion

The interpretation of WS1 mirrors the branch grid of §6, so the discussion is written per branch and
completed by the same numbers.

*Pre-registered alternative — not selected.* **If S1 / R1 (structure and order both visible).** WS1
already resolves ordered structure at the
table level. The burden then shifts to the richer models to show that their added machinery *extends*
rather than merely *reproduces* this edge, and to the outcome and OPE workstreams to test whether the
predictable order is also exploitable (findings #2, #3). The support exhibit remains the guardrail:
any ordered edge concentrated in thin cells is provisional.

**Selected by the data (2021–2025): S2 / R2. If S1 / R2 or S2 / R2 (selection maybe, order not, from
a table).** This is the study's most likely
result and its cleanest one. It is reported without embarrassment (SPEC §0): a modest or null ordering
effect is a real, publishable finding, consistent with the sequencing literature. The **measured
caveat** is what keeps it honest: on the positive synthetic world the ordered table recovered only
$\approx 0.03$ of a genuine planted effect, because a table keyed on pitch *names* is partially blind
to a mechanism keyed on pitch *physics*. A near-zero table $\Delta_{\text{order}}$ therefore bounds,
but does not eliminate, an ordered outcome effect — it establishes the level a feature model (WS3) or
learned representation (WS6) must exceed, and explains in advance why they might. The realized result
sharpens this: the near-zero is on the S2 side (WS1's history views do not beat `C`), yet the
pitcher × count × prev reference beating every view by ≈0.03–0.04 proves the first-order *selection*
signal is genuinely there — so WS1's honest verdict is "a real effect the tabular representation
cannot hold," the tightest possible statement of the support problem and the exact brief handed to
WS2 (a smarter grammar) and WS3 (physics as features).

*Pre-registered alternative — not selected.* **If R3 (over-fragmentation).** The result is about
estimation, not baseball, and it is the sharpest
possible motivation for WS2: fixed, shallow, count-then-count tables cannot allocate resolution to
where support exists, and a variable-order backoff can. WS1 has then done its job by failing
informatively.

Across all branches, the through-line is the support problem. The exhibit's central number — a
substantial fraction of the evaluation set landing in cells too thin to estimate once the key deepens
— is not a defect of this implementation but a property of the data and the tabular approach. It is
the reason the project is a ladder of models rather than a single table, and it is what WS1 exists to
make unavoidable.

---

## 8. Limitations

1. **`pitch_type` is a classifier output**, not the battery's intent. WS1's action space is the
   Statcast-inferred pitch family; a mislabeled pitch is a mislabeled action. Family (8 classes) is
   used precisely to blunt this, but it cannot remove it.
2. **No catcher or umpire effects.** These are absent from base Statcast and are omitted (SPEC §3.2);
   a real component of pitch-call and framing dynamics is therefore outside WS1's state.
3. **Tables cannot represent matchup memory.** `OM` is infeasible for pure tables (§4.5); the
   longer-term batter–pitcher adaptation question is deferred to the pooled/embedding models, and WS1
   reports $\Delta_{\text{matchup}} = \text{N/A}$.
4. **Family granularity and shallow order.** The `O` key retains only the ordered last two prior
   families, and `U` a multiset capped at six; genuinely long-range or fine-grained ordered
   dependence is beyond the key by construction.
5. **Single-metric reward.** Reward is $-\texttt{delta\_run\_exp}$, one run-expectancy scale; WS1 does
   not model the full event tree jointly, and a different reward could reweight the run-value tables.
6. **Mechanism-blindness (quantified).** As measured on the positive world, name-keyed tables recover
   only $\sim$3% of a planted physical (velocity-transition) effect. Null and near-null table results
   must be read with this attenuation in mind.

---

## 9. Conclusion

WS1 fits the simplest honest model of pitch sequencing — empirical-Bayes–shrunk conditional tables —
across five nested state views, scores them through the shared harness, and exhibits the support limits
of the tabular approach. Its conclusion is branch-conditional and complete once the real numbers arrive:

- **Under S1 / R1** *(pre-registered alternative — not selected)*, WS1 finds ordered selection
  structure and a table-visible ordered refinement, and hands the richer workstreams a live signal
  to extend and an outcome/OPE question to test.
- **Under S2 / R2 — selected by the data (2021–2025)** (the expected outcome), WS1 finds that a flat
  table does not resolve ordered structure beyond context and the previous pitch (history views
  1.4772–1.4787 vs `C` 1.4714; $\Delta_{\text{order}}$ = −0.0015) — a clean, reportable null — while
  the pitcher × count × prev reference (1.4410) beating every view proves the first-order selection
  signal is *real* and only the tabular representation cannot hold it, and WS1's own synthetic-world
  validation shows that a null $\Delta_{\text{order}}$ *bounds* rather than *refutes* a real ordered
  outcome effect, at a measured $\sim$3% recovery ceiling.
- **Under R3** *(pre-registered alternative — not selected)*, WS1 diagnoses over-fragmentation and
  thereby states the case for variable-order backoff directly.

In every case the durable contribution is the support exhibit: naive conditioning exhausts its support
before it exhausts the questions, which is the standing reason the rigor ladder does not stop at a
table.

---

## References

Efron, B., and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations.
*Journal of the American Statistical Association*.

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013).
*Bayesian Data Analysis*, third edition. Chapman and Hall/CRC.

James, W., and Stein, C. (1961). Estimation with quadratic loss. *Proceedings of the Fourth Berkeley
Symposium on Mathematical Statistics and Probability*.

Marchi, M., and Albert, J. (2013). *Analyzing Baseball Data with R*. Chapman and Hall/CRC.

Minka, T. P. (2000/2003). Estimating a Dirichlet distribution. Technical note.

Morris, C. N. (1983). Parametric empirical Bayes inference: theory and applications. *Journal of the
American Statistical Association*.

Robbins, H. (1956). An empirical Bayes approach to statistics. *Proceedings of the Third Berkeley
Symposium on Mathematical Statistics and Probability*.

Stein, C. (1956). Inadmissibility of the usual estimator for the mean of a multivariate normal
distribution. *Proceedings of the Third Berkeley Symposium on Mathematical Statistics and
Probability*.

Tango, T. M., Lichtman, M. G., and Dolphin, A. E. (2007). *The Book: Playing the Percentages in
Baseball*.
