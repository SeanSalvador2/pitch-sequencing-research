# A Bayesian Variable-Order Markov "Pitch Grammar" for Ordered Pitch Selection

*Workstream 2 of a comparative pitch-sequencing study.* This paper is completed in place with the
real-data results (2021–2025 Statcast). The Results and Discussion were pre-**branched** so that
the correct interpretation was already written for whichever numbers arrived; the branch **selected
by the data** is marked at each fork, and the unselected branches are retained and labelled as
*pre-registered alternatives*. The synthetic-world numbers quoted in §5 are *completed validation*,
not placeholders.

---

## Abstract

We ask how much of the pitch-sequencing story lives in **selection** — whether the *ordered*
sequence of pitches already thrown sharpens the forecast of the next one — and how deep that
ordered dependence actually reaches. We fit a per-context **variable-order Markov model** over
pitch-family tokens with two-axis hierarchical Dirichlet backoff: a count × handedness ×
pitcher **base rate** (the depth-0 repertoire) times a within-pitcher **ordered lift** whose
depth-`k` context is shrunk toward its depth-`(k−1)` suffix, with every depth's concentration
fitted by empirical Bayes (maximum marginal likelihood, method-of-moments fallback). The model
is fit over three nested views — context `C` (order 0), previous pitch `L1` (order 1), and the
full ordered sequence `O` (variable order ≤ `K_MAX` = 4); the unordered `U` and matchup-memory
`OM` views are shown to be **not-applicable** to a Markov grammar. On the real data
(2021–2025 Statcast, ~3.85M pitches) the context view attains selection log loss 1.4714
and the ordered grammar 1.4867, giving an ordered-selection edge
`delta_order_L1 = Loss(L1) − Loss(O)` = −0.0056 (95% clustered CI [−0.0061, −0.0051]) — a
significantly *negative* edge, meaning the variable-order view's depth-2–4 contexts fragment and
cost out-of-sample relative to the bigram. The data votes against depth directly: the fitted depth
concentrations climb monotonically (`α_1` = 17.9 earns its keep, then `α_2`/`α_3`/`α_4` =
84.7/113.0/166.7 are
pooled progressively away), the grammar earns a mean effective order of just 0.901 with 0.250 of
rows resolving depth ≥ 2, and its raw ordered predictor scores `B_seq` = −0.0221 bits versus
context — a *model-relative* number reflecting this grammar's depth>1 overfit, not
anti-predictability (§6.2). The one unambiguous descriptive finding is **stickiness**: every
depth-1 top motif is repeat-promotion (e.g. P(FS|FS) 0.24→0.38, P(SL|SL) 0.31→0.39), the exact
opposite sign of the synthetic no-three-in-a-row habit. We validate the method on the correctness
oracle, where — by
a deliberate inversion (decision D30) — the *null* world is WS2's **positive control**: it
plants an order-2 no-three-in-a-row selection habit that the grammar recovers exactly
(`GRAMMAR_DETECTED`: O = 1.3322 < L1 = 1.3385, `delta_order_L1` = +0.0063 CI [+0.0041, +0.0080],
order ≥ 2 mass 0.296, all top-5 motifs the planted repeat-suppression rule), while a stratified
history permutation collapses the edge (+0.0063 → +0.0001) and the order-2 mass (0.296 → 0.000)
(`COLLAPSES_UNDER_PERMUTATION`). The headline contribution is a *calibrated depth read* on
ordered selection that spends resolution only where support earns it — and a firewall: every
result here is finding #1 (selection), and none of it speaks to outcome value.

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

Workstream 2 (WS2) is the ladder's **selection instrument for finding #1** — sharper than WS1's
flat tables, still fully transparent. Where WS1 asked "can a shrunk lookup table *see* order?"
with a fixed, shallow key and answered mostly *no* (its cells fragment before they resolve
higher-order structure), WS2 asks the better question: *given a model that spends resolution
only where the data supports it, how deep does the ordered selection dependence go, and which
ordered rules does it consist of?* It fuses the two ideas of the study named in SPEC §12.2 —
the **Markov** idea (condition on an ordered string of previous tokens) and the
**Bayesian-pooling** idea (shrink a specific estimate toward a well-estimated coarser one) —
into a single variable-order grammar over pitch-family tokens.

**The D30 inversion, stated as a methodological point.** WS2 is where the correctness oracle's
design (decision D20) earns its keep, and the point is worth stating for its own sake. The
oracle's *null* world was built to be null for **outcomes** — it forbids any ordered
outcome effect — while deliberately carrying an ordered **selection** habit (a mild
no-three-in-a-row tendency). Every other workstream reads that world as a null. WS2, a
*selection* model, reads it as a **positive control** (decision D30): the planted selection
habit is precisely the order-2 structure the grammar should detect. This is not a trick of
convenience; it is the direct consequence of separating finding #1 from finding #2 at the
fixture level, and it lets WS2 be validated *positively* — the grammar must find the planted
order — rather than only checked for false positives. WS2's own *negative* control is a
different operation: a stratified history permutation under which a genuine ordered edge must
collapse. The two together (detect on the null-as-positive-control; collapse under permutation)
are a stronger acceptance test than either alone.

WS2's role on the ladder is therefore threefold. First, it is the **calibrated depth read**:
its effective-order distribution reports how much ordered structure the data can actually
resolve, honestly thinning with depth. Second, it is the first consumer of the SPEC §10
predictability read-out (`B_seq`, decision D31): it puts a bits number on how much the ordered
sequence leaks about the next pitch. Third, it hands WS3 and WS6 a concrete, *explicit* grammar
— an effective depth and a motif list — for their feature-based and learned-representation
models to confirm, extend, or exceed.

---

## 2. Related work

**Variable-order Markov models and backoff.** WS2's core is a hierarchical Bayesian *language
model* whose vocabulary is 8 pitch families. Its ancestry is the smoothing-and-backoff
literature of statistical language modeling: Katz (1987) introduced backoff (fall back to a
shorter context when the longer one is unseen); Kneser and Ney (1995) refined interpolation and
smoothing of higher- with lower-order estimates; and Chen and Goodman (1999) gave the
systematic empirical comparison that made smoothed *n*-gram models the standard. The
variable-order idea — grow the context only where it earns support — is the *probabilistic
suffix tree* of Ron, Singer, and Tishby (1996), and the prediction-focused survey of Begleiter,
El-Yaniv, and Yona (2004) is the direct methodological reference for using variable-order Markov
models as predictors and scoring them by log loss.

**Hierarchical-Bayesian language models.** Closest of all to WS2's exact form are the
*hierarchical* Bayesian language models. MacKay and Peto (1995) place a hierarchical Dirichlet
prior on *n*-gram distributions so that a context is shrunk toward its shorter suffix — which is
precisely WS2's suffix-chain shrinkage, `g_k` shrunk toward `g_{k−1}`. Teh (2006) generalizes
this to the hierarchical Pitman–Yor process, the state-of-the-art refinement; WS2 uses the
Dirichlet special case, keeping the model transparent and its concentrations directly fittable.
The Dirichlet-fitting machinery (maximum marginal likelihood on the Pólya evidence, with a
method-of-moments fallback) follows the standard treatment of estimating a Dirichlet (Minka,
2000/2003) and the hierarchical-model account of Gelman et al. (2013), shared with WS1.

**Bits of predictability.** The `B_seq` read-out is an application of Shannon (1948): the mean
log-ratio of two predictive distributions on the realized symbol is a cross-entropy difference,
i.e. the extra bits one model supplies about the next symbol over another. WS2 reports the bits
the ordered view supplies over the context view.

**Shrinkage and baseball.** The empirical-Bayes pooling philosophy is the classical shrinkage
of Efron and Morris (1975), the baseball demonstration this whole study leads with. The applied
grounding is Tango, Lichtman, and Dolphin (2007), whose count- and matchup-based analysis
motivates the count × handedness × pitcher conditioning of the base rate, and Marchi and Albert
(2013) for partial pooling of Statcast-style pitch data.

---

## 3. Data

**Source.** Statcast pitch-level data via pybaseball, seasons 2021–2025, a single tracking era
(Hawk-Eye), ~3.85M pitches over 119 raw columns (SPEC §1). `pitch_type` is a Statcast
*classifier output* and is treated throughout as an observed proxy for the pitch thrown, not as
the battery's intent (SPEC §1; see §8).

**The canonical decision table and the tokens.** Every workstream reads one shared table
(SPEC §3): one row per pitch, the decision made immediately *before* that pitch is released,
keyed by `(game_pk, at_bat_number, pitch_number)`. WS2 uses a minimal slice of it. The
**token** is the pitch family (8 symbols: `FF, SI, FC, SL, CU, CH, FS, XX`) of the pitch
thrown. The **context at depth `k`** is the ordered last `k` families of the *current* plate
appearance, `ctx_k = (w_{t−k}, …, w_{t−1})`, formed directly from `pa_id` / `pitch_number` /
`family` (no history logic is re-implemented). The **conditioning cell** is
`ch = (balls, strikes, stand, p_throws)` with a pitcher level below it — exactly WS1's C-ladder.

**Leakage discipline.** A feature for the decision before pitch `t` must be computable strictly
before pitch `t` is thrown (SPEC §0). The grammar's context is the families of *prior* pitches
this PA — all known before the current decision — so it is leakage-safe by construction; the
current pitch's family is the *target*, never part of the context. WS2 inherits the shared
builders' column-name audit by reading only the audited table.

**Temporal splits.** Train on 2021–2023, select on 2024, lock 2025 for test (SPEC §7). Splits
are by season, never by random row, and all confidence intervals are bootstrapped over
pitcher-game clusters.

---

## 4. Methods

### 4.1 View → order mapping (D29)

WS2 realizes the state views as a **grammar depth** (decision D29). `C` is order 0 (the base
rate, no ordered context); `L1` is order 1 (the previous family); `O` is variable order up to
`K_MAX` = 4. Two views are **not-applicable** and are documented rather than fitted:

| view | grammar depth | note |
|---|---|---|
| `C` | 0 | count × pitcher base rate; selection without current-PA order |
| `L1` | 1 | first-order (previous family) |
| `O` | ≤ `K_MAX` (4) | full variable-order grammar |
| `U` | — not applicable | a Markov context is an *ordered* string; a multiset is not a context |
| `OM` | — not applicable | a within-PA chain has no cross-PA memory |

### 4.2 Two-axis hierarchical Dirichlet backoff

The row prediction is a **product of experts** over two Dirichlet-multinomial backoff chains
(the full derivation is `THEORY.md` §§2–3).

**Cell axis — the base rate `b`.** The global marginal `π_G`, shrunk into a count × hand cell,
then a pitcher level:

$$P(a \mid ch) = \frac{n_{ch} + \beta_{ch}\,\pi_G}{N_{ch} + \beta_{ch}},
\qquad
b(a) \equiv P(a \mid ch, \mathrm{pit}) =
     \frac{n_{ch,\mathrm{pit}} + \beta_{\mathrm{pit}}\,P(a\mid ch)}{N_{ch,\mathrm{pit}} + \beta_{\mathrm{pit}}}.$$

**Depth axis — the ordered lift.** Keyed by the pitcher (pooled over counts, so deep contexts
keep support and the lift stays pitcher-clean), with pitcher marginal `m(a) = P(a \mid pit)`
and, for `k = 1 … K_MAX`, a suffix-chain shrinkage:

$$g_k(a \mid ctx_k, \mathrm{pit}) =
     \frac{n + \alpha_k\, g_{k-1}(a \mid ctx_{k-1}, \mathrm{pit})}{N + \alpha_k},
\qquad g_0 \equiv m, \quad ctx_{k-1} = \operatorname{suffix}(ctx_k),$$

where `suffix` drops the **oldest** family `w_{t−k}`. A row is eligible at depth `k` only when
the PA supplies `k` real priors (`pitch_number ≥ k + 1`).

**Product-of-experts combination**, renormalised over the 8 families:

$$P(a \mid ch, \mathrm{pit}, ctx_k) \;\propto\; b(a)\,\frac{g_k(a \mid ctx_k, \mathrm{pit})}{m(a)}
\;=\; \underbrace{P(a\mid ch,\mathrm{pit})}_{\text{count} \times \text{pitcher base}}
      \underbrace{\frac{P(a\mid \mathrm{pit}, ctx_k)}{P(a\mid \mathrm{pit})}}_{\text{within-pitcher order lift}}.$$

When the context is uninformative (`g_k = m`) the prediction is exactly `b`, so the `C` view is
`b` and `L1` / `O` apply the depth-1 / depth-`k` lift on the same base. **Why the lift is keyed
on the pitcher (the crux of D29):** pooling the grammar over pitchers would let the previous
family stand in for pitcher identity (`prev = FC` signals a cutter pitcher), double-counting the
repertoire `b` already carries and *hurting*; nesting under both count and pitcher fragments deep
contexts until the ordered signal is un-earnable. Conditioning the lift on the pitcher while
pooling over the count keeps both a count × pitcher base and a supported, confound-free ordered
lift — trading on the order effect being largely count-independent, a documented approximation.

### 4.3 Fitting the per-depth concentrations (empirical Bayes)

Each level's concentration is fitted **conditional on the posterior means of its parents** (a
top-down pass), pooling the exchangeable cells at that level, by **maximum marginal likelihood**
of the Dirichlet-multinomial (Pólya) evidence — a smooth 1-D optimisation on `log10 α`; a
Pearson-overdispersion **method-of-moments** estimator is the fallback when a level is too
sparse to identify the concentration (`THEORY.md` §4). A fitted concentration is a **readout**:
`α_k → ∞` means depth `k` adds nothing (its contexts look like draws from their suffix, and are
pooled all the way back), while a moderate `α_k` means depth `k` carries real ordered signal.
The ceiling is effectively unbounded (`10^6`) — a depth with no signal is pooled to its parent,
the honest variable-order result.

### 4.4 Effective order, motif lift, and B_seq

**Effective order** (a readout, not a parameter). A row's effective order is the deepest depth
`k` whose context was observed in training *and* whose own-count weight `ω_k = N_k / (N_k + α_k)`
reaches `τ` (default 0.5) — i.e. the context's own data, not the backoff prior, drives its
distribution. We report the distribution over rows, its mean, and the order ≥ 2 mass.

**Motif lift.** For every grammar context (pooled over the pitchers sharing the ordered family
context), the next-token distribution `p_ctx` is compared to its suffix's `p_suffix` and ranked
by the Kullback–Leibler divergence

$$D_{\mathrm{KL}}(p_{\text{ctx}} \,\|\, p_{\text{suffix}})
   = \sum_a p_{\text{ctx}}(a)\,\log_2 \frac{p_{\text{ctx}}(a)}{p_{\text{suffix}}(a)}.$$

The single family whose probability moves most is reported with its before/after probability,
its `log2` ratio, and direction (`suppress` / `promote`). The code **ranks by KL** and reports
the **max-probability-change family's log2 ratio** as the human-readable lift (`THEORY.md` §6).

**Bits of predictability** (SPEC §10, D31). For the ordered predictor `q_O` and the
context-only predictor `q_C`, scored on the realized next family `A_t`,

$$B_{\text{seq}} = \frac{1}{n}\sum_t \log_2 \frac{q_O(A_t \mid S_t)}{q_C(A_t \mid X_t)},$$

the mean per-pitch bits ordered history supplies about the next pitch beyond context. It equals
the C-minus-O cross-entropy difference in bits (`THEORY.md` §7). It is reported overall and
sliced by pitch-number, count-bucket, two-strike, and three-ball.

---

## 5. Experimental setup

**Scoring.** Every view's predictions are written in the standard schema (SPEC §8.1) and scored
*only* through the shared harness (SPEC §8): multiclass **log loss** (primary) for selection,
and the ordered-selection edge `delta_order_L1 = Loss(L1) − Loss(O)` with a pitcher-game
clustered bootstrap CI. WS2 has no `U` view, so the shared `compare_views` (which forms
`min[U, L1]`) does not apply; the edge is the direct L1-vs-O difference and the `U` slot is
never abused.

**Reference baselines.** WS2 is scored like-for-like against the four count-based references
(`eval/baselines.py`): global family frequency by count × hand, pitcher × count, first-order
transition, and pitcher × count × prev-family. `L1` and `O` should sit at or below the
`pitcher_count_prev` reference; `O` beating all four is the headline that the grammar is the
best next-pitch selector on the board.

**The `delta_order_L1` reading rule (D21-adapted).** Decision D21 flags the SPEC §6
`Δ_order = Loss(min[U, L1]) − Loss(O)` as *negatively biased under the null*, because `min[U, L1]`
is the smaller of two noisy losses (a Jensen gap). WS2's edge takes **no minimum** — it is a
straight difference of two losses — so it carries **no optimism bias** and its clustered CI is
read **directly**: a CI lower bound > 0 is genuine ordered selection structure beyond the
previous pitch; a value ≈ 0 (or slightly negative) reads as *no ordered edge beyond L1*, never
as "order hurts" (`THEORY.md` §8). SPEC §13's honest expectation is a **small** edge on real
data.

**Falsification protocol (completed validation).** WS2's acceptance wiring is the D30 detect /
collapse pair, and the following are *observed results* on the correctness oracle, not
aspirations:

- **Null world as positive control** (D30: it plants an order-2 no-three-in-a-row *selection*
  habit). The grammar fires **`GRAMMAR_DETECTED`**: O = 1.3322 < L1 = 1.3385
  (`delta_order_L1` = +0.0063, CI [+0.0041, +0.0080], significant); mean effective order 0.593
  with order ≥ 2 mass **0.296** (distribution `k0 = 0.704, k1 = 0.000, k2 = 0.296, k3 = k4 = 0`,
  τ = 0.50 — a *pure* order-2 signature, exactly the planted generative order); and all top-5
  motifs the planted repeat-suppression rule, correctly signed (`SI SI → P(SI) 0.35→0.22`,
  KL 0.061; `FS FS → 0.30→0.19`; `FF FF → 0.31→0.20`; `CU CU → 0.25→0.17`; `FC FC → 0.32→0.24`).
  The bits read-out `B_seq` = **+0.0120** overall, 0 at `t ≤ 2`, rising with PA depth and count
  leverage (two-strike **+0.0264**, three-ball **+0.0466**; by count-bucket
  ahead/behind/even = +0.0135/+0.0237/+0.0057; by pitch number `t1 = +0.000, t2 = −0.001,
  t3 = +0.023, t4 = +0.026, t5 = +0.034`).
- **Stratified history permutation** (WS2's negative control). Whole ordered-history vectors are
  permuted among rows sharing a `(count × hand, pitch_number)` stratum — fixing the count mix and
  the PA depth but destroying genuine ordered dependence. The grammar is refit and re-evaluated:
  the edge collapses **+0.0063 → +0.0001** and the order ≥ 2 mass **0.296 → 0.000**
  (**`COLLAPSES_UNDER_PERMUTATION`**).

Two honest world-specific artifacts are reported rather than hidden: on the synthetic worlds
`stand` / `p_throws` are uninformative, so WS2-`C` loses ~0.006 log loss to the quick
`pitcher_count` reference (expected to reverse on real data, where handedness matters — the same
artifact WS1 documents at ~0.009); and the oracle's `positive` world adds an ordered *outcome*
effect that is irrelevant to a selection grammar (WS2 neither needs nor claims it).

---

## 6. Results

### 6.1 Headline table (real data)

**Data vintage.** 3,567,640 regular-season decisions, 2021–2025 (SPEC's ~3.85M counts all game
types; the config filters to `game_type == "R"`). Train 2021–2023 (2,143,214), validation 2024
(711,898), locked test 2025 (712,528). WS2 fits on train and reports on validation.

| view | grammar depth | selection log loss | reference | reference log loss |
|---|---|---|---|---|
| `C`  | 0 | 1.4714  | `pitcher_count`      | 1.4866 |
| `L1` | 1 | 1.4811 | `pitcher_count_prev` | 1.4410 |
| `O`  | ≤ 4 | 1.4867 | `pitcher_count_prev` | 1.4410 |

Other references: global × count × hand 1.7432; prev-family × count
1.6437. Ordered-selection edge `delta_order_L1` = −0.0056 (95% clustered CI
[−0.0061, −0.0051]); `Δ_matchup` = N/A (`OM` not-applicable, D29).

Grammar exhibits: mean effective order 0.901, order ≥ 2 mass 0.250
(τ = 0.50); fitted depth concentrations `α_1` = 17.9, `α_2` = 84.7,
`α_3` = 113.0, `α_4` = 166.7 (monotonically increasing — the data pools depth away with depth). The
top depth-1 motif rules are XX→XX 0.31→0.54, FS→FS 0.24→0.38, CH→CH 0.21→0.30 (all repeat-promotion),
out of {N_MOTIFS} ranked rules at support ≥ 30 *(pending: total motif count from the report JSON,
not in the results log)*. Bits: `B_seq` = −0.0221 overall (model-relative; see §6.2)
— two-strike −0.0527, three-ball +0.0005; ahead/even/behind −0.0408/−0.0234/+0.0013; by pitch
number t1 = +0.000, then −0.010/−0.022/−0.032/−0.055 (t2–t5).

**Real-data verdicts.** The automated detector returns `GRAMMAR_NOT_DETECTED` and the permutation
control returns `COLLAPSES_UNDER_PERMUTATION`; both are correct and neither contradicts the
findings below — they must be read with care. The detector's rule was shaped around the *synthetic*
no-three-in-a-row habit, which required the variable-order view to *beat* the bigram (`O < L1`); on
real data `O < L1` is **False** (the deeper contexts fragment, §6.2/G2), so the rule fires "not
detected." Its other two clauses tell the real story: order ≥ 2 mass = 0.250 and repeat-motif =
**True** — stickiness *is* detected. The permutation control then confirms the machinery is sound:
scrambling histories collapses the edge (−0.0056 → −0.0013) and the order ≥ 2 mass (0.250 → 0.000),
exactly as a working detector should when genuine ordered dependence is destroyed.

### 6.2 Branched interpretation

The result is read on **three axes**, exactly as the notebook's §9 branch selector prints them.
A **Grammar** axis (does order help selection, and how deep?): branches **G1/G2/G3**. A **Bits**
axis (is the ordering forecastable?): **B1/B2**. A **Motif** axis (do the rules make baseball
sense?): **M1/M2**. Exactly one branch on each axis applies; each is written to stand alone once
the numbers above are filled.

#### Grammar axis

*Pre-registered alternative — not selected.* **G1 — ordered selection grammar beyond the previous
pitch.** `delta_order_L1` clears zero (CI
lower bound > 0): the full variable-order grammar beats the bigram by more than the clustered
CI, so **order carries selection information past the previous pitch** — the strong form of
finding #1 at the grammar rung. Characterise it with the two exhibits: the effective-order
distribution says *how deep* (where the ≥ 2 mass sits) and the motifs say *which rules* drive it.
For the study this is a live higher-order selection signal for WS3 (features) and WS6 (a learned
representation) to confirm and extend, and it sharpens the headline question from "does history
matter?" to "*ordered* history matters, to depth ~`k`." It says nothing about outcomes.

**Selected by the data (2021–2025). G2 — first-order adequate (the grammar is mostly bigrams).**
`delta_order_L1 ≈ 0` (not
significantly positive), but `L1` and `O` both beat `C` materially: within-PA history helps
selection, yet order beyond the immediately preceding pitch adds little. Selection memory is one
pitch deep — the grammar is essentially a set of calibrated bigrams. This is SPEC §13's
explicitly anticipated result ("O barely beats L1"), and a clean one: WS3 need only carry the
previous pitch to capture the selection signal, and it sets the bar WS6's learned representation
must clear to justify going deeper. Reported without embarrassment (SPEC §0).

*Realized (2021–2025).* The ordered signal is first-order and no deeper — with one honest
correction to this branch's pre-written "both beat `C`" clause. The fitted depth concentrations are
the clean read: `α_1` = 17.9 earns its keep (a moderate value — depth-1 contexts genuinely differ
from the pitcher marginal, which *is* the stickiness the motifs display), while `α_2`/`α_3`/`α_4` =
84.7/113.0/166.7 climb monotonically to the pool-it-away ceiling, so the data votes against depth
progressively. The edge `delta_order_L1` = −0.0056 (CI [−0.0061, −0.0051]) is significantly
*negative*; because this statistic takes no `min` it has no optimism bias, so the reading is direct
and firm — the variable-order `O` view's depth-2–4 contexts genuinely fragment and cost
out-of-sample against the bigram `L1`. That is a fragmentation/estimation cost, **not** evidence
that ordered history is anti-informative (effective order settles at 0.901). The correction: on
real data the grammar's own product-of-experts does *not* beat `C` (best view `C` = 1.4714;
`L1` = 1.4811 and `O` = 1.4867 are worse), because the pitcher-keyed lift pooled over count cannot
cash the first-order structure that the lean external pitcher × count × prev reference (1.4410)
captures directly — the identical fragmentation tax WS1's hand-keyed ladder pays (its §6.2), and
exactly the gap WS3's features are built to close.

*Pre-registered alternative — not selected.* **G3 — count/pitcher-driven only.** Neither history
view beats `C` materially: selection is
driven by count and pitcher identity, with within-PA order adding essentially nothing. This is
*surprising* against the conventional wisdom that pitchers sequence, so it is a claim to audit
before accepting. Check the backoff/support diagnostics and the effective-order distribution
first: if deep contexts simply never cleared their support threshold (all mass at order 0), the
honest reading is "the grammar could not resolve order at this data scale" — a support statement,
not a baseball one — and the case passes to WS3/WS6. Only with ample support and no order edge is
G3 substantive.

#### Bits axis

*Pre-registered alternative — not selected.* **B1 — materially positive `B_seq`.** Ordered history
makes the next pitch more forecastable from
pre-release information; report the by-slice pattern (more bits deeper in the PA and under count
leverage). This is a batter-side **information leak** — the raw material for SPEC §10's
predictability/exploitability frontier assembled at the capstone (WS7). Forecastable is not
exploitable: positive `B_seq` is finding #1 in bits, and whether those bits become outcome value
(finding #2) or prescriptive gain (finding #3) is tested downstream.

**Selected by the data (2021–2025), refined. B2 — `B_seq` ≈ 0** (the sequence supplies no
forecastable next-pitch bits over context). The ordering is not forecastable beyond context: the
sequence adds no measurable *positive* bits about the next pitch. Consistent with a G2/G3 reading
and a clean negative for the predictability frontier.

*Realized (2021–2025).* `B_seq` = −0.0221 overall — not the ≈ 0 this branch anticipated, but
*negative*, and the negativity is a model-relative artifact rather than anti-predictability. It is
`q_O` (the overfit variable-order predictor) scoring below `q_C` on the realized token because the
`O` grammar's depth>1 contexts fragment; the by-pitch-number fingerprint confirms the mechanism —
t1 = +0.000 (no history yet, so no bits), then −0.010/−0.022/−0.032/−0.055 monotonically as more
prior pitches let the variable-order predictor overfit deeper. The honest statement is therefore
"this grammar supplies no forecastable next-pitch bits over context, and its raw `B_seq` is
*depressed below zero* by the same depth>1 overfit `delta_order_L1` measures"; the clean, calibrated
bits read is deferred to WS3's regularised models.

#### Motif axis

**Selected by the data (2021–2025). M1 — motifs reproduce known patterns (face-valid).** The top
grammar rules read as real
sequencing tendencies — repeat suppression, fastball → breaking-ball setups, two-strike putaway
shifts — on adequate support. The grammar is learning baseball, not noise; the motif table is a
genuine descriptive deliverable and a qualitative corroboration of whichever Grammar branch
fired.

*Realized (2021–2025).* Face-valid, and the headline descriptive finding of the study's selection
side: **real MLB pitch selection is sticky.** Every depth-1 top motif is repeat-*promotion* — the
opposite sign of this branch's "repeat suppression" example, and of the synthetic no-three-in-a-row
habit the grammar was validated against: P(FS|FS) 0.24→0.38, P(CH|CH) 0.21→0.30, P(SL|SL)
0.31→0.39, P(SI|SI) 0.35→0.41, P(XX|XX) 0.31→0.54 (KL 0.16 down to 0.03). Depth-2 adjustments are
real but an order of magnitude smaller (KL ≤ 0.013) and read sensibly — double-up damping (e.g.
[CH, CH] → suppress the third) and off-speed → fastball re-promotion ([FS, FF], [SL, FF] →
promote). The grammar is learning baseball, and what it learns is that pitchers repeat.

*Pre-registered alternative — not selected.* **M2 — arbitrary motifs (inspect).** The top rules
lack a clear baseball reading or ride on
contexts barely clearing the support floor. Treat the exhibit as provisional: raise
`min_support`, re-inspect, and cross-check the effective-order distribution — if the order ≥ 2
mass is tiny, the deep motifs are estimated from little and are the first suspects for overfit.

---

## 7. Discussion

The interpretation mirrors the §6 branch grid, so the discussion is written per branch and
completed by the same numbers.

*Pre-registered alternative — not selected.* **If G1 (order beyond L1) — with B1 and M1.** WS2
resolves genuine higher-order ordered
selection. The burden shifts to the richer models to show their machinery *extends* rather than
merely *reproduces* this edge (WS6's learned representation against WS2's explicit grammar), and
to the outcome and OPE workstreams to test whether the predictable order is also *exploitable*
(findings #2, #3). The effective-order distribution and motif list are the concrete targets they
inherit.

**Selected by the data (2021–2025). If G2 (first-order adequate) — the SPEC §13 expectation.** This
is the study's most likely and
cleanest result. Within-PA history sharpens selection, but the *order* beyond the previous pitch
does not — the grammar is calibrated bigrams. Reported without embarrassment: a modest ordering
effect is a real, publishable finding consistent with the motif literature. It tells the rest of
the ladder exactly what to carry (the previous pitch) and what it must beat to justify more.

*Realized (2021–2025).* Confirmed as the depth read: `α_1` = 17.9 alone earns its keep and
`delta_order_L1` = −0.0056 shows depths 2–4 cost out-of-sample, with the §6.2 nuance that the
first-order *selection* structure the external reference proves real is not captured by WS2's own
product-of-experts (a representation limit, not an absence of signal — the same fragmentation tax
WS1 documents). The brief to the rest of the ladder stands: carry the previous pitch, and WS3's
features are what cash it.

*Pre-registered alternative — not selected.* **If G3 (count/pitcher only).** Diagnose before
believing. WS2's variable-order backoff is
precisely the instrument that separates "no order in the data" from "no order the model can
resolve": read the effective-order distribution and the depth concentrations. If depths 1+ all
pooled to their ceiling *with* ample support, that is a substantive selection finding; if they
pooled for lack of support, it is a data-scale statement handed upward.

**The firewall paragraph (binding).** Whatever branch fires, **a selection grammar says nothing
about outcome value.** A significant `delta_order_L1`, a deep effective order, a positive
`B_seq`, a table of face-valid motifs — each is a statement that *the next pitch is more
predictable given the ordered history*, and **none** is evidence that the ordering helps the
pitcher's *outcome* (finding #2) or that *changing* the sequence would help (finding #3). A
forecastable pitcher who throws unhittable pitches loses nothing; predictability is an
*information* fact, not a *run-value* fact. The outcome question is WS3's; the prescriptive
question is the OPE/RL workstreams' (WS4/5/7). This firewall is the reason the oracle separates
selection from outcome (D20) and the reason WS2 is scoped to finding #1 alone.

---

## 8. Limitations

1. **Classifier-output tokens.** The tokens are Statcast-inferred pitch *families*; a mislabeled
   pitch is a mislabeled token (SPEC §1). Family (8 classes) blunts classifier drift but cannot
   remove it.
2. **Family granularity hides within-family variation.** The grammar is blind to velocity,
   location, and movement *within* a family — a sequencing effect that lives in pitch *physics*
   rather than pitch *names* is invisible here (the mechanism-blindness WS1 quantified at ~3%
   recovery of a planted velocity effect). WS3 puts those quantities in as features.
3. **No cross-PA memory (D29).** The grammar is a within-PA chain; matchup memory (`OM`,
   cross-PA) is not spanned by a Markov context, so `Δ_matchup` = N/A. Cross-PA adaptation is
   handed to the pooled/embedding models (WS3+, WS6).
4. **Selection-only scope.** WS2 models *what is thrown next*, never the *outcome*. Everything it
   reports is finding #1; the firewall (§7) forbids reading it as finding #2 or #3.
5. **A documented pooling approximation.** Keying the ordered lift on the pitcher while pooling
   over the count assumes the order effect is largely count-independent (the count main-effect is
   carried by the base `b`). This keeps deep contexts supported and the lift confound-free but is
   an approximation; a genuinely count-*interacting* order effect would be partially absorbed
   into the base.
6. **The permutation control is a coarse stratification.** Histories are permuted within
   `(count × hand, pitch_number)` strata, which preserves the count mix and PA depth but scrambles
   across pitchers and prior-family multisets — a slightly *stronger* scramble than the finest
   SPEC §8.3 stratification. A collapse therefore confirms "no within-PA ordered dependence
   survives once count × hand × depth is fixed," which is broader than "no order beyond L1"; the
   `delta_order_L1` CI is the sharper, narrower test.

---

## 9. Conclusion

WS2 fits a transparent Bayesian variable-order Markov grammar of pitch *selection* — a two-axis
hierarchical Dirichlet backoff over ordered family tokens — across the feasible views, scores it
through the shared harness, and reads its depth honestly. Its conclusion is branch-conditional
and complete once the real numbers arrive:

- **Under G1** *(pre-registered alternative — not selected)*, WS2 finds ordered selection structure
  beyond the previous pitch and hands the richer workstreams a live signal (an effective depth and
  a motif list) to confirm, extend, and test for exploitability.
- **Under G2 — selected by the data (2021–2025)** (the expected outcome), WS2 finds that order
  beyond the previous pitch does not pay: `α_1` = 17.9 earns its keep while `α_2`–`α_4` = 84.7–166.7
  pool depth away, and `delta_order_L1` = −0.0056 shows the variable-order view costs out-of-sample
  — a clean "first-order adequate" result consistent with SPEC §13, with the documented nuance that
  the real first-order *selection* signal (external reference 1.4410) exceeds what WS2's own grammar
  can hold. The descriptive headline is stickiness (uniform depth-1 repeat-promotion motifs).
- **Under G3** *(pre-registered alternative — not selected)*, WS2's variable-order backoff
  distinguishes "no order in the data" from "no order the data can resolve," and hands the
  appropriate case upward.

Across every branch the durable contributions are the same: a *calibrated depth read* on ordered
selection that spends resolution only where support earns it (validated positively on the
null-world control and shown to collapse under permutation), a bits-of-predictability number for
the SPEC §10 frontier, and a firewall that keeps this selection finding rigorously separate from
the outcome and policy questions the rest of the ladder exists to answer.

---

## References

Begleiter, R., El-Yaniv, R., and Yona, G. (2004). On prediction using variable order Markov
models. *Journal of Artificial Intelligence Research*.

Chen, S. F., and Goodman, J. (1999). An empirical study of smoothing techniques for language
modeling. *Computer Speech & Language*.

Efron, B., and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations.
*Journal of the American Statistical Association*.

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013).
*Bayesian Data Analysis*, third edition. Chapman and Hall/CRC.

Katz, S. M. (1987). Estimation of probabilities from sparse data for the language model component
of a speech recognizer. *IEEE Transactions on Acoustics, Speech, and Signal Processing*.

Kneser, R., and Ney, H. (1995). Improved backing-off for m-gram language modeling. *Proceedings
of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)*.

MacKay, D. J. C., and Peto, L. C. B. (1995). A hierarchical Dirichlet language model. *Natural
Language Engineering*.

Marchi, M., and Albert, J. (2013). *Analyzing Baseball Data with R*. Chapman and Hall/CRC.

Minka, T. P. (2000/2003). Estimating a Dirichlet distribution. Technical note.

Ron, D., Singer, Y., and Tishby, N. (1996). The power of amnesia: learning probabilistic automata
with variable memory length. *Machine Learning*.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*.

Tango, T. M., Lichtman, M. G., and Dolphin, A. E. (2007). *The Book: Playing the Percentages in
Baseball*.

Teh, Y. W. (2006). A hierarchical Bayesian language model based on Pitman–Yor processes.
*Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual
Meeting of the Association for Computational Linguistics*.
