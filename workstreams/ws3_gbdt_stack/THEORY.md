# WS3 Theory: The Boosted Ablation, Derived

This document derives every formula the WS3 gradient-boosted stack uses, one named step at a
time, with a plain-words explanation after each block. Nothing is skipped. The boxed results
match `PAPER.md` and the code in `model.py` exactly; where the code names a function or a
constant, it is noted. WS3 shares the study's ablation and cluster-bootstrap machinery with
WS1/WS2, so §6 recaps the shared min-bias lemma and points to `../ws1_eb_tables/THEORY.md`
§10 for the part derived there, then derives the part that is new to WS3 — why a significantly
*negative* `Δ_matchup` is a real cost. Notation is introduced once in §1 and reused throughout.

---

## 1. Setup and notation

WS3 fits, **per state view** (all five: `C`, `U`, `L1`, `O`, `OM`), three gradient-boosted
pieces over the shared decision table (SPEC §3): a behavior model, a decomposed outcome stack,
and a direct-regression cross-check.

- `s` — the **state**, the feature vector of one decision row in a given view. `X_C ⊆ X_U ⊆ X_O
  ⊆ X_OM` and `X_C ⊆ X_L1 ⊆ X_O` — the views are information-nested (SPEC §6).
- `a ∈ 𝓐` — the **action**, one of the `K_fam = 8` pitch families
  (`FF, SI, FC, SL, CU, CH, FS, XX`, `FAMILIES` in `families.py`). Selection models predict
  `a`; outcome models **condition on** `a` (decision D22, `action_family` appended to the view).
- `c = (balls, strikes)` — the **count**. `count_key = balls·3 + strikes` (`_count_key`).
- The **outcome event tree** (decision D5, `outcomes.py`): level 1
  `o₁ ∈ 𝒪₁ = {ball, called_strike, whiff, foul, hbp, in_play}` (`OUTCOME1`, 6 classes);
  level 2, defined only when `o₁ = in_play`,
  `o₂ ∈ 𝒪₂ = {single, double, triple, home_run, out_or_other}` (`OUTCOME2`, 5 classes).
- `R = −delta_run_exp` — the reward, larger better for the pitcher (SPEC §5).
- `μ(a | s)` — the behavior policy (`BehaviorModel`). `P_A(o₁ | s, a)` — stage A
  (`OutcomeStack._stage_a`). `P_B(o₂ | s, a)` — stage B, fit on in-play rows
  (`OutcomeStack._stage_b`). `V(node, c)`, `S(node, c)` — count-conditional node value and
  within-node variance (`NodeValueTable`).

**In plain terms.** Every model in WS3 is a committee of small decision trees (a gradient-
boosted ensemble). We fit the same committee five times, once per "view" — five nested feature
sets that reveal progressively more of the pitch history. One committee guesses *what pitch
comes next* (behavior); another set of committees, plus a small reward lookup, estimates *what
happens when a given pitch is thrown* (the outcome stack). The differences between the five
views are the sequencing evidence, exactly as in WS1 and WS2 — but now the model can split on
pitch *physics* (velocity, location), not just pitch *names*.

---

## 2. Gradient boosting as functional gradient descent

Every WS3 model is a LightGBM gradient-boosted decision-tree (GBDT) ensemble. We derive the
additive update at sketch level, then specialize to the multiclass softmax objective the
selection and stage-A/stage-B models use.

### 2.1 The additive model and the functional gradient

**Step — write the additive model.** A GBDT builds a scalar score function as a sum of `M`
regression trees, grown one at a time:

$$F_M(x) = F_0(x) + \nu \sum_{m=1}^{M} h_m(x),$$

where each `h_m` is a regression tree, `ν` is the **learning rate** (`learning_rate`), and `F_0`
a constant initializer.

**Step — state the training objective.** For a differentiable per-example loss `ℓ(y, F)`, we
minimize the empirical risk `L(F) = Σᵢ ℓ(yᵢ, F(xᵢ))` over the function `F`. Boosting does this by
**steepest descent in function space**: at stage `m` we already hold `F_{m−1}` and want to add
the `h` that most decreases `L`.

**Step — take the functional gradient (the pseudo-residual).** The direction of steepest descent
at each training point is the negative gradient of the loss with respect to the current function
value,

$$g_{im} = -\left.\frac{\partial \ell(y_i, F)}{\partial F}\right|_{F = F_{m-1}(x_i)},$$

the **pseudo-residual**. Friedman's (2001) insight is to **fit the next tree to the
pseudo-residuals** (a least-squares fit), so the tree approximates the steepest-descent step
using only axis-aligned splits — hence *gradient* boosting.

**Step — set the leaf values by a Newton step.** LightGBM (Ke et al., 2017), following the
second-order refinement of Chen and Guestrin (2016), uses the Hessian too. With per-example
gradient `gᵢ` and Hessian `hᵢ = ∂²ℓ/∂F²`, a leaf `ℓ` containing example set `Iℓ` takes the
regularized Newton-optimal value

$$w_\ell^\star = -\frac{\sum_{i \in I_\ell} g_i}{\sum_{i \in I_\ell} h_i + \lambda},$$

with `λ = reg_lambda` (`DEFAULT_PARAMS["reg_lambda"] = 1.0`). The same ratio scores candidate
splits (the split gain), which is why the reported feature importance is a **gain** sum
(`feature_importance(importance_type="gain")`, `_RawClassifier.feature_gain`).

**In plain terms.** Boosting is gradient descent where the "parameter" being optimized is the
whole prediction function. Each round, we compute how wrong we are at every training row (the
pseudo-residual — for squared error, literally the residual `y − F`), fit one small tree to
those errors, and add a shrunken version of it. The learning rate keeps each step small so many
gentle trees beat a few greedy ones. LightGBM sharpens the step with curvature (the Hessian) and
a ridge penalty `λ`, which is also how it scores splits — and why "gain" is the natural
importance measure for the notebook's top-feature exhibits.

### 2.2 The multiclass softmax objective and why per-class trees

**Step — write the softmax model.** For a `K`-class target (family selection `K = 8`; stage A
`K = 6`; stage B `K = 5`), the model maintains **`K` score functions** `F⁽¹⁾, …, F⁽ᴷ⁾` and maps
them to probabilities by the softmax

$$p_k(x) = \frac{\exp F^{(k)}(x)}{\sum_{j=1}^{K} \exp F^{(j)}(x)}.$$

**Step — write the cross-entropy loss.** The multiclass log loss (the primary metric, SPEC §8.2)
is the negative log-likelihood

$$\ell(y, F) = -\sum_{k=1}^{K} \mathbf{1}[y = k]\,\log p_k = -\log p_y.$$

**Step — differentiate (the per-class gradient and Hessian).** The softmax derivative
`∂p_k/∂F⁽ʲ⁾ = p_k(δ_{kj} − p_j)` gives the clean gradient and diagonal Hessian

$$\frac{\partial \ell}{\partial F^{(k)}} = p_k - \mathbf{1}[y = k], \qquad \frac{\partial^2 \ell}{\partial (F^{(k)})^2} = p_k(1 - p_k).$$

**Step — read off "per-class trees."** The gradient for class `k` is just "predicted minus
observed probability" for that class. LightGBM therefore grows **one tree per class per boosting
round**, each fit to its class's pseudo-residual `1[y = k] − p_k`; the classes couple only
through the shared softmax normalizer. So after `M` rounds a `K`-class model holds `K·M` trees,
and the reported parameter count is

$$\boxed{\; \texttt{n\_params} = (\text{number of trees}) \times \texttt{num\_leaves} \;}$$

(`_RawClassifier.n_params`), summed over stage A and stage B for an `OutcomeStack`. This is the
compute coordinate logged for the SPEC §7 Pareto plot.

**In plain terms.** For an 8-way "which family next?" question, the model keeps eight running
scores and squashes them into probabilities with a softmax. The gradient turns out to be exactly
"how much did I over- or under-estimate this class here," so each round the ensemble grows one
little tree per class chasing that per-class miss. Eight families means eight trees per round; the
six-way pitch-result model grows six; the five-way in-play model grows five. Counting all of them
times the leaves per tree is the honest size of the model, which we log against wall-clock and RAM.

---

## 3. Why tabular ordered features are the right `O`-encoding for trees (D11)

A tree splits on one feature at a time; it cannot consume a variable-length sequence directly. So
the ordered within-PA history must be **encoded as fixed-width features**. Decision D11 fixes that
encoding for the tree family: the tabular `O` view is `C` + the unordered `U` block + the previous
pitch `L1` + an **ordered block** of last-3 positional slots, consecutive-difference features, and
a run-length feature (`states.py::_o_block`). This section makes the honest capability-and-limits
argument for that choice — it is a real modeling decision, not a shortcut.

**Step — state the information claim (D11).** The tabular `O` and the sequence-model `O`
(WS6's GRU) span the **same information set** — the ordered families, physics, and outcomes of the
prior pitches this PA — and differ only in *representation*. The tree reads that information through
columns; the RNN reads it through a recurrence. D11: "Same information set, representation differs
by model family."

**Step — show order is preserved by position.** The block carries **positional slots**
`o_s1_*, o_s2_*, o_s3_*` (slot `k` = the `k`-th most recent prior pitch), each with the prior
pitch's family, outcome token, release speed, and batter-relative location, plus `o_first_family`.
Because slot index is a distinct column, a tree can split on `o_s1_family = SL AND o_s2_family = FF`
differently from `o_s1_family = FF AND o_s2_family = SL` — i.e. it distinguishes *fastball→slider*
from *slider→fastball*. **Order is representable** even though the model is "just" a table of
features: the order lives in *which column* a family sits in.

**Step — show the mechanism is native, not proxied (the crux vs WS1).** The block also carries the
**consecutive-difference** features `o_velo_delta_last` (the velocity change into the previous
pitch, `|velo_{t−1} − velo_{t−2}|` up to sign), `o_velo_delta_mean`, `o_loc_delta_last/_mean`, and
the same-family `o_same_fam_run`. A single split `o_velo_delta_last ≥ 5` **directly** represents a
velocity-transition mechanism. This is exactly what WS1's name-keyed tables *cannot* do: they see a
velocity effect only through the pitch-name → velocity-band proxy and recover ~3% of a planted
velocity effect (WS1 mechanism-blindness). WS3 puts the physical difference in as a column, which is
why — on the positive synthetic world — `o_velo_delta_last` is the **top gain feature** of the `O`
and `OM` stage-A models and the recovery ratio jumps to ≈0.955 (§8, and `PAPER.md` §5). The
representation choice *is* the contrast exhibit.

**Step — state the limits honestly.** Three, none hand-waved:
1. **Horizon truncation.** Slots reach 3 prior pitches; beyond that, order survives only as
   *summaries* (`o_velo_delta_mean`, `o_same_fam_run`). A genuinely long-range order-4+ dependence
   is only partially represented — a real ceiling the sequence model (WS6) exists to test.
2. **Axis-aligned interactions only.** Trees find greedy axis-aligned splits; a smooth continuous
   "tunneling" geometry (release point + movement + location jointly) may be captured only coarsely
   by rectangular splits, where a learned continuous encoder could do better. WS3 measures what a
   *strong tabular* model can find, which bounds — but does not settle — what a learned representation
   could.
3. **Missingness structure.** Early-PA rows have missing slots; the block neutral-fills and adds
   explicit `*_missing` indicators (`o_s1_missing`, …). The tree must *learn* the missing-indicator
   interaction rather than getting it for free, so a first-pitch row and a deep-count row are handled
   by the same model through those flags.

**In plain terms.** Trees can't read a sentence, so we hand them the sentence as labeled boxes:
"most-recent pitch," "one before that," "two before that," plus "how much the speed jumped," "how
long the current same-pitch streak is." Because the boxes are position-labeled, the model still
knows *order* (which pitch was where), and because we include the actual speed and location changes
as numbers, it can see the *mechanism* a name-only table is blind to — the whole reason WS3 recovers
a planted velocity effect that WS1 barely detects. The honest costs are that we only look back three
pitches with full detail, trees carve the space into rectangles, and the model has to learn how to
treat the "no pitch yet" flags. Those are exactly the gaps WS6 is built to probe.

---

## 4. The D32 decomposition: `E[R | s, a]` and its residual-based sd

The outcome stack does **not** regress reward directly as its primary estimate. It assembles
`E[R | s, a]` from the event tree (decision D32), because the interpretable pieces —
`P_A`, `P_B`, and the count-conditional node values — are each independently reusable by WS4/5/7
(decision D33). Here we derive the assembly and its uncertainty exactly as coded
(`OutcomeStack._assemble`).

### 4.1 The expected-reward assembly

**Step — enumerate the leaves.** The two-level event tree has **10 leaves**: the 5 terminal
level-1 outcomes `{ball, called_strike, whiff, foul, hbp}` (each ends the pitch), plus the 5
level-2 refinements of `in_play`, `{single, double, triple, home_run, out_or_other}`. Write `_NON_INPLAY`
for the five non-in-play indices into `𝒪₁` and `_I_INPLAY` for the `in_play` index (`model.py`).

**Step — assign a value to each leaf (count-conditional).** Conditional on the count `c`, leaf `ℓ`
carries the train-fold mean reward of its event-tree node:

$$V_1(k, c) = \operatorname{mean}\{R_i : o_{1,i} = k,\ c_i = c\}, \qquad
  V_2(j, c) = \operatorname{mean}\{R_i : o_{2,i} = j,\ c_i = c,\ \text{in play}\},$$

each with a **global-node fallback** (the count-marginal node mean) when a `(node, count)` cell is
empty, and `0` when a node is entirely absent (`NodeValueTable.fit`, `._cell_stats`).

**Step — factorize the leaf probability (the event-tree chain rule).** By the tree structure, a
terminal level-1 leaf `k` has probability `P_A(o₁ = k | s, a)`; an in-play leaf `j` has probability
`P_A(o₁ = in_play | s, a) · P_B(o₂ = j | s, a)` (stage B is the distribution *conditional on
in-play*). Stack the ten leaf probabilities as
`p_leaf = [ P_A(·)|_{non-inplay} , P_A(in_play)·P_B(·) ]` (`_assemble`).

**Step — take the expectation over the leaf (law of total expectation).**
`E[R | s, a, c] = Σ_ℓ p_ℓ V_ℓ(c)`. Substituting the factorized probabilities and the node values:

$$\boxed{\;
\mathbb{E}[R \mid s, a, c] =
  \sum_{k \in \{\text{ball},\text{called\_strike},\text{whiff},\text{foul},\text{hbp}\}}
       P_A(o_1 = k \mid s, a)\; V_1(k, c)
  \;+\; P_A(o_1 = \text{in\_play} \mid s, a)
       \sum_{j \in \mathcal{O}_2} P_B(o_2 = j \mid s, a)\; V_2(j, c)
\;}$$

This is the exact formula in the `model.py` module docstring and `PAPER.md` §4. In code it is the
one-line `er = (p_leaf * v_leaf).sum(axis=1)` with `v_leaf = [V_1|_{non-inplay}, V_2]`.

**In plain terms.** "What run value do I expect if this pitch is thrown here?" is answered by
walking the event tree: with some probability the pitch is a ball, a called strike, a whiff, a foul,
a hit-by-pitch, or put in play; if put in play, it becomes a single/double/…/out with further
probabilities. Each of those ten endings is worth a certain number of runs *in this count* (learned
as a simple average on the training fold). Multiply each ending's probability by its run value and
add them up. The probabilities come from two boosted classifiers; the run values from a count-keyed
lookup — three interpretable parts, one number.

### 4.2 The residual-based standard deviation (law of total variance)

**Step — treat the leaf as a latent node.** Conditional on landing in leaf `ℓ`, reward has mean
`V_ℓ(c)` and **within-node variance** `S_ℓ(c)` (the train-fold reward variance inside that node,
`NodeValueTable`'s `s1`/`s2`). The leaf itself is latent with distribution `p_ℓ`.

**Step — apply the law of total variance.**
`Var[R] = E_ℓ[Var(R | ℓ)] + Var_ℓ(E[R | ℓ])`. The first term is `Σ_ℓ p_ℓ S_ℓ(c)`; the second is
`Σ_ℓ p_ℓ V_ℓ(c)² − (Σ_ℓ p_ℓ V_ℓ(c))² = Σ_ℓ p_ℓ V_ℓ² − E[R]²`. Adding:

$$\boxed{\;
\mathrm{Var}[R \mid s, a, c] = \sum_{\ell} p_\ell\big(S_\ell(c) + V_\ell(c)^2\big) - \mathbb{E}[R \mid s, a, c]^2,
\qquad \texttt{exp\_reward\_sd} = \sqrt{\max(\mathrm{Var}, 0)}
\;}$$

summed over the 10 leaves, exactly `second_moment = (p_leaf*(s_leaf + v_leaf**2)).sum(axis=1)` and
`var = maximum(second_moment − er**2, 0)`. The `max(·, 0)` guards tiny negative values from
finite-precision node estimates. **No bootstrap** — the uncertainty is purely residual, decomposing
into "which ending is uncertain" (`p_ℓ`) and "how noisy reward is within each ending" (`S_ℓ`).

**In plain terms.** The uncertainty in the expected reward has two sources: we're unsure *which*
ending happens (a coin-flip between whiff and foul, say), and even given the ending, the run value
scatters (an "out" in a 3-2 count varies). The law of total variance adds those two exactly — the
spread *between* endings plus the average spread *within* endings — giving a per-pitch error bar for
free, no resampling. That error bar is what WS4's uncertainty-aware bandit later consumes.

---

## 5. The decomposed-vs-direct cross-check (D32)

The stack also fits a **direct** regressor `Ê_dir[R | s, a]` — one LightGBM regression on the same
`(view + action)` features, target `R` (`_RawRegressor`, `objective="regression"`). It is a
consistency check on the assembly, not a second product.

**Step — define the disagreement statistic.** On an evaluation table,

$$\texttt{mean\_abs\_diff} = \frac{1}{n}\sum_i \big|\,\hat{E}_{\text{dec}}[R \mid s_i, a_i] - \hat{E}_{\text{dir}}[R \mid s_i, a_i]\,\big|,$$

flagged when it exceeds `DISAGREEMENT_FLAG_ABS = 0.03` on the `|R| ~ 0.05–0.3` reward scale
(`OutcomeStack.disagreement`).

**Step — say what a disagreement diagnoses.** The two estimators differ in what they assume. The
**decomposed** estimate imposes structure — the event-tree factorization *and* the assumption that,
given the leaf, reward depends on `(s, a)` only through `(node, count)` (the node value `V(node, c)`
ignores the rest of `s`). The **direct** regressor is unconstrained: it can let reward depend on any
`(s, a)` feature. So a large `mean_abs_diff` means one of:
1. the node-value lookups are mis-specified — a `(node, count)` cell too thin, falling back to the
   global node and discarding count structure the direct regressor still captures; or
2. the count-conditional-value assumption is violated — reward genuinely depends on `(s, a)` beyond
   `(node, count)`, which the direct model sees and the assembly cannot.

It is an **internal** check (both can be wrong together), not a ground-truth check; that is why it is
reported as a bounded-disagreement flag, not a pass/fail gate.

**Step — read the honest OM flag (completed validation).** On the untuned demo models the `OM`
disagreement lands *slightly over* 0.03 while the other four views sit at 0.023–0.026 (unflagged).
The reading: `OM`'s larger, sparser feature space fragments the stage-A/B fits relative to the
count-conditional node values (which do not carry the matchup dimension), nudging the two estimates
apart — a mild mis-alignment expected to tighten under `--tune`, not a correctness failure. It is
carried forward as an honest flag (dispatch log), and it is the same fragmentation story that §6's
negative `Δ_matchup` tells from the loss side.

**In plain terms.** We estimate the expected reward two ways — once by walking the event tree with a
count-keyed lookup (interpretable, reusable), once with a black-box regressor (unconstrained) — and
check they roughly agree. If they don't, either a run-value lookup cell was too empty to trust or the
reward really depends on something the tidy decomposition threw away. It is a smoke alarm for the
decomposition, not proof either one is right. The one view that trips the alarm on the demo is `OM`,
whose extra matchup features the tidy lookup ignores — the same reason `OM` will show a small
*out-of-sample cost* in the central table.

---

## 6. Ablation inference: the CIs, D21's `Δ_order` bias, and the *absence* of it in `Δ_matchup`

WS3's headline is the SPEC §6 ablation on **three** targets — selection log loss, outcome-1 log
loss, and run-value MAE — with pitcher-game clustered CIs. This section fixes how each is read.

### 6.1 The two ablation statistics and their clustered CIs

$$\Delta_{\text{order}} = \mathrm{Loss}(\min[U, L1]) - \mathrm{Loss}(O), \qquad
  \Delta_{\text{matchup}} = \mathrm{Loss}(O) - \mathrm{Loss}(OM),$$

both positive when the richer view has lower loss (helps out-of-sample). For run value the "loss" is
MAE and the same sign convention holds: `Δ_order = min(MAE_U, MAE_L1) − MAE_O`,
`Δ_matchup = MAE_O − MAE_OM` (`run_ws3.py::_runvalue_compare`). CIs come from a **cluster bootstrap**
that resamples pitcher-game clusters with replacement and recomputes the statistic
(`metrics.py::clustered_ci`), reported as `[lo, hi]`; the clustering respects SPEC §0/§7 (pitches in a
pitcher-game are dependent, so the resampling unit is the cluster, not the row).

### 6.2 D21's min-bias for `Δ_order` (recap — derived in WS1 THEORY §10)

`Δ_order` takes the **minimum** of two held-out losses. `min(·,·)` is concave, so by Jensen's
inequality, under the null where the three views share a true loss `L⋆`,

$$\mathbb{E}[\Delta_{\text{order}}] = \mathbb{E}[\min(\hat L_U, \hat L_{L1})] - \mathbb{E}[\hat L_O] \le L^\star - L^\star = 0.$$

So `Δ_order` is **negatively biased under the null** (decision D21): the criterion is *significantly
positive* (clustered CI lower bound > 0), and a small **negative** value reads as *consistent with no
ordering effect*, never "order hurts." This binds on all three WS3 targets. The full derivation and
the bias-size estimate `≈ ½·E|L̂_U − L̂_L1|` are in `../ws1_eb_tables/THEORY.md` §10; WS3 inherits the
rule unchanged. The outcome-1 `Δ_order` is the project's central number (finding #2); the selection
`Δ_order` is finding #1 and a separate claim.

### 6.3 Why a significantly *negative* `Δ_matchup` is a real cost, not the same bias

This is new to WS3 (the first workstream that can fit `OM`) and must be read differently from
`Δ_order`.

**Step — note there is no `min`.** `Δ_matchup = Loss(O) − Loss(OM)` is a **straight difference of two
held-out losses**, each an unbiased estimate of its view's expected loss (the same structure as WS2's
bias-free `delta_order_L1`, `../ws2_bayes_markov/THEORY.md` §8). There is **no Jensen gap**:

$$\mathbb{E}[\Delta_{\text{matchup}}] = \mathbb{E}[\hat L_O] - \mathbb{E}[\hat L_{OM}] = L_O - L_{OM}.$$

Under a matchup null (`L_O = L_OM`) the expectation is `0` — **unbiased**, not optimistically biased.

**Step — derive the direction of the finite-sample tilt.** `OM` **nests** `O` (`X_O ⊆ X_OM`, adding
~21 matchup features, most of them sparse or missing early in a game). With infinite data `L_OM ≤ L_O`
(more information cannot hurt the Bayes-optimal predictor). But at finite sample, a fitted model over
more features carries more **estimation variance**. Decompose the excess out-of-sample loss of `OM`
over `O` as

$$\underbrace{L_{OM} - L_O}_{\text{excess loss}} = \underbrace{-(\text{signal gained})}_{\le 0} \;+\; \underbrace{(\text{extra estimation variance})}_{\ge 0}.$$

If the matchup features carry **no signal** (the world has no matchup effect), the first term is 0 and
only the variance term survives, so `L_OM > L_O` and

$$\boxed{\; \Delta_{\text{matchup}} < 0 \ \text{(significantly)} \iff \text{OM pays estimation variance for features that carry no signal — a real out-of-sample fragmentation cost.} \;}$$

**Step — contrast the two negatives (the point).** A small negative `Δ_order` is a **statistical
artifact** of the `min` estimator (optimism), *not* a cost — hence "consistent with no effect." A
significantly negative `Δ_matchup` has **no such artifact to blame** (no `min`); it is the genuine
bias-variance signature of carrying many sparse features that add no signal. Both say "the phenomenon
(order / matchup) does not *help* here," but the *mechanism* differs: estimator optimism vs. real
generalization cost.

**Step — what it does and does not prove.** A negative `Δ_matchup` proves that, *in this data*, adding
the matchup block **hurts** out-of-sample outcome prediction. It does **not** prove matchup memory is
absent from baseball; it is "no detectable matchup outcome signal **and** a fragmentation cost from
the OM feature block." This is exactly what is **observed on both synthetic worlds by construction**
(no matchup effect planted): `Δ_matchup` outcome-1 `= −0.0078` CI `[−0.0145, −0.0018]` (null) and
`= −0.0129` CI `[−0.0193, −0.0080]` (positive). On real data the same negative pattern is the `M−`
branch (`PAPER.md` §6), read with a **support-diagnostics checklist** (are the OM features mostly
missing/low-support? then it is fragmentation, not a real negative matchup effect).

**In plain terms.** WS3 is the first model that can even *try* the matchup view, so the first honest
question is whether adding "what has this batter done against this pitcher" helps predict the outcome.
The comparison `O` vs `OM` has no `min` trick in it, so — unlike the order comparison — a negative
number isn't a statistical mirage. When the matchup features carry no real signal, cramming ~21 mostly
empty columns into the model just makes it noisier out-of-sample, and it scores *worse*. That is a
true cost, and we see it cleanly on both synthetic worlds where we *know* there is no matchup effect.
On real data a negative here means "no matchup outcome signal we can see, and the matchup features are
actively costing us" — after we check the support diagnostics to be sure it's fragmentation and not a
genuine backfire.

---

## 7. Calibration: the run-value regression and the probability reliability curve

WS3 reports two distinct calibrations, both through the shared harness
(`harness.py::evaluate_predictions`).

**Step — the run-value calibration regression.** Regress the realized reward on the predicted
expected reward across evaluation rows,

$$R_i = \text{intercept} + \text{slope}\cdot \hat E[R \mid s_i, a_i] + \varepsilon_i,$$

reported as `slope`, `intercept`, `r2` (`...["exp_reward"]["calibration"]`). **Perfect calibration is
`slope = 1`, `intercept = 0`.** A `slope < 1` means the predictions are *too spread out* (over-
confident extremes shrunk toward the mean would score better); `slope > 1` means *under-confident*; an
`intercept ≠ 0` is a global run-value bias. This is the run-value analogue of SPEC §8.2's "run-value
calibration intercept & slope."

**Step — the probability reliability curve.** For the discrete predictions (selection top-class
confidence; outcome-node probabilities), bin the predicted probability into deciles and plot the
observed frequency against the mean predicted probability in each bin (`metrics.py::reliability_table`
→ `mean_pred`, `frac_pos`, `count`). Points on the 45° line = calibrated; systematic sag below it =
over-confidence in that bin. Bin counts are annotated so sparse bins are discounted.

**In plain terms.** Calibration asks "when the model says 0.2 runs, does about 0.2 runs actually
happen?" For the reward we fit a line of *actual vs predicted* and hope for slope 1, intercept 0 —
slope under 1 means the model's highs are too high and lows too low. For the probabilities we bin the
forecasts and check the realized rate in each bin lands on the diagonal. A miscalibrated outcome model
can still rank actions correctly, but WS4/5/7 consume the *levels*, so calibration is load-bearing
downstream.

---

## 8. The counterfactual q̂ grid: prediction under a hypothetical action ≠ a causal claim

The q̂ grid (`q_grid`, decision D33) sweeps the action over all 8 families for every decision row:
build the view once, set `action_family = f`, re-run stages A/B, re-assemble
`q̂(s, f) = Ê[R | s, A = f]` (`OutcomeStack.q_grid`, shape `(n, 8)`). This section formalizes the
finding #2 / finding #3 firewall (SPEC §0) around it.

**Step — define the estimand actually computed.** For any family `f`,

$$\hat q(s, f) = \sum_{\ell} \hat P(\text{leaf} = \ell \mid s, A = f)\; V(\ell, c)$$

is a **well-defined functional of the fitted conditional model** — plug `A = f` into stages A/B and
assemble. It is defined for every `f`, including families not thrown in state `s`.

**Step — state what it is *not*.** Stages A/B are fit on **observed** `(s, a)` pairs. Evaluating them
at `A = f` for a row where `a ≠ f` was thrown is an **extrapolation along the action feature**, and it
identifies the **causal** `E[R | s, do(A = f)]` only under assumptions the data cannot establish:
1. **No unobserved confounding** (ignorability): `R(f) ⊥ A | s`. SPEC §0 says public data violates this
   — the catcher's target, the battery's intent, and scouting are unobserved, and they drive both the
   pitch chosen and its outcome.
2. **Overlap / support**: the pitcher must actually throw `f` in states like `s`, or the extrapolation
   has no data to stand on.

**Step — draw the firewall.** So `q̂` is **finding #2** material: a conditional expectation
"if family `f` is thrown here, the fitted model expects this outcome," honest as *prediction*. It
becomes **finding #3** (prescriptive: *changing* the pitch would help) **only** after passing the OPE
gate (WS4/5/7) with its behavior-policy anchoring, importance-weight and support diagnostics, and the
`INCONCLUSIVE`-if-estimators-disagree rule (SPEC §9). The notation

$$\hat q(s, f) = \hat E_{\text{model}}[R \mid s, A = f] \quad \ne \quad E[R \mid s, \mathrm{do}(A = f)]$$

is the firewall in symbols: the left side is what WS3 publishes; equating it with the right side is the
assumption the prescriptive workstreams must *test*, not one WS3 may *assume*. The counterfactual
sensitivity sanity check in the notebook (swap the action, watch outcome probabilities move) confirms
the grid *responds* to the action feature; it does not license a causal reading.

**In plain terms.** The q̂ grid is a "what-if card": for every decision it lists the model's expected
run value for each of the 8 pitches, by pretending each was the one thrown. That pretend-swap is a
legitimate *prediction* from a model that conditions on the pitch — but it is **not** proof that
throwing a different pitch would *cause* a better result, because we never see the pitch that wasn't
thrown, and the reasons a pitcher picked what he picked (the target, the scouting) are invisible and
affect the outcome too. So the grid is finding #2 (a careful prediction), and only the OPE machinery
downstream — which anchors to the actual behavior and refuses to answer when its estimators disagree —
can try to turn it into finding #3 (a recommendation). WS3 draws that line on purpose.

---

## 9. Hyperparameter-budget fairness: why equal budgets make the deltas comparable (D34)

**Step — state the budget.** Every view gets the **identical** predeclared `DEFAULT_PARAMS` and, under
`--tune`, the **identical** 12-combo `HP_GRID`
(`num_leaves ∈ {15,31,63} × min_child_samples ∈ {20,100} × learning_rate ∈ {0.03,0.1}`), selected by
**validation log loss on an internal holdout carved from the *train* seasons** — never the reported
validation/test folds (`select_hyperparams`, `_carve_tune_holdout`, `_resolve_params`). Stage A tunes;
stage B and the direct regressor **reuse** stage A's chosen params to keep the grid cost bounded
(`OutcomeStack.fit`).

**Step — argue comparability.** The ablation attributes a loss difference between two views to their
*information* (feature-set) difference. If `O` were allowed a larger search than `L1`, a positive
`Δ_order` could be a **budget** artifact (a bigger grid finds a better fit) rather than an
**information** artifact (order carries signal). Fixing the grid identical across views removes the
budget as a confounder: the only thing that varies view-to-view is the feature set, so the loss gap
isolates the information contribution. This is SPEC §7's equal-hyperparameter-budget discipline, made
operational.

**Step — note the holdout hygiene.** The tuning holdout is the latest **train** season (when ≥2 train
seasons exist and both sides clear `min_rows`), else a deterministic 25% `game_pk` hash — always inside
the train fold. The reported validation (2024) and locked test (2025) are never touched for selection,
so the central table is an honest out-of-sample read, and the chosen params are logged per view for the
Pareto plot.

**In plain terms.** To make "view `O` beats view `L1`" mean "order carries information" rather than
"`O` got a luckier tuning run," we give every view the exact same knob budget and pick knobs only on a
slice carved out of the training years — never on the validation or test years we report. Then the only
difference between the views is *what they can see*, which is precisely what the ablation is supposed to
measure. We log the chosen knobs and the model size for the compute-cost accounting.
