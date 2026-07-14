# WS2 Theory: The Grammar, Derived

This document derives every formula the WS2 variable-order Markov grammar uses, one named step
at a time, with a plain-words explanation after each block. Nothing is skipped. The boxed
results match `PAPER.md` and the code in `model.py` exactly; where the code names a function it
is noted. WS2 shares its Dirichlet-fitting machinery with WS1, so §4 states the shared lemmas
and points to `../ws1_eb_tables/THEORY.md` for the parts derived there, while staying
self-contained enough to follow on its own.

---

## 1. Setup and notation

We model a stream of **pitch-family tokens** within each plate appearance (PA). Fix a PA and a
decision point `t` (the choice made just before pitch `t`).

- `K = 8` — the number of pitch families (`FF, SI, FC, SL, CU, CH, FS, XX`); `a` indexes a
  family.
- `w_1, w_2, …` — the realized family tokens of the PA in chronological order. The **target** at
  `t` is the next token, family `A_t`.
- **Context at depth `k`** — the ordered last `k` families before `t`, written chronologically
  $$ctx_k = (w_{t-k}, \dots, w_{t-1}),$$
  with depth 0 the empty context. A row is **eligible at depth `k`** only if the PA supplies `k`
  real priors, i.e. `pitch_number ≥ k + 1`; otherwise a context never contains a "no-prior"
  sentinel.
- **Suffix** — $\operatorname{suffix}(ctx_k) = (w_{t-k+1}, \dots, w_{t-1}) = ctx_{k-1}$, i.e.
  drop the **oldest** family `w_{t−k}`. (Chronological order matters: the suffix keeps the
  *recent* tokens and forgets the distant one.)
- **Conditioning cell** — `ch = (balls, strikes, stand, p_throws)`, the count × handedness cell;
  `pit` is the pitcher.
- `π_G` — the global family marginal; `m(a) = P(a \mid pit)` — the pitcher marginal; `b(a)` — the
  count × pitcher base rate. `g_k(a \mid ctx_k, pit)` — the depth-`k` grammar distribution.
- Each depth `k` has a single shared **concentration** `α_k`; the cell axis has `β_ch`, `β_pit`.
  All are fit by empirical Bayes (§4).

**In plain terms.** Treat the PA as a short sentence and each pitch family as a word. A "context
of depth `k`" is the last `k` words; its "suffix" is the same window with the *oldest* word
dropped, so backing off one level means forgetting the most distant pitch and keeping the recent
ones. Everything below is about predicting the next word from as many recent words as the data
can support, and gracefully forgetting the rest.

---

## 2. The hierarchical Dirichlet suffix chain → the posterior mean

We model the next-family counts in a depth-`k` context cell as multinomial with an unknown
probability vector `p`, and put a Dirichlet prior on `p` centered on the **suffix parent's**
posterior mean.

**Step — write the likelihood (multinomial).** Let the depth-`k` context cell (for a given
pitcher) have next-family counts `n = (n_1, …, n_K)`, `N = Σ_a n_a`. Given `p`,

$$P(n \mid p) \;\propto\; \prod_{a=1}^{K} p_a^{\,n_a}.$$

**Step — write the prior (Dirichlet centered on the suffix parent).** Let
$\pi \equiv g_{k-1}(\cdot \mid ctx_{k-1}, pit)$ be the posterior mean of the **suffix** context
(one level shallower), already computed. With concentration `α_k` the prior pseudo-counts are
`α_k π_a`:

$$p \sim \mathrm{Dir}(\alpha_k \pi_1, \dots, \alpha_k \pi_K), \qquad
P(p) = \frac{1}{B(\alpha_k \pi)} \prod_{a=1}^{K} p_a^{\,\alpha_k \pi_a - 1}.$$

**Step — apply Bayes' rule (posterior ∝ prior × likelihood).**

$$P(p \mid n) \;\propto\; \prod_{a=1}^{K} p_a^{\,(\alpha_k \pi_a + n_a) - 1},$$

the kernel of a Dirichlet — the prior is **conjugate** to the multinomial.

**Step — recognize conjugacy and take the mean.** The posterior is
$\mathrm{Dir}(\alpha_k \pi_a + n_a)$; the mean of a Dirichlet is each parameter over their sum,
and $\sum_a(\alpha_k \pi_a + n_a) = \alpha_k + N$ because $\sum_a \pi_a = 1$. Hence

$$\boxed{\; g_k(a \mid ctx_k, pit) = \frac{n_a + \alpha_k\, g_{k-1}(a \mid ctx_{k-1}, pit)}{N + \alpha_k}, \qquad g_0 \equiv m \;}$$

which is `dirichlet_posterior_mean(counts, parent, alpha)` in `model.py`, with the parent being
the suffix-chain posterior mean. The recursion bottoms out at `g_0 = m`, the pitcher marginal
(itself shrunk toward `π_G`); an unseen context (`N = 0`) returns the suffix parent exactly —
that is **backoff**.

**The cell axis is the same conjugacy.** The base rate `b` is built by the identical update on
the cell ladder: `π_G` shrunk into `ch` with `β_ch`, then into `(ch, pit)` with `β_pit`,

$$P(a \mid ch) = \frac{n_{ch} + \beta_{ch}\,\pi_G}{N_{ch} + \beta_{ch}}, \qquad
b(a) = \frac{n_{ch,pit} + \beta_{pit}\,P(a\mid ch)}{N_{ch,pit} + \beta_{pit}}.$$

**Step — the product-of-experts row prediction.** The row prediction combines the base `b` and
the ordered lift `g_k / m` multiplicatively and renormalises:

$$\boxed{\; P(a \mid ch, pit, ctx_k) = \frac{b(a)\, g_k(a \mid ctx_k, pit)\,/\,m(a)}{\sum_{a'} b(a')\, g_k(a' \mid ctx_k, pit)\,/\,m(a')} \;}$$

When the context is uninformative, `g_k = m`, the ratio is 1 and the prediction is exactly `b`
(so the `C` view is `b`); `L1` / `O` apply the depth-1 / depth-`k` lift.

**In plain terms.** The Dirichlet is the natural prior over a probability vector and is conjugate
to the multinomial, so "updating" is just adding observed counts to prior pseudo-counts. Here the
prior for a depth-`k` context is its own *suffix* (the same context minus the oldest pitch),
scaled by a strength `α_k`. So each level's estimate is its own counts "topped up" by a blurrier,
better-supported shorter-context estimate. The final row prediction multiplies the pitcher's
count-conditioned repertoire (the base) by how the ordered context *reshapes* that repertoire
(the lift), then renormalises.

---

## 3. Unrolling the recursion → interpolation over all shallower depths

The one-step update in §2 hides a familiar object: written out, the depth-`k` posterior mean is
a **convex mixture over every shallower depth's own-count estimate**, exactly like interpolated
*n*-gram smoothing. We derive the mixture weights and name each move.

**Step — put the update in shrinkage form.** Split the boxed numerator of §2 and multiply the
data term by `N/N`:

$$g_k = \frac{N}{N + \alpha_k}\,\hat p_k^{\text{MLE}} + \frac{\alpha_k}{N + \alpha_k}\,g_{k-1}
     = w_k\,\hat p_k^{\text{MLE}} + (1 - w_k)\,g_{k-1}, \qquad w_k \equiv \frac{N_k}{N_k + \alpha_k},$$

where $\hat p_k^{\text{MLE}} = n / N$ is the context's own empirical distribution and `N_k = N`
its support. (This is the same algebra as WS1 THEORY §3; `w_k` is the "own-count weight".)

**Step — substitute one level down.** Replace `g_{k−1}` by its own shrinkage form
$g_{k-1} = w_{k-1}\hat p_{k-1}^{\text{MLE}} + (1 - w_{k-1}) g_{k-2}$:

$$g_k = w_k\,\hat p_k^{\text{MLE}} + (1 - w_k)\big[w_{k-1}\,\hat p_{k-1}^{\text{MLE}} + (1 - w_{k-1})\,g_{k-2}\big].$$

**Step — recurse to the bottom (`g_0 = m`) and collect terms.** Iterating the substitution down
to depth 0 and grouping the coefficient of each $\hat p_j^{\text{MLE}}$ gives the closed form

$$\boxed{\; g_k = \sum_{j=1}^{k} \underbrace{\Big[\, w_j \prod_{i=j+1}^{k} (1 - w_i)\Big]}_{\lambda_j}\, \hat p_j^{\text{MLE}} \; + \; \underbrace{\Big[\prod_{i=1}^{k} (1 - w_i)\Big]}_{\lambda_0}\, m \;}$$

**Step — check the weights are a convex combination.** By the telescoping identity
$1 - \prod_{i=1}^{k}(1 - w_i) = \sum_{j=1}^{k} w_j \prod_{i=j+1}^{k}(1 - w_i)$, the weights sum to
one: $\sum_{j=1}^{k}\lambda_j + \lambda_0 = 1$, and each is in $[0,1]$. So `g_k` is a genuine
mixture: weight $\lambda_j$ on depth-`j` own counts, weight $\lambda_0$ on the pitcher marginal.

**Step — read the weights.** Depth `j`'s weight $\lambda_j = w_j \prod_{i>j}(1 - w_i)$ is large
only if depth `j` itself has enough data (`w_j` near 1) **and** every deeper level was starved
(`w_i` small for `i > j`, so their $1 - w_i$ are near 1). This is precisely variable-order
backoff: the mixture concentrates on the **deepest well-supported depth** and leaks the rest to
shallower ones.

**In plain terms.** Although the model is defined one level at a time, unrolling it shows the
prediction is a weighted blend of "what happened after the last 1 pitch," "the last 2 pitches,"
… "the last `k` pitches," plus the pitcher's overall mix. The blend weight on a given depth is
high only when that depth has real data and all deeper depths do not. That is the whole point of
*variable* order: it automatically leans on the longest context the data can actually support and
smoothly forgets the rest — the same shape as interpolated Kneser–Ney, obtained here from
Dirichlet conjugacy.

---

## 4. Fitting `α_k`: the Pólya evidence at each depth

To *fit* each depth's concentration we need the probability of a context cell's counts with `p`
integrated out — the Dirichlet-multinomial (Pólya) marginal — summed over the exchangeable cells
at that depth, holding the **suffix parents fixed** from the level above (a top-down pass).

**Step — integrate out `p` (the shared lemma).** For one cell with prior
$\mathrm{Dir}(\alpha_k \pi)$ and counts `n`, integrating the multinomial likelihood against the
Dirichlet prior gives (derived in full in WS1 THEORY §4; the Dirichlet integral is
$\int_\Delta \prod_a p_a^{a_a - 1} dp = \prod_a\Gamma(a_a)/\Gamma(\sum_a a_a)$):

$$P(n \mid \alpha_k, \pi) = \binom{N}{n}\, \frac{\Gamma(\alpha_k)}{\Gamma(\alpha_k + N)} \prod_{a=1}^{K} \frac{\Gamma(\alpha_k \pi_a + n_a)}{\Gamma(\alpha_k \pi_a)}.$$

**Step — take logs, drop the `α`-free term, and sum over the level's cells.** The multinomial
coefficient does not depend on `α_k`; summing the log evidence over all cells `c` at depth `k`
(each with its own fixed suffix-parent mean $\pi_c \equiv g_{k-1}$) gives the objective the code
maximises (`_dirichlet_neg_log_evidence`, negated because the code minimises):

$$\boxed{\; \log P(\{n_c\} \mid \alpha_k) \;\overset{c}{=}\; \sum_c \Big[\log\Gamma(\alpha_k) - \log\Gamma(\alpha_k + N_c) + \sum_{a=1}^{K}\big(\log\Gamma(\alpha_k \pi_{c,a} + n_{c,a}) - \log\Gamma(\alpha_k \pi_{c,a})\big)\Big] \;}$$

with $\overset{c}{=}$ meaning "up to an additive constant in `α_k`". This is a smooth 1-D
function of the single scalar `α_k` (the parent means are fixed), optimised on `log10 α_k` over a
bounded interval by `scipy.optimize.minimize_scalar` — reported as method `"mml"`.

**Step — the method-of-moments fallback.** When a level has too few informative cells to identify
`α_k`, the code matches the pooled Pearson overdispersion to its Dirichlet-multinomial
expectation and solves (derived in WS1 THEORY §5; `_dirichlet_mom`):

$$\alpha_k = \frac{\sum_c (N_c - 1)}{\,S/(K-1) - C\,} - 1, \qquad S = \sum_c \sum_a \frac{(n_{c,a} - N_c \pi_{c,a})^2}{N_c \pi_{c,a}},$$

with `C` the number of cells; a non-positive denominator (under-dispersion) maps to heavy pooling
`α_k = α_max`. Fewer than two informative cells returns a weak default.

**Step — read the fitted value as a verdict.** `α_k → ∞` (up to the ceiling `10^6`) means depth
`k` adds nothing: from §3, `w_k = N_k/(N_k + α_k) → 0`, so the depth-`k` mixture weight vanishes
and the estimate is its suffix. A moderate `α_k` means the depth's own counts win. On the null
world this reads the *true generative order* straight off: depths 1–2 fit moderate `α` (the
planted order-2 habit is real), depths 3–4 run to the ceiling (no order-3/4 signal exists).

**In plain terms.** To choose how hard to shrink each depth, we ask "what single strength makes
the whole level's counts most probable, given the shorter-context estimates as the prior?" and
hill-climb on a log axis. When that is unstable we fall back to a sturdier moment match on how
overdispersed the cells are. Either way the fitted number is a *readout*: huge means "this depth
is useless, pool it away," moderate means "this depth carries real ordered signal."

---

## 5. Effective order: a readout via the own-count weight

**Step — define it.** A row's **effective order** is the deepest depth `k` (≤ the available
priors and ≤ `K_MAX`) whose context `(pit, ctx_k)` was observed in training **and** whose
own-count weight

$$\boxed{\; \omega_k = \frac{N_k}{N_k + \alpha_k} \;\ge\; \tau \;}$$

for a threshold `τ` (default `EFFECTIVE_ORDER_TAU = 0.5`); rows with no qualifying depth have
effective order 0 (`effective_order` in `model.py`).

**Step — connect it to §3.** `ω_k` is exactly the own-count weight `w_k` of the depth-`k`
shrinkage (§3). So "effective order ≥ `k`" means the depth-`k` context's own data — not the
backoff prior — carries at least fraction `τ` of the posterior effective sample size `N_k + α_k`.
The threshold reads *where the interpolation weight of §3 crosses a line*.

**Step — why it is a readout, not a parameter.** Nothing here is chosen to hit a target: `N_k` is
the data's support for that exact context and `α_k` is the *fitted* concentration from §4. `ω_k`
is therefore fully determined once the model is fit; `τ` only sets the reporting line ("own data
must outweigh the prior"). A depth fitted to its ceiling has `ω_k → 0` and can never qualify, so
the effective-order distribution *cannot* claim more order than the fitted concentrations
support. Changing `τ` slides the line but does not add or remove signal.

**In plain terms.** The effective order answers "how many pitches back did the data actually let
us look, on this row?" — and it answers it using the same own-count weight that decides the blend
in §3. It is a thermometer reading the fitted model, not a dial we set: a depth the fit pooled
away simply never clears the bar.

---

## 6. Motif lift: the KL ranking and the max-probability-ratio readout

The motifs are the grammar's "rules": ordered contexts whose next-token distribution moves most
versus their suffix. Two quantities appear — one for **ranking**, one for the **readout** — and
the code uses each for a specific job.

**Step — pool over pitchers sharing the ordered family context.** A given family context (e.g.
`FF FF`) is keyed per pitcher in the fit. The exhibit aggregates them by support-weighted average
of the depth-`k` posterior means and of the suffix-parent means:

$$p_{\text{ctx}}(a) = \frac{\sum_{pit} N_{pit}\, g_k(a \mid ctx_k, pit)}{\sum_{pit} N_{pit}}, \qquad
p_{\text{suffix}}(a) = \frac{\sum_{pit} N_{pit}\, g_{k-1}(a \mid \operatorname{suffix}(ctx_k), pit)}{\sum_{pit} N_{pit}},$$

with `N_pit` the pitcher's training support for the context (`top_motifs` accumulates
`N · post_mean` and `N · parent_post_mean`, then divides by `ΣN`).

**Step — rank by KL divergence (bits).** Contexts are ranked by how much the whole next-token
distribution moved from the suffix to the full context:

$$\boxed{\; D_{\mathrm{KL}}(p_{\text{ctx}} \,\|\, p_{\text{suffix}}) = \sum_{a} p_{\text{ctx}}(a)\, \log_2 \frac{p_{\text{ctx}}(a)}{p_{\text{suffix}}(a)} \;}$$

(base-2, so KL is in bits; the code clips both distributions to `[1e-12, 1]` before the log).
KL is the natural ranking score because it captures the *entire* distributional move, not one
family.

**Step — the readout: the max-probability-change family's log-ratio.** For the reported line, the
code picks the single family that moved most in absolute probability,
$j = \arg\max_a |p_{\text{ctx}}(a) - p_{\text{suffix}}(a)|$, and reports its before/after
probabilities, its `log2` ratio

$$\ell_j = \log_2 \frac{p_{\text{ctx}}(j)}{p_{\text{suffix}}(j)},$$

and a direction: **`suppress`** if $p_{\text{ctx}}(j) < p_{\text{suffix}}(j)$ (the family became
*less* likely — e.g. a same-family run `X X → P(X)` dropping, the no-three-in-a-row rule),
**`promote`** if it rose. So: **KL ranks; the max-probability-change family's log-ratio is the
human-readable lift.** (This is the "max-prob-ratio alternative" the brief asks about, and it is
what the printed motif text carries; KL is the sort key.)

**In plain terms.** To find the strongest rules we ask which ordered context reshapes the
next-pitch distribution the most overall — that is the KL divergence from its shorter suffix. To
*describe* each rule in one line we then name the single family that moved most and by what
factor, and whether it went up (a setup that promotes a pitch) or down (a repeat that suppresses
it). The ranking uses the full move; the sentence uses the headline family.

---

## 7. `B_seq`: from a log-likelihood ratio to bits, and to a cross-entropy difference

**Step — the per-row bit.** For validation row `t` with realized next family `A_t`, the
ordered-view probability `q_O(A_t)` and the context-view probability `q_C(A_t)`, the per-row bits
of predictability are

$$\beta_t = \log_2 \frac{q_O(A_t \mid S_t)}{q_C(A_t \mid X_t)}$$

(`bits_of_predictability`, base-2, both probabilities floored at `1e-12`).

**Step — average to `B_seq`.**

$$\boxed{\; B_{\text{seq}} = \frac{1}{n}\sum_{t=1}^{n} \log_2 \frac{q_O(A_t)}{q_C(A_t)} \;}$$

**Step — rewrite as a cross-entropy difference.** Split the log of the ratio and recognise each
average as a (negative) empirical cross-entropy $\mathrm{CE}(q) = -\frac1n\sum_t \log_2 q(A_t)$:

$$B_{\text{seq}} = \frac1n\sum_t \log_2 q_O(A_t) - \frac1n\sum_t \log_2 q_C(A_t)
= \big[-\mathrm{CE}(q_O)\big] - \big[-\mathrm{CE}(q_C)\big] = \mathrm{CE}(q_C) - \mathrm{CE}(q_O).$$

So `B_seq` is the **reduction in next-pitch cross-entropy** (in bits) from adding ordered context
— positive means `q_O` forecasts the realized pitch better than `q_C`.

**Step — relate to the log loss.** The harness log loss uses natural logs,
$\mathrm{Loss}(q) = -\frac1n\sum_t \ln q(A_t) = \ln 2 \cdot \mathrm{CE}(q)$. Therefore

$$B_{\text{seq}} = \frac{\mathrm{Loss}(C) - \mathrm{Loss}(O)}{\ln 2}\ \text{(bits)}.$$

Note this is the **C-vs-O** gap, distinct from the ordered-selection edge
`delta_order_L1 = Loss(L1) − Loss(O)` (the **L1-vs-O** gap, reported in nats). `B_seq` measures
all the bits ordered history adds over *no* history; `delta_order_L1` measures only what order
adds over the *previous pitch*.

**Step — interpret 0.012 bits in plain terms.** One full bit is one yes/no question answered.
`B_seq = 0.012` bits/pitch is about **one-eightieth of a yes/no question** of next-pitch
information per pitch; over a ~4-pitch PA, ~0.05 bits, ~1/20 of a yes/no question. Equivalently,
the next-pitch *perplexity* (effective number of equally likely choices) shrinks by a factor
$2^{-0.012} \approx 0.992$ — under 1%. It is real and correctly directional, but small: the
ordered sequence sharpens the next-pitch guess by a sliver, which is exactly SPEC §13's honest
expectation.

**In plain terms.** `B_seq` is "how many yes/no questions' worth of next-pitch information does
knowing the sequence buy you, per pitch, over knowing only the situation?" It is the same as the
drop in cross-entropy (or, up to the `ln 2` unit change, the C-minus-O log-loss gap). Twelve
thousandths of a bit is a real but tiny sharpening — a fraction of one yes/no question.

---

## 8. Why `delta_order_L1` lacks D21's min-bias (and what bias can remain)

**Step — recall the two statistics.** SPEC §6's ablation is
$\Delta_{\text{order}} = \mathrm{Loss}(\min[U, L1]) - \mathrm{Loss}(O)$. WS2's edge is
$\texttt{delta\_order\_L1} = \mathrm{Loss}(L1) - \mathrm{Loss}(O)$. Both use held-out losses,
which are random over the finite validation set.

**Step — the source of D21's bias (a `min` of noisy losses).** The function `min(a, b)` is
**concave**, so by Jensen's inequality
$\mathbb{E}[\min(\hat L_U, \hat L_{L1})] \le \min(\mathbb{E}\hat L_U, \mathbb{E}\hat L_{L1})$.
Under the null the three views share one true loss `L*`, so subtracting the (unbiased) `O` loss,

$$\mathbb{E}[\Delta_{\text{order}}] = \mathbb{E}[\min(\hat L_U, \hat L_{L1})] - L^\star \le 0,$$

a **negative** bias whose size grows with how noisy and different `L̂_U`, `L̂_L1` are (decision
D21). Taking the *better* of two noisy scores flatters the baseline.

**Step — why `delta_order_L1` has no such term.** There is no minimum. It is a straight
difference of two held-out loss estimates,
$\texttt{delta\_order\_L1} = \hat L_{L1} - \hat L_O$, each an **unbiased** estimate of its view's
expected loss. Hence

$$\boxed{\; \mathbb{E}[\texttt{delta\_order\_L1}] = \mathbb{E}[\hat L_{L1}] - \mathbb{E}[\hat L_O] = L_{L1} - L_O \;}$$

with **no Jensen gap**. Under the null of no ordered edge beyond L1 (`L_{L1} = L_O`) the
expectation is exactly 0 — *unbiased*, not negatively biased — so the clustered CI is read
directly: a lower bound above 0 is a genuine edge.

**Step — what bias can still remain (honest sketch).** Three residual effects, none of which
manufactures a false positive:

1. **Shared evaluation noise (helps, not hurts).** `L̂_L1` and `L̂_O` are computed on the *same*
   validation rows, so the difference is a **paired** statistic. The two losses are strongly
   positively correlated (they share the base `b`, the pitcher, and the previous pitch), so the
   *variance* of the difference is much smaller than that of either loss — the paired CI is
   tighter. This affects precision, not the point estimate; it is a feature.
2. **Nesting / extra flexibility (mildly conservative).** `O` nests `L1` (with `α_k → ∞` for
   `k ≥ 2`, `O` collapses to `L1`), but its concentrations are *estimated*, so under the null `O`
   carries slightly more estimation variance in its fitted predictions than `L1`. That nudges
   `E[L̂_O]` *up* a hair, so `E[delta_order_L1]` is if anything **slightly negative** under the
   null — the opposite of D21's optimism. A positive result is therefore understated, never
   inflated; "order hurts" is never the right reading of a small negative.
3. **Clustered dependence (a coverage caveat).** Pitches within a pitcher-game are dependent; the
   clustered bootstrap targets this, but with few clusters the CI can under-cover. This is a
   caveat about the *interval's* calibration, not a bias in the point estimate.

**In plain terms.** D21's warning is about taking the *minimum* of two lucky-or-unlucky scores,
which flatters the baseline and tilts the ablation negative even when nothing is there. WS2 never
takes a minimum — it just subtracts O's loss from L1's — so there is no built-in optimism, and
its interval means what it says. The only residual tilt is mildly *conservative* (O pays a small
price for its extra knobs), so a positive lower bound is a clean, if understated, ordered-selection
claim; the tight interval comes from scoring both views on the same pitches.

---

## 9. The permutation control: what is exchangeable under "no ordered selection"

**Step — state the null for a selection grammar.** The negative-control null is that the next
family is conditionally independent of the ordered within-PA history given the coarse
conditioning:

$$H_0:\quad A_t \;\perp\; H_t \;\big|\; (ch,\ \text{pitch\_number}),$$

where `H_t` is the full ordered history vector. Under `H_0`, the ordered context carries no
information the count × hand cell and the PA depth do not already carry.

**Step — identify what is exchangeable.** If `H_0` holds, then within a stratum that fixes
`(ch, pitch_number)` the history vectors are **exchangeable across rows**: reassigning row `i`'s
history to row `i'` leaves the joint distribution of `(A, ch, pitch_number, H)` unchanged, because
`A` does not depend on `H` given the stratum. This is exactly the exchangeability a permutation
test needs.

**Step — the operation (matches `permute_contexts_within_strata`).** Group rows by
`(ch, pitch_number)`; within each group, permute the whole lagged-context vectors among rows while
each row keeps its own target `A_t`. Fixing `pitch_number` fixes the number of real priors, so a
permuted context has the **right depth** (never a "no-prior" contradiction). The operation
**preserves** the count × hand marginal mix, the PA-depth distribution, and every target; it
**destroys** the association between the ordered history and the target.

**Step — what a collapse proves.** Refit `L1` and `O` on the permuted contexts and re-evaluate. A
*genuine* ordered edge — dependence `H_0` denies — cannot survive the scramble, so
`delta_order_L1` and the effective order must fall to the base (validated: edge +0.0063 → +0.0001,
order ≥ 2 mass 0.296 → 0.000). A *spurious* edge that was really the count × hand × depth marginal
in disguise would **survive**, because those marginals are preserved — so the collapse isolates
ordered dependence from base-rate structure. This is the negative control that turns "we detected
order" into "we detected order that is not the base rate re-labeled."

**Step — an honest caveat on the stratification.** The stratum is `(ch, pitch_number)` — it does
**not** fix the pitcher or the prior-family multiset. So the permutation also breaks the
pitcher ↔ history and multiset ↔ target links: it is a *stronger* scramble than the finest SPEC
§8.3 stratification (pitcher × count × pitch-number × handedness × prior-multiset). A collapse
therefore confirms the broad null "no within-PA history dependence survives once
count × hand × depth is fixed," which is *more* than "no order beyond L1." The narrower ordered
claim — order beyond the *previous pitch specifically* — rests on the `delta_order_L1` CI (§8),
which compares `L1` and `O` directly, both retaining the pitcher and the previous pitch. The two
controls are complementary: the permutation rules out base-rate mimicry broadly; the CI pins the
edge to *order beyond L1* precisely.

**In plain terms.** To check the grammar is not fooling itself, we shuffle the pitch histories
among situations that share the same count, handedness, and pitch number — keeping each pitch's
actual next-family, but pairing it with someone else's history. If the "order effect" was real it
vanishes (you have destroyed the order); if it was just the count-and-depth base rate wearing an
order costume, it stays. On the planted world the effect vanishes cleanly, which is exactly what a
true order effect must do. The one caveat is that this shuffle also scrambles pitcher and
multiset, so it tests a slightly broader "no history dependence" than "no order beyond the last
pitch" — for that precise claim we lean on the direct L1-versus-O interval of §8.
