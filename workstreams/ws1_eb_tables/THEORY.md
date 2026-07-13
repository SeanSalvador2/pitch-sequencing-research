# WS1 Theory: Every Formula, Derived

This document derives every formula the WS1 empirical-Bayes tables use, one named step at a
time, with a plain-words explanation after each block. Nothing is skipped. The boxed results
match `PAPER.md` and the code in `model.py` exactly; where the code names a function, it is
noted. Notation is introduced once in §1 and reused throughout.

---

## 1. Setup and notation

We fit conditional tables over the shared decision table (SPEC §3). Fix a **target** (either
`selection` or `run_value`) and a **state view** (one of `C`, `U`, `L1`, `O`). The view
determines a **conditioning key**; the set of rows sharing a key value is a **cell**.

- $K = 8$ — the number of pitch families ($\texttt{FF, SI, FC, SL, CU, CH, FS, XX}$).
- A cell $c$ collects the training rows whose key equals $c$.
- For **selection**: $n_c = (n_{c,1}, \dots, n_{c,K})$ is the vector of family counts in cell
  $c$, and $N_c = \sum_{k=1}^{K} n_{c,k}$ is the total count.
- For **run value**: $n_c$ is the number of observations in cell $c$, $\bar R_c$ their sample
  mean reward, and $\sigma^2$ a pooled within-cell reward variance.
- The cells are arranged in a **hierarchy of levels**. For selection the levels deepen as
  $() \to (ch) \to (ch, pit) \to (ch, pit, \text{hist})$ where $ch$ is the count × handedness
  code and $pit$ the pitcher; for run value each level additionally carries the current family
  (decision D22). Each cell $c$ has a **parent** $\text{parent}(c)$ one level up, whose fitted
  posterior mean we write $\pi_{\text{parent}(c)}$ (a distribution over families, selection) or
  $m_{\text{parent}(c)}$ (a scalar mean, run value).
- Level $l$ has a single shared **concentration** $\alpha_l$ (selection) or **shrinkage
  strength** $\kappa_l$ (run value), fit by empirical Bayes (§§4–5, §8).

**In plain terms.** A "cell" is one row of a lookup table — a specific game situation, at some
depth of detail. A shallow cell (just the count) has lots of data; a deep cell (count and
pitcher and the last two pitches) has little. Every deep cell has a shallower "parent" it can
lean on. The whole method is: estimate each cell by blending its own data with its parent's
estimate, and *learn* how much to blend.

---

## 2. Dirichlet–multinomial conjugacy → the posterior

We model the family counts in a cell as multinomial with an unknown family-probability vector
$p = (p_1, \dots, p_K)$, and place a Dirichlet prior on $p$ centered on the parent.

**Step — write the likelihood (multinomial).** Given $p$, the probability of the observed counts
$n_c$ is

$$P(n_c \mid p) = \binom{N_c}{n_{c,1}, \dots, n_{c,K}} \prod_{k=1}^{K} p_k^{\,n_{c,k}} \;\propto\; \prod_{k=1}^{K} p_k^{\,n_{c,k}}.$$

**Step — write the prior (Dirichlet centered on the parent).** With concentration $\alpha_l$ and
parent mean $\pi \equiv \pi_{\text{parent}(c)}$ (so the prior pseudo-counts are $\alpha_l \pi_k$),

$$p \sim \text{Dir}(\alpha_l \pi_1, \dots, \alpha_l \pi_K), \qquad P(p) = \frac{1}{B(\alpha_l \pi)} \prod_{k=1}^{K} p_k^{\,\alpha_l \pi_k - 1},$$

where $B(\cdot)$ is the multivariate Beta (normalizing) function.

**Step — apply Bayes' rule (posterior $\propto$ prior $\times$ likelihood).**

$$P(p \mid n_c) \;\propto\; \prod_{k=1}^{K} p_k^{\,\alpha_l \pi_k - 1} \cdot \prod_{k=1}^{K} p_k^{\,n_{c,k}} = \prod_{k=1}^{K} p_k^{\,(\alpha_l \pi_k + n_{c,k}) - 1}.$$

**Step — recognize conjugacy (same functional form).** The right-hand side is the kernel of a
Dirichlet, so the posterior is Dirichlet with updated parameters:

$$p \mid n_c \sim \text{Dir}\big(\alpha_l \pi_1 + n_{c,1}, \dots, \alpha_l \pi_K + n_{c,K}\big).$$

**Step — take the posterior mean (mean of a Dirichlet).** The mean of $\text{Dir}(a_1, \dots,
a_K)$ is $a_k / \sum_j a_j$. Here $\sum_j (\alpha_l \pi_j + n_{c,j}) = \alpha_l + N_c$ because
$\sum_j \pi_j = 1$. Therefore

$$\boxed{\;\hat p_{c,k} = \frac{n_{c,k} + \alpha_l\, \pi_k}{N_c + \alpha_l}\;}$$

which in vector form is `dirichlet_posterior_mean(counts, parent, alpha)` in `model.py`.

**In plain terms.** The Dirichlet is the natural "prior over a probability vector," and it is
*conjugate* to the multinomial: after seeing counts, you just add them to the prior pseudo-counts
and you are back to a Dirichlet. The posterior mean is then the total pseudo-count in each family
divided by the grand total — the parent's guess, "topped up" by the data actually seen.

---

## 3. Rearranging the posterior mean into shrinkage form

We rewrite the boxed posterior mean to expose it as a weighted average of the data and the parent.

**Step — split the numerator.** Separate the two additive pieces:

$$\hat p_{c,k} = \frac{n_{c,k}}{N_c + \alpha_l} + \frac{\alpha_l\, \pi_k}{N_c + \alpha_l}.$$

**Step — multiply the first term by $1 = N_c / N_c$ (introduce the MLE).** The maximum-likelihood
estimate is $\hat p^{\text{MLE}}_{c,k} = n_{c,k}/N_c$, so

$$\frac{n_{c,k}}{N_c + \alpha_l} = \frac{N_c}{N_c + \alpha_l} \cdot \frac{n_{c,k}}{N_c} = \frac{N_c}{N_c + \alpha_l}\, \hat p^{\text{MLE}}_{c,k}.$$

**Step — collect the two weights.** Defining $w_c = N_c/(N_c + \alpha_l)$,

$$\boxed{\;\hat p_c = w_c\, \hat p^{\text{MLE}}_c + (1 - w_c)\, \pi_{\text{parent}(c)}, \qquad w_c = \frac{N_c}{N_c + \alpha_l}\;}$$

since $1 - w_c = \alpha_l/(N_c + \alpha_l)$.

**Step — read off the two limits.** As $N_c \to \infty$, $w_c \to 1$ (trust the data). As
$N_c \to 0$, $w_c \to 0$ (return the parent exactly — this is §6, backoff).

**Step — posterior spread (for the reported variance).** The variance of a Dirichlet component
with total concentration $A = N_c + \alpha_l$ is $\hat p_{c,k}(1 - \hat p_{c,k})/(A + 1)$, giving
the reported

$$\text{Var}(p_{c,k}) = \frac{\hat p_{c,k}(1 - \hat p_{c,k})}{N_c + \alpha_l + 1}, \qquad \text{ESS}_c = N_c + \alpha_l.$$

**In plain terms.** This is the batting-average story in symbols. The estimate is a slider between
"what this cell actually did" and "what its parent expects," and the slider position $w_c$ depends
only on how much data the cell has relative to the concentration $\alpha_l$. A lot of data pushes
the slider to the cell; little data pushes it to the parent. The effective sample size $N_c +
\alpha_l$ is "real observations plus imaginary prior ones," and the posterior variance shrinks as
that grows.

---

## 4. The Pólya (Dirichlet–multinomial) marginal likelihood

To *fit* $\alpha_l$ we need the probability of a cell's counts with $p$ integrated out — the
marginal (a.k.a. evidence, or Pólya distribution).

**Step — write the marginal as an integral (integrate out $p$).**

$$P(n_c \mid \alpha_l, \pi) = \int_{\Delta} P(n_c \mid p)\, P(p \mid \alpha_l, \pi)\, dp = \binom{N_c}{n_c} \int_{\Delta} \frac{1}{B(\alpha_l \pi)} \prod_{k=1}^{K} p_k^{\,\alpha_l \pi_k + n_{c,k} - 1}\, dp.$$

**Step — use the Dirichlet integral (the integral of an unnormalized Dirichlet is its Beta
constant).** For any positive $a$, $\int_{\Delta} \prod_k p_k^{a_k - 1}\, dp = B(a) =
\frac{\prod_k \Gamma(a_k)}{\Gamma(\sum_k a_k)}$. Applying it with $a_k = \alpha_l \pi_k + n_{c,k}$:

$$P(n_c \mid \alpha_l, \pi) = \binom{N_c}{n_c}\, \frac{B(\alpha_l \pi + n_c)}{B(\alpha_l \pi)}.$$

**Step — expand the Beta functions into Gammas.** Using $B(a) = \prod_k \Gamma(a_k) / \Gamma(\sum
a_k)$ for both, and $\sum_k \alpha_l \pi_k = \alpha_l$, $\sum_k (\alpha_l \pi_k + n_{c,k}) =
\alpha_l + N_c$:

$$P(n_c \mid \alpha_l, \pi) = \binom{N_c}{n_c}\, \frac{\Gamma(\alpha_l)}{\Gamma(\alpha_l + N_c)} \prod_{k=1}^{K} \frac{\Gamma(\alpha_l \pi_k + n_{c,k})}{\Gamma(\alpha_l \pi_k)}.$$

**Step — take logs and drop the $\alpha$-free term.** The multinomial coefficient does not depend
on $\alpha_l$, so for optimization it is a constant. Summing the log over all cells at the level
gives the objective actually coded (`_dirichlet_neg_log_evidence`, negated because the code
minimizes):

$$\boxed{\;\log P(\{n_c\} \mid \alpha_l) \;\overset{c}{=}\; \sum_c \Big[\log \Gamma(\alpha_l) - \log \Gamma(\alpha_l + N_c) + \sum_{k=1}^{K} \big(\log \Gamma(\alpha_l \pi_{c,k} + n_{c,k}) - \log \Gamma(\alpha_l \pi_{c,k})\big)\Big]\;}$$

where $\overset{c}{=}$ means "equal up to an additive constant in $\alpha_l$."

**In plain terms.** If we do not commit to a single $p$ but average over all of them weighted by
the prior, a cell's counts follow the Pólya distribution — a multinomial with "memory," more
overdispersed than a plain multinomial. Its log-probability is a tidy ratio of Gamma functions.
Summed over all cells at a level, it is a smooth function of the one number $\alpha_l$, which is
exactly what we maximize next.

---

## 5. Fitting $\alpha_l$: maximum marginal likelihood and the method-of-moments fallback

### 5.1 Maximum marginal likelihood (MML), the primary estimator

**Step — state the objective.** Maximize the level-summed log evidence of §4 over the single
scalar $\alpha_l > 0$ (equivalently, minimize its negative).

**Step — why it is 1-D.** All exchangeable cells at a level share one concentration $\alpha_l$
(the parent means $\pi_{c}$ are fixed from the level above), so the entire level is one
one-dimensional optimization — cheap and robust, no multi-parameter surface to search.

**Step — why $\log_{10}$ scale.** $\alpha_l$ ranges over many orders of magnitude (from $\approx
10^{-3}$, heterogeneous cells, to the ceiling $10^6$, "this level adds nothing"). Optimizing
$\log_{10} \alpha_l$ on a bounded interval $[\,-3, \log_{10}(\alpha_{\max})\,]$ makes the search
well-conditioned; the code uses `scipy.optimize.minimize_scalar(method="bounded")` and reports
`"mml"` on success.

**Step — identifiability guard.** A level with fewer than two informative cells cannot identify a
shared concentration, so the code returns a weak default there (`"default"`).

### 5.2 Method-of-moments (MoM) fallback: Pearson-overdispersion matching

When the optimizer fails or the level is too sparse, we match the pooled Pearson statistic to its
expectation and solve for $\alpha_l$.

**Step — define the per-cell Pearson statistic.** For cell $c$ with expected counts
$E_{c,k} = N_c \pi_{c,k}$,

$$X_c = \sum_{k=1}^{K} \frac{(n_{c,k} - N_c \pi_{c,k})^2}{N_c \pi_{c,k}}.$$

**Step — state its expectation under the Dirichlet–multinomial (the overdispersion factor).** A
plain multinomial has $\mathbb{E}[X_c] = K - 1$. The Dirichlet–multinomial inflates this by the
overdispersion factor $1 + (N_c - 1)/(\alpha_l + 1)$:

$$\mathbb{E}[X_c] = (K - 1)\left(1 + \frac{N_c - 1}{\alpha_l + 1}\right).$$

**Step — pool over cells.** Summing $S = \sum_c X_c$ and matching to the summed expectation with
$C$ cells:

$$S = (K - 1)\left(C + \frac{\sum_c (N_c - 1)}{\alpha_l + 1}\right).$$

**Step — solve for $\alpha_l$ (rearrange).** Divide by $(K-1)$, isolate the fraction, and invert:

$$\frac{S}{K - 1} - C = \frac{\sum_c (N_c - 1)}{\alpha_l + 1} \;\;\Longrightarrow\;\; \boxed{\;\alpha_l = \frac{\sum_c (N_c - 1)}{\,S/(K-1) - C\,} - 1\;}$$

matching `_dirichlet_mom`. A non-positive denominator means under-dispersion (cells look *more*
uniform than multinomial noise), which maps to heavy pooling $\alpha_l = \alpha_{\max}$.

**In plain terms.** MML asks "what single blend strength makes the whole level's counts most
probable?" and answers with one number by hill-climbing on a log axis. When that is unstable, MoM
asks a cruder but sturdier question: "how much more spread out are the cells than pure multinomial
noise?" — because more spread means the cells are really different (small $\alpha$), and no extra
spread means they are interchangeable (large $\alpha$). Solving that spread equation gives the same
kind of answer without an optimizer.

---

## 6. Hierarchy and backoff as the zero-count limit

**Step — instantiate the shrinkage form at $N_c = 0$.** From §3, $w_c = N_c/(N_c + \alpha_l)$, so
$N_c = 0 \Rightarrow w_c = 0$ and

$$\hat p_c \big|_{N_c = 0} = 0 \cdot \hat p^{\text{MLE}}_c + 1 \cdot \pi_{\text{parent}(c)} = \pi_{\text{parent}(c)}.$$

**Step — recurse up the levels.** The parent is itself a shrinkage estimate with its own parent, so
an unseen cell resolves to the posterior mean of its **deepest observed ancestor**. In code,
`_assign` walks levels deepest-first and stops at the first key it has seen; `backoff_counts_`
records how many prediction rows resolved at each level.

**Step — note the run-value analogue.** The identical limit holds for §7's normal pooling:
$n_c = 0 \Rightarrow \hat m_c = m_{\text{parent}(c)}$.

**In plain terms.** "Backoff" is not a separate rule bolted on — it is what the shrinkage formula
*already does* when a cell has no data: the data weight goes to zero and the estimate is exactly the
parent's. Deep keys the table never saw at training time cost nothing at prediction time; they
simply fall back to the most specific situation the table *did* see.

---

## 7. Normal–normal partial pooling (run value)

Now the target is a mean reward, not a distribution. The model is: cell mean $\mu_c$ drawn around
the parent mean, observations drawn around $\mu_c$.

**Step — write the two-level Gaussian model.** With between-cell variance $\tau^2$ and within-cell
variance $\sigma^2$,

$$\mu_c \sim \mathcal{N}(m_{\text{parent}(c)},\, \tau^2), \qquad \bar R_c \mid \mu_c \sim \mathcal{N}\!\left(\mu_c,\, \frac{\sigma^2}{n_c}\right).$$

**Step — apply Bayes for a Gaussian mean (precision-weighted update).** For conjugate normals the
posterior mean is the precision-weighted average of prior mean and data, where precision $=
1/\text{variance}$. The prior precision is $1/\tau^2$; the data precision is $n_c/\sigma^2$:

$$\hat m_c = \frac{\frac{1}{\tau^2}\, m_{\text{parent}(c)} + \frac{n_c}{\sigma^2}\, \bar R_c}{\frac{1}{\tau^2} + \frac{n_c}{\sigma^2}}.$$

**Step — reparameterize with $\kappa_l = \sigma^2/\tau^2$ (a prior sample size).** Multiply
numerator and denominator by $\sigma^2$: the prior-precision term becomes $\sigma^2/\tau^2 =
\kappa_l$ and the data term becomes $n_c$. Hence

$$\hat m_c = \frac{\kappa_l\, m_{\text{parent}(c)} + n_c\, \bar R_c}{\kappa_l + n_c}.$$

**Step — write in shrinkage form.** Splitting the fraction,

$$\boxed{\;\hat m_c = \frac{n_c}{n_c + \kappa_l}\, \bar R_c + \frac{\kappa_l}{n_c + \kappa_l}\, m_{\text{parent}(c)}\;}$$

which is `normal_posterior_mean(n, cell_mean, parent_mean, kappa)`. So $\kappa_l$ is literally "how
many prior observations' worth of pull toward the parent."

**Step — posterior standard deviation.** The posterior precision is $1/\tau^2 + n_c/\sigma^2 =
(\kappa_l + n_c)/\sigma^2$, so the posterior variance is $\sigma^2/(n_c + \kappa_l)$ and

$$\boxed{\;\text{sd}(\hat m_c) = \frac{\sigma}{\sqrt{n_c + \kappa_l}}\;}$$

with $\sigma^2$ the pooled within-cell residual variance (`_pooled_sigma2`, a count-weighted average
of within-`(count_hand, family)` variances).

**In plain terms.** Averaging two noisy beliefs optimally means weighting each by how sure you are
of it (its precision) and renormalizing. The parent counts as $\kappa_l$ imaginary observations; the
cell brings $n_c$ real ones; the estimate is their pooled average, and its error bar shrinks like one
over the square root of the total. Same slider as selection, now for a mean.

---

## 8. Between-cell variance: a DerSimonian–Laird moment estimate (sketch)

We still need $\tau^2$ (equivalently $\kappa_l = \sigma^2/\tau^2$). MML on the normal evidence is
primary; the moment fallback is a DerSimonian–Laird–style estimate. *This is a derivation sketch,
honestly labeled — it states the moment identity and the estimator, not a full optimality proof.*

**Step — the marginal of a cell mean (integrate out $\mu_c$).** Adding the two Gaussian stages,

$$\bar R_c \sim \mathcal{N}\!\left(m_{\text{parent}(c)},\; \tau^2 + \frac{\sigma^2}{n_c}\right).$$

This gives the normal evidence used by MML (`_normal_neg_log_likelihood`):
$\tfrac12 \sum_c \big[\log(2\pi v_c) + (\bar R_c - m_{\text{parent}(c)})^2 / v_c\big]$ with
$v_c = \tau^2 + \sigma^2/n_c$.

**Step — take the moment of the squared deviation.** From the marginal,

$$\mathbb{E}\big[(\bar R_c - m_{\text{parent}(c)})^2\big] = \tau^2 + \frac{\sigma^2}{n_c}.$$

**Step — match the empirical mean deviation (the DL moment match).** Averaging the observed squared
deviations across cells and subtracting the known within-cell part gives the estimator

$$\hat\tau^2 = \frac{1}{C}\sum_c \left[(\bar R_c - m_{\text{parent}(c)})^2 - \frac{\sigma^2}{n_c}\right],$$

floored at a small positive value (a negative moment estimate means no detectable between-cell
spread).

**Step — convert to a shrinkage strength.** Then $\boxed{\;\kappa_l = \sigma^2 / \hat\tau^2\;}$
(`_normal_mom`); a floored-to-zero $\hat\tau^2$ maps to $\kappa_l = \kappa_{\max}$ (pool hard).

**In plain terms.** The observed scatter of cell means around their parent has two ingredients:
genuine differences between cells ($\tau^2$) and ordinary sampling noise ($\sigma^2/n_c$). Subtract
the sampling part you can compute, and what remains is an estimate of the genuine part. If nothing
remains, the cells are not really different and you pool them hard. This is a moment match, not a
likelihood optimum — hence "sketch."

---

## 9. Bias–variance of shrinkage: when it wins

Why shrink at all? Decompose the mean squared error of estimating a cell's true mean $\mu_c$ and
watch the trade.

**Step — write the two candidate estimators.** The unshrunk estimate is $\bar R_c$ (unbiased,
variance $\sigma^2/n_c$). The shrunk estimate is $\hat m_c = w\bar R_c + (1-w)m$ with $w =
n_c/(n_c+\kappa_l)$ and $m = m_{\text{parent}(c)}$.

**Step — bias of the shrunk estimator.** Since $\mathbb{E}[\bar R_c] = \mu_c$,

$$\text{Bias}(\hat m_c) = \mathbb{E}[\hat m_c] - \mu_c = (1 - w)(m - \mu_c).$$

Shrinking introduces a bias proportional to how far the cell's truth sits from the parent, scaled by
the shrink weight $1-w$.

**Step — variance of the shrunk estimator.** Only the $\bar R_c$ term is random:

$$\text{Var}(\hat m_c) = w^2\, \frac{\sigma^2}{n_c}.$$

Shrinking multiplies the raw variance by $w^2 < 1$.

**Step — assemble the MSE.**

$$\text{MSE}(\hat m_c) = w^2\frac{\sigma^2}{n_c} + (1 - w)^2 (m - \mu_c)^2.$$

**Step — the condition for shrinkage to help.** Compare to the unshrunk $\text{MSE}(\bar R_c) =
\sigma^2/n_c$. Shrinking a little from $w = 1$ lowers the MSE whenever the marginal variance
reduction beats the marginal bias increase; carrying out the first-order condition (or minimizing
the quadratic in $w$) gives the optimal weight

$$w^\star = \frac{(m - \mu_c)^2}{(m - \mu_c)^2 + \sigma^2/n_c}, \qquad \text{equivalently } \kappa^\star = \frac{\sigma^2}{(m-\mu_c)^2}.$$

So shrinkage wins most when $\sigma^2/n_c$ is large (little, noisy data) and when $(m - \mu_c)^2$ is
small (the parent is a good guess) — exactly the thin-cell regime. Empirical Bayes (§§5, 8) estimates
$\kappa^\star$ by pooling $(m - \mu_c)^2$ across the level rather than knowing it per cell.

**In plain terms.** Trusting a thin cell's own average is unbiased but wildly variable; pulling it
toward the parent trades a little bias for a large variance cut, and the net error drops. The dial
that minimizes total error pulls harder exactly when data is scarce and the parent is trustworthy —
which is what the fitted concentration is trying to find automatically.

---

## 10. Why $\Delta_{\text{order}}$ is negatively biased under the null

Recall the ablation (SPEC §6): $\Delta_{\text{order}} = \text{Loss}(\min[U, L1]) - \text{Loss}(O)$,
with the loss estimated on held-out data. We show it is biased below zero under the null, and why
WS1's version is *exactly* zero. *This is a formal sketch.*

**Step — set up the null.** Under the null, ordered history carries no information the parent does
not, so the three views' true expected losses are equal: $\mathbb{E}[\hat L_U] = \mathbb{E}[\hat
L_{L1}] = \mathbb{E}[\hat L_O] = L^\star$. The held-out loss estimates $\hat L_U, \hat L_{L1}$ are
random (finite validation set).

**Step — apply Jensen to the minimum (min is concave).** The function $\min(a, b)$ is concave, so by
Jensen's inequality

$$\mathbb{E}\big[\min(\hat L_U, \hat L_{L1})\big] \le \min\big(\mathbb{E}[\hat L_U], \mathbb{E}[\hat L_{L1}]\big) = L^\star.$$

**Step — subtract the (unbiased) $O$ loss.** With $\mathbb{E}[\hat L_O] = L^\star$,

$$\mathbb{E}[\Delta_{\text{order}}] = \mathbb{E}\big[\min(\hat L_U, \hat L_{L1})\big] - \mathbb{E}[\hat L_O] \le L^\star - L^\star = 0.$$

$$\boxed{\;\mathbb{E}[\Delta_{\text{order}}] \le 0 \quad \text{under the null.}\;}$$

**Step — size the bias.** The gap in Jensen's inequality is roughly $\tfrac12 \mathbb{E}|\hat L_U -
\hat L_{L1}|$: the noisier and more *different* the two history-lite loss estimates, the more
negative the expected $\Delta_{\text{order}}$. This is why the null criterion (decision **D21**) is
"not significantly positive," and why a small negative reads as *consistent with no ordering effect*,
never as "order hurts."

**Step — the WS1-specific collapse.** WS1's tables pool the history level away under the null: the
fitted history concentration $\alpha \to \alpha_{\max}$, so from §3 the history-level shrink weight
$w_c \to 0$ and every history cell returns its pitcher × count parent. Then $U$, $L1$, and $O$ make
**identical** predictions, so $\hat L_U = \hat L_{L1} = \hat L_O$ pointwise and

$$\Delta_{\text{order}} = \min(\hat L_U, \hat L_{L1}) - \hat L_O = 0 \quad \textbf{exactly.}$$

There is no Jensen gap because there is no variability across the views — which is why, for WS1
specifically, a *clearly negative* $\Delta_{\text{order}}$ on real data cannot be dismissed as the
optimism bias and instead diagnoses over-fragmentation (`PAPER.md` §6, branch R3).

**In plain terms.** Taking the *better* of two noisy scores flatters you: even when neither view is
truly better, the luckier of the two beats the honest baseline on average, so the difference tilts
negative. That is why we never celebrate a small negative — it is the statistic's built-in optimism,
not a real cost of order. WS1 has an extra twist: because its flat tables literally erase the history
level when it is useless, its three views become the same table and the difference is a clean zero.
When WS1's number is instead clearly negative, the erasure did not save it — the key simply
fragmented the data too far, and that is a real, diagnosable failure of tabular estimation, not of
baseball.
