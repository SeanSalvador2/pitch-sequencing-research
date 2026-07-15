# WS6 Theory: The Capacity-Matched GRU Ablation, Derived

This document derives every formula the WS6 deep-sequence stack uses, one named step at a time,
with a plain-words explanation after each block. Nothing is skipped. The boxed results match
`PAPER.md` and the code in `model.py` exactly; where the code names a function, a class, or a
constant, it is noted. WS6 shares the study's ablation and cluster-bootstrap machinery with
WS1/WS2/WS3, so §5 recaps the shared `Δ_order` min-bias lemma (derived in
`../ws1_eb_tables/THEORY.md` §10, reused in `../ws3_gbdt_stack/THEORY.md` §6) and the *absence* of
that bias in `Δ_matchup`, then derives the part that is new to WS6 — the **targeting arithmetic**
that explains why an effect binding on a subset is detected on that subset but diluted in the
aggregate. Notation is introduced once in §1 and reused throughout.

WS6's whole reason to exist is a representation question: does a *learned* encoding of the ordered
plate-appearance sequence beat the **engineered** tabular history (WS3, `../ws3_gbdt_stack/THEORY.md`
§3) and the **explicit** variable-order grammar (WS2)? To make that an honest measurement of
*information* rather than *capacity*, the five nested views `C / U / L1 / O / OM` (SPEC §6) are
realised as **capacity-matched neural architectures** (decision D45). §§2–4 derive why each view is
the deep analogue of its information rung; §4 states precisely what "capacity-matched" controls and
what it cannot.

---

## 1. Setup, notation, and the GRU cell derived gate by gate

### 1.1 Notation

WS6 fits, **per state view** (all five: `C`, `U`, `L1`, `O`, `OM`) and **per target**
(`selection`, `outcome1`), one capacity-matched network (`_SeqNet`) trained through `SeqModel`.

- `s` — the **state**, one decision row (SPEC §3). Its pieces: the audited static **C** context
  `x^C` (`ContextEncoder`), the ordered within-PA history tokens (below), and — for `OM` only —
  the matchup block `x^M` (`MatchupEncoder`).
- The **ordered history** of a decision row is the sequence of its *prior* pitches this PA,
  oldest-first, `h_{1:T}` with `T = ` `lengths` `≤ max_len = 15` (`build_sequences`, SPEC §3.3).
  Each history position `j` carries a token
  $$x_j = \big[\, E_{\text{fam}}(f_j)\; \Vert\; E_{\text{out}}(o_j)\; \Vert\; c_j \,\big] \in \mathbb{R}^{d_{\text{tok}}},
    \qquad d_{\text{tok}} = 2 d_{\text{emb}} + n_{\text{ch}} = 2(16) + 7 = 39,$$
  where `E_fam`, `E_out` are the family / outcome-token embeddings
  (`nn.Embedding(N_FAM+1, 16, padding_idx=FAM_PAD)` and `nn.Embedding(N_O1+1, 16, padding_idx=O1_PAD)`),
  and `c_j ∈ ℝ⁷` are the standardised float channels
  `SEQUENCE_FLOAT_CHANNELS = (plate_x_br, plate_z_norm, release_speed, pfx_x, pfx_z, dvelo, dloc)`
  (`_SeqNet._tokens`). Only prior pitches' realized (`exec_`) values enter `c_j`, so the
  representation is leakage-safe by the same reasoning as the state views (SPEC §0).
- `a ∈ 𝓐` — the **action**, one of `N_FAM = 8` families (`FF, SI, FC, SL, CU, CH, FS, XX`,
  `FAMILIES`). Selection predicts `a`; the outcome target **conditions on** `a` (decision D22): the
  current-action embedding `E_act(a)` is concatenated to the head input (`_SeqNet.forward`).
- `o₁ ∈ 𝒪₁ = {ball, called_strike, whiff, foul, hbp, in_play}` — the level-1 outcome
  (`OUTCOME1`, `N_O1 = 6`), the `outcome1` target.
- The per-row loss is multiclass **cross-entropy** (log loss, SPEC §8.2),
  `ℓ(y, p) = −log p_y` (`nn.CrossEntropyLoss`, scored out-of-sample by `log_loss_per_row`).

**In plain terms.** Every model in WS6 reads the at-bat like a short sentence: each prior pitch is
one "word" made of its family, its result, and seven physical numbers (speed, location, movement,
and how much the speed and location jumped from the pitch before). The five views differ only in
*how much of the sentence* each is allowed to read — the whole point of the workstream. One network
guesses the next pitch (selection); another, told which pitch was actually thrown, guesses its
result (outcome).

### 1.2 The GRU cell, gate by gate (the code's `nn.GRU`, one layer)

The `O`, `L1`, and `OM` views encode the token sequence with a **gated recurrent unit** (Cho et
al., 2014), `self.gru = nn.GRU(d_tok, hidden=64, num_layers=1, batch_first=True)`. A GRU carries a
hidden state `h_t ∈ ℝ^{64}` and updates it one token at a time. We derive the update exactly as
PyTorch computes it (the ground truth for the code), then note the equivalence to Cho's original.

**Step — the reset gate.** With input `x_t` and previous state `h_{t−1}`,
$$r_t = \sigma\!\big(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}\big),$$
`σ` the logistic sigmoid. `r_t ∈ (0,1)^{64}` decides, per coordinate, how much of the previous
state is *allowed into the candidate* below.

**Step — the update gate.**
$$z_t = \sigma\!\big(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}\big).$$
`z_t ∈ (0,1)^{64}` decides how much of the **old** state to keep versus overwrite.

**Step — the candidate state.** The reset gate multiplies the *recurrent* contribution only, and
the result passes through `tanh`:
$$\tilde h_t = \tanh\!\big(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})\big),$$
where `⊙` is elementwise product. When `r_t → 0` the candidate ignores history and depends on `x_t`
alone — the mechanism by which a GRU can "forget" on demand.

**Step — the state interpolation.** The new state is a per-coordinate convex blend of old state and
candidate:
$$\boxed{\; h_t = (1 - z_t)\odot \tilde h_t + z_t \odot h_{t-1}. \;}$$
(PyTorch's convention: `z_t` is the *keep-old* gate. Cho et al. write `h_t = (1-z)⊙h_{t-1} + z⊙\tilde h`
with `z` the *keep-new* gate; the two are identical under `z ↦ 1-z`, a relabelling that does not
change the function class.) The sequence is consumed with `pack_padded_sequence` so padded positions
never update the state (`_full_seq`); the view reads the **last layer's final hidden state** `h_T`,
zeroed for empty histories (`h * valid`).

**Step — from state to prediction.** Each present branch projects to `hidden`; the branches are
concatenated with the static-context branch (`static_enc`), the current-action embedding is appended
for the outcome target (D22), and a shared MLP head produces the class logits
(`_SeqNet.head = Linear → ReLU → Dropout → Linear`):
$$\text{logits} = \mathrm{Head}\big(\underbrace{\phi_C(x^C)}_{\text{static}} \,\Vert\, \underbrace{g_v(h_{1:T})}_{\text{sequence branch}} \,[\Vert\, \phi_M(x^M)] \,[\Vert\, E_{\text{act}}(a)]\big),$$
with `g_v` the view's sequence encoder (§§2–3), `p = softmax(logits)`.

**In plain terms.** A GRU reads the sentence left to right, keeping a running summary `h_t` in 64
numbers. At each word two "gates" decide how much of the running summary to keep and how much of the
new word to fold in, and a "reset" gate lets it wipe the summary when the new word makes the past
irrelevant. After the last prior pitch, that 64-number summary — plus the game context, plus (for
the outcome model) the pitch actually thrown — is fed to a small two-layer network that outputs the
probability of each next pitch (or each outcome). The gates are exactly what let one identical
network behave differently when fed one token (`L1`) versus the whole ordered history (`O`).

---

## 2. Why mean-pooling is order-invariant: `U` as the deep multiset view

The `U` view must answer SPEC §6's question — *does the unordered collection of prior pitches
matter?* — so its encoder must be **invariant to the order** of the tokens by construction, not by
training. WS6 realises it as a masked mean of per-token projections (`_SeqNet._mean_pool`).

**Step — write the encoder.** With `φ(x) = ReLU(W x + b)` the shared token projection (`tok_proj`)
and mask `m_j ∈ {0,1}`,
$$g_U(h_{1:T}) = \frac{1}{\sum_j m_j}\sum_{j=1}^{T} m_j\,\phi(x_j).$$

**Step — the permutation argument (one line).** For any permutation `π` of `{1,…,T}`, addition is
commutative, so
$$\sum_{j} m_{\pi(j)}\,\phi(x_{\pi(j)}) = \sum_{j} m_j\,\phi(x_j) \;\Longrightarrow\; g_U(h_{\pi(1:T)}) = g_U(h_{1:T}).$$

$$\boxed{\; g_U \text{ is invariant to token order: it is a function of the \emph{multiset} } \{x_1,\dots,x_T\}\ \text{only.} \;}$$

**Step — read off the implication.** `U` is the exact deep analogue of the tabular "unordered counts
/ means of prior pitches" view (SPEC §6): both discard order and keep the bag. Any gap between `O`
and `U` therefore isolates *order* information — but note the two encoders are **different function
classes** (mean-pool vs recurrence), so `O − U` mixes an information difference with an
architecture difference. The clean capacity-matched order comparison is `O` vs `L1` (§3, §4), which
is why the headline `Δ_order` in the code takes `min[U, L1]` as the reference (`ablation_deltas`):
`L1` supplies the capacity match, `U` supplies the pure order-blindness, and the minimum guards
against either reference being accidentally weak.

**In plain terms.** `U` is handed the same words as `O` but with their order scrambled, and it is
built so scrambling changes nothing: it averages the words into one blob. Because averaging does not
care what order you add things in, `U` literally cannot see sequence — it is the "bag of pitches"
model. That makes it the honest order-blind yardstick, and it is why `U` is the *deep* version of
the study's unordered rung.

---

## 3. Truncation as `L1`: the information-set argument

`L1` must be the "immediately-preceding-pitch" rung (SPEC §6) *and* the capacity match for `O`. WS6
gets both from one device: `L1` is the **same GRU as `O`**, fed only the last history token
(`_SeqNet._last_token`).

**Step — state what the truncated GRU computes.** Feed the length-1 sequence consisting of the last
valid token `x_T` (for a non-empty history) through the GRU initialised at `h_0 = 0`. One step of
§1.2 with `h_{t-1} = 0` gives
$$h^{L1} = (1 - z_1)\odot \tilde h_1 + z_1 \odot 0 = (1 - z_1)\odot \tilde h_1,
\quad z_1 = \sigma(W_{iz}x_T + b_{iz} + b_{hz}),\ \ \tilde h_1 = \tanh(W_{in}x_T + b_{in} + \sigma(\cdot)\odot b_{hn}).$$
Every term depends on `x_T` alone.

**Step — the information-set claim.** Since the head also reads the static context `x^C` (and, for
the outcome target, the action `a`),
$$\boxed{\; \hat p_{L1}(\cdot \mid s) = F\big(x^C,\, x_T,\, a\big) \quad\text{— a function of (context, last token, action) only.} \;}$$
`O`, by contrast, reads `h_T = ` (a function of the *whole ordered* `h_{1:T}`), so
`\hat p_O(\cdot | s) = F'(x^C, h_{1:T}, a)`. The **information set** of `L1` is a strict subset of
`O`'s: `σ(x^C, x_T, a) ⊆ σ(x^C, h_{1:T}, a)`. By the data-processing / conditional-Jensen argument
(same structure as WS3's D11 nesting, `../ws3_gbdt_stack/THEORY.md` §3), the Bayes-optimal `O` loss
is `≤` the Bayes-optimal `L1` loss: **more of the sequence cannot raise the achievable loss.** Any
*measured* `L1 − O` gap out-of-sample is thus an estimate of how much of that achievable order
information the learned `O` actually captured.

**Step — why this is the twin comparison (D47).** Because `L1` and `O` are the *identical network*
(§4), the fitted `L1 − O` loss difference removes model size as a confound entirely. The D47 grammar
gate is exactly this twin delta,
$$\Delta_{\text{twin}} = \mathrm{Loss}(L1) - \mathrm{Loss}(O),$$
`_l1_twin_delta_ci` in the code — read as "does the full ordered history beat the last pitch, at
equal capacity?" The null-world completed validation reports `Δ_twin = +0.0115`, CI
`[+0.0064, +0.0161]` on the repeat-context subset (§5, `PAPER.md` §5).

**In plain terms.** `L1` is not a different, smaller model — it is `O` with a blindfold that lets it
see only the most recent pitch. Formally, its prediction can depend on nothing but the context, that
one pitch, and (for outcomes) the pitch being thrown. So when `O` beats `L1`, it is beating *itself
with the blindfold off*: the only thing that changed is how much of the ordered history was fed in,
which is precisely the "does order matter?" question with the model-size answer held fixed.

---

## 4. Capacity matching: what is, and is not, controlled (the honest decomposition)

D45 requires the deep ablation to vary **information, not capacity**. This section states precisely
what that buys.

**Step — the parameter-count match.** `L1` and `O` instantiate the *same* `_SeqNet` (same
embeddings, same `nn.GRU`, same `Head`); they differ only in the `forward` path (`_last_token` vs
`_full_seq`). Therefore
$$\boxed{\; \texttt{n\_params}(L1) = \texttt{n\_params}(O) \quad\text{exactly.} \;}$$
The completed validation confirms it: `L1 = O = 31{,}112` parameters (`PAPER.md` §5). `C` (static MLP
only, no sequence branch) is `6{,}856`; `U` (adds the `tok_proj` mean-pool branch) `13{,}512`; `OM`
(adds the matchup branch `match_enc`) `36{,}616`. So the ladder's parameter counts are reported, not
forced identical (SPEC §7) — but the one comparison that carries the order claim, `L1` vs `O`, is
parameter-identical by construction.

**Step — decompose what "matched" means.** Three things could make `O` beat `L1`; capacity matching
neutralises only the first.

1. **Parameter count (controlled).** Identical, as boxed above. A larger `O` cannot be the
   explanation.
2. **Inductive bias (only partly controlled).** `L1` and `O` share the *architecture*, so their
   inductive bias is matched to each other. But `C` and `U` are necessarily *non-recurrent* (a
   static MLP; a permutation-invariant mean-pool), so they belong to different function classes.
   Matched *width* (`hidden = 64`, `embed = 16`) does **not** make them the same estimator. This is
   why `U − O` is read cautiously and `L1 − O` carries the headline.
3. **Optimization landscape (not controlled).** Feeding `O` the full ordered sequence gives it a
   longer computational graph, a harder (potentially vanishing-gradient) loss surface, and — in the
   data — a longer wall-clock (`O = 145.9 s` vs `L1 = 81.1 s` at equal parameters, `PAPER.md` §5).
   Early stopping on the validation fold (§7) is what lets `O` *realise* its lower achievable loss
   rather than overfit or under-optimise; without it a capacity-matched `O` could look worse than
   `L1` for optimisation reasons alone, which would be a training artifact, not an information
   finding.

$$\boxed{\; \text{matched parameters} \;\ne\; \text{matched inductive bias} \;\ne\; \text{matched optimization.} \;}$$

**Step — the honest consequence.** A **positive** `Δ_twin` (`O` beats `L1`) is strong evidence of
ordered information, because the only uncontrolled axis (optimisation) is *handicapping `O`*, not
helping it — `O` has the harder optimisation problem, so beating `L1` despite that is conservative. A
**null-or-negative** `Δ_twin` is correspondingly weaker as a claim of "no order information": it is
consistent with "no order information" **or** with "`O`'s harder optimisation problem swallowed a
small real effect at this scale." The training curves (§7) and the synthetic positive-world control
(which plants a real order effect the GRU must recover) are what disambiguate the two — a limitation
`PAPER.md` §8 states plainly.

**In plain terms.** We made the "does order matter?" test fair by giving the last-pitch model and the
full-sequence model *the exact same brain* — same size, same wiring — and changing only what we let
each read. That kills the obvious objection ("of course the bigger model won"). Two things it does
*not* equalise: the mean-pool and static models are a different kind of brain (so we lean on the
last-pitch twin, not those, for the headline), and the full-sequence model has the genuinely harder
learning job (a longer chain to train through). The second cuts in our favour when `O` wins — it won
with a handicap — and against a too-strong reading when `O` ties, which is why we read a tie as "no
*detected* order effect at this scale," never as proof order is absent.

---

## 5. The ablation statistics, the D21 min-bias, and the targeting lesson

WS6's headline is the SPEC §6 ablation, scored **only** through the shared harness
(`compare_views` → `Δ_order`, `Δ_matchup` with pitcher-game clustered CIs). This section fixes how
each is read and then derives the arithmetic that is new to WS6: why an effect that binds on a
*subset* is detected on that subset but diluted in the aggregate.

### 5.1 The two statistics and their clustered CIs

$$\Delta_{\text{order}} = \mathrm{Loss}(\min[U, L1]) - \mathrm{Loss}(O), \qquad
  \Delta_{\text{matchup}} = \mathrm{Loss}(O) - \mathrm{Loss}(OM),$$
both positive when the richer view has lower loss out-of-sample. CIs come from a **cluster
bootstrap** over pitcher-game clusters (`clustered_ci`), respecting SPEC §0/§7 dependence.

### 5.2 D21's min-bias for `Δ_order` (recap; derived in WS1 THEORY §10)

`Δ_order` takes the **minimum** of two held-out losses. `min(·,·)` is concave, so by Jensen, under a
null where `U`, `L1`, `O` share a true loss `L⋆`,
$$\mathbb{E}[\Delta_{\text{order}}] = \mathbb{E}[\min(\hat L_U, \hat L_{L1})] - \mathbb{E}[\hat L_O] \le L^\star - L^\star = 0.$$
So `Δ_order` is **negatively biased under the null** (decision D21): the criterion is *significantly
positive* (clustered CI lower bound `> 0`), and a small **negative** value reads as *consistent with
no ordering effect*, never "order hurts." This binds on both WS6 targets. The full derivation and the
bias-size estimate `≈ ½·E|L̂_U − L̂_{L1}|` are in `../ws1_eb_tables/THEORY.md` §10.

`Δ_matchup` takes **no minimum** — it is a straight difference of two held-out losses — so it carries
**no** Jensen bias and is read directly (a significant negative is a *real* out-of-sample cost, the
`M−` fragmentation reading derived for the tree stack in `../ws3_gbdt_stack/THEORY.md` §6.3). The
completed WS6 validation reports `Δ_matchup = −0.0036` with CIs excluding 0 on **both** targets — the
third model family (after WS3's GBDT and WS5's tabular MDP) to measure the matchup block's
out-of-sample cost where no matchup effect exists (§8; `PAPER.md` §5).

### 5.3 The targeting lesson, formalized

The null world plants an **order-2 selection habit** (a no-three-in-a-row tendency): a family is
suppressed as the next pitch **only** when the two most-recent prior pitches are the *same* family
(the "repeat context," `_repeat_context_mask`). So the ordered edge the GRU can find lives on a
**subset** of eval rows, and the aggregate `Δ_order` mixes signal rows with noise-only rows. Here is
the exact cost of that mixing.

**Step — set up support fraction and per-row effect.** Let the eval set have `n` rows. Let `S` be
the repeat-context subset with support fraction `ρ = |S| / n`. Model the per-row twin advantage
`d_i = ℓ_{L1,i} − ℓ_{O,i}` (positive when `O` is better) as
$$d_i = \delta\,\mathbf{1}[i \in S] + \varepsilon_i, \qquad \mathbb{E}[\varepsilon_i]=0,\ \ \mathrm{Var}(\varepsilon_i)=\sigma^2,$$
i.e. the effect `δ > 0` is present on `S` (where `O` can see the matching pair `L1` cannot) and
absent off `S` (where the earlier tokens carry no habit signal, so the twin ties in expectation).

**Step — the aggregate vs subset point estimates.** Averaging over all rows versus over `S`:
$$\bar d_{\text{agg}} = \frac{1}{n}\sum_{i} d_i \;\xrightarrow{}\; \rho\,\delta, \qquad
  \bar d_{\text{sub}} = \frac{1}{|S|}\sum_{i\in S} d_i \;\xrightarrow{}\; \delta.$$
$$\boxed{\; \text{aggregate effect } = \rho\,\delta \ \ (\text{diluted by the support fraction}); \qquad \text{subset effect } = \delta \ \ (\text{undiluted}). \;}$$

**Step — the variance/power arithmetic (why subset testing wins).** With per-row noise `σ²` and
(for simplicity) independent rows, the standard errors and one-sample `z`-statistics are
$$\mathrm{SE}_{\text{agg}} \approx \frac{\sigma}{\sqrt{n}},\quad z_{\text{agg}} \approx \frac{\rho\,\delta}{\sigma/\sqrt{n}} = \frac{\rho\,\delta\sqrt{n}}{\sigma};
\qquad
\mathrm{SE}_{\text{sub}} \approx \frac{\sigma}{\sqrt{\rho n}},\quad z_{\text{sub}} \approx \frac{\delta}{\sigma/\sqrt{\rho n}} = \frac{\delta\sqrt{\rho n}}{\sigma}.$$
Their ratio is
$$\boxed{\; \frac{z_{\text{sub}}}{z_{\text{agg}}} = \frac{\delta\sqrt{\rho n}/\sigma}{\rho\,\delta\sqrt{n}/\sigma} = \frac{1}{\sqrt{\rho}} \;>\; 1 \quad(\text{since } \rho < 1). \;}$$
Testing on the subset is **more powerful by a factor `1/√ρ`.** Discarding the `(1−ρ)n` noise-only
rows removes their variance contribution to the estimate faster than it removes signal (the signal
was already zero there), so the effect is easier to see, not harder. (Clustering by pitcher-game
inflates both SEs by the same design-effect factor, so the ratio `1/√ρ` is unchanged; the code
computes both CIs with the identical `clustered_ci`.)

**Step — read it against the numbers.** In the null-world validation the repeat context is a modest
slice (`n_repeat_rows = 1136`), so `ρ` is well below 1. The aggregate selection `Δ_order` is `+0.0006`
(not significant — the effect diluted by `ρ`, *and* pushed down by the §5.2 min-bias), while the
subset twin delta `Δ_twin |_S = +0.0115`, CI `[+0.0064, +0.0161]` (`> 0`) — the **same** habit,
detected once the evaluation targets the rows it acts on. This is the general lesson, formalized:
$$\boxed{\; \text{an effect with support fraction } \rho \text{ needs subset-targeted evaluation; the aggregate hides it by the factor } \rho \ (\text{point}) \text{ and } \sqrt{\rho}\ (\text{power}).\;}$$
The harness exposes exactly this via its slice machinery (`slice_masks`, and the `row_mask` argument
of `_l1_twin_delta_ci`), which is why the D47 gate reads the twin delta on the repeat-context slice,
not the aggregate.

**In plain terms.** The planted habit only bites in one specific situation — when the last two
pitches were the same. If you grade the model on *every* pitch, most pitches are situations where the
habit does nothing, so their random noise drowns the signal, and the average edge looks like nothing.
Grade the model *only on the pitches where the habit applies*, and the edge pops out. The math is
tidy: averaging over everything shrinks the true effect by the fraction of rows it lives on, and
shrinks your statistical power by the square root of that fraction. That is why WS6 checks the
repeat-context slice, and it is a lesson that generalises to any real effect that lives in a corner
of the data.

---

## 6. The motif probe as a functional read-out: why loss-detection and probability-exposure dissociate

The D47 exhibit asks a sharper question than the ablation: having established (via the loss) that the
`O` GRU *uses* the grammar, does its explicit output distribution *expose* the rule the way WS2's
grammar table does? The probe (`motif_repeat_probe`) is a gradient-free functional read-out.

**Step — define the two probe conditions.** For each family `X`, construct two synthetic length-2
histories that **end in the same token `X`** and differ only in the second-to-last pitch:
`[X, X]` (repeat) and `[Y, X]` for `Y ≠ X` (mixed, averaged over `Y`). Both end in `X`, the static
context is held at the training mean, and the float channels are held in-distribution (each token
carries its family's mean physical profile, `dvelo/dloc` re-derived; `_probe_next`). So an `L1` model
would predict identically across the two; **only `O` can react to the earlier token.**

**Step — the read-out.** With `p(X ∣ hist) = ` the model's predicted probability that the *next*
pitch is `X`,
$$\text{suppression}(X) = p(X \mid [Y,X]) - p(X \mid [X,X]) = p_{\text{mixed}}(X) - p_{\text{repeat}}(X),$$
positive when the third-consecutive `X` is made *less* likely after a repeat — the planted habit.
The summary is `mean_suppression = mean_X suppression(X)` and
`motif_present = (mean_suppression > 0) ∧ (frac_suppressed ≥ 0.5)` (`motif_repeat_probe`).

**Step — state the dissociation and why it is not a contradiction.** The completed null-world
validation reports `motif_present = False`: `mean_suppression = −0.0063`, only `43%` of families
suppressed (`FF: +0.05` suppressed, but `FC: −0.057` anti-suppressed) — *while the loss-based twin
delta on the same world is CI-positive* (§5). Loss-detection and probability-exposure measure
different functionals, and a model can pass the first without cleanly passing the second. Two exact
reasons:

1. **Loss weights by realized frequency; the probe weights families uniformly.** The out-of-sample
   loss on repeat-context rows is `E[−log p_{Y}]` over the *realized* next pitches `Y`, which — by
   the habit — are rarely the repeated family. A model lowers that loss by raising probability on the
   realized (non-`X`) families in aggregate; it need not lower the *specific level* `p(X∣[X,X])`
   uniformly across `X`. The probe, by contrast, is an **unweighted mean over the 8 families**, so a
   *rare* family (few `[FC,FC]` training contexts) contributes as much to `mean_suppression` as a
   *common* one (`FF`) contributes to the loss — and a rare family's synthetic `[X,X]` probe input is
   near the edge of support, where the learned function extrapolates. The observed `FF (+0.05)` vs
   `FC (−0.057)` split is exactly this frequency-vs-uniform mismatch.

2. **A tiny formal example where loss improves but a family's suppression sign flips.** Take two
   families `A, B` and, on repeat context `[A,A]`, let the true next-pitch distribution be
   `(A, B) = (0.1, 0.9)` (the habit suppresses a third `A`); on mixed `[B,A]` let it be `(0.4, 0.6)`.
   A model that outputs `\hat p(A∣[A,A]) = 0.15` and `\hat p(A∣[B,A]) = 0.35` has **lower** repeat-
   context cross-entropy than an order-blind model outputting `0.30` on both (it moved toward the
   realized `B`), yet its suppression for `A` is `0.35 − 0.15 = +0.20 > 0` — correctly signed. Now let
   a second family `C` be one the pitcher almost never repeats: the model has seen `[C,C]` a handful of
   times, its `\hat p(C∣[C,C]) = 0.22` is essentially noise, and `\hat p(C∣[B,C]) = 0.19`, giving
   suppression `−0.03 < 0`. **The average over `{A, C}` can be negative even though the loss on the
   frequent, in-support contexts fell.** The scalar loss is an average over the realized-label
   direction on in-distribution rows; the probe is a per-family counterfactual level on partly
   out-of-support inputs. They are allowed to disagree, and at demo scale they do.

$$\boxed{\; \text{The GRU's \emph{loss} detects the grammar (repeat-context } \Delta_{\text{twin}} > 0)\ \text{while its explicit \emph{probabilities} do not cleanly expose it: learned representations use patterns opaquely.} \;}$$

**Step — the interpretability contrast (WS2).** WS2's variable-order grammar stores the motif as an
explicit, inspectable conditional table — you can read "after `[X,X]`, `P(X)` drops" straight off the
model. WS6's GRU folds the same regularity into 64 continuous hidden units and a softmax; the rule is
*there* (the loss proves it) but not *legible* per family at this scale. That is the deep-learning
bargain in one exhibit, and what more training/data might change is a Phase-2 question (`PAPER.md` §7,
`{PLACEHOLDER}` for the real-data rerun).

**In plain terms.** "The model uses the rule" and "the model will tell you the rule" are different
claims. We proved the first: the GRU scores better exactly on the pitches where the rule lives. The
second — open the model, ask each pitch family "is a third-in-a-row less likely after a repeat?" —
comes back muddled: common pitches say yes, rare ones say no, and the average is a wash. That is not a
contradiction. The score is an average over what actually happened, weighted by how often each
situation occurs; the probe is a fussy per-family what-if on made-up inputs, some of which the model
barely saw in training. WS2's grammar answers the what-if cleanly because it *is* a table of what-ifs;
the GRU answers it opaquely because it packed the rule into arithmetic no one can read off. Opacity is
the price of the learned representation.

---

## 7. Early stopping as regularization (a bias–variance sketch)

WS6 trains with Adam and **early stopping on a validation fold**, restoring the best-val weights
(`SeqModel.fit`; when no external val fold exists — the falsification refits — a seeded
`internal_val_frac = 0.2` slice is held out). This section explains why that is regularization, and
why it is load-bearing for the ablation specifically.

**Step — the two curves.** Let `L_train(t)`, `L_val(t)` be train / held-out loss after `t` epochs.
Gradient training drives `L_train(t)` down monotonically; `L_val(t)` is U-shaped: it falls while the
model learns signal, then rises as it fits training noise. Early stopping returns the weights at
`t^\* = \arg\min_t L_val(t)` (patience `= 8`, D46).

**Step — the bias–variance reading.** Decompose held-out risk as `Bias²(t) + Var(t) + σ²_{noise}`.
More epochs *reduce bias* (the fit tracks the target more closely) and *increase variance* (it also
tracks noise). The minimiser satisfies the first-order condition
$$\left.\frac{d\,\mathrm{Bias}^2}{dt}\right|_{t^\*} = -\left.\frac{d\,\mathrm{Var}}{dt}\right|_{t^\*},$$
the standard bias–variance balance. For quadratic loss and (near-)linear models there is an exact
correspondence between **training time and shrinkage**: stopping after `t` gradient steps is
equivalent to an `L2` penalty that *decreases* with `t` — training longer = regularising less. So
`t^\*` is a data-chosen regularisation strength, not a fixed hyperparameter.

**Step — why it is load-bearing for the ablation (the capacity-match link).** `O` has the strictly
larger information set and the harder optimisation problem (§4). Without early stopping, `O` could
overfit its extra tokens and post a *higher* held-out loss than its capacity-matched twin `L1` — a
pure overfitting artifact that would masquerade as "order hurts." Early stopping on val caps `O`'s
effective capacity at the generalising optimum, so `Δ_twin = L1 − O` measures **information the fit
actually generalised**, not training-set memorisation. This is the deep-model analogue of WS3's
equal-hyperparameter-budget discipline (`../ws3_gbdt_stack/THEORY.md` §9): both make a view's
loss reflect its *information*, not its freedom to overfit.

**In plain terms.** If you train a flexible model too long it starts memorising the training data and
gets *worse* on new data. Early stopping watches a held-out slice and quits at the moment new-data
loss bottoms out — an automatic dial for "how much model is too much," set by the data. It matters
here because the full-sequence model has the most rope to hang itself with; stopping it at the right
point is what makes a fair comparison to the last-pitch twin about *information* rather than about who
overfit more.

---

## 8. The Pareto frontier formalized (SPEC §7): gain per log-parameter, per second

WS6 is the study's **compute peak on the predictive side** (SPEC §7), so the loss must be read
against its cost. Every fit logs `(n_params, seconds, peak_mem_mb)` via `runmeta` (D13), printed as
the **Pareto row**.

**Step — define dominance.** For view `v` with generalisation loss `L_v` and compute coordinate
`c_v = (\text{params}_v, \text{seconds}_v, \text{mem}_v)`, `v` is **Pareto-dominated** by `v'` iff
$$L_{v'} \le L_v \ \wedge\ c_{v'} \preceq c_v \quad\text{(componentwise)}, \ \text{with strict inequality somewhere.}$$
The **Pareto frontier** is the set of non-dominated views; the SPEC §7 deliverable is the
performance-vs-compute plot of `(c_v, L_v)`.

**Step — the marginal reading (gain per log-param / per second).** The decision "is the richer view
worth its cost?" is a marginal one. Between two rungs,
$$\text{gain per log-param} = \frac{L_{v} - L_{v'}}{\log \text{params}_{v'} - \log \text{params}_{v}},
\qquad
\text{gain per second} = \frac{L_{v} - L_{v'}}{\text{seconds}_{v'} - \text{seconds}_{v}}.$$
Log-parameters because returns to model size are diminishing and roughly log-linear; seconds because
wall-clock is what Phase-2 actually spends.

**Step — apply it to the capacity match (the sharp case).** `L1` and `O` have **identical**
parameters, so "gain per log-param" is undefined for that pair — the *only* separating cost is
**time**: `O = 145.9 s` vs `L1 = 81.1 s` (`PAPER.md` §5). So the `O`-over-`L1` question reduces to a
pure "gain per second":
$$\boxed{\; \text{Is } \tfrac{\Delta_{\text{twin}}}{\text{seconds}_O - \text{seconds}_{L1}} \text{ worth it? At equal parameters, the } O\text{ premium is entirely wall-clock (the recurrence over full sequences).} \;}$$
If `Δ_twin` is small (SPEC §13's expectation that "O barely beats L1"), `O` is Pareto-dominated by
`L1` and the cheap model wins the frontier — the `R=` result of `PAPER.md` §6. The full ladder's
verified costs (`C 6{,}856/62.6\text{s}`, `U 13{,}512/74.8\text{s}`, `L1 31{,}112/81.1\text{s}`,
`O 31{,}112/145.9\text{s}`, `OM 36{,}616/110.1\text{s}`, `~1.0` GB peak, 6 epochs) are the compute
coordinates read against the loss table; and the deeper question SPEC §7 poses for the whole study is
whether WS6's frontier point (orders of magnitude more compute than WS1–WS3) buys loss the cheaper
rungs could not.

**In plain terms.** Every model costs parameters, time, and memory, and a bigger model is only worth
it if the accuracy it buys beats the accuracy-per-cost of a smaller one. Because we deliberately made
`L1` and `O` the same size, the *only* thing `O` costs extra is **clock time** — reading the whole
sequence takes almost twice as long as reading the last pitch. So "is order worth it?" becomes "is the
tiny order gain worth doubling the training time?" If order barely helps (the outcome SPEC expects),
the honest answer is no — the last-pitch model sits on the efficient frontier and the full-sequence
model is a luxury. That verdict, and whether the whole deep rung earns its place against the cheap
tree and grammar models, is exactly what the Pareto plot is for.

---

*Every boxed result above appears verbatim in `PAPER.md` (§4–§6) and is implemented in
`workstreams/ws6_deep_seq/model.py` at the named function/class. The plain-English tour is
`SEAN-README.md`; the runnable pipeline and CLI are `run_ws6.py` and RUNBOOK §WS6.*
