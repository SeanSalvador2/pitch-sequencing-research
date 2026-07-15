# Learned Representations of Pitch Sequences: A Capacity-Matched GRU Ablation

*Workstream 6 of a comparative pitch-sequencing study — the deep-learning rung and the study's
compute peak on the predictive side.* This draft is written to be completed in place: every quantity
that depends on the real Statcast data is a `{PLACEHOLDER}`, and the Results and Discussion are
**branched** so that the correct interpretation is already written for whichever numbers arrive. The
synthetic-world numbers quoted in §5 are *completed validation* from the committed WS6a run, not
placeholders.

---

## Abstract

Workstream 6 (WS6) asks the study's representation-learning question: does a *learned* encoding of
the ordered plate-appearance sequence beat the **engineered** tabular history (WS3) and the
**explicit** variable-order grammar (WS2)? It fits the same five nested state views
(`C`, `U`, `L1`, `O`, `OM`; SPEC §6) as a single family of **capacity-matched neural architectures**
(decision D45), so the ablation measures *information*, not *capacity*: `C` is a static context MLP;
`U` is an order-invariant mean-pool of token embeddings; **`L1` and `O` are the *identical* GRU** —
`L1` fed only the last history token, `O` the full ordered sequence — so any `O`-over-`L1` gain is
attributable to ordered information and not to model size; `OM` appends a matchup-memory branch. Both
targets are run: next-family `selection` and the current-action-conditioned `outcome1` (decision
D22). The ablation reports `Δ_order = Loss(min[U, L1]) − Loss(O)` and `Δ_matchup = Loss(O) − Loss(OM)`
with pitcher-game clustered CIs on validation (2024) and the locked test (2025). On the real data
(2021–2025 Statcast, `~3.85M` pitches) the outcome-1 `Δ_order` is `{DELTA_ORDER_OUTCOME1}` (95%
clustered CI `{DELTA_ORDER_OUTCOME1_CI}`) on validation and `{DELTA_ORDER_OUTCOME1_TEST}`
(`{DELTA_ORDER_OUTCOME1_TEST_CI}`) on the locked test, `Δ_matchup` is `{DELTA_MATCHUP_OUTCOME1}`
(`{DELTA_MATCHUP_OUTCOME1_CI}`), and the learned `O` view's loss against WS3's engineered GBDT-`O`
and WS2's grammar is `{GRU_VS_WS3_WS2}` — the representation verdict. We validate the method on the
correctness oracle as a real-model test (decision D47). On the **null** world the D47 verdicts pass:
`GRU_GRAMMAR_DETECTED` — the ordered `O` selection GRU beats its capacity-matched twin `L1` on the
**repeat-context subset** (twin delta `+0.0115`, CI `[+0.0064, +0.0161]`, `n_repeat_rows = 1136`)
while the *aggregate* selection `Δ_order` is `+0.0006` (not significant) — and `GRU_OUTCOME_QUIET`
(outcome `Δ_order = −0.0004`, CI `[−0.0023, +0.0011]`, permutation `p = 0.091`, not fired; locked
test `−0.0011`, n.s.). Two exhibits carry the paper's methodological payload. First, the
**targeting lesson**: the planted habit binds only when the two most-recent pitches match, so it is
detected on that subset and *diluted to nothing in the aggregate* — an effect with support fraction
`ρ` costs the aggregate a factor `ρ` in point size and `√ρ` in power (`THEORY.md` §5), so effects
that bind on a subset need subset-targeted evaluation. Second, the **opacity exhibit**: a gradient-
free motif-rediscovery probe finds `motif_present = False` at demo scale (mean suppression `−0.0063`,
only `43%` of families suppressed; `FF +0.05` suppressed but `FC −0.057` anti-suppressed) — the GRU's
**loss** detects the grammar (the repeat-context twin delta is CI-positive) while its explicit per-
family **probabilities** do not cleanly expose the rule, the sharp interpretability contrast with
WS2's inspectable motif table. A methodological finding is carried forward: `Δ_matchup` is
significantly **negative** on both synthetic targets (`−0.0036`, CIs excluding 0) — WS6 is the
**third** model family, after WS3's GBDT and WS5's tabular MDP, to measure the out-of-sample cost of
the matchup feature block where no matchup effect exists. The full five-view Pareto row (params /
epochs / wall-clock) is reported for the SPEC §7 performance-vs-compute plot, with `L1` and `O`
printing **identical** parameter counts (`31,112`) — the capacity match made visible — and `O`
costing `~1.8×` `L1`'s wall-clock for the same size. WS6 ships **dual-path reproducibility** (decision
D46): a local CPU path and a self-contained free-Colab T4 path. The headline is a three-axis result
grid — does the *learned* representation beat the engineered features (R+/R=/R−), does *order* carry
value within WS6 (H1/H2/H3, read with the targeting lesson), and does *matchup memory* (M+/M0/M−) —
read under the study's finding-#2 firewall: a predictor is *predictive*, not causal.

---

## 1. Introduction

The study this workstream belongs to is organized as a **rigor ladder**, not a horse race (SPEC §0):

> How much evidence for sequencing survives progressively harder tests — from descriptive order
> patterns, to out-of-sample **outcome** dependence, to **counterfactual policy** value?

The ladder exists because three findings are routinely conflated and must be kept separate (SPEC §0,
verbatim):

> 1. **Selection structure** — prior pitches help predict *what is thrown next*.
> 2. **Predictive sequencing value** — prior pitches help predict the *outcome* of the current pitch,
>    after conditioning on the current pitch and game state.
> 3. **Prescriptive/causal value** — *changing* the sequence would improve outcomes.
>
> The first is easy; the second is hard; the third needs assumptions that public data cannot fully
> satisfy (we never see the pitch that wasn't thrown; `pitch_type` is a classifier output, not the
> battery's intent; scouting/target info is unobserved).

Workstreams 1–3 answered findings #1 and #2 with progressively stronger *engineered* and *explicit*
models: WS1's empirical-Bayes tables (the support problem), WS2's variable-order Markov grammar (the
selection-depth question, answered with an inspectable motif table), and WS3's gradient-boosted stack
(the centerpiece — the first uniform five-view outcome ablation, whose tabular `O` view encodes
ordered history as hand-designed positional slots and consecutive-difference features, decision D11).
**WS6 is the deep-learning rung.** It replaces the hand-designed encoding with a *learned* one and
asks the question the ladder was built to reach: when we stop engineering features and let a small
neural network read the at-bat as a sequence, does it find ordered signal the engineered features and
the explicit grammar missed?

Three properties make WS6 a clean measurement of that question rather than one more model on the pile.

**First, it varies information, not capacity (decision D45).** A naive "deep beats tabular" comparison
confounds two things — the learned representation *and* the extra parameters. WS6 removes the second
confound by realising the five views as **capacity-matched architectures**: the cleanest pair, `L1`
and `O`, is the *identical* network (same embeddings, same GRU, same head) differing *only* in how
much of the sequence is fed in — the last token versus the full ordered history. Their parameter
counts are therefore equal by construction (`31,112` each, §5), so a measured `O`-over-`L1` gain is
ordered information, full stop. `C` and `U` are the order-blind reference points, matched on
embedding and hidden width; per-view parameter counts are *reported* (the SPEC §7 Pareto axis) rather
than forced identical.

**Second, it is honest about compute.** WS6 is the study's **compute peak on the predictive side**
(SPEC §7): the recurrent `O`/`OM` fits over millions of padded sequences are the slowest predictive
step. So every fit logs params, wall-clock, and peak RAM, and the paper reads the (usually small)
ablation gain against that cost — a small `Δ_order` bought with orders of magnitude more compute than
WS1–WS3 is a *Pareto* statement, not just a loss statement. WS6 is also the study's only GPU-relevant
step, and it ships both a local CPU path and a self-contained free-Colab T4 path (decision D46).

**Third, it is the interpretability contrast.** WS2's grammar exposes the learned motif as a table
you can read; WS6's GRU folds the same regularity into continuous hidden units. WS6 makes that
contrast an explicit exhibit — a motif-rediscovery probe that asks whether the learned model's output
probabilities expose the rule its loss demonstrably uses.

**Contributions.**
1. **A capacity-matched deep ablation of all five state views**, with `L1` and `O` realised as one
   identical GRU so the order comparison is parameter-exact — the study's first deep `Δ_order` /
   `Δ_matchup`, on both targets, with a locked-test row.
2. **The targeting lesson, formalized and exhibited.** An effect that binds on a subset (support
   fraction `ρ`) is diluted by `ρ` in the aggregate and needs subset-targeted evaluation; the null
   world shows it live (aggregate selection `Δ_order` null, repeat-context twin delta CI-positive).
3. **The opacity exhibit.** A gradient-free motif probe showing the GRU's loss uses the grammar while
   its explicit per-family probabilities do not cleanly expose it — a concrete interpretability
   contrast with WS2's explicit motif table.
4. **The three-family `Δ_matchup` finding.** A significantly negative outcome `Δ_matchup` on both
   synthetic worlds, making WS6 the third model family (after WS3, WS5) to measure the matchup block's
   out-of-sample fragmentation cost — a cross-workstream pattern, pre-written as the live `M−` branch.
5. **Dual-path reproducibility** (decision D46): identical device-agnostic code on a local CPU and a
   self-contained free-Colab T4, with the ROCm/DirectML caveat documented and Colab recommended.

---

## 2. Related work

**Recurrent sequence encoders.** WS6's ordered views use a **gated recurrent unit** (GRU), introduced
by Cho et al. (2014) as the encoder half of an RNN encoder–decoder for machine translation: a
recurrence with multiplicative *update* and *reset* gates that carry information across steps while
mitigating the vanishing-gradient problem of a plain RNN. The GRU is the streamlined descendant of the
**long short-term memory** cell of Hochreiter and Schmidhuber (1997), whose constant-error-carousel
gating is the origin of the gated-recurrence idea; WS6 uses the GRU for its smaller parameter count at
comparable accuracy, which matters for a capacity-matched ablation and a CPU-runnable budget. The
gate-by-gate derivation of the exact cell WS6 trains (PyTorch's `nn.GRU`) is `THEORY.md` §1.2. The
broader use of such recurrences to *summarise a variable-length sequence into a fixed vector* for a
downstream predictor is the sequence-to-sequence framing of Sutskever, Vinyals, and Le (2014), which
is precisely how WS6 consumes the final hidden state.

**Order-invariant set encoders.** The `U` view must be invariant to token order by construction, which
is exactly the **permutation-invariant pooling** of Deep Sets (Zaheer et al., 2017): a sum/mean of
per-element embeddings is a universal encoder of a *multiset*. WS6's `U` is the masked-mean instance of
that construction (`THEORY.md` §2), making it the deep analogue of the study's unordered rung and a
principled order-blind yardstick rather than an ad-hoc baseline.

**Attention and convolutional alternatives (the demo's lineage, and the comparison not made).** The
Transformer (Vaswani et al., 2017) replaced recurrence with self-attention and is the architecture of
WS6's *optional* compact Transformer demo, which SPEC §12.4 scopes as a demonstration for the long
cross-PA test once the GRU shows signal — deliberately kept out of the acceptance gates
(`TransformerDemo`, exercised by a single smoke test). We also note, but do not adopt as a headline
model, the **temporal convolutional network**: Bai, Kolter, and Koltun (2018) is an arXiv empirical
evaluation reporting that dilated causal convolutions can match or exceed recurrent networks on
several sequence-modeling benchmarks; we cite it as the reason a convolutional encoder is a reasonable
Phase-2 alternative, not as an established theoretical result, and WS6 does not implement one.

**Optimisation and regularisation.** WS6 trains with the Adam optimiser (Kingma and Ba, 2014), uses
dropout (Srivastava et al., 2014) in the head and branches, and regularises by **early stopping** on
the validation fold — the training-time-as-shrinkage correspondence that makes a longer fit a weaker
regulariser (`THEORY.md` §7). These are load-bearing for the ablation specifically: they cap the
larger-information `O` view's freedom to overfit so that a measured `L1 − O` gap reflects generalised
information, not memorisation.

**Baseball analytics and sequencing.** The applied grounding is Tango, Lichtman, and Dolphin (2007),
whose count- and matchup-based analysis motivates the context (`C`) view, and Marchi and Albert (2013)
for the treatment of Statcast-style pitch data. WS6's selection views are scored against the study's
shrunk count-based reference baselines (decision D16), whose partial-pooling philosophy descends from
Efron and Morris (1975).

**Internal cross-references.** WS6 is read against its two predecessors on the representation ladder.
WS2 (variable-order Markov grammar) resolved the *selection* depth question with an **explicit,
inspectable** motif table; WS6's opacity exhibit (§5, §7) is the direct contrast — the same planted
motif, folded into a learned representation that uses it opaquely. WS3 (the GBDT centerpiece)
encoded ordered history as **engineered** tabular features and set the bar WS6's learned `O` must
clear (the R-axis of §6); its D11 same-information-different-representation argument
(`../ws3_gbdt_stack/THEORY.md` §3) is the premise WS6 tests, and its no-`min`-bias `Δ_matchup`
derivation (`../ws3_gbdt_stack/THEORY.md` §6.3) is reused here for the `M−` reading. The shared
`Δ_order` min-bias rule (decision D21) is derived in `../ws1_eb_tables/THEORY.md` §10 and binds on
WS6's papers and notebook unchanged.

---

## 3. Data

**Source.** Statcast pitch-level data via pybaseball, seasons 2021–2025, a single tracking era
(Hawk-Eye), `~3.85M` pitches over 119 raw columns (SPEC §1). The reward signal `delta_run_exp` is
populated on `~99.7%` of pitches. `pitch_type` is a Statcast *classifier output* and is treated
throughout as an observed proxy for the pitch thrown, not the battery's intent (SPEC §1).

**The decision table and the five views.** Every workstream reads one shared table (SPEC §3): one row
per pitch, the decision made immediately *before* that pitch is released, keyed by
`(game_pk, at_bat_number, pitch_number)`. WS6 consumes the five nested state views (SPEC §6) but — and
this is the point of the workstream — through their **raw sequence form**, not the engineered tabular
matrices the tree models read. The static context `C` is the audited `build_view(·, "C")` matrix
(`ContextEncoder`: continuous/ordinal features standardised, low-cardinality categoricals one-hot
encoded against a fitted vocabulary); the matchup block for `OM` is `history_features(·)["OM"]`,
standardised (`MatchupEncoder`). The *ordered history* is built by `build_sequences` (SPEC §3.3):

> For every decision row, the history is the ordered list of *prior* pitches in the same PA
> (pitches `1..t-1`), oldest first; it is empty for the first pitch of a PA. Tensors are left-aligned
> and padded to `max_len = 15` (keeping the most recent pitches when a PA runs longer), with a boolean
> mask and per-row lengths, and a `row_id` array so the tensors align back to the decision table.

Each history position carries a **family** token id and an **outcome** token id (with dedicated pad
indices `FAM_PAD = 8`, `O1_PAD = 6`) plus seven **float channels**,
`SEQUENCE_FLOAT_CHANNELS = (plate_x_br, plate_z_norm, release_speed, pfx_x, pfx_z, dvelo, dloc)`, where
`dvelo`/`dloc` are the release-speed and location *changes* from the previous history pitch (0 at
position 0). This is exactly the material a tree view can read only through engineered slots: the GRU
sees the raw ordered `(family, outcome, physics, transitions)` per prior pitch and learns its own
summary. The float channels live on very different scales (velocity `~90`, `dvelo ~0–10`), so
per-channel means/stds over the *valid* (masked) training positions are fitted and applied
(`FeatureEncoders`).

**The action and the outcome.** The action is the pitch family (8 classes: `FF, SI, FC, SL, CU, CH,
FS, XX`; SPEC §4). The `selection` target predicts it; the `outcome1` target predicts the level-1
outcome `o₁ ∈ {ball, called_strike, whiff, foul, hbp, in_play}` (6 classes) and **conditions on** the
current action by concatenating its embedding to the head input (decision D22) — SPEC §0's "after
conditioning on the current pitch," which keeps finding #1 out of finding #2.

**Leakage discipline.** A feature for the decision before pitch `t` must be computable strictly before
pitch `t` is thrown (SPEC §0). `build_sequences` uses only prior pitches' realized (`exec_`) values,
never the current pitch's execution, so the sequence representation is leakage-safe by the same
reasoning as the state views; the static and matchup branches inherit the shared column-name audit
(`leakage_audit` on the matchup block).

**Temporal splits.** Train on 2021–2023, select on 2024, lock 2025 for test (SPEC §7;
`temporal_split_masks`, with a `game_pk`-parity fallback for tiny synthetic fixtures). Splits are by
season, never by random row, and all confidence intervals are bootstrapped over pitcher-game clusters.

---

## 4. Methods

### 4.1 The capacity-matched architectures (the design exhibit)

Every view is one `_SeqNet` (view × target), sharing embedding width `d_emb = 16`, hidden width
`hidden = 64`, a single recurrent layer, dropout `0.1`, and the same MLP head. The token width is
`d_tok = 2·d_emb + n_ch = 39`. The five views differ *only* in the sequence branch:

- **`C`** — static context MLP only: `head(static_enc(x^C))`. No sequence branch; the order-blind
  context reference.
- **`U`** — `C` plus a **masked mean-pool** of per-token projections,
  `g_U = (1/Σ m_j) Σ_j m_j·ReLU(W x_j + b)` — **order-invariant by construction** (`_mean_pool`;
  `THEORY.md` §2). The deep multiset view.
- **`L1`** — `C` plus a **GRU fed only the last history token** (`_last_token`): a length-1 pass whose
  output is a function of `(context, last token, action)` only (`THEORY.md` §3).
- **`O`** — `C` plus the **same GRU over the full ordered** padded sequence (`_full_seq`,
  `pack_padded_sequence`, final hidden state). The headline ordered view.
- **`OM`** — `O` plus a matchup-memory branch `match_enc(x^M)` concatenated into the head.

The **cleanest capacity match is `L1` vs `O`**: the *identical* `_SeqNet`, differing only in the
`forward` path, so their parameter counts are equal by construction. Any `O`-over-`L1` gain is
ordered information, not model size (`THEORY.md` §4). A bigger `O` would confound information with
capacity, which is the exact failure D45 exists to prevent.

### 4.2 The exact matched dimensions and the head (D22)

The head input concatenates the present branches — `static_enc(x^C)` always, the sequence branch for
`U`/`L1`/`O`/`OM`, `match_enc(x^M)` for `OM` — and, **for the outcome target only**, the current-
action embedding `E_act(a)` (a separate `nn.Embedding(N_FAM, 16)`, D22). With `n_branches` present
branches and `act_dim ∈ {0, 16}`, the head is
`Linear(n_branches·64 + act_dim → 64) → ReLU → Dropout(0.1) → Linear(64 → K)`, `K = 8` (selection) or
`6` (outcome1). Class probabilities are `softmax(logits)`; the training loss is multiclass cross-
entropy (`nn.CrossEntropyLoss`), scored out-of-sample by `log_loss_per_row`.

### 4.3 Training discipline (D46)

Each model is fit with **Adam** (`lr = 3e-3`, `weight_decay = 1e-4`, `batch_size = 512`) for up to
`max_epochs` with **early stopping** on the validation fold (`patience = 8`, restoring best-val
weights); when no external validation fold exists — the falsification refits — a seeded
`internal_val_frac = 0.2` slice is held out so the ordered `O` view still stops before overfitting and
generalises as well as its twin `L1` (`THEORY.md` §7). The compact hyperparameters (embed `16`, hidden
`64`, `≤ 2` layers) are the predeclared D46 defaults, identical across views, so the only thing that
varies view-to-view is the information the encoder is allowed to read — the equal-budget discipline of
SPEC §7, the deep analogue of WS3's fixed LightGBM grid. Code is device-agnostic (`resolve_device`,
`--device auto`): the *same* code runs on a desktop CPU and a Colab T4 (D46). Every fit records
`epochs`, `seconds`, `n_params`, `best_val` (runmeta-compatible, D13).

### 4.4 The factory adapter (D17)

WS6's native falsification runs through a **model-callback adapter** (`_Ws6Adapter`,
`make_ws6_model_factory`): unlike WS3's factory — which hands back a learner consuming a *pre-built
tabular view matrix* — the WS6 adapter consumes a **decision-table slice** directly and builds the
ordered-token sequences itself, because the whole point of WS6 is the learned representation of the
raw ordered tokens that the leakage-audited tabular views deliberately drop. Each `fit` fits one
`FeatureEncoders` from its own `X` and stores it; `predict_proba` reuses that fitted encoder set (no
refit at predict time). Encoders are deliberately **not** cached across `fit` calls — a permuted or
re-sliced table (as the permutation control produces) must yield correspondingly different sequence
features, so a stale cache would be a correctness bug.

### 4.5 The motif-rediscovery probe (the method)

The probe (`motif_repeat_probe`) is a gradient-free functional read-out of the null-world *selection*
`O` GRU. For each family `X`, it constructs two synthetic length-2 histories that **end in the same
token `X`** and differ only in the second-to-last pitch — `[X, X]` (repeat) and `[Y, X]` averaged over
`Y ≠ X` (mixed) — holding the static context at the training mean and the float channels in-
distribution (each token carries its family's mean physical profile; `dvelo`/`dloc` re-derived;
`_probe_next`). Because both end in `X`, an `L1` model predicts identically; only the ordered `O` model
can react to the earlier token. The read-out is
`suppression(X) = P(next = X | [Y,X]) − P(next = X | [X,X])`, positive when a third consecutive `X` is
made less likely after a repeat — the planted no-three-in-a-row motif — summarised by
`mean_suppression` and `frac_suppressed`, with
`motif_present = (mean_suppression > 0) ∧ (frac_suppressed ≥ 0.5)`. This measures a *different*
functional than the loss (a per-family counterfactual level vs a frequency-weighted average over
realized labels), and §5 reports that the two dissociate at demo scale (`THEORY.md` §6).

### 4.6 Scoring and the ablation statistics

Every view writes standard-schema predictions (SPEC §8.1) scored **only** through the shared harness
(`evaluate_predictions`, `compare_views`): multiclass log loss (primary) across the SPEC §6 slices,
and `Δ_order = Loss(min[U, L1]) − Loss(O)`, `Δ_matchup = Loss(O) − Loss(OM)` with pitcher-game
clustered CIs on validation and the locked test. `Δ_order` is read under the D21 min-bias rule
(negatively biased under the null → the criterion is "significantly positive"); `Δ_matchup` takes no
minimum and is read directly (`THEORY.md` §5). The capacity-matched twin delta `Loss(L1) − Loss(O)`
(no `min`, no `U`) is the D47 grammar gate, reportable on any slice via a `row_mask`
(`_l1_twin_delta_ci`).

---

## 5. Experimental setup

**Scoring and slices.** As §4.6: log loss through the harness, ablations with clustered CIs on
validation (2024) and the locked test (2025), across the six SPEC §6 slices. Selection is scored
like-for-like against the four count-based references (`eval/baselines.py`). The learned `O` view's
losses are compared to WS3's engineered GBDT-`O` and WS2's grammar-`O` — a **read-off** (their
reported O-view losses are pasted into the headline comparison line; WS6 refits nothing from other
workstreams).

**The `Δ_order` reading rule (D21).** `Δ_order` uses `min[U, L1]` and is negatively biased under the
null, so the criterion is "**significantly positive**" (clustered CI lower bound `> 0`); a small
negative reads as *consistent with no ordering effect*, never "order hurts," and this binds on both
targets. `Δ_matchup` takes no minimum and carries no such bias.

**Falsification protocol (completed validation, decision D47).** WS6 reruns the SPEC §11 oracle as a
real-model test through the D17 factory adapter. The following are *observed results* on the committed
WS6a run (null world, 6 epochs, demo scale), not aspirations.

- **Null world** (an ordered *selection* habit — a no-three-in-a-row tendency — but outcomes depending
  only on context + current family, by construction). The D47 verdicts **pass**:
  - **`GRU_GRAMMAR_DETECTED`.** The ordered selection GRU beats its capacity-matched twin `L1` on the
    **repeat-context subset** (the rows whose two most-recent prior pitches are the same family, where
    the habit is active and `O` can see the matching pair `L1` cannot): twin delta `+0.0115`, clustered
    CI `[+0.0064, +0.0161]`, `n_repeat_rows = 1136`. The **aggregate** selection `Δ_order` is `+0.0006`
    (not significant) — the habit binds only on repeat contexts, and the aggregate dilutes it (§6's
    targeting lesson; `THEORY.md` §5). This is the crux of the exhibit: *the same effect is detected on
    the subset it acts on and hidden in the aggregate.*
  - **`GRU_OUTCOME_QUIET`.** On the outcome target no ordered dependence is found: `Δ_order = −0.0004`,
    CI `[−0.0023, +0.0011]` (lower bound not `> 0`), the token-order permutation test does not fire
    (`p = 0.091`), and the locked-test outcome `Δ_order` is `−0.0011` (n.s.) — the correct null read
    under D21.
  - **`Δ_matchup` significantly negative on both targets** (`−0.0036`, CIs excluding 0). Because
    `Δ_matchup` has no `min`-bias, this is a genuine out-of-sample cost of the matchup feature block
    where no matchup effect exists — WS6 is the **third** model family (after WS3's GBDT and WS5's
    tabular MDP D42/greedy-optimism exhibits) to measure it, a robust cross-workstream pattern
    (§7; pre-written as the live `M−` branch of §6).

- **The opacity exhibit (motif rediscovery).** On the same null world the motif probe reports
  `motif_present = False`: `mean_suppression = −0.0063`, only `43%` of families suppressed
  (`FF: +0.05` suppressed, but `FC: −0.057` anti-suppressed). So the GRU's **loss** detects the grammar
  (the repeat-context twin delta is CI-positive) while its explicit per-family **probabilities** do not
  cleanly expose the rule at demo scale — the learned representation uses the pattern opaquely, the
  interpretability contrast with WS2's explicit motif table (§7; `THEORY.md` §6).

- **Positive world** (a planted whiff boost when `|velo_{t−1} − velo_{t−2}| ≥ 5` mph — keyed on the
  transition *into* the prior pitch, per D21, so it separates `O` from `L1`; the `dvelo` channel
  carries the mechanism directly). Positive-world mechanism recovery is covered at **direction level**
  in the lean pipeline test at small scale: the outcome `O` model's predicted whiff lift on triggered
  vs untriggered rows is positive (`recovered_sign_ok`), the honest first-class verdict
  `GRU_MECHANISM_DIRECTIONAL` (mirroring WS4/WS5). Certifying the *magnitude* — a recovery ratio like
  WS3's `0.955`, the full `GRU_MECHANISM_RECOVERED` verdict requiring the aggregate outcome `Δ_order`
  CI above 0 **and** the permutation test to fire — is a full-scale Phase-2 question (`{PLACEHOLDER}`).

**The Pareto row (completed validation).** The five-view compute coordinates (selection target, 6
epochs, `~1.0` GB peak): `C 6,856` params / `62.6 s`; `U 13,512` / `74.8 s`; `L1 31,112` / `81.1 s`;
`O 31,112` / `145.9 s`; `OM 36,616` / `110.1 s`. **`L1` and `O` print identical parameter counts** —
the capacity match made visible — so the `O`-over-`L1` question is a pure gain-per-*second* question
(`O` costs `~1.8×` `L1`'s wall-clock for the same size, the recurrence over full sequences;
`THEORY.md` §8). The whole demo ran in `~28` minutes.

**Scope of the synthetic battery.** As with WS3, the permutation battery's refits are affordable only
at synthetic scale; on real data the central table carries the ablation and the synthetic worlds carry
the recovery/permutation checks (the Colab path skips `n_perm` on real data). The optional compact
Transformer (SPEC §12.4) is a *demo*, not an acceptance gate, exercised by a single smoke test.

---

## 6. Results

### 6.1 The central table (real data)

The headline: per-view losses on both targets, with the references and the WS2/WS3 comparison, plus
both ablations with clustered CIs.

| view | params | selection log loss | outcome-1 log loss | reference (selection) |
|---|---|---|---|---|
| `C`  | 6,856  | {C_SEL_LL}  | {C_O1_LL}  | `pitcher_count` {REF_PITCHER_COUNT} |
| `U`  | 13,512 | {U_SEL_LL}  | {U_O1_LL}  | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |
| `L1` | 31,112 | {L1_SEL_LL} | {L1_O1_LL} | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |
| `O`  | 31,112 | {O_SEL_LL}  | {O_O1_LL}  | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |
| `OM` | 36,616 | {OM_SEL_LL} | {OM_O1_LL} | `pitcher_count_prev` {REF_PITCHER_COUNT_PREV} |

**Representation read-off (the WS6 point):** learned `O` selection loss `{O_SEL_LL}` vs **WS3** GBDT-`O`
`{WS3_O_SEL_LL}` and **WS2** grammar-`O` `{WS2_O_SEL_LL}`; learned `O` outcome-1 loss `{O_O1_LL}` vs
**WS3** GBDT-`O` `{WS3_O_O1_LL}`.

**Ablation, validation (clustered 95% CI):**

| target | `Δ_order` = min[U,L1] − O | repeat-context twin `L1 − O` | `Δ_matchup` = O − OM |
|---|---|---|---|
| selection | {DELTA_ORDER_SEL} {DELTA_ORDER_SEL_CI} | {TWIN_SEL} {TWIN_SEL_CI} | {DELTA_MATCHUP_SEL} {DELTA_MATCHUP_SEL_CI} |
| outcome-1 | {DELTA_ORDER_OUTCOME1} {DELTA_ORDER_OUTCOME1_CI} | — | {DELTA_MATCHUP_OUTCOME1} {DELTA_MATCHUP_OUTCOME1_CI} |

**Locked-test confirmation (2025):** outcome-1 `Δ_order` = {DELTA_ORDER_OUTCOME1_TEST}
({DELTA_ORDER_OUTCOME1_TEST_CI}). Per-slice outcome-1 `Δ_order`: {DELTA_ORDER_BY_SLICE}. Pareto row on
real data (params / epochs / wall-clock per view): {PARETO_BY_VIEW}.

### 6.2 Branched interpretation — three axes

The result is read on three independent axes. The **representation** axis (R+/R=/R−) is WS6's actual
question — does the *learned* `O` beat WS3's *engineered* `O`? The **order** axis (H1/H2/H3) is the
within-WS6 `Δ_order`, read under D21 **and with the targeting lesson applied** (check the repeat-
context / slice deltas, not just the aggregate). The **matchup** axis (M+/M0/M−) is `Δ_matchup`, read
directly and in the three-family cross-workstream context. Exactly one branch on each axis applies;
the notebook's §9 branch selector prints which, and the write-ups below stand alone once the numbers
are filled.

#### Representation axis (learned `O` vs engineered WS3-`O`)

**R+ — the learned representation beats the engineered features.** GRU-`O` posts a lower loss than
WS3's GBDT-`O` (and WS2's grammar): the learned encoder found ordered signal the hand-designed
positional slots and consecutive-difference features (D11) missed. This is the strong deep-learning
result — but *before believing it*, inspect **what** it found: run the motif probe and the per-slice
deltas on the real-data model, confirm the edge concentrates where order should matter (`long_pa`,
`two_strike`) and is not an artifact of the GRU's extra smoothing of continuous physics. If it
survives, WS6 is the rung that justifies going beyond engineered features, and the cross-workstream
paper leads with it.

**R= — the learned representation ties the engineered features.** GRU-`O` `≈` WS3-`O` within CIs: the
engineered tabular encoding already captured the available ordered signal, and the learned
representation neither added nor lost much. This is a **Pareto** statement as much as a loss one
(`THEORY.md` §8): if the two tie on loss, the far cheaper GBDT wins the frontier, and the honest
headline is "engineered features suffice; the learned model is not worth its compute here." It is also
the most likely outcome given SPEC §13, and it is a clean, publishable finding — the deep model
*confirms* the engineered result rather than overturning it.

**R− — the learned representation underperforms the engineered features.** GRU-`O` posts a *higher*
loss than WS3-`O`. This is almost never "learning can't help in principle"; at this data/compute scale
it is an **optimisation or data limit** — the recurrent fit under-trained, the budget too small, the
sequences too short to reward a recurrence over a tree. Read the **training curves** before concluding
(did val loss still fall at the epoch cap? did early stopping fire too soon?), check the capacity-
matched twin `L1 − O` (a negative twin at equal parameters points at optimisation, `THEORY.md` §4),
and report it honestly: a strong tabular baseline beating an under-budget deep model is a real,
defensible result about *this* comparison, not a claim about deep learning.

#### Order axis (within WS6, outcome-1 `Δ_order`, read under D21 **with the targeting lesson**)

**H1 — `O` beats both `U` and `L1` (CI lower bound > 0): genuine ordered dependence.** The fully
ordered view lowers loss below the better of `U`/`L1` by more than the clustered CI — ordered
information the learned encoder captured. Because of the targeting lesson (§5; `THEORY.md` §5), a real
ordered effect may be **stronger on the subset it acts on than in the aggregate**: check the repeat-
context / per-slice twin deltas, not only the headline `Δ_order`, and read the locked-test row. On the
outcome target this is the live finding-#2 signal the prescriptive workstreams try to use.

**H2 — `O ≈ U/L1` but both beat `C`: history matters, order does not.** History lowers loss below
context-only, but the fully ordered `O` does not refine on the better of `U`/`L1` (`Δ_order` not
significantly positive, *and* the repeat-context twin delta not positive either — the targeting check
that keeps H2 honest). Within-PA history carries value but its fine order beyond the previous pitch /
unordered bag does not — the SPEC §13 expectation ("O barely beats L1"), reported without
embarrassment.

**H3 — nothing beats `C`: no sequencing signal at all.** No history view lowers loss below context-
only. The cleanest finding-#2 null for the learned model. This does not deny finding #1 (selection
structure may still exist — the null world shows exactly that separation). Audit calibration and the
training curves before accepting, since an under-optimised recurrence can masquerade as H3.

#### Matchup axis (outcome-1 `Δ_matchup`, read directly, three-family context)

**M+ — `OM` beats `O` (CI lower bound > 0): real batter–pitcher adaptation.** The matchup branch
lowers loss below `O` — longer-term adaptation the learned cross-PA representation captured, which the
`first_pitch` slice (where matchup memory is `OM`'s only added information) must corroborate. Notable
because it would be the *first* positive matchup signal in the study: WS3 and WS5 both found the
matchup block **costs** out-of-sample.

**M0 — `OM ≈ O`: no detectable matchup memory.** The matchup branch neither helps nor hurts beyond `O`
(CI spans 0). A clean read, expected if within-game/season rematches are too thin for even a pooled
encoder to resolve.

**M− — `OM` significantly below `O`: matchup features cost out-of-sample.** The matchup branch has
significantly higher loss than `O`. Because `Δ_matchup` has no `min`-bias this is a *real* cost, not an
artifact — the same fragmentation reading as WS3 (`../ws3_gbdt_stack/THEORY.md` §6.3). This is
**observed on both synthetic WS6 targets** (`−0.0036`), and it is now a **three-family pattern**: WS3
(GBDT), WS5 (tabular MDP), and WS6 (GRU) all pay an out-of-sample cost for the matchup block where no
matchup effect exists. The honest reading is "no matchup signal a learned encoder can see **and** a
real cost of carrying the features," checked with the support diagnostics — that three independent
model families agree makes it a robust methodological statement about sparse-feature fragmentation,
not a claim about baseball.

#### Cross-reading the grid

The honest headline is a triple `(R, H, M)`. The study's **most anticipated** cell is
**R= × H2 × M−**: the learned representation confirms the engineered result (order barely helps, so
the cheap model wins the frontier) and the matchup block costs — a modest, defensible finding with a
clean compute-vs-gain reading and a three-family methodological note. The **strongest** cell is
**R+ × H1 × M+**: the learned model finds ordered *and* matchup signal the engineered features missed
— a live signal for the whole prescriptive phase, believed only after the probes confirm *what* it
found. The **cleanest confirmation-of-null** is **R= × H3 × M0**. Whichever triple fires, the finding
is read under the firewall (§7): a predictor measures *prediction*, and only the OPE/RL workstreams
can test whether any of it is *prescriptive*.

---

## 7. Discussion

**On the representation axis — the whole point of the rung.** WS6 exists to ask whether a learned
encoding beats engineered features and the explicit grammar. Under **R+**, it does, and the burden
shifts to *interpreting* the gain: the opacity exhibit (below) is exactly why "the GRU won" is not
self-explanatory — a learned edge must be probed before it is believed. Under **R=** (the SPEC §13
expectation), the durable contribution is a **Pareto** one: the deep model confirms that WS3's
engineered `O` already captured the available ordered signal, so the ladder's honest conclusion is
"engineered features suffice, and the compute peak did not buy new predictive signal." Under **R−**,
the honest content is a scale statement — a strong tabular baseline beat an under-budget recurrence —
read from the training curves, never inflated into "deep learning fails."

**On the opacity exhibit and interpretability.** The sharpest thing WS6 shows is a *dissociation*:
the GRU's loss demonstrably **uses** the planted grammar (the repeat-context twin delta is
CI-positive) while its explicit per-family probabilities do **not** cleanly expose the rule
(`motif_present = False` at demo scale). This is not a contradiction — loss is a frequency-weighted
average over realized labels, the probe is a uniform per-family counterfactual level on partly out-of-
support inputs, and they are allowed to disagree (`THEORY.md` §6). It is the deep-learning bargain in
one exhibit: WS2's grammar says the rule out loud (an inspectable motif table); WS6's GRU uses the
same rule but will not cleanly say it. For the cross-workstream paper this is the concrete case that
**interpretability is a property to be measured, not assumed** — an accuracy claim about a learned
model does not license a mechanism claim, and what more training/data would change is a Phase-2
question (`{PLACEHOLDER}` for the real-data rerun).

**On the targeting lesson.** The null world makes a general methodological point live: the planted
effect binds on a subset (repeat contexts), so it is detected on that subset (twin delta CI-positive)
and diluted to nothing in the aggregate (`Δ_order` null). An effect with support fraction `ρ` costs
the aggregate a factor `ρ` in point size and `√ρ` in power (`THEORY.md` §5), so **effects that bind on
a subset need subset-targeted evaluation** — read the slice deltas, not just the headline. This binds
on the real-data reading (the H-axis): a real ordered effect might live only in `long_pa` /
`two_strike` and be invisible in the aggregate, and the harness's slice machinery is built to find it.

**On the three-family `Δ_matchup` pattern.** WS6 is the third model family — after WS3's GBDT and
WS5's tabular MDP — to find that the matchup feature block *costs* out-of-sample on synthetic worlds
where no matchup effect exists. Three independent function classes (trees, a tabular MDP, a recurrent
net) agreeing turns a single-workstream curiosity into a robust cross-workstream statement: carrying
many sparse matchup features adds estimation variance without signal, and the honest move on a
negative `Δ_matchup` is to read it as fragmentation (checked with support diagnostics), not as a claim
that matchup memory is absent from baseball.

**The firewall.** WS6 measures whether ordered history *predicts* selection and outcomes out-of-sample
(findings #1 and #2). It does **not** establish that *changing* the sequence would change outcomes
(finding #3); identifying that requires ignorability and overlap public data cannot satisfy (SPEC §0),
which is what the OPE/RL workstreams test rather than assume (`../ws3_gbdt_stack/THEORY.md` §8).

---

## 8. Limitations

1. **Predictive, not causal (the firewall).** WS6 measures out-of-sample prediction (findings #1/#2),
   not the effect of *changing* the sequence (finding #3). A learned `O`-over-`L1` gain is ordered
   *predictability*, which the prescriptive workstreams must test for *exploitability*, not assume.
2. **Demo-scale honesty.** The completed validation is a 6-epoch synthetic run; the D47 verdicts pass
   and the exhibits reproduce, but the motif probe's per-family readout and the positive-world
   *magnitude* recovery sharpen with scale. The real-data run (30 epochs, full data) is Phase 2, and
   the positive-world verdict at demo scale is honestly `GRU_MECHANISM_DIRECTIONAL`, not
   `RECOVERED`.
3. **Capacity matching's limits.** Matched parameter counts (`L1 = O`) do **not** match inductive
   bias (`C`/`U` are non-recurrent function classes) or the optimisation landscape (`O` has the harder
   fit), so `L1 − O` is the clean order comparison and `U − O` is read cautiously (`THEORY.md` §4). A
   null-or-negative twin is consistent with "no order information" *or* "`O`'s harder optimisation
   swallowed a small effect at this scale."
4. **The compact budget bounds "the GRU couldn't find it."** A null `Δ_order` under H2/H3 means a
   *compact* GRU (embed 16, hidden 64, `≤ 2` layers, D46) within an early-stopped budget did not
   resolve ordered signal — not that no network could. A much larger model or longer training could in
   principle differ; the budget is fixed for fairness and CPU-runnability (SPEC §7).
5. **The Transformer demo is optional and out of scope** (SPEC §12.4): a demonstration for the long
   cross-PA test, kept out of the acceptance gates and exercised by a single smoke test — no
   Transformer result is a WS6 finding.
6. **`pitch_type` is a classifier output** and the action is at **family** granularity (SPEC §1, §4);
   a mechanism below the family level, or a mislabeled pitch, is only partially represented.
7. **Falsification is synthetic-only** (D47): the permutation refits are infeasible at `~3.85M` rows,
   so the real-data run carries the ablation and the synthetic worlds carry the recovery/permutation
   checks.
8. **No catcher/umpire effects** (absent from base Statcast, SPEC §3.2) and a **single reward metric**
   (`−delta_run_exp`).

---

## 9. Conclusion

WS6 is the study's deep-learning rung and its compute peak on the predictive side: the first
capacity-matched neural ablation of all five state views, with `L1` and `O` realised as one identical
GRU so the order comparison is parameter-exact, run on both targets with a locked-test row and read
against WS3's engineered features and WS2's explicit grammar. Its conclusion is branch-conditional and
complete once the real numbers arrive, on three independent axes:

- **Representation axis.** Under **R+**, the learned representation beats the engineered features — the
  rung that justifies going deep, believed only after the probes confirm what it found. Under **R=**
  (the SPEC §13 expectation), the deep model *confirms* WS3's engineered result and the cheap model
  wins the Pareto frontier — a clean compute-vs-gain statement. Under **R−**, a strong tabular baseline
  beat an under-budget recurrence, read honestly from the training curves.
- **Order axis.** Under **H1**, genuine ordered dependence (checked on the subset it acts on, not just
  the aggregate). Under **H2** (the expectation), history helps but its fine order does not. Under
  **H3**, no sequencing signal at all — with the targeting lesson ensuring a subset-bound effect was
  not missed by aggregation.
- **Matchup axis.** Under **M+**, the study's first positive matchup signal. Under **M0**, none. Under
  **M−** — observed on both synthetic targets and now a **three-family** pattern (WS3, WS5, WS6) — the
  matchup block costs out-of-sample, read as sparse-feature fragmentation.

Across every cell the durable contributions are the same: a capacity-matched deep ablation that
measures information rather than size; the **targeting lesson**, formalized and shown live (subset-
bound effects need subset-targeted evaluation); the **opacity exhibit** (the GRU uses the grammar its
probabilities will not cleanly expose — interpretability measured, not assumed); the three-family
`Δ_matchup` finding; and dual-path reproducibility — all read under the finding-#2 firewall that keeps
a learned predictor from being mistaken for a causal one.

---

## References

Bai, S., Kolter, J. Z., and Koltun, V. (2018). An empirical evaluation of generic convolutional and
recurrent networks for sequence modeling. *arXiv:1803.01271*.

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y.
(2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation.
*Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

Efron, B., and Morris, C. (1975). Data analysis using Stein's estimator and its generalizations.
*Journal of the American Statistical Association*.

Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.

Kingma, D. P., and Ba, J. (2014). Adam: a method for stochastic optimization. *arXiv:1412.6980*
(International Conference on Learning Representations, 2015).

Marchi, M., and Albert, J. (2013). *Analyzing Baseball Data with R*. Chapman and Hall/CRC.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: a
simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*.

Sutskever, I., Vinyals, O., and Le, Q. V. (2014). Sequence to sequence learning with neural networks.
*Advances in Neural Information Processing Systems (NeurIPS)*.

Tango, T. M., Lichtman, M. G., and Dolphin, A. E. (2007). *The Book: Playing the Percentages in
Baseball*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and
Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems
(NeurIPS)*.

Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., and Smola, A. J. (2017). Deep
sets. *Advances in Neural Information Processing Systems (NeurIPS)*.
