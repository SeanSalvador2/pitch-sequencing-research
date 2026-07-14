# WS3 in Plain English: The Centerpiece

This is the intuition guide for Workstream 3. No equations you have to solve — just the ideas,
in order, so that when you read the paper or the notebook the math is already familiar. If you
only read one document about WS3, read this one first.

---

## What WS3 is, and why it's the centerpiece

The whole project is a **ladder**. Each rung is a model that asks *does the sequence of pitches
already thrown help you understand the next pitch — and its outcome?* WS1 was a lookup table
(the bottom rung, and it showed the *support problem*). WS2 was a grammar (it measured how deep
the ordered *selection* structure goes). **WS3 is the centerpiece**, and it earns that name for
three reasons at once: it is the **first model that runs all five views** (`C`, `U`, `L1`, `O`,
`OM`) through one strong engine, so the headline order-and-matchup comparison is finally
apples-to-apples across the whole ladder; it is the **first model that scores outcomes, not just
selection**, so it carries the project's central number — does ordered history help predict
*what happens* on the pitch, after we already know which pitch it is; and it is the **supply
depot for everything prescriptive** — WS4, WS5, and WS7 don't build their own outcome model,
they *import* WS3's. If WS3 says order doesn't help outcomes, be skeptical of anything fancier
that claims it does.

---

## The metaphor: five nested photographs of the same at-bat

Every model in the study looks at the same decision — the choice made just before a pitch — but
through five different amounts of detail. Picture five photographs of the same moment, each
showing a little more:

- **C (context).** A snapshot of *just the situation*: the count, the base-out state, the inning
  and score, who's batting and throwing, and this pitcher's general repertoire. No pitch history
  at all.
- **U (unordered).** C, plus *the bag of pitches thrown so far this at-bat* — two fastballs and a
  slider — but with the order scrambled. You know what's in the bag, not the sequence.
- **L1 (last pitch).** C, plus *the single most recent pitch*. "The last one was a curveball."
- **O (ordered).** C, plus *the full sequence in order* — the last three pitches in their slots,
  how much the velocity jumped, how long the current same-pitch streak is. This is the photograph
  that can tell *fastball-then-slider* from *slider-then-fastball*.
- **OM (matchup memory).** O, plus *the long-term history between this batter and this pitcher* —
  what happened in earlier at-bats this game, and this batter's tendencies against each pitch
  family.

The five photographs are **nested** — each contains the one before it — so the *differences*
between what the model can predict from each are the sequencing evidence. If the ordered
photograph `O` predicts outcomes better than the last-pitch photograph `L1`, then *order* carries
information beyond just "what was the last pitch." That comparison has a name, `Δ_order`, and
it's the headline. The matchup comparison, `OM` vs `O`, is `Δ_matchup`.

---

## Trees and boosting, in plain words

WS3's engine is a **gradient-boosted decision tree** ensemble (the library is LightGBM). Here's
the whole idea without a single formula.

A **decision tree** is a little flowchart of yes/no questions: *Is the count two strikes? Was the
last pitch a fastball? Did the velocity jump more than 5 mph?* Follow the answers down to a leaf,
and the leaf holds a guess. One tree is weak — a handful of questions can't capture much.

**Boosting** fixes that by building a *committee* of tiny trees, one after another, where each new
tree is trained to fix the mistakes the committee has made so far. The first tree makes a rough
guess; the second tree looks at where the first was wrong and patches it; the third patches what's
left; and so on for hundreds of little trees, each nudging the answer a bit (that "bit" is the
*learning rate*). The final prediction is the whole committee's vote. It's a committee of tiny
yes/no question-askers, each one hired to clean up after the last.

Why this engine for this job? Because trees can ask questions about **pitch physics**, not just
pitch names. WS1's tables were keyed on names (fastball, slider), so they could only see a
velocity effect indirectly. A tree can ask *"did the speed jump more than 5 mph from two pitches
ago?"* directly — and that turns out to matter enormously (see the two-world story below).

WS3 builds this committee **five times** — once per photograph — for two jobs: a **behavior**
model (guess the next pitch family) and an **outcome** model (guess what happens when a given
pitch is thrown). The outcome model is actually a little assembly: one committee guesses the
pitch *result* (ball/strike/whiff/foul/in-play), another guesses *what kind of hit* if it's put
in play, and a small lookup table says how many runs each ending is worth in that count. Multiply
and add, and you get the expected run value of the pitch.

---

## The central table, and exactly how to read it

The heart of WS3 is one table: for each of the five photographs, the selection loss, the outcome
loss, and the run-value error — plus the two headline comparisons. Two rules for reading it, and
they are different for the two comparisons.

### `Δ_order` — read it with the D21 rule

`Δ_order` compares the ordered view `O` against *the better of* `U` and `L1`. "The better of two"
is the catch. Even when order truly adds nothing, picking the luckier of two noisy scores flatters
the baseline, so `Δ_order` **leans slightly negative even under a true null**. The rule (decision
D21, binding on the whole study):

- **Significantly positive** (the confidence interval clears zero) = **real ordered effect.** This
  is the only result that claims order carries outcome value.
- **About zero, or a small negative** = **"consistent with no ordering effect."** *Never* read a
  small negative as "order hurts" — it's the built-in optimism of taking the better-of-two, not a
  real cost.

The outcome `Δ_order` is the project's central number. SPEC's honest expectation is that it's
**small** on real data — order barely beating the last pitch. That's a clean, publishable result,
not a disappointment.

### `Δ_matchup` — no bias, so a negative really means it hurt

`Δ_matchup` compares `OM` against `O`. Crucially, there's **no "better of two"** here — it's a
straight subtraction. So there's **no optimism to correct for**, and the reading is direct:

- **Significantly positive** = matchup memory genuinely helps.
- **About zero** = no detectable matchup effect.
- **Significantly negative** = the matchup features **actually made it worse out-of-sample.** And
  because there's no bias to blame, this one is *real*: piling ~21 mostly-empty matchup columns
  into the model makes it noisier without adding signal, so it generalizes worse. This is not a
  statistical mirage — it's a genuine cost.

We *saw* this negative on both synthetic test worlds (where we know there's no matchup effect), so
it's a live possibility on real data. If it happens, the honest reading is **"no matchup signal we
can see, and the matchup features are costing us"** — check the support diagnostics (are those
columns mostly empty?) to confirm it's fragmentation, then hand the cross-at-bat memory question
to the later learned-representation model (WS6) instead of forcing it into a tree.

---

## The two-world validation, and the 0.955-vs-0.03 contrast

Before trusting WS3 on real data, we run it on two synthetic worlds where we planted the truth
ourselves. These are **completed, reproducible results**, not aspirations.

**The null world.** Outcomes here depend only on the situation and the current pitch — there is
*no* ordered outcome effect to find. But the simulator does plant an ordered *selection* habit (a
mild "don't throw three of the same in a row"). WS3 gets both right, and this is the beautiful
part — it separates the two findings live:

- The **selection** `Δ_order` is **+0.0088** (clears zero): it *detects* the planted selection
  habit (finding #1). ✓
- The **outcome** `Δ_order` is **−0.0033** (does not clear zero): it correctly finds *no* outcome
  order effect (finding #2 is truly absent). ✓
- The permutation test doesn't fire (p = 0.88). Verdict: **`NULL_QUIET`** — the model does not
  hallucinate outcome structure that isn't there.

**The positive world.** Now we plant a real effect: when the pitcher makes a big *velocity jump*
into the previous pitch, the next pitch gets more whiffs. WS3 recovers it almost exactly:

- The **outcome** `Δ_order` is **+0.0277** on validation and **+0.0382** on the locked test — a
  clear, confirmed ordered outcome effect. ✓
- The mechanism ablation points straight at the velocity channel, and `o_velo_delta_last` (the
  velocity-jump feature) is the **single most important feature** in the outcome model. ✓
- It recovers **95.5%** of the planted whiff lift (0.294 recovered of 0.308 planted).

**The contrast — this is the exhibit that justifies the whole ladder.** Remember WS1's tables
recovered only **~3%** of this *same* planted effect, because they were keyed on pitch *names* and
couldn't see velocity. WS3 recovers **95.5%**, because it has the velocity jump as an actual
feature. Said plainly: **when the model can see the actual mechanism, it recovers 95% of it; when
it can only see pitch names, 3%.** Same effect, same data — the difference is entirely what the
model is allowed to look at. That's why one rung isn't enough, and why WS3 is the centerpiece.

---

## What the q̂ grid is, and who uses it

Besides the ablation, WS3 publishes a **q̂ grid** — think of it as a **what-if card for every
decision**. For a single pitch decision, the card lists the model's expected run value for *each
of the 8 pitch families*, as if each were the one thrown:

> *In this 1-2 count to this lefty, after fastball-then-slider: fastball → +0.012, sinker →
> +0.008, cutter → +0.015, slider → +0.021, curve → +0.018, change → +0.006, split → +0.014.*

Alongside it, WS3 publishes the **propensities** — the behavior model's guess at what the pitcher
*actually tends to throw* here — and the **feasibility mask** (which families this pitcher throws
often enough to be a real option). That combination — *what's it worth for each pitch, what does he
usually throw, what can he throw* — is exactly what the prescriptive workstreams consume:

- **WS4** (the bandit) reads the q̂ grid to pick a "best next pitch" as a target policy.
- **WS5** (the MDP) reads the node values to value a *setup* pitch that pays off later in the count.
- **WS7** (offline RL + OPE) reads both to evaluate whole-at-bat policies honestly.

None of them refit their own outcome model — they all import WS3's saved artifacts. That's the
"supply depot" role.

**One caution, and it's important.** The q̂ grid is a *prediction* — "if pitch X were thrown here,
here's the expected outcome." It is **not** a promise that *changing* to pitch X would *cause* a
better result. We never see the pitch that wasn't thrown, and the reasons a pitcher chose what he
chose (the catcher's target, the scouting report) are invisible and affect the outcome too. So the
grid is finding #2 (a careful prediction), and only the OPE machinery downstream can try to turn it
into finding #3 (a recommendation) — and it refuses to answer when its estimators disagree. WS3
draws that line on purpose.

---

## What the result will mean (the branches)

When we run WS3 on the real data, the answer falls on two independent questions. The notebook and
paper have the full write-ups; here's the plain version.

**Does *order* help predict outcomes? (the H-axis)**

- **H1 — yes, clearly.** `O` beats both `U` and `L1` (the interval clears zero). Real ordered
  outcome value. Check it concentrates where it should (longer at-bats, two-strike counts) and that
  the locked-2025 test confirms it, then hand it to the prescriptive phase.
- **H2 — history helps, order doesn't.** The history views beat context-only, but the *ordered* one
  doesn't beat the last-pitch/bag views. This is the *expected* result (SPEC predicted "O barely
  beats L1"): sequences carry a little outcome value, but their fine order doesn't. Clean and
  honest.
- **H3 — nothing beats context.** No history view helps outcomes at all. The cleanest finding-#2
  null. (It does *not* deny finding #1 — selection can still be structured, exactly as the null
  world shows.)

**Does *matchup memory* help? (the M-axis)**

- **M+ — yes.** `OM` beats `O`. Real batter-vs-pitcher adaptation — and it must show up on the
  first-pitch slice (where matchup memory is the *only* thing OM adds).
- **M0 — no effect.** `OM ≈ O`. No detectable matchup memory.
- **M− — it costs.** `OM` is significantly *worse* than `O`. The matchup features hurt
  out-of-sample (we saw this on both synthetic worlds). Read it as "no matchup signal + a real
  fragmentation cost," confirm with the support diagnostics, and route cross-at-bat memory to WS6.

The two axes combine into a pair — the most anticipated is "history helps a little, order doesn't,
matchup costs" (H2 × M−); the strongest is "real order *and* real matchup" (H1 × M+); the cleanest
null is H3 × M0.

---

## How to run it

Everything is in **RUNBOOK.md, step WS3** (sub-steps WS3.1, WS3.2, WS3.3). WS3 is the heaviest CPU
step so far, so it's **split into stages with per-view checkpointing** — a finished `(view, stage)`
writes a `.done` marker and a re-run skips it, so you can run a few views at a time and resume
cheaply. From the repo on your machine:

```powershell
conda activate statcast; cd ~\pitch-sequencing-research

# WS3.1 — behavior model mu(a|s) per view (next-pitch guess)
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage behavior --views C U L1 O OM --tune --threads 4

# WS3.2 — decomposed outcome stack per view (E[R|s,a])
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage outcome --views C U L1 O OM --tune --threads 4

# WS3.3 — assemble the q-grid + propensities, then score the central table
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage assemble --views C U L1 O OM --threads 4
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage eval --views C U L1 O OM
```

Run views one at a time if memory is tight (`--views O`, then `--views OM`, …) — each checkpoints
independently. Drop `--tune` for a faster single-fit run on the predeclared defaults. The two quick
synthetic checks need no data and take about a minute or two each (they run the whole pipeline plus
the falsification battery):

```powershell
python workstreams/ws3_gbdt_stack/run_ws3.py --synth null --stage all --out results/ws3_null/
python workstreams/ws3_gbdt_stack/run_ws3.py --synth positive --stage all --out results/ws3_pos/
```

**What to paste back**, per RUNBOOK: the printed **headline block**, one view's metadata sidecar
(`Get-Content results/ws3/outcome_O.json`), and the decomposed-vs-direct summary. **What to look at
first**, in order:

1. **The central table's outcome `Δ_order`**, read with D21. A CI above zero is the real ordered
   outcome effect (finding #2); small-negative is "consistent with no effect," never "order hurts."
   Then read the **locked-test** row as the 2025 confirmation.
2. **`Δ_matchup`**, read directly (no bias). A significant negative is a real fragmentation cost —
   check the support diagnostics before concluding anything about matchup baseball.
3. **Feature-importance sanity.** Count and pitcher features should dominate the `C` view; the
   history views add the slot/transition features on top.
4. **Decomposed-vs-direct agreement.** Should be small (well under 0.03) per view; a flagged view
   (the demo flags `OM`) means the tidy assembly and the black-box regressor disagree — investigate
   the node-value lookups.
5. **Calibration.** The run-value slope near 1 and intercept near 0.

---

## Mini-glossary

- **Boosting** — building a committee of tiny decision trees one at a time, each trained to fix the
  running committee's mistakes. The engine behind every WS3 model. More trees, each nudging gently.
- **Gain** — how much a feature improved the model's fit across all its splits; WS3's measure of
  feature importance. "The top-gain feature" = the question the trees found most useful.
- **Propensity** — the behavior model's estimate of how likely the pitcher is to throw each family
  in a given situation, `μ(a | s)`. The "what does he usually do here" number the OPE math needs.
- **q-value (q̂)** — the expected run value of a specific pitch in a specific situation,
  `E[R | s, A = f]`. The "what-if card." A *prediction* under a hypothetical pitch, not a causal
  promise.
- **Calibration** — whether the model's numbers mean what they say: when it predicts 0.2 runs, does
  ~0.2 runs actually happen? Reported as a slope (want 1) and intercept (want 0), and as reliability
  curves for the probabilities.
- **Ablation** — deliberately removing information (here, using a smaller photograph) and measuring
  how much worse the model gets. `Δ_order` and `Δ_matchup` are ablations; the loss difference *is*
  the value of the removed information.
- **Overfit / fragmentation** — when a model has so many features (relative to its data) that it
  fits noise and generalizes *worse*. The reason a negative `Δ_matchup` is a real cost: the extra
  matchup columns add noise, not signal.
- **The firewall** — the rule that keeps finding #2 (prediction) separate from finding #3 (cause).
  WS3 predicts; only the OPE/RL workstreams try to prescribe, and only within support.
