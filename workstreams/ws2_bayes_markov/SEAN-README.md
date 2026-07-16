# WS2 in Plain English: The Pitch Grammar

This is the intuition guide for Workstream 2. No equations you have to solve — just the ideas,
in order, so that when you read the paper or the notebook the math is already familiar. If you
only read one document about WS2, read this one first.

---

## What we actually found (2021–2025)

Here is the headline in plain words. We ran the grammar on five full seasons — about **3.57
million** regular-season pitch decisions — training on 2021–2023 and scoring on 2024.

**Real pitchers are sticky.** Every one of the top "rules" the grammar found is a *repeat* rule:
after a pitch of a given family, that same family becomes **more** likely next, not less.
Split-finger after split-finger jumps from 24% to 38%; slider after slider from 31% to 39%;
changeup after changeup from 21% to 30%. This is the single cleanest descriptive finding on the
whole selection side of the study — and it is the **exact opposite** of the "don't throw three in
a row" habit we planted in our synthetic test world. Real major-league pitchers double and triple
up more than the textbook expects.

**Order beyond the last pitch doesn't pay — and the data said so itself.** The grammar fits a dial
per depth for how much that depth is worth. Depth 1 (the previous pitch) earned a moderate dial
(**17.9**) — worth using. But depths 2, 3, and 4 came back at **84.7, 113.0, 166.7** — climbing
straight toward the "pool it away, it's worthless" ceiling. The model voted against depth all by
itself, and more strongly the deeper it looked. The mean "effective order" landed at about **0.9** —
under one pitch of real memory.

**In fact, using the deeper history made the forecast slightly *worse*.** The ordered view O scored
a touch worse than the last-pitch view L1 (the gap `delta_order_L1` was **−0.0056**, a clean
measurement with no bias to correct). That is not "order hurts baseball" — it is *fragmentation*:
chopping the data into depth-2/3/4 contexts spreads it too thin, so the deeper model overfits and
generalizes worse. The honest reading is "one pitch of memory is all this grammar can bank."

**A number that looks alarming but isn't: the "bits" came out negative.** `B_seq` — the extra
next-pitch predictability the sequence supplies — was **−0.0221** overall. Negative bits *sound*
like "the sequence makes the next pitch harder to guess," but that isn't what it means. It's the
same depth-overfit in a different mirror: the ordered predictor, overfit past depth 1, scores below
the plain context predictor on the realized pitch. The fingerprint gives it away — the bits are
exactly zero on the first pitch (no history yet) and get steadily *more* negative the deeper into
the at-bat you go, which is precisely how an overfit-with-depth artifact behaves. A clean,
calibrated bits number will come from WS3's regularized models, not this raw grammar.

**Why the automatic verdict says "GRAMMAR_NOT_DETECTED" — and why that's fine.** The detector was
built around the synthetic habit, which required the ordered view to *beat* the last-pitch view. On
real data it doesn't (that's the fragmentation above), so the switch flips to "not detected." But
its other readouts are the real story: stickiness *is* detected (the repeat motifs), and when we
shuffle the histories to destroy any real order, the machinery collapses exactly as it should. The
label is about the synthetic-shaped deep grammar; it is not a claim that real selection is
structureless.

One thing has *not* changed: this is all still about *selection* — what gets thrown next. None of
it says a sticky pitcher is easier to *hit*. That question (does the sequence change the *outcome*?)
is WS3's, and the firewall between the two is the whole point of the ladder.

---

## What WS2 is, and where it sits

The whole project is a **ladder**. Each rung is a model that tries to answer one question:
*does the sequence of pitches already thrown help you understand the next pitch?* We climb from
the simplest, most transparent model to the most powerful.

WS1, the bottom rung, was a **lookup table**: chop the data into ever-finer cells and read the
pitch mix off each. Its lesson was the *support problem* — deepen the key and the cells empty
out before they tell you anything, so a flat table can barely see order at all.

**WS2 is the next rung up: a *grammar*.** It keeps WS1's shrinkage idea but spends its
resolution intelligently — it uses as much pitch history as the data can actually support, and
falls back gracefully to less history where it can't. It answers a sharper question than WS1
could: not just "does order matter?" but *"how many pitches back does the order actually reach,
and what are the rules?"*

One thing to hold onto from the very start: **WS2 is only about selection — what gets thrown
next.** It says nothing about whether a sequence *works* (gets outs). That firewall matters, and
we'll come back to it.

---

## The one metaphor: plate appearances are sentences

Here is the whole idea in one picture.

Treat each **plate appearance as a sentence**, and each **pitch family as a word** (fastball,
slider, curve, …). A pitcher, over a plate appearance, is writing a little sentence:
*fastball, fastball, slider.* Some word sequences are common; some are avoided. The question WS2
asks is exactly the question a phone's autocomplete asks: **given the words so far, what's the
next word likely to be?**

That reframing is not just cute — it is literally the model. WS2 is a *language model* whose
vocabulary is 8 pitch families instead of English words. The same math that powers
"how-good-a-guess is the next word" for text powers "how good a guess is the next pitch" here.

- A **context** is the last few pitches of the current at-bat, *in order*.
- A pitcher who just threw *fastball, fastball* is in the context `(fastball, fastball)`.
- WS2 predicts the next family from that ordered context — and from the count, the batter's
  handedness, and who's pitching.

---

## The phone-keyboard analogy: use as much history as you can back up

Here is how WS2 spends its history, and it is exactly how a good autocomplete works.

Imagine your phone predicting your next word. If it has seen the exact three-word phrase you're
typing thousands of times, it uses all three words — that's a confident, specific guess. If it's
only seen your two-word phrase, it backs off and uses two. If even that's rare, it falls back to
"what word usually follows this one word," and in the worst case just "what word do you use a
lot." **It uses as much context as it has evidence for, and gracefully falls back when it runs
out.**

WS2 does exactly this over pitches, and the name for it is **backoff**:

- Start with the deepest context the at-bat allows — say the last 4 pitches.
- If that exact ordered sequence is well-supported in the data, lean on it.
- If it's thin, **drop the *oldest* pitch** and back off to the last 3 — then the last 2, then
  just the previous pitch, then finally no history at all (just "what does this pitcher throw in
  this count?").

Dropping the *oldest* pitch each time is the key move: the recent pitches matter most, so when
you have to forget one, you forget the distant one first. (In the paper this is called the
"suffix" — the tail end of the context.) The beautiful part is that WS2 doesn't switch between
these levels with a hard rule; it **blends** them, leaning hardest on the deepest level the data
supports and smoothly mixing in the shorter ones. That blend is the "variable-order" in
"variable-order Markov."

---

## What the fitted "concentration" numbers mean: the data votes

WS2 doesn't guess how much to trust each depth of history — it **fits a number for each depth**,
called the **concentration** (written `α`). You can read it as a vote:

- **A moderate concentration at depth `k`** means *"the data says pitch-`k`-back is worth
  paying attention to — trust these contexts, don't pool them away."*
- **A huge concentration at depth `k`** (it runs up to a ceiling of a million) means *"pitch-`k`-
  back adds nothing beyond the shorter context — pool it all the way back."*

So for each depth, **the data votes on whether that depth is worth anything at all**, and the
concentration is the tally. This is why WS2 can honestly say "order reaches about 2 pitches back"
or "only the previous pitch matters" — it's not our opinion, it's what the fitted concentrations
report.

The cleanest demonstration comes from our test world (next section): the concentrations for
depths 1 and 2 come back *moderate* (there's real order-2 structure to find), while depths 3 and
4 shoot to the ceiling (there's genuinely no order-3 or order-4 structure). The model finds
order *exactly as deep as the data goes* — and stops.

---

## The twist: our "null" world is WS2's *positive* control

Every other workstream in this project uses a **null world** — a synthetic world built to have
*no* effect — as a sanity check that the model doesn't hallucinate structure. For WS2, that same
world flips its meaning, and this is worth understanding because it's initially confusing.

Here's why. The null world was designed (decision D20) to separate two different questions:

- Does the sequence change the **outcome** (whiffs, outs)? In the null world: **no, never.** The
  outcome depends only on the situation and the current pitch.
- Does the sequence change **what gets thrown next** (selection)? In the null world: **yes, on
  purpose.** The simulator plants a habit — a mild *no-three-in-a-row* tendency, where a pitcher
  who just threw two of the same family is less likely to throw a third.

For an *outcome* model (WS1's run-value tables, WS3's outcome stack), the null world is a true
null — there's nothing to find, and finding nothing is the correct answer.

But **WS2 is a *selection* model**, and that planted no-three-in-a-row habit is *exactly* an
ordered selection rule — an order-2 grammar rule. So for WS2, **the null world is the world with
a signal in it.** It's a **positive control**: WS2 is *supposed* to find the planted order. If it
didn't, the grammar would be broken.

WS2's *negative* control — the "make sure it's not hallucinating" test — is a different
operation entirely: we **shuffle the pitch histories** among similar situations (same count, same
pitch number), keeping each pitch's real next-family but pairing it with someone else's history.
That destroys any real order. If WS2's "order effect" was genuine, it must **collapse** to nothing
under the shuffle. If it survived, it was never really about order. WS2 has to pass both: **detect**
the planted order on the straight null world, and **collapse** under the shuffle.

---

## Here is what our synthetic check actually showed

We ran the full WS2 pipeline on the null world (its positive-control role), and it behaved
exactly as it should. These are real, reproducible numbers, not aspirations:

**It detected the planted grammar (`GRAMMAR_DETECTED`).**

- The full-order view **O** predicted the next pitch better than the previous-pitch view **L1**:
  log loss **1.3322 vs 1.3385**. The gap (`delta_order_L1`) was **+0.0063**, with a confidence
  interval of **[+0.0041, +0.0080]** — comfortably above zero, so it's real, not noise.
- The **effective order** — how deep the grammar reached — was a clean order-2 signature: about
  **30% of pitches** earned depth 2, the rest fell back to the base, and *nothing* earned depth 1,
  3, or 4. That's the fingerprint of a *pure* order-2 planted habit: the model found order exactly
  where the simulator put it.
- Every one of the top **motifs** (the grammar's "rules") was the planted no-three-in-a-row habit,
  correctly signed: `SI SI → P(SI) 0.35→0.22`, `FF FF → 0.31→0.20`, and so on — "after two of a
  family, that family becomes *less* likely." Exactly right.

**It collapsed under the shuffle (`COLLAPSES_UNDER_PERMUTATION`).** Once we scrambled the
histories, the order edge fell from **+0.0063 to +0.0001**, and the depth-2 mass fell from
**0.30 to 0.00**. The "order" was genuinely order — destroy it, and it's gone.

**The bits read-out was textbook.** WS2 reports `B_seq` — the extra *bits* of next-pitch
predictability the sequence buys (more on this next). Overall it was **+0.012 bits**, essentially
**zero on the first two pitches** (no history yet), and rising as the at-bat went on and the count
got deeper (two-strike **+0.026**, three-ball **+0.047**). Bits appear exactly where order can
exist — later in longer at-bats, in higher-leverage counts.

One honest wrinkle we report rather than hide: on this synthetic world the context-only view **C**
loses about 0.006 to a quick "pitcher × count" baseline, because handedness carries no signal in
the simulator. That's a known artifact of the fake world (WS1 has the same one), and it flips on
real data where handedness matters.

---

## What `B_seq` means: bits the batter could steal

`B_seq` is the answer to a spy's question: *if a batter watched the sequence of pitches, how much
would it help him guess the next one — beyond just knowing the count and the pitcher?* We measure
that "help" in **bits**.

A **bit** is one yes/no question's worth of information. If knowing the sequence let a batter
answer one perfect yes/no question about the next pitch, that's 1 bit. So how big is our
**0.012 bits**? It's about **one-eightieth of a yes/no question per pitch** — real, and pointing
the right way, but *tiny*. Over a whole at-bat it adds up to maybe a twentieth of a yes/no
question. The sequence sharpens the guess by a sliver, not a landslide — which is exactly what the
honest literature expects.

**The crucial caveat, stated plainly:** a positive `B_seq` means the next pitch is *more
predictable* from the sequence. It does **not** mean the sequence *helps the batter get hits*, and
it certainly doesn't mean the pitcher should *change* his sequence. Predictable is not the same as
exploitable. A pitcher can be perfectly predictable and still unhittable. `B_seq` is an
*information* number, not a *runs* number — and keeping those two apart is the whole point of the
ladder.

---

## What the result will mean (the branches)

When we run WS2 on the real data, the answer falls into a branch on each of three questions. The
notebook and paper have the full write-ups; here's the plain version.

**Grammar — does order help selection, and how deep?**

- **G1 — yes, past the previous pitch.** The full grammar beats "just the last pitch" by more than
  noise. Real *ordered* structure: what you throw next depends on the last *few* pitches in order,
  not just the last one. We then read the effective-order distribution and the motifs to say *how
  deep* and *which rules*. This is a live signal for the fancier models (WS3, WS6) to confirm and
  extend.
- **G2 — history helps, but only one pitch deep.** The previous pitch matters, but the pitch
  *before* it adds nothing — the "grammar" is really just calibrated bigrams (last-pitch rules).
  This is the *expected* result (the project predicted "O barely beats L1"), and a clean one: it
  tells the later models they only need to carry the previous pitch.
- **G3 — nothing beats context.** Selection is driven by count and pitcher, with within-at-bat
  order adding nothing. That's *surprising* against conventional wisdom, so treat it as a claim to
  audit: check whether the deep contexts simply never had enough data (a support problem, handed
  up to WS3/WS6) before concluding order truly doesn't matter.

**Bits — is the ordering forecastable?**

- **B1 — yes, `B_seq` is clearly positive.** The sequence leaks real next-pitch information (more
  in deep counts, later in at-bats). This feeds the project's "predictability" capstone. Remember:
  forecastable ≠ exploitable.
- **B2 — no, about zero.** The sequence adds no measurable bits about the next pitch. A clean
  negative, consistent with a one-pitch-deep or context-only reading.

**Motifs — do the rules make baseball sense?**

- **M1 — yes, face-valid.** The top rules look like real pitching: repeat suppression (don't
  become predictable), fastball-then-breaking-ball setups, two-strike putaway shifts. The grammar
  is learning baseball.
- **M2 — no, arbitrary.** The top rules have no baseball reading, or ride on barely-supported
  contexts. Treat them as provisional and check for overfitting.

---

## How to run it

Everything is in **RUNBOOK.md, step WS2.1**. The short version, from the repo on your machine:

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws2_bayes_markov/run_ws2.py --table data/processed/decision_table.parquet --out results/ws2/ --views C L1 O
```

(That needs the decision table built first — RUNBOOK step 1. The two quick synthetic checks,
`--synth null` and `--synth positive`, need no data and take about five seconds each. Remember
`--synth null` is WS2's **positive control** — it *should* detect order.)

It writes, under `results/ws2/`:

- `predictions_real_<view>.parquet` — the per-view next-pitch predictions (fed to the shared
  scorer).
- `ws2_report_real.json` — the full report: every log loss, the baseline comparisons,
  `delta_order_L1` and its confidence interval, the `B_seq` slices, the effective-order
  distribution, the motifs, the fitted concentrations, and the detect/collapse verdicts.
- `motifs_real.csv` — the ranked grammar rules (ordered context → how it moves the next pitch).
- `ws2_real.runmeta.json` — timing and memory, for the compute-cost accounting.

**What to look at first**, in order:

1. **`delta_order_L1` and its CI**, read with the G-branches above. A lower bound above zero is
   the clean "order beats the previous pitch" claim. A small negative is *"no ordered edge beyond
   the last pitch"* — never "order hurts."
2. **The effective-order distribution.** Where does the mass sit? Mostly order 0–1 with a thin
   tail is the honest "how deep the data lets us go." A real chunk at order ≥ 2 is resolvable
   ordered structure.
3. **The motifs (`motifs_real.csv`).** Do the top rules look like baseball? Repeat suppression and
   fastball→offspeed setups are face-valid; nonsense rules on thin support are suspects.
4. **`B_seq`**, as bits of next-pitch predictability — and remember it's finding #1 (information),
   not finding #2 (outcomes).
5. **The baseline comparison.** L1 and O should be *at least as good* as the quick
   "pitcher × count × prev" reference. If the serious grammar lost to the quick baseline, something's
   off.

---

## Mini-glossary

- **Context** — the last few pitches of the current at-bat, *in order*. The grammar's input. (The
  last 2 pitches is a depth-2 context.)
- **Suffix** — a context with its *oldest* pitch dropped. Backing off one level = using the suffix.
  (`(fastball, fastball, slider)` → suffix `(fastball, slider)`.)
- **Backoff** — using less history when the full history is too rare to trust. Automatic and
  graceful, like autocomplete falling back from a 3-word phrase to a 2-word one.
- **Concentration (`α`)** — the fitted dial, one per depth, for how much to trust that depth of
  history. Moderate = "this depth is worth using"; huge (ceiling) = "this depth adds nothing, pool
  it away." The data sets it, not us.
- **Effective order** — how many pitches back the grammar's prediction actually reaches, on a given
  pitch. A *readout* of the fitted model, not a setting. "Mean effective order 0.6" means most
  pitches used little or no order.
- **Motif** — one of the grammar's "rules": an ordered context and how it reshapes the next-pitch
  distribution versus its suffix. E.g. "after two sinkers, another sinker gets less likely."
- **Lift** — how much a motif moves a pitch's probability, reported as the before→after change and
  a ratio. "Suppress" = made less likely; "promote" = made more likely.
- **Bits (`B_seq`)** — the extra next-pitch information the sequence supplies, in yes/no-questions'
  worth. Positive = the sequence makes the next pitch more forecastable. Information, not outcomes.
- **`delta_order_L1`** — the headline selection number: how much the full ordered grammar (O)
  out-predicts the previous-pitch grammar (L1). Positive-and-significant = order carries
  information past the last pitch; about zero = it doesn't (read it directly — no bias to correct,
  unlike WS1's `Δ_order`).
- **Positive control (D30)** — for WS2, the synthetic *null* world, because it plants an ordered
  *selection* habit the grammar should detect. WS2's *negative* control is the history shuffle,
  under which a real order effect must collapse.
