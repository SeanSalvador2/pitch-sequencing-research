# WS1 in Plain English: The Shrunk Lookup Table

This is the intuition guide for Workstream 1. No equations you have to solve — just the ideas,
in order, so that when you read the paper or the notebook the math is already familiar. If you
only read one document about WS1, read this one first.

---

## What we actually found (2021–2025)

Here is the headline in plain words, before any of the machinery below. We ran WS1 on five full
seasons — about **3.57 million** regular-season pitch decisions — training on 2021–2023 and scoring
on 2024.

**The lookup table could not make history pay.** Adding within-at-bat history to the table's key
did not sharpen the next-pitch guess at all. Context-only (**C**) scored a log loss of **1.4714**,
and the history views landed a hair *worse* (last-pitch **L1 = 1.4772**, ordered **O = 1.4787**).
So on WS1's own terms, history didn't help.

**But — and this is the real finding — history absolutely does matter; the table just can't hold
it.** A quick side-baseline keyed on "this pitcher, this count, and the *one* previous pitch"
scored **1.4410** — beating every WS1 view by a comfortable margin (about 0.03–0.04 log loss), and
beating the no-history version of the same baseline by **0.046**. That gap is proof that the
previous pitch carries real, material information about the next one. WS1 misses it not because the
signal is absent, but because its hand-built key (count × handedness × pitcher × previous pitch)
chops the data into too many tiny cells to use. We call that the **fragmentation tax**, and it is
the whole reason the project climbs past a lookup table.

**Order beyond the last pitch? Not visible to a table.** The order comparison `Δ_order` came out at
**−0.0015** — essentially zero, read as "no ordering effect a table can see" (a small negative is
the metric's built-in optimism, not a real cost). The support numbers show why the table is out of
room: distinct cells explode from about **30,000** (C) to over **320,000** (the unordered view),
and by the ordered view **more than a third** of the pitches we score land in cells we saw fewer
than 20 times in training.

**What the model "decided" is itself a finding.** WS1 fits a dial per level for how much to trust
that level (the *concentration*). Two dials tell a clean story. Pitchers came back extremely
*distinct* — the dial barely pooled them, confirming every pitcher really does have his own
repertoire. But the run-value side pegged its dial at the maximum "pool everything" setting on
every level: in plain words, the cell-by-cell "how many runs is this worth" tables held almost no
trustworthy signal, and the model correctly refused to trust them. Expected-reward error stayed
flat (**0.1200**) no matter how much history we added.

**One earlier prediction came true.** On our synthetic test worlds, context-only C lost a hair to a
"pitcher × count" baseline because handedness carried no signal there. We predicted that would flip
on real data, where lefty/righty matters — and it did: WS1-C (1.4714) now beats pitcher × count
(1.4866).

The bottom line: **first-order selection is real, but a lookup table is the wrong tool to express
it.** That is exactly the hand-off to WS2 (a smarter grammar) and WS3 (which puts pitch physics in
as features and can see what the table is blind to).

---

## What WS1 is, and where it sits

The whole project is a **ladder**. Each rung is a model that tries to answer one question:
*does the sequence of pitches already thrown help you understand the next pitch?* We climb from
the simplest, most transparent model to the most powerful, and we watch how much of the
"sequencing matters" story survives each harder test.

**WS1 is the bottom rung.** It is the simplest thing that could possibly work: a **lookup
table**. You want to know what a pitcher throws in a 1-2 count to a lefty? Look it up. You want
to know the expected value of throwing a slider there right after two fastballs? Look that up
too. That is the entire model. It is deliberately humble, and that is its job — it is the number
every fancier model has to beat before its complexity is worth anything. If a neural network
can't beat a lookup table, we've learned something important and cheap.

WS1 does two lookups:

1. **Selection** — given the situation, what's the probability distribution over the next pitch
   family (fastball, slider, curve, …)?
2. **Run value** — given the situation *and* the pitch actually thrown, what's the expected
   reward, and how sure are we?

---

## The one idea you need: batting-average shrinkage

Here is the whole statistical engine of WS1, told as a baseball story.

A hitter starts the season 9-for-30. His batting average is .300. Do you believe he's a .300
hitter? Of course not — 30 at-bats is almost nothing. Your honest guess for his *true* ability
is somewhere between his .300 and the league average of, say, .260. You'd **pull his .300 toward
.260** — and how hard you pull depends on how little data you have. Thirty at-bats? Pull hard.
Three hundred at-bats? Barely move it. Three thousand? Leave it alone.

That pulling-toward-the-average move is called **shrinkage**, and the famous result is that the
shrunk estimates predict the rest of the season *better* than the raw averages do. Raw averages
overreact to small samples; shrinkage fixes that.

WS1 is that trick, applied to pitches instead of hits, and done twice:

- For **selection**, each "cell" of the table (a specific situation) has its own pitch mix. A
  thin cell — one we've barely seen — gets pulled toward a broader, better-estimated "parent"
  mix. A well-populated cell keeps its own numbers.
- For **run value**, each cell has its own average reward, pulled toward the parent's average by
  the same logic.

The parent is just a shallower version of the same table — a less specific situation we have
much more data on. So a thin, ultra-specific cell borrows strength from the broad situation it
lives inside. That's the "empirical Bayes" in the name: we don't guess how hard to pull, we
**learn it from the data**.

---

## The five views, when your model is a lookup table

The study defines five "views" of each decision — five levels of detail about the pitch history.
For a lookup table, a view is simply **what goes into the lookup key**:

- **C (context only).** The key is *count × handedness × pitcher*. No pitch history at all. "What
  does this pitcher throw in this count to this kind of hitter?" This is the baseline.
- **L1 (last pitch).** The key adds *the previous pitch's family*. "…given that the last pitch
  was a fastball."
- **U (unordered priors).** The key adds *the bag of pitches thrown so far this at-bat, ignoring
  order*. "…given that he's already thrown two fastballs and a slider, in some order."
- **O (ordered priors).** The key adds *the last two pitches in order*. "…given that it went
  fastball-then-slider, specifically in that order."
- **OM (matchup memory).** The key would add *this specific batter-vs-pitcher history*. WS1
  can't do this one — see the support problem below.

The **differences between these views are the whole point.** If O predicts better than L1 and U,
then the *order* of pitches carries information beyond just "what was the last pitch" and "what's
in the bag." If they're all the same, order doesn't add much. That comparison has a name,
`Δ_order`, and it's the headline number.

---

## The support problem — why we can't just make the table more detailed

Here's the catch, and it's WS1's single most important lesson for the whole project.

Every time you add detail to the lookup key, you **split every cell into many smaller cells**.
That's great for precision — until you run out of data. A key so specific that each cell has been
seen twice is useless; you're estimating a pitch mix from two pitches.

We measured exactly how bad this gets, on our synthetic test world. Counting the number of
distinct cells each view creates:

| view | distinct cells | share of eval pitches landing in a cell with < 20 training pitches |
|---|---|---|
| **C**  | 288   | 0.044 |
| **L1** | 1,023 | 0.31 |
| **O**  | 2,691 | 0.52 |
| **U**  | 3,560 | 0.45 |

Read the first column: going from C to the history views, the number of cells **explodes** —
roughly 9× to 12×. Read the second column: with the ordered view O, **more than half** of the
pitches we want to score land in a cell we saw fewer than 20 times in training. You cannot
estimate anything trustworthy from that. (On the real data the explosion is even sharper; these
are the numbers from the synthetic world we validate on.)

The shrinkage saves us from *crashing* — an unseen cell just falls back to its parent (see
"backoff" below) — but it can't create information that isn't there. Once half your evaluation
lands in near-empty cells, the "extra detail" of the deep key is mostly decorative.

**That's why the ladder exists.** A lookup table runs out of data long before it runs out of
questions. The models above WS1 (a smart variable-order backoff in WS2, feature-based trees in
WS3, a learned representation in WS6) are all, in one way or another, ways to ask deeper
questions *without* shattering the data into unusable pieces. WS1's job is to make that
necessity undeniable.

### Why OM is impossible for a table

The matchup view OM would key on the specific batter-vs-pitcher pairing. But an average
pitcher-batter pair has only about **20 pitches** in their entire shared history. Splitting the
already-tiny O cells by *which batter* would leave almost every OM cell empty — they'd all just
fall back to O and add nothing. So WS1 honestly declares OM **infeasible for tables** and reports
the pitcher-batter counts that prove it. Matchup memory is a job for the pooling/embedding models
later in the project, not for a lookup table. Saying so clearly is part of WS1's contribution.

---

## What the fitted "concentration" numbers are telling you

WS1 doesn't guess how hard to shrink — it fits a number per level called the **concentration**
(written α for selection, κ for run value). You can read it like a verdict:

- **A small concentration** means "the cells at this level are genuinely different — trust each
  one's own data, shrink gently."
- **A huge concentration** means "the cells at this level all look the same — pool them together,
  this level of detail is adding nothing."

The clearest example comes from our **null test world**, which is built to have *no* real ordered
effect. When WS1 fits the history level there, the concentration **blows up to a million** (its
ceiling). In plain words: *the data told the model that within-at-bat order is useless here, so
the model pooled it all the way back to the pitcher-and-count cell.* Every view then makes the
exact same prediction, and `Δ_order` comes out to **exactly zero**. That's not a bug — that's the
model correctly refusing to invent a pattern that isn't there. It's the single best demonstration
that the method is honest.

---

## What the result will mean (the branches)

When we run WS1 on the real data, the answer falls into a branch on each of two questions. The
notebook and paper have the full write-ups; here's the plain version.

**Does history help predict the next pitch? (selection)**

- **S1 — yes, clearly.** The history views beat context-only by more than noise. Pitchers
  sequence their selection in a way even a table can see. This is the expected real-data result
  and means the selection signal is live for the fancier models to build on.
- **S2 — not really.** Adding history doesn't sharpen the next-pitch guess beyond count and
  pitcher. For a flat table this is common and doesn't mean pitchers don't sequence — it means a
  lookup table can't resolve it. That's a question for WS2/WS3.

**Does the *order* add anything beyond the last pitch? (`Δ_order`)**

- **R1 — yes, significantly** (the confidence interval clears zero). A real, table-visible
  ordered effect. Treat it with suspicion first: check that it isn't riding on near-empty cells,
  and that it shows up where it should (longer at-bats, two-strike counts). If it survives, it's
  the first real evidence that *order* matters.
- **R2 — no, it's about zero or slightly negative.** This is the most likely result, and it reads
  as *"consistent with no ordering effect a table can see"* — **not** as proof that order doesn't
  matter. (More on why in the next section.) It's a clean, honest null, and it sets the bar every
  later model must clear.
- **R3 — clearly negative.** The ordered key fragmented the data so badly that its predictions
  got *worse*. This is the support problem biting, and it's the sharpest possible argument for
  WS2's smarter backoff. It's a statement about running out of data, not about baseball.

---

## The mechanism-blindness lesson (why "≈ 0" isn't "nothing there")

This is the subtle part, and it matters for how you read a null result.

Our **positive test world** plants a real effect: when a pitcher makes a big *velocity change*
from two pitches ago to the last pitch, the next pitch gets more whiffs. It's a genuine ordered
effect, and we know its exact size.

Now — WS1's table is keyed on pitch **names** (fastball, slider, curve), not on velocity numbers.
It has no "velocity" column at all. So it can only "see" a velocity effect *indirectly*, through
the fact that pitch names correlate with speed (a curve is slow, a four-seam is fast). That's a
lossy, blurry proxy for the real thing.

How lossy? We measured it. WS1 recovers the effect in the **right direction** — and only the
ordered view O sees it, exactly as designed — but at about **3% of its true strength.** The table
is 97% blind to a real, planted effect, purely because it's keyed on the wrong vocabulary
(names, not physics).

The lesson: **when WS1 says `Δ_order ≈ 0` on real data, that does not prove order doesn't matter.**
It proves a *name-keyed lookup table* can't see it. A real ordered effect built on velocity, or
location, or movement, would show up faint-to-invisible in these tables. That 3% number is why
the project keeps climbing: WS3 puts velocity in as an actual feature, and WS6 learns its own
representation — both can see what the table is blind to. WS1's honest null is a floor, not a
verdict.

---

## How to run it

Everything is in **RUNBOOK.md, step WS1.1**. The short version, from the repo on your machine:

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws1_eb_tables/run_ws1.py --table data/processed/decision_table.parquet --out results/ws1/ --views C U L1 O --target both
```

(That needs the decision table built first — RUNBOOK step 1. The two quick synthetic checks,
`--synth null` and `--synth positive`, need no data and take about five seconds each; they're the
correctness self-tests.)

It writes, under `results/ws1/`:

- `predictions_real_<view>.parquet` — the per-view predictions (fed to the shared scorer).
- `ws1_report_real.json` — the full report: every log loss, the baseline comparisons, `Δ_order`
  and its confidence interval, the support table, and the fitted α/κ per level.
- `support_real.csv` — the support exhibit (the cell-count / thin-cell table above).
- `ws1_real.runmeta.json` — timing and memory, for the compute-cost accounting.

**What to look at first**, in order:

1. **The support table.** Confirm the cells explode from C to the history views and the thin-cell
   fraction climbs. This is WS1's headline exhibit — it should look like the table above, sharper.
2. **`Δ_order` and its CI**, read with the branches (R1/R2/R3) above. Remember: a small negative
   is *consistent with no effect*, never "order hurts."
3. **The baseline comparison.** WS1's history views should be at least as good as the quick
   `pitcher × count × prev` reference. If the serious version lost to the quick one, something's
   off. (On the synthetic worlds, context-only C can lose a hair to `pitcher × count` because
   handedness carries no signal there — that flips on real data.)
4. **`Δ_matchup` = N/A.** Expected and correct — tables can't do matchup memory.

---

## Mini-glossary

- **Shrinkage** — pulling a shaky, small-sample estimate toward a steadier "parent" estimate. The
  batting-average trick. Less data → pull harder.
- **Posterior** — your updated belief after combining a prior expectation with observed data. WS1
  reports the posterior *mean* (its best guess) and the posterior *spread* (how unsure it is).
- **Concentration (α / κ)** — the dial that sets how hard to shrink at each level. Big = "this
  level of detail adds nothing, pool it away." Small = "these cells are really different, trust
  them." WS1 fits it from the data.
- **Backoff** — what happens when a cell has no data: the estimate falls back to its parent (a
  shallower, better-supported situation). It's automatic — it's just what shrinkage does at zero
  data.
- **Log loss** — the score for a probability forecast; lower is better. It punishes confident
  wrong answers hard. It's the primary metric for the selection tables.
- **`Δ_order`** — the headline sequencing number: how much the fully *ordered* view improves on
  the better of "last pitch only" and "unordered bag of pitches." Positive and significant = order
  carries information a table can see; about zero = it doesn't (for a table); read it with the D21
  rule (a small negative means "consistent with no effect," not "order hurts").
