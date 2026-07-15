# WS6 in Plain English: Letting a Neural Net Read the At-Bat

This is the intuition guide for Workstream 6. No equations you have to solve — just the ideas, in
order, so that when you read the paper or the notebook the math is already familiar. If you only read
one document about WS6, read this one first.

---

## What WS6 is, and the one question it asks

The whole project is a **ladder**. Each rung is a model that asks *does the sequence of pitches
already thrown help you understand the next pitch — and its outcome?* WS1 was a lookup table. WS2 was
a **grammar** — it wrote the ordered rule down in a table you can read. WS3, the centerpiece, was a
big tree model fed **hand-crafted** history features: we, the humans, decided what to measure ("how
much did the speed jump from two pitches ago," "how long is the current same-pitch streak") and
handed those numbers to the model.

**WS6 stops hand-crafting.** Instead of us deciding which features matter, we let a small neural
network read the at-bat like a short sentence — one "word" per prior pitch, each word carrying the
pitch's type, its result, and its physical numbers — and figure out for itself what's worth
remembering. The one question WS6 exists to answer:

> When we stop engineering features and let the net read the raw sequence, does it find anything our
> engineered features (WS3) and our explicit grammar (WS2) missed?

That's it. Everything below is in service of asking that question *fairly*.

---

## The fairness rule: same-size brain, different blindfolds

Here's the trap WS6 is built to avoid. If you just compare "big fancy neural net" to "tree model,"
and the net wins, you can't tell *why* it won — was it the clever sequence-reading, or just that the
net had more knobs? So WS6 makes all five versions **the same size**. They get an identically-sized
brain; the *only* thing that differs is **what each is allowed to read**:

- **C (context).** Reads only the situation — the count, inning, score, who's up. No pitch history.
- **U (unordered).** Reads the *bag* of prior pitches this at-bat, with the order scrambled. Built so
  that scrambling changes nothing — it literally cannot see sequence.
- **L1 (last pitch).** Reads only the single most recent pitch.
- **O (ordered).** Reads the *full sequence in order*.
- **OM (matchup memory).** O, plus the long-term history between this batter and this pitcher.

The key pair is **L1 and O**: they are the *exact same network* — same wiring, same size — with one
difference. L1 gets a blindfold that lets it see only the last pitch; O sees the whole ordered
history. So when O beats L1, it's beating *itself with the blindfold off*. The only thing that changed
is how much of the at-bat it was allowed to read — which is exactly the "does order matter?" question,
with the "is it just a bigger model?" objection removed. On the test run, L1 and O printed the
**identical parameter count (31,112)** — proof the brains are the same size.

---

## Reading the at-bat like a sentence

A quick picture of what "read it like a sentence" means. Each prior pitch this at-bat becomes one
word. The word isn't a letter — it's a little bundle: *what family* (fastball, slider, …), *what
happened* (ball, called strike, whiff, …), and seven physical numbers (speed, where it crossed the
plate, movement, and how much the speed and location jumped from the pitch before). The net reads
these words left to right, keeping a running summary in its head, and updating that summary at each
word. After the last prior pitch, it uses that summary — plus the game situation, plus (for the
outcome model) the pitch actually thrown — to guess what comes next, or what the result will be.

The engine that keeps the running summary is a **GRU** (see the glossary). The important thing about
it: it has little "gates" that decide how much of the running summary to keep and how much of each new
word to fold in — and those gates are exactly what let *one* network behave differently when it's fed
one pitch (L1) versus the whole sequence (O).

---

## The targeting story: grade the model where the rule actually lives

This is the most important idea in WS6, and it's a lesson that reaches far beyond this workstream.

Our synthetic **null world** plants a habit: a pitcher tends *not* to throw three of the same pitch in
a row. But look closely — that habit only does anything **when the last two pitches already matched**.
If the last two were a fastball and a slider, "don't throw three fastballs in a row" is irrelevant. So
the rule lives in a *corner* of the data: the "repeat context" pitches where the previous two were the
same.

Now watch what happens when you grade the model two ways:

- **Grade it on every pitch.** Most pitches aren't repeat contexts, so the rule does nothing there —
  just random noise. That noise drowns the real signal, and the average edge looks like **nothing**
  (on the test run, the aggregate order-effect was `+0.0006` — statistically zero).
- **Grade it only on the repeat-context pitches** — the ones where the rule applies. Now the edge pops
  right out (on the test run, `+0.0115`, clearly above zero).

**Same model, same data — the only difference is which pitches you graded it on.** The plain lesson:
*an effect that lives in a corner of the data has to be measured in that corner.* Average over
everything and you'll hide it. The math backs this exactly — averaging shrinks a corner-bound effect
by the fraction of pitches it lives on, and shrinks your ability to detect it by the square root of
that fraction (`THEORY.md` §5). This is why WS6 checks the repeat-context slice, and why, on the real
data, a genuine order effect might show up only in long at-bats or two-strike counts and be invisible
in the headline number. **Read the slices, not just the total.**

---

## The opacity story, told straight

Here's the part that captures the whole deep-learning bargain in one exhibit.

We proved two things about the null-world GRU, and they sound contradictory until you sit with them:

1. **The net clearly USES the rule.** It scores *better* exactly on the repeat-context pitches where
   the no-three-in-a-row habit lives (that `+0.0115` edge). Its *loss* — how wrong it is — knows about
   the grammar.
2. **But when you open it up and ask "so what's the rule?", it won't say it cleanly.** We ran a probe:
   for each pitch family, we asked the net directly, "after two of these in a row, is a third less
   likely than normal?" A clean answer would be "yes" for every family. What we got was a muddle — for
   fastballs, yes (suppressed by `+0.05`); for cutters, actually the *opposite* (`−0.057`); overall
   only 43% of families said "yes," so the probe's verdict was `motif_present = False`.

Not a contradiction — two different questions. The *score* is an average over what actually happened,
weighted by how often each situation comes up. The *probe* is a fussy per-family what-if on made-up
inputs, some of which the net barely saw in training (a pitcher rarely throws two cutters in a row, so
"after two cutters…" is a situation the net has little experience with). The net can be right on
average about the common cases — which is what lowers its loss — while its answer to the rare-family
what-if is noise.

Now the contrast that matters: **WS2's grammar model says the rule out loud.** It *is* a table of
what-ifs, so you can read "after two-in-a-row, this pitch is less likely" straight off it. WS6's GRU
uses the same rule but packed it into arithmetic no one can read off. **That trade — more power to
learn, less ability to explain — is the whole deep-learning bargain**, and WS6 shows it happening on a
rule we planted ourselves and know is there. The takeaway for the final paper: *when a learned model
is accurate, that does not mean it will tell you why.* Interpretability is something you have to
measure, not assume.

---

## The three-strikes matchup pattern

One more result, and it's a cross-workstream one. Every model that can read "matchup memory" (the OM
view — what this batter has done against this pitcher) has been asked: does adding that memory help
predict outcomes? On our synthetic worlds, where we planted **no** matchup effect, the answer keeps
coming back the same: it doesn't help — it actively **costs** a little accuracy out-of-sample (WS6's
number: `−0.0036`, clearly below zero).

WS6 is the **third** model family to find this. WS3 (the tree model) found it. WS5 (the tabular MDP)
found it. Now WS6 (the neural net) finds it. Three completely different kinds of model, all agreeing:
piling in a bunch of mostly-empty matchup columns makes the model noisier without adding signal, so it
generalizes worse. When three unrelated model families agree, it's not a quirk of one — it's a real
lesson: *more features is not more information.* On the real data, if the matchup view looks worse than
plain O, the honest read is "no matchup signal we can see, plus a real cost of carrying those sparse
features" — not "matchup doesn't matter in baseball."

---

## Colab vs. your desktop, in plain words

WS6 is the one place in the whole study where a GPU actually helps — the sequence-reading (recurrent)
fits are slow. Two honest paths, and a warning:

- **The warning: don't fight your AMD card.** Your desktop GPU is AMD. PyTorch on Windows only really
  works with Nvidia (CUDA) cards; the AMD paths (ROCm, DirectML) are unofficial and flaky. Trying to
  make PyTorch use your AMD card is a rabbit hole. Don't.
- **The easy road: free Google Colab.** There's a ready-made notebook (`colab_ws6.ipynb`) that runs on
  a **free** Google Colab GPU (a "T4"). It's the same WS6 code you'd run locally, just on Google's
  hardware — about **an hour** for the full ordered-view fit, versus several hours on your desktop CPU.
- **The simple road: your desktop CPU.** The exact same code runs on your CPU with no GPU at all — just
  slower. It's honest and simple; use it for correctness or if Colab is a hassle.

Both are laid out step-by-step in **RUNBOOK.md, Step WS6** (WS6.1 = local CPU, WS6.2 = Colab). Nothing
else in the study needs a GPU — only WS6, and only its *optional* Transformer demo really prefers one.

---

## What the result will mean (the branches)

When WS6 runs on the real data, the answer falls on three separate questions. The notebook and paper
have the full write-ups; here's the plain version.

**Did the learned net beat our hand-crafted features? (the R-axis)**

- **R+ — yes, it beat WS3.** The net found ordered signal our engineered features missed. Exciting —
  but *before believing it*, run the probes and check the net's edge shows up where order should
  matter, not as some side-effect.
- **R= — it tied WS3.** Our engineered features already captured what's there; the net neither added
  nor lost much. This is the likely outcome, and a clean one: it means the cheap tree model wins,
  because it gets the same answer for a tiny fraction of the compute. "Engineered features suffice."
- **R− — it did worse than WS3.** Almost never "learning can't help" — at this scale it means the net
  under-trained or the budget was too small. Read the training curves before concluding anything.

**Did order help, inside WS6? (the H-axis)**

- **H1 — yes.** O beat both U and L1. Real ordered signal — and remember the targeting lesson: check
  the repeat-context and per-slice edges, not just the total.
- **H2 — history helps, order doesn't.** The history views beat context-only, but the fully-ordered
  view doesn't beat last-pitch. The *expected* result. Clean and honest.
- **H3 — nothing beats context.** No history view helps at all. The cleanest null (double-check the
  training curves, since an under-trained net can fake this).

**Did matchup memory help? (the M-axis)**

- **M+ — yes.** OM beat O. Would be the study's *first* positive matchup signal.
- **M0 — no effect.** OM ≈ O.
- **M− — it costs.** OM did worse than O — the three-family fragmentation pattern again. Read it as "no
  matchup signal + a real cost," not "matchup doesn't matter."

The honest headline is a triple. The most anticipated is **R= × H2 × M−**: the net confirms our tree's
answer (order barely helps, cheap model wins), and matchup memory costs. The strongest is
**R+ × H1 × M+**. The cleanest confirmation-of-null is **R= × H3 × M0**.

---

## How to run it

Everything is in **RUNBOOK.md, Step WS6** (WS6.1 local CPU, WS6.2 Colab). WS6 needs only the decision
table (Step 1) — it builds its own sequence tensors and refits nothing from other workstreams. First
install the deep extra (this is the only step that needs PyTorch):

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
pip install -e ".[deep]"
# Calibrate first: time one epoch of one view on your machine before the full run.
python workstreams/ws6_deep_seq/run_ws6.py --table data/processed/decision_table.parquet `
    --views O --targets selection --epochs 3 --device auto --out results/ws6_calib/
# Full ladder (resumable; checkpoints per view+target, so you can stop and resume):
python workstreams/ws6_deep_seq/run_ws6.py --table data/processed/decision_table.parquet `
    --device auto --epochs 30 --out results/ws6/
```

The two quick synthetic checks need no data and rebuild the world themselves (a few minutes each):

```powershell
python workstreams/ws6_deep_seq/run_ws6.py --synth null --out results/ws6_null/ --epochs 30
python workstreams/ws6_deep_seq/run_ws6.py --synth positive --out results/ws6_pos/ --epochs 30
```

**What to paste back**, per RUNBOOK: the whole printed **headline block** and the report's Pareto rows.
**What to look at first**, in order:

1. **The learned O vs WS3's engineered O** (the R-axis) — the whole point of WS6. Paste WS3's and
   WS2's O-view losses into the comparison line. If the GRU doesn't beat them, that's the honest,
   publishable finding.
2. **Order effect (Δ_order) on outcomes**, read with the D21 rule (a small negative is "consistent with
   no ordering effect," never "order hurts") **and** the targeting lesson (check the repeat-context /
   slice edges, not just the total).
3. **Matchup (Δ_matchup)** — expect it to be ≤ 0 (the three-family cost pattern).
4. **The Pareto row** — L1 and O should print identical parameter counts (the capacity match); the
   only thing O costs extra is wall-clock. Is the (usually small) order gain worth the extra compute?

The exact formulas live in `THEORY.md`; the exact code in `workstreams/ws6_deep_seq/model.py` and
`run_ws6.py`; the full paper in `PAPER.md`.

---

## Mini-glossary

- **GRU (gated recurrent unit)** — the little engine that reads the sequence one pitch at a time,
  keeping a running summary and using "gates" to decide what to keep and what to forget. The thing
  that lets one network read either the last pitch (L1) or the whole ordered history (O).
- **Embedding** — a learned numeric code for a category. Instead of "fastball = 1, slider = 2"
  (meaningless numbers), the net learns a small vector for each pitch family that captures how they
  relate. It's how the net turns "fastball" into something it can do math on.
- **Epoch** — one full pass through the training data. The net makes many passes, improving a little
  each time. WS6's real run does up to 30, stopping early when it stops improving.
- **Early stopping** — watching a held-out slice of data during training and quitting the moment the
  model stops getting better on it (before it starts memorizing). An automatic "how much model is too
  much" dial. It's what keeps the comparison between L1 and O fair — it stops the bigger-reading O from
  overfitting.
- **Parameter** — one of the numbers inside the network that training adjusts. "Same-size brain" means
  same number of parameters. L1 and O have the identical count (31,112) — that's the fairness rule made
  visible.
- **Pareto** — the accuracy-vs-cost trade-off. A model is only worth its extra cost if the accuracy it
  buys beats what a cheaper model gets. WS6 is the study's most expensive predictive step, so we always
  ask: was the (usually tiny) gain worth the compute?
- **Probe** — a diagnostic that pokes the trained model with hand-built inputs to ask what it learned.
  WS6's motif probe asks the net, family by family, "after two-in-a-row, is a third less likely?" — and
  found the net won't answer cleanly even though its score proves it knows the rule. The opacity
  exhibit.
