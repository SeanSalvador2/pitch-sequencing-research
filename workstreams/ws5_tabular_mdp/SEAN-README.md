# WS5 in Plain English: The Setup Pitch, and the Error Bar That Kept Us Honest

This is the intuition guide for Workstream 5. No equations you have to solve — just the ideas, in order, so
that when you read the paper or the notebook the math is already familiar. If you only read one document
about WS5, read this one first.

---

## What WS5 is, and the one-sentence version of the result

The project is a **ladder**. WS1–WS3 asked *does the sequence help you predict things?* WS4 asked the first
prescriptive question — *what should he throw?* — and hit a wall it named: a greedy, one-pitch-at-a-time
recommender can only cash the tiny part of a sequencing effect that's about *this* pitch, and the effect we
planted is mostly about *setting up the next* pitch. WS4 measured that wall (about `0.003` runs of a `0.03`-
run trick) and dared the next rung to climb over it.

**WS5 is the first rung that can even try.** It's the simplest model that thinks *one pitch ahead* — so it
can, in principle, value a **setup pitch**: throwing something now to make the next pitch play up. Here's the
one-sentence result, and it's a proud, honest "not yet":

> **We built a model that provably knows how to bank a setup — a unit test shows it deliberately throws the
> setup pitch and comes out ahead. But *proving it earns money in the wild* needs more at-bats than our
> synthetic test has, so the honest verdict is "the setup is real, we just can't certify it at this scale
> yet." And along the way we caught our own error bars lying and rebuilt them the expensive, honest way.**

---

## The board game: an MDP in plain words

Picture the at-bat as a little board game. The squares are the **count** — 0-0, 1-0, 0-1, 2-1, and so on —
and there are four "you're done" squares: walk, strikeout, hit-by-pitch, ball-in-play. Every pitch moves you
from one square toward the done squares (more balls, more strikes), and the at-bat always ends. That board
plus "which pitch do you throw on each square, and where does it tend to send you" is a **Markov decision
process** (an MDP) — a fancy name for a very readable little table.

The whole trick of WS5 is a **third design of the board with one extra feature: a lamp**. The lamp lights up
when the *last two pitches* differed a lot in speed (a big velocity jump). We build three boards, richer and
richer, matching the study's usual C / L1 / O ladder:

- **count** — just the count squares (16 squares). Our "context only" board.
- **count_prev** — the count *and* what family you just threw (112 squares).
- **count_prev_trigger** — the count, the last family, *and the lamp* (220 squares).

The lamp is computed only from **pitches that already happened**, so it never peeks at the current pitch —
that's the project's no-leakage rule. And the lamp is the whole point, as the next section explains.

---

## The setup pitch, and why greed can't pay for it

Here's the effect we planted in the test world: **when the pitcher makes a big velocity jump into a pitch,
the *next* pitch gets more whiffs.** Worth about `0.03` runs.

Now watch what a **greedy** chooser does — one that only cares about *this* pitch's reward. By the time it's
choosing the current pitch, the velocity jump either already happened or it didn't; the whiff bonus is
already spoken for, owed equally to every pitch it could throw right now. Greed can't *create* the bonus, so
it can't cash it. That was WS4's wall.

The **setup** is the move greed can't make: **throw something slow now, on purpose, so that the jump to the
next (fast) pitch lights the lamp — and the next pitch plays up.** That value doesn't show up on *this*
pitch's ledger; it shows up on the *next square you land on*. A one-pitch-at-a-time bandit is blind to it. But
WS5's board *remembers* the lamp, so its planner can see that throwing the slow pitch now moves you to a
lit-lamp square worth more — and it will choose the setup. The `count` board forgets what you just threw, so
it literally can't tell one choice sets up a better next square than another. **That one lamp is the entire
difference between a model that can value a setup and one that can't.**

---

## Planning the board: policy iteration in plain words

Given the board, how do we find the best plan? Two steps, repeated:

1. **Grade the current plan.** Because the at-bat always ends, "what's this plan worth from each square" is
   just a small system of equations we solve exactly — no guessing.
2. **Improve it.** On each square, switch to the pitch your own grades now like best (among the pitches this
   pitcher actually throws).

A one-line argument shows each switch can only *raise* the value, and since there are only so many possible
plans, you can't improve forever — you land on the best plan in a few passes. This is **policy iteration**,
and it's exact: no neural network, no approximation, a table you can read.

---

## Two ways to score the plan, and why we use both

A plan is only as trustworthy as our ability to *value* it, so WS5 scores the same softened plan **three
ways** and checks they agree (this is the "transparent simulator that cross-checks the OPE" job the project
gives WS5):

- **model-based** — what the board *itself* says the plan is worth (and a simulator that rolls the plan
  forward to double-check the board's own math).
- **step-wise DR** and **FQE** — two **off-policy** scores computed from the *real logged pitches*, judging
  the plan we *didn't* run from data logged under the plan we *did*.

If all three land near each other, great — trust the number. If the board's own value floats *above* what the
logged data supports, we call that **DIVERGES**, and we **report it, we don't quietly average it away**. A
divergence is informative: it means the board is a little too simple to be exactly right (the count-and-lamp
picture is missing something the real data feels). That's a signal, and it's a to-do for the capstone (WS7),
which gets a richer picture. The most extreme version of that optimism — the board's *greedy* value vs its
honest held-out score — we show as its own labelled exhibit, never as part of the verdict.

---

## The bug story: we caught our own gate lying

This is the part we're proudest of, and it's a bug we shipped, caught, and fixed.

The first version of WS5 declared victory: **`SETUP_EXPLOITED`** — "the setup pays off!" It was wrong, and not
because the arithmetic was wrong. It was wrong about its own **confidence**.

Here's the tell. One of our scorers (FQE) produces one number *per at-bat*, and its error bars come from
reshuffling those per-at-bat numbers. But **every at-bat starts on the exact same square** (0-0, no previous
pitch, lamp off). So FQE's per-at-bat number is the *same number every time*. Reshuffle a bag of identical
numbers and nothing moves — so the "error bar" came out as **zero width**: a fake, perfect certainty. The gate
saw a zero-width interval, thought "wow, this is nailed down," and lit the victory light.

The real uncertainty was never about the starting square (that never changes) — it's about **which at-bats we
happened to log**, which changes the model we fit. So we rebuilt the error bars the honest, expensive way: we
**resample whole at-bats and refit the entire FQE model from scratch, hundreds of times** (using the same
resample for every board, so the *comparison between boards* stays tight). We call this the **refit
bootstrap**, and it costs real time — about 3,000 refits, roughly an hour on the full synthetic run.

When we ran the honest error bars, the number told the truth: the setup gap is *directionally* positive in
all three scorers (`+0.008`, `+0.011`, `+0.028`), but its honest lower bound doesn't clear WS4's `0.003` bar
at this scale. So the verdict became the honest **`SETUP_INCONCLUSIVE`**. The lesson — the same one WS4 taught,
one rung up and sharper because this time the lie was *silent* — is a rule we now state before reading any
result:

> **The error bar is the claim.** We only declare a win on a *real* interval that clears the bar. A collapsed,
> zero-width interval is printed "not applicable," never as a result. A point estimate is evidence, not proof.

---

## What our synthetic check actually showed

We tested WS5 on two worlds, and both passed the trust gate first (the machinery can reproduce the pitcher's
own value exactly — the "prove the ruler works before you measure" rule).

- **Null world (no setup effect).** The lamp is inert. The board's "does creating a lit lamp add value?"
  read-out sits at zero (setup gap `+0.0020`, coin-flip positive-fraction `0.49`), and the richer boards are
  worth no more than the plain count board. Verdict: **`SEQ_NEUTRAL_MDP`** — the machine correctly finds
  nothing where there's nothing.
- **Positive world (the planted setup).** Now the read-outs light up and cleanly separate from the null: the
  "creating a lit lamp adds value" gap is `+0.0087` and positive two-thirds of the time (`0.68`), and the best
  pitch changes on **48%** of the lit-lamp squares (vs 43% under the null). All three scorers agree the setup
  gap is positive — but the honest error bars are too wide to certify it at this scale. Verdict:
  **`SETUP_INCONCLUSIVE`**.

**The proof that the machine isn't broken.** To be sure "inconclusive" is about the *data*, not a dead
detector, we built a tiny hand-checkable world where one pitch (a slow curve) clearly sets up the next, and
confirmed the trigger board's best plan **deliberately throws the setup curve** while the plain-count board
avoids it — and the setup board's value comes out clearly higher, with the simulator agreeing. So **the
machine provably knows how to bank the lamp.** Proving it earns money *in the wild* just needs more at-bats
than our synthetic test has — and the back-of-envelope says the real season (about 40× bigger) is comfortably
enough. That's a *Phase-2 prediction*, not a shrug.

---

## What the result will mean (the branches)

When we run WS5 on the real data, the answer is read on three axes. The notebook and paper have the full
write-ups; here's the plain version.

**Does the richer (lamp) board buy real setup value? (the design-ladder axis — the real question)**
- **S+** — yes: the setup gap's honest lower bound clears WS4's `0.003` bar. The setup is **cashed** — value a
  greedy bandit provably couldn't reach. A strong claim; cross-check it against WS3 (was order even
  predictive?) and WS4 (this is exactly the wall it dared us to climb), then hand it to WS7.
- **S0** — directional but uncertifiable: all three scorers lean positive and beat the null baseline, but the
  honest error bar doesn't clear the bar. This is the fixture's own reading — report the *evidence story* plus
  the "how many more at-bats would settle it" arithmetic, not a null.
- **S−** — the richer board *loses*: too many squares, too little data per square, so the extra detail just
  adds noise. Build the plan from the simpler board and read it as "the richer state overfit," not "order
  hurts."

**Do the three scorers agree? (the trust axis)**
- **A+** — yes, they land together; trust the value.
- **A−** — **DIVERGES**: the board's own value floats above what the logged data supports. Not a bug to
  average away — a signal that the count-and-lamp board is too simple to be exactly right, and a to-do for
  WS7's richer model. Lean on the honest held-out number, never the board's optimistic one.

**And the headline verdict:**
- **`SETUP_EXPLOITED`** — the setup is cashed, certified with a real error bar (never faked).
- **`SETUP_INCONCLUSIVE`** — real, representable, agreed-in-direction, but not certifiable yet (the fixture's
  verdict; a Phase-2 data-scale question).
- **`SEQ_NEUTRAL_MDP`** — nothing there (the null world's expected result).

The most anticipated real-data reading is **S0 + A− + `SETUP_INCONCLUSIVE`** at moderate scale — a real setup
below the error-bar floor, with the board visibly imperfect, motivating the capstone.

---

## How to run it

Everything is in **RUNBOOK.md, step WS5.1**. WS5 needs only the decision table (Step 1); passing
`--ws3-dir results/ws3/` reuses WS3's behavior model as the OPE denominator, but it's optional (without it,
WS5 fits a simpler behavior model from the training counts). From the repo on your machine:

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws5_tabular_mdp/run_ws5.py --table data/processed/decision_table.parquet --ws3-dir results/ws3/ --out results/ws5/
```

The two synthetic self-tests need no data and no dependency (WS5 builds and caches the world itself):
`--synth null --out results/ws5_null/` and `--synth positive --out results/ws5_pos/`.

**The one dial to know: `--fqe-boot`.** This is the count of honest refit-bootstrap replicates — the
expensive part, and the whole reason the error bars are real. The default `200` is about 3,000 model refits
(roughly an hour on the full run). **If it's too slow on the real data, lower it to 50–100** — it only makes
the error bars a little coarser; it never changes the point estimates. Any error bar that's still structurally
degenerate prints "not applicable," never a fake number.

**What to paste back**, per RUNBOOK: the entire printed **headline block** and the ladder CSV
(`Get-Content results/ws5/ladder_real.csv`). **What to look at first**, in order:

1. **The gate.** The line after `rows` must say `behavior recovery: PASS`. If it says `FAILED_GATE`, stop and
   paste it — nothing below it is real.
2. **The setup gap vs the ceiling**, in the "SETUP GAP" block — the *refit-bootstrap* FQE gap and whether its
   lower bound clears `+0.003`. That, not the board's own optimistic value, is the verdict.
3. **The verdict** — `SETUP_EXPLOITED` / `SETUP_INCONCLUSIVE` / `SEQ_NEUTRAL_MDP`.
4. **The D42 column and the greedy-optimism exhibit** — trust the value only where the scorers agree.

---

## Mini-glossary

- **MDP (Markov decision process)** — the little board game: count squares, "done" squares, and a table of
  where each pitch tends to send you. Readable, exact, and the simplest thing that can think one pitch ahead.
- **Setup pitch** — a pitch thrown to make the *next* pitch better (throw something slow now so the jump to
  the fast one lights the lamp). The value a greedy, one-pitch-at-a-time chooser is structurally blind to —
  and the thing WS5 exists to value.
- **The trigger (the lamp)** — the extra state feature that lights when the last two pitches differed in
  speed by at least 5 mph. Computed only from prior pitches (no leakage). It's what makes a setup
  *representable*: choosing a pitch that will light the next lamp is now a visible, valuable choice.
- **Policy iteration** — the exact two-step method (grade the plan, then improve it) that finds the best plan
  on the board in a few passes. No approximation, because the at-bat always ends.
- **Absorbing state** — a "done" square (walk, strikeout, hit-by-pitch, ball-in-play): once you're there the
  at-bat is over and no more reward accrues.
- **Model-based vs off-policy value** — two ways to score a plan: what the board itself says (model-based) vs
  what the real logged pitches say about a plan we didn't run (off-policy / OPE). We check they agree.
- **DIVERGES (the D42 check)** — when the board's own value floats outside what the logged data supports.
  Reported, never averaged away — a sign the board is a little too simple, and a to-do for WS7.
- **Refit bootstrap** — the honest, expensive error bar: resample whole at-bats and **refit the whole model
  from scratch**, hundreds of times, to measure how much the answer would wobble with different logged data.
  Its reason for existing is the bug we caught — see the bug story.
- **Degenerate CI** — a fake, zero-width error bar that happens when the thing you're resampling is a bag of
  identical numbers (every at-bat starts on the same square). We flag it and print "not applicable," never a
  result. Catching this was the workstream's headline lesson.
- **Markov-sufficient** — whether the board carries *enough* of the situation to be exactly right. If the
  true dynamics depend on more than count-plus-lamp, the board isn't Markov-sufficient, its own value is a bit
  off, and that's exactly what a DIVERGES tells you.
- **`SETUP_INCONCLUSIVE`** — the honest verdict when the setup is real and directionally agreed but the error
  bars can't certify it at this scale. Not "it works," not "it fails" — "not yet, and here's how many more
  at-bats it would take." A first-class result, written down in advance so a disappointing number can't be
  quietly retold as a win.
- **The myopic ceiling (WS4)** — the measured cap on what a greedy chooser can reach (about `0.003` runs).
  WS5's job is to climb over it by valuing the setup; beating it is the falsifiable target.
