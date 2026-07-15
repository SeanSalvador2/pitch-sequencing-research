# WS7 in Plain English: The Fanciest Model, the Most Safety Gear, and an Honest "Not Yet"

This is the intuition guide for Workstream 7 — the **capstone**, the last chapter of the study. No equations
you have to solve, just the ideas in order, so that when you read the paper or the notebook the math is already
familiar. If you only read one document about WS7, read this one first.

---

## What WS7 is, in one paragraph

The project is a **ladder**, and WS7 is the top rung. WS1–WS3 asked *does the sequence help you predict
things?* WS4 asked the first prescriptive question — *what should he throw?* — with a greedy, one-pitch-at-a-
time recommender, and hit a wall it named: greed can only cash the sliver of a sequencing effect that's about
*this* pitch. WS5 built the simplest model that could think one pitch ahead — a little board game — and proved
it *knows how* to value a setup pitch, but couldn't prove it earns money in the wild at our data size. WS7 is
the most ambitious model in the whole study: a flexible machine-learning value-improver (boosted trees inside a
"look one pitch ahead and keep the best follow-up" loop) that can represent far more than the board game — **and
it wears the most safety gear of anything we built.** Two trust gates before it's allowed to report a single
number, a "don't get greedy about pitches you barely saw" penalty baked into its planning, an honest
(expensive) error bar, and a rule that stops it from taking credit for the easy win. Its headline result is not
a number a coach acts on; it's the study's **final chart** — the frontier — and an honest verdict that matches
the two rungs below it: *the sequencing edge is probably real, but we can't certify it at this scale yet.*

---

## The two trust gates (prove the scale works before you weigh anything)

The project's iron rule is **OPE before policy**: before you trust a scorer to weigh a *new* plan, prove it can
weigh a *known* one. WS7 does this **twice**, and if either check fails the whole run stops and prints
`FAILED_GATE` — nothing below it means anything.

1. **Gate 1 — read back the pitcher's own value.** Point the scorer at the pitcher's *actual* policy and ask it
   to reproduce the value we already observed. If it can't read back a weight it already knows, the scale is
   broken (a wiring error — wrong propensities, a bad join). On our test worlds it passes exactly.
2. **Gate 2 — hit a known answer on a toy problem.** Run the scorer on a tiny made-up problem where we can
   compute the right answer by hand, and check it lands there. This catches a *bug in the scorer's math* that
   real data would hide, because real data never comes with an answer key. It passes too (it recovers a known
   `+0.6148` as `+0.6229`, within its error bar).

One gate calibrates the ruler on the real data; the other proves the ruler's mechanism on a problem with a
known answer. Only past **both** does the model get to recommend anything.

---

## Pessimism: don't pay for pitches the data barely saw

A flexible model has a bad habit: on pitches the real pitcher almost never threw in a situation, it has almost
no data, so its guess about that pitch's value is basically noise — and noise sometimes looks *great* by luck.
A greedy improver would happily recommend that lucky-looking, barely-seen pitch.

So we bake in **pessimism**. Before the model is allowed to prefer a pitch, we ask: *did the real pitcher throw
this here at least 2% of the time?* If not, we quietly dock a little from that pitch's value. That only ever
*lowers* the value of barely-seen pitches, and it only changes the recommendation when such a pitch was about
to win by a hair. The nice surprise: doing this actually **raised** the honest, held-out value a touch (from
`+0.0328` to `+0.0354`) — because the pitches it demoted were the over-rated flukes. It's "free honesty," and
it only bites about **2% of decisions**. (This is a plain-spoken version of a research idea called CQL; we say
straight up that ours is the simple stand-in, not the full thing.)

---

## The isolation story, told straight (this is the important part)

Here's the headline it would have been *easy* to write: **the fancy model beat the real pitchers' habits by a
mile** — a raw improvement of about `+0.035` in run value, with an error bar clear of zero. Sounds like a win
for sequencing, right?

**It isn't, and here's why.** Almost all of that win is just **count management** — throwing the right pitch
for the count (0-2 vs 3-1) — which *any* competent model gets right and which has **nothing to do with
sequencing**. The real pitchers' logged policy is habit-based, not optimized, so beating it on the count is
easy and expected.

To measure the *sequencing-specific* slice alone, we do the same trick WS4 and WS5 did: we run the **same**
value-improver on a stripped-down **count-only** version of the board, and subtract. Whatever the full ordered-
history model earns *over* the count-only model is the part that's actually about sequencing — everything else
cancels out.

That isolated slice is **tiny**: about `−0.0003`, with an error bar of `[−0.0129, +0.0157]` — i.e. buried in
noise, we can't even pin its sign. **But** — and this is the honest good news — that slice is reliably *bigger*
in the world where we *planted* a real sequencing effect than in the world where we didn't (`−0.0003` beats
`−0.0016`), and it stays bigger at every data size we tried. So our instrument is **discriminating** — it can
tell the two worlds apart — even though it can't yet put a confident number on the slice.

The verdict, therefore, is the same first-class "not yet" WS5 gave, now delivered by the fanciest model in the
study: **`RL_EVIDENCE_DIRECTIONAL`** — the evidence is real and points the right way, but the statistics aren't
enough to certify it at this scale. And to prove that's a *data-size* statement and not a broken detector, we
ship a unit test that feeds the machinery an obvious, clear-cut sequencing edge and confirms it *does* fire the
full **`RL_EVIDENCE_CERTIFIED`** verdict. So the machine can stamp "certified" — this data just isn't big
enough to earn it. On the **null** world (no planted effect) the verdict is `RL_NO_CLAIM`, and the headline
spells out that the raw gain there is count-driven, not sequencing.

---

## The exploitability story: the best policies are the most predictable

Here's the study's game-theoretic punchline, and it's a genuine trade-off, not a free lunch.

Our optimized policies **concentrate** — they pick a favorite pitch for each situation. That makes them *more
predictable*. So we measured what a batter who "guessed along" could claw back: we built a fixed, simple model
of how batters swing (by count and pitch type), turned each count into a little guessing game, and solved for
how exploitable each policy is.

The answer: our optimized policies are about **three times more exploitable than the real pitchers' natural
mixing** (the FQI policy scores `+0.0459`, the count-only policy `+0.0475`, versus real behavior's `+0.0137`).
A batter who sat on the model's predictable pick would get roughly three times the edge he'd get against a real
pitcher who keeps him honest by mixing.

So **value and predictability trade off**. The model can find a higher-value plan, but that plan is more
forecastable, and a smart batter takes some of the value back. This is *model-dependent* by design — we picked
one fixed, readable batter model — so the exact "3×" would move with a different batter model, but the
*direction* (concentrate → get exploited) is robust. It's the honest bottom line of the whole prescriptive
arc: there is no free prescription; predictability is the price.

(We also report a small number called `B_seq` — about `+0.0099` bits — which is just "how much does knowing the
ordered history help you *guess the next pitch*." Small, and rising in deeper counts. Forecastable ordering
isn't the same as ordering that *helps the batter*, but it's part of the same predictability picture.)

---

## The frontier: a Consumer Reports chart for pitching policies

WS7's actual deliverable — the last figure of the study — is the **frontier**. Think of it as a Consumer
Reports chart where every policy in the whole project gets a row and five grades:

- **value** — how much run value it earns (with an honest low-end estimate);
- **predictability (`B_seq`)** — how much the ordered history gives away the next pitch;
- **exploitability** — how much a guessing batter claws back;
- **deviation** — how far it strays from what real pitchers actually do;
- **compute** — how expensive it is to build.

The chart plots them all: value versus deviation (colored by exploitability) on the left, exploitability versus
predictability (sized by compute) on the right. The behavior policy, the WS4 bandit, the WS5 board-game
designs, and the WS7 flexible policies at each aggressiveness setting are all labeled points — **ten rows in
all**. We deliberately **don't** mash the five grades into one score, because that would mean deciding how many
runs a bit of predictability is worth — a judgment call the study doesn't get to make for you. The chart shows
the trade-offs; you pick what matters. That refusal to crown a single winner *is* the point: the honest summary
of a rigor ladder is a trade-off surface, not a leaderboard.

---

## The whole arc, in three sentences

This is the closing synthesis of the entire prescriptive half of the study:

- **WS4 (greedy) couldn't even see it.** A one-pitch-at-a-time recommender is *structurally blind* to a setup —
  it can't value a pitch whose payoff shows up on the *next* pitch. It measured that blind spot exactly
  (`~0.003` of a `~0.03` effect).
- **WS5 (the board game) saw it but couldn't prove it.** The simplest model that thinks one pitch ahead *can*
  represent a setup — a unit test shows it deliberately throws the setup pitch and wins — but its honest error
  bar couldn't certify the edge at our data size.
- **WS7 (the flexible model) discriminates it but is noisier still.** The fanciest model tells the planted-
  effect world from the null world reliably, yet — because flexibility comes with more wobble per data point —
  its error bar is *wider* than the simpler board game's, so it's even further from certifying.

**Everything points to the same place: the sequencing edge is real and representable, but small, and certifying
it is a question of data scale — you need the full season.** And there's a general lesson tucked inside:
**flexibility costs variance.** The more expressive model can *represent* more of the effect, but it *measures*
it less precisely at the same data size — so the right tool for *certifying* a small effect at moderate scale is
often the *simplest* model that can represent it, and the flexible model earns its keep only at full scale.

---

## What the result will mean (the branches)

When we run WS7 on the real data, the headline is read on two axes. The notebook and paper have the full
write-ups; here's the plain version.

**The verdict axis (does the isolated sequencing slice clear the bar?):**
- **`RL_EVIDENCE_CERTIFIED`** — yes: the isolated sequencing slice's honest lower bound clears WS4's `+0.003`
  bar, its second estimator agrees, *and* WS5 agrees in sign. The strong claim — a certifiable prescriptive
  sequencing edge at the top of the ladder. Run the support checklist before believing it.
- **`RL_EVIDENCE_DIRECTIONAL`** — the fixture's own reading and the most likely real-data outcome at moderate
  scale: the slice points the right way and tells the worlds apart, but the error bar can't certify it. Report
  the evidence story plus the "how much more data would settle it" arithmetic, not a null.
- **`RL_NO_CLAIM` / `RL_EVIDENCE_ABSENT`** — nothing beyond the count-driven gain. Read as "no sequencing edge
  at this scale," never as "order hurts."
- **`RL_INCONCLUSIVE`** — the two scorers disagree. Then the honest verdict is *inconclusive, not "it works"* —
  the project's rule, verbatim.

**The exploitability axis (did the value cost predictability?):**
- **`E-cheap`** — value gained with no extra exploitability. The happy case (not what the synthetic worlds
  show — treat it with suspicion and the full checklist).
- **`E-costly`** — value bought with predictability (the verified `~3×` pattern). The real trade-off; read it
  alongside the batter's-side story.

The most anticipated real-data cell is **`RL_EVIDENCE_DIRECTIONAL` + `E-costly`**: a real, world-discriminating
sequencing signal that's below the certification floor and bought at a predictability cost — the capstone's own
synthetic reading, lifted to the real season.

---

## How to run it

Everything is in **RUNBOOK.md, Step WS7** (WS7.1 does the heavy lifting; WS7.2's read-outs are folded into the
same run). WS7 **needs WS3's saved artifacts** (behavior model + `q̂` grid — run Step WS3 first) and
*optionally* a WS5 report for the cross-check. From the repo on your machine:

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws7_offline_rl/run_ws7.py --table data/processed/decision_table.parquet --ws3-dir results/ws3/ --ws5-report results/ws5/ws5_report_real.json --out results/ws7/ --fqe-boot 100
```

The two synthetic self-tests need no real data and no `--ws3-dir` (WS7 fits small WS3 stacks itself and
computes the WS5 cross-check fresh): `--synth null --out results/ws7_null/ --fqe-boot 40` and
`--synth positive --out results/ws7_pos/ --fqe-boot 40`.

**The dials to know:** `--fqe-boot` is the honest refit-bootstrap replicate count (the expensive part, and the
whole reason the error bars are real) — the default is fine on synthetic worlds; **lower it to 50–100 on the
full data** if it's slow (it only coarsens the error bars, never the point estimates). `--lam` / `--floor` set
the pessimism penalty; `--n-iter` the look-ahead passes; `--beta` the batter-anticipation strength in the
exploitability game.

**Heads-up on time:** WS7 is the **longest CPU step after WS3** — the flexible model refits several times, and
the honest error bars refit the scorer hundreds of times. Budget an hour or a few; it's all CPU.

**What to paste back**, per RUNBOOK: the entire printed **headline block** and the frontier CSV
(`Get-Content results/ws7/frontier_real.csv`). **What to look at first**, in order:

1. **The two gates.** The `GATES` line must read *both* `behavior-recovery PASS` and `logged-bandit PASS`. A
   `FAILED_GATE` stops everything — paste it and we fix the wiring first.
2. **The two gaps.** The headline prints a raw `FQI-O − behavior` gap (**count-driven, not the basis**) and the
   `FQI-O − FQI-count` **isolation** (the real sequencing measure). Read the isolation and whether its lower
   bound clears `+0.003`.
3. **The verdict** — `CERTIFIED` / `DIRECTIONAL` / `NO_CLAIM` / `ABSENT` / `INCONCLUSIVE`.
4. **Exploitability** — expect the optimized policies to be *more* exploitable than behavior (the honest cost).
5. **The frontier figure** — `frontier_real.png`, the study's final deliverable.

---

## Mini-glossary

- **FQI (fitted-Q iteration)** — the flexible value-improver: guess each pitch-and-situation's value, then
  repeatedly improve the guess by looking one pitch ahead and keeping the best feasible follow-up. WS7 runs it
  two ways — a flexible boosted-tree version on the rich ordered state, and an exact table version on a coarse
  count-only state — so their gap isolates sequencing.
- **Pessimism (the support penalty)** — the "don't pay for pitches the data barely saw" rule: dock a little
  value from any pitch the real pitcher threw less than 2% of the time here, so the model doesn't chase
  lucky-looking flukes. A simple stand-in for the research idea called CQL.
- **Isolation** — the trick that separates the *sequencing* win from the *count-management* win: run the same
  improver on a count-only board and subtract. Whatever the ordered-history model earns *over* the count-only
  model is the sequencing slice — everything else (mostly count value) cancels. This is the verdict's basis,
  not the easy raw win.
- **Equilibrium** — the best a pitcher can guarantee in the count "guessing game" against a batter who always
  guesses his most-exploitable option: found by mixing his pitches so no single guess hurts too much (a small
  linear program solves it).
- **Exploitability** — how far short of that equilibrium a policy falls — i.e. how much a guessing batter claws
  back. Zero if you mix perfectly, large if you're predictable. Our optimized policies are ~3× more exploitable
  than real pitchers' mixing: the price of predictability.
- **Frontier** — the study's final Consumer Reports chart: every policy graded on value, predictability,
  exploitability, deviation from real behavior, and compute, with the trade-offs shown and no single "winner"
  crowned — because that would require deciding how many runs a bit of predictability is worth.
- **The two gates** — behavior recovery (read back the pitcher's own value on real data) and the logged-bandit
  regression (hit a known answer on a toy problem). Both must pass before any policy value is reported.
- **`RL_EVIDENCE_DIRECTIONAL`** — the honest verdict when the sequencing slice points the right way and tells
  the worlds apart but the error bars can't certify it at this scale. Not "it works," not "it fails" — "real,
  small, needs the full data." The same first-class "not yet" WS4 and WS5 reported, now from the fanciest
  model. Written down in advance so a disappointing number can't be quietly retold as a win.
- **The myopic ceiling (WS4)** — the measured cap on what a greedy chooser can reach (`~0.003` runs). WS7's job
  is to beat it by valuing the setup; the isolation slice clearing it is the falsifiable target.
- **Flexibility costs variance** — the arc's general lesson: a more expressive model can *represent* more of an
  effect but *measures* it less precisely at the same data size, so it needs *more* data to certify a small
  edge than a simpler model does.
