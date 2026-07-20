# WS4 in Plain English: The First Recommendation, and the Ceiling on Greed

This is the intuition guide for Workstream 4. No equations you have to solve — just the ideas, in
order, so that when you read the paper or the notebook the math is already familiar. If you only read
one document about WS4, read this one first.

---

## What we actually found (2021–2025)

We ran WS4 on the full five seasons of real Statcast data (2021–2025) — training the pieces on
2021–2023 and scoring the `1,419,590` held-out pitches of 2024–2025. Here, in plain words, is what came
back. (The rest of this guide explains *how* each piece works; this section is the *result*.)

**The trust gate passed.** Before trusting a single recommendation, WS4 checks that the ruler works — it
asks the OPE machinery to recover the value of the pitcher's *actual* behavior, which we already know.
It did, exactly (the importance weights came out to 1, the tell-tale sign). So everything below is
trustworthy: when WS4 puts a number on a policy, we can believe the number.

**The greedy recommender does not beat real pitchers.** This is the headline. Every version of the
"best next pitch" recommender we tried — from a timid nudge away from what the pitcher did, all the way
to the bold "always throw the model's favorite" — scored **at or below** the pitchers' own choices. Not
once did it beat them, and the more boldly it deviated, the *worse* it did (the value slid from about
`−0.001` runs at a gentle setting to about `−0.009` at the boldest). Real MLB pitchers, it turns out,
are already very hard to out-guess one pitch at a time. That's not the machine failing — it's an honest,
informative result, and it's the same story we saw on our test worlds, now confirmed on real baseball.

**The sequencing-specific edge is real — but far too small to act on.** Remember the clean experiment:
give the *same* recommender either just the situation (`C`) or the situation plus the full pitch order
(`O`), grade both with the same ruler, and the `O − C` gap is the pure value of *knowing the order*. On
real data that gap is genuinely there — it's statistically distinguishable from zero once the
recommender deviates a little — but it is *minuscule*: about `+0.0001` to `+0.0003` runs, and it grows
only when the recommender wanders so far from real pitching that the estimate itself gets shaky. So
"does knowing the sequence let a greedy chooser make better calls?" gets a precise answer: *technically
yes, practically no.* It is the real-data twin of the ceiling we measured on the test world.

**Almost every "best pitch" call is a coin-flip.** When WS4 asks how confident it is that its top pick
really beats the runner-up, the average confidence is only about **57%** — barely better than a
toss-up — and about **99%** of its calls are statistical ties at the 95% level. This is honest, not
broken: at pitch-family resolution several pitches usually sit within each other's error bars, so "the
single best next pitch" is rarely a resolvable question. The right output is a *shortlist*, never a
cocky single pick.

**Why this points to the sequential models next.** Put it together: the recommender is trustworthy (the
gate passed), it can't beat real pitchers one pitch at a time, and the sliver of value that *order* adds
is real but too small for a greedy chooser to cash. That is exactly the *ceiling on greed* this chapter
is about — now confirmed on real data, not just the synthetic fixture. The value that's left lives in
the **setup**: throwing a pitch to make the *next* one better, which pays off in a future situation a
one-pitch-at-a-time model literally cannot see. Catching that is the job of the sequential rungs —
**WS5** (which can value a setup through the count) and **WS7** (offline RL) — and their bar to clear is
precisely this: beat the ceiling WS4 just measured.

---

## What WS4 is, and the one-sentence version of the result

The whole project is a **ladder**. WS1, WS2, and WS3 answered *does the pitch sequence help you
predict things?* — what's thrown next, and what happens. WS4 is the first rung that asks the next
question: ***what should he throw — and can we prove it?*** That's the jump from **prediction** to
**prescription**, and it's a big one, because a prediction can be checked against what happened but a
recommendation is about the pitch he *didn't* throw, which we never get to see.

Here's the one-sentence result, and it's an honest "inconclusive" with real content: **a greedy,
one-pitch-at-a-time recommender can't cash in the kind of sequencing effect we planted, because that
effect rewards *setting up* the next pitch — and we measured exactly how little of it greed can reach
(about 0.003 runs out of a 0.03-run trick).** So the "inconclusive" isn't a shrug; it's the machine
telling the truth about a real limit, and it's precisely why the later workstreams (WS5, WS7) exist.

---

## Prediction vs prescription (why this rung is different)

Every rung before this one was **predictive**: it looked at what actually happened and asked how well
the pitch history forecast it. You can always grade a prediction — the outcome is right there in the
data. **Prescription** is different: to say "he should have thrown a slider" you have to estimate the
value of the pitch he *didn't* throw, and there's no answer key for that. This is why the whole
prescriptive half of the project is wrapped in a special kind of caution called **off-policy
evaluation** (OPE) — a set of techniques for judging a policy you *didn't* run, from data logged under
the policy you *did* run. WS4's real job is to do that *honestly*, and to refuse to answer when it
can't.

---

## The trust gate: prove the ruler works before you measure anything

Before WS4 reports the value of any recommendation, it does something that sounds almost too obvious to
mention — and is the single most important discipline in the project. It takes the policy whose value we
*already know* (the pitcher's actual behavior — its value is just the average reward we observed) and
asks the OPE machinery to recover that known value. If the machinery can't even reproduce the value of
the policy we already ran, it has no business estimating the value of a policy we *didn't* run.

We call this **behavior-policy recovery**, and it is the **gate**: it runs *first*, and if it fails,
the pipeline prints `FAILED_GATE` and **stops** before printing a single recommendation value. Nothing
below a failed gate is trustworthy. On both our test worlds the gate passes exactly — the importance
weights come out to exactly 1, which is the tell-tale sign that "recommend what he already does" was
scored as literally identical to what he did. Only then does WS4 go on to score the *actual*
recommendations. (In the project's words: *OPE before policy.*)

---

## Thompson sampling in plain words: vote across plausible worlds

So how does WS4 turn WS3's numbers into a recommendation? WS3 hands it a **what-if card** for every
decision — the expected run value of each of the eight pitch families, if it were the one thrown — each
with an error bar. WS4 needs to turn "here are eight values, give or take" into "here's how to choose,"
and it does it the way a careful gambler would, using a classic idea called **Thompson sampling**:

> Imagine a thousand parallel worlds. In each world, draw a plausible value for every pitch from inside
> its error bar, and "throw" the pitch that came out best *in that world*. Then recommend each pitch as
> often as it won across the thousand worlds.

A pitch that's clearly best wins almost every world and gets recommended almost always. Two pitches that
are within a whisker of each other split the vote. The beauty is that the recommendation is automatically
a *distribution*, not a single cocky pick — the model's uncertainty is baked into how boldly it
distinguishes pitches. That's exactly what you want from a recommender that's honest about what it
doesn't know.

---

## The honest-κ story: whose error bars are these, anyway?

Here's a subtlety we could have hidden and chose not to. WS3's error bars measure the wrong thing for
Thompson sampling — and fixing it is a real judgment call we report out loud.

WS3's "give or take" on each pitch is about `0.17` runs. But that `0.17` is how much *one pitch's*
result bounces around — a whiff here, a home run there. It is **the noise of baseball**, not the model's
uncertainty about the *average* value of throwing that pitch. Thompson sampling needs the second thing:
how sure are we about the *average*, not how scattered are individual pitches. And averages are far more
certain than single outcomes — average a few hundred similar pitches and your uncertainty about the
average shrinks about twentyfold, from `0.17` down to under `0.01`.

If we'd sampled with the raw `0.17`, every pitch would look equally good and the recommender would shrug
at everything (a perfectly *honest* shrug, but a useless one). So we shrink the error bars by that
"square-root-of-a-few-hundred" factor — a single knob we call **κ**, set to `0.05` — to turn "the noise
of baseball" into "our uncertainty about the average." We expose that knob, we explain it, and we report
how the results depend on it. The honest sentence is: *the model's error bars were about the noise of
the game, not its own confidence, so we had to shrink them to something meaningful — and we say so
rather than burying it in the sampler.*

---

## The prescriptive ablation: same chooser, three qualities of information

The centerpiece exhibit is a clean experiment. We give the **same** Thompson recommender three different
qualities of information and see whether the richer information changes the *value* of its
recommendations:

- **C** — it knows only the *situation* (count, base-out, score, handedness, the pitcher's general
  repertoire).
- **L1** — the situation *plus the single last pitch*.
- **O** — the situation *plus the full ordered sequence* of the at-bat so far.

Then — and this is the trick that makes it clean — we grade all three recommenders with **one and the
same ruler** (the richest evaluator). If we graded the "full-order" recommender with a "full-order"
grader and the "context" recommender with a "context" grader, we couldn't tell whether the policy got
better or the *grader* just grades differently. Same ruler for all three means the *only* thing that
differs is what the recommender was allowed to know — so if the "full-order" recommender's value beats
the "context" one's, that gap (**O minus C**) is genuinely the value of *knowing the order*.

That **O − C gap** is the whole ballgame. And here's the discipline: the recommenders often beat the
pitcher's actual habits in raw value — but that's usually for **count reasons**, not sequencing (the
synthetic pitcher throws out of habit, not optimally, so a count-aware chooser beats him without knowing
anything about order). The raw "beats behavior" number is a trap. Only the **O − C gap** isolates
sequencing. That's why the paper leads with the gap and the notebook prints a warning whenever a raw
gain shows up.

---

## The myopic ceiling: the chapter's real story

Now the payoff, and the reason WS4's "inconclusive" is a *finding*, not a failure.

We tested WS4 on a world where we planted a real sequencing effect ourselves: **when the pitcher makes
a big velocity jump into the previous pitch, the next pitch gets more whiffs** — worth about `0.032`
runs. WS3 already proved it can *see* this effect (it recovered 95% of it as a prediction). So can WS4
turn it into a recommendation? Mostly **no** — and the reason is beautiful.

The trick rewards a velocity jump *that already happened*. By the time the current pitch is being chosen,
the jump is **history** — so the whole `0.032` reward-boost is already owed to *every* pitch he could
throw now, equally. A **greedy chooser** — one that only cares about *this* pitch's reward — can't
manufacture that boost by its choice, because the boost doesn't depend on the choice. All it can grab is
the sliver where some pitch types happen to finish a boosted count slightly better than others — and we
**measured that sliver: about `0.003` runs**, out of the `0.032` total.

Where's the other `0.029`? In the **setup**. The valuable move is throwing the pitch that *creates* the
velocity jump — one pitch *earlier* — so the boost is there for the *next* pitch. But that value pays off
in the *next* situation, and a greedy, one-pitch-at-a-time recommender literally cannot see the next
situation. It's structurally blind to setups.

So when WS4 says "I can't find a sequencing edge here," it is **exactly right**: the edge is real, but
it's a setup, and greed can't cash a setup. To be sure the machine isn't just broken, we ran a
self-test — we built a world with a *genuinely greedy* edge and confirmed the machinery *does* light up
("exploited") when there's something a greedy chooser can grab. So the machinery sees fine; the effect is
genuinely **sequential, not greedy-reachable**. In one line:

> **We measured the ceiling on greed at about `0.003` runs of a `0.03`-run effect. WS4's "inconclusive"
> is the machine telling the truth — and the sequel workstreams (WS5, WS7) exist precisely to catch what
> greed can't: the value of setting up the next pitch.**

That's not a loose end. It's a **falsifiable target**: WS5 and WS7 pass their version of this test by
*beating `0.003`* — by valuing the setup pitch a bandit provably can't.

---

## The ~99%-toss-up honesty

One more number every practitioner should hear. When WS4 looks at how confident it is that its *top*
recommended pitch really beats the *runner-up*, the average confidence is only about **62–66%**, and
about **99% of its recommendations are toss-ups** at the 95%-confidence level. That is not a bug — it's
an honest statement of how resolvable "the single best next pitch" actually is under a greedy model at
pitch-family resolution: several families are usually within their error bars of each other. The right
takeaway for a coach is a *shortlist*, not a *pick* — "these three are basically equivalent here," which
is exactly what the Thompson distribution already gives you. A confident single pick would be over-reading
noise, and the whole uncertainty machinery exists to stop that.

---

## What the result will mean (the branches)

When we run WS4 on the real data, the answer is read on a few axes. The notebook and paper have the full
write-ups; here's the plain version.

**First, the gate.**
- **G-PASS** — the ruler works; read on.
- **G-FAIL** — the ruler is broken on this data; **stop** and fix the propensities before believing
  anything.

**Does the recommender beat the pitcher's habits? (the value axis, at a moderate setting)**
- **V+** — yes, its worst-case value clears the pitcher's actual value. Real, but check it isn't riding
  on a few over-weighted pitches — and remember this alone isn't *sequencing* (it's usually count-driven).
- **V0** — a tie. The expected honest outcome for a conservative recommender on real baseball.
- **V−** — it loses to the pitcher. A model or support problem; diagnose before reading anything else.

**Does *order* actually help the recommendation? (the O − C gap — the real question)**
- **P+** — the ordered recommender beats the context one (the gap clears zero). Order is *greedily
  actionable* — this would **beat the ceiling** we measured, so it's a strong claim: cross-check it against
  WS3 (was the order even predictive there?) and hand it to WS7.
- **P0** — a tie. Two readings: either there's nothing there, *or* — the interesting one — the sequencing
  signal is present but **below the greedy floor** (it sits above the null world's overfitting baseline but
  its own error bar still covers zero). That's the WS4 fixture's own reading: a setup effect greed can't
  reach, routed to WS5/WS7.
- **P−** — the ordered recommender is *worse*. The extra ordered features overfit — the null-world
  signature. Build the policy from the simpler view, and read it as "order didn't help *and* cost a little,"
  not "order hurts."

**And a parallel check on every setting: do the OPE estimators agree?**
- **CONSISTENT** — they agree; the value is a coherent read.
- **INCONCLUSIVE** — they disagree by more than their own error bars, so we *refuse to call it* — "not
  it works." (Usually a high-deviation, low-sample-size setting.)

The most anticipated real-data reading is **G-PASS, tie on value, P0-sequential-not-myopic, estimators
consistent** at a moderate setting — the fixture's own story, and the motivation for the sequential rungs.

---

## How to run it

Everything is in **RUNBOOK.md, step WS4.1**. The one thing to know up front is the **dependency**: WS4
**consumes WS3's saved models** and trains nothing itself on real data, so **Step WS3 must have run
first** (the per-view `behavior_<view>.joblib` and `outcome_<view>.joblib` files under `results/ws3/`).
From the repo on your machine:

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws4_bandit/run_ws4.py --table data/processed/decision_table.parquet --ws3-dir results/ws3/ --out results/ws4/ --views C L1 O
```

It **loads** WS3's models and does no training, so it's much lighter than WS3 (estimate: tens of minutes,
mostly the Thompson simulation and the OPE bootstraps). The two quick synthetic checks need no data and no
WS3 dependency (WS4 fits small WS3 stacks itself and checkpoints them) — keep the game count small because
those internal WS3 fits are the cost:

```powershell
python workstreams/ws4_bandit/run_ws4.py --synth null --out results/ws4_null/
python workstreams/ws4_bandit/run_ws4.py --synth positive --out results/ws4_pos/
```

**What to paste back**, per RUNBOOK: the entire printed **headline block** and the frontier CSV
(`Get-Content results/ws4/frontier_real.csv`). **What to look at first**, in order:

1. **The gate.** The line after `rows` must say `behavior recovery: PASS`. If it says `FAILED_GATE`,
   stop and paste it — nothing below it is real.
2. **The O − C gap**, in the "SEQUENCING-PRESCRIPTION GAPS" block — *not* the raw "beats behavior"
   column. A gap whose lower bound clears zero (`SEQ_EXPLOITED`) is the strong result; anything else is
   an honest negative with its own meaning.
3. **The value-vs-α frontier and ESS.** ESS should fall from 100% as you dial toward the bandit — that's
   the price of deviating. A collapse to a few percent means the recommendation is far outside what the
   pitcher does, and its value should be read with suspicion.
4. **The ambiguity line.** A mean "top-beats-runner-up" confidence near 0.6 and a high toss-up share is
   the honest read on how resolvable the best pitch is — not a failure.

---

## Mini-glossary

- **Prescription vs prediction** — prediction grades what happened; prescription values a pitch he
  *didn't* throw. WS4 is the first prescriptive rung, so it lives inside the OPE caution.
- **Off-policy evaluation (OPE)** — judging a policy you didn't run from data logged under the policy you
  did run. The whole toolkit WS4's gate and estimators come from.
- **Propensity** — how likely the pitcher was to throw each family in a situation, `μ(a | s)` — WS3's
  behavior model. The "what does he usually do here" number the OPE math needs.
- **Importance weight** — the ratio of "how likely the *recommendation* was to throw this pitch" to "how
  likely *he* was." It's how OPE re-weights the logged pitches to look like the recommended policy; big
  weights are dangerous (a few pitches carry the whole estimate).
- **ESS (effective sample size)** — how many logged pitches your re-weighting *effectively* uses. It
  falls as the recommendation strays from behavior; low ESS = a shaky estimate.
- **Thompson sampling** — turn each pitch's error bar into a recommendation by drawing plausible values
  across many imagined worlds and voting on the winner. Naturally gives a distribution, not a cocky pick.
- **q̂ (the what-if card)** — WS3's expected run value for each pitch if it were thrown. A *prediction*
  under a hypothetical pitch, **not** a promise that switching to it would *cause* a better result.
- **κ (posterior scale)** — the shrink factor (`0.05`) that turns "the noise of baseball" into "our
  uncertainty about the average," so Thompson sampling has the right error bars. The one confidence knob,
  reported openly.
- **Myopic (greedy)** — judging a pitch only by *its own* reward, one pitch at a time — a bandit. Can't
  value a setup, because a setup pays off in the *next* situation.
- **Setup pitch** — a pitch thrown to make the *next* pitch better (e.g. creating the velocity jump that
  earns the next whiff). The value a greedy chooser is structurally blind to — and the thing WS5/WS7 exist
  to value.
- **The myopic ceiling** — the measured cap on how much of a sequencing effect greed can reach (about
  `0.003` runs of our `0.032`-run planted effect). WS4's honest "inconclusive," turned into a target for
  the sequential rungs to beat.
- **INCONCLUSIVE** — the verdict when the OPE estimators disagree by more than their own error bars. Not
  "it works," not "it fails" — "we refuse to call it." A first-class result, written down in advance so a
  disappointing number can't be quietly retold as a win.
