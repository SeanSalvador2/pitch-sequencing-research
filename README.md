# Pitch Sequencing Research

A comparative study of approaches to **pitch sequencing** in baseball — the decision of
what to throw next given the game state *and the sequence of pitches already thrown*.
Rather than chasing a single best predictor, the project builds seven methods across
paradigms (statistical, Bayesian, stochastic-process, classical ML, deep learning,
reinforcement learning) and compares them honestly on one shared benchmark.

The central question is a **rigor ladder**, not a horse race:

> How much evidence for sequencing survives progressively harder tests — from descriptive
> order patterns, to out-of-sample *outcome* dependence, to counterfactual *policy* value?

Three findings are routinely conflated, and the whole design exists to keep them apart
(SPEC §0):

1. **Selection structure** — prior pitches help predict *what is thrown next*.
2. **Predictive sequencing value** — prior pitches help predict the *outcome* of the
   current pitch, after conditioning on the current pitch and game state.
3. **Prescriptive / causal value** — *changing* the sequence would improve outcomes.

The first is easy; the second is hard; the third needs assumptions public data cannot
fully satisfy (we never see the pitch that wasn't thrown). Every model is trained on the
same **five nested state views** — `C` (context only), `U` (unordered prior pitches),
`L1` (previous pitch), `O` (full ordered sequence), `OM` (O + matchup memory) — and the
differences between them *are* the sequencing evidence, read through two headline
ablations: `Δ_order = Loss(min[U, L1]) − Loss(O)` and `Δ_matchup = Loss(O) − Loss(OM)`.

See **[SPEC.md](SPEC.md)** for the full design contract: the shared decision table, the
five views, the fixed action space and reward, the temporal split, the evaluation
harness, the off-policy-evaluation interface, and the synthetic-fixture correctness
oracle.

---

## Phase 1 status: COMPLETE

All of Workstream 0 (the shared `pitchseq` foundation) and all seven modeling
workstreams are code-complete, unit-tested, and documented against synthetic fixtures and
the committed 20k sample. Each workstream ships the same four things: **code** (`model.py`
+ a `run_wsN.py` CLI), **tests** in the top-level suite, a research-grade **notebook**
scaffold (validates and compiles, zero saved outputs), and three **docs** — `PAPER.md`,
`THEORY.md`, and a plain-English `SEAN-README.md`.

The full suite is **325 tests, green**. Phase 1 deliberately does **no** full-scale
training and **no** notebook execution — the 3.85M-pitch dataset is not in this
environment; that is Phase 2 (below).

| WS | What it is | Delivered | Suite |
|----|------------|-----------|-------|
| **0** | Shared foundation: decision table, five state views, reward, temporal splits, eval harness, OPE, synthetic worlds | `src/pitchseq/*` + `eval/*` + `synth.py` + build-table CLI | **119** |
| **1** | Empirical-Bayes conditional tables (Dirichlet-shrunk selection + partial-pooled run value) — the transparent baseline | code + tests + notebook + papers | **141** |
| **2** | Bayesian variable-order Markov "pitch grammar" (hierarchical Dirichlet backoff over ordered tokens) | code + tests + notebook + papers | **172** |
| **3** | GBDT behavior + decomposed outcome stack — **the centerpiece**: first model to run all five views; supplies the q̂ grid + propensities WS4/5/7 import | code + tests + notebook + papers | **203** |
| **4** | Bayesian contextual bandit — the first prescriptive rung (myopic "best next pitch"), evaluated only through OPE | code + tests + notebook + papers | **226** |
| **5** | Tabular MDP / controlled Markov reward process — the first *sequential* rung; can value a setup pitch; the transparent OPE cross-validator | code + tests + notebook + papers | +31 |
| **6** | Deep sequence: capacity-matched GRU ladder (+ optional compact Transformer demo) — learned representation vs engineered history | code + tests + notebook + papers | **297** |
| **7** | Conservative offline RL + OPE + exploitability read-out — **the capstone**; caps the ladder with the study's frontier figure | code + tests + notebook + papers | **325** |

(Suite column is the cumulative green count in the dispatch log as each unit landed; WS5
added 31 tests. The final full suite is **325**.)

### What the synthetic validation showed

Every workstream re-runs the SPEC §11 correctness oracle — a **null world** with no order
effect and a **positive world** with a planted previous-pitch mechanism — as its own
acceptance gate. The foundation's oracle passes deterministically: on the null world
`Δ_order = −0.0145` (permutation test does not fire) → NULL_CONFIRMED; on the positive
world `Δ_order = +0.0687` with the planted effect recovered at 0.71× magnitude → 
POSITIVE_CONFIRMED. Six findings carry through the ladder:

**Mechanism visibility (the 0.955-vs-0.03 contrast).** The lookup tables (WS1) can only
see the planted velo-transition effect *indirectly*, through pitch-family proxies, and
recover it at ~0.03 of full magnitude. The GBDT centerpiece (WS3), whose `O` view carries
the ordered velo transition directly, recovers the same planted mechanism at **0.955** of
magnitude. Same effect, same data — the difference is entirely what the model's state can
represent. This is the ladder's core lesson made quantitative.

**The myopic ceiling.** WS4's honest verdict on the positive world is
`SEQ_INCONCLUSIVE_MYOPIC`: a greedy one-pitch-at-a-time recommender can exploit only
~0.003 run of a ~0.03-run planted effect, because the effect rewards *setting up* the next
pitch, not the current one — and that sliver sits below the OPE noise floor. The effect is
present (WS3 found it) but not myopically prescriptive. Measuring that ceiling is WS4's
contribution and the motivation for the sequential rungs.

**The setup-certification variance story.** WS5 is the first rung that can value a setup.
On the positive world all three independent estimators point the same way
(`FQE +0.0079`, `step-wise-DR +0.0107`, model-based `+0.0277` at α=1), a unit test proves
the machinery deliberately banks a setup and comes out ahead — yet the lower confidence
bound does not clear the ceiling at synthetic scale, so the honest verdict is
`SETUP_INCONCLUSIVE`, with certification deferred to full-data scale. Along the way WS5
caught its own FQE error bars being structurally degenerate (zero-width by construction)
and rebuilt them the expensive, honest way (a refit cluster bootstrap). WS7 replicates the
pattern at the RL level: `RL_EVIDENCE_DIRECTIONAL`, robust across scales, certification
honestly deferred.

**The three-family Δ_matchup cost.** Three independent model families that fit the matchup
view — WS3's GBDT, WS5's MDP, and WS6's GRU — each show a significantly *negative* outcome
`Δ_matchup` on the synthetic worlds: adding matchup features where no matchup effect exists
is pure overfit cost (WS6's is `−0.0036`). It is a pre-written honest branch, not a
surprise.

**The opacity contrast.** WS2's grammar writes its ordered rules down in a table you can
read; the WS6 GRU uses the same grammar to lower its loss but its explicit probabilities do
not cleanly expose it (the motif-rediscovery probe reports no clean motif at demo scale).
Learned-but-opaque vs explicit-but-legible, measured rather than assumed.

**The exploitability trade.** WS7's capstone read-out shows the concentrated conservative
policy is roughly **3× more exploitable** than diffuse observed behavior — a fixed batter
response can sit on a predictable pitcher. Exploitability is reported as a *cost* the
frontier figure trades off against value, not a win.

### Honest-findings-first, in plain words

The study is built to report the truth either way — "order matters" and "order mostly
doesn't" are both clean, publishable results (SPEC §0 discipline 5, and §13). Three rules
keep it honest:

- A small *negative* order effect is read as **"consistent with no order effect," never as
  "order hurts"** — the way `Δ_order` is measured is optimistically biased downward, so we
  only claim an effect when it is clearly, significantly positive.
- An **"inconclusive" verdict is a first-class result we write up**, with its own
  interpretation — not a failure to paper over. Estimator disagreement means INCONCLUSIVE,
  not "it works."
- When an error bar turns out to be **structurally fake** (zero-width by construction), we
  catch it and rebuild it the expensive, honest way — and we never certify a result on a
  fake confidence interval.

---

## Phase 2: running it on the full data

The full `data/raw/statcast.db` (~3.85M pitches, 2021–2025) lives only on the local
desktop — **data locality, not hardware, is the gate.** Nearly everything is CPU work.
Phase 2 is driven entirely from **[RUNBOOK.md](RUNBOOK.md)**, a paste-back recipe:

1. Each step gives one line of *what / why*, exact copy-paste PowerShell commands, a rough
   runtime, and precisely *what to paste back*.
2. Every heavy script is resumable and checkpointed, writes big artifacts under gitignored
   `data/processed/`, `results/`, or `artifacts/`, and prints a compact **headline block**
   plus wall-clock and peak RAM — the raw material for the SPEC §7 performance-vs-compute
   Pareto plot.
3. Headline numbers are pasted back for review against the SPEC acceptance checks, then the
   real values are filled into the pre-written papers and notebooks.

Steps run in build order: **environment check → build the decision table → WS1 → WS2 → WS3
→ WS4 → WS5 → WS6 → WS7 → the frontier figure.** WS4/WS5/WS7 consume WS3's saved artifacts
rather than refitting their own outcome models.

**CPU-first.** WS1–WS5 and WS7 are CPU steps on the full data. **Only WS6** benefits from a
GPU, and it ships two paths: a local CPU path and a self-contained free **Colab T4**
notebook (`workstreams/ws6_deep_seq/colab_ws6.ipynb`) that trains the O-view GRU in about
**one hour** (vs several hours on CPU). The desktop's AMD GPU is not recommended (no stable
Windows ROCm build); use CPU for correctness or Colab for speed.

### Data

Statcast pitch-level data, 2021–2025 (~3.85M pitches, one Hawk-Eye tracking era). Pulled
locally with `pull_statcast.py` (data is gitignored — it lives on your machine, not in
git); only a 20k random-row sample (`sample_statcast_2024.parquet`) and `SCHEMA.csv` are
committed:

```bash
pip install -r requirements.txt
python pull_statcast.py --sqlite        # -> data/raw/statcast.db + per-season parquet
```

---

## Repository map

```
pitch-sequencing-research/
  SPEC.md                       # the design contract (read this first)
  README.md                     # this file
  ORCHESTRATION_PLAN.md         # dispatch log + every design decision (project memory)
  RUNBOOK.md                    # the Phase-2 paste-back recipe, in build order
  SCHEMA.csv                    # committed column schema
  sample_statcast_2024.parquet  # committed 20k random-row sample (schema/dtype validation)
  pyproject.toml                # package `pitchseq`; extras: [ml] LightGBM, [deep] torch
  pull_statcast.py / 00_pull_statcast.ipynb   # data pull (existing)
  configs/default.yaml          # pitch-family map, splits, windows, thresholds
  src/pitchseq/
    io, families, reward, rolling, outcomes, decision_table, states,
    sequences, splits, synth, runmeta, config, build_table    # WS0 foundation
    eval/
      predictions, metrics, baselines, falsification,
      predictability, ope, harness                            # the shared harness
  workstreams/
    ws1_eb_tables/       model.py  run_ws1.py  PAPER  THEORY  SEAN-README
    ws2_bayes_markov/    model.py  run_ws2.py  PAPER  THEORY  SEAN-README
    ws3_gbdt_stack/      model.py  run_ws3.py  PAPER  THEORY  SEAN-README   (centerpiece)
    ws4_bandit/          model.py  run_ws4.py  PAPER  THEORY  SEAN-README
    ws5_tabular_mdp/     model.py  run_ws5.py  PAPER  THEORY  SEAN-README
    ws6_deep_seq/        model.py  run_ws6.py  PAPER  THEORY  SEAN-README  colab_ws6.ipynb
    ws7_offline_rl/      model.py  run_ws7.py  PAPER  THEORY  SEAN-README   (capstone)
  notebooks/             ws1 … ws7 research-notebook scaffolds (synth-mode by default)
  tests/                 correctness suite against synthetic fixtures (325 tests)
  data/                  gitignored (raw/, processed/) — local only
```

Every `workstreams/wsN_*/` folder reads the shared decision table + state builders, writes
predictions in the standard schema, and is scored by the shared harness. No workstream
re-implements evaluation.

---

## How to verify

```bash
pip install -e ".[ml,deep]"     # [ml] = LightGBM (WS3+), [deep] = torch (WS6 only)
python -m pytest tests/ -q
```

Expected: **325 passed**. The suite is offline, CPU-only, and runs in minutes — it is the
gate that proves the harness is correct before any real model is trusted.

---

## Where to read next

- **[SPEC.md](SPEC.md)** — the contract: decision table, five views, reward, splits,
  harness, OPE, and the synthetic-oracle acceptance tests.
- **[ORCHESTRATION_PLAN.md](ORCHESTRATION_PLAN.md)** — the project's memory: every design
  decision with its rationale, and the full dispatch log with verdicts.
- **[RUNBOOK.md](RUNBOOK.md)** — the Phase-2 recipe: exact, copy-pasteable steps per
  workstream in build order.
- **`workstreams/*/SEAN-README.md`** — per-topic plain-English guides; start there for any
  one workstream, then read its `PAPER.md` and `THEORY.md`.
