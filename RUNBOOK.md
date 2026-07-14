# Phase-2 Runbook

This is the single place to run the pitch-sequencing study on the **full** data. The full
`data/raw/statcast.db` lives only on the local desktop, so every heavy step runs here and the
printed headline numbers are pasted back for review against the SPEC acceptance checks.

## How to use this runbook

Each step below gives:

1. **What / why** — one line on what the step produces and why.
2. **Commands** — copy-paste PowerShell, always prefixed with
   `conda activate statcast; cd ~\pitch-sequencing-research`. Multi-command lines are joined
   with `;` (PowerShell's statement separator).
3. **Expected** — rough runtime and what it prints/writes.
4. **Paste back** — exactly what to copy into the review thread.

Conventions:

- Every heavy script is **resumable and checkpointed**, writes big artifacts under
  gitignored `data/processed/`, `results/` or `artifacts/`, and ends with a compact
  **headline block** plus wall-clock and peak RAM (the raw material for the SPEC §7
  performance-vs-compute Pareto plot).
- Runtimes are **honest estimates** for a desktop CPU on ~3.85M pitches; your machine may be
  faster or slower. Nothing here needs a GPU (only WS6's optional Transformer variant does,
  and that step ships a separate Colab path).
- Steps are intentionally small so you can run a few at a time and cheaply re-run failures.

Later workstream steps (WS1 → WS7) append their own sections **below**, in build order, as
each unit lands. This file grows; earlier steps do not change.

---

## Step 0 — Environment check

**What / why.** Install the package (with the ML extra) into the `statcast` env and confirm
the whole test suite is green before touching real data. If the suite is red, stop and paste
the failure — nothing downstream can be trusted.

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
pip install -e ".[ml]"
python -m pytest tests/ -q
```

**Expected.** The install is quick (seconds if already present). The test suite runs in a
few minutes on CPU and ends with a line like `NNN passed in <secs>s`.

**Paste back.** The final ~15 lines of the pytest output (the summary line with the pass
count and timing).

---

## Step 1 — Build the canonical decision table

**What / why.** Build the one canonical per-decision table (SPEC §3) that every workstream
consumes, from raw Statcast. The builder writes a `decision_table_<season>.parquet`
checkpoint per season, concatenates them into `data/processed/decision_table.parquet`, runs
the mandatory reward **sign check** (SPEC §5), and writes a run-metadata JSON sidecar.

Each season is built with a trailing 365-day look-back context window and then filtered to
that season, so the rolling leakage-safe features are identical to a whole-dataset build
while `--resume` can still skip finished seasons after an interruption.

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python -m pitchseq.build_table --source data/raw/statcast.db --out data/processed/decision_table.parquet --resume
```

To (re)build only specific seasons, add e.g. `--seasons 2021 2022`. Omitting `--seasons`
builds every season present in the source. `--resume` skips any season whose checkpoint
already exists, so re-running after a crash resumes cheaply.

**Expected.** This is the heaviest single-pass step. **Estimate: order 10–40 minutes** on a
desktop CPU for the full ~3.85M pitches (it is an estimate — the rolling-window aggregation
and per-season builds dominate; a resume run that only concatenates existing checkpoints is
seconds). Writes:

- `data/processed/decision_table_<season>.parquet` — one checkpoint per season,
- `data/processed/decision_table.parquet` — the final concatenated table,
- `data/processed/decision_table.runmeta.json` — run metadata (timing, peak RAM, row counts,
  sign-check verdict, column list).

It ends by printing a headline block like:

```
====================================================================
 decision-table build - headline
====================================================================
 source         : data/raw/statcast.db
 output         : data/processed/decision_table.parquet
 seasons        : 2021, 2022, 2023, 2024, 2025
 rows / season  : 2021=..., 2022=..., 2023=..., 2024=..., 2025=...
 total rows     : ~3,850,000
 columns        : 81
 built / resumed: [...] / [...]
 sign check     : PASS   (ball=-.. called_strike=+.. home_run=-.. strikeout=+.. swinging_strike=+.. walk=-..)
 elapsed (s)    : ...
 peak mem (MB)  : ...
 runmeta        : data/processed/decision_table.runmeta.json
====================================================================
```

**Paste back.** Two things:

1. the entire printed **headline block**, and
2. the contents of `data/processed/decision_table.runmeta.json`:

```powershell
Get-Content data/processed/decision_table.runmeta.json
```

Review checks: the sign check must read **PASS** (SPEC §5: strikes/strikeouts positive,
balls/walks/home runs negative), the total row count should be ~3.85M, and every requested
season should appear in `rows / season`.

---

_Workstream steps (WS1 empirical-Bayes tables → WS7 offline RL) are appended below as each
unit is built._

---

## Step WS1.1 — Empirical-Bayes conditional tables

**What / why.** Fit the WS1 tables (SPEC §12.1) on the real decision table and score them
through the shared harness — the transparent statistical baseline every later workstream
must beat, and the exhibit that exposes the *support problem*. It fits, per state view, a
hierarchical Dirichlet-multinomial **selection** table (next-pitch family) and a hierarchical
normal partial-pooling **run-value** table (`E[R | cell, family]`), with each level's
concentration fitted by maximum marginal likelihood; then it emits standard-schema
predictions per view, scores selection log loss / run-value MAE and the `Delta_order`
ablation against the `eval/baselines` references, and writes the support-diagnostics exhibit.
`OM` is intentionally omitted — matchup memory cannot be tabulated (D25); the run prints why.

Runs on the train seasons (2021–2023), evaluates on validation (2024) via the config split.

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws1_eb_tables/run_ws1.py --table data/processed/decision_table.parquet --out results/ws1/ --views C U L1 O --target both
```

The synthetic Phase-1 CI equivalents (no real data needed, each ~5 s) are
`python workstreams/ws1_eb_tables/run_ws1.py --synth null --out results/ws1_null/` and
`... --synth positive --out results/ws1_pos/`. If the clustered-bootstrap CIs are slow on the
full validation season, lower `--n-boot` (default 200) — it changes only the CI widths.

**Expected.** **Estimate: order 5–20 minutes** on a desktop CPU (it is an estimate). The
table fitting itself is cheap — counting plus a handful of 1-D `scipy` optimisations per
level — so the cost is dominated by building the history keys over ~2.3M train rows and the
clustered-bootstrap CIs over the ~0.77M-row validation season; a `--n-boot 50` run is
markedly faster. Writes, under `results/ws1/`:

- `predictions_real_<view>.parquet` — standard-schema predictions per view (C/U/L1/O),
- `ws1_report_real.json` — the full report (per-view losses, baseline references,
  `Delta_order` + CI, support diagnostics, fitted α/κ per level),
- `support_real.csv` — the support-diagnostics exhibit,
- `ws1_real.runmeta.json` — timing / peak RAM / row counts.

It ends by printing a headline block like (numbers are illustrative; the synthetic-null demo
is shown, a real run has lower selection losses and a sharper support explosion):

```
============================================================================
 WS1 empirical-Bayes tables - headline
============================================================================
 world / target : real / both
 rows           : train=...  val=...
 selection log loss (val) vs count-based references:
   C  : ...  ref[pitcher_count]=...       (<= ref)
   U  : ...  ref[pitcher_count_prev]=...  (<= ref)
   L1 : ...  ref[pitcher_count_prev]=...  (<= ref)
   O  : ...  ref[pitcher_count_prev]=...  (<= ref)
   references     : global_count_hand=...  pitcher_count=...  transition=...  pitcher_count_prev=...
 run-value MAE (val):
   C  : MAE=...  RMSE=...
   ... (L1 / U / O)
 Delta_order    : <small>  CI[<lo>, <hi>]
                  <D21 reading: small negative = consistent with no ordering effect>
 Delta_matchup  : N/A - OM tables infeasible for pure tables (D25 support problem)
 support (distinct cells / eval frac n<20 / deepest-level backoff rate):
   C  : cells=...      n<20=...   deepest-hit=...
   U  : cells=...      n<20=...   deepest-hit=...
   L1 : cells=...      n<20=...   deepest-hit=...
   O  : cells=...      n<20=...   deepest-hit=...
 elapsed (s)    : ...
 peak mem (MB)  : ...
 outputs        : results/ws1/ws1_report_real.json
============================================================================
```

**Paste back.** Two things:

1. the entire printed **headline block**, and
2. the support diagnostics CSV:

```powershell
Get-Content results/ws1/support_real.csv
```

Review checks (what I look at):

- **`Delta_order` sign, read per D21.** `Delta_order = Loss(min[U,L1]) − Loss(O)` is
  negatively biased under the null; a small **negative** value reads as *consistent with no
  ordering effect for selection*, never as "order hurts". A **significantly positive**
  `Delta_order` (CI lower bound > 0) is the only result that claims ordered selection
  structure. WS1's flat shrinkage typically pools the within-PA order signal toward the
  pitcher×count cell, so ≈0 is the expected, honest result — the whole point of shipping the
  fancier workstreams (WS2/WS3) that can resolve it.
- **The support table.** Distinct cell counts must explode from C to the history views and
  the low-support fraction (eval rows in cells with n < 5 / 20 / 50) must climb with view
  depth — the concrete statement of why deeper history keys cannot be estimated by naïve
  counting. (U vs O distinct-cell order is world-dependent: U keys the full capped multiset,
  O only the ordered last two.)
- **The baseline comparison.** WS1's history views (L1/U/O) should be **≤** the
  `pitcher_count_prev` reference — the serious hierarchical version should not lose to the
  quick count baseline. WS1-C vs `pitcher_count` may tie within noise.
- **`Delta_matchup` = N/A** is expected and correct: pure tables cannot pool matchup memory
  (D25); the run prints the pitcher–batter support numbers that justify it.

---

## Step WS2.1 — Bayesian variable-order Markov grammar

**What / why.** Fit the WS2 "pitch grammar" (SPEC §12.2) on the real decision table and score
it through the shared harness — the variable-order Markov test of whether *ordered* within-PA
history sharpens next-pitch **selection** beyond context and the previous pitch. It fits, per
state view, a two-axis hierarchical Dirichlet backoff (decision D29): a count×hand-with-pitcher
base rate (WS1's C-ladder) times a within-pitcher, per-depth ordered *lift* over pitch-family
tokens, with each depth's concentration fitted by maximum marginal likelihood (MoM fallback).
View mapping: `C` = order 0 (base), `L1` = order 1, `O` = variable order ≤ `K_MAX` (default 4).
`U` and `OM` are **not-applicable** for a Markov grammar (inherently ordered; no cross-PA
pooling — D29); the run prints why rather than fitting them.

It emits standard-schema selection predictions per view, scores selection log loss against the
`eval/baselines` references, computes the WS2 order edge `delta_order_L1 = Loss(L1) − Loss(O)`
directly (the shared `compare_views` needs a `U` view WS2 does not have — the `U` slot is never
abused), reports SPEC §10 **bits-of-predictability** `B_seq` (decision D31: `q_O` vs `q_C`,
sliced overall / by pitch-number / by count-bucket / two-strike / three-ball), writes the
grammar exhibits (effective-order distribution, top motifs), and runs the D30 acceptance wiring
— **detect** the grammar and confirm it **collapses under a stratified history permutation**.

Runs on the train seasons (2021–2023), evaluates on validation (2024) via the config split.

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws2_bayes_markov/run_ws2.py --table data/processed/decision_table.parquet --out results/ws2/ --views C L1 O
```

The synthetic Phase-1 CI equivalents (no real data needed, each ~5 s) are
`python workstreams/ws2_bayes_markov/run_ws2.py --synth null --out results/ws2_null/` (WS2's
**positive control** — the null world plants an order-2 no-three-in-a-row selection habit) and
`... --synth positive --out results/ws2_pos/`. Lower `--n-boot` (default 200) if the
clustered-bootstrap CIs are slow on the full validation season; raise/lower `--k-max` to change
the grammar-order cap.

**Expected.** **Estimate: order 5–20 minutes** on a desktop CPU (it is an estimate). The
grammar fitting itself is cheap — counting plus a handful of 1-D `scipy` optimisations per
depth — so the cost is dominated by building the lagged token contexts over the ~2.3M train
rows, the clustered-bootstrap CIs over the ~0.77M-row validation season, and one permutation
refit for the D30 control. A `--n-boot 50` run is markedly faster. Writes, under `results/ws2/`:

- `predictions_real_<view>.parquet` — standard-schema selection predictions per view (C/L1/O),
- `ws2_report_real.json` — the full report (per-view losses, references, `delta_order_L1` + CI,
  `B_seq` slices, effective-order distribution, top motifs, D30 verdicts, fitted α per depth),
- `motifs_real.csv` — the ranked grammar rules (ordered context → next-token lift vs suffix),
- `ws2_real.runmeta.json` — timing / peak RAM / row counts.

It ends by printing a headline block like (numbers are the synthetic-null demo; a real run has
lower selection losses, a fuller effective-order tail, and baseball-plausible motifs):

```
============================================================================
 WS2 Bayesian variable-order Markov grammar - headline
============================================================================
 world / K_MAX  : null / 4
 rows           : train=29,756  val=9,961
 selection log loss (val) vs count-based references:
   C  : 1.3405  ref[pitcher_count]=1.3348  (> ref)
   L1 : 1.3385  ref[pitcher_count_prev]=1.3488  (<= ref)
   O  : 1.3322  ref[pitcher_count_prev]=1.3488  (<= ref)
   references     : global_count_hand=...  pitcher_count=...  transition=...  pitcher_count_prev=...
 delta_order_L1 : +0.0063  CI[+0.0041, +0.0080]  (Loss(L1)-Loss(O); significant)
 delta_matchup  : N/A - OM is not-applicable for a Markov grammar (D29)
 B_seq (bits)   : overall=+0.0120  two_strike=+0.0264  three_ball=+0.0466
   by count-bucket: ahead=...  behind=...  even=...
   by pitch_number: t1=+0.000  t2=-0.001  t3=+0.023  t4=+0.026  t5=+0.034
 effective order: mean=0.593  >=2 mass=0.296  (tau=0.50)
   distribution   : k0=0.704  k1=0.000  k2=0.296  k3=0.000  k4=0.000
 top motifs (ordered context -> next-token lift vs suffix):
   SI SI -> P(SI) 0.35->0.22 (suppress, KL=0.061)
   FF FF -> P(FF) 0.31->0.20 (suppress, KL=0.051)
   ...
 D30 detect     : GRAMMAR_DETECTED  (O<L1=True, >=2 mass=0.296, repeat-motif=True)
 D30 control    : COLLAPSES_UNDER_PERMUTATION  (edge +0.0063 -> +0.0001, >=2 mass -> 0.000)
 elapsed (s)    : ...
 peak mem (MB)  : ...
 outputs        : results/ws2/ws2_report_real.json
============================================================================
```

**Paste back.** Two things:

1. the entire printed **headline block**, and
2. the ranked motifs CSV:

```powershell
Get-Content results/ws2/motifs_real.csv
```

Review checks (what I look at):

- **`delta_order_L1`, read under D21's logic (adapted).** D21 flags the SPEC §6
  `Delta_order = Loss(min[U, L1]) − Loss(O)` as *negatively biased* because `min[U, L1]` is the
  smaller of two noisy losses. WS2's edge is `Loss(L1) − Loss(O)` — **no `min`, so no
  optimism bias** — and its clustered CI is read **directly**: a CI lower bound **> 0** is
  genuine ordered selection structure beyond the previous pitch; a value ≈ 0 (or slightly
  negative) reads as *no ordered selection edge beyond L1*, never as "order hurts". SPEC §13's
  honest expectation is a **small** edge on real data; the null-world positive control shows the
  edge is real and significant when a planted order-2 habit exists.
- **`B_seq` magnitude (SPEC §10).** A positive `B_seq` means ordered history makes the next
  pitch **more forecastable** from pre-release info — this is *finding #1* (selection
  structure), **not** proof that sequencing helps the batter's *outcome* (finding #2) or that
  changing the sequence would help (finding #3). Read it as bits of next-pitch predictability,
  broken out by count/pitch-number; expect more bits in deeper counts (more history to use).
- **The effective-order distribution.** The grammar depth actually *earned* (a context's own
  counts win over the backoff prior, `ω = N/(N+α) ≥ τ`). On real data expect most mass at order
  ≤ 1–2 and a thinning tail beyond — deep ordered contexts rarely clear their support threshold,
  which is the honest "how much order the data can resolve" statement. A materially non-zero
  ≥ 2 mass is real, resolvable order structure.
- **Motif sanity.** The top motifs (ranked by KL of the ordered context's next-token
  distribution vs its suffix) are the grammar's "rules". Sanity-check they are baseball-
  plausible — e.g. same-family-run suppression, or fastball-setup effects (a fastball raising
  the next-pitch probability of an offspeed/breaking family) if present. The synthetic-null
  motifs are all no-three-in-a-row suppression by construction; real motifs should look like
  real sequencing tendencies.
- **The `L1` / `O`-vs-references comparison.** `L1` and `O` should be **≤** the
  `pitcher_count_prev` reference — the serious hierarchical grammar should not lose to the quick
  prev-pitch count baseline. `C` vs `pitcher_count` may tie within noise; on the synthetic
  worlds `C` loses ~0.006 to `pitcher_count` because `stand`/`p_throws` are uninformative there
  (the same world-specific artifact WS1 documents), and would win on real data where handedness
  matters. `O` beating all four references is the headline: the grammar is the best next-pitch
  selector on the board.

---

## Step WS3 — GBDT behavior + decomposed outcome stack (the centerpiece)

**What / why.** WS3 (SPEC §12.3) is the project's centerpiece and its **heaviest CPU step so
far**. It fits, per state view — and it is the first workstream to run **all five** views
`C / U / L1 / O / OM` — a LightGBM **behavior** model `mu(a|s)` (next-pitch family) and a
**decomposed outcome** stack (decision D32): stage A `P(outcome1|s,a)`, stage B
`P(outcome2|s,a)` on in-play rows, stage C count-conditional node run-values, assembled into
`E[R|s,a]`, plus a direct `E[R|s,a]` regressor cross-check. It answers the project's central
question — *does ordered history (`O`) beat `U`/`L1` on **outcomes**, not just selection?* —
via the SPEC §6 `Delta_order`/`Delta_matchup` ablation with clustered CIs on **three** targets
(selection log loss, outcome1 log loss, run-value MAE), and it publishes the artifacts every
prescriptive workstream consumes (decision D33): the counterfactual value grid `qhat(s,a)` for
all 8 families and the behavior propensities `mu(a|s)`, saved with a `load_ws3_artifacts`
loader that **WS4/WS5/WS7 import** (they never refit their own outcome/behavior model).

Because it is heavy, WS3 is **split into stages with per-view checkpointing**: a completed
`(view, stage)` writes its model file(s) and a small `.done` marker, and a re-run skips it
(`--force` rebuilds). Run the three sub-steps below in order; within each, **run the views
sequentially** if RAM is tight (`--views C`, then `--views U`, …) — each view checkpoints
independently, so an interrupted run resumes cheaply.

Hyperparameters are the predeclared LightGBM defaults plus an optional `--tune` 12-combo grid
(`num_leaves × min_child_samples × learning_rate`, identical per view, selected by internal-
holdout validation log loss — decision D34; the holdout is carved from the **train** seasons,
never validation/test). `--threads N` raises LightGBM parallelism for speed (single-thread is
the reproducible default). Everything is runmeta-logged for the SPEC §7 Pareto plot.

Environment note: WS3 needs LightGBM (the `ml` extra). Step 0 already runs
`pip install -e ".[ml]"`; if you see an `ImportError` pointing at the `ml` extra, run that
first. Runs on train (2021–2023); scores on validation (2024) and the **locked test** (2025).

### Step WS3.1 — Behavior stage (`mu(a|s)` per view)

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage behavior --views C U L1 O OM --tune --threads 4
```

Run views one at a time if memory is tight, e.g. `... --stage behavior --views O`. Drop
`--tune` for a single-fit run on the predeclared defaults (markedly faster). The synthetic
Phase-1 CI equivalents (no real data; each ~1–2 min) are
`python workstreams/ws3_gbdt_stack/run_ws3.py --synth null --stage all --out results/ws3_null/`
and `... --synth positive --stage all --out results/ws3_pos/`.

**Expected.** **Estimate: order tens of minutes per view** on a desktop CPU for the full
~2.3M-row train fold (≈100–200 features; the multiclass fit dominates; `--tune` adds a
12-combo grid over a capped 250k-row subsample plus one full refit per view). It writes, under
`results/ws3/`, per view: `behavior_<view>.joblib` (+ `.json` metadata with chosen params and
top gain features), `pred_behavior_real_<view>.parquet` (standard-schema `action_probs` for
val+test), and a `.behavior_real_<view>.done` marker. **Paste back** the last ~10 console lines
(the per-view chosen params and elapsed/peak-RAM) and, for one view, the metadata sidecar:

```powershell
Get-Content results/ws3/behavior_O.json
```

Review checks: the fit should complete for every view and log its chosen params; the O-view
selection log loss (reported in WS3.3) should beat the `pitcher_count_prev` reference.

### Step WS3.2 — Outcome stage (decomposed `E[R|s,a]` per view)

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage outcome --views C U L1 O OM --tune --threads 4
```

**Expected.** **Estimate: order tens of minutes per view** (three LightGBM models per view —
stage A over all rows, stage B over the ~15% in-play rows, and the direct regressor — plus the
count-conditional node-value pass). Writes, per view: `outcome_<view>.joblib` (+ `.json` with
the node-value lookups, chosen params, decomposed-vs-direct disagreement and top gain
features), `pred_outcome_real_<view>.parquet` (standard-schema `outcome1`/`outcome2` probs +
assembled `exp_reward`/`exp_reward_sd`), and a `.outcome_real_<view>.done` marker. **Paste
back** the last ~10 console lines and, for one view, `Get-Content results/ws3/outcome_O.json`.

Review checks: the **decomposed-vs-direct** mean-|difference| (in each `outcome_<view>.json`)
should be small (well under the 0.03 flag on the |R|~0.05–0.3 reward scale); a flagged view
means the event-tree decomposition and the direct regressor disagree materially — investigate
the node-value lookups.

### Step WS3.3 — Assemble + eval (q̂ grid, propensities, the central table)

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage assemble --views C U L1 O OM --threads 4
python workstreams/ws3_gbdt_stack/run_ws3.py --table data/processed/decision_table.parquet --out results/ws3/ --stage eval --views C U L1 O OM
```

(Or run the whole thing in one shot with `--stage all`; the completed WS3.1/WS3.2 views are
skipped via their `.done` markers.) Lower `--n-boot` (default 200) if the clustered-bootstrap
CIs are slow on the full val+test seasons — it changes only the CI widths.

**Expected.** **Estimate: order 20–60 minutes.** `assemble` sweeps the 8-family q-grid and the
propensities across five views on the ~1.5M val+test rows (predict-heavy, no training) and
writes `qgrid_real_<view>.parquet` + `propensity_real_<view>.parquet` per view; `eval` scores
everything through the shared harness (per-view selection/outcome log loss, run-value MAE, the
three `Delta` blocks with clustered CIs) and writes `ws3_report_real.json` +
`ws3_real.runmeta.json`. The falsification battery runs on the synthetic worlds only (skipped
on real data). It ends by printing the **central table** headline (numbers below are the
synthetic-**positive** demo — a real run has lower losses and a real, likely small,
`Delta_order`):

```
==============================================================================
 WS3 GBDT behavior + decomposed outcome stack - headline
==============================================================================
 world / stage  : positive / all
 rows           : train=13,930  val=4,618  test=4,564
 CENTRAL TABLE (validation; log loss / MAE, lower is better):
   view   sel_ll  out1_ll  out2_ll   rv_mae
   C      1.3586   1.5794   1.0339   0.0926
   U      1.3521   1.5667   1.0392   0.0922
   L1     1.3623   1.5871   1.0390   0.0927
   O      1.3505   1.5417   1.0464   0.0921
   OM     1.3529   1.5478   1.0755   0.0914
   selection references: global_count_hand=1.7860 pitcher_count=1.3533 transition=1.6967 pitcher_count_prev=1.3736
 Delta_order (min[U,L1]-O), clustered CI:
   selection : +0.0016  CI[-0.0082, +0.0068]  not significantly positive: no evidence order helps beyond U/L1 (D21)
   outcome1  : +0.0250  CI[+0.0189, +0.0303]  significantly positive: ordered history helps out-of-sample
   run-value : +0.0001  CI[-0.0003, +0.0006]  (MAE; O lower error when >0)
 Delta_matchup (O-OM), clustered CI:
   selection : -0.0023  CI[-0.0082, +0.0041]
   outcome1  : -0.0061  CI[-0.0119, -0.0009]
   run-value : +0.0007  CI[+0.0002, +0.0012]
 LOCKED TEST outcome1 Delta_order: +0.0369  CI[+0.0278, +0.0460]
 decomposed vs direct (D32) mean|dec-direct|: C=0.0184 U=0.0199 L1=0.0192 O=0.0199 OM=0.0224  (all within tolerance)
 top gain features (C view): strikes=4166, pitcher_pitch_count=3201, balls=2536, batter_tend_chase_rate=2533
 --- D35 falsification (synthetic) ---
   order_ablation (outcome1) losses: C=1.5647 U=1.5545 L1=1.5680 O=1.5225
   delta_order=+0.0320  CI[+0.0258, +0.0397]  perm p=0.040  fired=True
   mechanism ablation (loss increase): velo_diff=0.0163 family_slots=0.0075 location=0.0013 outcome_history=0.0004
   mechanism top group: velo_diff
   recovered whiff-lift=0.2894  planted(empirical)=0.3067  recovery ratio=0.943
   (contrast: WS1 family-proxy attenuation ~0.03; floor 0.3)
 D35 verdict    : MECHANISM_RECOVERED
 elapsed (s)    : ...   peak mem (MB) : ...
 outputs        : results/ws3/ws3_report_real.json
==============================================================================
```

**Paste back.** Two things:

1. the entire printed **headline block**, and
2. the per-view decomposed-vs-direct + node-value summary from the report:

```powershell
python -c "import json;r=json.load(open('results/ws3/ws3_report_real.json'));print(json.dumps(r['disagreement'],indent=2))"
```

Review checks (what I look at):

- **`Delta_order` read per D21, on each target.** `Delta_order = Loss(min[U,L1]) − Loss(O)` is
  negatively biased under the null (min of two noisy losses), so the criterion is *significantly
  positive* (clustered CI lower bound > 0), and a small **negative** value reads as *consistent
  with no ordering effect*, never "order hurts". The **outcome1** `Delta_order` is the project's
  central number: a CI above 0 is genuine out-of-sample *sequencing value* (finding #2); SPEC
  §13's honest expectation is that it is **small** on real data (O barely beating L1). The
  **selection** `Delta_order` is finding #1 (order predicting the next pitch) and is a separate
  claim. Read the **locked-test** outcome1 `Delta_order` as the confirmation on unseen 2025 data.
- **`Delta_matchup` (O vs OM).** WS3 is the first workstream that can fit `OM`, so this is the
  first real read on longer-term batter–pitcher adaptation; expect it small.
- **Decomposed-vs-direct agreement (D32).** The `disagreement` block should show small
  mean-|difference| per view (unflagged); it is the internal check that the interpretable event-
  tree assembly matches a black-box regressor.
- **Feature-importance sanity.** In the `feature_gain` block, **count and pitcher features should
  dominate `C`** (in the demo: `strikes`, `pitcher_pitch_count`, `balls`); the history views add
  the slot/transition features on top.
- **Calibration.** The run-value calibration slope (in each view's `exp_reward.calibration`)
  should be near 1 and the intercept near 0.
- **WS1 / WS2 cross-reference.** Compare WS3's per-view **selection** log loss to WS2's grammar
  and WS1's tables (their own RUNBOOK steps): the GBDT should be at least competitive on
  selection, and its decomposed outcome model is what earlier tabular workstreams could not
  provide. On the **positive** synthetic world WS3 recovers the planted whiff-lift almost
  directly (recovery ratio ≈ 0.9 in the demo) because the `O` view carries the ordered velo
  transition — the **contrast exhibit** vs WS1's ~0.03 family-proxy attenuation (decision D35).

---

## Step WS4.1 — Bayesian contextual bandit (myopic prescription, OPE-gated)

**What / why.** WS4 (SPEC §12.5) is the **first prescriptive rung** (Phase B). It turns WS3's
decomposed outcome model into an uncertainty-aware **"best next pitch" target policy** and
evaluates it **offline** through the OPE gate (`eval/ope`) — prescription, never trusted
without OPE (SPEC §0.3). It **consumes WS3's saved artifacts** and never re-fits an outcome or
behavior model (decision D33): a Thompson target policy `π̃(a|s)` is built per state view from
WS3's counterfactual q̂ grid and its (scaled) uncertainty (decision D36), softened toward the
behavior policy across the SPEC §9 α grid (`π_α = (1−α)μ + α π̃`), and scored **strictly**
through `eval/ope.evaluate_policy` (decision D37). The core exhibit is the **D38 prescriptive
ablation**: the value of the policy built from the `C`, `L1` and `O` views — the `C → O` value
gap is the *sequencing-prescription* evidence, isolated from the (count-driven) raw
value-vs-behavior gain.

> **Dependency (explicit).** WS4.1 requires the WS3 artifacts from **Step WS3** to already
> exist under `results/ws3/` (the per-view `behavior_<view>.joblib` and `outcome_<view>.joblib`
> the D33 `load_ws3_artifacts` loader reads). Run Step WS3.1–WS3.2 first (the behavior +
> outcome stages; the `assemble`/`eval` stages are not required by WS4, only the fitted
> `.joblib` models). WS4 trains nothing on real data.

Runs on the same held-out rows WS3 scored — validation (2024) + the locked test (2025) — with
the behavior μ and q̂ read from WS3's train-fold (2021–2023) models.

**Commands.**

```powershell
conda activate statcast; cd ~\pitch-sequencing-research
python workstreams/ws4_bandit/run_ws4.py --table data/processed/decision_table.parquet --ws3-dir results/ws3/ --out results/ws4/ --views C L1 O
```

The synthetic Phase-1 CI equivalents (no real data, no WS3 dependency — WS4 fits small WS3
stacks itself and checkpoints them under `--out`) are
`python workstreams/ws4_bandit/run_ws4.py --synth null --out results/ws4_null/` and
`... --synth positive --out results/ws4_pos/`. Lower `--gap-boot` / `--n-boot` (defaults 400 /
300) if the clustered bootstraps are slow — it changes only the CI widths. Add `--no-fqe` to
skip the (verdict-excluded, D24) count-conditional FQE cross-check for a faster run.
`--posterior-scale` (default 0.05) is the single Thompson-confidence knob (see the model
docstring); `--n-samples` (default 1500) is the Monte-Carlo draw count.

**Expected.** **Estimate: order 10–40 minutes** on a desktop CPU (it is an estimate — WS4
**loads** WS3's models and does no training). The cost is the 8-family q̂/σ predict sweep and
the Thompson Monte-Carlo over the ~1.5M val+test rows for three views, plus the OPE estimator
bootstraps for five α per view and the clustered gap bootstraps; the optional FQE cross-check
adds a handful of `HistGradientBoosting` fits over a coarse count state (`--no-fqe` removes
them). Writes, under `results/ws4/`:

- `policy_real_<view>.parquet` — standard-schema `policy_prob` predictions per view,
- `frontier_real.csv` — the value-vs-α frontier overlay data (value, lower_95, ESS, support),
- `ws4_report_real.json` — the full report (gate, per-view/per-α OPE table, `C→O`/`L1→C` gaps
  with clustered CIs, deviation maps, ambiguity stats, verdict),
- `ws4_real.runmeta.json` — timing / peak RAM.

It ends by printing a headline block like (numbers are the synthetic-**positive** demo; a real
run has different values and, at full-data scale, potentially a resolvable `C→O` gap):

```
==================================================================================
 WS4 Bayesian contextual bandit - myopic prescription (OPE gate) - headline
==================================================================================
 world          : positive
 rows           : train=...  eval(scored)=...
 feasibility    : mean #feasible/row=...  empty-mask(no-rec)=...%  low-history=...%
 behavior recovery: PASS  (observed=...  IPS weights unit=True)  [gate; SPEC 0.3 / D37]
 behavior value V(mu) = ...   (alpha=0 baseline)

 D38 PRESCRIPTIVE-ABLATION TABLE (per view, per alpha; common evaluator = O):
   view alpha     value   lower95 d(vs beh)   ESS%    oos      verdict
   C     0.00   ...       ...       ...      100.0%   ...  INCONCLUSIVE
   ...  (L1 / O × 0.10 / 0.25 / 0.50 / 1.00)
   (D39: INCONCLUSIVE and lower95<V(mu) are first-class honest results, not failures.)

 SEQUENCING-PRESCRIPTION GAPS (clustered by pitcher-game; the O-vs-C isolation):
   alpha=1.00  O-C = ...  CI[..,..]  lower95=..
 ...
 --- D38 SYNTHETIC VERDICT ---
 prescriptive verdict : SEQ_INCONCLUSIVE_MYOPIC   (or SEQ_NEUTRAL_PRESCRIPTION on null)
==================================================================================
```

**Paste back.** Two things:

1. the entire printed **headline block**, and
2. the frontier overlay CSV:

```powershell
Get-Content results/ws4/frontier_real.csv
```

**Review checks (what I look at):**

- **Behavior-recovery PASS is the gate (SPEC §0.3 / D37).** The first line after `rows` must
  read `behavior recovery: PASS`. If it prints **`FAILED_GATE`**, the OPE harness cannot even
  recover the *observed* policy's value on this data — **stop and paste the FAILED_GATE block**;
  no target value below it can be trusted. (`IPS weights unit=True` confirms `π_0 = μ` gave
  exactly unit importance weights.)
- **The prescriptive-ablation reading (this is the point).** The sequencing-prescription
  evidence is the **`C → O` gap** in the "SEQUENCING-PRESCRIPTION GAPS" block, **not** the raw
  value-vs-behavior column. `SEQ_EXPLOITED` is emitted **only** when the real-reward-anchored
  `O − C` gap's one-sided 95% lower bound clears 0 at some α (the pipeline never fabricates it).
  Read the `d(vs beh)` column separately: the bandit often beats (or trails) the *habit-based*
  behavior policy for **count-driven** reasons that are **not** sequencing — the `C → O` gap is
  what isolates sequencing (D38).
- **D39 honest-negatives are results, not failures.** An `INCONCLUSIVE` per-α verdict (DM/SNIPS/
  DR disagree), a `lower95` **below** `V(mu)`, or a `SEQ_INCONCLUSIVE_MYOPIC` / `SEQ_NEUTRAL_
  PRESCRIPTION` overall verdict are **first-class outcomes with their own interpretation**, per
  SPEC §9's closing rule: *"If estimators disagree materially, the verdict is `INCONCLUSIVE`,
  not 'it works.'"* On the synthetic **positive** world the honest myopic verdict is expected to
  be **`SEQ_INCONCLUSIVE_MYOPIC`**: the planted effect is a velo-transition *state*-value effect,
  and a **myopic** policy can only exploit its small family-differential component (~0.003 run,
  below the OPE noise floor at synthetic scale). The effect is **present** (WS3 recovered it) but
  **not myopically prescriptive** — this is precisely the motivation for the *sequential*
  workstreams **WS5** (tabular MDP — values a setup pitch) and **WS7** (offline RL). On real
  data, whether the `C → O` gap resolves above 0 at full-data scale is the open question WS4
  poses and WS5/WS7 answer.
- **Support / ESS to eyeball.** `ESS%` must fall as α rises (100% at α=0, lower at α=1) — the
  price of deviating from behavior; a collapse to a few % at moderate α means the target is far
  outside behavior support (read its value with suspicion). `oos` (out-of-support fraction: mean
  target mass on actions with μ < 1%) should stay small; a large value means the policy
  recommends rarely-thrown actions the OPE cannot evaluate. The `empty-mask(no-rec)` fraction is
  the share of low-history decisions the target leaves at the observed action (no
  recommendation) — informational, higher early in a season.
- **Ambiguity.** `mean P(top>runner-up)` near 0.5 and high `ambiguous @80/@95` shares mean the
  recommendations are rarely confident distinctions (small q̂ gaps vs the posterior uncertainty)
  — an honest statement of how resolvable "best next pitch" is, not a bug.
