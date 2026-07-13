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
