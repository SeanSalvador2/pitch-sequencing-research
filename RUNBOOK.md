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

_Workstream steps (WS1 empirical-Bayes tables → WS7 offline RL) will be appended below as
each unit is built._
