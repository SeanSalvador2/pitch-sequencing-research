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
