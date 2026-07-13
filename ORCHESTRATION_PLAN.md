# Orchestration Plan

Running log for the pitch-sequencing comparative study. One implementation unit is
dispatched at a time, reviewed against SPEC.md, then committed. This file records the
plan, every dispatch, the review verdict, and design decisions made along the way.

## Mission recap

Build the rigor ladder described in SPEC.md and README.md: a shared foundation
(decision table, five nested state views C/U/L1/O/OM, reward, temporal splits,
evaluation harness, OPE) plus seven workstreams, each delivering code, a
research-grade notebook scaffold, PAPER.md, SEAN-README.md, and THEORY.md.

**Phase 0 (now):** the `pitchseq` package per SPEC §2–§11, genuinely running and
passing its correctness oracle offline on CPU against synthetic fixtures and the
committed 20k sample.

**Phase 1 (now):** all seven workstreams scaffolded and code-complete against
sample/fixtures — unit-tested where feasible, but **no full-scale training and no
notebook execution** here (the full data is not in this environment).

**Phase 2 (later):** real runs on the full data — **on Sean's desktop, because that
is where `statcast.db` lives** (data locality, not hardware, is the gate; nearly
everything is CPU work). The orchestrator plans each run and hands Sean exact,
copy-pasteable PowerShell steps; Sean runs them locally and pastes back the printed
headline numbers; the orchestrator reviews against SPEC acceptance checks and then
has the subagent fill real numbers + the chosen interpretation branch into the
pre-written papers/notebooks/READMEs.

## Phase-2 execution contract (locked)

- Sean's machine: Windows desktop, PowerShell, conda env `statcast`, repo at
  `~\pitch-sequencing-research`, data at `data/raw/statcast.db` and
  `data/raw/statcast_*.parquet`, AMD GPU, free Colab available.
- Every runnable step handed to Sean includes: (a) one line on what/why, (b) exact
  command(s) prefixed `conda activate statcast; cd ~\pitch-sequencing-research`,
  (c) rough expected runtime + what it outputs, (d) exactly what to paste back.
- All heavy scripts must be **resumable and checkpointed**, write outputs to
  `results/` or `artifacts/` (gitignored), and print headline numbers (log loss,
  Δ_order, Δ_matchup, OPE value + CI, ESS, …) plus wall-clock and peak RAM — the
  raw material for the SPEC §7 performance-vs-compute Pareto plot.
- CPU is the default everywhere. WS1–WS5 and WS7 are CPU steps on the full 3.85M
  pitches. Only WS6 is GPU-optional: ship both a CPU/local path and a
  self-contained free-Colab (T4) path; note the ROCm/DirectML caveat for the AMD
  GPU and recommend Colab as the simpler route. Only WS6's optional Transformer
  variant is genuinely GPU-preferred.
- Steps are kept small so Sean can run a few at a time and re-run failures cheaply.
- **RUNBOOK.md** at repo root accumulates the exact Phase-2 steps per workstream,
  in build order, as each workstream unit lands. It is the single place Sean works
  from in Phase 2.

## Hard constraints

- Temporal splits only; leakage is fatal; OPE before policy; falsifiable by
  construction; report the honest finding (SPEC §0).
- No full-scale training or notebook execution in this phase (no GPU; full data
  absent — `data/raw/statcast.db` exists only on the local machine).
- Workstreams plug into the shared harness; none re-implements evaluation.
- Plain commit messages, author Sean Salvador. Big data and model blobs stay out
  of git; only the 20k sample and tiny fixtures are committed.

## Build order

WS0 foundation (3 units) → WS1 EB tables → WS2 Bayes Markov → WS3 GBDT stack
(centerpiece) → OPE self-tests already in WS0 → WS4 bandit → WS5 tabular MDP →
WS6 GRU → WS7 offline RL capstone.

WS0 units:

- **0a — data layer:** pyproject + `io / families / reward / outcomes / rolling /
  decision_table / states / sequences / splits` + unit tests on the 20k sample and
  tiny synthetic inputs.
- **0b — fixtures + evaluation:** `synth.py` null/positive worlds, `eval/predictions`,
  `eval/metrics`, `eval/falsification`, `eval/predictability`, `eval/harness` + the
  null/positive acceptance tests of SPEC §11.
- **0c — OPE:** `eval/ope.py` (DM/IPS/SNIPS/DR/stepwise-DR/FQE + diagnostics),
  logged-bandit fixture, behavior-policy-recovery and known-value self-tests,
  leakage-guard audit test; whole suite green.

Each Phase-1 workstream then ships as one reviewed unit (WS3 may split into two:
code+tests, then docs+notebook).

## Locked design decisions

| # | decision | rationale |
|---|---|---|
| D1 | `configs/default.yaml` is canonical; duplicate root `default.yaml` removed | two copies of a contract drift |
| D2 | `.gitignore` covers `CLAUDE.md`, egg-info, pytest cache, model/artifact dirs; sample parquet + SCHEMA.csv explicitly un-ignored | keep the repo clean; the two committed data files stay visible to fresh clones |
| D3 | The 20k sample is a **random row sample** (19,050 PAs / 20,000 pitches): within-PA history is mostly absent. Sequence logic is validated on synthetic fixtures; the sample validates schema conformance, dtypes, family mapping, reward sign, and end-to-end runnability | discovered on inspection; affects what each test can honestly claim |
| D4 | Reward sign confirmed on sample: `delta_run_exp` < 0 on strikes/fouls, > 0 on balls, so `R = -delta_run_exp` is pitcher-positive as SPEC §5 assumes | verified empirically before building |
| D5 | Outcome event tree fixed at two levels. Level 1 (pitch result): `ball / called_strike / whiff / foul / hit_by_pitch / in_play` from `description`. Level 2 (conditional on `in_play`): `out / single / double / triple / home_run` from `events`. Node-level and final-event losses per SPEC §8.2 | SPEC names the event tree but leaves node granularity open; this is the minimal tree covering §8.2 |
| D6 | Batter-relative location: `plate_x_br = plate_x` for RHB, `−plate_x` for LHB (positive = outer half for either hand); `plate_z_norm = (plate_z − sz_bot)/(sz_top − sz_bot)` | removes handedness ambiguity in SPEC §3.3 |
| D7 | `row_id = "{game_pk}_{at_bat_number}_{pitch_number}"` (string) is the prediction-table key | deterministic, human-readable join key |
| D8 | Subagents never commit; the orchestrator reviews then commits | single point of review |
| D9 | PyTorch install deferred until WS6 | disk allowance; nothing earlier needs it |
| D10 | `base_state` bit-encodes occupancy: `on_1b→1, on_2b→2, on_3b→4` (0–7) | matches SPEC §3.2 derivation |
| D11 | Tabular **O** view = C + last-3 ordered pitch-token slots + consecutive-diff and run-length features; deep/Markov **O** = full ordered sequence via `sequences.py`. Same information set, representation differs by model family | SPEC §6 defines the information ladder, not the encoding |
| D12 | Every workstream ships CLI entry points designed for Sean's Windows box: argparse, `pathlib` paths, `if __name__ == "__main__"` guards (Windows spawn), checkpoint/resume, outputs under `results/`/`artifacts/`, headline numbers + wall-clock + peak RAM printed at the end | Phase-2 execution contract |
| D13 | Shared `pitchseq/runmeta.py` utility (psutil-based) records wall-clock, peak RSS, and run metadata to JSON next to each output — feeds the §7 Pareto plot | one implementation, used by every runnable step |
| D14 | `results/` added to `.gitignore`; only tiny summary JSON/CSVs may be committed deliberately after review | keep heavy artifacts out of git |
| D15 | WS0 gains a runnable full-data build step: `python -m pitchseq.build_table` (per-season checkpointed decision-table build, `data/processed/decision_table.parquet`) — first RUNBOOK step of Phase 2 | the decision table is the first thing Sean must build locally |

## Dispatch log

| unit | dispatched | verdict | commit |
|---|---|---|---|
| housekeeping (this file, .gitignore, dedupe config) | — (orchestrator) | — | 811db47 |
| Phase-2 execution contract | — (orchestrator) | — | a62b168 |
| 0a data layer | 2026-07-13 | Round 1: high quality, 42 tests green, SPEC-conformant. Review found (1) OM cross-matchup contamination — groupwise cumsum followed by a global `.shift(1)` leaks the previous matchup's totals into the first PA of the next (confirmed by probe); (2) `episode_returns` agreement guard falsely fires when a PA's last pitch has missing `delta_run_exp` (would break on full data); (3) `prev_pitch_type` categorical vocabulary was data-dependent. Sent back for targeted fixes + regression tests. | pending fixes |
| 0b fixtures + eval | | | |
| 0c OPE | | | |
| WS1 | | | |
| WS2 | | | |
| WS3 | | | |
| WS4 | | | |
| WS5 | | | |
| WS6 | | | |
| WS7 | | | |
