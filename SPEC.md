# Workstream 0 — Shared Foundation Spec

This is the design contract for the pitch-sequencing comparative study. Every one of
the seven modeling workstreams plugs into the shared pieces defined here: one canonical
**decision table**, the five nested **state views**, a fixed **action space** and
**reward**, a locked **temporal split**, one **evaluation harness**, and one **OPE**
interface. Build this before any model.

The guiding idea is a *rigor ladder*, not a horse race:

> How much evidence for sequencing survives progressively harder tests — from
> descriptive order patterns, to out-of-sample **outcome** dependence, to
> **counterfactual policy** value?

Three findings are routinely conflated and must be kept separate:

1. **Selection structure** — prior pitches help predict *what is thrown next*.
2. **Predictive sequencing value** — prior pitches help predict the *outcome* of the
   current pitch, after conditioning on the current pitch and game state.
3. **Prescriptive/causal value** — *changing* the sequence would improve outcomes.

The first is easy; the second is hard; the third needs assumptions that public data
cannot fully satisfy (we never see the pitch that wasn't thrown; `pitch_type` is a
classifier output, not the battery's intent; scouting/target info is unobserved).

---

## 0. Non-negotiable disciplines

1. **Leakage is fatal.** Every feature for the decision before pitch *t* must be
   computable strictly **before pitch *t* is released**. In particular, the current
   pitch's own physical measurements (velocity, movement, location, spin of pitch *t*)
   are **execution**, downstream of the decision — never state features. Rolling
   pitcher/batter features use only data dated **before the current game**.
2. **Temporal splits only.** Never random-row splits — pitches in the same PA/game/
   pitcher/season are highly dependent. Cluster confidence intervals by pitcher-game.
3. **OPE before policy.** Build and self-test the off-policy-evaluation harness (it must
   recover the observed policy's value) *before* optimizing any policy. Otherwise you
   produce recommendations you cannot evaluate.
4. **Falsifiable by construction.** Every workstream ships a runnable correctness check
   against synthetic ground truth (a null world with no order effect, a positive world
   with a planted effect). A model that "runs" has proven nothing.
5. **Report the honest finding.** A modest or null ordering effect (O barely beating L1)
   is a real, publishable result, consistent with the motif literature. Build to report
   it either way.

---

## 1. Data reality

- Source: Statcast via pybaseball, seasons **2021–2025**, all one tracking era
  (Hawk-Eye) — no cross-sensor drift to correct. `data/raw/statcast.db` (table
  `statcast`) and per-season parquet. ~3.85M pitches, 119 columns.
- `delta_run_exp` populated on ~99.7% of pitches — the reward signal.
- `pitch_type` is a Statcast classifier output: treat it as an observed **proxy** for
  the pitch thrown, not ground-truth intent.
- Optional pre-2020 extension (PITCHf/x/TrackMan eras) may be added **for descriptive /
  time-series analysis only** — never pooled into outcome models, due to measurement
  shift.

---

## 2. Repository structure

```
pitch-sequencing-research/
  SPEC.md                     # this file — the contract
  README.md
  requirements.txt            # data-pull deps (existing)
  pull_statcast.py            # data pull (existing)
  00_pull_statcast.ipynb      # data pull notebook (existing)
  pyproject.toml              # package `pitchseq` (added)
  configs/
    default.yaml              # pitch-family map, splits, windows, thresholds
  data/                       # gitignored (local only)
    raw/                      # statcast.db, statcast_*.parquet
    processed/                # decision table, sequence tensors (built artifacts)
  src/pitchseq/
    io.py                     # load raw statcast (db or parquet)
    families.py               # pitch_type -> family map, feasible-action masks
    reward.py                 # R = -delta_run_exp, sign check
    decision_table.py         # build the canonical per-decision table (WS0)
    states.py                 # C / U / L1 / O / OM feature builders (WS0)
    sequences.py              # variable-length history tensors for seq models
    splits.py                 # temporal split + clustered resampling
    rolling.py                # leakage-safe rolling repertoire / tendencies
    outcomes.py               # event-tree labels
    synth.py                  # synthetic null/positive worlds + logged-bandit fixture
    eval/
      predictions.py          # the standard prediction-output schema (contract)
      metrics.py              # log loss, Brier, calibration, top-k, ablation deltas
      falsification.py        # permutation, pseudo-history, semi-synthetic tests
      ope.py                  # DM / IPS / SNIPS / DR / stepwise-DR / FQE + diagnostics
      predictability.py       # bits-of-predictability, exploitability read-out
      harness.py              # run a model's outputs through the whole eval battery
  workstreams/
    ws1_eb_tables/            # empirical-Bayes conditional tables
    ws2_bayes_markov/         # Bayesian variable-order Markov
    ws3_gbdt_stack/           # GBDT behavior + decomposed outcome (CENTERPIECE)
    ws4_bandit/               # Bayesian contextual bandit (myopic prescription)
    ws5_tabular_mdp/          # controlled Markov reward process
    ws6_deep_seq/             # GRU (+ optional compact Transformer demo)
    ws7_offline_rl/           # conservative offline RL + OPE + exploitability
  tests/                      # correctness checks against synth fixtures
  notebooks/                  # exploratory + per-workstream demos
```

Each `workstreams/wsN_*/` folder reads the shared decision table + state builders,
writes predictions in the standard schema, and is scored by the shared harness. No
workstream re-implements evaluation.

---

## 3. The canonical decision table

**Grain:** one row per pitch = one decision point (the choice made immediately *before*
pitch *t*). Natural key: `(game_pk, at_bat_number, pitch_number)`.

**Columns** (leakage-safe; all knowable before pitch *t*):

### 3.1 Identity / keys
| field | source | notes |
|---|---|---|
| `game_pk, game_date, at_bat_number, pitch_number` | raw | key + ordering |
| `pitcher, batter` | raw | ids for pooling/embeddings |
| `season` | derived | from game_date |
| `pa_id` | derived | `game_pk * 1000 + at_bat_number` (episode id) |
| `is_seq_eligible` | derived | `pitch_number >= 2` |

### 3.2 Non-sequence context `X_t`  (this is the **C** view)
| field | source | notes |
|---|---|---|
| `balls, strikes` | raw | the count |
| `outs_when_up` | raw | |
| `base_state` | derived | 0–7 from `on_1b/on_2b/on_3b` occupancy |
| `inning, inning_topbot` | raw | |
| `score_diff_pitcher` | derived | `fld_score - bat_score` (pitcher's perspective) |
| `stand, p_throws` | raw | handedness; also `platoon = (stand==p_throws)` |
| `pitcher_pitch_count` | derived | cumulative pitches by this pitcher in this game, pre-*t* |
| `times_thru_order` | derived | batter's appearance index vs this pitcher, pre-*t* |
| `repertoire_mix_*` | rolling | trailing pre-game usage rate per family (8 cols) |
| `batter_tend_*` | rolling | trailing pre-game batter response tendencies |
| `park` | derived | home_team as proxy |

Catcher/umpire effects are not in base Statcast — omit (note as a known limitation).

### 3.3 Within-PA history `H^PA_t`  (adds **U / L1 / O**)
For each prior pitch *1..t-1* in the current PA, retain a token:
- `family`, exact `pitch_type`
- **batter-relative** location: `plate_x_br` (sign-flip for LHB), `plate_z_norm`
  (normalize by `sz_top/sz_bot`)
- `release_speed`, movement (`pfx_x, pfx_z`), `release_spin_rate`, release pos, extension
- outcome token: take / called-strike / ball / whiff / foul / in-play
- diffs from that pitcher's rolling family baseline
- pairwise diffs vs the immediately preceding pitch (velo gap, movement gap, release
  gap, plate-location contrast) — the raw material for tunneling features

### 3.4 Matchup memory `H^M_t`  (adds **OM**)
Separated buckets, each leakage-safe (strictly prior):
- earlier PAs **this game**
- earlier meetings **this season**
- older meetings
- general batter-vs-family, general pitcher-vs-handedness
Do not build unpooled exact matchup tables — an average pitcher-batter pair has ~20
career pitches. Use partial pooling / embeddings.

### 3.5 Execution `Z_t` and outcome `Y_t` (NOT state — labels/for outcome model only)
- `Z_t`: realized `release_speed, pfx_x, pfx_z, plate_x, plate_z, ...` of pitch *t*.
  Used **only** in the outcome model conditioned on execution, never as a pre-pitch feature.
- `Y_t`: `description, type, events, launch_speed, launch_angle,
  estimated_woba_using_speedangle` → event-tree labels (§6).
- `R_t`: reward (§5).

---

## 4. Action space

**Primary action = pitch family** (8 classes), mapped from `pitch_type`:

| family | pitch_type codes |
|---|---|
| `FF` four-seam | FF |
| `SI` sinker | SI, FT |
| `FC` cutter | FC |
| `SL` slider/sweeper | SL, ST, SV |
| `CU` curve | CU, KC, CS |
| `CH` change | CH |
| `FS` split | FS, FO |
| `XX` other/rare | KN, EP, SC, FA, PO, null, … (descriptive only; excluded from recommendations) |

Family is primary because it reduces classifier drift, keeps enough support for policy
evaluation, allows cross-pitcher comparison, and bounds the RL action space. Keep exact
`pitch_type` as a secondary benchmark.

**Feasible-action mask** (per pitcher, time-varying, leakage-safe): a family is feasible
only if the pitcher threw it **≥ 50 times AND ≥ 3%** over a trailing window ending
before the current game. Never use future-season repertoire.

---

## 5. Reward

$$R_t = -\,\texttt{delta\_run\_exp}_t$$

so **larger is better for the pitcher**. `delta_run_exp` is the change in run expectancy
across the pitch. **Sign check (mandatory before use):** verify empirically that $R_t$ is
positive on called strikes / swinging strikes / outs and negative on walks / home runs;
fail loudly if the sign is inverted. Episode return for a PA: $G = \sum_t R_t$; a terminal
reward equal to that sum must agree — disagreement is an indexing bug.

As a sensitivity analysis, also recompute a **training-fold-only** run-expectancy table
rather than trusting the supplied metric entirely.

---

## 6. The five nested state views (the measurement instrument)

Every suitable model is trained on the **same five state variants**. The differences
between them *are* the sequencing evidence.

| view | information | question it isolates |
|---|---|---|
| **C** | `X_t` only (context + rolling repertoire/tendencies) | selection without current-PA sequencing |
| **U** | C + **unordered** counts/means of prior PA pitches | does the *collection* of prior pitches matter? |
| **L1** | C + the **immediately preceding** pitch | simple previous-pitch dependence |
| **O** | C + the **full ordered** current-PA sequence | does true within-PA *order* matter? |
| **OM** | O + matchup memory `H^M` | longer-term batter–pitcher adaptation |

Headline comparisons:

$$\Delta_{\text{order}} = \text{Loss}(\min[U, L1]) - \text{Loss}(O)$$
$$\Delta_{\text{matchup}} = \text{Loss}(O) - \text{Loss}(OM)$$

Interpretation: O beating C but **not** U/L1 ⇒ history matters but *order* barely does.
O beating both U and L1 ⇒ genuine ordered dependence. Report every sequence metric
separately for: all pitches; sequence-eligible (`t≥2`); longer PAs (`t≥3`); two-strike /
three-ball counts; first pitch of a PA (only matchup memory can matter there).

---

## 7. Temporal split

- **Train:** 2021–2023  ·  **Validation / model selection:** 2024  ·  **Locked test:** 2025.
- Rolling replications: {train 21–22, tune 23, test 24} and {train 21–23, tune 24, test 25}.
- All rolling features use expanding/trailing windows ending before the pitch.
- Report: pitch-weighted overall; equal-weighted pitcher averages; established vs
  low-history/new pitchers; CIs clustered by pitcher-game (pitcher-level resampling as a
  robustness check).
- Predeclare roughly equal hyperparameter budgets; log wall-clock, peak RAM/VRAM, param
  count. Final deliverable includes a **performance-vs-compute Pareto plot**.

---

## 8. The evaluation harness (one contract for every model)

### 8.1 Standard prediction output (`eval/predictions.py`)
Every model writes a table keyed by `row_id` with whatever it produces:
- `action_probs` — distribution over the 8 families (behavior-policy models).
- `outcome_probs` — event-tree node probabilities (outcome models).
- `exp_reward`, `exp_reward_sd` — E[R] and uncertainty (outcome/value models).
- `policy_probs` — distribution over feasible families (prescriptive models).
- `state_view` — which of C/U/L1/O/OM produced this row.
- `model_id`, `seconds`, `peak_mem_mb`, `n_params`.

### 8.2 Metrics (`eval/metrics.py`)
- **Selection/next-pitch:** multiclass **log loss (primary)**, Brier, top-1/top-2
  accuracy (secondary), macro-F1, per-pitcher-macro and micro averages, calibration,
  log-loss skill vs a shrunk pitcher-by-count baseline.
- **Outcome:** node-level + final-event log loss, Brier, reliability curves by
  count/family/handedness, run-value MAE/RMSE, run-value calibration intercept & slope,
  observed-vs-predicted mean reward by action and by propensity decile.
- **Ablation deltas:** $\Delta_{\text{order}}$, $\Delta_{\text{matchup}}$ with clustered CIs.

Baselines every model is scored against: global family freq by count×handedness; shrunk
pitcher-by-count; first-order transition (prev pitch); pitcher×count×prev-pitch.

### 8.3 Sequence falsification (`eval/falsification.py`) — required for any O/OM model
- **Order ablation:** O must beat U and L1, not just C.
- **Stratified history permutation:** permute histories among PAs matched on
  pitcher×count×pitch-number×handedness×prior-pitch-multiset. An order-sensitive edge
  should collapse.
- **Pseudo-history control:** swap in a history from another PA by the same pitcher in a
  similar count — tests whether the model is merely identifying the pitcher.
- **Mechanism ablation:** separately remove pitch type / location / velo-diff / release
  similarity / outcome history.
- **Null semi-synthetic world:** outcomes depend only on context + current pitch; flexible
  models must **not** discover an ordered-history gain.
- **Positive semi-synthetic world:** inject a known previous-pitch effect (e.g. a benefit
  from a large velo differential); methods must recover its sign and rough magnitude.

---

## 9. OPE interface (`eval/ope.py`) — the prescriptive gate

Policies output a **distribution over feasible pitches**, never a deterministic pick. Use
a conservative class that stays near observed behavior:

$$\pi_\alpha(a\mid s) = (1-\alpha)\,\mu(a\mid s) + \alpha\,\widetilde\pi(a\mid s)$$

where $\mu$ is the estimated behavior policy (from WS3) and $\alpha$ controls deviation.

**Estimators:** Direct Method, IPS, self-normalized IPS, Doubly Robust, step-wise DR (for
full PAs), Fitted-Q Evaluation. Cluster-bootstrap CIs.

**Mandatory self-tests before trusting any policy value:**
1. **Behavior-policy recovery** — set target = behavior policy; OPE must return the
   observed held-out value.
2. **Semi-synthetic recovery** — on a logged-bandit fixture with a *known* target-policy
   value, each estimator recovers it within CI (unbiased ones unbiased).
3. Cross-fit propensity and outcome models; repeat with several propensity specs; drop
   low-overlap states and check persistence; placebo outcomes the pitch can't affect;
   perturb reward/transition within uncertainty; compare bandit (myopic) vs RL (full-PA).

**Every policy reports:** estimated run value + 95% lower bound; effective sample size;
fraction of decisions outside adequate support; max & upper-quantile importance ratios;
KL and total-variation distance from behavior; fraction of rare-action recommendations;
agreement among DM/DR/FQE. **If estimators disagree materially, the verdict is
`INCONCLUSIVE`, not "it works."**

---

## 10. Predictability & exploitability read-out (`eval/predictability.py`)

- **Predictability in bits:** train a calibrated "batter-side" next-pitch predictor from
  pre-release info only; report
  $B_{\text{seq}} = \frac{1}{n}\sum_t \log_2 \frac{q_O(A_t\mid S_t)}{q_C(A_t\mid X_t)}$
  — the extra bits about the next pitch supplied by ordered history — broken out by
  pitcher/count/pitch-number/rematch/family. Positive ⇒ forecastable ordering (not, by
  itself, proof it helps the batter).
- **Exploitability (light game-theoretic capstone):** a fixed, interpretable batter
  response model + the common outcome model give a per-state payoff; report each policy's
  average exploitability vs the equilibrium value, with payoff bootstrapped. Model-
  dependent by necessity — report robust/worst-case.
- **Final figure:** a frontier of OPE run value × predictability-in-bits × exploitability
  × distance-from-behavior × compute cost.

---

## 11. Synthetic fixtures & the WS0 correctness oracle (`synth.py`, `tests/`)

The foundation must be testable **without** the 3.85M-row dataset — same philosophy as
the rest of the portfolio.

- **Null world:** generate PAs where the next outcome depends only on context + current
  pitch (no order effect). Acceptance: the harness's $\Delta_{\text{order}}$ is
  statistically indistinguishable from 0; falsification permutation test does not fire.
- **Positive world:** inject a known previous-pitch effect of known sign/magnitude.
  Acceptance: the harness recovers the sign and approximate magnitude; the permutation
  test fires; a sequence model beats U/L1.
- **Logged-bandit fixture:** a small bandit with a known logging policy and an
  analytically computable target-policy value. Acceptance: OPE recovers the logging
  policy's value and the known target value within CI; ESS/weight diagnostics populate.
- **Reward/indexing checks:** sign check on $R$; $\sum_t R_t$ equals the terminal return;
  leakage guard (assert no current-pitch execution column appears in any state builder,
  e.g. via an import/column-name audit).

`tests/` runs all of these on tiny synthetic inputs, offline, fast — this is the gate that
proves the harness is correct before any real model is trusted.

---

## 12. Workstreams (build order) and how each plugs in

Two threads, each internally increasing in complexity; overall the offline-RL capstone is
the hardest thing in the project. Phase A alone (1–4 below on the predictive side + the
ablation) is a complete, honest paper if time runs short.

**Phase A — Descriptive & Predictive (does order matter?)**
1. **WS1 — Empirical-Bayes tables.** Dirichlet-shrunk selection + partial-pooled run-value
   tables. The transparent baseline everything must beat; exposes the support problem.
2. **WS2 — Bayesian variable-order Markov.** Backoff over ordered pitch tokens with
   hierarchical/Dirichlet shrinkage and calibrated uncertainty — the "pitch grammar" test
   (fuses the Markov and Bayesian-pooling ideas).
3. **WS3 — GBDT behavior + decomposed outcome stack (CENTERPIECE).** LightGBM for
   next-pitch *and* the event-tree outcome model, run across all five state views. Answers
   "does ordered history beat U/L1?" **and** supplies the behavior propensities + outcome
   model that WS4/5/7 consume. If O doesn't beat U/L1 here, be skeptical of everything
   fancier.
4. **WS6 — Deep sequence: GRU (+ optional compact Transformer demo).** Does a learned
   representation beat engineered history and the Markov model? Build the GRU/TCN; add a
   *compact* Transformer only as a demo if the GRU shows real signal and you want the long
   cross-PA (OM) test. Compute peak on the predictive side.

**Phase B — Prescriptive & honestly evaluated (what should he throw — and can we prove it?)**
5. **WS4 — Bayesian contextual bandit.** Myopic "best next pitch" as an uncertainty-aware
   *target* policy, evaluated **offline** (not learned online). Entry into prescription;
   reuses WS3's outcome model.
6. **WS5 — Tabular MDP / controlled Markov reward process.** The simplest sequential model
   that can value a *setup* pitch (accounts for future count states) and a transparent
   simulator used to validate the OPE code.
7. **WS7 — Conservative offline RL + OPE + exploitability read-out (CAPSTONE).** FQE/CQL
   for full-PA sequential value, evaluated honestly (behavior-policy recovery, DR/FQE
   agreement, support diagnostics), capped with the predictability/exploitability frontier
   (§10). Deferred to phase 2: model-based RL (highest model-error risk) and a full
   stochastic-game solver.

**Suggested order:** WS0 (this spec) → WS1 → WS2 → WS3 → build OPE self-tests → WS4 → WS5
→ WS6 → WS7. Build OPE before optimizing any policy.

---

## 13. Likely finding (set expectations now)

The most probable honest result is that **O barely beats L1** — full pitch *order* adds
little out-of-sample outcome signal beyond the immediately preceding pitch and context.
That is consistent with the recent motif literature and is a stronger, more credible
portfolio result than a fabricated policy gain. The study is designed so that either
outcome — "order matters" or "order mostly doesn't" — is a clean, defensible finding.
