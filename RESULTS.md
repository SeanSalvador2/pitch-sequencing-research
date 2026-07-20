# Phase-2 Results Log

Verbatim headline blocks from the real-data runs on the full 2021–2025 decision table
(3,567,640 regular-season decisions), as executed on the local desktop and reviewed
against the SPEC acceptance checks. This is the durable record the documentation
fill-in works from; the full machine-readable reports live in the gitignored
`results/` directory on the local machine.

Review verdicts and interpretation-branch selections for each entry are logged in
ORCHESTRATION_PLAN.md ("Phase-2 run log").

---

## Step 1 — decision-table build (2026-07-16)

```
 source         : data/raw/statcast.db
 seasons        : 2021, 2022, 2023, 2024, 2025
 rows / season  : 2021=712,320, 2022=710,210, 2023=720,684, 2024=711,898, 2025=712,528
 total rows     : 3,567,640
 columns        : 81
 sign check     : PASS   (ball=-0.0588  called_strike=+0.0651  home_run=-1.5311
                          strikeout=+0.2195  swinging_strike=+0.1141  walk=-0.2361)
 elapsed (s)    : 150.6      peak mem (MB): 21575.2
```

Notes: total is regular-season only (`game_type == "R"`; the ~3.85M figure counts all
game types). One fail-loud vocabulary stop on first contact (`foul_pitchout`, 1 pitch),
classified and committed (54fde3a) before the clean rebuild.

---

## WS1 — empirical-Bayes tables (2026-07-16)

```
 world / target : real / both
 rows           : train=2,143,214  val=711,898
 selection log loss (val) vs count-based references:
   C  : 1.4714  ref[pitcher_count]=1.4866  (<= ref)
   U  : 1.4785  ref[pitcher_count_prev]=1.4410  (> ref)
   L1 : 1.4772  ref[pitcher_count_prev]=1.4410  (> ref)
   O  : 1.4787  ref[pitcher_count_prev]=1.4410  (> ref)
   references     : global_count_hand=1.7432  pitcher_count=1.4866  transition=1.6437  pitcher_count_prev=1.4410
 run-value MAE (val):
   C  : MAE=0.1200  RMSE=0.2234      U  : MAE=0.1200  RMSE=0.2234
   L1 : MAE=0.1201  RMSE=0.2234      O  : MAE=0.1200  RMSE=0.2234
 Delta_order    : -0.0015  CI[-0.0017, -0.0013]   (read per D21: consistent with no ordering effect)
 Delta_matchup  : N/A - OM tables infeasible for pure tables (D25 support problem)
 support (distinct cells / eval frac n<20 / deepest-level backoff rate):
   C  : cells= 30430  n<20=0.072  deepest-hit=0.880
   U  : cells=320487  n<20=0.332  deepest-hit=0.751
   L1 : cells= 94992  n<20=0.222  deepest-hit=0.848
   O  : cells=212434  n<20=0.348  deepest-hit=0.803
 elapsed (s)    : 323.7     peak mem (MB): 6068.6
```

---

## WS2 — Bayesian variable-order Markov grammar (2026-07-16)

```
 world / K_MAX  : real / 4
 rows           : train=2,143,214  val=711,898
 selection log loss (val) vs count-based references:
   C  : 1.4714  ref[pitcher_count]=1.4866  (<= ref)
   L1 : 1.4811  ref[pitcher_count_prev]=1.4410  (> ref)
   O  : 1.4867  ref[pitcher_count_prev]=1.4410  (> ref)
 delta_order_L1 : -0.0056  CI[-0.0061, -0.0051]  (Loss(L1)-Loss(O); deeper contexts hurt OOS)
 B_seq (bits)   : overall=-0.0221  two_strike=-0.0527  three_ball=+0.0005
   by count-bucket: ahead=-0.0408  behind=+0.0013  even=-0.0234
   by pitch_number: t1=+0.000  t2=-0.010  t3=-0.022  t4=-0.032  t5=-0.055
   (model-relative: negative values reflect depth>1 overfit of this grammar, not anti-predictability)
 effective order: mean=0.901  >=2 mass=0.250  (tau=0.50)
   distribution   : k0=0.376  k1=0.373  k2=0.224  k3=0.026  k4=0.000
 top motifs (ordered context -> next-token lift vs suffix):
   XX -> P(XX) 0.31->0.54 (promote, KL=0.162)
   FS -> P(FS) 0.24->0.38 (promote, KL=0.083)
   CH -> P(CH) 0.21->0.30 (promote, KL=0.049)
   SL -> P(SL) 0.31->0.39 (promote, KL=0.028)
   SI -> P(SI) 0.35->0.41 (promote, KL=0.027)
 D30 detect     : GRAMMAR_NOT_DETECTED  (O<L1=False, >=2 mass=0.250, repeat-motif=True)
 D30 control    : COLLAPSES_UNDER_PERMUTATION  (edge -0.0056 -> -0.0013, >=2 mass -> 0.000)
 elapsed (s)    : 174.9     peak mem (MB): 5718.7
```

Headline reading: real MLB selection is **sticky at depth 1** (uniform repeat-promotion
motifs); depth beyond one pitch adds nothing this grammar can extract (significantly
negative order edge = fragmentation cost, effective order ≈ 0.9).

---

## WS3 — GBDT centerpiece: THE CENTRAL ABLATION (2026-07-16)

```
 rows           : train=2,143,214  val=711,898  test=712,528
 CENTRAL TABLE (validation; log loss / MAE, lower is better):
   view   sel_ll  out1_ll  out2_ll   rv_mae
   C      1.2404   1.4610   0.9237   0.1200
   U      1.2091   1.4584   0.9238   0.1201
   L1     1.2155   1.4585   0.9240   0.1201
   O      1.2002   1.4581   0.9240   0.1201
   OM     1.2000   1.4576   0.9238   0.1200
   selection references: global_count_hand=1.7432  pitcher_count=1.4866  transition=1.6437  pitcher_count_prev=1.4410
 Delta_order (min[U,L1]-O), clustered CI:
   selection : +0.0089  CI[+0.0085, +0.0093]  significantly positive
   outcome1  : +0.0003  CI[+0.0002, +0.0004]  significantly positive
   run-value : +0.0000  CI[-0.0000, +0.0000]  (MAE)
 Delta_matchup (O-OM), clustered CI:
   selection : +0.0002  CI[+0.0000, +0.0004]
   outcome1  : +0.0005  CI[+0.0004, +0.0006]
   run-value : +0.0001  CI[+0.0001, +0.0001]
 LOCKED TEST outcome1 Delta_order: +0.0003  CI[+0.0002, +0.0004]
 decomposed vs direct (D32) mean|dec-direct|: C=0.0041  U=0.0045  L1=0.0046  O=0.0046  OM=0.0048  (all within tolerance)
 top gain features (C view): strikes=3633152, action_family=1113673, balls=747929, batter_tend_whiff_rate=291169
 stages (s)     : behavior=1104.8  outcome=2709.6  assemble=2359.0  eval=553.2
 peak mem (MB)  : 12739.8 (outcome stage)
```

Headline reading: **ordered history predicts outcomes — statistically real, practically
tiny** (+0.0003 nats, replicated on the locked 2025 test); history-at-all (C→U) is worth
~0.0026, order adds ~0.0003 on top. **Matchup memory (+0.0005) carries more outcome
signal than within-PA order.** Selection order structure is robust (+0.0089). The SPEC
§13 expectation ("O barely beats L1") is vindicated in magnitude and, at this scale,
statistically certified rather than assumed.


---

## Detail digest (2026-07-16) — documentation inputs beyond the headlines

### First-pitch corroboration (scripts/first_pitch_check.py)
On 366,231 first pitches (val+test), where within-PA history cannot exist:
C=1.42138  L1=1.42060  U=1.42037  O=1.42076  OM=1.41952 (outcome1 log loss).
FIRST-PITCH Delta_matchup (O−OM) = +0.00124  CI[+0.00108, +0.00141] → **M+ CORROBORATED**
(~2.5× the overall Delta_matchup, concentrated exactly where only matchup memory can act).

### WS1 fitted concentrations (the "what did the data decide" story)
Selection: global=1, count_hand α=27.7, **pitcher α=2.35** (pitchers are extremely
distinct — almost no pooling), history α: L1=128.3, O=73.4, U=72.2 (history levels
shrunk hard toward the pitcher parent). Run value: **κ pegged at the 200 ceiling on
every non-global level** — cell-level run values are noise-dominated; the fitter pools
them as hard as allowed. Backoff at prediction (val rows): C resolves 626k at pitcher /
86k at count_hand / 0 global; L1 resolves 604k at history / 23k pitcher / 86k count_hand.

### WS2 per-depth concentrations + motifs
Depth concentrations: cell=27.7, pitcher=2.35, **depth1=17.9 (earns its keep)**,
depth2=84.7, depth3=113.0, depth4=166.7 — monotonically increasing: the data votes
"pool it away" progressively with depth. Permutation control on real data: effective-order
mass ≥2 → 0.000, mean 0.90 → 0.34 (machinery collapses correctly).
Top-20 motifs: depth-1 stickiness dominates (XX/FS/CH/SL/SI/CU all self-promote; the
largest KLs 0.16–0.03); depth-2 rows are real but an order of magnitude smaller
(KL ≤ 0.013), including double-up damping (e.g. [CH, CH] → suppress) and
fastball-re-promotion after off-speed pairs ([FS FF], [SL FF] → promote).

### WS3 detail
Top-10 gain, O view: strikes, action_family, balls, batter_tend_whiff/swing,
**prev_release_speed (193,583)**, u_n_prior, base_state, batter_tend_inplay/chase —
the dominant history feature is the previous pitch's speed; no single engineered
order feature dominates (the +0.0003 order effect is **diffuse**, unlike the synthetic
world where o_velo_delta_last was rank 1). OM's top-10 adds prev_pitch_type and shrinks
the batter_tend_* gains (matchup features absorb general-tendency signal).
TEST central (locked): C=1.4560, L1=1.4534, U=1.4532, O=1.4529, OM=1.4524;
test Δ_order=+0.000262 CI[+0.000178,+0.000363]; test Δ_matchup=+0.000539
CI[+0.000444,+0.000627]. Calibration ECE 0.0035–0.0037; top-1 acc 0.367–0.370;
skill vs marginal ~5.8–5.9%.
Chosen hyperparameters (D34 grid): behavior num_leaves 31 (C/U/L1) vs 63 (O/OM),
min_child_samples 100, lr 0.03, best iters 188–276; outcome num_leaves 15 (C) / 31,
min_child_samples 100 (C/U/L1) vs 20 (O/OM), iters 226–297. Param counts 46,624–96,264.

---

## WS4 — Bayesian contextual bandit (myopic prescription, OPE gate) (2026-07-17)

```
 world          : real
 rows           : train=2,143,214  eval(scored)=1,419,590
 feasibility    : mean #feasible/row=3.71  empty-mask(no-rec)=4.9%  low-history=4.6%
 behavior recovery: PASS  (observed=+0.0000  IPS weights unit=True)  [gate; SPEC 0.3 / D37]
 behavior value V(mu) = -0.0001   (alpha=0 baseline)

 D38 PRESCRIPTIVE-ABLATION (per view/alpha; common evaluator = O):
   every softened policy value <= V(mu); d(vs beh) negative and growing with alpha
   (C/L1/O near-identical): alpha .10 ~ -0.0009, .25 ~ -0.0022, .50 ~ -0.0045, 1.0 ~ -0.0089
   ESS% collapses 100 -> 24.7 -> 5.5 -> 1.7 -> 0.6 as alpha rises; oos_support ~ 0
   per-cell verdict: alpha 0 and 1 INCONCLUSIVE, alpha .10/.25/.50 CONSISTENT

 SEQUENCING-PRESCRIPTION GAPS (O-vs-C isolation, clustered by pitcher-game):
   alpha 0.00  O-C = +0.0000  CI[+0.0000,+0.0000]
   alpha 0.10  O-C = +0.0000  CI[+0.0000,+0.0000]  (CI excludes 0)
   alpha 0.25  O-C = +0.0001  CI[+0.0001,+0.0001]  (CI excludes 0)
   alpha 0.50  O-C = +0.0002  CI[+0.0001,+0.0002]  (CI excludes 0)
   alpha 1.00  O-C = +0.0003  CI[+0.0002,+0.0004]  (CI excludes 0);  L1-C tracks O-C

 AMBIGUITY : mean P(top>runner-up) ~0.57; ambiguous @95 = 1.00 (nearly every rec a toss-up)
             decidable=1,320,705  no-rec=69,823
 DEVIATION : mean TV from behavior at alpha=1: C=0.290 L1=0.293 O=0.296
 elapsed (s): 34182 (~9.5 h; FQE-in-loop dominates)   peak mem (MB): 8991.5
```

Headline reading: **the myopic prescriptive layer does not beat observed MLB behavior.**
The OPE gate PASSES (estimates are trustworthy); every deviation from behavior *lowers*
estimated value, and value-vs-behavior is negative at all alpha>0. The sequencing-specific
prescriptive isolation (O-vs-C) is statistically nonzero (CI excludes 0 from alpha>=0.1) but
**practically negligible** (~+0.0001 to +0.0003 run), and grows only as the policy moves
off-support (ESS -> 0.6%). ~99% of "best next pitch" recommendations are toss-ups at 95%
confidence. Consistent with WS3 finding #2 (order helps outcomes, but barely): a greedy
one-pitch-ahead recommender cannot convert that sliver into decision value. This is the
honest D39 result and the motivation for the sequential rungs (WS5 setup value, WS7 RL).
