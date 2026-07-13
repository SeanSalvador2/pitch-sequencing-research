# Pitch Sequencing Research

A comparative study of approaches to **pitch sequencing** in baseball — the decision of
what to throw next given the game state *and the sequence of pitches already thrown*.
Rather than chasing a single best predictor, the project builds seven methods across
paradigms (statistical, Bayesian, stochastic-process, classical ML, deep learning,
reinforcement learning) and compares them honestly on one shared benchmark.

The central question is a **rigor ladder**, not a horse race:

> How much evidence for sequencing survives progressively harder tests — from descriptive
> order patterns, to out-of-sample *outcome* dependence, to counterfactual *policy* value?

See **[SPEC.md](SPEC.md)** for the full design contract: the shared decision table, the
five nested state views (C / U / L1 / O / OM) that isolate sequencing, the fixed action
space and reward, the temporal split, the evaluation harness, the off-policy-evaluation
interface, and the synthetic-fixture correctness oracle.

## Data

Statcast pitch-level data, 2021–2025 (~3.85M pitches, one tracking era). Pulled locally
with `pull_statcast.py` (data is gitignored — it lives on your machine, not in git):

```bash
pip install -r requirements.txt
python pull_statcast.py --sqlite         # -> data/raw/statcast.db  + per-season parquet
```

## Workstreams

| # | Workstream | Family | Role |
|---|---|---|---|
| 0 | Shared foundation | — | decision table, state builders, eval harness, OPE (SPEC.md) |
| 1 | Empirical-Bayes tables | Statistical/Bayesian | transparent baseline |
| 2 | Bayesian variable-order Markov | Probabilistic | pitch-grammar / next-pitch |
| 3 | GBDT behavior + outcome stack | Classical ML | **centerpiece**: the C/U/L1/O/OM ablation + propensities + outcome model |
| 4 | Bayesian contextual bandit | Bayesian decision | myopic prescription |
| 5 | Tabular MDP | Stochastic process | sequential, interpretable prescription |
| 6 | Deep sequence (GRU, opt. Transformer) | Deep learning | learned representation vs engineered history |
| 7 | Conservative offline RL + OPE | RL / game theory | full-PA prescription, honestly evaluated |

Build order: WS0 → WS1 → WS2 → WS3 → OPE self-tests → WS4 → WS5 → WS6 → WS7.
**Phase A (WS1–3 + WS6) is a complete study on its own** if time is short; the
prescriptive phase is upside.

## Likely finding

The most probable honest result is that full pitch *order* (state O) adds little
out-of-sample signal beyond the last pitch (L1) — consistent with recent motif work. The
study is built so that either outcome is a clean, reportable finding.
