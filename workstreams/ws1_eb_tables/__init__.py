"""Workstream 1 -- empirical-Bayes conditional tables (SPEC ``12.1``; decisions D25-D28).

The transparent statistical baseline every later workstream must beat, and the exhibit that
exposes the *support problem*: as the conditioning key deepens (C -> L1 -> U -> O) the number
of distinct cells explodes and per-cell pitch support collapses, so naïve conditional
tables cannot see ordered structure without hierarchical shrinkage.

* :mod:`.model` -- the empirical-Bayes machinery. Selection tables are a hierarchical
  Dirichlet-multinomial over the D25 key ladder with per-level concentrations fitted by
  maximum marginal likelihood (method-of-moments fallback); run-value tables are a
  hierarchical normal partial-pooling model with per-level shrinkage strengths fitted the
  same way. Both expose posterior uncertainty and backoff-usage diagnostics.
* :mod:`.run_ws1` -- the runnable step (D12): temporal split, fit + predict per view, score
  through the shared harness, write predictions / report / support-diagnostics, print a
  compact headline block.
"""
