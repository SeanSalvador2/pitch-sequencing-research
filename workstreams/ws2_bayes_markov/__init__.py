"""Workstream 2 -- Bayesian variable-order Markov grammar (SPEC ``12.2``; decisions D29-D31).

The "pitch grammar" test: a per-context variable-order Markov model over pitch-family tokens
with hierarchical Dirichlet backoff, fusing the Markov and Bayesian-pooling ideas of the
study. Ordered within-PA history is the context; the next family is the token; the depth of
context actually used is *earned* by the data through fitted per-depth concentrations.

* :mod:`.model` -- the grammar. A two-axis hierarchical Dirichlet backoff (decision D29):
  the **depth axis** is the primary variable-order chain ``P_k(next | ctx_k, cell)`` shrunk
  toward the suffix distribution ``P_{k-1}(next | ctx_{k-1}, cell)`` with a per-depth
  concentration fitted by maximum marginal likelihood (method-of-moments fallback); the
  **cell axis** is WS1's C-ladder (global -> count x hand -> count x hand x pitcher) supplying
  the depth-0 base. Exposes posterior-mean predictions, per-row predictive variance and
  Dirichlet ESS, the effective-order read-out, the top-motif "grammar rules", and the
  order-usage summary. ``U`` and ``OM`` are declared not-applicable for a Markov grammar.
* :mod:`.run_ws2` -- the runnable step (D12): temporal split, fit C/L1/O, score through the
  shared harness, the SPEC ``10`` bits-of-predictability read-out (decision D31), the grammar
  exhibits, and the D30 detect / collapse-under-permutation acceptance verdicts.
"""
