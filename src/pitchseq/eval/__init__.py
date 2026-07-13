"""Shared evaluation harness for the pitch-sequencing study (SPEC ``8`` / ``10`` / ``11``).

Every workstream writes predictions in the standard schema (:mod:`.predictions`) and is
scored by the one harness (:mod:`.harness`); no workstream re-implements evaluation. The
pieces:

* :mod:`.predictions` -- the standard prediction-output contract + parquet round-trip.
* :mod:`.metrics` -- pure scoring functions (log loss, Brier, calibration, top-k,
  run-value calibration, ablation deltas, clustered CIs).
* :mod:`.baselines` -- the four count-based reference next-pitch models (decision D16).
* :mod:`.falsification` -- order ablation, history-permutation, pseudo-history and
  mechanism-ablation tests through a model-callback interface (decision D17).
* :mod:`.predictability` -- bits-of-predictability (SPEC ``10``).
* :mod:`.harness` -- ``evaluate_predictions`` / ``compare_views`` / ``write_report``.
"""

from __future__ import annotations

from . import baselines, falsification, harness, metrics, predictability, predictions

__all__ = [
    "predictions",
    "metrics",
    "baselines",
    "falsification",
    "predictability",
    "harness",
]
