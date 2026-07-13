"""``pitchseq`` -- shared foundation for the pitch-sequencing comparative study.

Workstream 0: the canonical decision table, the five nested state views (C / U / L1 / O /
OM), the fixed action space and reward, leakage-safe rolling features, sequence tensors,
and temporal splits. See ``SPEC.md`` for the design contract.
"""

from __future__ import annotations

from .config import load_config
from .families import FAMILIES, add_family, feasible_action_mask
from .io import load_statcast
from .outcomes import OUTCOME1, OUTCOME2, add_outcomes
from .reward import add_reward, episode_returns, sign_check
from .rolling import add_rolling
from .decision_table import (
    EXECUTION_COLS,
    LABEL_COLS,
    STATE_ELIGIBLE_COLS,
    build_decision_table,
    leakage_audit,
)
from .states import STATE_VIEWS, build_view
from .sequences import build_sequences
from .splits import cluster_bootstrap_indices, make_splits

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_config",
    "load_statcast",
    "FAMILIES",
    "add_family",
    "feasible_action_mask",
    "OUTCOME1",
    "OUTCOME2",
    "add_outcomes",
    "add_reward",
    "sign_check",
    "episode_returns",
    "add_rolling",
    "build_decision_table",
    "leakage_audit",
    "STATE_ELIGIBLE_COLS",
    "EXECUTION_COLS",
    "LABEL_COLS",
    "STATE_VIEWS",
    "build_view",
    "build_sequences",
    "make_splits",
    "cluster_bootstrap_indices",
]
