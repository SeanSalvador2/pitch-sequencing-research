"""The standard prediction-output contract (SPEC ``8.1``).

Every model -- regardless of workstream -- writes a single **wide** table keyed by
``row_id`` (decision D7: ``"{game_pk}_{at_bat_number}_{pitch_number}"``). A model fills in
whatever column groups it produces and leaves the rest absent:

======================  ===========================================================
group                   columns
======================  ===========================================================
``action_prob``         ``action_prob_<FAM>`` for the 8 families (behavior models)
``outcome1_prob``       ``outcome1_prob_<c>`` for the 6 level-1 outcomes
``outcome2_prob``       ``outcome2_prob_<c>`` for the 5 level-2 outcomes
``exp_reward``          ``exp_reward`` (+ optional ``exp_reward_sd``)
``policy_prob``         ``policy_prob_<FAM>`` for the 8 families (prescriptive models)
======================  ===========================================================

Required metadata columns on every row: ``model_id``, ``state_view`` (one of
``C/U/L1/O/OM`` or ``"NA"``), ``seconds``, ``peak_mem_mb``, ``n_params`` (the last three
come straight from :func:`pitchseq.runmeta.track_run`).

A probability group is validated only where it is fully populated: rows in which the whole
group is ``NaN`` are treated as *not applicable* (e.g. ``outcome2`` on a pitch that was not
put in play) and skipped; a row with a partially-filled group is an error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..families import FAMILIES
from ..outcomes import OUTCOME1, OUTCOME2
from ..states import STATE_VIEWS

__all__ = [
    "ROW_ID",
    "ACTION_PROB_COLS",
    "OUTCOME1_PROB_COLS",
    "OUTCOME2_PROB_COLS",
    "POLICY_PROB_COLS",
    "META_COLS",
    "PROB_GROUPS",
    "LEGAL_STATE_VIEWS",
    "validate_predictions",
    "save_predictions",
    "load_predictions",
    "prediction_groups_present",
]

ROW_ID = "row_id"

ACTION_PROB_COLS = [f"action_prob_{f}" for f in FAMILIES]
OUTCOME1_PROB_COLS = [f"outcome1_prob_{c}" for c in OUTCOME1]
OUTCOME2_PROB_COLS = [f"outcome2_prob_{c}" for c in OUTCOME2]
POLICY_PROB_COLS = [f"policy_prob_{f}" for f in FAMILIES]

META_COLS = ["model_id", "state_view", "seconds", "peak_mem_mb", "n_params"]

#: Named probability groups and their (ordered) column lists.
PROB_GROUPS: dict[str, list[str]] = {
    "action_prob": ACTION_PROB_COLS,
    "outcome1_prob": OUTCOME1_PROB_COLS,
    "outcome2_prob": OUTCOME2_PROB_COLS,
    "policy_prob": POLICY_PROB_COLS,
}

LEGAL_STATE_VIEWS = set(STATE_VIEWS) | {"NA"}


def prediction_groups_present(df: pd.DataFrame) -> list[str]:
    """Return the names of the probability groups fully present in ``df``."""
    present = []
    for name, cols in PROB_GROUPS.items():
        have = [c for c in cols if c in df.columns]
        if have:
            present.append(name)
    return present


def _check_group(df: pd.DataFrame, name: str, cols: list[str], tol: float, problems: list[str]) -> None:
    """Validate one probability group (see module docstring for the rule)."""
    missing = [c for c in cols if c not in df.columns]
    if missing and len(missing) != len(cols):
        problems.append(
            f"group {name!r} partially present: missing {missing} (a group is all-or-nothing)"
        )
        return
    if missing:  # entire group absent -> nothing to check
        return

    block = df[cols].to_numpy(dtype=float)
    all_nan = np.isnan(block).all(axis=1)
    any_nan = np.isnan(block).any(axis=1)
    partial = any_nan & ~all_nan
    if partial.any():
        idx = np.flatnonzero(partial)[:5].tolist()
        problems.append(f"group {name!r} has partially-NaN rows (row positions {idx} ...)")

    active = ~all_nan
    if not active.any():
        return
    active_block = block[active]
    if np.nanmin(active_block) < -tol:
        problems.append(f"group {name!r} has negative probabilities")
    sums = np.nansum(active_block, axis=1)
    bad = np.abs(sums - 1.0) > tol
    if bad.any():
        worst = float(np.max(np.abs(sums - 1.0)))
        problems.append(
            f"group {name!r} rows do not sum to 1 (max deviation {worst:.3g} > tol {tol:g})"
        )


def validate_predictions(df: pd.DataFrame, tol: float = 1e-6) -> bool:
    """Validate a prediction table against the SPEC ``8.1`` contract.

    Checks: ``row_id`` present and unique; every fully-present probability group is
    non-negative and sums to 1 within ``tol`` on its applicable rows; ``state_view`` is
    legal; the required metadata columns are present.

    Parameters
    ----------
    df : pandas.DataFrame
        The prediction table.
    tol : float, optional
        Absolute tolerance for the sum-to-one / non-negativity checks (default ``1e-6``).

    Returns
    -------
    bool
        ``True`` when valid.

    Raises
    ------
    ValueError
        With a message naming every specific problem found.
    """
    problems: list[str] = []

    if ROW_ID not in df.columns:
        problems.append(f"missing required key column {ROW_ID!r}")
    else:
        dup = df[ROW_ID].duplicated()
        if dup.any():
            examples = df.loc[dup, ROW_ID].unique()[:5].tolist()
            problems.append(f"{ROW_ID} is not unique (e.g. {examples})")

    for meta in META_COLS:
        if meta not in df.columns:
            problems.append(f"missing required metadata column {meta!r}")

    if "state_view" in df.columns:
        illegal = set(df["state_view"].dropna().astype(str).unique()) - LEGAL_STATE_VIEWS
        if illegal:
            problems.append(
                f"illegal state_view value(s) {sorted(illegal)}; legal = {sorted(LEGAL_STATE_VIEWS)}"
            )

    present_any = False
    for name, cols in PROB_GROUPS.items():
        if any(c in df.columns for c in cols):
            present_any = True
        _check_group(df, name, cols, tol, problems)

    if not present_any and "exp_reward" not in df.columns:
        problems.append("no probability group and no exp_reward present -- nothing to score")

    if problems:
        raise ValueError("Invalid prediction table (SPEC 8.1):\n  - " + "\n  - ".join(problems))
    return True


def save_predictions(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a prediction table to parquet, preserving dtypes on round-trip.

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction table (validated first).
    path : str or pathlib.Path
        Destination ``.parquet`` file.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    validate_predictions(df)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read a prediction table written by :func:`save_predictions`."""
    return pd.read_parquet(Path(path), engine="pyarrow")
