"""The canonical per-decision table (SPEC ``3``).

Grain: one row per pitch = the decision made immediately *before* pitch ``t``. Columns are
grouped so leakage audits are mechanical:

1. **identity / keys** -- ordering and join keys;
2. **context ``X_t``** -- everything knowable before the pitch (this is the ``C`` view);
3. **rolling baselines** -- pitcher per-family execution baselines (pre-game, used later
   for history diffs-from-baseline);
4. **action** -- the family / pitch_type actually chosen (the decision itself);
5. **execution ``Z_t``** -- realized physical measurements of pitch ``t``, every column
   ``exec_``-prefixed; downstream of the decision, never a state feature;
6. **labels ``Y_t`` / reward** -- event-tree outcomes and ``R``.

The batter-relative location conventions ``plate_x_br`` (sign-flipped for LHB so positive
is the outer half either way) and ``plate_z_norm`` (strike-zone-normalized height) are
applied here and stored under the ``exec_`` prefix, because a pitch's *own* location is
execution -- it is only legitimate history material for the *next* pitch's decision.

On the committed 20k row-sample most within-PA histories are absent (it is a random row
sample), so per-pitch history material is frequently a length-0 prefix; the sequence and
state builders handle that by using only the pitches actually present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config
from .families import FAMILIES, add_family
from .io import SORT_KEYS
from .outcomes import add_outcomes
from .reward import add_reward
from .rolling import add_rolling, batter_tend_columns, repertoire_mix_columns

__all__ = [
    "build_decision_table",
    "leakage_audit",
    "decision_table_column_groups",
    "IDENTITY_COLS",
    "CONTEXT_COLS",
    "ROLLING_BASELINE_COLS",
    "ACTION_COLS",
    "EXECUTION_COLS",
    "LABEL_COLS",
    "STATE_ELIGIBLE_COLS",
]

IDENTITY_COLS = [
    "game_pk",
    "game_date",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "season",
    "pa_id",
    "is_seq_eligible",
    "row_id",
]

# Context X_t == the C view (SPEC 3.2 / 6). Rolling repertoire/tendency columns are part
# of context because they use only pre-game data.
_CONTEXT_BASE = [
    "balls",
    "strikes",
    "outs_when_up",
    "base_state",
    "inning",
    "inning_topbot",
    "score_diff_pitcher",
    "stand",
    "p_throws",
    "platoon",
    "pitcher_pitch_count",
    "times_thru_order",
    "park",
]
_CONTEXT_ROLLING = (
    repertoire_mix_columns()
    + batter_tend_columns()
    + [
        "pitcher_n_trailing_pitches",
        "pitcher_low_history",
        "batter_n_trailing_pitches",
        "batter_low_history",
    ]
)
CONTEXT_COLS = _CONTEXT_BASE + _CONTEXT_ROLLING

ROLLING_BASELINE_COLS = [
    f"pitcher_base_{fam}_{measure}"
    for fam in FAMILIES
    for measure in ("release_speed", "pfx_x", "pfx_z")
]

ACTION_COLS = ["family", "pitch_type"]

# Execution Z_t -- realized measurements of pitch t. The ``exec_`` prefix is what the
# leakage audit keys on.
EXECUTION_COLS = [
    "exec_release_speed",
    "exec_pfx_x",
    "exec_pfx_z",
    "exec_plate_x",
    "exec_plate_z",
    "exec_plate_x_br",
    "exec_plate_z_norm",
    "exec_release_spin_rate",
    "exec_release_extension",
    "exec_release_pos_x",
    "exec_release_pos_z",
]

LABEL_COLS = ["outcome1", "outcome2", "is_swing", "is_whiff", "R"]

# Columns permitted to appear in a pre-decision state (identity is not a feature per se
# but is safe/pre-known). Actions are the decision/target, executions and labels are
# downstream -- none of those may be state features.
STATE_ELIGIBLE_COLS = IDENTITY_COLS + CONTEXT_COLS + ROLLING_BASELINE_COLS

# Raw execution source columns, in the same order as EXECUTION_COLS (plate_x_br /
# plate_z_norm are derived below, hence None).
_EXEC_SOURCE = {
    "exec_release_speed": "release_speed",
    "exec_pfx_x": "pfx_x",
    "exec_pfx_z": "pfx_z",
    "exec_plate_x": "plate_x",
    "exec_plate_z": "plate_z",
    "exec_release_spin_rate": "release_spin_rate",
    "exec_release_extension": "release_extension",
    "exec_release_pos_x": "release_pos_x",
    "exec_release_pos_z": "release_pos_z",
}


def decision_table_column_groups() -> dict[str, list[str]]:
    """Return the canonical column grouping (for documentation and tests)."""
    return {
        "identity": list(IDENTITY_COLS),
        "context": list(CONTEXT_COLS),
        "rolling_baseline": list(ROLLING_BASELINE_COLS),
        "action": list(ACTION_COLS),
        "execution": list(EXECUTION_COLS),
        "label": list(LABEL_COLS),
    }


def leakage_audit(df_or_cols) -> bool:
    """Assert that no execution or label column is being used as a state feature.

    Parameters
    ----------
    df_or_cols : pandas.DataFrame or iterable of str
        A frame (its columns are audited) or a list of proposed state feature names.

    Returns
    -------
    bool
        ``True`` when clean.

    Raises
    ------
    AssertionError
        Naming every offending execution (``exec_``-prefixed) or label column (SPEC ``0`` /
        ``11``).
    """
    cols = list(df_or_cols.columns) if isinstance(df_or_cols, pd.DataFrame) else list(df_or_cols)
    bad_exec = [c for c in cols if c in EXECUTION_COLS or c.startswith("exec_")]
    bad_label = [c for c in cols if c in LABEL_COLS]
    if bad_exec or bad_label:
        raise AssertionError(
            "Leakage audit failed (SPEC 0/11): execution/label columns present as state "
            f"features -- exec={bad_exec}, label={bad_label}"
        )
    return True


def _int(series: pd.Series) -> np.ndarray:
    return series.astype("Int64").astype("int64").to_numpy()


def _float(series: pd.Series) -> np.ndarray:
    return series.astype("Float64").astype("float64").to_numpy()


def build_decision_table(raw_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Build the canonical decision table from raw Statcast rows.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw Statcast pitches (e.g. from :func:`~pitchseq.io.load_statcast`).
    config : dict, optional
        Parsed config; loaded from the default path when ``None``.

    Returns
    -------
    pandas.DataFrame
        One row per pitch with columns ordered by group (see
        :func:`decision_table_column_groups`).
    """
    if config is None:
        config = load_config()

    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)

    # Derived labels / action / reward / rolling context.
    df = add_family(df, config)
    df = add_outcomes(df)
    df = add_reward(df)
    df = add_rolling(df, config)

    out = pd.DataFrame(index=df.index)

    # --- identity / keys ---
    out["game_pk"] = _int(df["game_pk"])
    out["game_date"] = df["game_date"].to_numpy()
    out["at_bat_number"] = _int(df["at_bat_number"])
    out["pitch_number"] = _int(df["pitch_number"])
    out["pitcher"] = _int(df["pitcher"])
    out["batter"] = _int(df["batter"])
    out["season"] = df["game_date"].dt.year.astype("int64").to_numpy()
    out["pa_id"] = out["game_pk"].to_numpy() * 1000 + out["at_bat_number"].to_numpy()
    out["is_seq_eligible"] = out["pitch_number"].to_numpy() >= 2
    out["row_id"] = (
        out["game_pk"].astype(str)
        + "_"
        + out["at_bat_number"].astype(str)
        + "_"
        + out["pitch_number"].astype(str)
    )

    # --- context X_t ---
    out["balls"] = _int(df["balls"])
    out["strikes"] = _int(df["strikes"])
    out["outs_when_up"] = _int(df["outs_when_up"])
    on1 = df["on_1b"].notna().to_numpy()
    on2 = df["on_2b"].notna().to_numpy()
    on3 = df["on_3b"].notna().to_numpy()
    out["base_state"] = (on1 * 1 + on2 * 2 + on3 * 4).astype("int64")
    out["inning"] = _int(df["inning"])
    out["inning_topbot"] = df["inning_topbot"].astype("object").to_numpy()
    out["score_diff_pitcher"] = (_int(df["fld_score"]) - _int(df["bat_score"])).astype("int64")
    stand = df["stand"].astype("object").to_numpy()
    p_throws = df["p_throws"].astype("object").to_numpy()
    out["stand"] = stand
    out["p_throws"] = p_throws
    out["platoon"] = stand == p_throws
    # Cumulative pitches by this pitcher in this game, strictly before t (rows are in
    # pitch order within a game, so cumcount is the pre-t count).
    out["pitcher_pitch_count"] = df.groupby(["game_pk", "pitcher"]).cumcount().astype("int64").to_numpy()
    out["times_thru_order"] = _int(df["n_thruorder_pitcher"])
    out["park"] = df["home_team"].astype("object").to_numpy()
    for col in _CONTEXT_ROLLING:
        out[col] = df[col].to_numpy()

    # --- rolling baselines ---
    for col in ROLLING_BASELINE_COLS:
        out[col] = df[col].to_numpy()

    # --- action ---
    out["family"] = pd.Categorical(df["family"].astype("object").to_numpy(), categories=list(FAMILIES))
    out["pitch_type"] = df["pitch_type"].astype("object").to_numpy()

    # --- execution Z_t (batter-relative location conventions applied here) ---
    for exec_col, src in _EXEC_SOURCE.items():
        out[exec_col] = _float(df[src])
    plate_x = _float(df["plate_x"])
    plate_z = _float(df["plate_z"])
    sz_top = _float(df["sz_top"])
    sz_bot = _float(df["sz_bot"])
    # plate_x_br: unchanged for RHB, sign-flipped for LHB (positive = outer half either way).
    out["exec_plate_x_br"] = np.where(stand == "L", -plate_x, plate_x)
    denom = sz_top - sz_bot
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(denom != 0, (plate_z - sz_bot) / denom, np.nan)
    out["exec_plate_z_norm"] = norm

    # --- labels / reward ---
    out["outcome1"] = df["outcome1"].to_numpy()
    out["outcome2"] = df["outcome2"].to_numpy()
    out["is_swing"] = df["is_swing"].to_numpy()
    out["is_whiff"] = df["is_whiff"].to_numpy()
    out["R"] = df["R"].to_numpy()

    ordered = (
        IDENTITY_COLS
        + CONTEXT_COLS
        + ROLLING_BASELINE_COLS
        + ACTION_COLS
        + EXECUTION_COLS
        + LABEL_COLS
    )
    return out[ordered]
