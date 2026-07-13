"""Shared test fixtures and synthetic-frame factories.

The committed 20k parquet is a random *row* sample (19,050 PAs / 20,000 pitches), so
within-PA histories are almost all incomplete. It is therefore used only for schema
conformance, dtype handling, family coverage, reward sign, and end-to-end smoke tests.
Every test that asserts exact sequence / rolling / leakage values uses a small handcrafted
synthetic frame built here, where the ground truth is known.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "sample_statcast_2024.parquet"

# Column defaults for a single synthetic pitch. Only the fields a test cares about need to
# be overridden per row.
_DEFAULTS = dict(
    game_type="R",
    pitch_type="FF",
    game_date="2023-04-01",
    game_pk=1,
    at_bat_number=1,
    pitch_number=1,
    pitcher=100,
    batter=200,
    description="ball",
    events=None,
    balls=0,
    strikes=0,
    outs_when_up=0,
    on_1b=None,
    on_2b=None,
    on_3b=None,
    inning=1,
    inning_topbot="Top",
    fld_score=0,
    bat_score=0,
    stand="R",
    p_throws="R",
    home_team="NYY",
    away_team="BOS",
    n_thruorder_pitcher=1,
    release_speed=93.0,
    pfx_x=0.5,
    pfx_z=1.2,
    plate_x=0.0,
    plate_z=2.5,
    sz_top=3.4,
    sz_bot=1.6,
    release_spin_rate=2200,
    release_extension=6.5,
    release_pos_x=-1.0,
    release_pos_z=5.8,
    zone=5,
    delta_run_exp=-0.02,
)

# Columns that must stay nullable-friendly (occupancy holds a runner id or is empty).
_NULLABLE = ("on_1b", "on_2b", "on_3b")


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-Statcast-like DataFrame from partial row specifications.

    Parameters
    ----------
    rows : list of dict
        Each dict overrides :data:`_DEFAULTS` for one pitch.

    Returns
    -------
    pandas.DataFrame
        Columns and dtypes close enough to real Statcast for the pipeline to run.
    """
    records = []
    for row in rows:
        rec = dict(_DEFAULTS)
        rec.update(row)
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    # Nullable occupancy columns as object so ``None`` survives (notna -> base_state).
    for col in _NULLABLE:
        df[col] = df[col].astype("object")
    return df


@pytest.fixture(scope="session")
def sample_raw() -> pd.DataFrame:
    """The committed 20k row-sample loaded via the package loader (regular season)."""
    from pitchseq.io import load_statcast

    if not SAMPLE_PATH.is_file():
        pytest.skip(f"sample parquet not found at {SAMPLE_PATH}")
    return load_statcast(str(SAMPLE_PATH))


@pytest.fixture(scope="session")
def sample_table(sample_raw) -> pd.DataFrame:
    """Decision table built from the 20k sample (session-scoped, built once)."""
    from pitchseq.decision_table import build_decision_table

    return build_decision_table(sample_raw)
