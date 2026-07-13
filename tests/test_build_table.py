"""End-to-end test of the ``python -m pitchseq.build_table`` CLI (D15).

Drives the builder on the committed 20k sample via ``--sample``, writing to a temp dir, and
checks the per-season checkpoint, the final parquet and the runmeta JSON sidecar exist with
matching row counts. The sample is a single season (2024), so one checkpoint is produced.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pitchseq import build_table
from pitchseq.runmeta import read_runmeta

from conftest import SAMPLE_PATH


@pytest.fixture()
def out_path(tmp_path):
    return tmp_path / "processed" / "decision_table.parquet"


def _require_sample():
    if not SAMPLE_PATH.is_file():
        pytest.skip(f"sample parquet not found at {SAMPLE_PATH}")


def test_build_table_end_to_end(out_path, sample_table):
    _require_sample()
    expected_rows = len(sample_table)  # whole-sample decision table

    meta = build_table.main(["--sample", str(SAMPLE_PATH), "--out", str(out_path)])

    checkpoint = out_path.parent / "decision_table_2024.parquet"
    sidecar = out_path.parent / "decision_table.runmeta.json"

    # All three artifacts exist.
    assert checkpoint.is_file()
    assert out_path.is_file()
    assert sidecar.is_file()

    # Row counts agree across checkpoint, final table and the whole-sample build.
    final = pd.read_parquet(out_path)
    cp = pd.read_parquet(checkpoint)
    assert len(final) == expected_rows
    assert len(cp) == expected_rows
    assert meta["total_rows"] == expected_rows
    assert meta["rows_per_season"] == {2024: expected_rows}

    # The final table equals the whole-sample build row-for-row (per-season == whole build).
    assert set(final["row_id"]) == set(sample_table["row_id"])
    assert list(final.columns) == list(sample_table.columns)

    # Sign check passed and was recorded.
    assert meta["sign_check"]["status"] == "PASS"

    # Runmeta sidecar round-trips and carries the timing / memory / build fields.
    saved = read_runmeta(sidecar)
    assert saved["total_rows"] == expected_rows
    assert saved["sign_check"]["status"] == "PASS"
    assert saved["built_seasons"] == [2024]
    assert saved["seconds"] >= 0.0
    assert saved["peak_mem_mb"] > 0.0
    assert isinstance(saved["columns"], list) and len(saved["columns"]) == saved["n_columns"]
    # It is valid JSON on disk.
    with open(sidecar, encoding="utf-8") as handle:
        assert isinstance(json.load(handle), dict)


def test_build_table_resume_skips_existing(out_path):
    _require_sample()
    # First build creates the checkpoint.
    build_table.main(["--sample", str(SAMPLE_PATH), "--out", str(out_path)])
    checkpoint = out_path.parent / "decision_table_2024.parquet"
    first_mtime = checkpoint.stat().st_mtime_ns

    # Resume with an explicit season list -> skip the build, checkpoint untouched.
    meta = build_table.main(
        ["--sample", str(SAMPLE_PATH), "--out", str(out_path), "--seasons", "2024", "--resume"]
    )
    assert meta["resumed_seasons"] == [2024]
    assert meta["built_seasons"] == []
    assert checkpoint.stat().st_mtime_ns == first_mtime  # not rewritten


def test_build_table_requires_a_source():
    with pytest.raises(SystemExit):
        build_table.main(["--out", "unused.parquet"])
