"""Run-metadata instrumentation (decision D13 / D19)."""

from __future__ import annotations

import time

import numpy as np

from pitchseq.runmeta import read_runmeta, track_run, write_runmeta


def test_track_run_measures_time_and_memory():
    with track_run(model_id="unit_test", view="O", n_params=123) as meta:
        big = np.ones(8_000_000, dtype=np.float64)  # ~64 MB, touched by np.ones
        big += 1.0
        time.sleep(0.12)
        keep = float(big.sum())  # keep the array resident through the block

    assert keep > 0
    assert meta["seconds"] >= 0.10
    assert meta["peak_mem_mb"] >= meta["start_mem_mb"]
    # The ~64 MB allocation must show up in the peak (generous slack for the sampler).
    assert meta["peak_mem_mb"] - meta["start_mem_mb"] >= 15.0
    # Extra fields and identifying context are recorded.
    assert meta["model_id"] == "unit_test"
    assert meta["view"] == "O"
    assert meta["n_params"] == 123
    assert isinstance(meta["hostname"], str) and meta["hostname"]
    assert isinstance(meta["started_at"], str) and "T" in meta["started_at"]


def test_write_and_read_runmeta_roundtrip(tmp_path):
    with track_run(model_id="m") as meta:
        time.sleep(0.01)
    path = tmp_path / "sub" / "runmeta.json"
    written = write_runmeta(path, meta)
    assert written.is_file()
    back = read_runmeta(path)
    assert back["model_id"] == "m"
    assert back["seconds"] == meta["seconds"]
    assert back["peak_mem_mb"] == meta["peak_mem_mb"]
    assert back["hostname"] == meta["hostname"]


def test_track_run_records_extra_even_on_exception():
    captured = {}
    try:
        with track_run(model_id="boom") as meta:
            captured = meta
            raise ValueError("expected")
    except ValueError:
        pass
    assert captured.get("model_id") == "boom"
    assert "seconds" in captured  # finally-block filled the timing in
    assert captured["seconds"] >= 0.0
