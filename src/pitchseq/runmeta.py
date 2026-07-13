"""Wall-clock and peak-memory instrumentation (decision D13 / D19).

Every runnable step in the study records the same run metadata -- wall-clock seconds,
peak resident-set size (RSS) in megabytes, and identifying context -- so the
performance-vs-compute Pareto plot (SPEC ``7``) and the prediction schema's ``seconds`` /
``peak_mem_mb`` fields (SPEC ``8.1``) are populated from one implementation.

Peak RSS is measured with :mod:`psutil` and is portable to Windows -- deliberately *not*
via the Unix-only :mod:`resource` module. A lightweight background thread samples
``Process.memory_info().rss`` at a fixed interval while the tracked block runs and keeps
the maximum; the current reading is also taken at entry and exit so even a block shorter
than one sampling interval yields a sane peak.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import psutil

__all__ = ["track_run", "write_runmeta", "read_runmeta"]

_BYTES_PER_MB = 1024.0 * 1024.0


class _PeakSampler:
    """Background sampler tracking the peak RSS of the current process.

    Parameters
    ----------
    interval_s : float
        Delay between samples, in seconds.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self._proc = psutil.Process()
        self._interval = float(interval_s)
        self._peak_bytes = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="runmeta-peak", daemon=True)

    def _sample(self) -> None:
        try:
            rss = self._proc.memory_info().rss
        except psutil.Error:  # pragma: no cover - process vanished; keep last peak
            return
        if rss > self._peak_bytes:
            self._peak_bytes = rss

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)

    def start(self) -> "_PeakSampler":
        self._thread.start()
        return self

    def stop(self) -> float:
        """Stop sampling and return the peak RSS observed, in megabytes."""
        self._stop.set()
        self._thread.join()
        self._sample()  # final reading, in case the peak is at the very end
        return self._peak_bytes / _BYTES_PER_MB

    @property
    def start_mb(self) -> float:
        return self._proc.memory_info().rss / _BYTES_PER_MB


@contextmanager
def track_run(interval_s: float = 0.02, **extra: Any) -> Iterator[dict]:
    """Context manager measuring wall-clock seconds and peak RSS of a block.

    The yielded dict is filled in on exit, so a caller keeps a reference and reads the
    measurements afterwards::

        with track_run(model_id="ws3_gbdt_O") as meta:
            train_and_predict()
        print(meta["seconds"], meta["peak_mem_mb"])

    Parameters
    ----------
    interval_s : float, optional
        Peak-RSS sampling interval in seconds (default ``0.02``).
    **extra : Any
        Arbitrary extra fields copied verbatim into the metadata dict (e.g. ``model_id``,
        ``view``, ``n_params``). They are recorded immediately so they are present even if
        the block raises.

    Yields
    ------
    dict
        Mutable metadata dict. After the block completes it holds ``seconds`` (float),
        ``peak_mem_mb`` (float), ``start_mem_mb`` (float), ``started_at`` (ISO-8601 UTC
        string), ``hostname`` (str), plus every key in ``extra``.
    """
    meta: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }
    meta.update(extra)

    sampler = _PeakSampler(interval_s=interval_s)
    meta["start_mem_mb"] = sampler.start_mb
    start = time.perf_counter()
    sampler.start()
    try:
        yield meta
    finally:
        elapsed = time.perf_counter() - start
        peak_mb = sampler.stop()
        meta["seconds"] = float(elapsed)
        meta["peak_mem_mb"] = float(peak_mb)


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder for numpy scalars / datetimes / paths."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_runmeta(path: str | Path, meta: dict) -> Path:
    """Write a run-metadata dict to a JSON file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file. Parent directories are created if needed.
    meta : dict
        The metadata (typically from :func:`track_run`, possibly augmented).

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True, default=_json_default)
    return out


def read_runmeta(path: str | Path) -> dict:
    """Read a run-metadata JSON file written by :func:`write_runmeta`."""
    with open(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)
