"""Configuration loader.

The single source of truth for family maps, split boundaries, rolling windows, and
thresholds is ``configs/default.yaml`` at the repository root (SPEC ``2``). This module
locates and parses it so every other module reads the same contract.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "default_config_path"]

_CONFIG_ENV_VAR = "PITCHSEQ_CONFIG"
_DEFAULT_REL = Path("configs") / "default.yaml"


def default_config_path() -> Path:
    """Locate ``configs/default.yaml``.

    Resolution order:

    1. the ``PITCHSEQ_CONFIG`` environment variable, if set;
    2. the first ancestor of this file that contains ``configs/default.yaml``
       (works for an editable ``src`` layout install);
    3. ``configs/default.yaml`` under the current working directory.

    Returns
    -------
    pathlib.Path
        Path to the configuration file (existence is not guaranteed for case 3).
    """
    env = os.environ.get(_CONFIG_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()

    # Walk up from this module: src/pitchseq/config.py -> repo root holds configs/.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _DEFAULT_REL
        if candidate.is_file():
            return candidate

    return (Path.cwd() / _DEFAULT_REL).resolve()


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the study configuration as a plain dictionary.

    Parameters
    ----------
    path : str or os.PathLike, optional
        Explicit path to a YAML config. When ``None`` the canonical
        ``configs/default.yaml`` is used (see :func:`default_config_path`).

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    cfg_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {cfg_path} did not parse to a mapping.")
    return cfg
