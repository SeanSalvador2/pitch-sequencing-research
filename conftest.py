"""Repository-root pytest configuration.

Its sole job is to put the repository root on ``sys.path`` so the test suite can import the
modeling workstreams (``import workstreams.wsN_*...``), which live at the repository root
rather than inside the installed ``pitchseq`` package. ``pitchseq`` itself is imported from
its editable install (``pip install -e``); only ``workstreams`` needs this shim.

pytest already imports this file (a rootdir ``conftest.py``) before collecting tests, and
the explicit insert keeps the behaviour identical whether the suite is launched from the
repository root or elsewhere, on POSIX or Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
