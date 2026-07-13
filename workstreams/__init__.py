"""Modeling workstreams for the pitch-sequencing comparative study.

Each ``wsN_*`` subpackage reads the shared :mod:`pitchseq` foundation (decision table,
state views, reward, splits) and the shared evaluation harness, writes standard-schema
predictions, and is scored **only** through :mod:`pitchseq.eval.harness` -- no workstream
re-implements evaluation (SPEC ``2`` / ``8``).

``workstreams`` is a regular package (this file makes it importable as
``workstreams.wsN_*``). It lives at the repository root rather than under ``src/`` because
it is not part of the installable ``pitchseq`` library; the runnable steps add the
repository root to ``sys.path`` so ``python workstreams/wsN_*/run_*.py`` works from a
fresh clone on any platform.
"""
