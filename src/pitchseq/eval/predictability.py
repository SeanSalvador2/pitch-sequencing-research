r"""Predictability in bits (SPEC ``10``).

The batter-side read-out: how many extra **bits** about the next pitch does ordered history
supply beyond non-sequence context? For a next-pitch predictor :math:`q_O` trained on the
ordered view and :math:`q_C` on context only,

.. math:: B_{\text{seq}} = \frac{1}{n}\sum_t \log_2 \frac{q_O(A_t \mid S_t)}{q_C(A_t \mid X_t)}

is the mean per-pitch bits (a positive value means the ordered history makes the actual next
pitch more forecastable). It is reported broken out by pitcher / count / pitch-number /
family per SPEC ``10``.

.. note::
   The **exploitability** read-out of SPEC ``10`` (a batter response model + the common
   outcome model giving a per-state payoff, then each policy's average exploitability vs the
   equilibrium value) is a game-theoretic capstone that depends on the offline-RL outcome
   and policy machinery. It belongs to **WS7**, not this shared foundation, and is
   deliberately not implemented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..families import FAMILIES

__all__ = ["bits_of_predictability", "bits_summary"]

_FAM_INDEX = {f: i for i, f in enumerate(FAMILIES)}


def _action_indices(actions) -> np.ndarray:
    """Map family labels (or already-integer indices) to family column indices."""
    a = np.asarray(actions)
    if a.dtype.kind in ("i", "u"):
        return a.astype(np.int64)
    return np.array([_FAM_INDEX[str(x)] for x in a], dtype=np.int64)


def bits_of_predictability(q_o, q_c, actions, eps: float = 1e-12) -> np.ndarray:
    r"""Per-row bits of predictability :math:`\log_2 \frac{q_O(a_t)}{q_C(a_t)}`.

    Parameters
    ----------
    q_o : array-like, shape (n, 8)
        Ordered-view next-pitch probabilities (family columns in :data:`FAMILIES` order).
    q_c : array-like, shape (n, 8)
        Context-view next-pitch probabilities, same column order.
    actions : array-like, shape (n,)
        The realised next pitch's family (labels or integer indices into ``FAMILIES``).
    eps : float, optional
        Floor applied to both probabilities before the log (default ``1e-12``).

    Returns
    -------
    numpy.ndarray, shape (n,)
        Per-row bits. The mean is :math:`B_{\text{seq}}`.
    """
    qo = np.asarray(q_o, dtype=np.float64)
    qc = np.asarray(q_c, dtype=np.float64)
    if qo.shape != qc.shape:
        raise ValueError(f"q_o {qo.shape} and q_c {qc.shape} must have the same shape")
    idx = _action_indices(actions)
    n = qo.shape[0]
    po = np.clip(qo[np.arange(n), idx], eps, 1.0)
    pc = np.clip(qc[np.arange(n), idx], eps, 1.0)
    return np.log2(po / pc)


def bits_summary(df: pd.DataFrame, bits, by=("pitcher",)) -> pd.DataFrame:
    r"""Mean bits of predictability sliced by one or more grouping columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Carries the grouping columns (e.g. ``pitcher``, ``balls``, ``strikes``,
        ``pitch_number``, ``family``) aligned row-for-row to ``bits``.
    bits : array-like, shape (n,)
        Per-row bits from :func:`bits_of_predictability`.
    by : sequence of str, optional
        Grouping columns (default ``("pitcher",)``). Use e.g. ``["balls", "strikes"]`` for
        by-count, ``["pitch_number"]`` for by-depth, ``["family"]`` for by-pitch.

    Returns
    -------
    pandas.DataFrame
        One row per group with ``mean_bits``, ``sum_bits`` and ``count``, sorted by the
        grouping columns.
    """
    by = list(by)
    work = df[by].copy()
    work["_bits"] = np.asarray(bits, dtype=np.float64)
    grp = work.groupby(by, observed=True)["_bits"]
    out = grp.agg(mean_bits="mean", sum_bits="sum", count="size").reset_index()
    return out.sort_values(by).reset_index(drop=True)
