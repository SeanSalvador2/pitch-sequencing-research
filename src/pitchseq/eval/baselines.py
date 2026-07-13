"""Count-based reference next-pitch (family) baselines (decision D16; SPEC ``8.2``).

Four transparent references every workstream model is scored against. Each predicts a
distribution over the 8 families and is **Dirichlet-smoothed toward a parent distribution**
so low-support cells shrink to a better-estimated ancestor and unseen keys fall back up the
hierarchy automatically.

Shrinkage design. A cell with observed family counts :math:`n = (n_1,\dots,n_8)` and parent
distribution :math:`\pi` (which sums to 1) predicts

.. math:: \hat p_c = \frac{n_c + \alpha\,\pi_c}{\sum_c n_c + \alpha}.

:math:`\alpha` is the pseudo-count strength (the "prior sample size"). When the cell is
unseen (:math:`\sum_c n_c = 0`) this returns exactly the parent :math:`\pi` -- i.e. the
model backs off one level up. The four models form a hierarchy of parents:

* **(a) global family frequency by count x batter-hand** -- parent = the overall family
  marginal;
* **(b) pitcher x count** -- parent = model (a) at the row's count and hand;
* **(c) first-order transition (prev-family x count)** -- parent = model (a);
* **(d) pitcher x count x prev-family** -- parent = the arithmetic mean of (b) and (c)
  (the two count-matched partial-pooling ancestors), so an unseen (pitcher, count,
  prev-family) cell backs off to the average of the pitcher-cell and transition-cell
  predictions, and thence -- through their own smoothing -- to (a) and the global marginal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..families import FAMILIES

__all__ = [
    "GlobalCountHandBaseline",
    "PitcherCountBaseline",
    "TransitionBaseline",
    "PitcherCountPrevBaseline",
    "prev_family_series",
    "build_baselines",
]

_FAM_INDEX = {f: i for i, f in enumerate(FAMILIES)}
_N_FAM = len(FAMILIES)
_NONE_PREV = "__NONE__"


def prev_family_series(table: pd.DataFrame) -> np.ndarray:
    """Previous family within each PA (``"__NONE__"`` on the first pitch of a PA)."""
    t = table[["pa_id", "pitch_number", "family"]].copy()
    order = np.lexsort((t["pitch_number"].to_numpy(), t["pa_id"].to_numpy()))
    prev = np.full(len(t), _NONE_PREV, dtype=object)
    fam = t["family"].astype("object").to_numpy()
    pa = t["pa_id"].to_numpy()
    fam_s = fam[order]
    pa_s = pa[order]
    prev_s = np.full(len(t), _NONE_PREV, dtype=object)
    for i in range(1, len(t)):
        if pa_s[i] == pa_s[i - 1]:
            prev_s[i] = fam_s[i - 1]
    prev[order] = prev_s
    return prev


def _family_counts(fam_labels: np.ndarray) -> np.ndarray:
    """Length-8 count vector over :data:`FAMILIES` for an array of family labels."""
    out = np.zeros(_N_FAM, dtype=np.float64)
    idx, counts = np.unique(fam_labels, return_counts=True)
    for f, c in zip(idx, counts):
        if f in _FAM_INDEX:
            out[_FAM_INDEX[f]] += c
    return out


def _smooth(counts: np.ndarray, parent: np.ndarray, alpha: float) -> np.ndarray:
    """Dirichlet-smooth a count vector toward a parent distribution."""
    total = counts.sum()
    return (counts + alpha * parent) / (total + alpha)


def _keys(table: pd.DataFrame, cols: list[str]) -> list[tuple]:
    """Row-wise tuple keys for the given columns."""
    arrs = [table[c].to_numpy() for c in cols]
    return list(zip(*[a.tolist() for a in arrs]))


class GlobalCountHandBaseline:
    """(a) Global family frequency by ``(balls, strikes, stand)``.

    Parameters
    ----------
    alpha : float, optional
        Dirichlet pseudo-count strength (default 8.0).
    """

    def __init__(self, alpha: float = 8.0) -> None:
        self.alpha = float(alpha)
        self.global_ = np.full(_N_FAM, 1.0 / _N_FAM)
        self.cells_: dict[tuple, np.ndarray] = {}

    def fit(self, table: pd.DataFrame) -> "GlobalCountHandBaseline":
        fam = table["family"].astype("object").to_numpy()
        self.global_ = _family_counts(fam)
        self.global_ = (
            self.global_ / self.global_.sum() if self.global_.sum() > 0 else np.full(_N_FAM, 1.0 / _N_FAM)
        )
        keys = _keys(table, ["balls", "strikes", "stand"])
        agg: dict[tuple, np.ndarray] = {}
        for key, f in zip(keys, fam):
            vec = agg.setdefault(key, np.zeros(_N_FAM))
            if f in _FAM_INDEX:
                vec[_FAM_INDEX[f]] += 1.0
        self.cells_ = agg
        return self

    def _row_proba(self, balls, strikes, stand) -> np.ndarray:
        counts = self.cells_.get((balls, strikes, stand), np.zeros(_N_FAM))
        return _smooth(counts, self.global_, self.alpha)

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        keys = _keys(table, ["balls", "strikes", "stand"])
        return np.vstack([self._row_proba(*k) for k in keys])


class PitcherCountBaseline:
    """(b) Pitcher x count, shrunk toward (a)."""

    def __init__(self, alpha: float = 8.0) -> None:
        self.alpha = float(alpha)
        self.parent = GlobalCountHandBaseline(alpha=alpha)
        self.cells_: dict[tuple, np.ndarray] = {}

    def fit(self, table: pd.DataFrame) -> "PitcherCountBaseline":
        self.parent.fit(table)
        fam = table["family"].astype("object").to_numpy()
        keys = _keys(table, ["pitcher", "balls", "strikes"])
        agg: dict[tuple, np.ndarray] = {}
        for key, f in zip(keys, fam):
            vec = agg.setdefault(key, np.zeros(_N_FAM))
            if f in _FAM_INDEX:
                vec[_FAM_INDEX[f]] += 1.0
        self.cells_ = agg
        return self

    def _row_proba(self, pitcher, balls, strikes, stand) -> np.ndarray:
        parent = self.parent._row_proba(balls, strikes, stand)
        counts = self.cells_.get((pitcher, balls, strikes), np.zeros(_N_FAM))
        return _smooth(counts, parent, self.alpha)

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        keys = _keys(table, ["pitcher", "balls", "strikes", "stand"])
        return np.vstack([self._row_proba(*k) for k in keys])


class TransitionBaseline:
    """(c) First-order transition ``prev_family x count``, shrunk toward (a)."""

    def __init__(self, alpha: float = 8.0) -> None:
        self.alpha = float(alpha)
        self.parent = GlobalCountHandBaseline(alpha=alpha)
        self.cells_: dict[tuple, np.ndarray] = {}

    def fit(self, table: pd.DataFrame) -> "TransitionBaseline":
        self.parent.fit(table)
        fam = table["family"].astype("object").to_numpy()
        prev = prev_family_series(table)
        balls = table["balls"].to_numpy()
        strikes = table["strikes"].to_numpy()
        agg: dict[tuple, np.ndarray] = {}
        for p, b, s, f in zip(prev, balls, strikes, fam):
            vec = agg.setdefault((p, b, s), np.zeros(_N_FAM))
            if f in _FAM_INDEX:
                vec[_FAM_INDEX[f]] += 1.0
        self.cells_ = agg
        return self

    def _row_proba(self, prev, balls, strikes, stand) -> np.ndarray:
        parent = self.parent._row_proba(balls, strikes, stand)
        counts = self.cells_.get((prev, balls, strikes), np.zeros(_N_FAM))
        return _smooth(counts, parent, self.alpha)

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        prev = prev_family_series(table)
        keys = list(zip(prev.tolist(), table["balls"].tolist(), table["strikes"].tolist(), table["stand"].tolist()))
        return np.vstack([self._row_proba(*k) for k in keys])


class PitcherCountPrevBaseline:
    """(d) Pitcher x count x prev-family, shrunk toward the mean of (b) and (c)."""

    def __init__(self, alpha: float = 8.0) -> None:
        self.alpha = float(alpha)
        self.b = PitcherCountBaseline(alpha=alpha)
        self.c = TransitionBaseline(alpha=alpha)
        self.cells_: dict[tuple, np.ndarray] = {}

    def fit(self, table: pd.DataFrame) -> "PitcherCountPrevBaseline":
        self.b.fit(table)
        self.c.fit(table)
        fam = table["family"].astype("object").to_numpy()
        prev = prev_family_series(table)
        pit = table["pitcher"].to_numpy()
        balls = table["balls"].to_numpy()
        strikes = table["strikes"].to_numpy()
        agg: dict[tuple, np.ndarray] = {}
        for pi, b, s, p, f in zip(pit, balls, strikes, prev, fam):
            vec = agg.setdefault((pi, b, s, p), np.zeros(_N_FAM))
            if f in _FAM_INDEX:
                vec[_FAM_INDEX[f]] += 1.0
        self.cells_ = agg
        return self

    def _row_proba(self, pitcher, balls, strikes, prev, stand) -> np.ndarray:
        parent_b = self.b._row_proba(pitcher, balls, strikes, stand)
        parent_c = self.c._row_proba(prev, balls, strikes, stand)
        parent = 0.5 * (parent_b + parent_c)
        parent = parent / parent.sum()
        counts = self.cells_.get((pitcher, balls, strikes, prev), np.zeros(_N_FAM))
        return _smooth(counts, parent, self.alpha)

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        prev = prev_family_series(table)
        keys = list(
            zip(
                table["pitcher"].tolist(),
                table["balls"].tolist(),
                table["strikes"].tolist(),
                prev.tolist(),
                table["stand"].tolist(),
            )
        )
        return np.vstack([self._row_proba(*k) for k in keys])


def build_baselines(alpha: float = 8.0) -> dict:
    """Instantiate the four reference baselines keyed by short name.

    Returns
    -------
    dict
        ``{'global_count_hand', 'pitcher_count', 'transition', 'pitcher_count_prev'}``.
    """
    return {
        "global_count_hand": GlobalCountHandBaseline(alpha=alpha),
        "pitcher_count": PitcherCountBaseline(alpha=alpha),
        "transition": TransitionBaseline(alpha=alpha),
        "pitcher_count_prev": PitcherCountPrevBaseline(alpha=alpha),
    }
