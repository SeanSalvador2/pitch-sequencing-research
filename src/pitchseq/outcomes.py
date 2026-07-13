"""Event-tree outcome labels (SPEC ``3.5`` / ``6`` / ``8.2``).

Two nested levels, locked for the whole study:

* **Level 1** ``outcome1`` from ``description`` -- the pitch result:
  ``ball / called_strike / whiff / foul / hbp / in_play``.
* **Level 2** ``outcome2`` from ``events``, defined only on ``in_play`` rows:
  ``single / double / triple / home_run / out_or_other``.

Plus two swing flags. ``description`` is a controlled Statcast vocabulary, so an unseen
value is treated as a data-integrity failure and raises (the fail-loud discipline of
SPEC ``0`` / ``11``): a silently mis-bucketed pitch result would corrupt every downstream
loss and reward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "OUTCOME1",
    "OUTCOME2",
    "DESCRIPTION_TO_OUTCOME1",
    "HIT_EVENTS",
    "SWING_OUTCOMES",
    "add_outcomes",
    "assert_outcome_coverage",
]

#: Level-1 categories, in a fixed order.
OUTCOME1: tuple[str, ...] = ("ball", "called_strike", "whiff", "foul", "hbp", "in_play")

#: Level-2 categories (conditional on ``in_play``), in a fixed order.
OUTCOME2: tuple[str, ...] = ("single", "double", "triple", "home_run", "out_or_other")

#: ``description`` -> level-1 outcome. Covers the full Statcast vocabulary, including the
#: automatic (pitch-timer / rule) calls: ``automatic_ball`` is a ball, ``automatic_strike``
#: is a called strike -- confirmed against the ``type`` column (B / S) on the sample.
DESCRIPTION_TO_OUTCOME1: dict[str, str] = {
    # ball
    "ball": "ball",
    "blocked_ball": "ball",
    "pitchout": "ball",
    "automatic_ball": "ball",
    # called strike
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    # whiff (swing producing a strike with no fair contact; foul_tip is caught -> a strike)
    "swinging_strike": "whiff",
    "swinging_strike_blocked": "whiff",
    "missed_bunt": "whiff",
    "foul_tip": "whiff",
    # foul
    "foul": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    # hit by pitch
    "hit_by_pitch": "hbp",
    # in play
    "hit_into_play": "in_play",
}

#: Level-2 events that are their own bucket; every other event on an in-play ball
#: collapses to ``out_or_other``.
HIT_EVENTS: frozenset[str] = frozenset({"single", "double", "triple", "home_run"})

#: Level-1 outcomes that involve a swing.
SWING_OUTCOMES: frozenset[str] = frozenset({"whiff", "foul", "in_play"})


def assert_outcome_coverage(df: pd.DataFrame) -> None:
    """Raise if any ``description`` value is outside the locked vocabulary.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``description``.

    Raises
    ------
    ValueError
        Listing every unrecognised (or null) ``description`` value and its count.
    """
    desc = df["description"]
    known = set(DESCRIPTION_TO_OUTCOME1)
    mask_unknown = ~desc.isin(known)  # null is not in ``known`` -> flagged
    if mask_unknown.any():
        offenders = desc[mask_unknown].value_counts(dropna=False)
        raise ValueError(
            "Unrecognised description value(s) not in the locked outcome vocabulary "
            f"(fail-loud, SPEC 0/11):\n{offenders.to_string()}"
        )


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``outcome1``, ``outcome2``, ``is_swing`` and ``is_whiff``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``description`` and ``events``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the four label columns added. ``outcome1`` / ``outcome2``
        are ``category`` dtype; ``outcome2`` is ``NA`` on non-in-play rows.

    Raises
    ------
    ValueError
        If an unseen ``description`` value is present (see :func:`assert_outcome_coverage`).
    """
    assert_outcome_coverage(df)
    out = df.copy()

    o1 = out["description"].map(DESCRIPTION_TO_OUTCOME1)
    out["outcome1"] = pd.Categorical(o1, categories=list(OUTCOME1))

    in_play = o1.to_numpy() == "in_play"
    o2 = pd.Series(pd.NA, index=out.index, dtype="object")
    if in_play.any():
        ev = out.loc[in_play, "events"]
        o2.loc[in_play] = ev.where(ev.isin(HIT_EVENTS), "out_or_other").astype("object")
    out["outcome2"] = pd.Categorical(o2, categories=list(OUTCOME2))

    is_swing = np.isin(o1.to_numpy(), list(SWING_OUTCOMES))
    out["is_swing"] = is_swing
    out["is_whiff"] = o1.to_numpy() == "whiff"
    return out
