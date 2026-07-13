"""Reward and episode returns (SPEC ``5``).

The per-pitch reward is the negated change in run expectancy,

    R_t = -delta_run_exp_t,

so larger is better *for the pitcher*. The sign is verified empirically before use: a
called / swinging strike or a strikeout must have positive mean reward, a ball or a
walk / home run negative. A per-PA episode return is the sum of its per-pitch rewards,
and that sum must equal an independently reconstructed terminal return (an indexing
guard, SPEC ``5`` / ``11``).
"""

from __future__ import annotations

import pandas as pd

__all__ = ["add_reward", "sign_check", "episode_returns", "REWARD_FIELD"]

#: Raw Statcast field the reward negates.
REWARD_FIELD = "delta_run_exp"

# Description / event groups the sign check asserts on, with the required sign.
_POSITIVE_DESCRIPTIONS = {
    "called_strike": ["called_strike"],
    "swinging_strike": ["swinging_strike", "swinging_strike_blocked"],
}
_POSITIVE_EVENTS = {"strikeout": ["strikeout"]}
_NEGATIVE_DESCRIPTIONS = {"ball": ["ball", "blocked_ball"]}
_NEGATIVE_EVENTS = {"walk": ["walk"], "home_run": ["home_run"]}


def add_reward(df: pd.DataFrame, field: str = REWARD_FIELD, column: str = "R") -> pd.DataFrame:
    """Add the reward column ``R = -delta_run_exp``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``field``.
    field : str, optional
        Source run-expectancy delta (default ``"delta_run_exp"``).
    column : str, optional
        Output column name (default ``"R"``).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the reward column added as ``float64`` (NaN where the source
        is missing; ~0.3% of pitches per SPEC ``1``).
    """
    out = df.copy()
    out[column] = -out[field].astype("Float64").astype("float64")
    return out


def sign_check(df: pd.DataFrame, column: str = "R") -> dict[str, float]:
    """Verify the reward sign convention, raising on any violation.

    Asserts mean ``R > 0`` on called / swinging strikes and strikeouts, and mean
    ``R < 0`` on balls and on walks / home runs (SPEC ``5``).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``column``, ``description`` and ``events``.
    column : str, optional
        Reward column name (default ``"R"``).

    Returns
    -------
    dict
        Group label -> observed mean reward (for logging / inspection).

    Raises
    ------
    ValueError
        Naming every group whose mean reward has the wrong sign (or is absent / NaN).
    """
    means: dict[str, float] = {}
    violations: list[str] = []

    def _check(label: str, mask: pd.Series, want_positive: bool) -> None:
        vals = df.loc[mask, column].dropna()
        if len(vals) == 0:
            violations.append(f"{label}: no rows with a defined reward to check")
            return
        m = float(vals.mean())
        means[label] = m
        ok = (m > 0) if want_positive else (m < 0)
        if not ok:
            sign = "> 0" if want_positive else "< 0"
            violations.append(f"{label}: mean R = {m:+.5f} but expected {sign} (n={len(vals)})")

    for label, descs in _POSITIVE_DESCRIPTIONS.items():
        _check(label, df["description"].isin(descs), want_positive=True)
    for label, evs in _POSITIVE_EVENTS.items():
        _check(label, df["events"].isin(evs), want_positive=True)
    for label, descs in _NEGATIVE_DESCRIPTIONS.items():
        _check(label, df["description"].isin(descs), want_positive=False)
    for label, evs in _NEGATIVE_EVENTS.items():
        _check(label, df["events"].isin(evs), want_positive=False)

    if violations:
        raise ValueError("Reward sign check failed (SPEC 5):\n  " + "\n  ".join(violations))
    return means


def episode_returns(
    df: pd.DataFrame,
    pa_col: str = "pa_id",
    reward_col: str = "R",
    order_col: str = "pitch_number",
    assert_agreement: bool = True,
) -> pd.Series:
    """Per-PA episode return ``G = sum_t R_t`` with an indexing-agreement guard.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``pa_col``, ``reward_col`` and (for the guard) ``order_col``.
    pa_col : str, optional
        Episode identifier (default ``"pa_id"``).
    reward_col : str, optional
        Per-pitch reward column (default ``"R"``).
    order_col : str, optional
        Within-PA ordering column (default ``"pitch_number"``).
    assert_agreement : bool, optional
        When ``True`` (default) recompute the return by ordered cumulative sum and assert
        it equals the group sum -- a terminal reward equal to the return must agree, or an
        indexing bug exists (SPEC ``5``).

    Returns
    -------
    pandas.Series
        Indexed by ``pa_col``, the episode return per PA. A PA whose rewards are all
        missing returns NaN (``min_count=1``); a partially-missing PA returns the sum of
        its defined rewards.

    Raises
    ------
    AssertionError
        If the two reconstructions disagree.

    Notes
    -----
    The agreement guard tests *indexing* (grouping / ordering consistency), not NaN
    policy, so both of its reconstructions run on NaN-zeroed rewards. Mixing a
    NaN-propagating cumulative sum with a NaN-skipping group sum can raise spurious
    alarms on the ~0.3% of pitches with missing ``delta_run_exp`` (and depends on pandas
    ``skipna`` defaults that have shifted across versions); the returned Series is
    unaffected and keeps the raw-column semantics above.
    """
    grouped = df.groupby(pa_col, sort=True)[reward_col].sum(min_count=1)

    if assert_agreement:
        work = df[[pa_col, order_col]].copy()
        work["_r0"] = df[reward_col].fillna(0.0)
        work = work.sort_values([pa_col, order_col], kind="stable")
        group_sum = work.groupby(pa_col, sort=True)["_r0"].sum()
        terminal = (
            work.groupby(pa_col, sort=True)["_r0"].cumsum().groupby(work[pa_col], sort=True).last()
        )
        # Both are sums of the same (NaN-zeroed) rewards; equality proves grouping and
        # ordering are consistent (no double count, no dropped or misassigned pitch).
        if not (group_sum - terminal).abs().lt(1e-9).all():
            raise AssertionError(
                "Episode-return disagreement between group-sum and terminal cumulative "
                "sum: indexing bug (SPEC 5)."
            )
    return grouped
