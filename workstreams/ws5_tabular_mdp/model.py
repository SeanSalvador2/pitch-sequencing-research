r"""Tabular MDP / controlled Markov reward process -- the setup-valuing rung (SPEC ``12.6``).

WS5 is the simplest *sequential* prescriptive model: a small tabular MDP over the count and a
sliver of ordered history, estimated from logged transition/reward counts, planned with
undiscounted policy iteration, and used two ways -- as a **transparent simulator** and as a
**policy source** whose softened target is scored through the shared OPE gate
(:mod:`pitchseq.eval.ope`). It is the first rung that can value a *setup* pitch: an action
whose payoff is not the current pitch's reward but the *state* it creates for the next pitch
(decisions D40-D43).

The state-space ladder (decision D43)
=====================================
Three nested tabular state designs mirror the C / L1 / O measurement ladder in state space:

* ``"count"``               -- ``(balls, strikes)`` only (12 cells)                -- the ``C`` analogue;
* ``"count_prev"``          -- ``x`` previous family (9, incl. ``NONE``) = 108      -- the ``L1`` analogue;
* ``"count_prev_trigger"``  -- ``x`` a velo-gap **trigger flag** (2) = 216          -- an ``O``-lite analogue,

each plus four absorbing terminals ``{walk, strikeout, hbp, in_play_end}``. The **trigger
flag** for the current decision is ``1`` iff both prior pitches exist and
``|exec_release_speed_{t-1} - exec_release_speed_{t-2}| >= threshold`` (config-echoed
``5.0``) -- computed from **prior** pitches only, so it is leakage-safe (SPEC ``0``). It is
exactly the planted positive-world mechanism (:mod:`pitchseq.synth`): a large ordered velo
transition *into* the previous pitch. The flag makes the setup effect *representable* -- from a
state with previous family ``p``, choosing a current family ``a`` whose velo band differs from
``p``'s by ``>= threshold`` drives the **next** state's trigger flag to 1, and a triggered
state carries a higher reward (the planted whiff boost). Policy iteration therefore values
actions for the trigger they *create*, which the count / count+prev designs cannot express.

Why an undiscounted MDP (``gamma = 1``)
=======================================
The episode is the plate appearance. Its return is the undiscounted sum of per-pitch rewards
``sum_t R_t`` -- by SPEC ``5`` exactly the total run-expectancy change across the PA, and the
quantity the OPE estimators evaluate. Discounting would distort that run-value semantics
(a strike-three worth less than a strike-one), so ``gamma = 1`` is the correct, not merely
convenient, choice. Every PA terminates with probability one (absorbing terminals), so the
undiscounted values are finite and :func:`policy_iteration` solves them exactly by the linear
system ``(I - P_pi) V = R_pi`` on the non-terminal block (terminals pinned at ``V = 0``).

The estimators (documented formulas below on each function)
===========================================================
* **transitions** ``P_hat(s'|s,a)`` -- Dirichlet-smoothed transition counts over the
  per-source-state *reachable* set (:func:`estimate_mdp`);
* **rewards** ``R_hat(s,a)`` -- mean reward shrunk toward the state mean then the global mean
  (two-level, :func:`estimate_mdp`);
* **feasibility** -- the SPEC ``4`` per-row masks aggregated to a per-state feasible action set
  by a share rule (:func:`estimate_mdp`);
* **behavior** ``mu_hat(a|s)`` -- Dirichlet-smoothed action frequencies per state
  (:func:`estimate_behavior_policy`), the fallback logging policy when WS3 propensities are absent.

:mod:`workstreams.ws5_tabular_mdp.run_ws5` wires these into the checkpointed pipeline, the D42
model-based-vs-OPE cross-check and the D43 ladder verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitchseq.config import load_config
from pitchseq.families import FAMILIES, OTHER_FAMILY

__all__ = [
    "FAMILIES",
    "STATE_DESIGNS",
    "RICHEST_DESIGN",
    "TERMINALS",
    "PREV_NONE",
    "XX_INDEX",
    "DEFAULT_VELO_GAP_THRESHOLD",
    "DEFAULT_ALPHA_T",
    "DEFAULT_ALPHA_R",
    "DEFAULT_ALPHA_B",
    "DEFAULT_FEAS_STATE_SHARE",
    "N_COUNT",
    "N_PREV",
    "StateSpace",
    "TabularMDP",
    "count_index",
    "terminal_type",
    "prev_and_trigger",
    "encode_states",
    "family_velo_bands",
    "estimate_mdp",
    "estimate_behavior_policy",
    "policy_evaluation",
    "policy_iteration",
    "greedy_policy_matrix",
    "soften_policy",
    "simulate",
    "onehot_tabular_regressor",
    "policy_to_row_probs",
    "setup_diagnostics",
]

#: The D43 state-space ladder, coarsest to richest.
STATE_DESIGNS: tuple[str, ...] = ("count", "count_prev", "count_prev_trigger")

#: The richest design -- its behavior model / states form the common OPE evaluator (like WS4's
#: COMMON_EVAL_VIEW) so the trigger-vs-count value gap isolates the policy's information.
RICHEST_DESIGN = "count_prev_trigger"

#: Absorbing terminal states, appended after the non-terminal states of every design (fixed order).
TERMINALS: tuple[str, ...] = ("walk", "strikeout", "hbp", "in_play_end")

#: Previous-family sentinel for the first pitch of a PA (no prior pitch).
PREV_NONE = "NONE"

#: Index of the descriptive-only ``XX`` family (never a feasible action; SPEC ``4``).
XX_INDEX = FAMILIES.index(OTHER_FAMILY)

#: Velo-gap threshold (mph) for the trigger flag; echoes :mod:`pitchseq.synth`'s planted value.
DEFAULT_VELO_GAP_THRESHOLD = 5.0

#: Dirichlet smoothing for the transition counts (pseudo-count per reachable next-state).
DEFAULT_ALPHA_T = 1.0

#: Shrinkage strength (pseudo-count) for the two-level reward table (cell -> state -> global).
DEFAULT_ALPHA_R = 5.0

#: Dirichlet smoothing for the empirical behavior policy ``mu_hat(a|s)`` (pseudo-count per action).
DEFAULT_ALPHA_B = 0.5

#: A state's feasible action set keeps a family iff it is SPEC-4-feasible for at least this share
#: of the decision rows mapped to the state (majority rule; see :func:`estimate_mdp`).
DEFAULT_FEAS_STATE_SHARE = 0.5

#: 12 count cells: balls in 0..3, strikes in 0..2.
N_COUNT = 12
#: 9 previous-family slots: the 8 :data:`FAMILIES` plus ``NONE``.
N_PREV = len(FAMILIES) + 1
_N_FAM = len(FAMILIES)
_FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}
_PREV_NONE_IDX = _N_FAM  # NONE occupies the slot after the 8 families.

# Non-terminal state counts per design (terminals are appended after these).
_N_NONTERMINAL = {
    "count": N_COUNT,
    "count_prev": N_COUNT * N_PREV,
    "count_prev_trigger": N_COUNT * N_PREV * 2,
}


# -------------------------------------------------------------------------------------
# state indexing
# -------------------------------------------------------------------------------------


def count_index(balls, strikes) -> np.ndarray:
    """Count-cell index ``balls * 3 + strikes`` (0..11) for pre-pitch counts.

    Parameters
    ----------
    balls, strikes : array-like of int
        Pre-pitch balls (0..3) and strikes (0..2). A pre-pitch count never reaches 4 balls /
        3 strikes (those terminate the PA), which this asserts.

    Returns
    -------
    numpy.ndarray
        Integer count-cell index per element.
    """
    b = np.asarray(balls, dtype=np.int64)
    s = np.asarray(strikes, dtype=np.int64)
    if (b < 0).any() or (b > 3).any() or (s < 0).any() or (s > 2).any():
        raise ValueError("pre-pitch count out of range (need balls in 0..3, strikes in 0..2)")
    return b * 3 + s


def _prev_idx(prev_family) -> np.ndarray:
    """Previous-family slot index: ``FAMILIES`` index for a family, :data:`_PREV_NONE_IDX` for NONE."""
    out = np.empty(len(prev_family), dtype=np.int64)
    for i, f in enumerate(np.asarray(prev_family, dtype=object)):
        out[i] = _PREV_NONE_IDX if (f is None or f == PREV_NONE) else _FAMILY_TO_IDX.get(str(f), XX_INDEX)
    return out


def _nonterminal_id(design: str, count_id, prev_idx, trigger) -> np.ndarray:
    r"""Vectorised non-terminal state id for a design.

    ``count`` -> ``count_id``; ``count_prev`` -> ``count_id * 9 + prev_idx``;
    ``count_prev_trigger`` -> ``(count_id * 9 + prev_idx) * 2 + trigger``.
    """
    count_id = np.asarray(count_id, dtype=np.int64)
    if design == "count":
        return count_id.copy()
    prev_idx = np.asarray(prev_idx, dtype=np.int64)
    base = count_id * N_PREV + prev_idx
    if design == "count_prev":
        return base
    if design == "count_prev_trigger":
        return base * 2 + np.asarray(trigger, dtype=np.int64)
    raise ValueError(f"unknown design {design!r}; choose from {STATE_DESIGNS}")


def terminal_type(outcome1, balls, strikes) -> str | None:
    """Map a pitch outcome + pre-count to its absorbing terminal, or ``None`` if non-terminal.

    A PA ends on: an in-play ball (``in_play_end``); a hit-by-pitch (``hbp``); ball four
    (``ball`` at ``balls == 3`` -> ``walk``); strike three (``called_strike`` / ``whiff`` at
    ``strikes == 2`` -> ``strikeout``). Fouls never terminate; a non-terminal outcome returns
    ``None``.

    Parameters
    ----------
    outcome1 : str
        Level-1 outcome (``ball``/``called_strike``/``whiff``/``foul``/``hbp``/``in_play``).
    balls, strikes : int
        Pre-pitch count.

    Returns
    -------
    str or None
        One of :data:`TERMINALS`, or ``None`` when the PA continues.
    """
    o = str(outcome1)
    if o == "in_play":
        return "in_play_end"
    if o == "hbp":
        return "hbp"
    if o == "ball":
        return "walk" if int(balls) >= 3 else None
    if o in ("called_strike", "whiff"):
        return "strikeout" if int(strikes) >= 2 else None
    return None  # foul or anything else: not terminal


def prev_and_trigger(
    table: pd.DataFrame, threshold: float = DEFAULT_VELO_GAP_THRESHOLD
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Leakage-safe previous family and velo-gap trigger flag per decision row.

    Within each PA (``pa_id``, ordered by ``pitch_number``) the previous family is the family
    of pitch ``t-1`` (``NONE`` on the first pitch) and the trigger flag is

    .. math::
       \mathrm{trigger}_t = \mathbf{1}\!\left[\,t \ge 3 \ \wedge\
         \bigl|\,\mathtt{exec\_release\_speed}_{t-1} - \mathtt{exec\_release\_speed}_{t-2}\,\bigr|
         \ge \tau\,\right],

    with threshold :math:`\tau` (``threshold``). Only **prior** pitches' execution speeds enter
    (a shift of 1 and 2 within the PA), so no current-pitch execution leaks (SPEC ``0``). Both
    priors must exist, so the flag is 0 for the first two pitches of every PA.

    Parameters
    ----------
    table : pandas.DataFrame
        Decision rows with ``pa_id``, ``pitch_number``, ``family`` and ``exec_release_speed``.
    threshold : float, optional
        Velo-gap threshold in mph (default :data:`DEFAULT_VELO_GAP_THRESHOLD`).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        ``prev_family`` (object, ``NONE`` when absent), ``trigger`` (int 0/1) and
        ``has_two_priors`` (bool), all aligned to ``table`` row order.
    """
    # Stable sort by (pa_id, pitch_number) via lexsort (last key is primary), keep an inverse to
    # restore the original row order after the within-PA shifts.
    key = np.lexsort((table["pitch_number"].to_numpy(), table["pa_id"].to_numpy()))
    inv = np.empty_like(key)
    inv[key] = np.arange(len(key))

    srt = table.iloc[key]
    g = srt.groupby("pa_id", sort=False)
    prev_fam = g["family"].shift(1).astype(object)
    v1 = g["exec_release_speed"].shift(1)
    v2 = g["exec_release_speed"].shift(2)
    has2 = v1.notna() & v2.notna()
    trig = (((v1 - v2).abs() >= float(threshold)) & has2).astype(np.int64)
    prev_fam = prev_fam.where(prev_fam.notna(), PREV_NONE)

    # Restore original row order.
    prev_out = prev_fam.to_numpy()[inv]
    trig_out = trig.to_numpy()[inv]
    has2_out = has2.to_numpy()[inv]
    return prev_out.astype(object), trig_out.astype(np.int64), has2_out.astype(bool)


@dataclass
class StateSpace:
    """Encoded state ids and metadata for one design (return of :func:`encode_states`).

    Attributes
    ----------
    design : str
        One of :data:`STATE_DESIGNS`.
    state_ids : numpy.ndarray
        Current non-terminal state id per decision row (aligned to the input table).
    n_states : int
        Total states, non-terminal + 4 terminals.
    n_nonterminal : int
        Non-terminal state count (terminals are ids ``n_nonterminal .. n_nonterminal + 3``).
    terminal_ids : dict
        ``name -> id`` for each of :data:`TERMINALS`.
    count_id, prev_idx, trigger : numpy.ndarray
        Per-row state components (``prev_idx`` / ``trigger`` are still computed for every design
        so diagnostics can use them even when the design ignores them).
    prev_family, has_two_priors : numpy.ndarray
        Per-row previous family (object) and the two-priors-exist flag.
    threshold : float
        The velo-gap threshold used for ``trigger``.
    """

    design: str
    state_ids: np.ndarray
    n_states: int
    n_nonterminal: int
    terminal_ids: dict
    count_id: np.ndarray
    prev_idx: np.ndarray
    trigger: np.ndarray
    prev_family: np.ndarray
    has_two_priors: np.ndarray
    threshold: float

    def terminal_mask(self) -> np.ndarray:
        """Boolean length-``n_states`` mask, ``True`` on the four terminal ids."""
        m = np.zeros(self.n_states, dtype=bool)
        for tid in self.terminal_ids.values():
            m[tid] = True
        return m

    def describe(self, sid: int) -> str:
        """Human-readable label for a state id (for diagnostics / tests)."""
        for name, tid in self.terminal_ids.items():
            if sid == tid:
                return f"TERM:{name}"
        if self.design == "count":
            b, s = divmod(sid, 3)
            return f"{b}-{s}"
        if self.design == "count_prev":
            cid, pidx = divmod(sid, N_PREV)
            b, s = divmod(cid, 3)
            pf = PREV_NONE if pidx == _PREV_NONE_IDX else FAMILIES[pidx]
            return f"{b}-{s}|prev={pf}"
        cid_p, trig = divmod(sid, 2)
        cid, pidx = divmod(cid_p, N_PREV)
        b, s = divmod(cid, 3)
        pf = PREV_NONE if pidx == _PREV_NONE_IDX else FAMILIES[pidx]
        return f"{b}-{s}|prev={pf}|trig={trig}"


def encode_states(
    table: pd.DataFrame, design: str, threshold: float = DEFAULT_VELO_GAP_THRESHOLD
) -> StateSpace:
    """Encode each decision row's current (non-terminal) state id for a design (decision D43).

    See the module docstring for the id scheme. Terminal ids are appended after the
    non-terminal block: ``walk, strikeout, hbp, in_play_end`` at
    ``n_nonterminal + 0..3``.

    Parameters
    ----------
    table : pandas.DataFrame
        Decision rows with ``balls``, ``strikes``, ``pa_id``, ``pitch_number``, ``family`` and
        ``exec_release_speed``.
    design : str
        One of :data:`STATE_DESIGNS`.
    threshold : float, optional
        Velo-gap threshold for the trigger flag.

    Returns
    -------
    StateSpace
        Per-row state ids and the metadata needed to build and read the MDP.
    """
    if design not in STATE_DESIGNS:
        raise ValueError(f"unknown design {design!r}; choose from {STATE_DESIGNS}")
    count_id = count_index(table["balls"].to_numpy(), table["strikes"].to_numpy())
    prev_family, trigger, has2 = prev_and_trigger(table, threshold)
    prev_idx = _prev_idx(prev_family)
    state_ids = _nonterminal_id(design, count_id, prev_idx, trigger)
    n_nt = _N_NONTERMINAL[design]
    terminal_ids = {name: n_nt + i for i, name in enumerate(TERMINALS)}
    return StateSpace(
        design=design,
        state_ids=state_ids.astype(np.int64),
        n_states=n_nt + len(TERMINALS),
        n_nonterminal=n_nt,
        terminal_ids=terminal_ids,
        count_id=count_id,
        prev_idx=prev_idx,
        trigger=trigger,
        prev_family=prev_family,
        has_two_priors=has2,
        threshold=float(threshold),
    )


# -------------------------------------------------------------------------------------
# MDP estimation
# -------------------------------------------------------------------------------------


def family_velo_bands(table: pd.DataFrame) -> np.ndarray:
    """Empirical mean ``exec_release_speed`` per family (the velo band), :data:`FAMILIES` order.

    Used to classify a (previous family, action) pair as *trigger-creating* when the band gap
    is ``>= threshold`` (the representability of the setup effect). Unseen families get ``NaN``.

    Returns
    -------
    numpy.ndarray, shape (8,)
        Mean release speed per family, ``NaN`` where a family never appears.
    """
    fam = table["family"].astype(object).to_numpy()
    velo = pd.to_numeric(table["exec_release_speed"], errors="coerce").to_numpy()
    bands = np.full(_N_FAM, np.nan, dtype=np.float64)
    for j, f in enumerate(FAMILIES):
        m = (fam == f) & np.isfinite(velo)
        if m.any():
            bands[j] = float(velo[m].mean())
    return bands


def _next_state_ids(table: pd.DataFrame, ss: StateSpace) -> tuple[np.ndarray, np.ndarray]:
    """Successor state id and a validity flag per decision row (for transition counts).

    Within a PA (ordered by ``pitch_number``), a non-last pitch's successor is the next pitch's
    encoded state; the last pitch's successor is the terminal implied by
    :func:`terminal_type`. A last pitch whose outcome is non-terminal (a truncated PA on real
    data -- never in the completed synthetic PAs) is routed to ``in_play_end`` and marked so it
    can be excluded from reachability if desired.
    """
    key = np.lexsort((table["pitch_number"].to_numpy(), table["pa_id"].to_numpy()))
    inv = np.empty_like(key)
    inv[key] = np.arange(len(key))
    srt = table.iloc[key].reset_index(drop=True)
    cur_sorted = ss.state_ids[key]

    pa = srt["pa_id"].to_numpy()
    is_last = np.ones(len(srt), dtype=bool)
    is_last[:-1] = pa[1:] != pa[:-1]  # last row of each pa_id block

    nxt = np.empty(len(srt), dtype=np.int64)
    valid = np.ones(len(srt), dtype=bool)
    # Non-last: successor is the next sorted row's state.
    nxt[:-1] = np.where(is_last[:-1], -1, cur_sorted[1:])
    nxt[-1] = -1
    # Last rows: map to a terminal.
    o1 = srt["outcome1"].astype(object).to_numpy()
    balls = srt["balls"].to_numpy()
    strikes = srt["strikes"].to_numpy()
    last_rows = np.flatnonzero(is_last)
    for r in last_rows:
        tt = terminal_type(o1[r], balls[r], strikes[r])
        if tt is None:  # truncated PA (real-data guard); route to in_play_end.
            nxt[r] = ss.terminal_ids["in_play_end"]
            valid[r] = False
        else:
            nxt[r] = ss.terminal_ids[tt]
    return nxt[inv], valid[inv]


@dataclass
class TabularMDP:
    """An estimated tabular MDP for one design (return of :func:`estimate_mdp`).

    Attributes
    ----------
    design : str
        The state design.
    P : numpy.ndarray, shape (S, A, S)
        Smoothed transition kernel ``P_hat(s' | s, a)``.
    R : numpy.ndarray, shape (S, A)
        Two-level-shrunk reward table ``R_hat(s, a)`` (terminals are 0).
    feasible : numpy.ndarray of bool, shape (S, A)
        Per-state feasible action set (``XX`` never feasible; terminals all-False).
    support : numpy.ndarray, shape (S, A)
        Transition support count ``N(s, a)`` (rows observed in that state-action).
    n_states, n_actions, n_nonterminal : int
    terminal_ids : dict
    start_dist : numpy.ndarray, shape (S,)
        Empirical initial-state distribution (the first pitch of each PA).
    velo_bands : numpy.ndarray, shape (8,)
        Per-family mean release speed (:func:`family_velo_bands`).
    state_space : StateSpace
        The encoding used (carries ``describe`` and the terminal mask).
    alpha_t, alpha_r : float
        The smoothing strengths used.
    """

    design: str
    P: np.ndarray
    R: np.ndarray
    feasible: np.ndarray
    support: np.ndarray
    n_states: int
    n_actions: int
    n_nonterminal: int
    terminal_ids: dict
    start_dist: np.ndarray
    velo_bands: np.ndarray
    state_space: StateSpace
    alpha_t: float
    alpha_r: float

    def terminal_mask(self) -> np.ndarray:
        return self.state_space.terminal_mask()


def _aggregate_feasibility(
    state_ids: np.ndarray,
    feasible_rows: np.ndarray | None,
    n_states: int,
    n_actions: int,
    support: np.ndarray,
    terminal_mask: np.ndarray,
    share: float,
) -> np.ndarray:
    r"""Aggregate per-row SPEC-4 masks to a per-state feasible action set (majority-share rule).

    A family ``a`` is feasible in state ``s`` iff it is feasible for at least ``share`` of the
    decision rows mapped to ``s``. If that leaves a (reachable) state with an empty set, it falls
    back to the actions actually **observed** in the state (``support > 0``); a state with no
    observations at all falls back to all non-``XX`` families. ``XX`` is never feasible (SPEC
    ``4``); terminals have no feasible actions.
    """
    feas = np.zeros((n_states, n_actions), dtype=bool)
    if feasible_rows is None:
        feas[:] = True
        feas[:, XX_INDEX] = False
    else:
        num = np.zeros((n_states, n_actions), dtype=np.float64)
        den = np.zeros(n_states, dtype=np.float64)
        np.add.at(num, state_ids, feasible_rows.astype(np.float64))
        np.add.at(den, state_ids, 1.0)
        with np.errstate(invalid="ignore"):
            frac = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0)
        feas = frac >= float(share)
        feas[:, XX_INDEX] = False
        # Fallback for reachable states left with an empty set.
        empty = (~feas.any(axis=1)) & (support.sum(axis=1) > 0)
        obs = support > 0
        obs[:, XX_INDEX] = False
        feas[empty] = obs[empty]
    # Unobserved non-terminal states: allow all non-XX families (well-definedness only).
    unobserved = (support.sum(axis=1) == 0) & (~terminal_mask)
    feas[unobserved] = True
    feas[unobserved, XX_INDEX] = False
    feas[terminal_mask] = False
    return feas


def estimate_mdp(
    table: pd.DataFrame,
    design: str,
    alpha_t: float = DEFAULT_ALPHA_T,
    alpha_r: float = DEFAULT_ALPHA_R,
    threshold: float = DEFAULT_VELO_GAP_THRESHOLD,
    feasible_rows: np.ndarray | None = None,
    feas_state_share: float = DEFAULT_FEAS_STATE_SHARE,
    reward_col: str = "R",
    action_col: str = "family",
) -> TabularMDP:
    r"""Estimate the tabular MDP for a design from logged transition/reward counts.

    **Transitions.** With counts ``N(s,a,s')`` and the per-source-state *reachable* set
    ``succ(s) = { s' : N(s,\cdot,s') > 0 }`` (of size ``K_s``),

    .. math::
       \hat P(s'\!\mid s,a) = \frac{N(s,a,s') + \alpha_t}{N(s,a) + \alpha_t\,K_s},
       \qquad s' \in succ(s),

    (Dirichlet smoothing with pseudo-count ``alpha_t`` over the reachable set; zero elsewhere).
    A state-action never taken (``N(s,a)=0``) **backs off to the state's action-marginal**
    transition ``N(s,\cdot,s')``; an unobserved state routes deterministically to ``in_play_end``
    (unreachable in practice). Terminals self-loop. The dominant mass comes from the observed
    ``N(s,a,s')`` (which carry the exact ``prev' = a`` and the velo-gap ``trigger'``), so the
    setup mechanism is preserved; ``alpha_t`` only regularises thin cells.

    **Rewards (two-level shrinkage).** With global mean ``r_bar``, state mean shrunk toward it,
    then the cell mean shrunk toward the state mean:

    .. math::
       \hat r(s) &= \frac{\sum_{i:s_i=s} R_i + \alpha_r\,\bar r}{n_s + \alpha_r}, \\
       \hat R(s,a) &= \frac{\sum_{i:(s_i,a_i)=(s,a)} R_i + \alpha_r\,\hat r(s)}{n_{s,a} + \alpha_r}.

    Thin ``(s,a)`` cells fall back smoothly to the state mean, then the global mean (no NaNs).
    Terminals carry ``R = 0`` (absorbing, no further reward).

    **Feasibility.** The per-row SPEC ``4`` masks (``feasible_rows``) are aggregated to a
    per-state action set by the majority-share rule (:func:`_aggregate_feasibility`,
    ``feas_state_share``); ``feasible_rows=None`` makes every non-``XX`` family feasible.

    Parameters
    ----------
    table : pandas.DataFrame
        Training decision rows (must carry the columns :func:`encode_states` needs plus
        ``outcome1`` and ``reward_col``).
    design : str
        One of :data:`STATE_DESIGNS`.
    alpha_t, alpha_r : float, optional
        Transition Dirichlet pseudo-count and reward shrinkage strength.
    threshold : float, optional
        Velo-gap threshold for the trigger flag.
    feasible_rows : numpy.ndarray of bool, shape (n, 8), optional
        Per-row SPEC ``4`` feasible mask (computed on the full table's trailing window and
        subset to ``table``). ``None`` -> all non-``XX`` families feasible.
    feas_state_share : float, optional
        Share threshold for the per-state feasibility aggregation.
    reward_col, action_col : str, optional
        Reward and action column names (defaults ``"R"`` / ``"family"``).

    Returns
    -------
    TabularMDP
        The estimated ``(P, R, feasible, support, start_dist, ...)`` bundle.
    """
    ss = encode_states(table, design, threshold)
    S, A = ss.n_states, _N_FAM
    states = ss.state_ids
    actions = np.array(
        [_FAMILY_TO_IDX.get(str(f), XX_INDEX) for f in table[action_col].astype(object).to_numpy()],
        dtype=np.int64,
    )
    rewards = pd.to_numeric(table[reward_col], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(rewards)
    next_ids, _valid = _next_state_ids(table, ss)

    # --- transition counts N(s, a, s') over finite-reward decisions ---
    N = np.zeros((S, A, S), dtype=np.float64)
    np.add.at(N, (states[finite], actions[finite], next_ids[finite]), 1.0)
    N_sa = N.sum(axis=2)  # (S, A)
    N_marg = N.sum(axis=1)  # (S, S): state action-marginal successors

    P = np.zeros((S, A, S), dtype=np.float64)
    term_mask = ss.terminal_mask()
    for s in range(S):
        if term_mask[s]:
            P[s, :, s] = 1.0  # absorbing self-loop
            continue
        succ = np.flatnonzero(N_marg[s] > 0)
        if succ.size == 0:  # unobserved state: route to in_play_end (well-definedness only)
            P[s, :, ss.terminal_ids["in_play_end"]] = 1.0
            continue
        K = succ.size
        for a in range(A):
            if N_sa[s, a] > 0:
                num = N[s, a, succ] + alpha_t
                P[s, a, succ] = num / (N_sa[s, a] + alpha_t * K)
            else:  # unobserved action: back off to the state action-marginal
                num = N_marg[s, succ] + alpha_t
                P[s, a, succ] = num / (N_marg[s].sum() + alpha_t * K)

    # --- two-level reward shrinkage ---
    r_global = float(rewards[finite].mean()) if finite.any() else 0.0
    sum_s = np.zeros(S, dtype=np.float64)
    cnt_s = np.zeros(S, dtype=np.float64)
    np.add.at(sum_s, states[finite], rewards[finite])
    np.add.at(cnt_s, states[finite], 1.0)
    r_state = (sum_s + alpha_r * r_global) / (cnt_s + alpha_r)
    sum_sa = np.zeros((S, A), dtype=np.float64)
    cnt_sa = np.zeros((S, A), dtype=np.float64)
    np.add.at(sum_sa, (states[finite], actions[finite]), rewards[finite])
    np.add.at(cnt_sa, (states[finite], actions[finite]), 1.0)
    R = (sum_sa + alpha_r * r_state[:, None]) / (cnt_sa + alpha_r)
    R[term_mask, :] = 0.0

    # --- feasibility ---
    feasible = _aggregate_feasibility(states, feasible_rows, S, A, N_sa, term_mask, feas_state_share)

    # --- empirical start distribution (first pitch of each PA) ---
    first_mask = table["pitch_number"].to_numpy() == table.groupby("pa_id")["pitch_number"].transform("min").to_numpy()
    start = np.zeros(S, dtype=np.float64)
    np.add.at(start, states[first_mask], 1.0)
    start = start / start.sum() if start.sum() > 0 else start

    return TabularMDP(
        design=design,
        P=P,
        R=R,
        feasible=feasible,
        support=N_sa,
        n_states=S,
        n_actions=A,
        n_nonterminal=ss.n_nonterminal,
        terminal_ids=ss.terminal_ids,
        start_dist=start,
        velo_bands=family_velo_bands(table),
        state_space=ss,
        alpha_t=float(alpha_t),
        alpha_r=float(alpha_r),
    )


def estimate_behavior_policy(
    table: pd.DataFrame,
    ss: StateSpace,
    alpha_b: float = DEFAULT_ALPHA_B,
    action_col: str = "family",
) -> np.ndarray:
    r"""Dirichlet-smoothed empirical behavior policy ``mu_hat(a | s)`` per state (fallback logging policy).

    .. math:: \hat\mu(a\mid s) = \frac{N(s,a) + \alpha_b}{N(s) + \alpha_b\,A}

    (full support, so importance ratios stay finite; SPEC ``9``). This is the mode used when no
    WS3 propensities are supplied -- a state-conditional empirical logging policy from train
    counts; it is coarser than WS3's contextual GBDT ``mu`` (which the pipeline prefers when a
    ``--ws3-dir`` is given).

    Parameters
    ----------
    table : pandas.DataFrame
        Rows to count (typically the training rows).
    ss : StateSpace
        The encoding for ``table`` (its ``state_ids`` align to ``table``).
    alpha_b : float, optional
        Dirichlet pseudo-count per action.
    action_col : str, optional
        Action column (default ``"family"``).

    Returns
    -------
    numpy.ndarray, shape (n_states, 8)
        ``mu_hat(a | s)`` per state (rows for unobserved states are uniform over the 8 families).
    """
    S, A = ss.n_states, _N_FAM
    states = ss.state_ids
    actions = np.array(
        [_FAMILY_TO_IDX.get(str(f), XX_INDEX) for f in table[action_col].astype(object).to_numpy()],
        dtype=np.int64,
    )
    N = np.zeros((S, A), dtype=np.float64)
    np.add.at(N, (states, actions), 1.0)
    return (N + alpha_b) / (N.sum(axis=1, keepdims=True) + alpha_b * A)


# -------------------------------------------------------------------------------------
# planning (undiscounted policy iteration; gamma = 1)
# -------------------------------------------------------------------------------------


def _values_for_policy(
    P: np.ndarray, R: np.ndarray, pi: np.ndarray, terminal_mask: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Exact ``(V, Q)`` for a stochastic policy ``pi`` by solving ``(I - gamma P_pi) V = R_pi``.

    Terminals are pinned at ``V = 0`` and excluded from the linear system (their reward is 0 and
    they self-loop). The solve is on the non-terminal block only; a proper policy (every PA
    terminates w.p. 1) makes ``I - gamma P_pi`` non-singular, so the undiscounted ``gamma = 1``
    values are finite and exact. A singular / non-finite solve (an improper policy, e.g. an
    all-foul cycle that never ends) falls back to iterative fixed-point evaluation (unreachable
    in practice: the smoothed kernel plus complete-PA training data makes every policy proper).
    """
    S = P.shape[0]
    nt = np.flatnonzero(~terminal_mask)
    R_pi = (pi * R).sum(axis=1)  # (S,)
    P_pi = np.einsum("sa,sat->st", pi, P)  # (S, S)
    A_mat = np.eye(len(nt)) - gamma * P_pi[np.ix_(nt, nt)]
    b = R_pi[nt]
    V = np.zeros(S, dtype=np.float64)
    try:
        V[nt] = np.linalg.solve(A_mat, b)
        if not np.isfinite(V[nt]).all():
            raise np.linalg.LinAlgError("non-finite solve")
    except np.linalg.LinAlgError:  # improper policy: iterate to the fixed point
        v = np.zeros(S, dtype=np.float64)
        for _ in range(20000):
            v_new = R_pi + gamma * (P_pi @ v)
            v_new[terminal_mask] = 0.0
            if np.max(np.abs(v_new - v)) < 1e-12:
                v = v_new
                break
            v = v_new
        V = v
    Q = R + gamma * np.einsum("sat,t->sa", P, V)
    return V, Q


def policy_evaluation(
    P: np.ndarray, R: np.ndarray, policy_probs: np.ndarray, gamma: float = 1.0, terminal_mask=None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Exact value and Q-function of a (possibly stochastic) policy in the tabular MDP.

    Solves the Bellman equation ``V = R_pi + gamma P_pi V`` on the non-terminal states (SPEC
    ``5`` undiscounted return; see the module docstring). This is the "return-to-go under the
    target" the doubly-robust / step-wise-DR estimators want as ``q_hat``.

    Parameters
    ----------
    P : numpy.ndarray, shape (S, A, S)
        Transition kernel.
    R : numpy.ndarray, shape (S, A)
        Reward table.
    policy_probs : numpy.ndarray, shape (S, A)
        Per-state action distribution.
    gamma : float, optional
        Discount (default 1.0; see the module docstring on why undiscounted is correct).
    terminal_mask : numpy.ndarray of bool, shape (S,), optional
        Terminal states (pinned at ``V = 0``). When ``None`` the terminals are inferred as the
        absorbing self-loop states (``P[s, :, s] == 1``).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``V`` (shape ``(S,)``) and ``Q`` (shape ``(S, A)``).
    """
    if terminal_mask is None:
        terminal_mask = np.isclose(P[np.arange(P.shape[0]), 0, np.arange(P.shape[0])], 1.0)
    pi = np.asarray(policy_probs, dtype=np.float64)
    return _values_for_policy(np.asarray(P, np.float64), np.asarray(R, np.float64), pi, terminal_mask, gamma)


def _greedy_indices(Q: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Feasible arg-max action per state (ties broken by lowest index; all-infeasible -> 0)."""
    masked = np.where(feasible, Q, -np.inf)
    idx = masked.argmax(axis=1)
    idx[~np.isfinite(masked.max(axis=1))] = 0  # no feasible action -> placeholder (terminals)
    return idx.astype(np.int64)


def greedy_policy_matrix(policy_idx: np.ndarray, n_actions: int) -> np.ndarray:
    """One-hot ``(n_states, n_actions)`` policy matrix from per-state action indices."""
    P = np.zeros((len(policy_idx), n_actions), dtype=np.float64)
    P[np.arange(len(policy_idx)), np.asarray(policy_idx, dtype=np.int64)] = 1.0
    return P


def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    feasible: np.ndarray,
    gamma: float = 1.0,
    terminal_mask=None,
    max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Undiscounted policy iteration to the optimal deterministic feasible policy (decision D41).

    Alternates exact policy evaluation (:func:`policy_evaluation`) and greedy improvement
    restricted to each state's feasible actions until the policy is stable (guaranteed in
    finitely many steps for a finite MDP; ``max_iter`` guards against a non-converging estimate).
    ``gamma = 1`` is the correct undiscounted PA return (module docstring).

    Parameters
    ----------
    P : numpy.ndarray, shape (S, A, S)
    R : numpy.ndarray, shape (S, A)
    feasible : numpy.ndarray of bool, shape (S, A)
        Per-state feasible action set (the improvement's arg-max is restricted to it).
    gamma : float, optional
        Discount (default 1.0).
    terminal_mask : numpy.ndarray of bool, shape (S,), optional
        Terminal states; inferred from the absorbing self-loops when ``None``.
    max_iter : int, optional
        Maximum improvement sweeps (a convergence guard; raises if exceeded).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        ``policy`` (per-state action index, shape ``(S,)``), ``Q`` (shape ``(S, A)``) and ``V``
        (shape ``(S,)``) of the optimal policy.
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    S, A = R.shape
    if terminal_mask is None:
        terminal_mask = np.isclose(P[np.arange(S), 0, np.arange(S)], 1.0)
    # A feasible-everywhere fallback keeps the greedy arg-max well-defined for any reachable state.
    feas = feasible.copy()
    no_feas = (~feas.any(axis=1)) & (~terminal_mask)
    feas[no_feas] = True
    if A == _N_FAM:  # on the family action space keep XX out (SPEC 4); generic MDPs are untouched
        feas[no_feas, XX_INDEX] = False

    policy = _greedy_indices(R, feas)  # init: myopic greedy on immediate reward
    V = np.zeros(S)
    Q = np.zeros((S, A))
    for _ in range(max_iter):
        pi = greedy_policy_matrix(policy, A)
        V, Q = _values_for_policy(P, R, pi, terminal_mask, gamma)
        new_policy = _greedy_indices(Q, feas)
        if np.array_equal(new_policy, policy):
            return new_policy, Q, V
        policy = new_policy
    raise RuntimeError(f"policy_iteration did not converge within {max_iter} sweeps")


def soften_policy(
    q_table: np.ndarray, feasible: np.ndarray, temperature: float = 0.02
) -> np.ndarray:
    r"""Temperature-softened policy over feasible actions -- a masked Boltzmann distribution.

    .. math:: \pi(a\mid s) \propto \exp\!\big(Q(s,a)/T\big)\ \mathbf{1}[a \text{ feasible}]

    ``T -> 0`` recovers the greedy arg-max; large ``T`` approaches uniform over feasible actions.
    This is the *temperature* softening variant of the conservative target; the pipeline's
    **primary** target is instead the SPEC ``9`` mixture ``pi_alpha = (1-alpha) mu + alpha
    pi_greedy`` (built with :func:`pitchseq.eval.ope.pi_alpha` at the row level), documented in
    :mod:`workstreams.ws5_tabular_mdp.run_ws5`. Infeasible actions get exactly zero mass; a state
    with no feasible action yields a uniform row (unreachable / terminal placeholder).

    Parameters
    ----------
    q_table : numpy.ndarray, shape (S, A)
        Per-state action values.
    feasible : numpy.ndarray of bool, shape (S, A)
        Feasible actions per state.
    temperature : float, optional
        Boltzmann temperature ``T`` (default 0.02, near-greedy on the reward scale).

    Returns
    -------
    numpy.ndarray, shape (S, A)
        Per-state action distribution (rows sum to 1; zero on infeasible actions).
    """
    q = np.asarray(q_table, dtype=np.float64)
    feas = np.asarray(feasible, dtype=bool)
    T = float(temperature)
    if T <= 0:
        raise ValueError("temperature must be > 0 (use policy_iteration for the greedy limit)")
    out = np.full_like(q, 1.0 / q.shape[1])  # placeholder uniform for unreachable/terminal rows
    any_feas = feas.any(axis=1)
    if any_feas.any():
        scores = np.where(feas, q / T, -np.inf)
        scores = scores - np.where(any_feas[:, None], scores.max(axis=1, keepdims=True), 0.0)
        exp = np.where(feas, np.exp(scores), 0.0)
        denom = exp.sum(axis=1, keepdims=True)
        probs = np.divide(exp, denom, out=np.zeros_like(exp), where=denom > 0)
        out[any_feas] = probs[any_feas]
    return out


# -------------------------------------------------------------------------------------
# transparent simulator (decision D42)
# -------------------------------------------------------------------------------------


def simulate(
    P: np.ndarray,
    R: np.ndarray,
    policy_probs: np.ndarray,
    n_episodes: int,
    rng,
    start_dist: np.ndarray,
    terminal_mask=None,
    max_len: int = 200,
) -> tuple[float, list]:
    r"""Roll out a policy in the estimated MDP -- the transparent simulator (decision D42).

    Samples ``n_episodes`` episodes: draw ``s_0`` from ``start_dist``; at each step draw an
    action ``a ~ policy_probs[s]``, accrue reward ``R[s, a]``, and transition ``s' ~ P[s, a]``;
    stop at a terminal (absorbing self-loop) or after ``max_len`` steps. Returns the Monte-Carlo
    estimate of the undiscounted PA return and the per-episode traces. Deterministic for a fixed
    ``rng``. On a proper policy the estimate converges to :func:`policy_evaluation`'s exact
    ``sum_s start_dist[s] V(s)`` -- the simulator's self-consistency with the analytic value is
    the D42 validation.

    Parameters
    ----------
    P : numpy.ndarray, shape (S, A, S)
    R : numpy.ndarray, shape (S, A)
    policy_probs : numpy.ndarray, shape (S, A)
        Per-state action distribution to roll out.
    n_episodes : int
        Number of episodes to simulate.
    rng : numpy.random.Generator or int
        Generator / seed for reproducible rollouts.
    start_dist : numpy.ndarray, shape (S,)
        Initial-state distribution.
    terminal_mask : numpy.ndarray of bool, shape (S,), optional
        Terminal states; inferred from absorbing self-loops when ``None``.
    max_len : int, optional
        Maximum steps per episode (guards a rare non-terminating rollout).

    Returns
    -------
    (float, list)
        The mean simulated return and a list of ``(states, actions, rewards)`` tuples (each a
        list) per episode.
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    pi = np.asarray(policy_probs, dtype=np.float64)
    S, A = R.shape
    if terminal_mask is None:
        terminal_mask = np.isclose(P[np.arange(S), 0, np.arange(S)], 1.0)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    start = np.asarray(start_dist, dtype=np.float64)
    start = start / start.sum()

    returns = np.empty(n_episodes, dtype=np.float64)
    traces: list = []
    a_idx = np.arange(A)
    s_idx = np.arange(S)
    for e in range(n_episodes):
        s = int(rng.choice(s_idx, p=start))
        states_e, actions_e, rewards_e = [], [], []
        total = 0.0
        for _ in range(max_len):
            if terminal_mask[s]:
                break
            a = int(rng.choice(a_idx, p=pi[s]))
            r = float(R[s, a])
            states_e.append(s)
            actions_e.append(a)
            rewards_e.append(r)
            total += r
            s = int(rng.choice(s_idx, p=P[s, a]))
        returns[e] = total
        traces.append((states_e, actions_e, rewards_e))
    return float(returns.mean()), traces


# -------------------------------------------------------------------------------------
# FQE regressor callback (for the refit cluster bootstrap)
# -------------------------------------------------------------------------------------


def onehot_tabular_regressor(n_actions: int = _N_FAM):
    r"""Fast exact group-mean regressor for FQE's one-hot ``[state | action]`` features.

    :func:`pitchseq.eval.ope.fqe` builds its feature matrix as ``[one_hot(state, n_states),
    one_hot(action, n_actions)]`` and accepts any sklearn-style learner through its documented
    ``regressor_factory`` callback -- this factory supplies one (WS5 does **not** re-implement
    FQE). It is *numerically identical* to :func:`pitchseq.eval.ope.tabular_regressor` on those
    features -- the exact per-``(state, action)`` group mean, with unseen cells falling back to
    the global training mean -- but decodes the two one-hot blocks to integer codes and uses
    ``numpy.bincount`` instead of hashing every row, which is orders of magnitude faster. That
    speed is what makes the **refit cluster bootstrap** of the FQE values feasible (hundreds of
    re-fits per run; see ``run_ws5``).

    Parameters
    ----------
    n_actions : int, optional
        Width of the trailing action one-hot block (default the 8 families). The state block is
        everything before it, so the same factory serves any state-space size.

    Returns
    -------
    callable
        A zero-argument factory returning a fresh regressor with ``fit(X, y)`` / ``predict(X)``.
    """

    class _OneHotGroupMean:
        def __init__(self) -> None:
            self._means: np.ndarray | None = None
            self._global = 0.0

        def _codes(self, X) -> tuple[np.ndarray, int]:
            X = np.asarray(X)
            n_state_cols = X.shape[1] - n_actions
            s = X[:, :n_state_cols].argmax(axis=1)
            a = X[:, n_state_cols:].argmax(axis=1)
            return s * n_actions + a, n_state_cols

        def fit(self, X, y):
            y = np.asarray(y, dtype=np.float64)
            codes, n_state_cols = self._codes(X)
            size = max(int(n_state_cols) * n_actions, 1)
            counts = np.bincount(codes, minlength=size).astype(np.float64)
            sums = np.bincount(codes, weights=y, minlength=size)
            self._global = float(y.mean()) if len(y) else 0.0
            self._means = np.divide(
                sums, counts, out=np.full(size, self._global, dtype=np.float64), where=counts > 0
            )
            return self

        def predict(self, X):
            codes, _ = self._codes(X)
            return self._means[codes]

    return _OneHotGroupMean





def policy_to_row_probs(policy_probs: np.ndarray, state_ids: np.ndarray) -> np.ndarray:
    """Map a per-state policy to per-row target probabilities (align a policy to logged rows).

    Gathers ``policy_probs[state_ids[i]]`` for each decision row ``i`` -- the target-policy row
    the OPE gate consumes (:func:`pitchseq.eval.ope.evaluate_policy`). Works for the greedy
    one-hot policy, the temperature-softened policy or any per-state distribution.

    Parameters
    ----------
    policy_probs : numpy.ndarray, shape (n_states, 8)
        Per-state action distribution.
    state_ids : array-like of int, shape (n,)
        Each row's (non-terminal) state id.

    Returns
    -------
    numpy.ndarray, shape (n, 8)
        Per-row target probabilities.
    """
    return np.asarray(policy_probs, dtype=np.float64)[np.asarray(state_ids, dtype=np.int64)]


# -------------------------------------------------------------------------------------
# setup diagnostics (decision D43 -- "what did the trigger knowledge change")
# -------------------------------------------------------------------------------------


def setup_diagnostics(
    mdp_trigger: TabularMDP,
    mdp_count_prev: TabularMDP,
    threshold: float = DEFAULT_VELO_GAP_THRESHOLD,
) -> dict:
    r"""The D43 setup exhibits: setup Q-gaps and the optimal-action change from trigger knowledge.

    Two read-outs, both computed on the richest (``count_prev_trigger``) MDP:

    1. **Setup Q-gap.** For each *reachable, non-terminal* state, split the feasible actions into
       **trigger-creating** (velo band ``|v(a) - v(prev)| >= threshold``, so choosing ``a`` drives
       the next state's trigger flag to 1) and non-creating, and report
       ``max_a-creating Q(s,a) - max_a-noncreating Q(s,a)`` -- how much extra value the MDP
       assigns to *setting up* a trigger. A positive mean is the model valuing the setup; ``~0``
       (the null world) means the trigger is inert.
    2. **Optimal-action change.** Map each ``count_prev_trigger`` state to its ``count_prev``
       parent (drop the trigger flag) and report the share of (reachable) states whose optimal
       action differs from the parent design's optimal action -- what the trigger knowledge
       *changed* in the policy.

    Parameters
    ----------
    mdp_trigger : TabularMDP
        The ``count_prev_trigger`` MDP (with a planned policy available via
        :func:`policy_iteration`).
    mdp_count_prev : TabularMDP
        The ``count_prev`` MDP.
    threshold : float, optional
        Velo-gap threshold for classifying trigger-creating actions.

    Returns
    -------
    dict
        ``setup_gap_mean`` / ``setup_gap_median`` / ``setup_gap_positive_frac`` (over states with
        both an creating and a non-creating feasible action), ``n_states_with_gap``,
        ``optimal_action_change_frac`` and ``n_states_compared`` (reachable trigger states with a
        count_prev parent), plus ``mean_creating_actions`` (mean # trigger-creating feasible
        actions per state).
    """
    pol_t, Q_t, _ = policy_iteration(mdp_trigger.P, mdp_trigger.R, mdp_trigger.feasible)
    pol_p, _, _ = policy_iteration(mdp_count_prev.P, mdp_count_prev.R, mdp_count_prev.feasible)
    ss = mdp_trigger.state_space
    bands = mdp_trigger.velo_bands
    term = ss.terminal_mask()
    reachable = (mdp_trigger.support.sum(axis=1) > 0) & (~term)

    gaps = []
    n_creating = []
    changes = []
    for s in np.flatnonzero(reachable):
        # Decode prev family index of this trigger-design state.
        cid_p, _trig = divmod(int(s), 2)
        _cid, pidx = divmod(cid_p, N_PREV)
        feas = np.flatnonzero(mdp_trigger.feasible[s])
        if feas.size == 0:
            continue
        if pidx == _PREV_NONE_IDX or not np.isfinite(bands[pidx]):
            creating = np.zeros(feas.size, dtype=bool)
        else:
            gap_a = np.abs(bands[feas] - bands[pidx])
            creating = np.isfinite(gap_a) & (gap_a >= float(threshold))
        n_creating.append(int(creating.sum()))
        if creating.any() and (~creating).any():
            gaps.append(float(Q_t[s, feas[creating]].max() - Q_t[s, feas[~creating]].max()))
        # Optimal-action change vs the count_prev parent.
        parent = cid_p  # count_prev id == (count_id*9 + prev_idx) == cid_p
        if parent < mdp_count_prev.n_nonterminal:
            changes.append(int(pol_t[s]) != int(pol_p[parent]))

    gaps = np.asarray(gaps, dtype=np.float64)
    changes = np.asarray(changes, dtype=bool)
    return {
        "setup_gap_mean": float(gaps.mean()) if gaps.size else float("nan"),
        "setup_gap_median": float(np.median(gaps)) if gaps.size else float("nan"),
        "setup_gap_positive_frac": float((gaps > 0).mean()) if gaps.size else float("nan"),
        "n_states_with_gap": int(gaps.size),
        "optimal_action_change_frac": float(changes.mean()) if changes.size else float("nan"),
        "n_states_compared": int(changes.size),
        "mean_creating_actions": float(np.mean(n_creating)) if n_creating else float("nan"),
        "threshold": float(threshold),
    }
