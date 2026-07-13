"""Synthetic null / positive worlds -- the WS0 correctness oracle (SPEC ``11``; D18, D20).

Two worlds share one simulation engine and emit **raw-Statcast-schema-compatible** frames
(the minimal column subset the builders consume), so a fixture flows through
``build_decision_table -> build_view (all five) -> build_sequences`` and exercises the
leakage-safe path end to end (D18).

The engine separates the three findings SPEC ``0`` warns are routinely conflated:

* **Selection structure (finding #1)** -- pitchers have order-dependent *selection* habits
  (a reduced probability of a third consecutive pitch of the same family). This is present
  in **both** worlds; it lets ordered history predict *what is thrown next*.
* **Predictive sequencing value (finding #2)** -- ordered history predicting the *outcome*
  after conditioning on the current pitch and game state. This is **absent by
  construction** in the null world (outcomes depend only on ``(count, platoon, current
  family)``) and **planted** in the positive world.

Positive-world effect (D20). The obvious "``|velo_t - velo_{t-1}|`` boosts whiffs" effect
cannot satisfy the SPEC ``11`` acceptance that a sequence model beats **L1** as well as U:
an outcome model conditions on the current pitch (so ``velo_t`` is known through the chosen
family) and the L1 view already carries ``prev_release_speed`` (``velo_{t-1}``), so L1
captures any pure previous-pitch effect and ``Delta_order`` collapses to 0 by construction.
We therefore key the planted effect on the **last ordered transition**
``|velo_{t-1} - velo_{t-2}|`` -- the velocity change *into* the previous pitch. Only the
ordered **O** view represents this quantity (via ``o_velo_delta_last`` / the slot speeds);
U keeps unordered means and L1 keeps a single previous pitch, so neither can express it.
When that transition is large the batter is still adjusting, and the whiff probability on
the current pitch is boosted by a known amount. The effect requires two prior pitches, so
it lives in longer plate appearances (``pitch_number >= 3``).

Both generators return ``(raw_df, truth_meta)``; ``truth_meta`` carries the ground truth
the acceptance tests check against.

Two further fixtures support the off-policy-evaluation harness (SPEC ``9`` / ``11``); they
are self-contained toy environments, *not* the pitch simulator above:

* :func:`make_logged_bandit` -- a small discrete contextual bandit with a known,
  non-uniform, state-dependent logging policy, a known mean-reward table ``E[R|s,a]`` and
  Gaussian reward noise. Its :class:`BanditTruth` exposes an exact analytic evaluator
  ``policy_value(pi)`` for *any* policy matrix, plus a couple of canonical target policies
  with precomputed values -- the ground truth for the OPE known-value self-test.
* :func:`make_two_step_mdp` -- a tiny two-step episodic decision process (a controlled
  Markov reward process) whose :class:`TwoStepTruth` gives the exact policy value and the
  exact target-policy Q-functions, so the sequential estimators (step-wise DR, FQE) have an
  analytically computable target to recover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import load_config
from .outcomes import OUTCOME1

__all__ = [
    "make_null_world",
    "make_positive_world",
    "simulate_world",
    "make_logged_bandit",
    "make_two_step_mdp",
    "BanditTruth",
    "TwoStepTruth",
]

# --- fixed categories / mappings -------------------------------------------------------

#: Families the engine actually throws (``XX`` is left out of repertoires so feasibility
#: and family features stay well populated; the family map still supports it downstream).
_ENGINE_FAMILIES = ("FF", "SI", "FC", "SL", "CU", "CH", "FS")

#: Representative raw ``pitch_type`` code emitted per family (mapped back by ``add_family``).
_FAMILY_PITCH_TYPE = {
    "FF": "FF",
    "SI": "SI",
    "FC": "FC",
    "SL": "SL",
    "CU": "CU",
    "CH": "CH",
    "FS": "FS",
}

# OUTCOME1 index positions (kept in sync with pitchseq.outcomes.OUTCOME1).
_I_BALL = OUTCOME1.index("ball")
_I_CALLED = OUTCOME1.index("called_strike")
_I_WHIFF = OUTCOME1.index("whiff")
_I_FOUL = OUTCOME1.index("foul")
_I_HBP = OUTCOME1.index("hbp")
_I_INPLAY = OUTCOME1.index("in_play")

_DESCRIPTION = {
    _I_BALL: "ball",
    _I_CALLED: "called_strike",
    _I_WHIFF: "swinging_strike",
    _I_FOUL: "foul",
    _I_HBP: "hit_by_pitch",
    _I_INPLAY: "hit_into_play",
}

# Physical baselines per family (league-ish), before per-archetype velocity shift.
_FAM_VELO = {"FF": 94.5, "SI": 93.5, "FC": 90.0, "SL": 85.5, "CU": 79.0, "CH": 85.0, "FS": 86.5}
_FAM_PFX_X = {"FF": 0.6, "SI": 1.4, "FC": -0.2, "SL": -0.9, "CU": -0.7, "CH": 1.1, "FS": 0.8}
_FAM_PFX_Z = {"FF": 1.5, "SI": 0.9, "FC": 0.8, "SL": 0.2, "CU": -0.8, "CH": 0.6, "FS": 0.1}
_FAM_SPIN = {"FF": 2300, "SI": 2150, "FC": 2450, "SL": 2500, "CU": 2650, "CH": 1750, "FS": 1500}

# Behavioural propensities per family (used by the outcome model).
_FAM_SWING = {"FF": 0.47, "SI": 0.48, "FC": 0.47, "SL": 0.45, "CU": 0.42, "CH": 0.46, "FS": 0.46}
_FAM_WHIFF = {"FF": 0.16, "SI": 0.13, "FC": 0.21, "SL": 0.30, "CU": 0.28, "CH": 0.27, "FS": 0.31}
_FAM_ZONE = {"FF": 0.54, "SI": 0.52, "FC": 0.50, "SL": 0.44, "CU": 0.42, "CH": 0.46, "FS": 0.45}


@dataclass
class _Archetype:
    name: str
    throws: str  # 'R' or 'L'
    velo_shift: float
    weights: dict  # family -> base selection weight (unnormalised)


#: A handful of distinct pitcher archetypes (distinct repertoires + velo bands) so the
#: rolling repertoire / baseline features are non-trivial (SPEC ``11``).
_ARCHETYPES = [
    _Archetype("power_rhp", "R", 2.5, {"FF": 0.40, "SL": 0.30, "CH": 0.18, "CU": 0.12}),
    _Archetype("sinker_rhp", "R", 0.5, {"SI": 0.42, "SL": 0.28, "CH": 0.20, "FF": 0.10}),
    _Archetype("finesse_lhp", "L", -2.0, {"FF": 0.34, "CH": 0.28, "CU": 0.22, "SL": 0.16}),
    _Archetype("cutter_rhp", "R", 0.5, {"FC": 0.34, "FF": 0.26, "SL": 0.22, "CU": 0.18}),
    _Archetype("splitter_rhp", "R", 1.0, {"FF": 0.38, "FS": 0.30, "SL": 0.20, "SI": 0.12}),
    _Archetype("curve_lhp", "L", -1.0, {"FF": 0.32, "CU": 0.30, "SL": 0.20, "CH": 0.18}),
]

_TEAMS = ["NYY", "BOS", "LAD", "SFG", "CHC", "STL", "HOU", "ATL"]

_SEASON_START = {yr: pd.Timestamp(f"{yr}-04-01") for yr in range(2019, 2031)}


def _choice(rng: np.random.Generator, probs: np.ndarray) -> int:
    """Sample an index from a (normalised) probability vector via inverse-CDF."""
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r < c:
            return i
    return len(probs) - 1


@dataclass
class _World:
    """Mutable accumulators for one simulated world."""

    world: str
    effect_size: float
    threshold: float
    rng: np.random.Generator
    cols: dict = field(default_factory=dict)
    trigger_flags: list = field(default_factory=list)
    applied_boost: list = field(default_factory=list)
    whiff_flags: list = field(default_factory=list)


def _outcome1_probs(balls: int, strikes: int, platoon: bool, family: str) -> np.ndarray:
    """Length-6 outcome-1 distribution given ``(count, platoon, family)`` only.

    Returns probabilities aligned to :data:`pitchseq.outcomes.OUTCOME1`
    ``(ball, called_strike, whiff, foul, hbp, in_play)``. This is the *entire* outcome
    mechanism of the null world: it depends on nothing about the ordered history, so
    finding #2 is absent by construction.
    """
    p_hbp = 0.008

    swing = _FAM_SWING[family]
    if strikes == 2:
        swing += 0.08
    if balls == 3 and strikes < 2:
        swing -= 0.10
    if balls == 0 and strikes == 0:
        swing -= 0.03
    if platoon:
        swing -= 0.02
    swing = float(np.clip(swing, 0.20, 0.75))

    noswing = 1.0 - p_hbp - swing

    cs_share = _FAM_ZONE[family]
    if balls > strikes:  # behind in the count -> throw more strikes
        cs_share += 0.08
    elif strikes > balls:
        cs_share -= 0.05
    cs_share = float(np.clip(cs_share, 0.10, 0.85))
    p_ball = noswing * (1.0 - cs_share)
    p_cs = noswing * cs_share

    wh_share = _FAM_WHIFF[family]
    if strikes == 2:
        wh_share += 0.03
    if platoon:
        wh_share += 0.04
    wh_share = float(np.clip(wh_share, 0.05, 0.60))
    foul_share = min(0.38, 1.0 - wh_share - 0.10)
    inplay_share = 1.0 - wh_share - foul_share
    p_whiff = swing * wh_share
    p_foul = swing * foul_share
    p_inplay = swing * inplay_share

    vec = np.array([p_ball, p_cs, p_whiff, p_foul, p_hbp, p_inplay], dtype=np.float64)
    return vec / vec.sum()


def _apply_effect(vec: np.ndarray, effect_size: float) -> tuple[np.ndarray, float]:
    """Move up to ``effect_size`` of probability into whiff, from foul + in-play.

    Returns the adjusted vector and the actual boost applied (bounded by the available
    foul + in-play mass). Mass is drawn proportionally so the two swing outcomes keep
    their relative weight.
    """
    available = vec[_I_FOUL] + vec[_I_INPLAY]
    if available <= 0:
        return vec, 0.0
    boost = float(min(effect_size, 0.9 * available))
    out = vec.copy()
    out[_I_WHIFF] += boost
    out[_I_FOUL] -= boost * (vec[_I_FOUL] / available)
    out[_I_INPLAY] -= boost * (vec[_I_INPLAY] / available)
    return out, boost


def _batted_event(rng: np.random.Generator) -> str:
    """Sample an in-play event (BABIP-ish)."""
    probs = np.array([0.680, 0.205, 0.058, 0.006, 0.051])  # out, 1B, 2B, 3B, HR
    events = ("field_out", "single", "double", "triple", "home_run")
    return events[_choice(rng, probs)]


# Base run-expectancy magnitudes; sign fixed so R = -delta_run_exp obeys SPEC 5.
_DELTA_BASE = {
    "ball": +0.038,
    "called_strike": -0.045,
    "whiff": -0.050,
    "foul_pre2": -0.020,
    "foul_2k": -0.002,
    "walk": +0.310,
    "strikeout": -0.100,
    "hbp": +0.300,
    "field_out": -0.250,
    "single": +0.450,
    "double": +0.750,
    "triple": +1.030,
    "home_run": +1.400,
}


def _delta_run_exp(rng: np.random.Generator, key: str) -> float:
    """A sign-consistent ``delta_run_exp`` for an outcome (lognormal spread keeps sign)."""
    base = _DELTA_BASE[key]
    return float(base * np.exp(0.22 * rng.standard_normal()))


def _physics(rng: np.random.Generator, arch: _Archetype, family: str) -> dict:
    """Sample the realised physical measurements of one pitch."""
    throw_sign = 1.0 if arch.throws == "R" else -1.0
    velo = _FAM_VELO[family] + arch.velo_shift + 1.3 * rng.standard_normal()
    plate_x = 0.70 * rng.standard_normal()
    plate_z = 2.50 + 0.60 * rng.standard_normal()
    in_zone = (abs(plate_x) < 0.83) and (1.6 < plate_z < 3.4)
    return {
        "release_speed": velo,
        "pfx_x": _FAM_PFX_X[family] * throw_sign + 0.15 * rng.standard_normal(),
        "pfx_z": _FAM_PFX_Z[family] + 0.20 * rng.standard_normal(),
        "plate_x": plate_x,
        "plate_z": plate_z,
        "sz_top": 3.40,
        "sz_bot": 1.60,
        "release_spin_rate": _FAM_SPIN[family] + 120.0 * rng.standard_normal(),
        "release_extension": 6.40 + 0.20 * rng.standard_normal(),
        "release_pos_x": -1.6 * throw_sign + 0.20 * rng.standard_normal(),
        "release_pos_z": 5.80 + 0.20 * rng.standard_normal(),
        "zone": 5 if in_zone else 12,
    }


def _select_family(rng: np.random.Generator, arch: _Archetype, count: tuple, hist_fams: list) -> str:
    """Order-dependent selection habit (finding #1): dampen a 3rd consecutive same family."""
    fams = list(arch.weights.keys())
    w = np.array([arch.weights[f] for f in fams], dtype=np.float64)
    balls, strikes = count
    for i, f in enumerate(fams):
        if strikes == 2:
            if f in ("SL", "CU", "FS"):
                w[i] *= 1.30
            elif f == "FF":
                w[i] *= 0.90
        if balls == 3 and strikes < 2:
            if f in ("FF", "SI"):
                w[i] *= 1.30
            elif f in ("SL", "CU", "FS"):
                w[i] *= 0.80
        # No-three-in-a-row habit: if the last two pitches were this family, downweight it.
        if len(hist_fams) >= 2 and hist_fams[-1] == f and hist_fams[-2] == f:
            w[i] *= 0.30
    w /= w.sum()
    return fams[_choice(rng, w)]


class _RowSink:
    """Column-oriented row accumulator (fast frame assembly)."""

    _COLUMNS = (
        "game_type", "pitch_type", "game_date", "game_pk", "at_bat_number", "pitch_number",
        "pitcher", "batter", "description", "events", "delta_run_exp", "balls", "strikes",
        "outs_when_up", "on_1b", "on_2b", "on_3b", "inning", "inning_topbot", "fld_score",
        "bat_score", "stand", "p_throws", "n_thruorder_pitcher", "home_team", "away_team",
        "release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot",
        "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "zone",
    )

    def __init__(self) -> None:
        self.data = {c: [] for c in self._COLUMNS}

    def add(self, **kw) -> None:
        for c in self._COLUMNS:
            self.data[c].append(kw[c])

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        for c in ("on_1b", "on_2b", "on_3b", "events"):
            df[c] = df[c].astype("object")
        return df


def _advance_bases(bases: list, event: str, batter: int) -> int:
    """Advance runners for a terminal event; return runs scored. Mutates ``bases`` in place.

    ``bases`` is ``[first, second, third]`` holding runner ids or ``None``. A simplified,
    force-free advancement model (no double plays / extra bases): enough to make base-out
    context vary realistically (it never feeds outcomes).
    """
    runs = 0
    if event in ("walk", "hit_by_pitch"):
        # Force only as far as needed.
        if bases[0] is None:
            bases[0] = batter
        elif bases[1] is None:
            bases[1] = batter  # runner from 1st forced to 2nd, but keep simple
            bases[0] = batter
        elif bases[2] is None:
            bases[2] = bases[1]
            bases[1] = batter
            bases[0] = batter
        else:
            runs += 1  # bases loaded -> forced run
            bases[2] = bases[1]
            bases[1] = batter
            bases[0] = batter
        return runs
    if event == "single":
        if bases[2] is not None:
            runs += 1
        bases[2] = bases[1]
        bases[1] = bases[0]
        bases[0] = batter
    elif event == "double":
        runs += int(bases[2] is not None) + int(bases[1] is not None)
        bases[2] = bases[0]
        bases[1] = batter
        bases[0] = None
    elif event == "triple":
        runs += sum(1 for b in bases if b is not None)
        bases[:] = [None, None, batter]
    elif event == "home_run":
        runs += sum(1 for b in bases if b is not None) + 1
        bases[:] = [None, None, None]
    return runs


def simulate_world(
    n_games: int,
    seed: int,
    config: dict | None = None,
    world: str = "null",
    effect_size: float = 0.20,
    velo_gap_threshold: float = 5.0,
    innings_per_game: int = 9,
    n_pitchers: int = 12,
    n_batters: int = 18,
) -> tuple[pd.DataFrame, dict]:
    """Simulate one world and return ``(raw_df, truth_meta)``.

    Parameters
    ----------
    n_games : int
        Number of games to simulate.
    seed : int
        Seed for the single :class:`numpy.random.Generator` driving the world.
    config : dict, optional
        Parsed config (used for the season list). Loaded from default when ``None``.
    world : {'null', 'positive'}
        Which world to build.
    effect_size : float
        Positive world only: absolute whiff-probability boost on triggered pitches.
    velo_gap_threshold : float
        Positive world only: ``|velo_{t-1} - velo_{t-2}|`` (mph) at/above which the boost
        fires.
    innings_per_game : int
        Innings simulated per game (smaller keeps fixtures fast).
    n_pitchers, n_batters : int
        Sizes of the reusable player pools (reuse across games/seasons makes the trailing
        rolling features and matchup memory non-trivial).

    Returns
    -------
    (pandas.DataFrame, dict)
        Raw-Statcast-schema-compatible frame and the ground-truth metadata.
    """
    if world not in ("null", "positive"):
        raise ValueError(f"world must be 'null' or 'positive', got {world!r}")
    if config is None:
        config = load_config()
    seasons = list(config.get("data", {}).get("seasons", [2021, 2022, 2023, 2024, 2025]))
    rng = np.random.default_rng(seed)

    # Player pools (fixed identity / archetype / handedness).
    pitchers = [1000 + i for i in range(n_pitchers)]
    pitcher_arch = {pid: _ARCHETYPES[i % len(_ARCHETYPES)] for i, pid in enumerate(pitchers)}
    batters = [2000 + i for i in range(n_batters)]
    batter_stand = {bid: ("L" if i % 3 == 0 else "R") for i, bid in enumerate(batters)}

    sink = _RowSink()
    trigger_flags: list = []
    applied_boost: list = []
    whiff_flags: list = []

    n_seasons = len(seasons)
    for g in range(n_games):
        season = seasons[g % n_seasons]
        game_date = _SEASON_START[season] + pd.Timedelta(days=3 * (g // n_seasons))
        game_pk = 500000 + g
        home_team = _TEAMS[g % len(_TEAMS)]
        away_team = _TEAMS[(g + 1) % len(_TEAMS)]
        home_pitcher = pitchers[(2 * g) % n_pitchers]
        away_pitcher = pitchers[(2 * g + 1) % n_pitchers]
        # Rotate lineups so matchups recur but vary.
        home_lineup = [batters[(g + k) % n_batters] for k in range(9)]
        away_lineup = [batters[(g + 9 + k) % n_batters] for k in range(9)]

        home_score = away_score = 0
        at_bat_number = 0
        faced = {home_pitcher: 0, away_pitcher: 0}
        order_idx = {"home": 0, "away": 0}

        for inning in range(1, innings_per_game + 1):
            for topbot in ("Top", "Bot"):
                if topbot == "Top":
                    batting, pitcher, lineup, side = "away", home_pitcher, away_lineup, "away"
                else:
                    batting, pitcher, lineup, side = "home", away_pitcher, home_lineup, "home"
                arch = pitcher_arch[pitcher]
                outs = 0
                bases: list = [None, None, None]

                while outs < 3:
                    at_bat_number += 1
                    batter = lineup[order_idx[side]]
                    order_idx[side] = (order_idx[side] + 1) % 9
                    faced[pitcher] += 1
                    ntho = 1 + (faced[pitcher] - 1) // 9
                    stand = batter_stand[batter]
                    p_throws = arch.throws
                    platoon = stand == p_throws

                    balls = strikes = 0
                    hist_fams: list = []
                    hist_velo: list = []
                    pitch_number = 0

                    # Scores as recorded pre-pitch (constant within a PA).
                    if topbot == "Top":
                        fld_score, bat_score = home_score, away_score
                    else:
                        fld_score, bat_score = away_score, home_score
                    b1, b2, b3 = bases[0], bases[1], bases[2]
                    outs_pre = outs

                    while True:
                        pitch_number += 1
                        family = _select_family(rng, arch, (balls, strikes), hist_fams)
                        phys = _physics(rng, arch, family)
                        velo = phys["release_speed"]

                        trigger = (
                            world == "positive"
                            and len(hist_velo) >= 2
                            and abs(hist_velo[-1] - hist_velo[-2]) >= velo_gap_threshold
                        )
                        vec = _outcome1_probs(balls, strikes, platoon, family)
                        boost = 0.0
                        if trigger:
                            vec, boost = _apply_effect(vec, effect_size)
                        o1 = _choice(rng, vec)

                        balls_before = balls
                        strikes_before = strikes
                        terminal = False
                        events = None
                        delta_key = None

                        if o1 == _I_BALL:
                            balls += 1
                            if balls >= 4:
                                terminal, events, delta_key = True, "walk", "walk"
                            else:
                                delta_key = "ball"
                        elif o1 == _I_CALLED:
                            strikes += 1
                            if strikes >= 3:
                                terminal, events, delta_key = True, "strikeout", "strikeout"
                            else:
                                delta_key = "called_strike"
                        elif o1 == _I_WHIFF:
                            strikes += 1
                            if strikes >= 3:
                                terminal, events, delta_key = True, "strikeout", "strikeout"
                            else:
                                delta_key = "whiff"
                        elif o1 == _I_FOUL:
                            if strikes < 2:
                                strikes += 1
                            delta_key = "foul_pre2" if strikes_before < 2 else "foul_2k"
                        elif o1 == _I_HBP:
                            terminal, events, delta_key = True, "hit_by_pitch", "hbp"
                        else:  # in play
                            events = _batted_event(rng)
                            terminal, delta_key = True, events

                        delta = _delta_run_exp(rng, delta_key)

                        sink.add(
                            game_type="R",
                            pitch_type=_FAMILY_PITCH_TYPE[family],
                            game_date=game_date,
                            game_pk=game_pk,
                            at_bat_number=at_bat_number,
                            pitch_number=pitch_number,
                            pitcher=pitcher,
                            batter=batter,
                            description=_DESCRIPTION[o1],
                            events=events,
                            delta_run_exp=delta,
                            balls=balls_before,
                            strikes=strikes_before,
                            outs_when_up=outs_pre,
                            on_1b=b1,
                            on_2b=b2,
                            on_3b=b3,
                            inning=inning,
                            inning_topbot=topbot,
                            fld_score=fld_score,
                            bat_score=bat_score,
                            stand=stand,
                            p_throws=p_throws,
                            n_thruorder_pitcher=ntho,
                            home_team=home_team,
                            away_team=away_team,
                            release_speed=phys["release_speed"],
                            pfx_x=phys["pfx_x"],
                            pfx_z=phys["pfx_z"],
                            plate_x=phys["plate_x"],
                            plate_z=phys["plate_z"],
                            sz_top=phys["sz_top"],
                            sz_bot=phys["sz_bot"],
                            release_spin_rate=phys["release_spin_rate"],
                            release_extension=phys["release_extension"],
                            release_pos_x=phys["release_pos_x"],
                            release_pos_z=phys["release_pos_z"],
                            zone=phys["zone"],
                        )
                        trigger_flags.append(trigger)
                        applied_boost.append(boost)
                        whiff_flags.append(o1 == _I_WHIFF)

                        hist_fams.append(family)
                        hist_velo.append(velo)

                        if terminal:
                            if events in ("strikeout",):
                                outs += 1
                            elif events == "field_out":
                                outs += 1
                            else:
                                runs = _advance_bases(bases, events, batter)
                                if topbot == "Top":
                                    away_score += runs
                                else:
                                    home_score += runs
                            break

    raw = sink.frame()

    trig = np.asarray(trigger_flags, dtype=bool)
    boost_arr = np.asarray(applied_boost, dtype=float)
    whiff = np.asarray(whiff_flags, dtype=bool)
    truth: dict = {
        "world": world,
        "n_games": n_games,
        "n_pitches": int(len(raw)),
        "seed": seed,
        "mechanism": "velo_gap_prev_transition" if world == "positive" else "none",
    }
    if world == "positive":
        affected = float(trig.mean()) if len(trig) else 0.0
        realized = float(boost_arr[trig].mean()) if trig.any() else 0.0
        w_trig = float(whiff[trig].mean()) if trig.any() else 0.0
        w_untrig = float(whiff[~trig].mean()) if (~trig).any() else 0.0
        truth.update(
            {
                "effect": float(effect_size),
                "threshold": float(velo_gap_threshold),
                "affected_rate": affected,
                "realized_boost": realized,
                "empirical_whiff_lift": w_trig - w_untrig,
            }
        )
    else:
        truth.update({"effect": 0.0, "threshold": None, "affected_rate": 0.0})
    return raw, truth


def make_null_world(
    n_games: int = 60,
    seed: int = 20260713,
    config: dict | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Null world: order-dependent *selection*, but outcomes depend only on context.

    See the module docstring. Acceptance (SPEC ``11``): the harness's ``Delta_order`` for
    outcome prediction is indistinguishable from 0 and the permutation test does not fire.

    Returns
    -------
    (pandas.DataFrame, dict)
        ``(raw_df, truth_meta)`` with ``truth_meta['world'] == 'null'``.
    """
    return simulate_world(n_games, seed, config, world="null", **kwargs)


def make_positive_world(
    n_games: int = 60,
    seed: int = 20260713,
    config: dict | None = None,
    effect_size: float = 0.20,
    velo_gap_threshold: float = 5.0,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Positive world: the null engine plus a planted ordered previous-transition effect.

    When ``|velo_{t-1} - velo_{t-2}| >= velo_gap_threshold`` the whiff probability on the
    current pitch is boosted by ``effect_size`` (bounded by available swing mass). See the
    module docstring for why the effect is keyed on the *ordered transition* rather than a
    pure previous-pitch gap. Ground truth is returned in ``truth_meta``.

    Returns
    -------
    (pandas.DataFrame, dict)
        ``(raw_df, truth_meta)`` with ``truth_meta['world'] == 'positive'`` and keys
        ``effect``, ``threshold``, ``affected_rate``, ``realized_boost``,
        ``empirical_whiff_lift``.
    """
    return simulate_world(
        n_games,
        seed,
        config,
        world="positive",
        effect_size=effect_size,
        velo_gap_threshold=velo_gap_threshold,
        **kwargs,
    )


# =====================================================================================
# Off-policy-evaluation fixtures (SPEC 9 / 11)
# =====================================================================================


def _softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax over ``axis``."""
    shifted = scores - scores.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _floored_policy(scores: np.ndarray, floor: float) -> np.ndarray:
    """A softmax policy mixed with a uniform floor so every action keeps mass ``>= floor``.

    Returns ``floor + (1 - A*floor) * softmax(scores)`` row-wise, which is a valid
    distribution (sums to 1) whose smallest entry is exactly ``floor`` on the least-favoured
    action. A strictly positive floor guarantees full support, so importance ratios
    ``pi/mu`` stay finite and bounded by ``1/floor``. Requires ``A * floor < 1``.
    """
    n_actions = scores.shape[-1]
    if n_actions * floor >= 1.0:
        raise ValueError(f"floor {floor} too large for {n_actions} actions (need A*floor < 1)")
    return floor + (1.0 - n_actions * floor) * _softmax(scores, axis=-1)


def _epsilon_greedy(values: np.ndarray, epsilon: float) -> np.ndarray:
    """Epsilon-greedy policy on a ``(states, actions)`` value table.

    Puts ``1 - epsilon + epsilon/A`` on ``argmax_a values[s, a]`` and ``epsilon/A`` on the
    rest, so the policy keeps full support (every action has probability ``>= epsilon/A``).
    """
    n_states, n_actions = values.shape
    pi = np.full((n_states, n_actions), epsilon / n_actions, dtype=np.float64)
    best = values.argmax(axis=1)
    pi[np.arange(n_states), best] += 1.0 - epsilon
    return pi


def _sample_categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Vectorised categorical sampling: one draw per row of the ``(n, A)`` matrix ``probs``.

    Uses per-row inverse-CDF (``searchsorted``-free): counts how many cumulative thresholds
    the uniform draw exceeds. Deterministic for a fixed ``rng`` state.
    """
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(probs.shape[0])
    # Compare against the A-1 interior thresholds; the count is the sampled index in [0, A).
    return (u[:, None] >= cdf[:, :-1]).sum(axis=1).astype(np.int64)


@dataclass
class BanditTruth:
    """Ground truth for a :func:`make_logged_bandit` fixture.

    Attributes
    ----------
    q_table : numpy.ndarray, shape (n_states, n_actions)
        The exact mean-reward table ``E[R | s, a]``.
    state_marginal : numpy.ndarray, shape (n_states,)
        The true state distribution ``p(s)`` the contexts are drawn from.
    behavior_policy : numpy.ndarray, shape (n_states, n_actions)
        The logging policy ``mu(a | s)`` (full support, non-uniform, state-dependent).
    targets : dict
        Named canonical target policies, each a ``(n_states, n_actions)`` matrix.
    n_states, n_actions : int
        Sizes.
    reward_noise : float
        Standard deviation of the Gaussian reward noise.
    seed : int
        Generating seed.
    """

    q_table: np.ndarray
    state_marginal: np.ndarray
    behavior_policy: np.ndarray
    targets: dict
    n_states: int
    n_actions: int
    reward_noise: float
    seed: int

    def policy_value(self, pi: np.ndarray) -> float:
        r"""Exact analytic value of a policy ``pi``.

        .. math:: V(\pi) = \sum_s p(s) \sum_a \pi(a\mid s)\, \mathbb{E}[R\mid s, a]

        Parameters
        ----------
        pi : numpy.ndarray, shape (n_states, n_actions)
            Any policy matrix (rows sum to 1).

        Returns
        -------
        float
            The exact expected reward of ``pi`` under this world.
        """
        pi = np.asarray(pi, dtype=np.float64)
        if pi.shape != self.q_table.shape:
            raise ValueError(f"pi shape {pi.shape} != q_table shape {self.q_table.shape}")
        return float((self.state_marginal[:, None] * pi * self.q_table).sum())

    @property
    def behavior_value(self) -> float:
        """Exact value of the logging policy ``V(mu)``."""
        return self.policy_value(self.behavior_policy)

    def target_value(self, name: str) -> float:
        """Exact value of a named canonical target policy."""
        return self.policy_value(self.targets[name])

    def target_probs_for(self, states, name_or_matrix) -> np.ndarray:
        """Per-row target probabilities for the given logged ``states``.

        Parameters
        ----------
        states : array-like of int
            Logged state indices (length ``n``).
        name_or_matrix : str or numpy.ndarray
            A key into :attr:`targets` or an explicit ``(n_states, n_actions)`` matrix.

        Returns
        -------
        numpy.ndarray, shape (n, n_actions)
            The target policy row for each logged state.
        """
        pi = self.targets[name_or_matrix] if isinstance(name_or_matrix, str) else np.asarray(name_or_matrix)
        return pi[np.asarray(states, dtype=np.int64)]

    def behavior_probs_for(self, states) -> np.ndarray:
        """Per-row behavior probabilities ``mu(.|s)`` for the given logged ``states``."""
        return self.behavior_policy[np.asarray(states, dtype=np.int64)]


def make_logged_bandit(
    n_rounds: int = 8000,
    seed: int = 20260713,
    n_states: int = 6,
    n_actions: int = 5,
    reward_noise: float = 0.30,
    propensity_floor: float = 0.04,
    epsilon: float = 0.10,
) -> tuple[pd.DataFrame, BanditTruth]:
    r"""A small discrete contextual bandit with a known logging policy (SPEC ``9`` / ``11``).

    Contexts (states) are drawn i.i.d. from a fixed non-uniform marginal ``p(s)``; the
    action is drawn from a known, non-uniform, state-dependent logging policy
    ``mu(a | s)`` with a strictly positive floor (so importance ratios vary but stay
    bounded by ``1/floor``); the reward is ``E[R | s, a]`` plus Gaussian noise. Every
    quantity needed for an exact off-policy evaluation is returned in the
    :class:`BanditTruth`.

    Parameters
    ----------
    n_rounds : int, optional
        Number of logged rounds (rows). Default 8000.
    seed : int, optional
        Seed for the single generator driving states, actions and reward noise.
    n_states, n_actions : int, optional
        Context and action counts. Defaults 6 and 5.
    reward_noise : float, optional
        Standard deviation of the Gaussian reward noise (default 0.30).
    propensity_floor : float, optional
        Minimum logging propensity for any action (default 0.04). Must satisfy
        ``n_actions * propensity_floor < 1``.
    epsilon : float, optional
        Exploration rate of the epsilon-greedy canonical target (default 0.10).

    Returns
    -------
    (pandas.DataFrame, BanditTruth)
        ``logged_df`` has one row per round with columns:

        ``state`` (int context), ``action`` (int taken), ``reward`` (float),
        ``mu_prob`` (the logging propensity of the taken action ``mu(a_t | s_t)``),
        ``mu_prob_0 ... mu_prob_{A-1}`` (the full logging distribution per row),
        ``pa_id`` (episode id -- one round is a one-step episode) and ``step`` (all 0).

        ``truth`` is a :class:`BanditTruth` carrying ``q_table`` (``E[R|s,a]``),
        ``state_marginal`` (``p(s)``), ``behavior_policy`` (``mu``), the analytic
        ``policy_value`` evaluator and two canonical targets:

        * ``"greedy"`` -- epsilon-greedy on the true ``q_table``;
        * ``"shift"``  -- a deliberately different-from-``mu`` policy (softmax on the
          negated logging scores), which puts mass where ``mu`` does not.
    """
    if n_actions * propensity_floor >= 1.0:
        raise ValueError("n_actions * propensity_floor must be < 1 for a valid floored policy")
    rng = np.random.default_rng(seed)

    # True mean-reward table E[R|s,a]: a spread of values with a clear best action per state.
    q_table = rng.normal(0.0, 0.6, size=(n_states, n_actions))
    q_table[np.arange(n_states), rng.integers(0, n_actions, size=n_states)] += 0.5

    # Non-uniform state marginal p(s).
    sw = rng.uniform(1.0, 3.0, size=n_states)
    state_marginal = sw / sw.sum()

    # Logging policy mu(a|s): state-dependent softmax with a positive floor.
    logging_scores = rng.normal(0.0, 1.0, size=(n_states, n_actions))
    mu = _floored_policy(logging_scores, propensity_floor)

    # Canonical target policies.
    pi_greedy = _epsilon_greedy(q_table, epsilon)
    pi_shift = _floored_policy(-1.5 * logging_scores, propensity_floor)
    targets = {"greedy": pi_greedy, "shift": pi_shift}

    # Sample the logged data.
    states = _sample_categorical(rng, np.tile(state_marginal, (n_rounds, 1)))
    actions = _sample_categorical(rng, mu[states])
    mean_reward = q_table[states, actions]
    rewards = mean_reward + reward_noise * rng.standard_normal(n_rounds)

    data = {
        "state": states.astype(np.int64),
        "action": actions.astype(np.int64),
        "reward": rewards.astype(np.float64),
        "mu_prob": mu[states, actions].astype(np.float64),
    }
    full_mu = mu[states]
    for a in range(n_actions):
        data[f"mu_prob_{a}"] = full_mu[:, a].astype(np.float64)
    data["pa_id"] = np.arange(n_rounds, dtype=np.int64)  # one round == one 1-step episode
    data["step"] = np.zeros(n_rounds, dtype=np.int64)
    logged_df = pd.DataFrame(data)

    truth = BanditTruth(
        q_table=q_table,
        state_marginal=state_marginal,
        behavior_policy=mu,
        targets=targets,
        n_states=n_states,
        n_actions=n_actions,
        reward_noise=reward_noise,
        seed=seed,
    )
    return logged_df, truth


@dataclass
class TwoStepTruth:
    """Ground truth for a :func:`make_two_step_mdp` fixture.

    A two-step episodic decision process: step 0 in one of ``n_s0`` states, a deterministic
    transition ``T`` to one of ``n_s1`` step-1 states, then a terminal step-1 reward. The
    return is ``r0 + r1`` (undiscounted). Attributes hold every table needed for exact
    evaluation; :meth:`policy_value` and :meth:`q_functions` give the analytic target value
    and target Q-functions the sequential estimators must recover.
    """

    p_s0: np.ndarray  # (n_s0,)
    R0: np.ndarray  # (n_s0, n_actions) mean step-0 reward
    R1: np.ndarray  # (n_s1, n_actions) mean step-1 reward
    T: np.ndarray  # (n_s0, n_actions) -> step-1 state index
    mu0: np.ndarray  # (n_s0, n_actions) behavior at step 0
    mu1: np.ndarray  # (n_s1, n_actions) behavior at step 1
    targets: dict  # name -> (pi0, pi1)
    n_s0: int
    n_s1: int
    n_actions: int
    reward_noise: float
    seed: int

    @property
    def n_states_total(self) -> int:
        """Total distinct global state ids (step-0 states then step-1 states)."""
        return self.n_s0 + self.n_s1

    def q_functions(self, pi1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Exact Q-functions under a step-1 policy ``pi1``.

        .. math::
            Q_1(s_1, a) &= \mathbb{E}[R_1 \mid s_1, a] \\
            Q_0(s_0, a) &= \mathbb{E}[R_0 \mid s_0, a]
                           + \sum_{a'} \pi_1(a' \mid T(s_0, a))\, Q_1(T(s_0, a), a')

        Returns ``(Q0, Q1)`` with shapes ``(n_s0, n_actions)`` and ``(n_s1, n_actions)``.
        These are the "return-to-go" values of taking an action then following the target,
        i.e. exactly the outcome model a doubly-robust or fitted-Q estimator should learn.
        """
        pi1 = np.asarray(pi1, dtype=np.float64)
        q1 = self.R1.copy()
        v1 = (pi1 * q1).sum(axis=1)  # (n_s1,) value of each step-1 state under pi1
        q0 = self.R0 + v1[self.T]  # (n_s0, n_actions)
        return q0, q1

    def policy_value(self, pi0: np.ndarray, pi1: np.ndarray) -> float:
        r"""Exact expected return ``E[r0 + r1]`` under the target policy ``(pi0, pi1)``.

        .. math:: V(\pi) = \sum_{s_0} p(s_0) \sum_{a} \pi_0(a\mid s_0)\, Q_0(s_0, a)
        """
        pi0 = np.asarray(pi0, dtype=np.float64)
        q0, _ = self.q_functions(pi1)
        v0 = (pi0 * q0).sum(axis=1)  # (n_s0,)
        return float((self.p_s0 * v0).sum())

    def target_value(self, name: str) -> float:
        """Exact value of a named canonical target policy."""
        pi0, pi1 = self.targets[name]
        return self.policy_value(pi0, pi1)

    def per_row_target(self, logged: pd.DataFrame, name: str) -> np.ndarray:
        """Per-row target probabilities for a logged frame (rows aligned to ``logged``)."""
        pi0, pi1 = self.targets[name]
        return self._per_row_matrix(logged, pi0, pi1)

    def per_row_q(self, logged: pd.DataFrame, name: str) -> np.ndarray:
        """Per-row exact target Q-values ``(n_rows, n_actions)`` for a logged frame."""
        pi0, pi1 = self.targets[name]
        q0, q1 = self.q_functions(pi1)
        return self._per_row_matrix(logged, q0, q1)

    def _per_row_matrix(self, logged: pd.DataFrame, tab0: np.ndarray, tab1: np.ndarray) -> np.ndarray:
        """Assemble a per-row matrix from step-0 / step-1 lookup tables via the global state id."""
        step = logged["step"].to_numpy()
        state = logged["state"].to_numpy()
        out = np.empty((len(logged), self.n_actions), dtype=np.float64)
        m0 = step == 0
        m1 = step == 1
        out[m0] = tab0[state[m0]]
        out[m1] = tab1[state[m1] - self.n_s0]  # step-1 global ids are offset by n_s0
        return out


def make_two_step_mdp(
    n_episodes: int = 5000,
    seed: int = 20260713,
    n_s0: int = 3,
    n_s1: int = 4,
    n_actions: int = 2,
    reward_noise: float = 0.20,
    propensity_floor: float = 0.10,
    epsilon: float = 0.10,
) -> tuple[pd.DataFrame, TwoStepTruth]:
    r"""A tiny two-step episodic decision process for the sequential OPE self-tests.

    Each episode: draw ``s0 ~ p(s0)``; take ``a0 ~ mu0(.|s0)`` earning ``r0 = E[R0|s0,a0] +
    noise``; transition deterministically to ``s1 = T(s0, a0)``; take ``a1 ~ mu1(.|s1)``
    earning terminal ``r1 = E[R1|s1,a1] + noise``. The undiscounted return is ``r0 + r1``.
    The behavior policies are floored softmaxes (full support); the canonical target is a
    one-step-lookahead epsilon-greedy improvement, whose exact value and Q-functions the
    :class:`TwoStepTruth` provides.

    Global state ids in ``logged_df`` encode the level: step-0 rows carry ``state = s0`` in
    ``[0, n_s0)``; step-1 rows carry ``state = n_s0 + s1``. This keeps a single ``state``
    column unambiguous for fitted-Q feature construction.

    Returns
    -------
    (pandas.DataFrame, TwoStepTruth)
        ``logged_df`` has two rows per episode (``step`` 0 then 1) with columns
        ``pa_id``, ``step``, ``state``, ``action``, ``reward``, ``mu_prob`` and
        ``mu_prob_0 ... mu_prob_{A-1}``. ``truth`` is a :class:`TwoStepTruth`.
    """
    rng = np.random.default_rng(seed)

    sw = rng.uniform(1.0, 2.5, size=n_s0)
    p_s0 = sw / sw.sum()

    R0 = rng.normal(0.0, 0.5, size=(n_s0, n_actions))
    R1 = rng.normal(0.0, 0.6, size=(n_s1, n_actions))
    # Deterministic transition table s1 = T(s0, a0).
    T = rng.integers(0, n_s1, size=(n_s0, n_actions))

    mu0 = _floored_policy(rng.normal(0.0, 1.0, size=(n_s0, n_actions)), propensity_floor)
    mu1 = _floored_policy(rng.normal(0.0, 1.0, size=(n_s1, n_actions)), propensity_floor)

    # One-step-lookahead epsilon-greedy target: pi1 greedy on R1, then pi0 greedy on the
    # induced Q0 (well-defined because pi1 is fixed before pi0).
    pi1_target = _epsilon_greedy(R1, epsilon)
    v1_target = (pi1_target * R1).sum(axis=1)
    q0_target = R0 + v1_target[T]
    pi0_target = _epsilon_greedy(q0_target, epsilon)
    targets = {"lookahead_greedy": (pi0_target, pi1_target)}

    # Sample episodes.
    s0 = _sample_categorical(rng, np.tile(p_s0, (n_episodes, 1)))
    a0 = _sample_categorical(rng, mu0[s0])
    r0 = R0[s0, a0] + reward_noise * rng.standard_normal(n_episodes)
    s1 = T[s0, a0]
    a1 = _sample_categorical(rng, mu1[s1])
    r1 = R1[s1, a1] + reward_noise * rng.standard_normal(n_episodes)

    # Interleave step-0 / step-1 rows in episode-major order.
    pa_id = np.repeat(np.arange(n_episodes, dtype=np.int64), 2)
    step = np.tile(np.array([0, 1], dtype=np.int64), n_episodes)
    state = np.empty(2 * n_episodes, dtype=np.int64)
    state[0::2] = s0
    state[1::2] = n_s0 + s1
    action = np.empty(2 * n_episodes, dtype=np.int64)
    action[0::2] = a0
    action[1::2] = a1
    reward = np.empty(2 * n_episodes, dtype=np.float64)
    reward[0::2] = r0
    reward[1::2] = r1

    mu_taken = np.empty(2 * n_episodes, dtype=np.float64)
    mu_taken[0::2] = mu0[s0, a0]
    mu_taken[1::2] = mu1[s1, a1]
    full_mu = np.empty((2 * n_episodes, n_actions), dtype=np.float64)
    full_mu[0::2] = mu0[s0]
    full_mu[1::2] = mu1[s1]

    data = {
        "pa_id": pa_id,
        "step": step,
        "state": state,
        "action": action,
        "reward": reward,
        "mu_prob": mu_taken,
    }
    for a in range(n_actions):
        data[f"mu_prob_{a}"] = full_mu[:, a]
    logged_df = pd.DataFrame(data)

    truth = TwoStepTruth(
        p_s0=p_s0,
        R0=R0,
        R1=R1,
        T=T,
        mu0=mu0,
        mu1=mu1,
        targets=targets,
        n_s0=n_s0,
        n_s1=n_s1,
        n_actions=n_actions,
        reward_noise=reward_noise,
        seed=seed,
    )
    return logged_df, truth
