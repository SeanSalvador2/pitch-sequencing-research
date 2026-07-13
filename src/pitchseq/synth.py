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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import load_config
from .outcomes import OUTCOME1

__all__ = ["make_null_world", "make_positive_world", "simulate_world"]

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
