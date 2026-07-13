"""SPEC 11 positive-world acceptance: the planted ordered effect must be recovered.

Design (deterministic under fixed seeds):

* World: ``make_positive_world(n_games=60, seed=3, innings_per_game=6, effect_size=0.30,
  velo_gap_threshold=5.0)`` -- the null engine plus a whiff boost on the current pitch when
  the ordered velo transition ``|velo_{t-1} - velo_{t-2}|`` clears 5 mph. Only the O view
  can represent that transition (``o_velo_delta_last`` / slot speeds), so O must beat U and
  L1, not just C.
* Split / learner: as in the null test (train 2021-2023, eval 2024-2025 long PAs, the same
  regularised LightGBM).
* Acceptance: the permutation test fires (``p < 0.05``); ``Delta_order`` is significantly
  positive (cluster-CI lower bound ``> 0``); the O-view outcome model recovers the effect's
  sign and rough magnitude (predicted whiff-probability lift on triggered pitches, within a
  factor of ~3 of the planted lift).
"""

from __future__ import annotations

from model_factories import make_lgbm_factory

from pitchseq.decision_table import build_decision_table
from pitchseq.eval.falsification import (
    mechanism_ablation,
    pseudo_history_control,
    run_positive_world_acceptance,
)
from pitchseq.synth import make_positive_world

_PARAMS = dict(
    num_leaves=8, min_child_samples=60, reg_lambda=5.0, n_estimators=50, max_depth=4, min_split_gain=1e-3
)
_THR = 5.0


def _positive_world():
    raw, truth = make_positive_world(
        n_games=60, seed=3, innings_per_game=6, effect_size=0.30, velo_gap_threshold=_THR
    )
    table = build_decision_table(raw)
    train_mask = table["season"].isin([2021, 2022, 2023]).to_numpy()
    long_pa = (table["pitch_number"] >= 3).to_numpy()
    eval_mask = table["season"].isin([2024, 2025]).to_numpy() & long_pa
    return table, truth, train_mask, eval_mask


def test_positive_world_order_effect_recovered():
    table, truth, train_mask, eval_mask = _positive_world()
    factory = make_lgbm_factory(**_PARAMS)
    verdict = run_positive_world_acceptance(
        table, factory, train_mask, eval_mask,
        n_permutations=24, seed=1, velo_gap_threshold=_THR, ci_boot=300,
    )

    # O beats U and L1: Delta_order significantly positive.
    assert verdict["delta_order"] > 0.0
    assert verdict["delta_order_ci"]["lo"] > 0.0
    assert verdict["losses"]["O"] < verdict["losses"]["U"]
    assert verdict["losses"]["O"] < verdict["losses"]["L1"]
    # Permutation test fires.
    assert verdict["permutation_fired"] is True
    assert verdict["permutation_p"] < 0.05
    # Effect direction and rough magnitude recovered from the O-view outcome model.
    ref = truth["empirical_whiff_lift"]
    assert verdict["recovered_whiff_lift"] > 0.0
    assert 0.3 * ref < verdict["recovered_whiff_lift"] < 3.0 * ref
    assert verdict["order_effect_detected"] is True
    assert verdict["verdict"] == "POSITIVE_CONFIRMED"


def test_positive_world_mechanism_and_pseudo_history():
    # Lighter checks on the same world: the velo-diff mechanism carries the effect, and
    # scrambling genuine history (same pitcher + count) hurts the outcome model.
    table, _, train_mask, eval_mask = _positive_world()
    factory = make_lgbm_factory(**_PARAMS)

    mech = mechanism_ablation(
        table, factory, target="outcome1", train_mask=train_mask, eval_mask=eval_mask,
        restrict_min_pitch=3,
    )
    # Dropping the velocity / velo-difference features hurts the O model the most.
    assert mech["deltas"]["velo_diff"] > 0.0
    assert mech["deltas"]["velo_diff"] == max(mech["deltas"].values())

    pseudo = pseudo_history_control(
        table, factory, seed=1, target="outcome1", train_mask=train_mask, eval_mask=eval_mask,
        restrict_min_pitch=3,
    )
    # Swapping in another PA's history (same pitcher, same count) raises the loss:
    # the model was using real ordered history, not just pitcher identity.
    assert pseudo["delta"] > 0.0
