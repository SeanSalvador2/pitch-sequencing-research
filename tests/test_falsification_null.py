"""SPEC 11 null-world acceptance: no ordered outcome dependence may be discovered.

Design (all deterministic under fixed seeds):

* World: ``make_null_world(n_games=60, seed=3, innings_per_game=6)`` -- order-dependent
  *selection* but outcomes depend only on ``(count, platoon, current family)``.
* Split: train = seasons 2021-2023, eval = seasons 2024-2025 **restricted to long PAs**
  (``pitch_number >= 3``), the only region where a two-prior-pitch ordered effect could
  live, so this is where the null must hold most sharply.
* Learner: a compact, regularised, deterministic LightGBM (heavy shrinkage so the O view's
  ~40 extra columns cannot overfit the tiny variance of the null).
* Acceptance: the permutation test does not fire (``p >= 0.05``) and ``Delta_order`` is not
  significantly *positive* (its cluster-CI lower bound ``<= 0``) and small in magnitude.

Why "not significantly positive" rather than "CI covers 0": ``Delta_order =
Loss(min[U, L1]) - Loss(O)`` is biased *negative* under the null because ``min[U, L1]``
optimistically picks the luckier of two noisy view-losses, so a small negative null delta
is the correct null behaviour. The falsification concern is a spurious *positive* order
effect, which is exactly what this asserts against.
"""

from __future__ import annotations

from model_factories import make_lgbm_factory

from pitchseq.decision_table import build_decision_table
from pitchseq.eval.falsification import run_null_world_acceptance
from pitchseq.synth import make_null_world

_PARAMS = dict(
    num_leaves=8, min_child_samples=60, reg_lambda=5.0, n_estimators=50, max_depth=4, min_split_gain=1e-3
)


def test_null_world_no_ordered_outcome_effect():
    raw, truth = make_null_world(n_games=60, seed=3, innings_per_game=6)
    assert truth["world"] == "null"
    table = build_decision_table(raw)

    train_mask = table["season"].isin([2021, 2022, 2023]).to_numpy()
    long_pa = (table["pitch_number"] >= 3).to_numpy()
    eval_mask = table["season"].isin([2024, 2025]).to_numpy() & long_pa

    factory = make_lgbm_factory(**_PARAMS)
    verdict = run_null_world_acceptance(
        table, factory, train_mask, eval_mask,
        n_permutations=24, seed=1, ci_boot=300, delta_threshold=0.03,
    )

    # Permutation test must NOT fire.
    assert verdict["permutation_fired"] is False
    assert verdict["permutation_p"] >= 0.05
    # Delta_order not significantly positive, and small in magnitude.
    assert verdict["delta_order_ci"]["lo"] <= 0.0
    assert abs(verdict["delta_order"]) < 0.03
    # No ordered effect detected -> null confirmed.
    assert verdict["order_effect_detected"] is False
    assert verdict["verdict"] == "NULL_CONFIRMED"
