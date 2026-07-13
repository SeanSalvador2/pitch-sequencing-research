"""The five nested state views C / U / L1 / O / OM (SPEC 6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import make_raw
from pitchseq.decision_table import build_decision_table, leakage_audit
from pitchseq.states import STATE_VIEWS, build_view

# --- a handcrafted 2-PA game (same batter vs same pitcher), known values throughout ---
# PA1 (at_bat 1): FF@95, SL@85, then a decision pitch (2 priors).
# PA2 (at_bat 2): first-pitch decision -> only matchup memory from PA1 can matter.
_ZTOP, _ZBOT = 3.5, 1.5  # so plate_z_norm = (plate_z - 1.5) / 2


def _base(**kw):
    common = dict(game_pk=1, pitcher=100, batter=200, sz_top=_ZTOP, sz_bot=_ZBOT, stand="R")
    common.update(kw)
    return common


def _two_pa_table():
    rows = [
        # PA1
        _base(at_bat_number=1, pitch_number=1, pitch_type="FF", release_speed=95.0, pfx_x=1.0, pfx_z=1.5,
              plate_x=0.5, plate_z=2.5, description="ball"),
        _base(at_bat_number=1, pitch_number=2, pitch_type="SL", release_speed=85.0, pfx_x=-1.0, pfx_z=0.5,
              plate_x=-0.5, plate_z=2.0, description="called_strike"),
        _base(at_bat_number=1, pitch_number=3, pitch_type="FF", release_speed=96.0, pfx_x=1.1, pfx_z=1.4,
              plate_x=0.0, plate_z=2.5, description="swinging_strike"),
        # PA2
        _base(at_bat_number=2, pitch_number=1, pitch_type="CH", release_speed=88.0, pfx_x=0.8, pfx_z=0.9,
              plate_x=0.2, plate_z=2.2, description="foul"),
        _base(at_bat_number=2, pitch_number=2, pitch_type="FF", release_speed=95.0, pfx_x=1.0, pfx_z=1.5,
              plate_x=0.1, plate_z=2.6, description="ball"),
    ]
    return build_decision_table(make_raw(rows))


def _row(X, table, ab, pn):
    idx = table.index[(table["at_bat_number"] == ab) & (table["pitch_number"] == pn)][0]
    return X.loc[idx]


def test_u_block_exact():
    table = _two_pa_table()
    X, meta = build_view(table, "U")
    r = _row(X, table, ab=1, pn=3)  # priors = [FF@95 (x=0.5,z=0.5), SL@85 (x=-0.5,z=0.25)]
    assert r["u_n_prior"] == 2
    assert r["u_count_FF"] == 1
    assert r["u_count_SL"] == 1
    assert r["u_count_CH"] == 0
    assert r["u_n_ball"] == 1
    assert r["u_n_called_strike"] == 1
    assert r["u_n_whiff"] == 0
    assert r["u_mean_release_speed"] == pytest.approx(90.0)
    assert r["u_mean_plate_x_br"] == pytest.approx(0.0)
    assert r["u_mean_plate_z_norm"] == pytest.approx((0.5 + 0.25) / 2)
    assert r["u_prior_missing"] == 0
    leakage_audit(X)


def test_l1_block_equals_previous_pitch():
    table = _two_pa_table()
    X, _ = build_view(table, "L1")
    r = _row(X, table, ab=1, pn=3)  # previous pitch = SL@85
    assert str(r["prev_family"]) == "SL"
    assert str(r["prev_pitch_type"]) == "SL"
    assert str(r["prev_outcome1"]) == "called_strike"
    assert r["prev_release_speed"] == pytest.approx(85.0)
    assert r["prev_plate_x_br"] == pytest.approx(-0.5)
    assert r["prev_plate_z_norm"] == pytest.approx(0.25)
    assert r["prev_pfx_x"] == pytest.approx(-1.0)
    assert r["l1_prev_missing"] == 0


def test_o_block_ordered_features_exact():
    table = _two_pa_table()
    X, _ = build_view(table, "O")
    r = _row(X, table, ab=1, pn=3)
    # slot 1 = most recent prior (SL), slot 2 = FF, slot 3 = missing.
    assert str(r["o_s1_family"]) == "SL"
    assert r["o_s1_release_speed"] == pytest.approx(85.0)
    assert str(r["o_s2_family"]) == "FF"
    assert r["o_s2_release_speed"] == pytest.approx(95.0)
    assert r["o_s3_missing"] == 1
    assert str(r["o_first_family"]) == "FF"  # first pitch of the PA
    # velocity delta into the last prior transition = 85 - 95 = -10 (one transition only).
    assert r["o_velo_delta_last"] == pytest.approx(-10.0)
    assert r["o_velo_delta_mean"] == pytest.approx(-10.0)
    # location delta = distance between the two priors.
    expect_loc = math.hypot(-0.5 - 0.5, 0.25 - 0.5)
    assert r["o_loc_delta_last"] == pytest.approx(expect_loc)
    assert r["o_loc_delta_mean"] == pytest.approx(expect_loc)
    assert r["o_same_fam_run"] == pytest.approx(1.0)  # priors FF,SL -> last run length 1


def test_o_same_family_run_length():
    # Three same-family priors then a decision -> run length 3.
    rows = [
        _base(at_bat_number=1, pitch_number=1, pitch_type="SL", description="ball"),
        _base(at_bat_number=1, pitch_number=2, pitch_type="SL", description="ball"),
        _base(at_bat_number=1, pitch_number=3, pitch_type="SL", description="called_strike"),
        _base(at_bat_number=1, pitch_number=4, pitch_type="FF", description="ball"),
    ]
    table = build_decision_table(make_raw(rows))
    X, _ = build_view(table, "O")
    assert _row(X, table, 1, 4)["o_same_fam_run"] == pytest.approx(3.0)


def test_om_within_game_matchup_counts():
    table = _two_pa_table()
    X, _ = build_view(table, "OM")
    # PA2 first pitch: earlier PA1 (same batter/pitcher, this game) had 3 pitches, 1 swing,
    # 1 whiff (the swinging_strike on PA1 pitch 3).
    r = _row(X, table, ab=2, pn=1)
    assert r["om_wg_n_pitches"] == pytest.approx(3.0)
    assert r["om_wg_n_swings"] == pytest.approx(1.0)
    assert r["om_wg_n_whiffs"] == pytest.approx(1.0)
    assert r["om_wg_n_prior_pa"] == pytest.approx(1.0)
    assert r["om_wg_missing"] == 0
    # PA1 first pitch has no earlier matchup this game.
    r0 = _row(X, table, ab=1, pn=1)
    assert r0["om_wg_n_pitches"] == pytest.approx(0.0)
    assert r0["om_wg_missing"] == 1


def test_om_no_cross_matchup_contamination():
    # Regression: a groupwise cumsum followed by a GLOBAL shift leaked the previous
    # matchup's totals into the first PA of the next matchup. Interleaved scenario:
    # pitcher 99 faces batter 10 (at_bat 1: 3 pitches, at_bat 3: 2 pitches) and
    # batter 20 (at_bat 2: 2 pitches) in one game.
    rows = []
    for pn, desc in ((1, "ball"), (2, "ball"), (3, "swinging_strike")):
        rows.append(_base(pitcher=99, batter=10, at_bat_number=1, pitch_number=pn, description=desc))
    for pn in (1, 2):
        rows.append(_base(pitcher=99, batter=20, at_bat_number=2, pitch_number=pn, description="ball"))
    for pn in (1, 2):
        rows.append(_base(pitcher=99, batter=10, at_bat_number=3, pitch_number=pn, description="ball"))
    table = build_decision_table(make_raw(rows))
    X, _ = build_view(table, "OM")

    # Batter 20's only PA has NO earlier meeting: all zeros, flagged missing.
    r20 = X.loc[table.index[table["batter"] == 20]]
    assert (r20["om_wg_n_pitches"] == 0.0).all()
    assert (r20["om_wg_n_swings"] == 0.0).all()
    assert (r20["om_wg_n_whiffs"] == 0.0).all()
    assert (r20["om_wg_n_prior_pa"] == 0.0).all()
    assert (r20["om_wg_missing"] == 1.0).all()

    # Batter 10's at_bat 3 sees exactly its own at_bat 1 (3 pitches, 1 swing, 1 whiff).
    r10 = X.loc[table.index[(table["batter"] == 10) & (table["at_bat_number"] == 3)]]
    assert (r10["om_wg_n_pitches"] == 3.0).all()
    assert (r10["om_wg_n_swings"] == 1.0).all()
    assert (r10["om_wg_n_whiffs"] == 1.0).all()
    assert (r10["om_wg_n_prior_pa"] == 1.0).all()
    assert (r10["om_wg_missing"] == 0.0).all()

    # Internal consistency on every row: missing <=> no prior PA of this matchup.
    assert ((X["om_wg_missing"] == 1.0) == (X["om_wg_n_prior_pa"] == 0.0)).all()


def test_prev_pitch_type_fixed_vocabulary():
    # Regression: categories were derived from the data at hand, so two separately
    # built matrices encoded codes differently. The vocabulary is now fixed from config.
    def build(codes):
        rows = [
            _base(at_bat_number=1, pitch_number=i + 1, pitch_type=c, description="ball")
            for i, c in enumerate(codes)
        ]
        return build_decision_table(make_raw(rows))

    t1 = build(["FF", "SL", "FF"])
    t2 = build(["CH", "KC", "CU"])  # disjoint pitch types
    x1, _ = build_view(t1, "L1")
    x2, _ = build_view(t2, "L1")
    cats1 = list(x1["prev_pitch_type"].cat.categories)
    cats2 = list(x2["prev_pitch_type"].cat.categories)
    assert cats1 == cats2
    assert cats1[-2:] == ["__UNK__", "__NONE__"]

    # A prior pitch with an unseen or null pitch_type lands in __UNK__ (a real prior
    # pitch exists, so l1_prev_missing stays 0); __NONE__ is only "no prior pitch".
    t3 = build(["ZZ", "FF"])  # ZZ not in the config vocabulary
    x3, _ = build_view(t3, "L1")
    r = _row(x3, t3, ab=1, pn=2)
    assert str(r["prev_pitch_type"]) == "__UNK__"
    assert r["l1_prev_missing"] == 0
    t4 = build([None, "FF"])  # null pitch_type on a real prior pitch
    x4, _ = build_view(t4, "L1")
    r4 = _row(x4, t4, ab=1, pn=2)
    assert str(r4["prev_pitch_type"]) == "__UNK__"
    assert r4["l1_prev_missing"] == 0
    assert str(_row(x4, t4, ab=1, pn=1)["prev_pitch_type"]) == "__NONE__"


def test_pitch1_neutral_fills_and_indicators():
    table = _two_pa_table()
    XO, _ = build_view(table, "O")
    r = _row(XO, table, ab=1, pn=1)  # very first pitch: no within-PA history
    assert r["u_n_prior"] == 0
    assert r["u_prior_missing"] == 1
    assert r["u_mean_release_speed"] == 0.0  # neutral fill
    assert r["l1_prev_missing"] == 1
    assert str(r["prev_family"]) == "__NONE__"
    assert r["prev_release_speed"] == 0.0
    assert r["o_s1_missing"] == 1 and r["o_s2_missing"] == 1 and r["o_s3_missing"] == 1
    assert str(r["o_first_family"]) == "__NONE__"
    assert r["o_same_fam_run"] == 0.0


def test_u_order_invariant_l1_o_order_sensitive():
    def build(order):
        rows = [
            _base(at_bat_number=1, pitch_number=1, pitch_type=order[0], release_speed=95.0 if order[0] == "FF" else 85.0,
                  plate_x=0.5, plate_z=2.5, description="ball"),
            _base(at_bat_number=1, pitch_number=2, pitch_type=order[1], release_speed=95.0 if order[1] == "FF" else 85.0,
                  plate_x=-0.5, plate_z=2.0, description="called_strike"),
            _base(at_bat_number=1, pitch_number=3, pitch_type="CH", description="foul"),
        ]
        return build_decision_table(make_raw(rows))

    t_ab = build(["FF", "SL"])
    t_ba = build(["SL", "FF"])
    u_ab = _row(build_view(t_ab, "U")[0], t_ab, 1, 3)
    u_ba = _row(build_view(t_ba, "U")[0], t_ba, 1, 3)
    # Unordered aggregates are identical regardless of prior order.
    for col in ["u_count_FF", "u_count_SL", "u_mean_release_speed", "u_n_ball", "u_n_called_strike"]:
        assert u_ab[col] == pytest.approx(u_ba[col])
    # But L1 and O ordered features flip with the order.
    l1_ab = _row(build_view(t_ab, "L1")[0], t_ab, 1, 3)
    l1_ba = _row(build_view(t_ba, "L1")[0], t_ba, 1, 3)
    assert str(l1_ab["prev_family"]) != str(l1_ba["prev_family"])
    o_ab = _row(build_view(t_ab, "O")[0], t_ab, 1, 3)
    o_ba = _row(build_view(t_ba, "O")[0], t_ba, 1, 3)
    assert str(o_ab["o_first_family"]) != str(o_ba["o_first_family"])


def test_prev_vs_baseline_diff():
    # Prior game establishes an FF baseline of 100 mph; the decision game's previous pitch
    # is an FF at 96 -> diff = -4.
    rows = [
        _base(game_pk=1, game_date="2023-04-01", at_bat_number=1, pitch_number=1, pitch_type="FF", release_speed=100.0, description="ball"),
        _base(game_pk=1, game_date="2023-04-01", at_bat_number=2, pitch_number=1, pitch_type="FF", release_speed=100.0, description="ball"),
        _base(game_pk=2, game_date="2023-04-08", at_bat_number=1, pitch_number=1, pitch_type="FF", release_speed=96.0, description="ball"),
        _base(game_pk=2, game_date="2023-04-08", at_bat_number=1, pitch_number=2, pitch_type="SL", description="ball"),
    ]
    table = build_decision_table(make_raw(rows))
    X, _ = build_view(table, "L1")
    r = _row(X, table, ab=1, pn=2)  # in game 2; prev = FF@96, baseline FF = 100
    assert r["prev_release_speed_vs_base"] == pytest.approx(-4.0)


def test_view_nesting_and_shapes():
    table = _two_pa_table()
    cols = {v: set(build_view(table, v)[0].columns) for v in STATE_VIEWS}
    assert cols["C"] <= cols["U"]
    assert cols["C"] <= cols["L1"]
    assert cols["U"] <= cols["O"]
    assert cols["L1"] <= cols["O"]
    assert cols["O"] <= cols["OM"]
    # Every view is leakage-clean (no exec/label columns).
    for v in STATE_VIEWS:
        leakage_audit(build_view(table, v)[0])
