"""The five nested state views C / U / L1 / O / OM (SPEC ``6``).

Every suitable model is trained on the same five state variants; the *differences* between
them are the sequencing evidence:

* **C**  -- non-sequence context ``X_t`` only.
* **U**  -- C + order-invariant aggregates of the prior pitches this PA.
* **L1** -- C + the immediately-preceding pitch.
* **O**  -- C + U + L1 + the full ordered current-PA sequence (slots, deltas, run length).
* **OM** -- O + matchup memory (within-game batter-vs-pitcher, trailing batter-vs-family).

The views are **information-nested** by construction (``cols(C) ⊆ cols(U) ⊆ cols(O)`` etc.).
Every history feature is built from *prior* pitches only -- a prior pitch's realized
``exec_`` values are fully observed before the current decision, so using them is legitimate;
the current pitch's own execution is never touched (enforced by
:func:`~pitchseq.decision_table.leakage_audit`). History features are neutral-filled for the
first pitch of a PA, with explicit ``*_missing`` indicator columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decision_table import CONTEXT_COLS, leakage_audit
from .families import FAMILIES, family_map_from_config
from .outcomes import OUTCOME1
from .rolling import batter_family_rolling

__all__ = ["STATE_VIEWS", "build_view", "history_features"]

STATE_VIEWS: tuple[str, ...] = ("C", "U", "L1", "O", "OM")

_NONE = "__NONE__"  # categorical sentinel for "no prior pitch"
_UNK = "__UNK__"  # prior pitch exists, but its pitch_type is null / outside the vocabulary
_FAM_CATS = list(FAMILIES) + [_NONE]
_O1_CATS = list(OUTCOME1) + [_NONE]


def _pitch_type_categories() -> list[str]:
    """Fixed ``prev_pitch_type`` vocabulary.

    Every concrete ``pitch_type`` code from the config's ``pitch_families`` lists (in
    config order, YAML ``null`` excluded) plus the ``__UNK__`` and ``__NONE__`` sentinels.
    Deriving the vocabulary from config rather than from the data at hand guarantees two
    separately built matrices encode categories identically.
    """
    return list(family_map_from_config().keys()) + [_UNK, _NONE]

# Prior-outcome counts tracked in the U (unordered) block.
_U_OUTCOMES = ("ball", "called_strike", "whiff", "foul")


def _sorted_working(table: pd.DataFrame) -> pd.DataFrame:
    """Return ``table`` sorted within-PA by pitch order (defensive; usually a no-op)."""
    return table.sort_values(["pa_id", "pitch_number"], kind="stable")


def _fill_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df[c] = df[c].fillna(0.0)


def _as_cat(values, categories) -> pd.Categorical:
    obj = pd.Series(values).astype("object")
    obj = obj.where(obj.notna(), _NONE)
    return pd.Categorical(obj, categories=categories)


def _u_block(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Unordered aggregates of the prior pitches this PA (order-invariant)."""
    g = t.groupby("pa_id", sort=False)
    n = len(t)
    out = pd.DataFrame(index=t.index)

    n_prior = g.cumcount().to_numpy().astype("float64")  # pitches before t in this PA
    out["u_n_prior"] = n_prior

    fam = t["family"].astype("object").to_numpy()
    scratch = pd.DataFrame(index=t.index)
    for f in FAMILIES:
        scratch[f"ind_{f}"] = (fam == f).astype("float64")
    o1 = t["outcome1"].astype("object").to_numpy()
    for oc in _U_OUTCOMES:
        scratch[f"oc_{oc}"] = (o1 == oc).astype("float64")
    for m in ("exec_plate_x_br", "exec_plate_z_norm", "exec_release_speed"):
        v = pd.to_numeric(t[m], errors="coerce").astype("float64").to_numpy()
        scratch[f"val_{m}"] = np.nan_to_num(v)
        scratch[f"vld_{m}"] = (~np.isnan(v)).astype("float64")
    scratch["pa_id"] = t["pa_id"].to_numpy()

    gs = scratch.groupby("pa_id", sort=False)
    # prior sum = cumulative-including-current minus current row's own value
    for f in FAMILIES:
        incl = gs[f"ind_{f}"].cumsum().to_numpy()
        out[f"u_count_{f}"] = incl - scratch[f"ind_{f}"].to_numpy()
    for oc in _U_OUTCOMES:
        incl = gs[f"oc_{oc}"].cumsum().to_numpy()
        out[f"u_n_{oc}"] = incl - scratch[f"oc_{oc}"].to_numpy()
    for m, short in (
        ("exec_plate_x_br", "plate_x_br"),
        ("exec_plate_z_norm", "plate_z_norm"),
        ("exec_release_speed", "release_speed"),
    ):
        s_incl = gs[f"val_{m}"].cumsum().to_numpy()
        v_incl = gs[f"vld_{m}"].cumsum().to_numpy()
        prior_sum = s_incl - scratch[f"val_{m}"].to_numpy()
        prior_vld = v_incl - scratch[f"vld_{m}"].to_numpy()
        mean = np.full(n, np.nan)
        np.divide(prior_sum, prior_vld, out=mean, where=prior_vld > 0)
        out[f"u_mean_{short}"] = mean

    missing = n_prior == 0
    out["u_prior_missing"] = missing.astype("float64")
    numeric_cols = [c for c in out.columns if c != "u_prior_missing"]
    _fill_numeric(out, numeric_cols)

    meta = {"columns": list(out.columns), "missing_indicators": ["u_prior_missing"]}
    return out, meta


def _l1_block(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The immediately-preceding pitch (token + diffs from the pitcher's family baseline).

    ``prev_pitch_type`` carries the fixed vocabulary of :func:`_pitch_type_categories`:
    ``__NONE__`` is reserved for "no prior pitch" (first pitch of a PA, mirrored by the
    ``l1_prev_missing`` indicator); a prior pitch whose ``pitch_type`` is null or outside
    the config vocabulary maps to the dedicated ``__UNK__`` bucket, keeping the two
    situations distinct.
    """
    g = t.groupby("pa_id", sort=False)
    out = pd.DataFrame(index=t.index)

    prev_family = g["family"].shift(1).astype("object").to_numpy()
    prev_pt = g["pitch_type"].shift(1).astype("object").to_numpy()
    prev_o1 = g["outcome1"].shift(1).astype("object").to_numpy()
    # family is never null on a real pitch (unknown -> XX), so a null prev_family means
    # exactly "no prior pitch in this PA".
    missing = np.asarray(pd.isna(prev_family))
    out["prev_family"] = _as_cat(prev_family, _FAM_CATS)
    pt_cats = _pitch_type_categories()
    known = pd.Series(prev_pt, dtype="object").isin(pt_cats[:-2]).to_numpy()
    pt_norm = np.where(known, prev_pt, np.where(missing, _NONE, _UNK))
    out["prev_pitch_type"] = pd.Categorical(pt_norm, categories=pt_cats)
    out["prev_outcome1"] = _as_cat(prev_o1, _O1_CATS)

    prev_vals = {}
    for m, short in (
        ("exec_plate_x_br", "plate_x_br"),
        ("exec_plate_z_norm", "plate_z_norm"),
        ("exec_release_speed", "release_speed"),
        ("exec_pfx_x", "pfx_x"),
        ("exec_pfx_z", "pfx_z"),
    ):
        pv = g[m].shift(1).to_numpy().astype("float64")
        prev_vals[short] = pv
        out[f"prev_{short}"] = pv

    # Diffs of the previous pitch from that pitcher's trailing baseline for the prev
    # pitch's family (same pitcher throughout the PA, so the current row's baseline
    # columns apply). Legitimate: t-1 is fully observed before decision t.
    for short, base_measure in (
        ("release_speed", "release_speed"),
        ("pfx_x", "pfx_x"),
        ("pfx_z", "pfx_z"),
    ):
        base_for_prev = np.full(len(t), np.nan)
        for f in FAMILIES:
            mask = prev_family == f
            if mask.any():
                base_for_prev[mask] = t[f"pitcher_base_{f}_{base_measure}"].to_numpy()[mask]
        out[f"prev_{short}_vs_base"] = prev_vals[short] - base_for_prev

    out["l1_prev_missing"] = missing.astype("float64")
    numeric_cols = [c for c in out.columns if c.startswith("prev_") and not _is_cat(out[c])]
    _fill_numeric(out, numeric_cols)

    meta = {"columns": list(out.columns), "missing_indicators": ["l1_prev_missing"]}
    return out, meta


def _is_cat(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)


def _o_block(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ordered current-PA features: last-3 slots, first family, deltas, run length."""
    g = t.groupby("pa_id", sort=False)
    n = len(t)
    out = pd.DataFrame(index=t.index)
    n_prior = g.cumcount().to_numpy()

    # Last-3 pitch slots (slot k = the k-th most recent prior pitch).
    slot_speed = {}
    slot_x = {}
    slot_z = {}
    for k in (1, 2, 3):
        fam_k = g["family"].shift(k).astype("object").to_numpy()
        o1_k = g["outcome1"].shift(k).astype("object").to_numpy()
        out[f"o_s{k}_family"] = _as_cat(fam_k, _FAM_CATS)
        out[f"o_s{k}_outcome1"] = _as_cat(o1_k, _O1_CATS)
        sp = g["exec_release_speed"].shift(k).to_numpy().astype("float64")
        xx = g["exec_plate_x_br"].shift(k).to_numpy().astype("float64")
        zz = g["exec_plate_z_norm"].shift(k).to_numpy().astype("float64")
        out[f"o_s{k}_release_speed"] = sp
        out[f"o_s{k}_plate_x_br"] = xx
        out[f"o_s{k}_plate_z_norm"] = zz
        out[f"o_s{k}_missing"] = (n_prior < k).astype("float64")
        slot_speed[k], slot_x[k], slot_z[k] = sp, xx, zz

    # First pitch of the PA (a prior only when t >= 2); neutral for pitch 1.
    first_fam = g["family"].transform("first").astype("object").to_numpy()
    first_fam = np.where(n_prior == 0, None, first_fam)
    out["o_first_family"] = _as_cat(first_fam, _FAM_CATS)

    # Per-step transitions among consecutive pitches (defined from the PA's 2nd pitch on):
    # step into pitch i = attribute(i) - attribute(i-1).
    speed = pd.to_numeric(t["exec_release_speed"], errors="coerce").astype("float64").to_numpy()
    xbr = pd.to_numeric(t["exec_plate_x_br"], errors="coerce").astype("float64").to_numpy()
    znorm = pd.to_numeric(t["exec_plate_z_norm"], errors="coerce").astype("float64").to_numpy()
    prev_speed_in = g["exec_release_speed"].shift(1).to_numpy().astype("float64")
    prev_x_in = g["exec_plate_x_br"].shift(1).to_numpy().astype("float64")
    prev_z_in = g["exec_plate_z_norm"].shift(1).to_numpy().astype("float64")
    step_velo = speed - prev_speed_in  # NaN for first pitch of PA
    step_dist = np.hypot(xbr - prev_x_in, znorm - prev_z_in)

    step = pd.DataFrame(
        {
            "pa_id": t["pa_id"].to_numpy(),
            "sv": np.nan_to_num(step_velo),
            "sv_valid": (~np.isnan(step_velo)).astype("float64"),
            "sd": np.nan_to_num(step_dist),
            "sd_valid": (~np.isnan(step_dist)).astype("float64"),
        },
        index=t.index,
    )
    gstep = step.groupby("pa_id", sort=False)
    # Mean over prior transitions = cumulative (before t) sum / valid count.
    for name, val, vld in (("velo", "sv", "sv_valid"), ("loc", "sd", "sd_valid")):
        s_incl = gstep[val].cumsum().to_numpy()
        v_incl = gstep[vld].cumsum().to_numpy()
        prior_sum = s_incl - step[val].to_numpy()
        prior_vld = v_incl - step[vld].to_numpy()
        mean = np.full(n, np.nan)
        np.divide(prior_sum, prior_vld, out=mean, where=prior_vld > 0)
        out[f"o_{name}_delta_mean"] = mean
    # Last transition among priors = the transition into pitch t-1.
    out["o_velo_delta_last"] = gstep["sv"].shift(1).to_numpy().astype("float64") * np.where(
        gstep["sv_valid"].shift(1).to_numpy().astype("float64") > 0, 1.0, np.nan
    )
    out["o_loc_delta_last"] = gstep["sd"].shift(1).to_numpy().astype("float64") * np.where(
        gstep["sd_valid"].shift(1).to_numpy().astype("float64") > 0, 1.0, np.nan
    )

    # Same-family repeat run length ending at the most recent prior pitch.
    fam_obj = t["family"].astype("object")
    changed = fam_obj.to_numpy() != g["family"].shift(1).astype("object").to_numpy()
    changed_series = pd.Series(changed, index=t.index)
    block_id = changed_series.groupby(t["pa_id"], sort=False).cumsum()
    run_incl = (
        pd.Series(1, index=t.index).groupby([t["pa_id"], block_id], sort=False).cumsum()
    )
    out["o_same_fam_run"] = _g_shift(run_incl, t, "pa_id")

    numeric_cols = [
        c for c in out.columns if not _is_cat(out[c]) and not c.endswith("_missing")
    ]
    _fill_numeric(out, numeric_cols)
    missing_inds = [c for c in out.columns if c.endswith("_missing")]

    meta = {"columns": list(out.columns), "missing_indicators": missing_inds}
    return out, meta


def _g_shift(series: pd.Series, t: pd.DataFrame, key: str) -> np.ndarray:
    """Shift ``series`` by one within each ``key`` group, aligned to ``t``'s index."""
    return series.groupby(t[key], sort=False).shift(1).to_numpy().astype("float64")


def _om_block(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Matchup memory: within-game batter-vs-pitcher + trailing batter-vs-family."""
    out = pd.DataFrame(index=t.index)

    # Within-game batter-vs-this-pitcher from strictly-earlier PAs (at_bat_number <).
    per_pa = (
        t.groupby(["game_pk", "pitcher", "batter", "at_bat_number"], sort=True)
        .agg(
            _n=("pitch_number", "size"),
            _sw=("is_swing", "sum"),
            _wh=("is_whiff", "sum"),
        )
        .reset_index()
        .sort_values(["game_pk", "pitcher", "batter", "at_bat_number"], kind="stable")
    )
    mkey = ["game_pk", "pitcher", "batter"]
    gm = per_pa.groupby(mkey, sort=False)
    # Cumulative over strictly-earlier PAs of this matchup via the exclusive-cumsum idiom
    # (groupwise cumulative-including-current minus the current PA's own value), the same
    # construction the U block uses. NOTE: a groupwise ``cumsum()`` followed by a plain
    # ``.shift(1)`` is NOT groupwise -- the shift runs globally across the frame, so the
    # first PA of each matchup would inherit the previous matchup's totals.
    per_pa["om_wg_n_pitches"] = (gm["_n"].cumsum() - per_pa["_n"]).astype("float64").to_numpy()
    per_pa["om_wg_n_swings"] = (gm["_sw"].cumsum() - per_pa["_sw"]).astype("float64").to_numpy()
    per_pa["om_wg_n_whiffs"] = (gm["_wh"].cumsum() - per_pa["_wh"]).astype("float64").to_numpy()
    per_pa["om_wg_n_prior_pa"] = gm.cumcount().to_numpy().astype("float64")
    # First PA of a matchup gets exactly 0 everywhere (its cumsum equals its own value).

    merged = t[["game_pk", "pitcher", "batter", "at_bat_number"]].merge(
        per_pa[mkey + ["at_bat_number", "om_wg_n_pitches", "om_wg_n_swings", "om_wg_n_whiffs", "om_wg_n_prior_pa"]],
        on=["game_pk", "pitcher", "batter", "at_bat_number"],
        how="left",
    )
    merged.index = t.index
    for c in ["om_wg_n_pitches", "om_wg_n_swings", "om_wg_n_whiffs", "om_wg_n_prior_pa"]:
        out[c] = merged[c].to_numpy()
    out["om_wg_missing"] = (out["om_wg_n_prior_pa"].to_numpy() == 0).astype("float64")

    # Trailing batter-vs-family (pre-game, from the rolling machinery).
    bvf = batter_family_rolling(t)
    for c in bvf.columns:
        out[c] = bvf[c].to_numpy()

    numeric_cols = [c for c in out.columns if c != "om_wg_missing"]
    _fill_numeric(out, numeric_cols)

    meta = {"columns": list(out.columns), "missing_indicators": ["om_wg_missing"]}
    return out, meta


def history_features(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute all history blocks (U, L1, O, OM) once, aligned to ``table``'s index."""
    t = _sorted_working(table)
    u, _ = _u_block(t)
    l1, _ = _l1_block(t)
    o, _ = _o_block(t)
    om, _ = _om_block(t)
    reindex = table.index
    return {
        "U": u.loc[reindex],
        "L1": l1.loc[reindex],
        "O": o.loc[reindex],
        "OM": om.loc[reindex],
    }


def build_view(table: pd.DataFrame, view: str) -> tuple[pd.DataFrame, dict]:
    """Build one nested state view.

    Parameters
    ----------
    table : pandas.DataFrame
        A decision table from :func:`~pitchseq.decision_table.build_decision_table`.
    view : str
        One of :data:`STATE_VIEWS` (``"C"``, ``"U"``, ``"L1"``, ``"O"``, ``"OM"``).

    Returns
    -------
    (pandas.DataFrame, dict)
        The feature matrix ``X`` (indexed like ``table``) and a ``meta`` dict with
        ``view``, ``feature_names``, ``dtypes``, ``n_features`` and ``missing_indicators``.
    """
    if view not in STATE_VIEWS:
        raise ValueError(f"Unknown view {view!r}; expected one of {STATE_VIEWS}")

    context = table[CONTEXT_COLS].copy()
    missing_inds: list[str] = []
    parts = [context]

    if view != "C":
        t = _sorted_working(table)
        reindex = table.index
        if view == "U":
            blk, meta = _u_block(t)
            parts.append(blk.loc[reindex])
            missing_inds += meta["missing_indicators"]
        elif view == "L1":
            blk, meta = _l1_block(t)
            parts.append(blk.loc[reindex])
            missing_inds += meta["missing_indicators"]
        elif view in ("O", "OM"):
            ublk, um = _u_block(t)
            lblk, lm = _l1_block(t)
            oblk, om_ = _o_block(t)
            parts += [ublk.loc[reindex], lblk.loc[reindex], oblk.loc[reindex]]
            missing_inds += um["missing_indicators"] + lm["missing_indicators"] + om_["missing_indicators"]
            if view == "OM":
                mblk, mm = _om_block(t)
                parts.append(mblk.loc[reindex])
                missing_inds += mm["missing_indicators"]

    X = pd.concat(parts, axis=1)
    # No execution or label column may ever appear in a state view (SPEC 0/11).
    leakage_audit(X)

    meta = {
        "view": view,
        "feature_names": list(X.columns),
        "dtypes": {c: str(X[c].dtype) for c in X.columns},
        "n_features": X.shape[1],
        "missing_indicators": missing_inds,
    }
    return X, meta
