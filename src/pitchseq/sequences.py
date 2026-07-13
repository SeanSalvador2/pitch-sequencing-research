"""Variable-length ordered history tensors (SPEC ``3.3``, for the deep / Markov threads).

For every decision row, the history is the ordered list of *prior* pitches in the same PA
(pitches ``1..t-1``), oldest first; it is empty for the first pitch of a PA. The tensors are
left-aligned and padded to ``max_len`` (keeping the most recent ``max_len`` pitches when a PA
runs longer), with a boolean mask and per-row lengths, and a ``row_id`` array so the tensors
align back to the decision table.

Only prior pitches' realized (``exec_``) values are used -- never the current pitch's -- so the
representation is leakage-safe by the same reasoning as the state views.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .families import FAMILIES
from .outcomes import OUTCOME1

__all__ = ["build_sequences", "SEQUENCE_FLOAT_CHANNELS"]

#: Float channels emitted per history position, in order.
SEQUENCE_FLOAT_CHANNELS = (
    "plate_x_br",
    "plate_z_norm",
    "release_speed",
    "pfx_x",
    "pfx_z",
    "dvelo",  # release_speed change from the previous history pitch (0 at position 0)
    "dloc",   # euclidean location change from the previous history pitch (0 at position 0)
)

_FAM_INDEX = {fam: i for i, fam in enumerate(FAMILIES)}
_O1_INDEX = {oc: i for i, oc in enumerate(OUTCOME1)}
_FAM_PAD = len(FAMILIES)
_O1_PAD = len(OUTCOME1)


def build_sequences(table: pd.DataFrame, max_len: int = 15) -> dict:
    """Build padded ordered-history tensors aligned to the decision table.

    Parameters
    ----------
    table : pandas.DataFrame
        A decision table from :func:`~pitchseq.decision_table.build_decision_table`.
    max_len : int, optional
        Maximum history length kept (most recent pitches; default 15).

    Returns
    -------
    dict
        ``family_idx`` / ``outcome1_idx`` -- ``(N, max_len)`` int token arrays (padded with
        a dedicated pad index); one ``(N, max_len)`` float array per
        :data:`SEQUENCE_FLOAT_CHANNELS`; ``mask`` ``(N, max_len)``; ``lengths`` ``(N,)``;
        ``row_id`` ``(N,)`` aligned to ``table``; plus vocab / pad-index metadata.
    """
    n = len(table)
    pa = table["pa_id"].to_numpy()
    pn = table["pitch_number"].to_numpy()

    fam_idx_all = np.array(
        [_FAM_INDEX.get(f, _FAM_PAD) for f in table["family"].astype("object").to_numpy()],
        dtype=np.int64,
    )
    o1_idx_all = np.array(
        [_O1_INDEX.get(o, _O1_PAD) for o in table["outcome1"].astype("object").to_numpy()],
        dtype=np.int64,
    )
    xbr = _f(table, "exec_plate_x_br")
    znorm = _f(table, "exec_plate_z_norm")
    speed = _f(table, "exec_release_speed")
    pfx_x = _f(table, "exec_pfx_x")
    pfx_z = _f(table, "exec_pfx_z")
    row_id_all = table["row_id"].astype("object").to_numpy()

    # Sort into (pa_id, pitch_number) order; remember how to scatter back.
    order = np.lexsort((pn, pa))
    pa_s = pa[order]

    family_idx = np.full((n, max_len), _FAM_PAD, dtype=np.int64)
    outcome1_idx = np.full((n, max_len), _O1_PAD, dtype=np.int64)
    channels = {ch: np.zeros((n, max_len), dtype=np.float64) for ch in SEQUENCE_FLOAT_CHANNELS}
    mask = np.zeros((n, max_len), dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int64)

    fam_s = fam_idx_all[order]
    o1_s = o1_idx_all[order]
    xs = np.nan_to_num(xbr[order])
    zs = np.nan_to_num(znorm[order])
    ss = np.nan_to_num(speed[order])
    pxs = np.nan_to_num(pfx_x[order])
    pzs = np.nan_to_num(pfx_z[order])

    # PA block boundaries in sorted order.
    if n > 0:
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = pa_s[1:] != pa_s[:-1]
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], n)
        for s, e in zip(starts, ends):
            block = slice(s, e)
            fam_b = fam_s[block]
            o1_b = o1_s[block]
            x_b, z_b, sp_b, px_b, pz_b = xs[block], zs[block], ss[block], pxs[block], pzs[block]
            length = e - s
            for i in range(length):
                p = order[s + i]  # original row position for the i-th pitch of this PA
                lo = max(0, i - max_len)
                h = i - lo  # history length placed
                if h == 0:
                    continue
                sl = slice(lo, i)
                family_idx[p, :h] = fam_b[sl]
                outcome1_idx[p, :h] = o1_b[sl]
                hx, hz, hsp = x_b[sl], z_b[sl], sp_b[sl]
                channels["plate_x_br"][p, :h] = hx
                channels["plate_z_norm"][p, :h] = hz
                channels["release_speed"][p, :h] = hsp
                channels["pfx_x"][p, :h] = px_b[sl]
                channels["pfx_z"][p, :h] = pz_b[sl]
                if h >= 2:
                    dv = np.diff(hsp)
                    dl = np.hypot(np.diff(hx), np.diff(hz))
                    channels["dvelo"][p, 1:h] = dv
                    channels["dloc"][p, 1:h] = dl
                mask[p, :h] = 1.0
                lengths[p] = h

    result = {
        "family_idx": family_idx,
        "outcome1_idx": outcome1_idx,
        "mask": mask,
        "lengths": lengths,
        "row_id": row_id_all,
        "max_len": max_len,
        "family_vocab": list(FAMILIES),
        "outcome1_vocab": list(OUTCOME1),
        "family_pad_idx": _FAM_PAD,
        "outcome1_pad_idx": _O1_PAD,
        "float_channels": list(SEQUENCE_FLOAT_CHANNELS),
    }
    result.update(channels)
    return result


def _f(table: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(table[col], errors="coerce").astype("float64").to_numpy()
