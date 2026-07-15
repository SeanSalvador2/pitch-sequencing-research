"""WS6 deep sequence models -- capacity-matched GRU views (SPEC 12.4; decisions D45-D47).

The question WS6 exists to answer: does a *learned* representation of the ordered
plate-appearance sequence beat the engineered tabular history (WS3) and the explicit
variable-order grammar (WS2)? To measure that -- and to keep the comparison an honest
measurement of *information* rather than *capacity* -- the same five nested state views
C / U / L1 / O / OM (SPEC 6) are realised as **capacity-matched neural architectures**
(D45), all sharing embedding / hidden widths and the same MLP head:

* **C**  -- a static MLP on the context features only (the C view; categoricals one-hot
  encoded with a fitted vocabulary, ordinal/continuous features standardised).
* **U**  -- mean-pooled token embeddings (**order-invariant by construction**) + the static
  head. The pooled set of prior-pitch tokens, with order thrown away.
* **L1** -- a GRU consuming **only the last history token** + the static head.
* **O**  -- a GRU over the **full ordered** padded sequence + the static head.
* **OM** -- O + the matchup-memory columns appended through a second static branch.

The cleanest capacity match is **L1 vs O**: they are the *identical* network (same GRU,
same head) and differ *only* in how much of the sequence is fed in -- the last token
versus the full ordered history. Any O-over-L1 gain is therefore attributable to ordered
information, not to model size. C and U are the order-blind reference points, matched on
embedding / hidden width but necessarily non-recurrent. Per-view parameter counts are
reported (the SPEC 7 Pareto axis) rather than forced identical.

Targets (D22): ``'selection'`` (8-way softmax over families) and ``'outcome1'`` (6-way; the
**current action** is embedded and concatenated to the head, so ordered history can only
help by predicting the *outcome* after conditioning on the pitch actually thrown -- finding
#2, never finding #1).

Sequence material comes from :func:`pitchseq.sequences.build_sequences` (family / outcome
token ids with pad indices, the float channels incl. ``dvelo`` / ``dloc``, the mask and
lengths, all ``row_id``-aligned to the decision table); the static context is the audited
**C** view (:func:`pitchseq.states.build_view`) and the matchup branch is the **OM** block
(:func:`pitchseq.states.history_features`). No execution / label column ever reaches a
feature -- the same leakage discipline as the tabular workstreams.

PyTorch is an optional dependency; importing this module without it raises a clear
``ImportError`` pointing at ``pip install -e ".[deep]"``.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:  # torch is the [deep] extra (decision D46); fail loudly with the install hint.
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "WS6 (deep sequence models) requires PyTorch, which is not installed. "
        'Install the optional dependency group:  pip install -e ".[deep]"'
    ) from exc

from pitchseq.decision_table import leakage_audit
from pitchseq.eval.metrics import ablation_deltas, clustered_ci, log_loss_per_row
from pitchseq.families import FAMILIES
from pitchseq.outcomes import OUTCOME1
from pitchseq.sequences import SEQUENCE_FLOAT_CHANNELS, build_sequences
from pitchseq.states import build_view, history_features

__all__ = [
    "STATE_VIEWS",
    "TARGETS",
    "resolve_device",
    "seed_everything",
    "ContextEncoder",
    "MatchupEncoder",
    "FeatureEncoders",
    "SeqModel",
    "TransformerDemo",
    "make_ws6_model_factory",
    "ws6_order_ablation",
    "ws6_permutation_test",
    "ws6_null_acceptance",
    "ws6_positive_acceptance",
    "motif_repeat_probe",
    "temporal_split_masks",
]

STATE_VIEWS: tuple[str, ...] = ("C", "U", "L1", "O", "OM")
TARGETS: tuple[str, ...] = ("selection", "outcome1")

_FAM = list(FAMILIES)
_O1 = list(OUTCOME1)
N_FAM = len(_FAM)
N_O1 = len(_O1)
FAM_PAD = N_FAM  # pad index emitted by build_sequences for family tokens
O1_PAD = N_O1  # pad index emitted by build_sequences for outcome tokens
_FAM_INDEX = {f: i for i, f in enumerate(_FAM)}
_O1_INDEX = {o: i for i, o in enumerate(_O1)}
_N_CH = len(SEQUENCE_FLOAT_CHANNELS)
# Positions of the base (non-derived) float channels used when re-deriving dvelo / dloc.
_CH_SPEED = SEQUENCE_FLOAT_CHANNELS.index("release_speed")
_CH_X = SEQUENCE_FLOAT_CHANNELS.index("plate_x_br")
_CH_Z = SEQUENCE_FLOAT_CHANNELS.index("plate_z_norm")
_CH_DVELO = SEQUENCE_FLOAT_CHANNELS.index("dvelo")
_CH_DLOC = SEQUENCE_FLOAT_CHANNELS.index("dloc")
# Base (non-derived) channels: everything except the ordered dvelo / dloc diffs, which are
# re-derived after a reorder rather than permuted directly.
_CH_BASE = tuple(i for i in range(_N_CH) if i not in (_CH_DVELO, _CH_DLOC))


# =====================================================================================
# device / determinism helpers
# =====================================================================================

def resolve_device(device: str | None = "auto") -> "torch.device":
    """Resolve ``'auto'`` / ``'cpu'`` / ``'cuda'`` to a concrete :class:`torch.device`.

    ``'auto'`` (and ``None``) select CUDA when available, else CPU -- so the *same* code
    runs on Sean's desktop CPU and on a Colab T4 (decision D46).
    """
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_everything(seed: int) -> None:
    """Seed numpy + torch (and CUDA if present) for reproducible fits."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)


# =====================================================================================
# fitted feature encoders (numeric matrices for the static + matchup branches)
# =====================================================================================

def _numeric(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(frame), 0), dtype=np.float64)
    return frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def _fit_moments(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardisation moments robust to all-NaN columns (mean 0, std 1 for those)."""
    if arr.shape[1] == 0:
        return np.zeros(0), np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> handled below
        m = np.nanmean(arr, axis=0)
        s = np.nanstd(arr, axis=0)
    ok = np.isfinite(arr).any(axis=0)
    mean = np.where(ok, np.nan_to_num(m), 0.0)
    std = np.where(ok & (s >= 1e-6), np.nan_to_num(s, nan=1.0), 1.0)
    return mean, std


class ContextEncoder:
    """Turn the audited **C** view into a fixed-width numeric matrix.

    Continuous / ordinal columns are standardised (train mean / std); low-cardinality
    categoricals (``stand``, ``p_throws``, ``inning_topbot``, ``park``, ...) are one-hot
    encoded against a **fitted vocabulary**, so two separately built matrices encode
    columns identically and unseen categories map to an all-zero block. Booleans are
    treated as numeric 0/1. The vocabulary / moments are captured at :meth:`fit`.
    """

    def __init__(self) -> None:
        self.numeric_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.cat_vocab: dict[str, list[str]] = {}
        self.mean: list[float] = []
        self.std: list[float] = []
        self.width: int = 0

    @staticmethod
    def _is_categorical(series: pd.Series) -> bool:
        return isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object

    def fit(self, table: pd.DataFrame) -> "ContextEncoder":
        X, _ = build_view(table, "C")  # leakage-audited context view
        num, cat = [], {}
        for c in X.columns:
            s = X[c]
            if self._is_categorical(s):
                vals = pd.Series(s.astype("object")).dropna().astype(str).unique().tolist()
                cat[c] = sorted(vals)
            else:  # numeric or bool
                num.append(c)
        arr = _numeric(X, num)
        mean, std = _fit_moments(arr)
        self.numeric_cols, self.cat_cols, self.cat_vocab = num, list(cat), cat
        self.mean, self.std = mean.tolist(), std.tolist()
        self.width = len(num) + sum(len(v) for v in cat.values())
        return self

    def transform(self, table: pd.DataFrame) -> np.ndarray:
        X, _ = build_view(table, "C")
        arr = _numeric(X, self.numeric_cols)
        if arr.shape[1]:
            arr = np.nan_to_num((arr - np.asarray(self.mean)) / np.asarray(self.std))
        parts = [arr.astype(np.float32)]
        for c in self.cat_cols:
            vocab = self.cat_vocab[c]
            idx = {v: i for i, v in enumerate(vocab)}
            oh = np.zeros((len(X), len(vocab)), dtype=np.float32)
            vals = X[c].astype("object").astype(str).to_numpy()
            for i, v in enumerate(vals):
                j = idx.get(v)
                if j is not None:
                    oh[i, j] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((len(X), 0), np.float32)

    def state_dict(self) -> dict:
        return {
            "numeric_cols": self.numeric_cols,
            "cat_cols": self.cat_cols,
            "cat_vocab": self.cat_vocab,
            "mean": self.mean,
            "std": self.std,
            "width": self.width,
        }

    @classmethod
    def from_state(cls, d: dict) -> "ContextEncoder":
        enc = cls()
        enc.numeric_cols = list(d["numeric_cols"])
        enc.cat_cols = list(d["cat_cols"])
        enc.cat_vocab = {k: list(v) for k, v in d["cat_vocab"].items()}
        enc.mean = list(d["mean"])
        enc.std = list(d["std"])
        enc.width = int(d["width"])
        return enc


class MatchupEncoder:
    """Standardise the **OM** matchup-memory block into a numeric matrix.

    The OM block (:func:`pitchseq.states.history_features`) is all numeric (within-game
    batter-vs-pitcher counts + trailing batter-vs-family tendencies). Columns and train
    moments are captured at :meth:`fit`; :meth:`transform` re-standardises.
    """

    def __init__(self) -> None:
        self.cols: list[str] = []
        self.mean: list[float] = []
        self.std: list[float] = []
        self.width: int = 0

    def fit(self, table: pd.DataFrame) -> "MatchupEncoder":
        M = history_features(table)["OM"]
        leakage_audit(M)
        self.cols = list(M.columns)
        arr = _numeric(M, self.cols)
        mean, std = _fit_moments(arr)
        self.mean, self.std = mean.tolist(), std.tolist()
        self.width = len(self.cols)
        return self

    def transform(self, table: pd.DataFrame) -> np.ndarray:
        M = history_features(table)["OM"]
        arr = _numeric(M, self.cols)
        if arr.shape[1]:
            arr = np.nan_to_num((arr - np.asarray(self.mean)) / np.asarray(self.std))
        return arr.astype(np.float32)

    def state_dict(self) -> dict:
        return {"cols": self.cols, "mean": self.mean, "std": self.std, "width": self.width}

    @classmethod
    def from_state(cls, d: dict) -> "MatchupEncoder":
        enc = cls()
        enc.cols = list(d["cols"])
        enc.mean = list(d["mean"])
        enc.std = list(d["std"])
        enc.width = int(d["width"])
        return enc


class FeatureEncoders:
    """All fitted stats a view needs, in one place (static + matchup + float-channel scale).

    The float channels emitted by :func:`build_sequences` (velo ~90 mph, plate coords ~1-3,
    ``dvelo`` ~0-10) live on very different scales; the GRU trains far more stably when they
    are standardised, so per-channel means / stds over the *valid* (masked) training
    positions are captured here too. ``fit_match`` skips the (relatively expensive) matchup
    block for non-OM views.
    """

    def __init__(self, max_len: int = 15) -> None:
        self.max_len = int(max_len)
        self.ctx = ContextEncoder()
        self.match = MatchupEncoder()
        self.has_match = False
        self.ch_mean = np.zeros(_N_CH, dtype=np.float64)
        self.ch_std = np.ones(_N_CH, dtype=np.float64)
        self.fam_profile = np.zeros((N_FAM, _N_CH), dtype=np.float64)  # per-family mean channels

    def fit(self, table: pd.DataFrame, fit_match: bool = True) -> "FeatureEncoders":
        self.ctx.fit(table)
        if fit_match:
            self.match.fit(table)
            self.has_match = True
        seq = build_sequences(table, max_len=self.max_len)
        chans = np.stack([seq[c] for c in SEQUENCE_FLOAT_CHANNELS], axis=-1)  # (n, L, C)
        m = seq["mask"].astype(bool)
        flat = chans[m] if m.any() else np.zeros((1, _N_CH))
        self.ch_mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        self.ch_std = np.where(std < 1e-6, 1.0, std)
        # Per-family mean physical profile over valid history tokens (for realistic,
        # in-distribution motif-probe contexts). Families never thrown fall back to ch_mean.
        prof = np.tile(self.ch_mean, (N_FAM, 1))
        fam_flat = seq["family_idx"][m] if m.any() else np.zeros(0, dtype=np.int64)
        for f in range(N_FAM):
            sel = fam_flat == f
            if sel.any():
                prof[f] = flat[sel].mean(axis=0)
        self.fam_profile = prof
        return self

    def _scaled_channels(self, seq: dict) -> np.ndarray:
        chans = np.stack([seq[c] for c in SEQUENCE_FLOAT_CHANNELS], axis=-1).astype(np.float64)
        chans = (chans - self.ch_mean) / self.ch_std
        chans *= seq["mask"][..., None]  # zero the padded positions after scaling
        return chans.astype(np.float32)

    def make(self, table: pd.DataFrame, view: str, target: str) -> dict:
        """Build the numpy tensor bundle for ``table`` under ``view`` / ``target``."""
        seq = build_sequences(table, max_len=self.max_len)
        n = len(table)
        fam = table["family"].astype("object").to_numpy()
        action_idx = np.array([_FAM_INDEX.get(f, FAM_PAD) for f in fam], dtype=np.int64)
        # A family outside the 8-way vocab (only 'XX'-mapped nulls) would be FAM_PAD; the
        # action embedding has no pad row, so clamp such rare rows to 0 (documented).
        action_idx = np.where(action_idx >= N_FAM, 0, action_idx)
        bundle = {
            "static": self.ctx.transform(table),
            "fam_idx": seq["family_idx"].astype(np.int64),
            "out_idx": seq["outcome1_idx"].astype(np.int64),
            "float_ch": self._scaled_channels(seq),
            "mask": seq["mask"].astype(np.float32),
            "lengths": seq["lengths"].astype(np.int64),
            "action_idx": action_idx,
            "row_id": seq["row_id"],
            "n": n,
        }
        if view == "OM":
            if not self.has_match:
                self.match.fit(table)
                self.has_match = True
            bundle["matchup"] = self.match.transform(table)
        else:
            bundle["matchup"] = np.zeros((n, 0), dtype=np.float32)
        if target == "selection":
            y = np.array([_FAM_INDEX.get(f, 0) for f in fam], dtype=np.int64)
        else:
            o1 = table["outcome1"].astype("object").to_numpy()
            y = np.array([_O1_INDEX.get(o, 0) for o in o1], dtype=np.int64)
        bundle["y"] = y
        return bundle

    def state_dict(self) -> dict:
        return {
            "max_len": self.max_len,
            "ctx": self.ctx.state_dict(),
            "match": self.match.state_dict(),
            "has_match": self.has_match,
            "ch_mean": self.ch_mean.tolist(),
            "ch_std": self.ch_std.tolist(),
            "fam_profile": self.fam_profile.tolist(),
        }

    @classmethod
    def from_state(cls, d: dict) -> "FeatureEncoders":
        enc = cls(max_len=int(d["max_len"]))
        enc.ctx = ContextEncoder.from_state(d["ctx"])
        enc.match = MatchupEncoder.from_state(d["match"])
        enc.has_match = bool(d["has_match"])
        enc.ch_mean = np.asarray(d["ch_mean"], dtype=np.float64)
        enc.ch_std = np.asarray(d["ch_std"], dtype=np.float64)
        if "fam_profile" in d:
            enc.fam_profile = np.asarray(d["fam_profile"], dtype=np.float64)
        return enc


# =====================================================================================
# the capacity-matched network
# =====================================================================================

class _SeqNet(nn.Module):
    """One capacity-matched view network (see the module docstring for the ladder).

    All branches project to ``hidden``; present branches are concatenated and passed to a
    shared MLP head. For ``target='outcome1'`` the current-action embedding is concatenated
    to the head input (D22). ``L1`` and ``O`` share the *same* GRU -- L1 simply feeds the
    single last token.
    """

    def __init__(
        self,
        view: str,
        target: str,
        static_dim: int,
        match_dim: int,
        n_float_ch: int = _N_CH,
        embed_dim: int = 16,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.view = view
        self.target = target
        self.hidden = hidden
        self.fam_emb = nn.Embedding(N_FAM + 1, embed_dim, padding_idx=FAM_PAD)
        self.out_emb = nn.Embedding(N_O1 + 1, embed_dim, padding_idx=O1_PAD)
        self.tok_dim = 2 * embed_dim + n_float_ch

        self.static_enc = nn.Sequential(
            nn.Linear(static_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        n_branches = 1  # static, always present

        self.tok_proj = None
        self.gru = None
        self.match_enc = None
        if view == "U":
            self.tok_proj = nn.Sequential(nn.Linear(self.tok_dim, hidden), nn.ReLU())
            n_branches += 1
        elif view in ("L1", "O", "OM"):
            self.gru = nn.GRU(
                self.tok_dim,
                hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            n_branches += 1
        if view == "OM":
            self.match_enc = nn.Sequential(
                nn.Linear(match_dim, hidden), nn.ReLU(), nn.Dropout(dropout)
            )
            n_branches += 1

        self.act_emb = None
        act_dim = 0
        if target == "outcome1":
            self.act_emb = nn.Embedding(N_FAM, embed_dim)  # current action (D22)
            act_dim = embed_dim

        n_out = N_FAM if target == "selection" else N_O1
        head_in = n_branches * hidden + act_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out)
        )

    def _tokens(self, fam_idx, out_idx, float_ch):
        return torch.cat([self.fam_emb(fam_idx), self.out_emb(out_idx), float_ch], dim=-1)

    def _mean_pool(self, tok, mask):
        h = self.tok_proj(tok)  # (B, L, hidden)
        m = mask.unsqueeze(-1)
        summ = (h * m).sum(dim=1)
        cnt = m.sum(dim=1).clamp(min=1.0)
        return summ / cnt  # masked mean -> permutation-invariant

    def _last_token(self, tok, lengths):
        b = tok.shape[0]
        idx = (lengths - 1).clamp(min=0)
        last = tok[torch.arange(b, device=tok.device), idx]  # (B, tok_dim)
        valid = (lengths > 0).float().unsqueeze(-1)
        out, _ = self.gru((last * valid).unsqueeze(1))  # length-1 sequence
        return out[:, -1, :] * valid  # zero for empty histories

    def _full_seq(self, tok, lengths):
        eff = lengths.clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            tok, eff.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        h = h_n[-1]  # last layer's final hidden state (B, hidden)
        valid = (lengths > 0).float().unsqueeze(-1)
        return h * valid

    def forward(self, batch: dict) -> "torch.Tensor":
        parts = [self.static_enc(batch["static"])]
        if self.view in ("U", "L1", "O", "OM"):
            tok = self._tokens(batch["fam_idx"], batch["out_idx"], batch["float_ch"])
            if self.view == "U":
                parts.append(self._mean_pool(tok, batch["mask"]))
            elif self.view == "L1":
                parts.append(self._last_token(tok, batch["lengths"]))
            else:  # O / OM
                parts.append(self._full_seq(tok, batch["lengths"]))
        if self.view == "OM":
            parts.append(self.match_enc(batch["matchup"]))
        h = torch.cat(parts, dim=-1)
        if self.target == "outcome1":
            h = torch.cat([h, self.act_emb(batch["action_idx"])], dim=-1)
        return self.head(h)


class SeqModel:
    """Trainer / predictor wrapping one :class:`_SeqNet` (view x target).

    Two fit entry points:

    * :meth:`fit` -- the low-level contract: Adam + early stopping on a validation tensor
      bundle, over pre-built numpy bundles (the ``build_sequences`` dict + aligned static /
      matchup / action / label arrays produced by :meth:`FeatureEncoders.make`).
    * :meth:`fit_table` -- the ergonomic entry: fit / reuse encoders from a decision table,
      build the bundles, then call :meth:`fit`.

    :meth:`predict_proba` runs on a bundle; :meth:`predict_proba_table` on a decision table.
    :meth:`fit` records ``epochs`` / ``seconds`` / ``n_params`` / ``best_val`` in ``meta_``
    (runmeta-compatible), and :meth:`save` / :meth:`load` round-trip the net state_dict +
    a JSON meta carrying the encoders.
    """

    def __init__(
        self,
        view: str,
        target: str,
        *,
        embed_dim: int = 16,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 15,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 40,
        patience: int = 5,
        internal_val_frac: float = 0.2,
        seed: int = 0,
        device: str | None = "auto",
    ) -> None:
        if view not in STATE_VIEWS:
            raise ValueError(f"Unknown view {view!r}; expected one of {STATE_VIEWS}")
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}; expected one of {TARGETS}")
        self.view = view
        self.target = target
        self.hp = dict(
            embed_dim=embed_dim,
            hidden=hidden,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            internal_val_frac=internal_val_frac,
            seed=seed,
        )
        self.seed = seed
        self.device = resolve_device(device)
        self.net: _SeqNet | None = None
        self.encoders: FeatureEncoders | None = None
        self.classes_ = list(_FAM) if target == "selection" else list(_O1)
        self.meta_: dict = {}

    # -- net construction -------------------------------------------------------------

    def _build_net(self, static_dim: int, match_dim: int) -> _SeqNet:
        seed_everything(self.seed)
        net = _SeqNet(
            self.view,
            self.target,
            static_dim=static_dim,
            match_dim=match_dim,
            embed_dim=self.hp["embed_dim"],
            hidden=self.hp["hidden"],
            num_layers=self.hp["num_layers"],
            dropout=self.hp["dropout"],
        ).to(self.device)
        return net

    def _batch(self, bundle: dict, idx: np.ndarray) -> dict:
        dev = self.device
        return {
            "static": torch.as_tensor(bundle["static"][idx], device=dev),
            "fam_idx": torch.as_tensor(bundle["fam_idx"][idx], device=dev),
            "out_idx": torch.as_tensor(bundle["out_idx"][idx], device=dev),
            "float_ch": torch.as_tensor(bundle["float_ch"][idx], device=dev),
            "mask": torch.as_tensor(bundle["mask"][idx], device=dev),
            "lengths": torch.as_tensor(bundle["lengths"][idx], device=dev),
            "action_idx": torch.as_tensor(bundle["action_idx"][idx], device=dev),
            "matchup": torch.as_tensor(bundle["matchup"][idx], device=dev),
        }

    # -- low-level fit over bundles ---------------------------------------------------

    @staticmethod
    def _slice_bundle(bundle: dict, idx: np.ndarray) -> dict:
        out = {}
        for k, v in bundle.items():
            if k == "n":
                continue
            out[k] = v[idx] if isinstance(v, np.ndarray) else v
        out["n"] = int(len(idx))
        return out

    def fit(self, train_tensors: dict, val_tensors: dict | None = None, verbose: bool = False) -> "SeqModel":
        """Train on a bundle with Adam + early stopping.

        When ``val_tensors`` is ``None`` and ``internal_val_frac > 0``, a seeded slice of the
        training rows is held out for early stopping -- so the falsification refits (which
        have no external validation fold) still stop before overfitting, which is what lets
        the ordered O view generalise as well as its capacity-matched L1 twin.
        """
        import time

        if val_tensors is None and self.hp["internal_val_frac"] > 0 and int(train_tensors["n"]) >= 40:
            n_all = int(train_tensors["n"])
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n_all)
            cut = max(1, min(int(round(n_all * (1.0 - self.hp["internal_val_frac"]))), n_all - 1))
            val_tensors = self._slice_bundle(train_tensors, perm[cut:])
            train_tensors = self._slice_bundle(train_tensors, perm[:cut])
        static_dim = int(train_tensors["static"].shape[1])
        match_dim = int(train_tensors["matchup"].shape[1])
        self.net = self._build_net(static_dim, match_dim)
        opt = torch.optim.Adam(
            self.net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"]
        )
        loss_fn = nn.CrossEntropyLoss()
        n = int(train_tensors["n"])
        y_train = torch.as_tensor(train_tensors["y"], device=self.device)
        gen = torch.Generator().manual_seed(self.seed)
        bs = self.hp["batch_size"]

        best_val = float("inf")
        best_state = copy.deepcopy(self.net.state_dict())
        best_epoch = 0
        bad = 0
        epochs_run = 0
        start = time.perf_counter()
        for epoch in range(self.hp["max_epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=gen).numpy()
            for s in range(0, n, bs):
                idx = perm[s : s + bs]
                batch = self._batch(train_tensors, idx)
                opt.zero_grad()
                logits = self.net(batch)
                loss = loss_fn(logits, y_train[idx])
                loss.backward()
                opt.step()
            epochs_run = epoch + 1
            val_loss = self._eval_loss(val_tensors, loss_fn) if val_tensors is not None else None
            if val_tensors is not None:
                if val_loss < best_val - 1e-5:
                    best_val, best_epoch, bad = val_loss, epochs_run, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                    if bad >= self.hp["patience"]:
                        break
            if verbose:
                print(f"[{self.view}/{self.target}] epoch {epochs_run} val={val_loss}")
        if val_tensors is not None:
            self.net.load_state_dict(best_state)  # restore best-val weights
        seconds = time.perf_counter() - start
        self.meta_ = {
            "epochs": epochs_run,
            "best_epoch": best_epoch if val_tensors is not None else epochs_run,
            "seconds": float(seconds),
            "n_params": int(sum(p.numel() for p in self.net.parameters())),
            "best_val": float(best_val) if val_tensors is not None else None,
            "view": self.view,
            "target": self.target,
        }
        return self

    @torch.no_grad()
    def _eval_loss(self, tensors: dict, loss_fn) -> float:
        self.net.eval()
        n = int(tensors["n"])
        y = torch.as_tensor(tensors["y"], device=self.device)
        bs = self.hp["batch_size"]
        total = 0.0
        for s in range(0, n, bs):
            idx = np.arange(s, min(s + bs, n))
            logits = self.net(self._batch(tensors, idx))
            total += float(loss_fn(logits, y[idx])) * len(idx)
        return total / max(n, 1)

    # -- table-level convenience ------------------------------------------------------

    def fit_table(
        self,
        train_table: pd.DataFrame,
        val_table: pd.DataFrame | None = None,
        encoders: FeatureEncoders | None = None,
    ) -> "SeqModel":
        """Fit / reuse encoders from ``train_table``, build bundles, then :meth:`fit`."""
        if encoders is None:
            encoders = FeatureEncoders(max_len=self.hp["max_len"]).fit(
                train_table, fit_match=(self.view == "OM")
            )
        self.encoders = encoders
        train_b = encoders.make(train_table, self.view, self.target)
        val_b = encoders.make(val_table, self.view, self.target) if val_table is not None else None
        return self.fit(train_b, val_b)

    @torch.no_grad()
    def predict_proba(self, tensors: dict) -> np.ndarray:
        """Softmax probabilities ``(n, K)`` for a bundle (columns follow ``classes_``)."""
        if self.net is None:
            raise RuntimeError("SeqModel.predict_proba called before fit")
        self.net.eval()
        n = int(tensors["n"])
        bs = self.hp["batch_size"]
        out = []
        for s in range(0, n, bs):
            idx = np.arange(s, min(s + bs, n))
            logits = self.net(self._batch(tensors, idx))
            out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64) if out else np.zeros((0, len(self.classes_)))

    def predict_proba_table(self, table: pd.DataFrame) -> np.ndarray:
        if self.encoders is None:
            raise RuntimeError("SeqModel.predict_proba_table needs fitted encoders (use fit_table)")
        bundle = self.encoders.make(table, self.view, self.target)
        return self.predict_proba(bundle)

    # -- persistence ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save the net state_dict (``.pt``) + a JSON meta (hyperparams + encoders)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.net is None or self.encoders is None:
            raise RuntimeError("nothing to save: fit the model (with encoders) first")
        torch.save(self.net.state_dict(), out.with_suffix(".pt"))
        meta = {
            "view": self.view,
            "target": self.target,
            "hp": self.hp,
            "encoders": self.encoders.state_dict(),
            "static_dim": int(self.net.static_enc[0].in_features),
            "match_dim": int(self.net.match_enc[0].in_features) if self.net.match_enc else 0,
            "meta_": self.meta_,
        }
        with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=float)
        return out.with_suffix(".pt")

    @classmethod
    def load(cls, path: str | Path, device: str | None = "auto") -> "SeqModel":
        out = Path(path)
        with open(out.with_suffix(".json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        hp = meta["hp"]
        model = cls(meta["view"], meta["target"], device=device, **{
            k: hp[k] for k in (
                "embed_dim", "hidden", "num_layers", "dropout", "max_len", "lr",
                "weight_decay", "batch_size", "max_epochs", "patience",
                "internal_val_frac", "seed",
            )
        })
        model.encoders = FeatureEncoders.from_state(meta["encoders"])
        model.net = model._build_net(int(meta["static_dim"]), int(meta["match_dim"]))
        state = torch.load(out.with_suffix(".pt"), map_location=model.device, weights_only=True)
        model.net.load_state_dict(state)
        model.net.eval()
        model.meta_ = meta.get("meta_", {})
        return model


# =====================================================================================
# optional compact Transformer demo (SPEC 12.4: a *demo*, not an acceptance gate)
# =====================================================================================

class TransformerDemo(nn.Module):
    """A compact Transformer encoder over the ordered tokens (O-view only, ``outcome1``/
    ``selection``). SPEC 12.4 asks for this **only as an optional demo** for the long
    cross-PA test once the GRU shows signal -- it is deliberately kept out of the acceptance
    gates and exercised by a single smoke test.

    Same embedding sizes as the GRU; 1-2 encoder blocks, 2 attention heads, learned
    positional embeddings, masked mean-pool of the token states + the static branch.
    """

    def __init__(
        self,
        target: str = "outcome1",
        static_dim: int = 8,
        n_float_ch: int = _N_CH,
        embed_dim: int = 16,
        hidden: int = 64,
        n_heads: int = 2,
        n_blocks: int = 2,
        dropout: float = 0.1,
        max_len: int = 15,
    ) -> None:
        super().__init__()
        if target not in TARGETS:
            raise ValueError(f"Unknown target {target!r}")
        self.target = target
        self.fam_emb = nn.Embedding(N_FAM + 1, embed_dim, padding_idx=FAM_PAD)
        self.out_emb = nn.Embedding(N_O1 + 1, embed_dim, padding_idx=O1_PAD)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.tok_proj = nn.Linear(2 * embed_dim + n_float_ch, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=2 * hidden,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.static_enc = nn.Sequential(nn.Linear(static_dim, hidden), nn.ReLU())
        self.act_emb = nn.Embedding(N_FAM, embed_dim) if target == "outcome1" else None
        act_dim = embed_dim if target == "outcome1" else 0
        n_out = N_FAM if target == "selection" else N_O1
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + act_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out)
        )
        self.max_len = max_len

    def forward(self, batch: dict) -> "torch.Tensor":
        tok = torch.cat(
            [self.fam_emb(batch["fam_idx"]), self.out_emb(batch["out_idx"]), batch["float_ch"]], dim=-1
        )
        b, ln, _ = tok.shape
        h = self.tok_proj(tok)
        pos = torch.arange(ln, device=tok.device).clamp(max=self.max_len - 1)
        h = h + self.pos_emb(pos).unsqueeze(0)
        pad_mask = batch["mask"] < 0.5  # True where padded (ignored by attention)
        # rows with an all-pad (empty) history would make a full row masked; feed those
        # through a no-key-mask path is unnecessary -- pooled output is zeroed by the mask.
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        m = batch["mask"].unsqueeze(-1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        parts = [pooled, self.static_enc(batch["static"])]
        z = torch.cat(parts, dim=-1)
        if self.target == "outcome1":
            z = torch.cat([z, self.act_emb(batch["action_idx"])], dim=-1)
        return self.head(z)


# =====================================================================================
# the D17 falsification adapter (fit / predict_proba over DECISION TABLE slices)
# =====================================================================================

class _Ws6Adapter:
    """Falsification adapter (decision D17): ``fit(X, y)`` / ``predict_proba(X)`` where
    ``X`` is a **decision-table slice** (not a pre-built view matrix -- WS6 needs the raw
    ordered tokens, which the audited view matrices deliberately drop). The adapter builds
    the sequences / static / matchup features for its ``view`` internally.

    ``classes_`` is the fixed target vocabulary so the shared harness can re-align columns.
    Efficiency / caching (documented): each ``fit`` fits **one** :class:`FeatureEncoders`
    from its own ``X`` and stores it on the model; ``predict_proba`` reuses that fitted
    encoder set (no re-fit at predict time). Encoders are deliberately **not** cached across
    ``fit`` calls -- a permuted / re-sliced table (as the permutation control produces) must
    yield correspondingly-different sequence features, so a stale cache would be a bug.
    """

    def __init__(self, view: str, target: str, params: dict) -> None:
        self.view = view
        self.target = target
        self.params = dict(params or {})
        self.classes_ = list(_FAM) if target == "selection" else list(_O1)
        self.model: SeqModel | None = None

    def _seqmodel(self) -> SeqModel:
        keys = (
            "embed_dim", "hidden", "num_layers", "dropout", "max_len", "lr",
            "weight_decay", "batch_size", "max_epochs", "patience", "internal_val_frac",
            "seed", "device",
        )
        kw = {k: self.params[k] for k in keys if k in self.params}
        return SeqModel(self.view, self.target, **kw)

    def fit(self, X: pd.DataFrame, y=None) -> "_Ws6Adapter":
        # y is accepted for interface parity but the labels are read from the table slice
        # (build_sequences / the target column need the full decision-table row anyway).
        self.model = self._seqmodel().fit_table(X, val_table=None)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("adapter used before fit")
        return self.model.predict_proba_table(X)


def make_ws6_model_factory(params: dict | None = None, target: str = "selection"):
    """Return ``factory(view) -> _Ws6Adapter`` for the D17 falsification interface.

    Unlike the WS3 factory (which hands back a learner that consumes a pre-built tabular
    view matrix), the WS6 adapter consumes the **decision-table slice** directly and builds
    the ordered-token sequences itself -- the whole point of WS6 is the learned
    representation of the raw ordered tokens, which the leakage-audited tabular views drop.
    ``params`` overrides the compact defaults (embed 16 / hidden 64 / <=2 layers, D46) and,
    for the permutation refits, a small ``max_epochs`` keeps the battery affordable.

    Parameters
    ----------
    params : dict, optional
        :class:`SeqModel` hyperparameter overrides.
    target : {'selection', 'outcome1'}
        The prediction target the returned adapters train for.
    """
    merged = dict(params or {})

    def factory(view: str) -> _Ws6Adapter:
        return _Ws6Adapter(view, target, merged)

    return factory


# =====================================================================================
# WS6-native falsification (sequence-native order ablation + token-order permutation)
# =====================================================================================

def temporal_split_masks(table: pd.DataFrame, config: dict | None = None) -> dict:
    """Season-based train / val / test boolean masks (SPEC 7).

    Uses the config's first ``splits.rolling`` entry (train 2021-2023 / val 2024 / test
    2025) when the seasons are present, else falls back to a two-way parity split so tiny
    synthetic fixtures still work.
    """
    from pitchseq.config import load_config

    if config is None:
        config = load_config()
    seasons_present = set(table["season"].unique().tolist()) if "season" in table.columns else set()
    rolling = config.get("splits", {}).get("rolling", [])
    spec = rolling[0] if rolling else {"train": [2021, 2022, 2023], "val": [2024], "test": [2025]}
    tr = set(spec.get("train", [])) & seasons_present
    va = set(spec.get("val", [])) & seasons_present
    te = set(spec.get("test", [])) & seasons_present
    if tr and va:
        s = table["season"].to_numpy()
        return {
            "train": np.isin(s, list(tr)),
            "val": np.isin(s, list(va)),
            "test": np.isin(s, list(te)) if te else np.zeros(len(table), bool),
        }
    gp = table["game_pk"].to_numpy()
    train = gp % 2 == 0
    return {"train": train, "val": ~train, "test": np.zeros(len(table), bool)}


def _target_labels(target: str) -> list:
    return list(_FAM) if target == "selection" else list(_O1)


def _target_values(table: pd.DataFrame, target: str) -> np.ndarray:
    col = "family" if target == "selection" else "outcome1"
    return table[col].astype("object").to_numpy()


def ws6_order_ablation(
    table: pd.DataFrame,
    target: str,
    params: dict | None = None,
    train_mask=None,
    eval_mask=None,
    views=("C", "U", "L1", "O", "OM"),
) -> dict:
    """Fit one SeqModel per view on the train rows; return per-view eval losses + deltas.

    The sequence-native analogue of :func:`pitchseq.eval.falsification.order_ablation`: the
    factory is driven with **decision-table slices** (not tabular view matrices), so each
    view's learned encoder sees exactly the information SPEC 6 grants it. Returns per-row
    eval losses (for clustered CIs), per-view mean losses and the SPEC 6 ablation deltas.
    """
    n = len(table)
    if train_mask is None or eval_mask is None:
        m = temporal_split_masks(table)
        train_mask = m["train"] if train_mask is None else train_mask
        eval_mask = m["val"] if eval_mask is None else eval_mask
    train_pos = np.flatnonzero(np.asarray(train_mask, bool))
    eval_pos = np.flatnonzero(np.asarray(eval_mask, bool))
    labels = _target_labels(target)
    y = _target_values(table, target)

    factory = make_ws6_model_factory(params, target=target)
    per_row: dict[str, np.ndarray] = {}
    losses: dict[str, float] = {}
    for v in views:
        model = factory(v).fit(table.iloc[train_pos])
        proba = model.predict_proba(table.iloc[eval_pos])
        pr = log_loss_per_row(y[eval_pos], proba, labels)
        per_row[v] = pr
        losses[v] = float(pr.mean())
    deltas = ablation_deltas(losses) if {"U", "L1", "O"} <= set(losses) else {"delta_order": None, "delta_matchup": None}
    return {
        "target": target,
        "views": list(views),
        "losses": losses,
        "per_row": per_row,
        "eval_pos": eval_pos,
        "train_pos": train_pos,
        "deltas": deltas,
    }


def _delta_order_ci(ab: dict, table: pd.DataFrame, seed: int, n_boot: int) -> dict:
    """Clustered CI for ``min(Loss_U, Loss_L1) - Loss_O`` on the eval rows (D21)."""
    per_row, eval_pos = ab["per_row"], ab["eval_pos"]
    eval_df = table.iloc[eval_pos][["pitcher", "game_pk"]].reset_index(drop=True)
    lu, ll, lo = per_row["U"], per_row["L1"], per_row["O"]

    def metric(idx):
        return float(min(lu[idx].mean(), ll[idx].mean()) - lo[idx].mean())

    return clustered_ci(metric, eval_df, cluster="pitcher_game", n_boot=n_boot, seed=seed)


def _repeat_context_mask(table: pd.DataFrame) -> np.ndarray:
    """Rows whose two most-recent prior pitches are the **same family** (the habit is active).

    On exactly these rows the planted no-three-in-a-row habit suppresses that family, and it
    is precisely what the ordered O view can see and its capacity-matched L1 twin (last token
    only) cannot -- so it is where a learned ordered edge should show up cleanly.
    """
    t = table.sort_values(["pa_id", "pitch_number"], kind="stable")
    g = t.groupby("pa_id", sort=False)["family"]
    p1 = g.shift(1).astype("object")
    p2 = g.shift(2).astype("object")
    same = (p1.to_numpy() == p2.to_numpy()) & p1.notna().to_numpy() & p2.notna().to_numpy()
    return pd.Series(same, index=t.index).reindex(table.index).fillna(False).to_numpy().astype(bool)


def _l1_twin_delta_ci(ab: dict, table: pd.DataFrame, seed: int, n_boot: int, row_mask=None) -> dict:
    """Clustered CI for the capacity-matched twin delta ``Loss_L1 - Loss_O`` (D47 grammar
    gate: "O beats L1"), optionally restricted to ``row_mask`` (a full-table boolean)."""
    per_row, eval_pos = ab["per_row"], ab["eval_pos"]
    ll, lo = per_row["L1"], per_row["O"]
    ev = table.iloc[eval_pos].reset_index(drop=True)
    if row_mask is not None:
        keep = np.asarray(row_mask, bool)[eval_pos]
        ll, lo = ll[keep], lo[keep]
        ev = ev.loc[keep].reset_index(drop=True)
    if len(ll) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}

    def metric(idx):
        return float(ll[idx].mean() - lo[idx].mean())

    ci = clustered_ci(metric, ev[["pitcher", "game_pk"]], cluster="pitcher_game", n_boot=n_boot, seed=seed)
    ci["n"] = int(len(ll))
    return ci


def _permute_order_bundle(bundle: dict, rng: np.random.Generator, ch_mean, ch_std) -> dict:
    """Return a copy of ``bundle`` with each row's history **token order** shuffled.

    Within each row's valid prefix the base channels (family / outcome ids, plate_x_br,
    plate_z_norm, release_speed, pfx_x, pfx_z) are permuted together, then the ordered
    ``dvelo`` / ``dloc`` channels are **re-derived** from the shuffled sequence -- so the
    permuted bundle is exactly the same *multiset* of pitches in a scrambled order (U is
    unchanged; only order information is destroyed). Padded positions are untouched.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bundle.items()}
    fam, o1 = out["fam_idx"], out["out_idx"]
    ch = out["float_ch"].copy()  # (n, L, C) scaled
    lengths = out["lengths"]
    # Recover the *unscaled* base channels to re-derive diffs after shuffling.
    ch_un = ch * ch_std[None, None, :] + ch_mean[None, None, :]
    n = fam.shape[0]
    for i in range(n):
        L = int(lengths[i])
        if L <= 1:
            continue
        p = rng.permutation(L)
        fam[i, :L] = fam[i, :L][p]
        o1[i, :L] = o1[i, :L][p]
        for c in _CH_BASE:  # permute ALL per-pitch base channels together (keep them aligned)
            ch_un[i, :L, c] = ch_un[i, :L, c][p]
        # re-derive dvelo / dloc for the shuffled order (0 at position 0)
        speed = ch_un[i, :L, _CH_SPEED]
        x = ch_un[i, :L, _CH_X]
        z = ch_un[i, :L, _CH_Z]
        dv = np.zeros(L)
        dl = np.zeros(L)
        dv[1:] = np.diff(speed)
        dl[1:] = np.hypot(np.diff(x), np.diff(z))
        ch_un[i, :L, _CH_DVELO] = dv
        ch_un[i, :L, _CH_DLOC] = dl
    ch = (ch_un - ch_mean[None, None, :]) / ch_std[None, None, :]
    ch *= out["mask"][..., None]
    out["fam_idx"], out["out_idx"], out["float_ch"] = fam, o1, ch.astype(np.float32)
    return out


def ws6_permutation_test(
    table: pd.DataFrame,
    target: str = "outcome1",
    params: dict | None = None,
    n_permutations: int = 12,
    seed: int = 0,
    train_mask=None,
    eval_mask=None,
    reference_view: str = "U",
    ordered_view: str = "O",
    restrict_min_pitch: int = 3,
    alpha: float = 0.05,
) -> dict:
    r"""Token-order permutation control for a learned ordered edge (SPEC 8.3, D47).

    The observed statistic is the order edge ``edge = Loss(U) - Loss(O)`` on the eval rows.
    Under the null of no ordered dependence, scrambling each history's **token order**
    (preserving its multiset exactly, so U is invariant) and refitting O leaves the edge
    unchanged in expectation. The one-sided p-value is
    ``p = (1 + #{b: edge_b >= edge_obs}) / (1 + B)``.

    This is the sequence-native counterpart of the shared tabular history-permutation test:
    because the learned model consumes raw ordered tokens (not engineered order columns),
    order must be permuted at the *token* level and the O model refit each time. Kept modest
    (``n_permutations`` refits) by design.
    """
    if params is None:
        params = {}
    if train_mask is None or eval_mask is None:
        m = temporal_split_masks(table)
        train_mask = m["train"] if train_mask is None else train_mask
        eval_mask = m["val"] if eval_mask is None else eval_mask
    restrict = table["pitch_number"].to_numpy() >= restrict_min_pitch
    train_pos = np.flatnonzero(np.asarray(train_mask, bool) & restrict)
    eval_pos = np.flatnonzero(np.asarray(eval_mask, bool) & restrict)
    labels = _target_labels(target)
    y = _target_values(table, target)
    if len(train_pos) == 0 or len(eval_pos) == 0:
        return {"observed_edge": float("nan"), "p_value": 1.0, "fired": False, "null_edges": []}

    max_len = params.get("max_len", 15)
    enc = FeatureEncoders(max_len=max_len).fit(table.iloc[train_pos], fit_match=False)

    def _fit_eval_O(train_b, eval_b) -> np.ndarray:
        keys = ("embed_dim", "hidden", "num_layers", "dropout", "max_len", "lr",
                "weight_decay", "batch_size", "max_epochs", "patience", "internal_val_frac", "seed", "device")
        kw = {k: params[k] for k in keys if k in params}
        model = SeqModel(ordered_view, target, **kw)
        model.fit(train_b, None)
        proba = model.predict_proba(eval_b)
        return log_loss_per_row(y[eval_pos], proba, labels)

    # Reference (U) loss is order-invariant -> computed once.
    ref_model = make_ws6_model_factory(params, target=target)(reference_view)
    ref_model.fit(table.iloc[train_pos])
    ref_loss = float(log_loss_per_row(y[eval_pos], ref_model.predict_proba(table.iloc[eval_pos]), labels).mean())

    train_O = enc.make(table.iloc[train_pos], ordered_view, target)
    eval_O = enc.make(table.iloc[eval_pos], ordered_view, target)
    obs_O = float(_fit_eval_O(train_O, eval_O).mean())
    observed_edge = ref_loss - obs_O

    rng = np.random.default_rng(seed)
    null_edges = []
    for _ in range(n_permutations):
        tb = _permute_order_bundle(train_O, rng, enc.ch_mean, enc.ch_std)
        eb = _permute_order_bundle(eval_O, rng, enc.ch_mean, enc.ch_std)
        perm_O = float(_fit_eval_O(tb, eb).mean())
        null_edges.append(ref_loss - perm_O)
    null = np.asarray(null_edges, dtype=np.float64)
    p_value = float((1 + int(np.sum(null >= observed_edge))) / (1 + n_permutations))
    return {
        "observed_edge": observed_edge,
        "loss_reference": ref_loss,
        "loss_ordered": obs_O,
        "null_edges": null_edges,
        "null_edge_mean": float(null.mean()) if len(null) else float("nan"),
        "p_value": p_value,
        "fired": p_value < alpha,
        "n_train": int(len(train_pos)),
        "n_eval": int(len(eval_pos)),
    }


def ws6_null_acceptance(
    table: pd.DataFrame,
    params: dict | None = None,
    train_mask=None,
    eval_mask=None,
    n_permutations: int = 12,
    seed: int = 0,
    ci_boot: int = 60,
    alpha: float = 0.05,
) -> dict:
    """D47 null-world verdicts for the GRU.

    Two verdicts:

    * ``GRU_GRAMMAR_DETECTED`` -- on the **selection** target the ordered GRU-O beats L1
      with a clustered ``Delta_order`` CI strictly above 0 (the planted no-three-in-a-row
      selection habit is learned).
    * ``GRU_OUTCOME_QUIET`` -- on the **outcome1** target no ordered dependence is found:
      the ``Delta_order`` CI lower bound is not above 0 **and** the permutation test does
      not fire (D21 -- ``Delta_order`` is negatively biased under the null, so the criterion
      is "not significantly positive").
    """
    if train_mask is None or eval_mask is None:
        m = temporal_split_masks(table)
        train_mask = m["train"] if train_mask is None else train_mask
        eval_mask = m["val"] if eval_mask is None else eval_mask

    sel = ws6_order_ablation(table, "selection", params, train_mask, eval_mask, views=("C", "U", "L1", "O"))
    sel_agg_ci = _delta_order_ci(sel, table, seed=seed, n_boot=ci_boot)  # min(U,L1)-O, aggregate
    rc = _repeat_context_mask(table)
    grammar_ci = _l1_twin_delta_ci(sel, table, seed=seed, n_boot=ci_boot, row_mask=rc)  # L1-O on repeat rows
    grammar_detected = bool(grammar_ci["lo"] > 0)

    out = ws6_order_ablation(table, "outcome1", params, train_mask, eval_mask, views=("C", "U", "L1", "O"))
    out_ci = _delta_order_ci(out, table, seed=seed, n_boot=ci_boot)
    perm = ws6_permutation_test(table, "outcome1", params, n_permutations=n_permutations,
                                seed=seed, train_mask=train_mask, eval_mask=eval_mask, alpha=alpha)
    outcome_quiet = bool(out_ci["lo"] <= 0 and not perm["fired"])

    return {
        "world": "null",
        "grammar_verdict": "GRU_GRAMMAR_DETECTED" if grammar_detected else "GRU_GRAMMAR_MISSED",
        "outcome_verdict": "GRU_OUTCOME_QUIET" if outcome_quiet else "GRU_OUTCOME_NOISY",
        "grammar_detected": grammar_detected,
        "outcome_quiet": outcome_quiet,
        "selection_losses": sel["losses"],
        "selection_delta_order": sel["deltas"]["delta_order"],
        "selection_delta_order_ci": sel_agg_ci,
        "grammar_repeat_delta": grammar_ci["point"],
        "grammar_repeat_ci": grammar_ci,
        "grammar_repeat_n": grammar_ci.get("n", 0),
        "outcome_losses": out["losses"],
        "outcome_delta_order": out["deltas"]["delta_order"],
        "outcome_delta_order_ci": out_ci,
        "permutation_p": perm["p_value"],
        "permutation_fired": bool(perm["fired"]),
        "pass": bool(grammar_detected and outcome_quiet),
    }


def ws6_positive_acceptance(
    table: pd.DataFrame,
    params: dict | None = None,
    train_mask=None,
    eval_mask=None,
    n_permutations: int = 12,
    seed: int = 0,
    ci_boot: int = 60,
    velo_gap_threshold: float = 5.0,
    alpha: float = 0.05,
    planted_lift: float | None = None,
) -> dict:
    """D47 positive-world verdict for the GRU: the ordered whiff mechanism is recovered.

    ``GRU_MECHANISM_RECOVERED`` requires the **outcome1** GRU-O to beat U/L1 with a
    clustered ``Delta_order`` CI above 0 **and** the token-order permutation test to fire.
    The recovered-vs-planted ratio (predicted whiff lift on triggered vs untriggered eval
    rows, over the planted empirical lift) is reported like WS3 did.
    """
    from pitchseq.eval.falsification import velo_gap_prev_transition

    if train_mask is None or eval_mask is None:
        m = temporal_split_masks(table)
        train_mask = m["train"] if train_mask is None else train_mask
        eval_mask = m["val"] if eval_mask is None else eval_mask

    ab = ws6_order_ablation(table, "outcome1", params, train_mask, eval_mask, views=("C", "U", "L1", "O"))
    ci = _delta_order_ci(ab, table, seed=seed, n_boot=ci_boot)
    perm = ws6_permutation_test(table, "outcome1", params, n_permutations=n_permutations,
                                seed=seed, train_mask=train_mask, eval_mask=eval_mask, alpha=alpha)
    detected = bool(ci["lo"] > 0 and perm["fired"])

    # Recover the effect from the O-view outcome model's predicted whiff probability.
    train_pos = np.flatnonzero(np.asarray(train_mask, bool))
    eval_pos = np.flatnonzero(np.asarray(eval_mask, bool))
    o_model = make_ws6_model_factory(params, target="outcome1")("O")
    o_model.fit(table.iloc[train_pos])
    proba = o_model.predict_proba(table.iloc[eval_pos])
    whiff = proba[:, _O1_INDEX["whiff"]]
    gap = velo_gap_prev_transition(table)[eval_pos]
    trig = gap >= velo_gap_threshold
    recovered = float("nan")
    if trig.any() and (~trig).any():
        recovered = float(whiff[trig].mean() - whiff[~trig].mean())
    sign_ok = bool(np.isfinite(recovered) and recovered > 0)
    ratio = float(recovered / planted_lift) if (planted_lift and np.isfinite(recovered)) else float("nan")

    return {
        "world": "positive",
        "verdict": "GRU_MECHANISM_RECOVERED" if (detected and sign_ok) else "GRU_MECHANISM_MISSED",
        "order_effect_detected": detected,
        "outcome_losses": ab["losses"],
        "delta_order": ab["deltas"]["delta_order"],
        "delta_order_ci": ci,
        "permutation_p": perm["p_value"],
        "permutation_fired": bool(perm["fired"]),
        "recovered_whiff_lift": recovered,
        "planted_whiff_lift": planted_lift,
        "recovery_ratio": ratio,
        "recovered_sign_ok": sign_ok,
        "pass": bool(detected and sign_ok),
    }


# =====================================================================================
# motif-rediscovery probe (D47): does the null-world selection GRU learn WS2's motif?
# =====================================================================================

def _constant_context_static(model: SeqModel, table: pd.DataFrame) -> np.ndarray:
    """A single representative (mean) static context row for the probe."""
    static = model.encoders.ctx.transform(table)
    return static.mean(axis=0, keepdims=True).astype(np.float32)


def motif_repeat_probe(model: SeqModel, table: pd.DataFrame, families=None) -> dict:
    """Probe whether the null-world **selection** GRU-O rediscovers WS2's repeat-suppression
    motif (D47), with no gradients / attention analysis.

    For each family ``X`` we compare two synthetic two-token histories that **end in the same
    token** ``X`` and differ only in the *second-to-last* pitch:

    * ``[X, X]`` -- X repeated twice (the "repeat" context the habit acts on);
    * ``[Y, X]`` -- a different family then X (a "mixed" context), averaged over ``Y != X``.

    Both end in X, so a previous-pitch (L1) model would predict identically; only the ordered
    O model can react to the earlier token. The static context is held at the training mean
    and the float channels are held identical across the two conditions, so the *only* input
    difference is the second-to-last family id. If the planted no-three-in-a-row habit was
    learned, ``P(next = X | X, X)`` is **suppressed** below ``P(next = X | Y, X)`` -- the third
    consecutive same-family pitch is made less likely.

    Returns a per-family table plus the mean suppression (``p_mixed - p_repeat``, positive
    when the motif is present).
    """
    if model.target != "selection":
        raise ValueError("motif_repeat_probe expects a selection-target model")
    fams = list(families) if families is not None else list(_ENGINE_FAMILIES_DEFAULT)
    static0 = _constant_context_static(model, table)
    enc = model.encoders

    rows = []
    for x in fams:
        xi = _FAM_INDEX[x]
        p_repeat = _probe_next(model, static0, [xi, xi], enc)[xi]
        mixed = []
        for y in fams:
            if y == x:
                continue
            mixed.append(_probe_next(model, static0, [_FAM_INDEX[y], xi], enc)[xi])
        p_mixed = float(np.mean(mixed)) if mixed else float("nan")
        rows.append({
            "family": x,
            "p_same_after_repeat": float(p_repeat),
            "p_same_after_mixed": p_mixed,
            "suppression": float(p_mixed - p_repeat),
        })
    supp = np.array([r["suppression"] for r in rows], dtype=float)
    return {
        "per_family": rows,
        "mean_suppression": float(supp.mean()) if len(supp) else float("nan"),
        "frac_suppressed": float((supp > 0).mean()) if len(supp) else float("nan"),
        "motif_present": bool(len(supp) and supp.mean() > 0 and (supp > 0).mean() >= 0.5),
    }


_ENGINE_FAMILIES_DEFAULT = ("FF", "SI", "FC", "SL", "CU", "CH", "FS")


def _probe_next(model: SeqModel, static0: np.ndarray, hist_fams: list[int], enc: FeatureEncoders) -> np.ndarray:
    """Predicted next-family distribution given a synthetic ordered history ``hist_fams``.

    Each history token carries its family's **mean physical profile** (in-distribution, so
    the probe is not evaluated on out-of-support inputs), and the ordered ``dvelo`` / ``dloc``
    channels are re-derived from the constructed sequence (so ``[X, X]`` naturally has a
    near-zero velocity step and a mixed context a realistic one) before standardising.
    """
    L = enc.max_len
    fam_idx = np.full((1, L), FAM_PAD, dtype=np.int64)
    out_idx = np.full((1, L), O1_PAD, dtype=np.int64)
    ch_un = np.zeros((1, L, _N_CH), dtype=np.float64)
    mask = np.zeros((1, L), dtype=np.float32)
    h = len(hist_fams)
    for j, f in enumerate(hist_fams):
        fam_idx[0, j] = f
        out_idx[0, j] = _O1_INDEX["called_strike"]  # a neutral, fixed outcome token
        ch_un[0, j] = enc.fam_profile[f]
        mask[0, j] = 1.0
    if h >= 2:
        speed = ch_un[0, :h, _CH_SPEED]
        x = ch_un[0, :h, _CH_X]
        z = ch_un[0, :h, _CH_Z]
        ch_un[0, 1:h, _CH_DVELO] = np.diff(speed)
        ch_un[0, 1:h, _CH_DLOC] = np.hypot(np.diff(x), np.diff(z))
    float_ch = ((ch_un - enc.ch_mean) / enc.ch_std) * mask[..., None]
    bundle = {
        "static": static0,
        "fam_idx": fam_idx,
        "out_idx": out_idx,
        "float_ch": float_ch.astype(np.float32),
        "mask": mask,
        "lengths": np.array([h], dtype=np.int64),
        "action_idx": np.array([0], dtype=np.int64),
        "matchup": np.zeros((1, 0), dtype=np.float32),
        "n": 1,
    }
    return model.predict_proba(bundle)[0]
