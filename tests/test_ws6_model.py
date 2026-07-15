"""WS6 deep-model unit tests (SPEC 12.4; decisions D45-D47).

Pin the capacity-matched architecture contracts on tiny, fast, deterministic inputs:

* forward shapes per view x target;
* **U is order-invariant** (permuting the history leaves the output unchanged) while **O is
  not** -- the core of the SPEC 6 measurement;
* **L1 truncation** -- the L1 output is invariant to changes in any token *before* the last;
* **OM uses the matchup features** (zeroing them changes the output);
* **action conditioning** is present for the outcome target (changing the current action
  changes the output) and absent for the selection target (D22);
* training reduces loss on a separable problem, early stopping fires, a save/load round-trip
  reproduces predictions exactly, and fits are deterministic under a fixed seed;
* the D17 falsification factory yields a usable classifier;
* the optional Transformer demo runs (one forward + one epoch).

torch runs single-threaded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from pitchseq.decision_table import build_decision_table  # noqa: E402
from pitchseq.families import FAMILIES  # noqa: E402
from pitchseq.outcomes import OUTCOME1  # noqa: E402
from pitchseq.synth import make_null_world  # noqa: E402
from workstreams.ws6_deep_seq import model as M  # noqa: E402


# --- controlled batch builders ---------------------------------------------------------

def _batch(B=5, L=6, static_dim=5, match_dim=3, lengths=None, seed=1):
    g = np.random.default_rng(seed)
    if lengths is None:
        base = [4, 3, 5, 0, 2]
        lengths = np.array([base[i % len(base)] for i in range(B)])
    lengths = np.asarray(lengths, dtype=np.int64)
    lengths = np.clip(lengths, 0, L)
    fam = np.full((B, L), M.FAM_PAD, dtype=np.int64)
    out = np.full((B, L), M.O1_PAD, dtype=np.int64)
    fch = np.zeros((B, L, M._N_CH), np.float32)
    mask = np.zeros((B, L), np.float32)
    for i in range(B):
        for j in range(int(lengths[i])):
            fam[i, j] = g.integers(0, M.N_FAM)
            out[i, j] = g.integers(0, M.N_O1)
            fch[i, j] = g.standard_normal(M._N_CH)
            mask[i, j] = 1.0
    return {
        "static": g.standard_normal((B, static_dim)).astype(np.float32),
        "fam_idx": fam, "out_idx": out, "float_ch": fch, "mask": mask, "lengths": lengths,
        "action_idx": g.integers(0, M.N_FAM, B).astype(np.int64),
        "matchup": g.standard_normal((B, match_dim)).astype(np.float32), "n": B,
    }


def _forward(view, target, batch, hidden=16, embed=8, seed=0):
    M.seed_everything(seed)
    net = M._SeqNet(view, target, static_dim=batch["static"].shape[1],
                    match_dim=batch["matchup"].shape[1], embed_dim=embed, hidden=hidden)
    net.eval()
    tb = {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    with torch.no_grad():
        return net(tb).numpy()


def _copy(batch):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in batch.items()}


# --- forward shapes --------------------------------------------------------------------

@pytest.mark.parametrize("view", M.STATE_VIEWS)
@pytest.mark.parametrize("target", M.TARGETS)
def test_forward_shapes(view, target):
    batch = _batch()
    logits = _forward(view, target, batch)
    k = M.N_FAM if target == "selection" else M.N_O1
    assert logits.shape == (batch["n"], k)
    assert np.isfinite(logits).all()


# --- U order-invariance vs O order-sensitivity -----------------------------------------

def _permute_prefix(batch, row=0, seed=3):
    b = _copy(batch)
    L = int(b["lengths"][row])
    p = np.random.default_rng(seed).permutation(L)
    for key in ("fam_idx", "out_idx"):
        b[key][row, :L] = b[key][row, :L][p]
    b["float_ch"][row, :L] = b["float_ch"][row, :L][p]
    return b


def test_U_is_order_invariant():
    batch = _batch()
    perm = _permute_prefix(batch)
    u0, u1 = _forward("U", "selection", batch), _forward("U", "selection", perm)
    assert np.allclose(u0, u1, atol=1e-6)


def test_O_is_order_sensitive():
    batch = _batch()
    perm = _permute_prefix(batch)
    o0, o1 = _forward("O", "selection", batch), _forward("O", "selection", perm)
    assert not np.allclose(o0[0], o1[0], atol=1e-6)  # the permuted row changes


# --- L1 truncation ---------------------------------------------------------------------

def test_L1_invariant_to_tokens_before_last():
    """Changing a non-last token (length fixed) leaves the L1 output unchanged."""
    batch = _batch()
    b2 = _copy(batch)
    b2["fam_idx"][0, 0] = (b2["fam_idx"][0, 0] + 1) % M.N_FAM
    b2["float_ch"][0, 0] += 5.0  # perturb the earliest token hard
    l0, l1 = _forward("L1", "selection", batch), _forward("L1", "selection", b2)
    assert np.allclose(l0, l1, atol=1e-6)


def test_L1_changes_with_last_token():
    batch = _batch()
    b3 = _copy(batch)
    last = int(batch["lengths"][0]) - 1
    b3["fam_idx"][0, last] = (b3["fam_idx"][0, last] + 1) % M.N_FAM
    l0, l1 = _forward("L1", "selection", batch), _forward("L1", "selection", b3)
    assert not np.allclose(l0[0], l1[0], atol=1e-6)


# --- OM uses matchup -------------------------------------------------------------------

def test_OM_uses_matchup_features():
    batch = _batch()
    bz = _copy(batch)
    bz["matchup"] = np.zeros_like(bz["matchup"])
    m0, m1 = _forward("OM", "selection", batch), _forward("OM", "selection", bz)
    assert not np.allclose(m0, m1, atol=1e-6)


# --- action conditioning (D22) ---------------------------------------------------------

def test_outcome_conditions_on_action():
    batch = _batch()
    ba = _copy(batch)
    ba["action_idx"] = (ba["action_idx"] + 1) % M.N_FAM
    o0, o1 = _forward("O", "outcome1", batch), _forward("O", "outcome1", ba)
    assert not np.allclose(o0, o1, atol=1e-6)


def test_selection_ignores_action():
    batch = _batch()
    ba = _copy(batch)
    ba["action_idx"] = (ba["action_idx"] + 1) % M.N_FAM
    s0, s1 = _forward("O", "selection", batch), _forward("O", "selection", ba)
    assert np.allclose(s0, s1, atol=1e-8)


# --- training: separable problem, early stopping, save/load, determinism ---------------

def _toy_bundle(n=400, k=None, static_dim=None, seed=0, noise=0.0):
    """A separable selection problem: the class is one-hot-encoded in the static features."""
    k = M.N_FAM if k is None else k
    static_dim = k if static_dim is None else static_dim
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n).astype(np.int64)
    static = np.zeros((n, static_dim), np.float32)
    static[np.arange(n), y % static_dim] = 1.0
    static += noise * rng.standard_normal((n, static_dim)).astype(np.float32)
    L = 4
    return {
        "static": static,
        "fam_idx": np.full((n, L), M.FAM_PAD, np.int64),
        "out_idx": np.full((n, L), M.O1_PAD, np.int64),
        "float_ch": np.zeros((n, L, M._N_CH), np.float32),
        "mask": np.zeros((n, L), np.float32),
        "lengths": np.zeros(n, np.int64),
        "action_idx": np.zeros(n, np.int64),
        "matchup": np.zeros((n, 0), np.float32),
        "y": y, "n": n,
    }


def test_training_reduces_loss_on_separable_problem():
    train = _toy_bundle(n=400, seed=0)
    val = _toy_bundle(n=200, seed=1)
    model = M.SeqModel("C", "selection", hidden=16, embed_dim=8, max_epochs=30, patience=10,
                       lr=1e-2, batch_size=128, seed=0, device="cpu")
    model.fit(train, val)
    proba = model.predict_proba(val)
    loss = -np.log(np.clip(proba[np.arange(val["n"]), val["y"]], 1e-9, 1.0)).mean()
    assert loss < np.log(M.N_FAM) - 0.3  # comfortably below the uniform baseline


def test_early_stopping_fires():
    train = _toy_bundle(n=300, seed=0)
    val = _toy_bundle(n=150, seed=1)
    model = M.SeqModel("C", "selection", hidden=16, max_epochs=200, patience=3, lr=2e-2,
                       batch_size=128, seed=0, device="cpu")
    model.fit(train, val)
    assert model.meta_["epochs"] < 200  # converged and stopped early


def test_save_load_roundtrip_identical(tmp_path):
    raw, _ = make_null_world(n_games=12, seed=7, innings_per_game=6)
    tab = build_decision_table(raw)
    m = M.temporal_split_masks(tab)
    tr, va = tab.loc[m["train"]], tab.loc[m["val"]]
    model = M.SeqModel("O", "selection", hidden=24, embed_dim=8, max_epochs=5, max_len=8,
                       seed=0, device="cpu").fit_table(tr, va)
    p0 = model.predict_proba_table(va)
    model.save(tmp_path / "m")
    reloaded = M.SeqModel.load(tmp_path / "m", device="cpu")
    p1 = reloaded.predict_proba_table(va)
    assert np.array_equal(p0, p1)


def test_determinism_same_seed():
    train = _toy_bundle(n=300, seed=0)
    val = _toy_bundle(n=150, seed=1)

    def fit_predict():
        return M.SeqModel("C", "selection", hidden=16, max_epochs=8, lr=1e-2, batch_size=128,
                          seed=0, device="cpu").fit(train, val).predict_proba(val)

    assert np.array_equal(fit_predict(), fit_predict())


# --- factory adapter (the shared D17 interface) ----------------------------------------

def test_ws6_model_factory_interface():
    raw, _ = make_null_world(n_games=12, seed=7, innings_per_game=6)
    tab = build_decision_table(raw)
    m = M.temporal_split_masks(tab)
    tr, va = tab.loc[m["train"]], tab.loc[m["val"]]
    factory = M.make_ws6_model_factory(
        {"hidden": 16, "embed_dim": 8, "max_epochs": 4, "max_len": 8, "seed": 0, "device": "cpu"},
        target="selection",
    )
    clf = factory("O").fit(tr)
    proba = clf.predict_proba(va)
    assert proba.shape == (len(va), len(FAMILIES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert list(clf.classes_) == list(FAMILIES)


def test_factory_outcome_target_shape():
    raw, _ = make_null_world(n_games=12, seed=7, innings_per_game=6)
    tab = build_decision_table(raw)
    m = M.temporal_split_masks(tab)
    factory = M.make_ws6_model_factory(
        {"hidden": 16, "embed_dim": 8, "max_epochs": 4, "max_len": 8, "seed": 0, "device": "cpu"},
        target="outcome1",
    )
    clf = factory("O").fit(tab.loc[m["train"]])
    proba = clf.predict_proba(tab.loc[m["val"]])
    assert proba.shape == (int(m["val"].sum()), len(OUTCOME1))


# --- unknown view/target guards --------------------------------------------------------

def test_unknown_view_raises():
    with pytest.raises(ValueError):
        M.SeqModel("ZZ", "selection")


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        M.SeqModel("O", "banana")


# --- optional Transformer demo (SPEC 12.4: a demo, not a gate) -------------------------

def test_transformer_demo_smoke():
    batch = _batch(B=6, L=8)
    batch["y"] = np.random.default_rng(0).integers(0, M.N_O1, batch["n"]).astype(np.int64)
    M.seed_everything(0)
    demo = M.TransformerDemo(target="outcome1", static_dim=batch["static"].shape[1],
                             n_blocks=1, n_heads=2, hidden=32, embed_dim=8, max_len=8)
    tb = {k: (torch.as_tensor(v) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    logits = demo(tb)
    assert logits.shape == (batch["n"], M.N_O1)
    # one optimisation step runs without error
    opt = torch.optim.Adam(demo.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()(demo(tb), torch.as_tensor(batch["y"]))
    opt.zero_grad(); loss.backward(); opt.step()
    assert np.isfinite(float(loss.detach()))
