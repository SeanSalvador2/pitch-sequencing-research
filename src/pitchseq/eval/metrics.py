"""Scoring metrics (SPEC ``8.2``). Pure functions over aligned arrays / frames.

Every function is documented with its exact formula. Conventions: ``proba`` is an
``(n, k)`` array of class probabilities whose columns follow a fixed ``labels`` order;
``y_true`` is either an array of class labels (mapped through ``labels``) or an array of
integer class indices. Losses are *lower is better*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..splits import cluster_bootstrap_indices

__all__ = [
    "log_loss",
    "log_loss_per_row",
    "brier_multiclass",
    "top_k_accuracy",
    "macro_f1",
    "per_pitcher_macro",
    "micro_mean",
    "reliability_table",
    "ece",
    "run_value_calibration",
    "run_value_mae_rmse",
    "skill_vs_baseline",
    "ablation_deltas",
    "clustered_ci",
]


def _to_indices(y_true, labels, k: int) -> np.ndarray:
    """Map labels/indices to integer class indices in ``[0, k)``."""
    y = np.asarray(y_true)
    if labels is not None:
        lookup = {lab: i for i, lab in enumerate(labels)}
        try:
            return np.array([lookup[v] for v in y], dtype=np.int64)
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"y_true value {exc.args[0]!r} not in labels {list(labels)}") from exc
    idx = y.astype(np.int64)
    if idx.min() < 0 or idx.max() >= k:
        raise ValueError(f"integer y_true out of range [0,{k}) -- pass labels= for label arrays")
    return idx


def log_loss_per_row(y_true, proba, labels=None, eps: float = 1e-12) -> np.ndarray:
    r"""Per-row multiclass log loss :math:`-\log \hat p_{i, y_i}` (probabilities clipped).

    The true-class probability is clipped to ``[eps, 1]`` before the log, bounding the
    contribution of a zero-probability true class.

    Parameters
    ----------
    y_true : array-like
        Class labels (if ``labels`` given) or integer indices.
    proba : array-like, shape (n, k)
        Predicted class probabilities in ``labels`` column order.
    labels : sequence, optional
        Column ordering for ``proba`` and the vocabulary for ``y_true``.
    eps : float, optional
        Clipping floor (default ``1e-12``).

    Returns
    -------
    numpy.ndarray, shape (n,)
        Per-row losses.
    """
    proba = np.asarray(proba, dtype=np.float64)
    n, k = proba.shape
    idx = _to_indices(y_true, labels, k)
    p = np.clip(proba[np.arange(n), idx], eps, 1.0)
    return -np.log(p)


def log_loss(y_true, proba, labels=None, eps: float = 1e-12, sample_weight=None) -> float:
    r"""Mean multiclass log loss.

    .. math:: L = -\frac{1}{\sum_i w_i} \sum_{i=1}^{n} w_i \log \hat p_{i, y_i}

    with :math:`\hat p` clipped to ``[eps, 1]`` (default weights :math:`w_i = 1`).
    """
    per_row = log_loss_per_row(y_true, proba, labels, eps)
    if sample_weight is None:
        return float(per_row.mean())
    w = np.asarray(sample_weight, dtype=np.float64)
    return float(np.average(per_row, weights=w))


def brier_multiclass(y_true, proba, labels=None) -> float:
    r"""Multiclass Brier score.

    .. math:: BS = \frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{k}\big(\hat p_{ic} - \mathbb{1}[y_i=c]\big)^2

    Ranges in :math:`[0, 2]`; lower is better.
    """
    proba = np.asarray(proba, dtype=np.float64)
    n, k = proba.shape
    idx = _to_indices(y_true, labels, k)
    onehot = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), idx] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def top_k_accuracy(y_true, proba, k: int = 1, labels=None) -> float:
    r"""Top-:math:`k` accuracy: fraction of rows whose true class is among the ``k`` highest
    predicted probabilities.

    .. math:: \mathrm{Acc}@k = \frac{1}{n}\sum_i \mathbb{1}\big[y_i \in \mathrm{top}_k(\hat p_i)\big]
    """
    proba = np.asarray(proba, dtype=np.float64)
    n, n_classes = proba.shape
    idx = _to_indices(y_true, labels, n_classes)
    k = min(k, n_classes)
    # Indices of the top-k columns per row (unordered).
    topk = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    hit = (topk == idx[:, None]).any(axis=1)
    return float(hit.mean())


def macro_f1(y_true, y_pred, labels=None) -> float:
    r"""Macro-averaged F1 over classes.

    For each class :math:`c`, :math:`F1_c = \frac{2 P_c R_c}{P_c + R_c}` (0 when the
    denominator is 0), and the macro average is :math:`\frac{1}{|C|}\sum_c F1_c`.

    Parameters
    ----------
    y_true, y_pred : array-like
        Predicted and true class labels (or indices). ``y_pred`` is typically
        ``argmax`` of a probability matrix mapped back to labels.
    labels : sequence, optional
        The class set to average over (default: sorted union of observed labels).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    f1s = []
    for c in labels:
        tp = int(np.sum((yt == c) & (yp == c)))
        fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s)) if f1s else 0.0


def per_pitcher_macro(per_row, pitcher) -> float:
    r"""Equal-weight pitcher average of a per-row metric.

    .. math:: M_{\text{macro}} = \frac{1}{|P|}\sum_{p\in P} \frac{1}{n_p}\sum_{i: \text{pit}_i=p} m_i

    i.e. average within each pitcher, then average across pitchers (each pitcher counts
    once). Contrast with :func:`micro_mean` (pitch-weighted).
    """
    per_row = np.asarray(per_row, dtype=np.float64)
    pit = np.asarray(pitcher)
    uniq = pd.unique(pit)
    means = [per_row[pit == p].mean() for p in uniq]
    return float(np.mean(means)) if means else float("nan")


def micro_mean(per_row, weights=None) -> float:
    r"""Pitch-weighted (micro) mean of a per-row metric.

    .. math:: M_{\text{micro}} = \frac{\sum_i w_i m_i}{\sum_i w_i} \quad (w_i = 1 \text{ by default}).
    """
    per_row = np.asarray(per_row, dtype=np.float64)
    if weights is None:
        return float(per_row.mean())
    return float(np.average(per_row, weights=np.asarray(weights, dtype=np.float64)))


def reliability_table(y_binary, p, n_bins: int = 10) -> pd.DataFrame:
    r"""Binned reliability (calibration) table for a binary target.

    Predictions are grouped into ``n_bins`` equal-width bins on ``[0, 1]``; each bin
    reports the mean predicted probability, the observed positive fraction, and the count.

    Returns
    -------
    pandas.DataFrame
        Columns ``bin_lo``, ``bin_hi``, ``mean_pred``, ``frac_pos``, ``count`` (empty bins
        omitted).
    """
    y = np.asarray(y_binary, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index in [0, n_bins-1]; the right edge (p == 1) folds into the last bin.
    b = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for j in range(n_bins):
        m = b == j
        if not m.any():
            continue
        rows.append(
            {
                "bin_lo": float(edges[j]),
                "bin_hi": float(edges[j + 1]),
                "mean_pred": float(p[m].mean()),
                "frac_pos": float(y[m].mean()),
                "count": int(m.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "mean_pred", "frac_pos", "count"])


def ece(y_binary, p, n_bins: int = 10) -> float:
    r"""Expected Calibration Error.

    .. math:: \mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{n}\,\big|\,\bar y_b - \bar p_b\,\big|

    where :math:`\bar y_b` is the observed positive fraction and :math:`\bar p_b` the mean
    predicted probability in bin :math:`b`. For multiclass confidence calibration, pass the
    max-probability confidence as ``p`` and the correctness indicator as ``y_binary``.
    """
    tbl = reliability_table(y_binary, p, n_bins)
    if tbl.empty:
        return float("nan")
    n = tbl["count"].sum()
    return float((tbl["count"] / n * (tbl["frac_pos"] - tbl["mean_pred"]).abs()).sum())


def run_value_calibration(observed, predicted) -> dict:
    r"""OLS calibration of observed reward on predicted expected reward.

    Fits :math:`R_i = a + b\,\hat R_i + \varepsilon_i` by ordinary least squares and returns
    the intercept :math:`a`, slope :math:`b`, and :math:`R^2`. Perfect calibration is
    :math:`a=0,\; b=1`.

    Returns
    -------
    dict
        ``{'intercept', 'slope', 'r2', 'n'}``.
    """
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return {"intercept": float("nan"), "slope": float("nan"), "r2": float("nan"), "n": int(n)}
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (a + b * x)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"intercept": float(a), "slope": float(b), "r2": float(r2), "n": int(n)}


def run_value_mae_rmse(observed, predicted) -> dict:
    r"""Mean absolute error and root-mean-square error of a reward prediction.

    .. math:: \mathrm{MAE} = \frac{1}{n}\sum_i |R_i - \hat R_i|, \qquad
              \mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_i (R_i - \hat R_i)^2}
    """
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    d = y[ok] - x[ok]
    if len(d) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return {"mae": float(np.abs(d).mean()), "rmse": float(np.sqrt((d ** 2).mean())), "n": int(len(d))}


def skill_vs_baseline(loss_model: float, loss_baseline: float) -> dict:
    r"""Skill of a model's loss relative to a baseline loss.

    .. math:: \text{ratio} = \frac{L_{\text{model}}}{L_{\text{base}}}, \quad
              \text{diff} = L_{\text{base}} - L_{\text{model}}, \quad
              \text{skill} = 1 - \frac{L_{\text{model}}}{L_{\text{base}}}

    Positive ``diff`` / ``skill`` means the model beats the baseline.
    """
    ratio = float(loss_model / loss_baseline) if loss_baseline != 0 else float("nan")
    return {
        "loss_ratio": ratio,
        "loss_diff": float(loss_baseline - loss_model),
        "skill": float("nan") if np.isnan(ratio) else 1.0 - ratio,
    }


def ablation_deltas(losses: dict) -> dict:
    r"""The SPEC ``6`` headline sequencing deltas from a ``{view: loss}`` mapping.

    .. math:: \Delta_{\text{order}} = \mathrm{Loss}(\min[U, L1]) - \mathrm{Loss}(O), \qquad
              \Delta_{\text{matchup}} = \mathrm{Loss}(O) - \mathrm{Loss}(OM)

    ``delta_order`` needs ``U``, ``L1`` and ``O``; ``delta_matchup`` needs ``O`` and ``OM``
    and is ``None`` when ``OM`` is absent. Positive ``delta_order`` means the fully ordered
    view beats the better of the unordered / previous-pitch views.
    """
    for req in ("U", "L1", "O"):
        if req not in losses:
            raise KeyError(f"ablation_deltas needs loss for view {req!r}; got {sorted(losses)}")
    delta_order = float(min(losses["U"], losses["L1"]) - losses["O"])
    delta_matchup = float(losses["O"] - losses["OM"]) if "OM" in losses else None
    return {"delta_order": delta_order, "delta_matchup": delta_matchup}


def clustered_ci(
    metric_fn,
    df: pd.DataFrame,
    cluster: str = "pitcher_game",
    n_boot: int = 200,
    seed: int | None = None,
    alpha: float = 0.05,
) -> dict:
    r"""Percentile confidence interval for a metric via cluster bootstrap.

    Resamples whole clusters (``pitcher_game`` by default) with replacement using
    :func:`pitchseq.splits.cluster_bootstrap_indices`, recomputing ``metric_fn`` on each
    replicate; the CI is the ``[alpha/2, 1-alpha/2]`` percentile interval of the replicate
    statistics. Reproducible for a fixed ``seed``.

    Parameters
    ----------
    metric_fn : callable
        ``metric_fn(pos_idx: np.ndarray) -> float`` where ``pos_idx`` are **positional**
        indices into ``df`` (0..len(df)-1). It must compute the statistic on that subset.
    df : pandas.DataFrame
        Rows to resample; must carry the columns the cluster spec needs (``pitcher`` and
        ``game_pk`` for ``pitcher_game``).
    cluster : str, optional
        Cluster unit (default ``"pitcher_game"``).
    n_boot : int, optional
        Number of bootstrap replicates (default 200).
    seed : int, optional
        Seed for reproducibility.
    alpha : float, optional
        Two-sided miscoverage (default 0.05 -> a 95% CI).

    Returns
    -------
    dict
        ``{'point', 'lo', 'hi', 'se', 'n_boot'}``.
    """
    df = df.reset_index(drop=True)
    point = float(metric_fn(np.arange(len(df))))
    stats = []
    for idx in cluster_bootstrap_indices(df, cluster=cluster, n_boot=n_boot, seed=seed):
        val = metric_fn(np.asarray(idx))
        if np.isfinite(val):
            stats.append(val)
    if not stats:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "n_boot": 0}
    arr = np.asarray(stats, dtype=np.float64)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "n_boot": int(len(arr)),
    }
