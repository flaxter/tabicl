"""NaN-safe attribution-evaluation metrics.

Each metric operates per-dataset (one vector of predictions of length
``p`` versus one vector of ground-truth labels of length ``p``) and the
module exposes a single :func:`aggregate_metrics` helper that averages
across valid datasets.

Labels from the Phase 3 sampler are NaN-padded past the active feature
count, and Head I's labels are entirely NaN on non-identifiable
samples. Every function here masks the NaN positions before doing
arithmetic and reports ``nan`` when the valid pool is too small to
compute anything meaningful (<3 positions or constant input).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Per-dataset primitives
# ---------------------------------------------------------------------------


def _finite_pair_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean mask of positions where both ``a`` and ``b`` are finite."""
    return np.isfinite(a) & np.isfinite(b)


def _rank(x: np.ndarray) -> np.ndarray:
    """Average-rank of each entry in ``x`` (ties share the mean rank)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # Handle ties: assign each tied group its mean rank.
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            mean_rank = (i + j - 1) / 2.0
            for idx in order[i:j]:
                ranks[idx] = mean_rank
        i = j
    return ranks


def spearman_per_dataset(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation on the finite overlap of ``pred`` and ``target``.

    Returns ``nan`` if the finite overlap has fewer than 3 positions or
    if either side is constant on the overlap.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape; got {pred.shape} vs {target.shape}"
        )

    mask = _finite_pair_mask(pred, target)
    if mask.sum() < 3:
        return float("nan")

    rp = _rank(pred[mask])
    rt = _rank(target[mask])
    if rp.std() == 0.0 or rt.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(rp, rt)[0, 1])


def pearson_per_dataset(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation on the finite overlap of ``pred`` and ``target``.

    Returns ``nan`` if the finite overlap has fewer than 3 positions or
    either side is constant.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape; got {pred.shape} vs {target.shape}"
        )

    mask = _finite_pair_mask(pred, target)
    if mask.sum() < 3:
        return float("nan")

    p = pred[mask]
    t = target[mask]
    if p.std() == 0.0 or t.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def topk_recall_per_dataset(
    pred: np.ndarray, target: np.ndarray, k: int
) -> float:
    """Fraction of the top-``k`` target features that appear in the top-``k`` prediction.

    Ranking is by absolute value on the finite subset of both arrays.
    Returns ``nan`` if the valid pool is smaller than ``k`` or if the
    target is constant-zero on the valid pool.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape; got {pred.shape} vs {target.shape}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")

    mask = _finite_pair_mask(pred, target)
    if mask.sum() < k:
        return float("nan")

    valid_idx = np.flatnonzero(mask)
    t_abs = np.abs(target[valid_idx])
    p_abs = np.abs(pred[valid_idx])
    if t_abs.max() == 0.0:
        return float("nan")

    # argpartition gives the indices into the valid_idx subset.
    top_t = set(valid_idx[np.argpartition(-t_abs, k - 1)[:k]].tolist())
    top_p = set(valid_idx[np.argpartition(-p_abs, k - 1)[:k]].tolist())
    return len(top_t & top_p) / k


# ---------------------------------------------------------------------------
# Batch aggregation
# ---------------------------------------------------------------------------


@dataclass
class HeadMetrics:
    """Aggregated metrics across a batch of datasets for one head."""

    spearman: float
    pearson: float
    top1: float
    top3: float
    top5: float
    n_valid: int


def aggregate_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    ks: Sequence[int] = (1, 3, 5),
) -> HeadMetrics:
    """Compute per-dataset metrics, then nan-mean across a batch.

    Parameters
    ----------
    preds, targets : ndarray of shape (n_datasets, max_features)
        NaN-padded per PLAN §Phase 3.
    ks : sequence of int
        Top-k values to compute. Must contain exactly three entries for
        the ``top1``/``top3``/``top5`` fields; pass ``(1, 3, 5)`` for
        the standard set. Smaller ``k`` values are fine — the default
        tuple is what the paper reports.

    Returns
    -------
    HeadMetrics
        Aggregated results. ``n_valid`` is the number of datasets
        contributing a finite Spearman correlation (the strictest
        aggregator).
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if preds.shape != targets.shape:
        raise ValueError(
            f"preds and targets must have the same shape; got {preds.shape} vs {targets.shape}"
        )
    if len(ks) != 3:
        raise ValueError(f"ks must have exactly 3 entries; got {ks}")

    n_datasets = preds.shape[0]
    spear = np.array([spearman_per_dataset(preds[i], targets[i]) for i in range(n_datasets)])
    pear = np.array([pearson_per_dataset(preds[i], targets[i]) for i in range(n_datasets)])
    topk = np.array(
        [
            [topk_recall_per_dataset(preds[i], targets[i], k) for k in ks]
            for i in range(n_datasets)
        ]
    )

    return HeadMetrics(
        spearman=float(np.nanmean(spear)) if np.isfinite(spear).any() else float("nan"),
        pearson=float(np.nanmean(pear)) if np.isfinite(pear).any() else float("nan"),
        top1=float(np.nanmean(topk[:, 0])) if np.isfinite(topk[:, 0]).any() else float("nan"),
        top3=float(np.nanmean(topk[:, 1])) if np.isfinite(topk[:, 1]).any() else float("nan"),
        top5=float(np.nanmean(topk[:, 2])) if np.isfinite(topk[:, 2]).any() else float("nan"),
        n_valid=int(np.isfinite(spear).sum()),
    )
