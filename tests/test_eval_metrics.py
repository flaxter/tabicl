"""Phase 6e — attribution-metric unit tests.

Covers the pure-numpy primitives in :mod:`tabicl.eval.metrics`:

- Perfect predictions → Spearman = Pearson = 1.0.
- Monotone-but-nonlinear predictions → Spearman = 1.0 but Pearson < 1.
- Reversed predictions → Spearman = -1.0.
- NaN positions in either side are masked before the correlation.
- ``nan`` is returned when the valid overlap is too small or constant.
- Top-k recall matches the expected set-intersection cardinality.
- Batch :func:`aggregate_metrics` takes the nan-mean across datasets.
"""
from __future__ import annotations

import numpy as np
import pytest

from tabicl.eval.metrics import (
    aggregate_metrics,
    pearson_per_dataset,
    spearman_per_dataset,
    topk_recall_per_dataset,
)


# ---------------------------------------------------------------------------
# Spearman / Pearson per-dataset
# ---------------------------------------------------------------------------


def test_spearman_perfect_agreement_is_one():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert spearman_per_dataset(p, t) == pytest.approx(1.0)
    assert pearson_per_dataset(p, t) == pytest.approx(1.0)


def test_spearman_reversed_is_negative_one():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert spearman_per_dataset(p, t) == pytest.approx(-1.0)


def test_spearman_robust_to_nonlinearity_but_pearson_is_not():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = p ** 3  # monotone nonlinear transform → preserves ranks, breaks linearity
    assert spearman_per_dataset(p, t) == pytest.approx(1.0)
    # Pearson on a perfect cubic is still high but not exactly 1.
    assert pearson_per_dataset(p, t) < 1.0
    assert pearson_per_dataset(p, t) > 0.9


def test_nan_positions_are_masked():
    p = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    t = np.array([10.0, 20.0, 30.0, 40.0, np.nan])
    # Valid overlap: first three positions.
    assert spearman_per_dataset(p, t) == pytest.approx(1.0)
    assert pearson_per_dataset(p, t) == pytest.approx(1.0)


def test_nan_when_too_few_finite_overlap():
    p = np.array([1.0, np.nan, 3.0])
    t = np.array([np.nan, 2.0, np.nan])
    assert np.isnan(spearman_per_dataset(p, t))
    assert np.isnan(pearson_per_dataset(p, t))


def test_nan_when_constant_on_overlap():
    p = np.array([1.0, 1.0, 1.0, 1.0])
    t = np.array([10.0, 20.0, 30.0, 40.0])
    assert np.isnan(spearman_per_dataset(p, t))
    assert np.isnan(pearson_per_dataset(p, t))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        spearman_per_dataset(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# Top-k recall
# ---------------------------------------------------------------------------


def test_topk_recall_perfect_overlap():
    # Top-3 by |·|: positions 0, 2, 3 in both.
    p = np.array([5.0, 0.1, -4.0, 3.0, 0.2])
    t = np.array([10.0, 0.5, -8.0, 7.0, 0.1])
    assert topk_recall_per_dataset(p, t, k=3) == pytest.approx(1.0)
    assert topk_recall_per_dataset(p, t, k=1) == pytest.approx(1.0)


def test_topk_recall_partial_overlap():
    # Target top-2: positions 0, 2.  Pred top-2: positions 3, 0. Overlap = {0}.
    p = np.array([5.0, 0.1, 0.2, 6.0])
    t = np.array([10.0, 0.5, 8.0, 0.1])
    assert topk_recall_per_dataset(p, t, k=2) == pytest.approx(0.5)


def test_topk_recall_nan_when_valid_pool_too_small():
    p = np.array([5.0, np.nan])
    t = np.array([10.0, np.nan])
    assert np.isnan(topk_recall_per_dataset(p, t, k=3))


def test_topk_recall_nan_when_target_all_zero():
    p = np.array([5.0, 3.0, 1.0])
    t = np.array([0.0, 0.0, 0.0])
    assert np.isnan(topk_recall_per_dataset(p, t, k=2))


def test_topk_recall_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        topk_recall_per_dataset(np.zeros(3), np.zeros(3), k=0)


# ---------------------------------------------------------------------------
# Aggregation across a batch
# ---------------------------------------------------------------------------


def test_aggregate_metrics_averages_per_dataset_scores():
    # Two datasets: first perfect, second reversed → mean Spearman = 0.
    preds = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    targets = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    m = aggregate_metrics(preds, targets)
    assert m.spearman == pytest.approx(0.0)
    assert m.pearson == pytest.approx(0.0)
    assert m.n_valid == 2


def test_aggregate_metrics_handles_all_nan_rows():
    # Dataset 0 valid, dataset 1 entirely NaN in targets → skipped.
    preds = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    targets = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    m = aggregate_metrics(preds, targets)
    assert m.spearman == pytest.approx(1.0)
    assert m.n_valid == 1


def test_aggregate_metrics_all_nan_gives_nan_output():
    preds = np.full((2, 4), np.nan)
    targets = np.full((2, 4), np.nan)
    m = aggregate_metrics(preds, targets)
    assert np.isnan(m.spearman)
    assert np.isnan(m.pearson)
    assert np.isnan(m.top1)
    assert m.n_valid == 0


def test_aggregate_metrics_requires_three_ks():
    with pytest.raises(ValueError):
        aggregate_metrics(np.zeros((1, 3)), np.zeros((1, 3)), ks=(1, 3))
