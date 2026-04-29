"""Unit tests for :mod:`tabicl.eval.explainer_eval`.

Uses a minimal ``_DummyExplainer`` that returns pre-set per-state vectors
so the tests exercise metric and CSV plumbing without running the trunk.
"""
from __future__ import annotations

import csv
from itertools import chain, combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pytest

from tabicl.eval.explainer_eval import (
    DatasetScore,
    EvalCase,
    EvalSuite,
    GroundTruth,
    evaluate_explainer,
    write_scores_csv,
)


# ---------------------------------------------------------------------------
# Dummy explainer
# ---------------------------------------------------------------------------


class _DummyExplainer:
    """Returns pre-set vectors keyed by ``frozenset(S)``.

    Implements the minimal explainer surface ``evaluate_explainer`` calls:
    ``fit``, ``conditional_predictive_values``, ``greedy_predictive_path``,
    plus the cached attributes ``predictive_sufficiency_``,
    ``predictive_necessity_``, and ``base_estimator_``.
    """

    def __init__(
        self,
        preds_by_state: Mapping,
        p: int,
        sufficiency: np.ndarray | None = None,
        necessity: np.ndarray | None = None,
    ):
        self._preds = {
            frozenset(k): np.asarray(v, dtype=np.float64)
            for k, v in preds_by_state.items()
        }
        self._p = p
        self.predictive_sufficiency_ = (
            np.asarray(sufficiency, dtype=np.float64)
            if sufficiency is not None
            else np.full(p, np.nan, dtype=np.float64)
        )
        self.predictive_necessity_ = (
            np.asarray(necessity, dtype=np.float64)
            if necessity is not None
            else np.full(p, np.nan, dtype=np.float64)
        )
        self.base_estimator_ = _ConstantPredictor()

    def fit(self, X, y) -> "_DummyExplainer":
        return self

    def conditional_predictive_values(self, S: Sequence[int]) -> np.ndarray:
        key = frozenset(int(i) for i in S)
        if key not in self._preds:
            raise KeyError(f"unknown state {set(key)}")
        return self._preds[key].copy()

    def greedy_predictive_path(self):
        # Deterministic path: walk the empty-state ranking.
        vec = self._preds.get(frozenset(), np.zeros(self._p))
        order = list(np.argsort(-np.nan_to_num(vec, nan=-np.inf)))
        gains = [float(vec[i]) for i in order]
        return order, gains


class _ConstantPredictor:
    """Stand-in base estimator for performance_curve_for_ranking; predicts
    a constant so the curve has well-defined values without needing a real
    sklearn fit/predict. Pre-fitted at construction so callers don't need
    to invoke ``fit`` first."""

    def __init__(self, y_mean: float = 0.0):
        self._y_mean = float(y_mean)
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self._y_mean = float(np.asarray(y, dtype=np.float64).mean())
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._y_mean, dtype=np.float64)

    def predict_proba(self, X):
        # Used by the classification path of performance_curve_for_ranking;
        # return a flat 50/50 vector so the curve stays bounded.
        return np.full((X.shape[0], 2), 0.5, dtype=np.float64)


# ---------------------------------------------------------------------------
# GroundTruth + evaluate_explainer
# ---------------------------------------------------------------------------


def _powerset(p: int):
    xs = range(p)
    return chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1))


def _make_case(p: int = 4) -> tuple[EvalCase, _DummyExplainer]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, p))
    y = rng.standard_normal(20)

    value_by_state: dict[frozenset, np.ndarray] = {}
    value_by_state[frozenset()] = np.array([0.9, 0.2, 0.7, 0.3])
    # Every LOO state — needed for the necessity vector.
    for i in range(p):
        row = np.full(p, np.nan)
        row[i] = 0.1 * (i + 1)
        value_by_state[frozenset(j for j in range(p) if j != i)] = row
    # Singleton state.
    value_by_state[frozenset([0])] = np.array([np.nan, 0.4, 0.5, 0.1])

    gt = GroundTruth(value_by_state=value_by_state, y_var=1.0)
    preds = {k: v.copy() for k, v in value_by_state.items()}  # perfect predictions
    expl = _DummyExplainer(
        preds_by_state=preds,
        p=p,
        sufficiency=np.array([0.9, 0.2, 0.7, 0.3]),       # matches empty-state row
        necessity=np.array([0.1, 0.2, 0.3, 0.4]),         # matches loo diagonal
    )
    case = EvalCase(
        dataset_id="d0",
        task="regression",
        X_train=X, y_train=y,
        X_test=X, y_test=y,
        ground_truth=gt,
    )
    return case, expl


def _evaluate_one(case: EvalCase, expl: _DummyExplainer) -> DatasetScore:
    suite = EvalSuite(name="unit", cases=[case])
    scores = evaluate_explainer(lambda c: expl, suite)
    assert len(scores) == 1
    return scores[0]


def test_evaluate_explainer_perfect_predictions_all_metrics_high():
    case, expl = _make_case()
    sc = _evaluate_one(case, expl)

    assert sc.dataset_id == "d0"
    assert sc.p == 4
    assert sc.spearman_value == pytest.approx(1.0)
    assert sc.pearson_value == pytest.approx(1.0)
    assert sc.mse_value == pytest.approx(0.0)
    assert sc.mae_value == pytest.approx(0.0)
    assert sc.top1_next_feature == pytest.approx(1.0)
    assert sc.top3_next_feature == pytest.approx(1.0)
    assert sc.spearman_sufficiency == pytest.approx(1.0)
    assert sc.spearman_necessity == pytest.approx(1.0)


def test_evaluate_explainer_acquisition_auc_finite_with_full_powerset():
    """Ground truth at every state along the path → AUC well-defined."""
    p = 3
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, p))
    y = (rng.standard_normal(10) > 0).astype(int)  # binary y for classification path

    def truth_vec(S):
        row = np.full(p, np.nan)
        for i in range(p):
            if i in S:
                continue
            row[i] = 0.1 + (p - len(S)) * 0.2 + i * 0.05
        return row

    value_by_state = {frozenset(S): truth_vec(S) for S in _powerset(p)}
    gt = GroundTruth(value_by_state=value_by_state)
    expl = _DummyExplainer({k: v.copy() for k, v in value_by_state.items()}, p=p)
    case = EvalCase(
        dataset_id="full", task="classification",
        X_train=X, y_train=y, X_test=X, y_test=y,
        ground_truth=gt,
    )
    sc = _evaluate_one(case, expl)
    assert np.isfinite(sc.acquisition_auc)


def test_evaluate_explainer_reversed_predictions_negative_sufficiency_spearman():
    case, _ = _make_case()
    preds = {k: v.copy() for k, v in case.ground_truth.value_by_state.items()}
    truth_empty = case.ground_truth.value_by_state[frozenset()].copy()
    preds[frozenset()] = -truth_empty
    expl = _DummyExplainer(
        preds, p=4,
        sufficiency=-truth_empty,
        necessity=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    sc = _evaluate_one(case, expl)
    assert sc.spearman_value < 1.0
    assert sc.spearman_sufficiency == pytest.approx(-1.0)


def test_evaluate_explainer_acquisition_auc_nan_when_greedy_path_crashes():
    """Greedy-acquisition support is optional. When ``greedy_predictive_path``
    raises, ``_score_one`` must still produce all the other metrics for the
    dataset (Spearman / Pearson / MAE / top-k / sufficiency / necessity) and
    set ``acquisition_auc=NaN`` rather than aborting the whole sweep.

    Locks down the policy restored in
    ``explainer_eval: tolerate greedy_predictive_path failures with NaN auc``
    (PR #4). If a future refactor drops the try/except again, this test
    fires."""
    case, _ = _make_case()
    preds = {k: v.copy() for k, v in case.ground_truth.value_by_state.items()}

    class _NoGreedyExplainer(_DummyExplainer):
        def greedy_predictive_path(self):
            raise RuntimeError("no greedy")

    expl = _NoGreedyExplainer(
        preds, p=4,
        sufficiency=preds[frozenset()].copy(),
        necessity=np.array([0.1, 0.2, 0.3, 0.4]),
    )
    sc = _evaluate_one(case, expl)
    # Acquisition AUC is NaN; everything else still scored on perfect preds.
    assert np.isnan(sc.acquisition_auc)
    assert sc.spearman_value == pytest.approx(1.0)
    assert sc.mae_value == pytest.approx(0.0)
    assert sc.spearman_sufficiency == pytest.approx(1.0)
    assert sc.spearman_necessity == pytest.approx(1.0)


def test_evaluate_explainer_writes_csv(tmp_path: Path):
    case, expl = _make_case()
    suite = EvalSuite(name="unit", cases=[case, case])
    out = tmp_path / "out" / "scores.csv"

    scores = evaluate_explainer(lambda c: expl, suite, out_csv=out)
    assert len(scores) == 2
    assert isinstance(scores[0], DatasetScore)
    assert out.exists()

    with out.open() as f:
        rows = list(csv.DictReader(f))
    # 2 dataset rows + 1 mean row.
    assert len(rows) == 3
    assert "spearman_value" in rows[0]
    assert "acquisition_auc" in rows[0]
    assert rows[-1]["dataset_id"] == "mean"
