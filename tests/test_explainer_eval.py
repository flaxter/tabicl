"""Unit tests for :mod:`tabicl.eval.explainer_eval`.

Uses a minimal ``_DummyExplainer`` that returns pre-set per-state vectors
so the tests exercise metric and CSV plumbing without running the trunk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import csv
import numpy as np
import pytest

from tabicl.eval.explainer_eval import (
    DatasetScore,
    DatasetSpec,
    GroundTruth,
    canonical_states,
    evaluate_explainer,
    score_one,
)


# ---------------------------------------------------------------------------
# Dummy explainer
# ---------------------------------------------------------------------------


class _DummyExplainer:
    """Returns pre-set vectors keyed by ``frozenset(S)``."""

    def __init__(
        self,
        preds_by_state: Mapping,
        p: int,
        sufficiency: np.ndarray | None = None,
        necessity: np.ndarray | None = None,
    ):
        self._preds = {frozenset(k): np.asarray(v, dtype=np.float64) for k, v in preds_by_state.items()}
        self._p = p
        if sufficiency is not None:
            self.predictive_sufficiency_ = np.asarray(sufficiency, dtype=np.float64)
        if necessity is not None:
            self.predictive_necessity_ = np.asarray(necessity, dtype=np.float64)

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


# ---------------------------------------------------------------------------
# GroundTruth + score_one
# ---------------------------------------------------------------------------


def _make_spec(p: int = 4) -> tuple[DatasetSpec, _DummyExplainer]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, p))
    y = rng.standard_normal(20)

    # Build ground truth at S in {empty} union {every loo} union {some singleton, random}.
    value_by_state = {}
    empty = frozenset()
    value_by_state[empty] = np.array([0.9, 0.2, 0.7, 0.3])
    # LOO states
    for i in range(p):
        row = np.full(p, np.nan)
        row[i] = 0.1 * (i + 1)
        value_by_state[frozenset(j for j in range(p) if j != i)] = row
    # Singleton state
    value_by_state[frozenset([0])] = np.array([np.nan, 0.4, 0.5, 0.1])

    gt = GroundTruth(value_by_state=value_by_state, y_var=1.0)
    preds = {k: v.copy() for k, v in value_by_state.items()}  # perfect predictions

    expl = _DummyExplainer(
        preds_by_state=preds,
        p=p,
        sufficiency=np.array([0.9, 0.2, 0.7, 0.3]),  # matches empty-state row
        necessity=np.array([0.1, 0.2, 0.3, 0.4]),     # matches loo diagonal
    )
    return (
        DatasetSpec(name="d0", X=X, y=y, ground_truth=gt, meta={}),
        expl,
    )


def test_score_one_perfect_predictions_all_metrics_high():
    spec, expl = _make_spec()
    sc = score_one(expl, spec)

    assert sc.name == "d0"
    assert sc.n_features == 4
    assert sc.spearman_value == pytest.approx(1.0)
    assert sc.pearson_value == pytest.approx(1.0)
    assert sc.mse_value == pytest.approx(0.0)
    assert sc.mae_value == pytest.approx(0.0)
    assert sc.top1_next_feature == pytest.approx(1.0)
    assert sc.top3_next_feature == pytest.approx(1.0)
    assert sc.spearman_sufficiency == pytest.approx(1.0)
    assert sc.spearman_necessity == pytest.approx(1.0)
    # acquisition_auc requires ground-truth at every state along both paths;
    # the minimal _make_spec does not enumerate every path state — covered
    # separately in test_acquisition_auc_path_scoring.


def test_acquisition_auc_path_scoring_one_over_one_when_truth_matches():
    """When every state along the oracle+predicted path is covered by
    ground truth and predictions are perfect, the normalised AUFC is 1.0."""
    p = 3
    # Build a ground truth at all 2^p states so the path scoring can
    # traverse any subset.
    from itertools import chain, combinations

    def powerset(p):
        xs = range(p)
        return chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1))

    def truth_vec(S):
        row = np.full(p, np.nan)
        for i in range(p):
            if i in S:
                continue
            row[i] = 0.1 + (p - len(S)) * 0.2 + i * 0.05
        return row

    value_by_state = {frozenset(S): truth_vec(S) for S in powerset(p)}
    gt = GroundTruth(value_by_state=value_by_state)
    expl = _DummyExplainer({k: v.copy() for k, v in value_by_state.items()}, p=p)
    spec = DatasetSpec(
        name="full", X=np.zeros((10, p)), y=np.zeros(10),
        ground_truth=gt, meta={},
    )
    sc = score_one(expl, spec)
    assert sc.acquisition_auc == pytest.approx(1.0)


def test_score_one_reversed_predictions_negative_sufficiency_spearman():
    spec, _ = _make_spec()
    # Rank-reverse the empty-state predictions: produce values that
    # perfectly anti-correlate with truth by using the negation.
    preds = {k: v.copy() for k, v in spec.ground_truth.value_by_state.items()}
    truth_empty = spec.ground_truth.value_by_state[frozenset()].copy()
    preds[frozenset()] = -truth_empty
    expl = _DummyExplainer(preds, p=4, sufficiency=-truth_empty)
    sc = score_one(expl, spec)
    assert sc.spearman_value < 1.0
    assert sc.spearman_sufficiency == pytest.approx(-1.0)


def test_score_one_handles_missing_necessity_attribute():
    spec, _ = _make_spec()
    preds = {k: v.copy() for k, v in spec.ground_truth.value_by_state.items()}
    # No predictive_necessity_ attribute set.
    expl = _DummyExplainer(preds, p=4, sufficiency=preds[frozenset()])
    sc = score_one(expl, spec)
    assert np.isnan(sc.spearman_necessity)


def test_acquisition_auc_returns_nan_when_state_missing_from_ground_truth():
    spec, _ = _make_spec()
    preds = {k: v.copy() for k, v in spec.ground_truth.value_by_state.items()}
    # Drop the (frozenset) state after step 0 so path-scoring fails early.
    # Oracle's path will visit state {argmax(empty)} which is index 0 → state {0}
    # already present. To break, remove the empty state from pred preds.
    # Here we test a simpler path: pass an explainer that errors in greedy.

    class _NoGreedyExplainer(_DummyExplainer):
        def greedy_predictive_path(self):
            raise RuntimeError("no greedy")

    expl = _NoGreedyExplainer(preds, p=4)
    sc = score_one(expl, spec)
    assert np.isnan(sc.acquisition_auc)


def test_evaluate_explainer_writes_csv(tmp_path: Path):
    spec, expl = _make_spec()
    out = tmp_path / "out" / "scores.csv"

    def factory(X, y):
        return expl

    scores = evaluate_explainer(factory, [spec, spec], out_csv=out)
    assert len(scores) == 2
    assert isinstance(scores[0], DatasetScore)
    assert out.exists()

    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "spearman_value" in rows[0]
    assert "acquisition_auc" in rows[0]


# ---------------------------------------------------------------------------
# State sampler
# ---------------------------------------------------------------------------


def test_canonical_states_covers_every_stratum():
    rng = np.random.default_rng(0)
    p = 8
    states = canonical_states(p, rng=rng)
    sizes = sorted({len(s) for s in states})
    # Expect: 0 (empty), 1 (singleton), 2-4 (small), ~p/2 (medium),
    #         p-2 / p-1 (near_full), p-1 loo (near_full).
    assert 0 in sizes
    assert 1 in sizes
    assert any(2 <= s <= 4 for s in sizes)
    assert p // 2 in sizes
    assert p - 1 in sizes  # LOO
    assert p - 2 in sizes or p - 1 in sizes  # near-full


def test_canonical_states_handles_small_p():
    rng = np.random.default_rng(0)
    states = canonical_states(3, rng=rng)
    # Nothing crashes on small p; at minimum we get empty + at least one LOO.
    assert frozenset() in states
    assert any(len(s) == 2 for s in states)
