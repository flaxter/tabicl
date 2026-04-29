"""Tests for the predictive-oracle TabICLExplainer API.

These use a monkey-patched tiny TabICL (no HF-Hub download) to keep the
tests fast; we only care about the wrapper semantics, not trunk quality.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tabicl.model.heads import ConditionalPredictiveValueHead
from tabicl.sklearn.explainer import TabICLExplainer


# ---------------------------------------------------------------------------
# Tiny in-memory base estimator stub
# ---------------------------------------------------------------------------


class _DummyUniqueFilter:
    def __init__(self, keep: np.ndarray):
        self.features_to_keep_ = keep


class _DummyPreprocessor:
    def __init__(self, X_transformed: np.ndarray):
        self.X_transformed_ = X_transformed


class _DummyEnsembleGenerator:
    def __init__(self, X_filtered: np.ndarray, y: np.ndarray, keep: np.ndarray):
        self.preprocessors_ = {"quantile_normal": _DummyPreprocessor(X_filtered)}
        self.y_ = y
        self.unique_filter_ = _DummyUniqueFilter(keep)


class _DummyTabICL(nn.Module):
    """Trunk stub that emits deterministic per-column embeddings."""

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self._proj = nn.Linear(1, embed_dim, bias=False)
        nn.init.eye_(self._proj.weight.view(embed_dim, 1)[:1, :])  # identity-ish

    def forward(self, X, y_train=None, return_column_embeddings=False, inference_config=None):
        # X: (1, T, H). Take per-column mean over T, project to (1, H, E).
        col_mean = X.mean(dim=1, keepdim=True).transpose(1, 2)  # (1, H, 1)
        col_emb = torch.tanh(self._proj(col_mean))  # (1, H, E)
        logits = torch.zeros(1, 1, 2)  # dummy
        if return_column_embeddings:
            return logits, col_emb
        return logits


class _DummyBaseEstimator:
    def __init__(self, n_features: int = 5, n_kept: int = 5):
        self.n_features = n_features
        self.n_kept = n_kept

    def fit(self, X, y):
        self.n_features_in_ = self.n_features
        keep = np.zeros(self.n_features, dtype=bool)
        keep[: self.n_kept] = True
        X_filtered = np.asarray(X, dtype=np.float32)[:, :self.n_kept]
        y_arr = np.asarray(y, dtype=np.float32)
        self.ensemble_generator_ = _DummyEnsembleGenerator(X_filtered, y_arr, keep)
        self.model_ = _DummyTabICL(embed_dim=8)
        self.device_ = torch.device("cpu")
        self.inference_config_ = None
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fit_explainer(n_features: int = 5, n_kept: int = 5) -> TabICLExplainer:
    torch.manual_seed(0)
    base = _DummyBaseEstimator(n_features=n_features, n_kept=n_kept)
    head = ConditionalPredictiveValueHead(embed_dim=8)
    expl = TabICLExplainer(base_estimator=base, value_head=head)
    X = np.random.default_rng(0).standard_normal((30, n_features)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 2, size=30)
    expl.fit(X, y)
    return expl


# ---------------------------------------------------------------------------
# API semantics
# ---------------------------------------------------------------------------


def test_conditional_predictive_values_shape_is_n_features_in():
    expl = _fit_explainer(n_features=5)
    out = expl.conditional_predictive_values([])
    assert out.shape == (5,)


def test_conditional_predictive_values_NaN_inside_S():
    expl = _fit_explainer(n_features=5)
    out = expl.conditional_predictive_values([1, 3])
    assert np.isnan(out[1])
    assert np.isnan(out[3])
    # Positions outside S should be finite (dummy-head outputs are finite).
    assert np.isfinite(out[[0, 2, 4]]).all()


def test_conditional_predictive_values_NaN_for_dropped_features():
    """Constant-filtered features are not in the trunk's feature set and
    should come back as NaN regardless of S."""
    expl = _fit_explainer(n_features=5, n_kept=3)
    out = expl.conditional_predictive_values([])
    # Features 3 and 4 were filtered out — must be NaN.
    assert np.isnan(out[3])
    assert np.isnan(out[4])
    assert np.isfinite(out[[0, 1, 2]]).all()


def test_predictive_sufficiency_equals_empty_query():
    expl = _fit_explainer(n_features=5)
    empty_query = expl.conditional_predictive_values([])
    suff = expl.predictive_sufficiency_
    finite = np.isfinite(suff)
    assert np.allclose(suff[finite], empty_query[finite], atol=1e-6)


def test_predictive_necessity_shape_and_diagonal_logic():
    """predictive_necessity_[i] should equal conditional_predictive_values(
    [j for j != i])[i] for every valid (not filtered) feature."""
    expl = _fit_explainer(n_features=4, n_kept=4)
    nec = expl.predictive_necessity_
    assert nec.shape == (4,)
    for i in range(4):
        S = [j for j in range(4) if j != i]
        out = expl.conditional_predictive_values(S)
        assert np.isclose(nec[i], out[i], atol=1e-6), (
            f"necessity[{i}]={nec[i]:.4f} vs conditional[{i}|S]={out[i]:.4f}"
        )


def test_greedy_predictive_path_returns_valid_permutation_of_kept():
    expl = _fit_explainer(n_features=4, n_kept=4)
    path, gains = expl.greedy_predictive_path(k=4)
    assert sorted(path) == list(range(4)), f"path not a permutation: {path}"
    assert len(gains) == 4


def test_greedy_path_skips_dropped_features():
    expl = _fit_explainer(n_features=5, n_kept=3)
    path, _ = expl.greedy_predictive_path()
    # Must never pick a filtered-out feature.
    assert all(i < 3 for i in path), f"greedy path included dropped features: {path}"
    # Budget exhausts the kept features.
    assert len(path) == 3


def test_conditional_value_graph_shape_and_diag_false():
    expl = _fit_explainer(n_features=4, n_kept=4)
    g = expl.conditional_value_graph(threshold=-1e9)  # permissive, fill with True
    assert g.shape == (4, 4)
    assert not g.diagonal().any()


def test_invalid_S_index_raises():
    expl = _fit_explainer(n_features=3)
    with pytest.raises(ValueError):
        expl.conditional_predictive_values([5])  # out of range
