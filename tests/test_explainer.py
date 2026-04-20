"""Phase 5 — scikit-learn attribution API tests.

Verify interface correctness of :class:`TabICLExplainer`:

- ``fit`` populates the static attribution attributes
  (``observational_relevance_``, ``interventional_effects_``,
  ``identifiability_scope_``) with the right shapes and types.
- ``predict`` / ``predict_proba`` delegate unchanged to the base
  estimator.
- Conditioning-set semantics for Head C: entries inside ``S`` are NaN,
  entries for constant-dropped features are NaN, threshold extremes of
  ``conditional_relevance_graph`` produce all-False / all-True matrices.
- Head-checkpoint round-trip: save + load via
  ``heads_checkpoint_path`` gives bit-identical Head A scores.
- Same flow works end-to-end for ``TabICLRegressor``.

These tests do **not** verify attribution accuracy — the heads here are
freshly-initialised and untrained. Attribution quality is Phase 6.

To avoid depending on the HF Hub pretrained-checkpoint download, each
test builds a tiny in-memory ``TabICL`` model and monkey-patches the
base estimator's ``_load_model`` to install it.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from tabicl import TabICLClassifier, TabICLRegressor, TabICLExplainer
from tabicl.model.heads import (
    ObservationalHead,
    InterventionalHead,
    ConditionalHead,
)
from tabicl.model.tabicl import TabICL


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


EMBED_DIM = 32


def _build_tiny_tabicl(max_classes: int, seed: int = 0) -> TabICL:
    torch.manual_seed(seed)
    return TabICL(
        max_classes=max_classes,
        embed_dim=EMBED_DIM,
        col_num_blocks=2,
        col_nhead=4,
        col_num_inds=8,
        icl_num_blocks=2,
        icl_nhead=4,
        row_num_blocks=2,
        row_nhead=4,
        row_num_cls=2,
        ff_factor=2,
    )


def _install_tiny_model(estimator, max_classes: int) -> None:
    """Replace the estimator's ``_load_model`` to use an in-memory tiny TabICL.

    Avoids downloading a pretrained checkpoint from HF Hub; Phase 5
    tests care about the attribution interface, not trunk weights.
    """
    tiny = _build_tiny_tabicl(max_classes=max_classes)

    def _fake_load_model(self=estimator):
        self.model_path_ = None
        self.model_ = tiny
        self.model_config_ = {
            "max_classes": max_classes,
            "embed_dim": EMBED_DIM,
        }
        self.model_.eval()

    estimator._load_model = _fake_load_model


def _fresh_heads(embed_dim: int = EMBED_DIM) -> dict:
    torch.manual_seed(0)
    return {
        "observational": ObservationalHead(embed_dim=embed_dim),
        "interventional": InterventionalHead(embed_dim=embed_dim),
        "conditional": ConditionalHead(embed_dim=embed_dim),
    }


def _small_classification_dataset(n=40, p=5, seed=0):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=p, n_redundant=0,
        n_classes=2, random_state=seed,
    )
    X = X.astype(np.float32)
    return X[:30], y[:30], X[30:]


def _small_regression_dataset(n=40, p=5, seed=0):
    X, y = make_regression(n_samples=n, n_features=p, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    return X[:30], y[:30], X[30:]


def _build_fitted_classifier_explainer(X_train, y_train, heads=None, **kwargs):
    clf = TabICLClassifier(n_estimators=2, random_state=0, verbose=False)
    _install_tiny_model(clf, max_classes=10)  # generous upper bound
    expl = TabICLExplainer(
        base_estimator=clf,
        heads=heads if heads is not None else _fresh_heads(),
        **kwargs,
    )
    expl.fit(X_train, y_train)
    return expl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_explainer_fit_populates_attribution_attributes():
    X_train, y_train, _ = _small_classification_dataset()
    expl = _build_fitted_classifier_explainer(X_train, y_train)

    assert expl.observational_relevance_.shape == (X_train.shape[1],)
    assert expl.interventional_effects_.shape == (X_train.shape[1],)
    assert expl.observational_relevance_.dtype == np.float64
    assert expl.interventional_effects_.dtype == np.float64
    assert isinstance(expl.identifiability_scope_, str)
    assert len(expl.identifiability_scope_) > 0


def test_explainer_predict_delegates_to_base():
    X_train, y_train, X_test = _small_classification_dataset()
    expl = _build_fitted_classifier_explainer(X_train, y_train)

    pred_base = expl.base_estimator_.predict(X_test)
    pred_expl = expl.predict(X_test)
    np.testing.assert_array_equal(pred_base, pred_expl)

    proba_base = expl.base_estimator_.predict_proba(X_test)
    proba_expl = expl.predict_proba(X_test)
    np.testing.assert_allclose(proba_base, proba_expl)


def test_unique_filter_gives_nan_on_constant_features():
    X_train, y_train, _ = _small_classification_dataset(p=5)
    # Make column 2 constant — UniqueFeatureFilter will drop it.
    X_train = X_train.copy()
    X_train[:, 2] = 3.14

    expl = _build_fitted_classifier_explainer(X_train, y_train)

    assert np.isnan(expl.observational_relevance_[2])
    assert np.isnan(expl.interventional_effects_[2])
    # Surviving features must be finite.
    kept = [i for i in range(5) if i != 2]
    assert np.isfinite(expl.observational_relevance_[kept]).all()
    assert np.isfinite(expl.interventional_effects_[kept]).all()


def test_marginal_conditional_contributions_masks_S():
    X_train, y_train, _ = _small_classification_dataset(p=5)
    expl = _build_fitted_classifier_explainer(X_train, y_train)

    scores = expl.marginal_conditional_contributions(S=[1])
    assert scores.shape == (5,)
    assert np.isnan(scores[1])
    # Everyone else finite.
    for j in (0, 2, 3, 4):
        assert np.isfinite(scores[j]), f"Expected finite score at position {j}, got {scores[j]}"

    # Empty S — all features finite.
    scores_empty = expl.marginal_conditional_contributions(S=[])
    assert np.isfinite(scores_empty).all()


def test_conditional_relevance_graph_threshold_extremes():
    X_train, y_train, _ = _small_classification_dataset(p=4)
    expl = _build_fitted_classifier_explainer(X_train, y_train)

    # Very high threshold → no edges pass.
    empty_graph = expl.conditional_relevance_graph(threshold=1e9)
    assert empty_graph.shape == (4, 4)
    assert empty_graph.dtype == bool
    assert not empty_graph.any()

    # Very low threshold → every off-diagonal entry is True.
    full_graph = expl.conditional_relevance_graph(threshold=-1e9)
    assert full_graph.dtype == bool
    off_diag = ~np.eye(4, dtype=bool)
    assert full_graph[off_diag].all()
    # Diagonal is always False by convention.
    assert not np.diag(full_graph).any()


def test_regressor_explainer_works_end_to_end():
    X_train, y_train, X_test = _small_regression_dataset(p=4)

    reg = TabICLRegressor(n_estimators=2, random_state=0, verbose=False)
    _install_tiny_model(reg, max_classes=0)
    expl = TabICLExplainer(base_estimator=reg, heads=_fresh_heads())
    expl.fit(X_train, y_train)

    assert expl.observational_relevance_.shape == (4,)
    assert expl.interventional_effects_.shape == (4,)

    # Regressor has predict() but not predict_proba.
    pred = expl.predict(X_test)
    assert pred.shape == (X_test.shape[0],)
    with pytest.raises(AttributeError):
        expl.predict_proba(X_test)


def test_heads_checkpoint_roundtrip(tmp_path: Path):
    X_train, y_train, _ = _small_classification_dataset(p=5)

    heads = _fresh_heads()
    expl_direct = _build_fitted_classifier_explainer(
        X_train, y_train, heads=copy.deepcopy(heads),
    )
    direct_scores = expl_direct.observational_relevance_.copy()

    ckpt_path = tmp_path / "phase5_heads.pt"
    torch.save(
        {
            "heads": {
                "observational": heads["observational"].state_dict(),
                "interventional": heads["interventional"].state_dict(),
                "conditional": heads["conditional"].state_dict(),
                "config": {"embed_dim": EMBED_DIM, "hidden_dim": None},
            },
        },
        ckpt_path,
    )

    clf = TabICLClassifier(n_estimators=2, random_state=0, verbose=False)
    _install_tiny_model(clf, max_classes=10)
    expl_loaded = TabICLExplainer(
        base_estimator=clf,
        heads_checkpoint_path=ckpt_path,
    )
    expl_loaded.fit(X_train, y_train)

    np.testing.assert_allclose(
        direct_scores, expl_loaded.observational_relevance_,
        rtol=0, atol=0,
    )


def test_rejects_ambiguous_head_sources(tmp_path: Path):
    """Can't pass both heads and heads_checkpoint_path; can't pass neither."""
    X_train, y_train, _ = _small_classification_dataset(p=4)

    clf = TabICLClassifier(n_estimators=2, random_state=0, verbose=False)
    _install_tiny_model(clf, max_classes=10)

    expl_none = TabICLExplainer(base_estimator=clf)
    with pytest.raises(ValueError, match="No attribution heads"):
        expl_none.fit(X_train, y_train)

    dummy_ckpt = tmp_path / "x.pt"
    torch.save({"heads": {}}, dummy_ckpt)
    expl_both = TabICLExplainer(
        base_estimator=clf,
        heads=_fresh_heads(),
        heads_checkpoint_path=dummy_ckpt,
    )
    with pytest.raises(ValueError, match="at most one"):
        expl_both.fit(X_train, y_train)
