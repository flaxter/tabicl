"""Phase 6a — end-to-end attribution-quality harness tests.

Cover the four public suite builders plus ``evaluate_explainer``
end-to-end on a tiny in-memory TabICL regressor with freshly-initialised
heads. These tests verify **interface correctness**, not attribution
accuracy — untrained heads are near-zero by init and aren't expected to
pass quality gates (that's the paper run).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from tabicl import TabICLExplainer, TabICLRegressor
from tabicl.eval.explainer_eval import (
    DatasetScore,
    EvalCase,
    EvalSuite,
    GroundTruth,
    build_collider_suite,
    build_held_out_prior_suite,
    build_id_boundary_suite,
    build_in_distribution_suite,
    evaluate_explainer,
    write_scores_csv,
)
from tabicl.model.heads import (
    ConditionalHead,
    InterventionalHead,
    ObservationalHead,
)
from tabicl.model.tabicl import TabICL


EMBED_DIM = 32


# ---------------------------------------------------------------------------
# Fixtures (mirrors tests/test_explainer.py)
# ---------------------------------------------------------------------------


def _build_tiny_tabicl(seed: int = 0) -> TabICL:
    torch.manual_seed(seed)
    return TabICL(
        max_classes=0,  # regressor
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


def _install_tiny_model(estimator) -> None:
    tiny = _build_tiny_tabicl()

    def _fake_load_model(self=estimator):
        self.model_path_ = None
        self.model_ = tiny
        self.model_config_ = {"max_classes": 0, "embed_dim": EMBED_DIM}
        self.model_.eval()

    estimator._load_model = _fake_load_model


def _fresh_heads() -> dict:
    torch.manual_seed(0)
    return {
        "observational": ObservationalHead(embed_dim=EMBED_DIM),
        "interventional": InterventionalHead(embed_dim=EMBED_DIM),
        "conditional": ConditionalHead(embed_dim=EMBED_DIM),
    }


def _explainer_factory():
    def _make():
        reg = TabICLRegressor(n_estimators=2, random_state=0, verbose=False)
        _install_tiny_model(reg)
        return TabICLExplainer(base_estimator=reg, heads=_fresh_heads())

    return _make


# ---------------------------------------------------------------------------
# Suite-builder shape tests
# ---------------------------------------------------------------------------


def test_in_distribution_suite_shape():
    suite = build_in_distribution_suite(
        n_datasets=3, seed=0, n_rows=128, min_features=5, max_features=6,
        n_mc=64, k_cond_triples=4,
    )
    assert suite.name == "in_distribution"
    assert len(suite.cases) == 3
    for case in suite.cases:
        assert case.X.ndim == 2 and case.X.shape[0] == 128
        p = case.X.shape[1]
        assert case.y.shape == (128,)
        gt = case.ground_truth
        assert gt.observational.shape == (p,)
        assert gt.is_identifiable is True
        assert gt.interventional is not None
        assert gt.interventional.shape == (p,)
        assert len(gt.conditional_triples) == 4
        for (i, S, c_star) in gt.conditional_triples:
            assert 0 <= i < p
            assert isinstance(S, frozenset)
            assert i not in S
            assert np.isfinite(c_star)


def test_held_out_prior_suite_shape():
    suite = build_held_out_prior_suite(
        n_datasets=2, seed=0, n_rows=128, min_features=5, max_features=6,
        n_mc=64, k_cond_triples=4,
    )
    assert suite.name == "held_out_prior"
    assert len(suite.cases) == 2
    for case in suite.cases:
        gt = case.ground_truth
        assert gt.is_identifiable is False
        # MLPSCM has no ``do()`` ground truth — interventional should be None.
        assert gt.interventional is None
        # Head A label still defined via binned plug-in estimator.
        assert np.isfinite(gt.observational).any()


def test_collider_suite_analytical_ground_truth():
    suite = build_collider_suite(
        n_datasets=1, seed=0, n_rows=200,
        n_parents=2, n_independent=1, n_downstream=2, k_cond_triples=4,
    )
    assert suite.name == "collider"
    case = suite.cases[0]
    p = case.X.shape[1]
    assert p == 5  # 2 parents + 1 indep + 2 downstream

    gt = case.ground_truth
    # Independent feature (index 2) must have Head I == 0 exactly.
    assert gt.interventional is not None
    assert gt.interventional[2] == pytest.approx(0.0)
    # Downstream features (indices 3, 4) must have Head I == 0 exactly.
    assert gt.interventional[3] == pytest.approx(0.0)
    assert gt.interventional[4] == pytest.approx(0.0)
    # Parent features (indices 0, 1) must have Head I > 0.
    assert gt.interventional[0] > 0
    assert gt.interventional[1] > 0
    # Independent feature should have ~0 observational info.
    # (The exact 0 holds because the analytical covariance has sig_Xy[indep]=0.)
    assert gt.observational[2] == pytest.approx(0.0, abs=1e-9)
    # Parent and downstream should carry non-trivial observational info.
    assert gt.observational[0] > 0
    assert gt.observational[3] > 0


def test_id_boundary_suite_alternates_identifiability():
    suite = build_id_boundary_suite(
        n_datasets=4, seed=0, n_rows=128, min_features=5, max_features=6,
        n_mc=64, k_cond_triples=4,
    )
    assert suite.name == "id_boundary"
    id_flags = [c.ground_truth.is_identifiable for c in suite.cases]
    assert id_flags == [True, False, True, False]
    # True structural coefficients are still available as Head I ground
    # truth on the non-identifiable half — what the model *should*
    # predict if it could identify the mechanism; that is the whole
    # point of the paired figure.
    for c in suite.cases:
        assert c.ground_truth.interventional is not None


# ---------------------------------------------------------------------------
# evaluate_explainer end-to-end
# ---------------------------------------------------------------------------


def test_evaluate_explainer_end_to_end(tmp_path: Path):
    suite = build_in_distribution_suite(
        n_datasets=2, seed=0, n_rows=64, min_features=5, max_features=5,
        n_mc=32, k_cond_triples=3,
    )
    out_csv = tmp_path / "scores.csv"
    scores = evaluate_explainer(
        _explainer_factory(), suite, out_csv=out_csv, verbose=False
    )
    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, DatasetScore)
        assert s.suite == "in_distribution"
        assert s.n == 64 and s.p == 5
        # Untrained heads → metrics should at least be finite numbers
        # (Spearman may be nan on degenerate outputs; just make sure
        # nothing crashes and the harness populates every field).
        assert s.fit_seconds >= 0
        assert s.n_triples_C >= 0

    # CSV persisted with header + 2 data rows + 1 mean row = 4 lines.
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[-1]["dataset_id"] == "mean"


def test_evaluate_explainer_collider_headline_shape(tmp_path: Path):
    """End-to-end on the collider suite — the headline §6.1 figure path."""
    suite = build_collider_suite(
        n_datasets=1, seed=1, n_rows=64,
        n_parents=2, n_independent=1, n_downstream=2, k_cond_triples=3,
    )
    scores = evaluate_explainer(_explainer_factory(), suite, verbose=False)
    assert len(scores) == 1
    s = scores[0]
    assert s.suite == "collider"
    # p = 2 + 1 + 2 = 5
    assert s.p == 5


def test_write_scores_csv_refuses_empty(tmp_path: Path):
    with pytest.raises(ValueError, match="No scores"):
        write_scores_csv(tmp_path / "x.csv", [])
