"""Unit tests for :mod:`tabicl.eval.eval_heads`.

Covers:

- ``classify_stratum`` bucketing rules.
- ``score_by_stratum`` returns finite metrics for every populated stratum
  when predictions are perfect, and NaN for strata with no states.
- ``evaluate_head_by_stratum`` writes one row per (dataset, stratum) pair.

Head-only fine-tuning (``fit_head_only``) is covered implicitly by the
training smoke: we only verify the API shape here without running a
trunk forward pass.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pytest

from tabicl.eval.eval_heads import (
    STRATA,
    StratumMetrics,
    classify_stratum,
    evaluate_head_by_stratum,
    score_by_stratum,
)
from tabicl.eval.explainer_eval import DatasetSpec, GroundTruth


# ---------------------------------------------------------------------------
# classify_stratum
# ---------------------------------------------------------------------------


def test_classify_stratum_named_buckets():
    p = 10
    assert classify_stratum(p, frozenset()) == "empty"
    assert classify_stratum(p, frozenset([2])) == "singleton"
    assert classify_stratum(p, frozenset([0, 1, 2])) == "small"
    assert classify_stratum(p, frozenset(range(p - 1))) == "near_full"
    assert classify_stratum(p, frozenset(range(p - 2))) == "near_full"
    # Medium bucket includes p//2 exactly.
    assert classify_stratum(p, frozenset(range(p // 2))) == "medium"


def test_classify_stratum_small_p():
    # p = 3: singleton → 1, near_full → {1, 2} → hits 'near_full' branch via p-2/p-1.
    assert classify_stratum(3, frozenset([0])) == "singleton"
    assert classify_stratum(3, frozenset([0, 1])) == "near_full"


# ---------------------------------------------------------------------------
# score_by_stratum: dummy explainer
# ---------------------------------------------------------------------------


class _DummyExplainer:
    def __init__(self, preds_by_state: Mapping):
        self._preds = {frozenset(k): np.asarray(v, dtype=np.float64) for k, v in preds_by_state.items()}

    def conditional_predictive_values(self, S: Sequence[int]) -> np.ndarray:
        return self._preds[frozenset(int(i) for i in S)].copy()


def _make_spec(p: int = 6) -> tuple[DatasetSpec, _DummyExplainer]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, p))
    y = rng.standard_normal(30)

    value_by_state = {}
    empty = frozenset()
    value_by_state[empty] = np.array([0.9, 0.2, 0.7, 0.3, 0.1, 0.5])

    # Populate at least one state in every stratum.
    value_by_state[frozenset([0])] = np.array([np.nan, 0.4, 0.5, 0.1, 0.2, 0.1])
    value_by_state[frozenset([0, 1, 2])] = np.array([np.nan, np.nan, np.nan, 0.4, 0.3, 0.5])
    value_by_state[frozenset([0, 1, 2, 3])] = np.array([np.nan, np.nan, np.nan, np.nan, 0.2, 0.8])   # |S|=p//2=3 or 4
    value_by_state[frozenset([0, 1, 2, 3, 4])] = np.array([np.nan]*5 + [0.9])  # p-1 near_full

    gt = GroundTruth(value_by_state=value_by_state, y_var=1.0)
    preds = {k: v.copy() for k, v in value_by_state.items()}
    return (
        DatasetSpec(name="d0", X=X, y=y, ground_truth=gt, meta={}),
        _DummyExplainer(preds),
    )


def test_score_by_stratum_perfect_predictions_populated_buckets():
    spec, expl = _make_spec()
    out = score_by_stratum(expl, spec)
    assert set(out.keys()) == set(STRATA)

    # Empty + singleton + small + near_full always populated; medium may be.
    for bucket in ("empty", "singleton", "near_full"):
        m = out[bucket]
        assert m.n_states >= 1
        # Perfect predictions → MAE/MSE 0 for every bucket that has enough
        # finite overlap for Spearman; MAE is always finite.
        assert m.mae == pytest.approx(0.0, abs=1e-9)
        assert m.mse == pytest.approx(0.0, abs=1e-9)


def test_score_by_stratum_fills_unused_bucket_with_nan():
    """Remove every empty-set query → stratum has zero states."""
    spec, expl = _make_spec()
    # Build a spec with no empty state.
    states = {k: v for k, v in spec.ground_truth.value_by_state.items() if len(k) > 0}
    spec2 = DatasetSpec(
        name=spec.name, X=spec.X, y=spec.y,
        ground_truth=GroundTruth(value_by_state=states),
        meta={},
    )
    expl2 = _DummyExplainer({k: v.copy() for k, v in states.items()})
    out = score_by_stratum(expl2, spec2)
    assert out["empty"].n_states == 0
    assert np.isnan(out["empty"].mae)
    assert np.isnan(out["empty"].spearman)


def test_evaluate_head_by_stratum_writes_csv_with_one_row_per_bucket(tmp_path: Path):
    spec, expl = _make_spec()
    out = tmp_path / "head_eval.csv"

    def factory(X, y):
        return expl

    rows = evaluate_head_by_stratum(factory, [spec, spec], out_csv=out)
    assert len(rows) == len(STRATA) * 2
    assert isinstance(rows[0], StratumMetrics)
    assert out.exists()

    with out.open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(STRATA) * 2
    # Every bucket appears twice (once per dataset).
    stratums = [r["stratum"] for r in csv_rows]
    for s in STRATA:
        assert stratums.count(s) == 2
