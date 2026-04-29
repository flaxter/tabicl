"""Unit tests for :mod:`tabicl.eval.eval_heads`.

Covers the post-pivot API:

- ``stratum_of_state`` bucketing rules.
- ``evaluate_dataset`` returns one S11_1 row per stratum and a single
  S11_2 row per dataset, with finite metrics on perfect predictions.
- ``write_s11_1`` writes a CSV with per-row data plus per-stratum and
  across-strata summary rows.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pytest

from tabicl.eval.eval_heads import (
    STRATUM_ORDER,
    S11_1Row,
    S11_2Row,
    evaluate_dataset,
    stratum_of_state,
    write_s11_1,
)
from tabicl.eval.explainer_eval import EvalCase, GroundTruth


# ---------------------------------------------------------------------------
# stratum_of_state
# ---------------------------------------------------------------------------


def test_stratum_of_state_named_buckets():
    p = 10
    assert stratum_of_state(frozenset(), p) == "empty"
    assert stratum_of_state(frozenset([2]), p) == "singleton"
    assert stratum_of_state(frozenset([0, 1, 2]), p) == "small"
    assert stratum_of_state(frozenset(range(p - 1)), p) == "near_full"
    assert stratum_of_state(frozenset(range(p - 2)), p) == "near_full"
    # Medium bucket is exactly p // 2.
    assert stratum_of_state(frozenset(range(p // 2)), p) == "medium"


def test_stratum_of_state_small_p():
    # At p=3, |S|=2 hits the 'small' branch (2 <= s <= 4) before the
    # near_full branch — the small bucket dominates when ranges overlap.
    assert stratum_of_state(frozenset([0]), 3) == "singleton"
    assert stratum_of_state(frozenset([0, 1]), 3) == "small"


def test_stratum_of_state_unclassified_returns_none():
    # |S|=5 with p=10 isn't in any bucket (small=2..4, medium=5 only at p//2=5,
    # but 5 != p//2=5 only for p=10 — actually 10//2=5, so |S|=5 IS medium.)
    # Pick |S|=6 at p=10: not small, not medium (5), not near_full (8 or 9).
    assert stratum_of_state(frozenset(range(6)), 10) is None


# ---------------------------------------------------------------------------
# evaluate_dataset: dummy explainer
# ---------------------------------------------------------------------------


class _DummyExplainer:
    """Returns pre-set per-state vectors. ``fit`` is a no-op."""

    def __init__(
        self,
        preds_by_state: Mapping,
        sufficiency: np.ndarray | None = None,
        necessity: np.ndarray | None = None,
    ):
        self._preds = {
            frozenset(k): np.asarray(v, dtype=np.float64)
            for k, v in preds_by_state.items()
        }
        if sufficiency is not None:
            self.predictive_sufficiency_ = np.asarray(sufficiency, dtype=np.float64)
        if necessity is not None:
            self.predictive_necessity_ = np.asarray(necessity, dtype=np.float64)

    def fit(self, X, y) -> "_DummyExplainer":
        return self

    def conditional_predictive_values(self, S: Sequence[int]) -> np.ndarray:
        return self._preds[frozenset(int(i) for i in S)].copy()


def _make_case(p: int = 6) -> tuple[EvalCase, _DummyExplainer]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, p))
    y = rng.standard_normal(30)

    value_by_state: dict[frozenset, np.ndarray] = {}
    value_by_state[frozenset()] = np.array([0.9, 0.2, 0.7, 0.3, 0.1, 0.5])
    # Populate at least one state in every stratum we want to exercise.
    value_by_state[frozenset([0])] = np.array([np.nan, 0.4, 0.5, 0.1, 0.2, 0.1])
    value_by_state[frozenset([0, 1, 2])] = np.array(
        [np.nan, np.nan, np.nan, 0.4, 0.3, 0.5]
    )
    value_by_state[frozenset([0, 1, 2])] = np.array(
        [np.nan, np.nan, np.nan, 0.4, 0.3, 0.5]
    )
    # |S| = p // 2 = 3 → already in 'small' (2..4); pick a state that lands
    # exactly on the medium boundary by using p=6 so p//2=3 is small. Use a
    # different p instead: switch to p=10 below in dedicated cases.
    # |S| = p - 1 → near_full
    value_by_state[frozenset(range(p - 1))] = np.full(p, np.nan)
    value_by_state[frozenset(range(p - 1))][p - 1] = 0.9
    # LOO for necessity vector: |S| = p-1 with one feature missing per row.
    for i in range(p):
        loo = frozenset(j for j in range(p) if j != i)
        if loo not in value_by_state:
            row = np.full(p, np.nan)
            row[i] = 0.1 * (i + 1)
            value_by_state[loo] = row

    gt = GroundTruth(value_by_state=value_by_state, y_var=1.0)
    preds = {k: v.copy() for k, v in value_by_state.items()}
    sufficiency = preds[frozenset()].copy()
    necessity = np.array([0.1 * (i + 1) for i in range(p)])
    case = EvalCase(
        dataset_id="d0",
        task="regression",
        X_train=X, y_train=y,
        X_test=X, y_test=y,
        ground_truth=gt,
    )
    return case, _DummyExplainer(preds, sufficiency=sufficiency, necessity=necessity)


def test_evaluate_dataset_returns_one_row_per_stratum():
    case, expl = _make_case()
    rows_11_1, row_11_2, s_pairs, n_pairs = evaluate_dataset(case, expl)
    assert len(rows_11_1) == len(STRATUM_ORDER)
    assert {r.stratum for r in rows_11_1} == set(STRATUM_ORDER)
    assert isinstance(row_11_2, S11_2Row)


def test_evaluate_dataset_perfect_predictions_zero_mae_on_populated_strata():
    case, expl = _make_case()
    rows_11_1, _, _, _ = evaluate_dataset(case, expl)
    populated = [r for r in rows_11_1 if r.n_states > 0]
    assert populated, "test fixture must populate at least one stratum"
    for r in populated:
        # Perfect predictions → mean MAE 0 (within float tolerance).
        assert r.mae == pytest.approx(0.0, abs=1e-9)


def test_evaluate_dataset_unpopulated_stratum_has_nan():
    """Drop the empty-state query → 'empty' stratum has zero states."""
    case, expl = _make_case()
    states = {
        k: v for k, v in case.ground_truth.value_by_state.items() if len(k) > 0
    }
    case2 = EvalCase(
        dataset_id=case.dataset_id, task=case.task,
        X_train=case.X_train, y_train=case.y_train,
        X_test=case.X_test, y_test=case.y_test,
        ground_truth=GroundTruth(value_by_state=states, y_var=1.0),
    )
    expl2 = _DummyExplainer(
        {k: v.copy() for k, v in states.items()},
        # No sufficiency target → still need the attribute for §11.2 path.
        sufficiency=np.zeros(case.X_train.shape[1]),
        necessity=np.zeros(case.X_train.shape[1]),
    )
    rows_11_1, _, _, _ = evaluate_dataset(case2, expl2)
    by_name = {r.stratum: r for r in rows_11_1}
    assert by_name["empty"].n_states == 0
    assert np.isnan(by_name["empty"].spearman)
    assert np.isnan(by_name["empty"].mae)


def test_write_s11_1_writes_per_dataset_and_summary_rows(tmp_path: Path):
    case, expl = _make_case()
    rows_a, _, _, _ = evaluate_dataset(case, expl)
    rows_b, _, _, _ = evaluate_dataset(case, expl)
    all_rows = list(rows_a) + list(rows_b)

    out = tmp_path / "s11_1.csv"
    write_s11_1(out, all_rows)
    assert out.exists()

    with out.open() as f:
        csv_rows = list(csv.DictReader(f))

    # 2 datasets × 5 strata = 10 per-row rows + 5 stratum-mean rows
    # + 1 across-strata row.
    assert len(csv_rows) == len(STRATUM_ORDER) * 2 + len(STRATUM_ORDER) + 1
    stratum_col = [r["stratum"] for r in csv_rows]
    assert "across_strata" in stratum_col
    for s in STRATUM_ORDER:
        # Each stratum appears twice as per-dataset rows + once as the
        # stratum-mean summary row.
        assert stratum_col.count(s) == 3
