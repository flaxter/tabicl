"""Phase 6 — attribution evaluation utilities.

This subpackage ships the metrics and evaluation harness used to score
trained attribution heads against the Phase 3 ground-truth labels.

Phase 6e (head-only fine-tuning) is the first consumer; Phase 6a–d
reuse the same primitives on different prior configurations.
"""
from .metrics import (
    spearman_per_dataset,
    pearson_per_dataset,
    topk_recall_per_dataset,
    aggregate_metrics,
)
from .explainer_eval import (
    GroundTruth,
    DatasetSpec,
    DatasetScore,
    canonical_states,
    score_one,
    evaluate_explainer,
    build_in_distribution_suite,
    build_held_out_prior_suite,
)

__all__ = [
    "spearman_per_dataset",
    "pearson_per_dataset",
    "topk_recall_per_dataset",
    "aggregate_metrics",
    "GroundTruth",
    "DatasetSpec",
    "DatasetScore",
    "canonical_states",
    "score_one",
    "evaluate_explainer",
    "build_in_distribution_suite",
    "build_held_out_prior_suite",
]
