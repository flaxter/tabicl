"""Attribution-evaluation utilities for the conditional predictive value head."""
from .explainer_eval import (
    DatasetScore,
    EvalCase,
    EvalSuite,
    GroundTruth,
    build_held_out_prior_suite,
    build_in_distribution_suite,
    evaluate_explainer,
    write_scores_csv,
)
from .metrics import (
    aggregate_metrics,
    pearson_per_dataset,
    spearman_per_dataset,
    topk_recall_per_dataset,
)

__all__ = [
    "DatasetScore",
    "EvalCase",
    "EvalSuite",
    "GroundTruth",
    "aggregate_metrics",
    "build_held_out_prior_suite",
    "build_in_distribution_suite",
    "evaluate_explainer",
    "pearson_per_dataset",
    "spearman_per_dataset",
    "topk_recall_per_dataset",
    "write_scores_csv",
]
