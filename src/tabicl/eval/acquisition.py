"""Feature-acquisition utilities for predictive-value evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score


@dataclass(frozen=True)
class AcquisitionResult:
    method: str
    ranking: list[int]
    performance_curve: list[float]
    normalized_auc: float


def mask_unselected_features(X: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    """Return a copy of ``X`` with unrevealed features replaced by NaN."""
    X = np.asarray(X, dtype=np.float64)
    out = np.full_like(X, np.nan, dtype=np.float64)
    idx = np.asarray(list(selected), dtype=int)
    if idx.size > 0:
        out[:, idx] = X[:, idx]
    return out


def score_estimator(estimator: Any, X: np.ndarray, y: np.ndarray, task: str) -> float:
    """Score a fitted estimator on masked features."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if task == "regression":
        pred = np.asarray(estimator.predict(X), dtype=np.float64)
        return float(r2_score(y, pred))
    if task == "classification":
        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                score = proba[:, 1]
            else:
                score = proba.reshape(-1)
        else:
            score = np.asarray(estimator.predict(X), dtype=np.float64)
        return float(roc_auc_score(y, score))
    raise ValueError(f"Unknown task {task!r}; expected 'regression' or 'classification'")


def performance_curve_for_ranking(
    estimator: Any,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    ranking: Sequence[int],
    *,
    task: str,
    max_features: int | None = None,
) -> tuple[list[float], float]:
    """Held-out performance curve and normalized AUFC for a fixed ranking."""
    X_eval = np.asarray(X_eval, dtype=np.float64)
    budget = min(len(ranking), X_eval.shape[1]) if max_features is None else min(int(max_features), len(ranking))

    full_score = max(0.0, score_estimator(estimator, X_eval, y_eval, task))
    curve: list[float] = []
    for k in range(budget + 1):
        masked = mask_unselected_features(X_eval, ranking[:k])
        score = max(0.0, score_estimator(estimator, masked, y_eval, task))
        if full_score > 0.0:
            score = min(score, full_score)
        curve.append(float(score))

    if full_score <= 0.0 or budget == 0:
        return curve, float("nan")

    auc = float(np.trapezoid(np.asarray(curve, dtype=np.float64), dx=1.0))
    normalized = auc / (full_score * budget)
    return curve, float(np.clip(normalized, 0.0, 1.0))


def oracle_ranking(explainer: Any, *, k: int | None = None) -> list[int]:
    path, _ = explainer.greedy_predictive_path(k=k)
    return [int(i) for i in path]


def marginal_correlation_ranking(X: np.ndarray, y: np.ndarray) -> list[int]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    scores = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.nanstd(col) == 0.0 or np.nanstd(y) == 0.0:
            scores[j] = 0.0
            continue
        scores[j] = abs(np.corrcoef(col, y)[0, 1])
        if not np.isfinite(scores[j]):
            scores[j] = 0.0
    return np.argsort(-scores).astype(int).tolist()


def permutation_importance_ranking(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    n_repeats: int = 10,
    random_state: int = 42,
) -> list[int]:
    scoring = "r2" if task == "regression" else "roc_auc"
    result = permutation_importance(
        estimator,
        np.asarray(X, dtype=np.float64),
        np.asarray(y),
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )
    return np.argsort(-result.importances_mean).astype(int).tolist()


def kernel_shap_ranking(
    estimator: Any,
    X: np.ndarray,
    *,
    task: str,
    nsamples: int = 512,
    background_size: int = 100,
    explain_size: int = 100,
    random_state: int = 0,
) -> list[int]:
    import shap

    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    explain_idx = rng.choice(X.shape[0], size=min(explain_size, X.shape[0]), replace=False)
    X_eval = X[explain_idx]
    background = shap.sample(X, min(background_size, X.shape[0]), random_state=random_state)

    if task == "classification" and hasattr(estimator, "predict_proba"):
        predict_fn = estimator.predict_proba
    else:
        predict_fn = estimator.predict

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_eval, nsamples=nsamples, silent=True)
    values = _collapse_shap_values(shap_values)
    scores = np.abs(values).mean(axis=0)
    return np.argsort(-scores).astype(int).tolist()


def random_ranking(p: int, *, random_state: int = 42) -> list[int]:
    rng = np.random.default_rng(random_state)
    return rng.permutation(int(p)).astype(int).tolist()


def evaluate_acquisition_methods(
    explainer: Any,
    X_rank: np.ndarray,
    y_rank: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    *,
    task: str,
    methods: Sequence[str] = ("oracle", "kernel_shap", "permutation_importance", "marginal_correlation", "random"),
    max_features: int | None = None,
    shap_nsamples: int = 512,
    permutation_repeats: int = 10,
    random_state: int = 42,
) -> list[AcquisitionResult]:
    """Evaluate oracle and static rankings under the same masked-feature protocol."""
    estimator = getattr(explainer, "base_estimator_", explainer)
    p = int(np.asarray(X_eval).shape[1])
    out: list[AcquisitionResult] = []

    for method in methods:
        if method == "oracle":
            ranking = oracle_ranking(explainer, k=max_features)
        elif method == "kernel_shap":
            ranking = kernel_shap_ranking(
                estimator,
                X_rank,
                task=task,
                nsamples=shap_nsamples,
                random_state=random_state,
            )
        elif method == "permutation_importance":
            ranking = permutation_importance_ranking(
                estimator,
                X_rank,
                y_rank,
                task=task,
                n_repeats=permutation_repeats,
                random_state=random_state,
            )
        elif method == "marginal_correlation":
            ranking = marginal_correlation_ranking(X_rank, y_rank)
        elif method == "random":
            ranking = random_ranking(p, random_state=random_state)
        else:
            raise ValueError(f"Unknown acquisition method {method!r}")

        curve, auc = performance_curve_for_ranking(
            estimator,
            X_eval,
            y_eval,
            ranking,
            task=task,
            max_features=max_features,
        )
        out.append(
            AcquisitionResult(
                method=method,
                ranking=ranking,
                performance_curve=curve,
                normalized_auc=auc,
            )
        )
    return out


def _collapse_shap_values(shap_values: Any) -> np.ndarray:
    """Normalize different SHAP return shapes to ``(n_samples, p)``."""
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            return np.asarray(shap_values[1], dtype=np.float64)
        return np.asarray(shap_values[0], dtype=np.float64)

    values = np.asarray(shap_values, dtype=np.float64)
    if values.ndim == 3:
        # Common binary-classification shape: (n_samples, p, 2).
        return values[..., 1] if values.shape[-1] >= 2 else values[..., 0]
    return values
