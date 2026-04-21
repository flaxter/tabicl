"""OpenML real-data loaders and predictive acquisition benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from tabicl.eval.acquisition import AcquisitionResult, evaluate_acquisition_methods


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    openml_id: int
    task: str


@dataclass(frozen=True)
class OpenMLDataset:
    name: str
    openml_id: int
    task: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


OPENML_DATASETS = {
    "bike_sharing": DatasetSpec("bike_sharing", 42713, "regression"),
    "adult_income": DatasetSpec("adult_income", 1590, "classification"),
    "breast_cancer_wisconsin": DatasetSpec("breast_cancer_wisconsin", 1510, "classification"),
}


def load_openml_dataset(name: str, *, random_state: int = 0) -> OpenMLDataset:
    """Fetch, preprocess, and split one locked OpenML dataset."""
    if name not in OPENML_DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {sorted(OPENML_DATASETS)}")

    spec = OPENML_DATASETS[name]
    bunch = fetch_openml(data_id=spec.openml_id, as_frame=True)
    X_frame = bunch.data.copy()
    y = bunch.target.copy()

    transformer = _build_transformer(X_frame)
    X_all = transformer.fit_transform(X_frame).astype(np.float64)
    feature_names = list(transformer.get_feature_names_out())

    if spec.task == "classification":
        y = LabelEncoder().fit_transform(np.asarray(y))
        stratify = y
    else:
        y = np.asarray(y, dtype=np.float64)
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    return OpenMLDataset(
        name=spec.name,
        openml_id=spec.openml_id,
        task=spec.task,
        X_train=X_train,
        X_test=X_test,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        feature_names=feature_names,
    )


def run_openml_greedy_benchmark(
    explainer_factory: Callable[[OpenMLDataset], Any],
    *,
    dataset_names: Sequence[str] = ("bike_sharing", "adult_income", "breast_cancer_wisconsin"),
    random_state: int = 0,
    methods: Sequence[str] = ("oracle", "kernel_shap", "permutation_importance", "marginal_correlation", "random"),
    shap_nsamples: int = 512,
) -> dict[str, list[AcquisitionResult]]:
    """Fit one explainer per dataset and compare acquisition policies."""
    results: dict[str, list[AcquisitionResult]] = {}
    for name in dataset_names:
        dataset = load_openml_dataset(name, random_state=random_state)
        explainer = explainer_factory(dataset)
        explainer.fit(dataset.X_train, dataset.y_train)
        results[name] = evaluate_acquisition_methods(
            explainer,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            dataset.y_test,
            task=dataset.task,
            methods=methods,
            shap_nsamples=shap_nsamples,
            random_state=42,
        )
    return results


def _build_transformer(X_frame: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [
        col for col in X_frame.columns
        if pd.api.types.is_object_dtype(X_frame[col]) or pd.api.types.is_categorical_dtype(X_frame[col])
    ]
    num_cols = [col for col in X_frame.columns if col not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
