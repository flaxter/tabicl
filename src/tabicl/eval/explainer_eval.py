"""Predictive-oracle end-to-end evaluation harness."""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from tabicl.eval.acquisition import performance_curve_for_ranking
from tabicl.eval.metrics import (
    nanmean as _nanmean,
    pearson_per_dataset,
    spearman_per_dataset,
    topk_recall_per_dataset,
)
from tabicl.prior.dataset import Prior
from tabicl.prior.hp_sampling import HpSamplerList
from tabicl.prior.labels import build_oracle_context, delta_vector_for_S, sample_value_queries_meta
from tabicl.prior.mlp_scm import MLPSCM
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from tabicl.prior.reg2cls import Reg2Cls
from tabicl.prior.tree_scm import TreeSCM


_PRIOR_REGISTRY = {
    "mlp_scm": MLPSCM,
    "tree_scm": TreeSCM,
}


@dataclass(frozen=True)
class GroundTruth:
    value_by_state: Mapping[frozenset[int], np.ndarray]
    y_var: float | None = None


@dataclass(frozen=True)
class EvalCase:
    dataset_id: str
    task: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ground_truth: GroundTruth


@dataclass(frozen=True)
class EvalSuite:
    name: str
    cases: Sequence[EvalCase]


@dataclass
class DatasetScore:
    suite: str
    dataset_id: str
    task: str
    n_train: int
    n_test: int
    p: int
    spearman_value: float
    pearson_value: float
    mse_value: float
    mae_value: float
    top1_next_feature: float
    top3_next_feature: float
    spearman_sufficiency: float
    spearman_necessity: float
    acquisition_auc: float
    fit_seconds: float


def evaluate_explainer(
    explainer_factory: Callable[[EvalCase], Any],
    suite: EvalSuite,
    *,
    out_csv: str | Path | None = None,
    verbose: bool = False,
) -> list[DatasetScore]:
    """Fit a fresh explainer for every case and score it end-to-end."""
    scores: list[DatasetScore] = []
    for case in suite.cases:
        explainer = explainer_factory(case)
        t0 = time.perf_counter()
        explainer.fit(case.X_train, case.y_train)
        fit_seconds = time.perf_counter() - t0
        score = _score_one(suite.name, case, explainer, fit_seconds)
        scores.append(score)
        if verbose:
            print(
                f"[{suite.name}] {case.dataset_id}: "
                f"rho={score.spearman_value:.3f} "
                f"top1={score.top1_next_feature:.3f} "
                f"auc={score.acquisition_auc:.3f}"
            )

    if out_csv is not None:
        write_scores_csv(out_csv, scores)
    return scores


def build_in_distribution_suite(
    n_datasets: int,
    seed: int,
    *,
    n_rows: int = 512,
    min_features: int = 5,
    max_features: int = 10,
    n_oracle: int = 512,
    mixture: str = "default",
) -> EvalSuite:
    rng = np.random.default_rng(seed)
    cases = [
        _generate_case(
            dataset_id=f"in_dist_{i:03d}",
            prior_type=_sample_prior_type(rng, "mix_scm"),
            rng=rng,
            n_rows=n_rows,
            min_features=min_features,
            max_features=max_features,
            n_oracle=n_oracle,
            mixture=mixture,
        )
        for i in range(n_datasets)
    ]
    return EvalSuite(name="in_distribution", cases=cases)


def build_held_out_prior_suite(
    n_datasets: int,
    seed: int,
    *,
    prior_type: str = "tree_scm",
    n_rows: int = 512,
    min_features: int = 5,
    max_features: int = 10,
    n_oracle: int = 512,
    mixture: str = "default",
) -> EvalSuite:
    rng = np.random.default_rng(seed)
    cases = [
        _generate_case(
            dataset_id=f"held_out_{prior_type}_{i:03d}",
            prior_type=prior_type,
            rng=rng,
            n_rows=n_rows,
            min_features=min_features,
            max_features=max_features,
            n_oracle=n_oracle,
            mixture=mixture,
        )
        for i in range(n_datasets)
    ]
    return EvalSuite(name="held_out_prior", cases=cases)


def write_scores_csv(path: str | Path, scores: Sequence[DatasetScore]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not scores:
        raise ValueError("No scores to write.")

    rows = [asdict(score) for score in scores]
    mean_row = {
        "suite": rows[0]["suite"],
        "dataset_id": "mean",
        "task": "",
        "n_train": int(sum(r["n_train"] for r in rows)),
        "n_test": int(sum(r["n_test"] for r in rows)),
        "p": "",
    }
    numeric_cols = list(DatasetScore.__dataclass_fields__.keys())[6:]
    for col in numeric_cols:
        values = [r[col] for r in rows if isinstance(r[col], (int, float)) and np.isfinite(r[col])]
        mean_row[col] = float(np.mean(values)) if values else float("nan")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DatasetScore.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        writer.writerow(mean_row)


def _score_one(suite_name: str, case: EvalCase, explainer: Any, fit_seconds: float) -> DatasetScore:
    gt = case.ground_truth
    spearman_rows: list[float] = []
    pearson_rows: list[float] = []
    top1_rows: list[float] = []
    top3_rows: list[float] = []
    sq_errors: list[float] = []
    abs_errors: list[float] = []

    for state, target in gt.value_by_state.items():
        pred = np.asarray(explainer.conditional_predictive_values(sorted(state)), dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        spearman_rows.append(spearman_per_dataset(pred, target))
        pearson_rows.append(pearson_per_dataset(pred, target))
        top1_rows.append(topk_recall_per_dataset(pred, target, 1))
        top3_rows.append(topk_recall_per_dataset(pred, target, min(3, len(target))))

        mask = np.isfinite(pred) & np.isfinite(target)
        if mask.any():
            diff = pred[mask] - target[mask]
            sq_errors.extend((diff ** 2).tolist())
            abs_errors.extend(np.abs(diff).tolist())

    suff_target = gt.value_by_state.get(frozenset())
    suff_spearman = (
        spearman_per_dataset(explainer.predictive_sufficiency_, suff_target)
        if suff_target is not None
        else float("nan")
    )

    necessity_target = _necessity_vector(gt.value_by_state, case.X_train.shape[1])
    necessity_spearman = spearman_per_dataset(explainer.predictive_necessity_, necessity_target)

    try:
        path, _ = explainer.greedy_predictive_path()
        _, acquisition_auc = performance_curve_for_ranking(
            explainer.base_estimator_,
            case.X_test,
            case.y_test,
            path,
            task=case.task,
        )
    except Exception:
        # Greedy-acquisition support is optional: if the explainer doesn't
        # implement greedy_predictive_path (or it crashes on this case), we
        # still want the other 9 metrics in DatasetScore, so silently set
        # acquisition_auc=NaN rather than aborting the whole sweep.
        acquisition_auc = float("nan")

    return DatasetScore(
        suite=suite_name,
        dataset_id=case.dataset_id,
        task=case.task,
        n_train=int(case.X_train.shape[0]),
        n_test=int(case.X_test.shape[0]),
        p=int(case.X_train.shape[1]),
        spearman_value=_nanmean(spearman_rows),
        pearson_value=_nanmean(pearson_rows),
        mse_value=float(np.mean(sq_errors)) if sq_errors else float("nan"),
        mae_value=float(np.mean(abs_errors)) if abs_errors else float("nan"),
        top1_next_feature=_nanmean(top1_rows),
        top3_next_feature=_nanmean(top3_rows),
        spearman_sufficiency=suff_spearman,
        spearman_necessity=necessity_spearman,
        acquisition_auc=acquisition_auc,
        fit_seconds=float(fit_seconds),
    )


def _necessity_vector(value_by_state: Mapping[frozenset[int], np.ndarray], p: int) -> np.ndarray:
    out = np.full(p, np.nan, dtype=np.float64)
    for i in range(p):
        state = frozenset(j for j in range(p) if j != i)
        target = value_by_state.get(state)
        if target is not None and len(target) > i:
            out[i] = float(target[i])
    return out




def _sample_prior_type(rng: np.random.Generator, prior_type: str) -> str:
    if prior_type == "mix_scm":
        return str(rng.choice(["mlp_scm", "tree_scm"], p=[0.7, 0.3]))
    return prior_type


def _sample_params(
    prior_type: str,
    *,
    rng: np.random.Generator,
    n_rows: int,
    min_features: int,
    max_features: int,
) -> dict[str, Any]:
    hp_sampler = HpSamplerList(DEFAULT_SAMPLED_HP, device="cpu")
    sampled = hp_sampler.sample()
    sampled = {k: v() if callable(v) else v for k, v in sampled.items()}

    num_features = int(rng.integers(min_features, max_features + 1))
    train_size = max(2, int(0.8 * n_rows))
    params = {
        **DEFAULT_FIXED_HP,
        **sampled,
        "seq_len": n_rows,
        "train_size": train_size,
        "max_features": num_features,
        "prior_type": prior_type,
        "num_features": num_features,
        "num_classes": 2,
        "device": "cpu",
        "permute_features": False,
        "permute_labels": False,
    }
    return params


def _generate_case(
    *,
    dataset_id: str,
    prior_type: str,
    rng: np.random.Generator,
    n_rows: int,
    min_features: int,
    max_features: int,
    n_oracle: int,
    mixture: str,
) -> EvalCase:
    prior_cls = _PRIOR_REGISTRY[prior_type]
    local_seed = int(rng.integers(0, 2**31 - 1))
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

    while True:
        params = _sample_params(
            prior_type,
            rng=rng,
            n_rows=n_rows,
            min_features=min_features,
            max_features=max_features,
        )
        with torch.no_grad():
            scm = prior_cls(**params)
            X_raw, y_raw = scm()

            X_cls, y_cls = Reg2Cls(params)(X_raw.detach(), y_raw.detach())
        X_batch = X_cls.unsqueeze(0)
        y_batch = y_cls.unsqueeze(0)
        d = torch.tensor([params["num_features"]], dtype=torch.long)
        X_filtered, d_filtered = Prior.delete_unique_features(X_batch, d)
        if int(d_filtered[0].item()) != params["num_features"]:
            continue
        if not Prior.sanity_check(X_filtered, y_batch, params["train_size"]):
            continue

        p = int(params["num_features"])
        oracle_rng = np.random.default_rng(local_seed + 1)
        context = build_oracle_context(scm, p, n_oracle=n_oracle, rng=oracle_rng)
        if p <= 12:
            states = _all_states(p)
        else:
            states = {
                frozenset(S.tolist())
                for S, _ in sample_value_queries_meta(p, oracle_rng, mixture=mixture)
            }
            states.add(frozenset())
            states.update(frozenset(j for j in range(p) if j != i) for i in range(p))

        value_by_state = {
            state: np.sqrt(
                np.clip(
                    delta_vector_for_S(context, np.array(sorted(state), dtype=int)),
                    a_min=0.0,
                    a_max=None,
                )
            )
            for state in states
        }

        X_np = X_filtered.squeeze(0).cpu().numpy()[:, :p].astype(np.float64)
        y_np = y_batch.squeeze(0).cpu().numpy().astype(np.int64)
        train_size = int(params["train_size"])
        return EvalCase(
            dataset_id=dataset_id,
            task="classification",
            X_train=X_np[:train_size],
            y_train=y_np[:train_size],
            X_test=X_np[train_size:],
            y_test=y_np[train_size:],
            ground_truth=GroundTruth(value_by_state=value_by_state, y_var=context.y_var),
        )


def _all_states(p: int) -> set[frozenset[int]]:
    states: set[frozenset[int]] = set()
    for mask in range(1 << p):
        states.add(frozenset(i for i in range(p) if mask & (1 << i)))
    return states
