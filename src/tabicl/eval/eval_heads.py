"""Checkpoint scorer for the conditional predictive value head."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from tabicl.eval.explainer_eval import EvalSuite, build_held_out_prior_suite, build_in_distribution_suite
from tabicl.eval.metrics import pearson_per_dataset, spearman_per_dataset, topk_recall_per_dataset
from tabicl.model.heads import ConditionalPredictiveValueHead
from tabicl.model.tabicl import TabICL


@dataclass(frozen=True)
class LoadedValueModel:
    trunk: TabICL
    value_head: ConditionalPredictiveValueHead
    device: torch.device


@dataclass
class HeadEvalRow:
    suite: str
    dataset_id: str
    stratum: str
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


def load_value_model(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> LoadedValueModel:
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    trunk = TabICL(**checkpoint["config"])
    trunk.load_state_dict(checkpoint["state_dict"])
    trunk.to(device).eval()

    heads = checkpoint.get("heads")
    if heads is None or "value" not in heads:
        raise KeyError(f"Checkpoint at {path} does not contain a value head.")

    cfg = heads.get("config", {})
    embed_dim = cfg.get("embed_dim") or checkpoint["config"].get("embed_dim")
    hidden_dim = cfg.get("hidden_dim")
    value_head = ConditionalPredictiveValueHead(embed_dim=embed_dim, hidden_dim=hidden_dim)
    value_head.load_state_dict(heads["value"])
    value_head.to(device).eval()
    return LoadedValueModel(trunk=trunk, value_head=value_head, device=torch.device(device))


def evaluate_value_model(model: LoadedValueModel, suite: EvalSuite) -> list[HeadEvalRow]:
    rows: list[HeadEvalRow] = []
    for case in suite.cases:
        col_emb = _column_embeddings(model, case)
        necessity = _necessity_vector(case.ground_truth.value_by_state, case.X_train.shape[1])
        suff = case.ground_truth.value_by_state[frozenset()]

        for stratum in ("all", "empty", "singleton", "small", "medium", "near_full"):
            states = _states_for_stratum(case.ground_truth.value_by_state, stratum, case.X_train.shape[1])
            if not states:
                continue
            row = _score_states(
                suite.name,
                case.dataset_id,
                stratum,
                col_emb,
                model,
                states,
                suff_target=suff,
                necessity_target=necessity,
                all_states=case.ground_truth.value_by_state,
            )
            rows.append(row)
    return rows


def write_rows_csv(path: str | Path, rows: Sequence[HeadEvalRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(HeadEvalRow.__dataclass_fields__.keys())
    data = [asdict(row) for row in rows]
    mean_row = {"suite": data[0]["suite"], "dataset_id": "mean", "stratum": "all", "p": ""}
    for col in fieldnames[4:]:
        vals = [row[col] for row in data if np.isfinite(row[col])]
        mean_row[col] = float(np.mean(vals)) if vals else float("nan")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        writer.writerow(mean_row)


def _column_embeddings(model: LoadedValueModel, case: Any) -> torch.Tensor:
    X_full = np.concatenate([case.X_train, case.X_test], axis=0)[None, ...]
    X_t = torch.as_tensor(X_full, dtype=torch.float, device=model.device)
    y_train = torch.as_tensor(case.y_train[None, ...], dtype=torch.long, device=model.device)
    d = torch.tensor([case.X_train.shape[1]], dtype=torch.long, device=model.device)
    with torch.no_grad():
        _, col_emb = model.trunk(
            X_t,
            y_train=y_train,
            d=d,
            return_column_embeddings=True,
        )
    return col_emb


def _predict_state(col_emb: torch.Tensor, model: LoadedValueModel, state: frozenset[int]) -> np.ndarray:
    p = col_emb.shape[1]
    mask = torch.zeros((1, p), dtype=torch.bool, device=model.device)
    if state:
        idx = torch.as_tensor(sorted(state), dtype=torch.long, device=model.device)
        mask[0, idx] = True
    with torch.no_grad():
        pred = model.value_head(col_emb, mask).squeeze(0).cpu().numpy().astype(np.float64)
    if state:
        pred[np.asarray(sorted(state), dtype=int)] = np.nan
    return pred


def _score_states(
    suite: str,
    dataset_id: str,
    stratum: str,
    col_emb: torch.Tensor,
    model: LoadedValueModel,
    states: Sequence[frozenset[int]],
    *,
    suff_target: np.ndarray,
    necessity_target: np.ndarray,
    all_states: Mapping[frozenset[int], np.ndarray],
) -> HeadEvalRow:
    spearman_rows: list[float] = []
    pearson_rows: list[float] = []
    top1_rows: list[float] = []
    top3_rows: list[float] = []
    sq_errors: list[float] = []
    abs_errors: list[float] = []

    for state in states:
        pred = _predict_state(col_emb, model, state)
        target = np.asarray(all_states[state], dtype=np.float64)
        spearman_rows.append(spearman_per_dataset(pred, target))
        pearson_rows.append(pearson_per_dataset(pred, target))
        top1_rows.append(topk_recall_per_dataset(pred, target, 1))
        top3_rows.append(topk_recall_per_dataset(pred, target, min(3, len(target))))
        mask = np.isfinite(pred) & np.isfinite(target)
        if mask.any():
            diff = pred[mask] - target[mask]
            sq_errors.extend((diff ** 2).tolist())
            abs_errors.extend(np.abs(diff).tolist())

    if stratum == "all":
        suff_pred = _predict_state(col_emb, model, frozenset())
        necessity_pred = np.array(
            [
                _predict_state(col_emb, model, frozenset(j for j in range(col_emb.shape[1]) if j != i))[i]
                for i in range(col_emb.shape[1])
            ],
            dtype=np.float64,
        )
        acquisition_auc = _oracle_acquisition_auc(col_emb, model, all_states)
        suff_spearman = spearman_per_dataset(suff_pred, suff_target)
        necessity_spearman = spearman_per_dataset(necessity_pred, necessity_target)
    else:
        suff_spearman = float("nan")
        necessity_spearman = float("nan")
        acquisition_auc = float("nan")

    return HeadEvalRow(
        suite=suite,
        dataset_id=dataset_id,
        stratum=stratum,
        p=int(col_emb.shape[1]),
        spearman_value=_nanmean(spearman_rows),
        pearson_value=_nanmean(pearson_rows),
        mse_value=float(np.mean(sq_errors)) if sq_errors else float("nan"),
        mae_value=float(np.mean(abs_errors)) if abs_errors else float("nan"),
        top1_next_feature=_nanmean(top1_rows),
        top3_next_feature=_nanmean(top3_rows),
        spearman_sufficiency=suff_spearman,
        spearman_necessity=necessity_spearman,
        acquisition_auc=acquisition_auc,
    )


def _states_for_stratum(value_by_state: Mapping[frozenset[int], np.ndarray], stratum: str, p: int) -> list[frozenset[int]]:
    states = list(value_by_state.keys())
    if stratum == "all":
        return states
    if stratum == "empty":
        return [s for s in states if len(s) == 0]
    if stratum == "singleton":
        return [s for s in states if len(s) == 1]
    if stratum == "small":
        return [s for s in states if 2 <= len(s) <= 4]
    if stratum == "medium":
        return [s for s in states if len(s) == p // 2]
    if stratum == "near_full":
        return [s for s in states if len(s) in {max(p - 2, 0), max(p - 1, 0)}]
    raise ValueError(f"Unknown stratum {stratum!r}")


def _necessity_vector(value_by_state: Mapping[frozenset[int], np.ndarray], p: int) -> np.ndarray:
    out = np.full(p, np.nan, dtype=np.float64)
    for i in range(p):
        state = frozenset(j for j in range(p) if j != i)
        target = value_by_state.get(state)
        if target is not None:
            out[i] = float(target[i])
    return out


def _oracle_acquisition_auc(
    col_emb: torch.Tensor,
    model: LoadedValueModel,
    value_by_state: Mapping[frozenset[int], np.ndarray],
) -> float:
    p = int(col_emb.shape[1])
    selected: list[int] = []
    used: set[int] = set()
    curve = [0.0]
    cumulative = 0.0
    for _ in range(p):
        state = frozenset(selected)
        pred = _predict_state(col_emb, model, state)
        candidates = [i for i in range(p) if i not in used]
        i_best = max(candidates, key=lambda i: pred[i])
        true_gain = float(value_by_state[state][i_best] ** 2)
        cumulative += true_gain
        curve.append(cumulative)
        selected.append(i_best)
        used.add(i_best)

    total = curve[-1]
    if total <= 0.0:
        return float("nan")
    auc = float(np.trapezoid(np.asarray(curve, dtype=np.float64), dx=1.0))
    return float(np.clip(auc / (total * p), 0.0, 1.0))


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite", choices=("in_distribution", "held_out_prior"), default="in_distribution")
    parser.add_argument("--n_datasets", type=int, default=20)
    parser.add_argument("--n_rows", type=int, default=512)
    parser.add_argument("--min_features", type=int, default=5)
    parser.add_argument("--max_features", type=int, default=10)
    parser.add_argument("--n_oracle", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = load_value_model(args.checkpoint_path, device=args.device)
    if args.suite == "in_distribution":
        suite = build_in_distribution_suite(
            args.n_datasets,
            args.seed,
            n_rows=args.n_rows,
            min_features=args.min_features,
            max_features=args.max_features,
            n_oracle=args.n_oracle,
        )
    else:
        suite = build_held_out_prior_suite(
            args.n_datasets,
            args.seed,
            n_rows=args.n_rows,
            min_features=args.min_features,
            max_features=args.max_features,
            n_oracle=args.n_oracle,
        )
    rows = evaluate_value_model(model, suite)
    write_rows_csv(args.out, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
