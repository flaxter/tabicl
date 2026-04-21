"""Head-only fine-tune scorer for the conditional predictive value head.

Two responsibilities, kept deliberately small:

1. Fine-tune only the value head on a frozen TabICL trunk over a stream
   of labelled datasets (``fit_head_only``). Produces a fresh head
   checkpoint that can be loaded by :class:`TabICLExplainer`.
2. Score a trained head against a held-out suite by |S|-stratum
   (``evaluate_head_by_stratum``). Reports Spearman / Pearson / MAE on
   the RMS scale plus top-k recall of best-next-feature, per
   preregistration §11.1.

For end-to-end explainer scoring (including greedy-path AUFC and endpoint
attributes), use :func:`tabicl.eval.explainer_eval.evaluate_explainer`
instead. This module is the finer-grained label-fidelity cousin.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from tabicl.eval.explainer_eval import DatasetSpec, Factory
from tabicl.eval.metrics import (
    pearson_per_dataset,
    spearman_per_dataset,
    topk_recall_per_dataset,
)


# ---------------------------------------------------------------------------
# |S|-stratum classification
# ---------------------------------------------------------------------------


STRATA: Tuple[str, ...] = ("empty", "singleton", "small", "medium", "near_full")


def classify_stratum(p: int, S: FrozenSet[int]) -> str:
    """Bucket ``S`` into the preregistration's |S| strata.

    Rule order (first match wins, so edge cases on small ``p`` fall into
    the most specialised bucket):

    - ``empty``     if ``|S|=0``
    - ``singleton`` if ``|S|=1``
    - ``near_full`` if ``|S| in {p-2, p-1}``
    - ``small``     if ``2 <= |S| <= 4``
    - ``medium``    if ``|S|`` is within 1 of ``p//2``
    - catch-all: ``near_full`` if ``|S| > p/2`` else ``medium``.
    """
    size = len(S)
    if size == 0:
        return "empty"
    if size == 1:
        return "singleton"
    if size in (p - 2, p - 1):
        return "near_full"
    if 2 <= size <= 4:
        return "small"
    mid = p // 2
    if abs(size - mid) <= 1:
        return "medium"
    return "near_full" if size > p // 2 else "medium"


# ---------------------------------------------------------------------------
# Per-stratum metrics
# ---------------------------------------------------------------------------


@dataclass
class StratumMetrics:
    """One |S|-stratum's pooled metrics for one dataset."""

    name: str
    stratum: str
    n_states: int
    n_features: int
    spearman: float = float("nan")
    pearson: float = float("nan")
    mae: float = float("nan")
    mse: float = float("nan")
    top1_next_feature: float = float("nan")
    top3_next_feature: float = float("nan")


def _per_state_metrics(
    pred: np.ndarray, truth: np.ndarray
) -> Tuple[float, float, float, float, float, float]:
    """Return (spearman, pearson, mae, mse, top1, top3) for a single ``S``."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(truth)
    if mask.sum() < 1:
        return (float("nan"),) * 6
    diff = pred[mask] - truth[mask]
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    if mask.sum() >= 3:
        sp = spearman_per_dataset(pred, truth)
        pe = pearson_per_dataset(pred, truth)
        t3 = topk_recall_per_dataset(pred, truth, k=3)
    else:
        sp = pe = t3 = float("nan")
    t1 = topk_recall_per_dataset(pred, truth, k=1)
    return sp, pe, mae, mse, t1, t3


def score_by_stratum(
    explainer: Any, dataset: DatasetSpec
) -> Dict[str, StratumMetrics]:
    """Compute stratum-pooled metrics for one fitted explainer on one dataset."""
    p = int(dataset.X.shape[1])
    buckets: Dict[str, Dict[str, List[float]]] = {
        s: {"spearman": [], "pearson": [], "mae": [], "mse": [], "top1": [], "top3": []}
        for s in STRATA
    }
    counts: Dict[str, int] = {s: 0 for s in STRATA}

    for S, truth in dataset.ground_truth.value_by_state.items():
        bucket = classify_stratum(p, S)
        pred = np.asarray(
            explainer.conditional_predictive_values(sorted(S)), dtype=np.float64
        )
        sp, pe, mae, mse, t1, t3 = _per_state_metrics(pred, truth)
        buckets[bucket]["spearman"].append(sp)
        buckets[bucket]["pearson"].append(pe)
        buckets[bucket]["mae"].append(mae)
        buckets[bucket]["mse"].append(mse)
        buckets[bucket]["top1"].append(t1)
        buckets[bucket]["top3"].append(t3)
        counts[bucket] += 1

    out: Dict[str, StratumMetrics] = {}
    for bucket in STRATA:
        entries = buckets[bucket]

        def _mean(xs: List[float]) -> float:
            arr = np.asarray(xs, dtype=np.float64)
            if arr.size == 0 or not np.isfinite(arr).any():
                return float("nan")
            return float(np.nanmean(arr))

        out[bucket] = StratumMetrics(
            name=dataset.name,
            stratum=bucket,
            n_states=counts[bucket],
            n_features=p,
            spearman=_mean(entries["spearman"]),
            pearson=_mean(entries["pearson"]),
            mae=_mean(entries["mae"]),
            mse=_mean(entries["mse"]),
            top1_next_feature=_mean(entries["top1"]),
            top3_next_feature=_mean(entries["top3"]),
        )
    return out


def evaluate_head_by_stratum(
    factory: Factory,
    suite: Sequence[DatasetSpec],
    *,
    out_csv: Optional[str | Path] = None,
) -> List[StratumMetrics]:
    """Fit factory per dataset and return stratum-pooled metrics.

    One :class:`StratumMetrics` row is produced per (dataset, stratum) pair.
    If ``out_csv`` is set, rows are written there.
    """
    rows: List[StratumMetrics] = []
    for spec in suite:
        expl = factory(spec.X, spec.y)
        by_stratum = score_by_stratum(expl, spec)
        for bucket in STRATA:
            rows.append(by_stratum[bucket])

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [f.name for f in StratumMetrics.__dataclass_fields__.values()]  # type: ignore[attr-defined]
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    return rows


# ---------------------------------------------------------------------------
# Head-only fine-tuning
# ---------------------------------------------------------------------------


@dataclass
class FitHeadResult:
    """Summary of ``fit_head_only``."""

    steps: int
    final_loss: float
    head_state_dict: Mapping[str, Any]
    log_sigma2: Optional[List[float]] = None


def fit_head_only(
    model,
    value_head,
    batch_iterator: Callable[[], Iterable],
    *,
    num_steps: int,
    lr: float = 1e-3,
    huber_delta: float = 1.0,
    weighting: str = "uncertainty",
    device: str = "cpu",
) -> FitHeadResult:
    """Fine-tune ``value_head`` only; trunk ``model`` stays frozen.

    ``batch_iterator`` is a callable returning an iterable of ``(X, y, d,
    labels)`` tuples — the same 4-tuple that :class:`PriorDataset` emits
    after batching. Typical caller wraps a :class:`PriorDataset` in a
    closure that yields ``num_steps`` batches.

    Returns a :class:`FitHeadResult` containing the final head state dict.
    The trunk is not modified in place beyond its ``eval()`` call.
    """
    import torch
    from torch import nn
    from tabicl.train.multi_task_loss import PredictiveValueLoss

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    loss_module = PredictiveValueLoss(
        value_head=value_head,
        weighting=weighting,
        huber_delta=huber_delta,
    ).to(device)
    # Optimiser sees only head (+ uncertainty weights) parameters.
    opt = torch.optim.Adam(loss_module.parameters(), lr=lr)

    final_loss = float("nan")
    it = iter(batch_iterator())
    loss_module.train()
    for step in range(num_steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        X, y, d, labels = batch
        X = X.to(device)
        y = y.to(device)
        d = d.to(device)
        with torch.no_grad():
            # Trunk: return per-column embeddings (frozen forward).
            logits, col_emb = model(
                X, y_train=y, return_column_embeddings=True
            )
        total, _ = loss_module(logits=logits, y_true=y, col_emb=col_emb, labels=labels, d=d)
        opt.zero_grad()
        total.backward()
        opt.step()
        final_loss = float(total.detach().item())

    log_sigma2 = None
    if weighting == "uncertainty":
        log_sigma2 = loss_module.log_sigma2.detach().cpu().tolist()
    return FitHeadResult(
        steps=num_steps,
        final_loss=final_loss,
        head_state_dict={k: v.detach().cpu() for k, v in value_head.state_dict().items()},
        log_sigma2=log_sigma2,
    )


__all__ = [
    "STRATA",
    "classify_stratum",
    "StratumMetrics",
    "score_by_stratum",
    "evaluate_head_by_stratum",
    "FitHeadResult",
    "fit_head_only",
]
