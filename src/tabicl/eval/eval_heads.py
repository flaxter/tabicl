"""§11.1 + §11.2 evaluation of the conditional predictive value head.

Consumes a heads-only checkpoint (produced by training with the value head
enabled) and writes two CSVs read directly by the paper:

- ``--out_s11_1`` (§11.1): per-dataset / per-|S|-stratum Spearman, Pearson,
  MAE on RMS scale, top-1 recall, top-3 recall. Strata are
  ``empty`` (|S|=0), ``singleton`` (|S|=1), ``small`` (|S| in {2,3,4}),
  ``medium`` (|S| ~ p/2), ``near_full`` (|S| in {p-2, p-1}) — locked by the
  preregistration.
- ``--out_s11_2`` (§11.2): per-dataset Spearman/Pearson/MAE for the two
  endpoints ``predictive_sufficiency_`` (s_i = r_{i|emptyset}) and
  ``predictive_necessity_`` (n_i = r_{i|[p]\\{i}}). Trailing rows report the
  pooled calibration slope and intercept across every (target, pred) pair.

The synthetic datasets are drawn through
:func:`tabicl.eval.explainer_eval.build_in_distribution_suite` at
``mix_scm`` — the same prior mixture the head was trained on. Features are
sampled in ``[min_features, max_features]``; the default range
(13-20) is chosen to force the mixture-sampling path in the suite
builder (p<=12 triggers full 2^p enumeration, which blows up).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from tabicl import TabICLClassifier
from tabicl.eval.explainer_eval import EvalCase, build_in_distribution_suite
from tabicl.eval.metrics import (
    pearson_per_dataset,
    spearman_per_dataset,
    topk_recall_per_dataset,
)
from tabicl.sklearn.explainer import TabICLExplainer


STRATUM_ORDER = ("empty", "singleton", "small", "medium", "near_full")


def stratum_of_state(state: frozenset, p: int) -> str | None:
    """Classify ``S`` into one of the preregistration strata or None."""
    s = len(state)
    if s == 0:
        return "empty"
    if s == 1:
        return "singleton"
    if 2 <= s <= 4:
        return "small"
    if s == max(p // 2, 0):
        return "medium"
    if p >= 2 and s in (p - 2, p - 1):
        return "near_full"
    return None


@dataclass
class S11_1Row:
    dataset_id: str
    p: int
    stratum: str
    n_states: int
    spearman: float
    pearson: float
    mae: float
    top1: float
    top3: float


@dataclass
class S11_2Row:
    dataset_id: str
    p: int
    spearman_s: float
    pearson_s: float
    mae_s: float
    spearman_n: float
    pearson_n: float
    mae_n: float


def _nanmean(xs) -> float:
    arr = np.asarray(list(xs), dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def _pooled_slope_intercept(pairs: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """OLS slope + intercept of pred = slope * target + intercept."""
    if len(pairs) < 2:
        return float("nan"), float("nan")
    arr = np.asarray(pairs, dtype=np.float64)
    t = arr[:, 0]
    y = arr[:, 1]
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2 or np.std(t[mask]) == 0.0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    return float(slope), float(intercept)


def evaluate_dataset(
    case: EvalCase,
    explainer: TabICLExplainer,
) -> tuple[list[S11_1Row], S11_2Row, list[tuple[float, float]], list[tuple[float, float]]]:
    """Evaluate one dataset; return §11.1 rows, §11.2 row, pooled calib pairs."""
    p = int(case.X_train.shape[1])
    explainer.fit(case.X_train, case.y_train)

    per_stratum: dict[str, dict[str, list[float]]] = {
        k: {"spearman": [], "pearson": [], "mae": [], "top1": [], "top3": []}
        for k in STRATUM_ORDER
    }
    per_stratum_counts = {k: 0 for k in STRATUM_ORDER}

    # Cache already-computed states so we don't double-query.
    seen: dict[frozenset, np.ndarray] = {}

    for state, target in case.ground_truth.value_by_state.items():
        stratum = stratum_of_state(state, p)
        if stratum is None:
            continue
        if state in seen:
            pred = seen[state]
        else:
            pred = np.asarray(
                explainer.conditional_predictive_values(sorted(state)),
                dtype=np.float64,
            )
            seen[state] = pred
        target = np.asarray(target, dtype=np.float64)

        per_stratum[stratum]["spearman"].append(spearman_per_dataset(pred, target))
        per_stratum[stratum]["pearson"].append(pearson_per_dataset(pred, target))
        per_stratum[stratum]["top1"].append(topk_recall_per_dataset(pred, target, 1))
        if p >= 3:
            per_stratum[stratum]["top3"].append(
                topk_recall_per_dataset(pred, target, min(3, p))
            )
        mask = np.isfinite(pred) & np.isfinite(target)
        if mask.any():
            per_stratum[stratum]["mae"].append(
                float(np.mean(np.abs(pred[mask] - target[mask])))
            )
        per_stratum_counts[stratum] += 1

    rows_11_1: list[S11_1Row] = []
    for stratum in STRATUM_ORDER:
        acc = per_stratum[stratum]
        rows_11_1.append(
            S11_1Row(
                dataset_id=case.dataset_id,
                p=p,
                stratum=stratum,
                n_states=per_stratum_counts[stratum],
                spearman=_nanmean(acc["spearman"]),
                pearson=_nanmean(acc["pearson"]),
                mae=_nanmean(acc["mae"]),
                top1=_nanmean(acc["top1"]),
                top3=_nanmean(acc["top3"]),
            )
        )

    # §11.2 endpoints. The explainer caches sufficiency/necessity at fit().
    suff_pred = np.asarray(explainer.predictive_sufficiency_, dtype=np.float64)
    suff_target_raw = case.ground_truth.value_by_state.get(frozenset())
    if suff_target_raw is None:
        suff_target = np.full(p, np.nan, dtype=np.float64)
    else:
        suff_target = np.asarray(suff_target_raw, dtype=np.float64)

    necessity_pred = np.asarray(explainer.predictive_necessity_, dtype=np.float64)
    necessity_target = np.full(p, np.nan, dtype=np.float64)
    for i in range(p):
        loo_state = frozenset(j for j in range(p) if j != i)
        t = case.ground_truth.value_by_state.get(loo_state)
        if t is not None and len(t) > i and np.isfinite(t[i]):
            necessity_target[i] = float(t[i])

    finite_s = np.isfinite(suff_pred) & np.isfinite(suff_target)
    finite_n = np.isfinite(necessity_pred) & np.isfinite(necessity_target)

    row_11_2 = S11_2Row(
        dataset_id=case.dataset_id,
        p=p,
        spearman_s=spearman_per_dataset(suff_pred, suff_target),
        pearson_s=pearson_per_dataset(suff_pred, suff_target),
        mae_s=float(np.mean(np.abs(suff_pred[finite_s] - suff_target[finite_s])))
        if finite_s.any()
        else float("nan"),
        spearman_n=spearman_per_dataset(necessity_pred, necessity_target),
        pearson_n=pearson_per_dataset(necessity_pred, necessity_target),
        mae_n=float(np.mean(np.abs(necessity_pred[finite_n] - necessity_target[finite_n])))
        if finite_n.any()
        else float("nan"),
    )

    s_pairs = list(zip(suff_target[finite_s].tolist(), suff_pred[finite_s].tolist()))
    n_pairs = list(
        zip(necessity_target[finite_n].tolist(), necessity_pred[finite_n].tolist())
    )

    return rows_11_1, row_11_2, s_pairs, n_pairs


def write_s11_1(path: str | Path, rows: Sequence[S11_1Row]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(S11_1Row.__dataclass_fields__.keys())

    by_stratum: dict[str, list[S11_1Row]] = {k: [] for k in STRATUM_ORDER}
    for r in rows:
        by_stratum[r.stratum].append(r)

    stratum_means: list[dict] = []
    for stratum in STRATUM_ORDER:
        rs = by_stratum[stratum]
        stratum_means.append(
            {
                "dataset_id": "mean",
                "p": "",
                "stratum": stratum,
                "n_states": int(sum(r.n_states for r in rs)),
                "spearman": _nanmean([r.spearman for r in rs]),
                "pearson": _nanmean([r.pearson for r in rs]),
                "mae": _nanmean([r.mae for r in rs]),
                "top1": _nanmean([r.top1 for r in rs]),
                "top3": _nanmean([r.top3 for r in rs]),
            }
        )

    across = {
        "dataset_id": "mean",
        "p": "",
        "stratum": "across_strata",
        "n_states": "",
        "spearman": _nanmean([m["spearman"] for m in stratum_means]),
        "pearson": _nanmean([m["pearson"] for m in stratum_means]),
        "mae": _nanmean([m["mae"] for m in stratum_means]),
        "top1": _nanmean([m["top1"] for m in stratum_means]),
        "top3": _nanmean([m["top3"] for m in stratum_means]),
    }

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
        for m in stratum_means:
            w.writerow(m)
        w.writerow(across)


def write_s11_2(
    path: str | Path,
    rows: Sequence[S11_2Row],
    s_pairs: Sequence[tuple[float, float]],
    n_pairs: Sequence[tuple[float, float]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(S11_2Row.__dataclass_fields__.keys()) + [
        "calib_slope",
        "calib_intercept",
    ]

    s_slope, s_intercept = _pooled_slope_intercept(s_pairs)
    n_slope, n_intercept = _pooled_slope_intercept(n_pairs)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            d["calib_slope"] = ""
            d["calib_intercept"] = ""
            w.writerow(d)

        w.writerow(
            {
                "dataset_id": "mean",
                "p": "",
                "spearman_s": _nanmean([r.spearman_s for r in rows]),
                "pearson_s": _nanmean([r.pearson_s for r in rows]),
                "mae_s": _nanmean([r.mae_s for r in rows]),
                "spearman_n": _nanmean([r.spearman_n for r in rows]),
                "pearson_n": _nanmean([r.pearson_n for r in rows]),
                "mae_n": _nanmean([r.mae_n for r in rows]),
                "calib_slope": "",
                "calib_intercept": "",
            }
        )
        w.writerow(
            {
                "dataset_id": "calib_s_pooled",
                "p": "",
                "spearman_s": "",
                "pearson_s": "",
                "mae_s": "",
                "spearman_n": "",
                "pearson_n": "",
                "mae_n": "",
                "calib_slope": s_slope,
                "calib_intercept": s_intercept,
            }
        )
        w.writerow(
            {
                "dataset_id": "calib_n_pooled",
                "p": "",
                "spearman_s": "",
                "pearson_s": "",
                "mae_s": "",
                "spearman_n": "",
                "pearson_n": "",
                "mae_n": "",
                "calib_slope": n_slope,
                "calib_intercept": n_intercept,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--out_s11_1", required=True)
    parser.add_argument("--out_s11_2", required=True)
    parser.add_argument("--n_datasets", type=int, default=200)
    parser.add_argument("--n_rows", type=int, default=512)
    parser.add_argument("--min_features", type=int, default=13)
    parser.add_argument("--max_features", type=int, default=20)
    parser.add_argument("--n_oracle", type=int, default=512)
    parser.add_argument(
        "--mixture", default="backup", choices=("default", "backup"),
        help="Conditioning-state mixture. 'backup' matches the training run "
             "activated 2026-04-21 (preregistration §1 backup clause).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--log_every", type=int, default=10,
        help="Print a progress line every N datasets.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[eval_heads] checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    print(
        f"[eval_heads] building in_distribution suite: "
        f"n_datasets={args.n_datasets} p in [{args.min_features},{args.max_features}] "
        f"n_rows={args.n_rows} n_oracle={args.n_oracle} mixture={args.mixture}",
        flush=True,
    )
    suite = build_in_distribution_suite(
        n_datasets=args.n_datasets,
        seed=args.seed,
        n_rows=args.n_rows,
        min_features=args.min_features,
        max_features=args.max_features,
        n_oracle=args.n_oracle,
        mixture=args.mixture,
    )
    print(f"[eval_heads] suite built: {len(suite.cases)} cases", flush=True)

    rows_11_1: list[S11_1Row] = []
    rows_11_2: list[S11_2Row] = []
    all_s_pairs: list[tuple[float, float]] = []
    all_n_pairs: list[tuple[float, float]] = []

    for i, case in enumerate(suite.cases):
        base = TabICLClassifier(device=args.device)
        explainer = TabICLExplainer(
            base_estimator=base,
            heads_checkpoint_path=str(checkpoint_path),
            device=args.device,
        )
        ds_11_1, ds_11_2, s_pairs, n_pairs = evaluate_dataset(case, explainer)
        rows_11_1.extend(ds_11_1)
        rows_11_2.append(ds_11_2)
        all_s_pairs.extend(s_pairs)
        all_n_pairs.extend(n_pairs)

        if (i + 1) % max(1, args.log_every) == 0 or i == 0 or (i + 1) == len(suite.cases):
            print(
                f"[eval_heads] {i+1}/{len(suite.cases)} p={case.X_train.shape[1]} "
                f"rho_suff={ds_11_2.spearman_s:.3f} rho_nec={ds_11_2.spearman_n:.3f}",
                flush=True,
            )

    write_s11_1(args.out_s11_1, rows_11_1)
    write_s11_2(args.out_s11_2, rows_11_2, all_s_pairs, all_n_pairs)

    # Terse grep summary for slurm log.
    by_stratum_mean = {
        stratum: _nanmean([r.spearman for r in rows_11_1 if r.stratum == stratum])
        for stratum in STRATUM_ORDER
    }
    across_strata = _nanmean(list(by_stratum_mean.values()))
    suff_mean = _nanmean([r.spearman_s for r in rows_11_2])
    nec_mean = _nanmean([r.spearman_n for r in rows_11_2])
    print(
        "[eval_heads] §11.1 mean spearman per stratum: "
        + " ".join(f"{k}={v:.3f}" for k, v in by_stratum_mean.items())
        + f"  across_strata={across_strata:.3f}",
        flush=True,
    )
    print(
        f"[eval_heads] §11.2 mean: spearman_s={suff_mean:.3f} spearman_n={nec_mean:.3f}",
        flush=True,
    )
    print(f"[eval_heads] wrote {args.out_s11_1} and {args.out_s11_2}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
