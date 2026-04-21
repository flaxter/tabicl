"""Label-cost benchmark for compute_value_queries.

Measures per-dataset cost of the simulator-oracle plug-in estimator
under the locked §7.2 mixture (10 states default / 6 states backup).
Writes one CSV row per (prior_type, n, p, mixture, n_oracle) cell.

Usage:
    uv run python scripts/bench_labels.py \\
        --out /path/to/bench_labels.csv \\
        --n-datasets 30 \\
        --priors mlp_scm tree_scm \\
        --n-values 512 1024 \\
        --p-values 8 16 20 \\
        --n-oracle-values 256 512 \\
        --mixtures default backup

The Task 9 smoke target is >=500 ms/dataset on the most expensive
(prior, p, n_oracle) cell; if the primary mixture exceeds that target
on any cell in the training regime, the preregistration's backup
mixture kicks in.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from tabicl.prior.labels import compute_value_queries
from tabicl.prior.mlp_scm import MLPSCM
from tabicl.prior.tree_scm import TreeSCM
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from tabicl.prior.hp_sampling import HpSamplerList


_REGISTRY = {
    "mlp_scm": MLPSCM,
    "tree_scm": TreeSCM,
}


@torch.no_grad()
def _build_scm(prior_type: str, seq_len: int, num_features: int, rng_seed: int) -> Any:
    """Construct an SCM instance with sensible defaults.

    Uses the same HP-sampler machinery as PriorDataset so the SCM
    hyperparameters resemble training-time distributions.
    """
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    sampler = HpSamplerList(DEFAULT_SAMPLED_HP, device="cpu")
    sampled = sampler.sample()
    params: Dict[str, Any] = {
        **DEFAULT_FIXED_HP,
        **{k: (v() if callable(v) else v) for k, v in sampled.items()},
        "seq_len": seq_len,
        "num_features": num_features,
        "num_classes": 2,
        "device": "cpu",
    }
    cls = _REGISTRY[prior_type]
    return cls(**params), params


def _bench_one(
    prior_type: str,
    seq_len: int,
    num_features: int,
    n_oracle: int,
    mixture: str,
    n_datasets: int,
    rng_seed: int = 0,
) -> Dict[str, float]:
    """Measure compute_value_queries cost averaged over n_datasets draws."""
    times: List[float] = []
    query_counts: List[int] = []
    for ds_idx in range(n_datasets):
        scm, _ = _build_scm(prior_type, seq_len, num_features, rng_seed + ds_idx)
        X, y = scm()
        if isinstance(X, torch.Tensor):
            X = X.detach()
            y = y.detach()
        rng = np.random.default_rng(rng_seed + ds_idx)
        t0 = time.perf_counter()
        payload = compute_value_queries(
            scm,
            X,
            y,
            n_oracle=n_oracle,
            mixture=mixture,
            rng=rng,
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        query_counts.append(len(payload["value_queries"]))
    arr = np.array(times)
    return {
        "prior_type": prior_type,
        "n": seq_len,
        "p": num_features,
        "n_oracle": n_oracle,
        "mixture": mixture,
        "n_datasets": n_datasets,
        "mean_ms": float(arr.mean() * 1000),
        "median_ms": float(np.median(arr) * 1000),
        "p95_ms": float(np.quantile(arr, 0.95) * 1000),
        "max_ms": float(arr.max() * 1000),
        "queries_per_ds": float(np.mean(query_counts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-datasets", type=int, default=20)
    ap.add_argument("--priors", nargs="+", default=["mlp_scm", "tree_scm"])
    ap.add_argument("--n-values", nargs="+", type=int, default=[512, 1024])
    ap.add_argument("--p-values", nargs="+", type=int, default=[8, 16, 20])
    ap.add_argument("--n-oracle-values", nargs="+", type=int, default=[256, 512])
    ap.add_argument("--mixtures", nargs="+", default=["default", "backup"])
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prior_type", "n", "p", "n_oracle", "mixture", "n_datasets",
        "mean_ms", "median_ms", "p95_ms", "max_ms", "queries_per_ds",
    ]
    first_write = not args.out.exists()
    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if first_write:
            writer.writeheader()

        total_cells = (
            len(args.priors) * len(args.n_values) * len(args.p_values)
            * len(args.n_oracle_values) * len(args.mixtures)
        )
        cell = 0
        for prior_type in args.priors:
            for n in args.n_values:
                for p in args.p_values:
                    for n_oracle in args.n_oracle_values:
                        for mixture in args.mixtures:
                            cell += 1
                            print(
                                f"[{cell}/{total_cells}] prior={prior_type} n={n} "
                                f"p={p} n_oracle={n_oracle} mixture={mixture}",
                                flush=True,
                            )
                            row = _bench_one(
                                prior_type=prior_type,
                                seq_len=n,
                                num_features=p,
                                n_oracle=n_oracle,
                                mixture=mixture,
                                n_datasets=args.n_datasets,
                                rng_seed=args.rng_seed,
                            )
                            print(
                                f"  mean={row['mean_ms']:.1f}ms "
                                f"p95={row['p95_ms']:.1f}ms "
                                f"queries_per_ds={row['queries_per_ds']:.1f}",
                                flush=True,
                            )
                            writer.writerow(row)
                            f.flush()


if __name__ == "__main__":
    main()
