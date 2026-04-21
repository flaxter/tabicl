"""CLI driver for the attribution-quality harness.

Builds a suite (in-distribution or held-out-prior), constructs a
:class:`TabICLExplainer`-based factory from a checkpoint, and runs
:func:`tabicl.eval.explainer_eval.evaluate_explainer`. Writes a per-dataset
CSV with every metric in :class:`DatasetScore`.

A ``--smoke`` flag runs a tiny 3-dataset / small-p sweep for arc devel
partitions.

Example::

    uv run python scripts/bench_explainer_eval.py \\
        --checkpoint path/to/heads.pt \\
        --suite in_distribution \\
        --n-datasets 200 \\
        --num-features 8 \\
        --out results/bench_explainer_eval.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _build_suite(args: argparse.Namespace):
    from tabicl.eval.explainer_eval import (
        build_held_out_prior_suite,
        build_in_distribution_suite,
    )

    if args.suite == "in_distribution":
        return build_in_distribution_suite(
            n_datasets=args.n_datasets,
            num_features=args.num_features,
            seq_len=args.seq_len,
            seed=args.seed,
            n_oracle=args.n_oracle,
            n_bins=args.n_bins,
        )
    if args.suite == "held_out_prior":
        return build_held_out_prior_suite(
            n_datasets=args.n_datasets,
            num_features=args.num_features,
            seq_len=args.seq_len,
            seed=args.seed,
            n_oracle=args.n_oracle,
            n_bins=args.n_bins,
        )
    raise ValueError(f"Unknown suite {args.suite!r}")


def _build_factory(args: argparse.Namespace):
    """Return a ``(X, y) -> TabICLExplainer`` factory."""
    from tabicl import TabICLClassifier, TabICLRegressor
    from tabicl.sklearn.explainer import TabICLExplainer

    def factory(X: np.ndarray, y: np.ndarray):
        is_classification = np.issubdtype(np.asarray(y).dtype, np.integer) or (
            np.unique(y).size <= 10 and np.all(np.mod(np.asarray(y, dtype=float), 1) == 0)
        )
        base = (
            TabICLClassifier(device=args.device)
            if is_classification
            else TabICLRegressor(device=args.device)
        )
        expl = TabICLExplainer(
            base_estimator=base,
            heads_checkpoint_path=args.checkpoint,
            device=args.device,
        )
        return expl.fit(X, y)

    return factory


def _smoke_args_override(args: argparse.Namespace) -> argparse.Namespace:
    """Tiny-suite knobs for ``--smoke`` (arc devel partition)."""
    args.n_datasets = 3
    args.num_features = 4
    args.seq_len = 200
    args.n_oracle = 400
    return args


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a training checkpoint with the value-head state dict.",
    )
    parser.add_argument(
        "--suite", default="in_distribution",
        choices=("in_distribution", "held_out_prior"),
    )
    parser.add_argument("--n-datasets", type=int, default=200)
    parser.add_argument("--num-features", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--n-oracle", type=int, default=2000)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cpu",
        help="Device for the TabICL base estimators and explainer.",
    )
    parser.add_argument(
        "--out", default="results/bench_explainer_eval.csv",
        help="Output CSV path (will be created).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny sweep for arc devel: 3 datasets / p=4 / seq_len=200.",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args = _smoke_args_override(args)

    if not Path(args.checkpoint).exists():
        print(f"[bench_explainer_eval] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    from tabicl.eval.explainer_eval import evaluate_explainer

    t0 = time.time()
    suite = _build_suite(args)
    t_suite = time.time() - t0
    print(
        f"[bench_explainer_eval] built {args.suite} suite of {len(suite)} datasets "
        f"in {t_suite:.1f}s",
        flush=True,
    )

    factory = _build_factory(args)
    t1 = time.time()
    scores = evaluate_explainer(factory, suite, out_csv=args.out)
    t_eval = time.time() - t1
    print(
        f"[bench_explainer_eval] scored {len(scores)} datasets in {t_eval:.1f}s "
        f"-> {args.out}",
        flush=True,
    )

    # Terse summary for stdout / SLURM log grep.
    def _nanmean(attr: str) -> float:
        xs = np.array([getattr(s, attr) for s in scores], dtype=np.float64)
        if not np.isfinite(xs).any():
            return float("nan")
        return float(np.nanmean(xs))

    print(
        "  spearman_value={:.3f}  pearson_value={:.3f}  mae={:.3f}  "
        "top1={:.3f}  top3={:.3f}  suff={:.3f}  nec={:.3f}  auc={:.3f}".format(
            _nanmean("spearman_value"),
            _nanmean("pearson_value"),
            _nanmean("mae_value"),
            _nanmean("top1_next_feature"),
            _nanmean("top3_next_feature"),
            _nanmean("spearman_sufficiency"),
            _nanmean("spearman_necessity"),
            _nanmean("acquisition_auc"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
