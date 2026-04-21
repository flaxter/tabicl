"""Phase 6a driver — end-to-end attribution-quality benchmark.

Runs one or more ``EvalSuite``s through :func:`tabicl.eval.explainer_eval.evaluate_explainer`
using a ``TabICLExplainer`` factory and writes per-suite CSVs under
``--out-dir``.

Two operating modes:

- ``--heads-checkpoint <path>`` — production: wrap a real Phase 4
  checkpoint (trunk + heads) and evaluate attribution quality.
- no checkpoint — smoke / plumbing mode: evaluate **randomly-initialised
  heads** on a tiny in-memory TabICL. Useful as a CSV-shape regression
  check and as a baseline Spearman ≈ 0 datum for the paper-run delta.

Typical invocation::

    python scripts/bench_explainer_eval.py \
        --heads-checkpoint /data/.../head_finetune.ckpt \
        --suites in_distribution,held_out_prior,collider,id_boundary \
        --n-datasets 50 --seed 0 \
        --out-dir results/bench_explainer_eval

Smoke::

    python scripts/bench_explainer_eval.py --smoke --out-dir results/smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch

from tabicl import TabICLExplainer, TabICLRegressor
from tabicl.eval.explainer_eval import (
    EvalSuite,
    build_collider_suite,
    build_held_out_prior_suite,
    build_id_boundary_suite,
    build_in_distribution_suite,
    evaluate_explainer,
)
from tabicl.model.heads import (
    ConditionalHead,
    InterventionalHead,
    ObservationalHead,
)
from tabicl.model.tabicl import TabICL


SUITE_BUILDERS = {
    "in_distribution": build_in_distribution_suite,
    "held_out_prior": build_held_out_prior_suite,
    "collider": build_collider_suite,
    "id_boundary": build_id_boundary_suite,
}


def build_smoke_factory() -> Callable[[], TabICLExplainer]:
    """Factory that builds a tiny in-memory TabICL + fresh random heads.

    Matches the ``tests/test_explainer_eval.py`` pattern so this path
    requires no pretrained checkpoint and no GPU — suitable for arc
    devel smokes and CSV-plumbing regression.
    """
    EMBED_DIM = 32

    def _make_tiny() -> TabICL:
        torch.manual_seed(0)
        return TabICL(
            max_classes=0,
            embed_dim=EMBED_DIM,
            col_num_blocks=2,
            col_nhead=4,
            col_num_inds=8,
            icl_num_blocks=2,
            icl_nhead=4,
            row_num_blocks=2,
            row_nhead=4,
            row_num_cls=2,
            ff_factor=2,
        )

    def _install(estimator):
        tiny = _make_tiny()

        def _fake_load_model(self=estimator):
            self.model_path_ = None
            self.model_ = tiny
            self.model_config_ = {"max_classes": 0, "embed_dim": EMBED_DIM}
            self.model_.eval()

        estimator._load_model = _fake_load_model

    def _make():
        reg = TabICLRegressor(n_estimators=2, random_state=0, verbose=False)
        _install(reg)
        torch.manual_seed(0)
        heads = {
            "observational": ObservationalHead(embed_dim=EMBED_DIM),
            "interventional": InterventionalHead(embed_dim=EMBED_DIM),
            "conditional": ConditionalHead(embed_dim=EMBED_DIM),
        }
        return TabICLExplainer(base_estimator=reg, heads=heads)

    return _make


def build_production_factory(
    heads_checkpoint: str | Path,
    *,
    n_estimators: int,
    device: str,
) -> Callable[[], TabICLExplainer]:
    """Factory that loads a real Phase 4 checkpoint's heads for each case."""
    heads_checkpoint = Path(heads_checkpoint)

    def _make():
        reg = TabICLRegressor(
            n_estimators=n_estimators, random_state=0, verbose=False,
            device=device,
        )
        return TabICLExplainer(
            base_estimator=reg,
            heads_checkpoint_path=heads_checkpoint,
            device=device,
        )

    return _make


def _resolve_suite(
    name: str, *, n_datasets: int, seed: int, n_rows: int, n_mc: int, k_cond: int
) -> EvalSuite:
    builder = SUITE_BUILDERS[name]
    if name == "collider":
        return builder(
            n_datasets=n_datasets, seed=seed, n_rows=n_rows, k_cond_triples=k_cond,
        )
    return builder(
        n_datasets=n_datasets, seed=seed, n_rows=n_rows,
        n_mc=n_mc, k_cond_triples=k_cond,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True, help="Directory for per-suite CSVs.")
    p.add_argument(
        "--suites",
        default="in_distribution,held_out_prior,collider,id_boundary",
        help="Comma-separated subset of: " + ", ".join(SUITE_BUILDERS.keys()),
    )
    p.add_argument("--n-datasets", type=int, default=50)
    p.add_argument("--n-rows", type=int, default=500)
    p.add_argument("--n-mc", type=int, default=512)
    p.add_argument("--k-cond-triples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--heads-checkpoint", default=None,
        help="Path to Phase 4 checkpoint with trunk + head state dicts. "
             "If omitted, runs the tiny-TabICL smoke factory.",
    )
    p.add_argument("--n-estimators", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--smoke", action="store_true",
        help="Tiny-plumbing defaults: n_datasets=2, n_rows=64, 1 suite.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        n_datasets = 2
        n_rows = 64
        n_mc = 32
        k_cond = 3
        suites = ["in_distribution", "collider"]
    else:
        n_datasets = args.n_datasets
        n_rows = args.n_rows
        n_mc = args.n_mc
        k_cond = args.k_cond_triples
        suites = [s.strip() for s in args.suites.split(",") if s.strip()]

    unknown = [s for s in suites if s not in SUITE_BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown suite(s): {unknown}. Known: {list(SUITE_BUILDERS)}")

    if args.heads_checkpoint:
        factory = build_production_factory(
            args.heads_checkpoint,
            n_estimators=args.n_estimators,
            device=args.device,
        )
        mode_tag = "prod"
    else:
        factory = build_smoke_factory()
        mode_tag = "smoke"

    for suite_name in suites:
        print(f"[{mode_tag}] Building suite '{suite_name}' (n={n_datasets}, rows={n_rows})")
        suite = _resolve_suite(
            suite_name, n_datasets=n_datasets, seed=args.seed,
            n_rows=n_rows, n_mc=n_mc, k_cond=k_cond,
        )
        out_csv = out_dir / f"{suite_name}.csv"
        print(f"[{mode_tag}] Evaluating → {out_csv}")
        evaluate_explainer(factory, suite, out_csv=out_csv, verbose=args.verbose)

    print(f"Done. Wrote {len(suites)} CSV(s) under {out_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
