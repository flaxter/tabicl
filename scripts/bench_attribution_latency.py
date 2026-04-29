"""KernelSHAP vs. conditional predictive value oracle — latency comparison.

Runs the grid locked in preregistration §11.5:

    p   in {8, 16, 32}
    n   in {500, 2000}
    n_Q in {1, 10, 100}    # number of information states queried

For each (p, n), fit the base estimator once. Then:

- ``kernelshap``: build a ``shap.KernelExplainer`` and call
  ``shap_values`` on the full training matrix. This is a single-call
  KernelSHAP budget (the standard SHAP reporting unit).
- ``oracle``: sample ``n_Q`` random information states ``S`` and call
  ``TabICLExplainer.conditional_predictive_values(S)`` for each.

Reports wall-clock seconds per operation + relative speedup to a CSV.

The §11.5 success criterion is that at ``n_Q = 10`` the total cost
(one fit + 10 queries) is at most the cost of a single KernelSHAP call.
No claim is made that one forward pass computes all ``S`` — we report
cost per information state honestly.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np


P_GRID = (8, 16, 32)
N_GRID = (500, 2000)
N_Q_GRID = (1, 10, 100)


@dataclass
class LatencyRow:
    method: str
    p: int
    n: int
    n_Q: int
    seconds_fit: float
    seconds_query: float
    seconds_total: float
    seconds_per_state: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def _synth(p: int, n: int, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Linear-Gaussian synthetic: ``Y = sum(beta_i X_i) + eps``, ``beta`` sparse."""
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    active = rng.choice(p, size=max(1, p // 3), replace=False)
    beta[active] = rng.uniform(0.5, 2.0, size=active.size) * rng.choice(
        [-1.0, 1.0], size=active.size
    )
    y = X @ beta + 0.1 * rng.standard_normal(n)
    return X, y


def _sample_states(p: int, n_Q: int, *, rng: np.random.Generator) -> List[List[int]]:
    """Draw ``n_Q`` random information states ``S``, covering diverse |S|."""
    states: List[List[int]] = []
    for _ in range(n_Q):
        # Uniform |S| across [0, p-1], then uniform feature subset of that size.
        k = int(rng.integers(0, p))
        sel = rng.choice(p, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        states.append(sorted(int(j) for j in sel))
    return states


# ---------------------------------------------------------------------------
# KernelSHAP backend
# ---------------------------------------------------------------------------


def _time_kernelshap(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_shap: int,
    n_background: int,
    nsamples: int,
    device: str,
) -> LatencyRow:
    import shap
    from tabicl import TabICLRegressor

    t0 = time.time()
    base = TabICLRegressor(device=device).fit(X, y)
    fit_time = time.time() - t0

    bg = shap.sample(X, n_background, random_state=42)
    explainer = shap.KernelExplainer(base.predict, bg)
    idx = np.random.default_rng(0).choice(X.shape[0], size=min(n_shap, X.shape[0]), replace=False)

    t1 = time.time()
    _ = explainer.shap_values(X[idx], nsamples=nsamples, silent=True)
    query_time = time.time() - t1

    return LatencyRow(
        method="kernelshap",
        p=int(X.shape[1]),
        n=int(X.shape[0]),
        n_Q=n_shap,          # reuse column: "n_Q" = n_shap rows explained
        seconds_fit=fit_time,
        seconds_query=query_time,
        seconds_total=fit_time + query_time,
        seconds_per_state=query_time / max(n_shap, 1),
        notes=f"nsamples={nsamples} bg={n_background}",
    )


# ---------------------------------------------------------------------------
# Oracle backend
# ---------------------------------------------------------------------------


def _time_oracle(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_Q: int,
    checkpoint: str,
    device: str,
    rng: np.random.Generator,
) -> LatencyRow:
    from tabicl import TabICLRegressor
    from tabicl.sklearn.explainer import TabICLExplainer

    t0 = time.time()
    base = TabICLRegressor(device=device)
    expl = TabICLExplainer(
        base_estimator=base,
        heads_checkpoint_path=checkpoint,
        device=device,
    ).fit(X, y)
    fit_time = time.time() - t0

    states = _sample_states(int(X.shape[1]), n_Q, rng=rng)
    t1 = time.time()
    for S in states:
        _ = expl.conditional_predictive_values(S)
    query_time = time.time() - t1

    return LatencyRow(
        method="oracle",
        p=int(X.shape[1]),
        n=int(X.shape[0]),
        n_Q=n_Q,
        seconds_fit=fit_time,
        seconds_query=query_time,
        seconds_total=fit_time + query_time,
        seconds_per_state=query_time / max(n_Q, 1),
        notes=f"device={device}",
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _write_csv(rows: Sequence[LatencyRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in LatencyRow.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def run_grid(
    *,
    checkpoint: str,
    out_csv: Path,
    p_grid: Sequence[int] = P_GRID,
    n_grid: Sequence[int] = N_GRID,
    n_Q_grid: Sequence[int] = N_Q_GRID,
    shap_background: int = 100,
    shap_nsamples: int = 512,
    shap_rows_explained: int = 10,
    device: str = "cpu",
    seed: int = 0,
) -> List[LatencyRow]:
    """Run the full grid and return rows. Writes CSV when ``out_csv`` is given."""
    rng = np.random.default_rng(seed)
    rows: List[LatencyRow] = []
    for p in p_grid:
        for n in n_grid:
            X, y = _synth(p, n, rng=rng)

            # One KernelSHAP measurement per (p, n) (it is not n_Q-dependent).
            ks = _time_kernelshap(
                X, y,
                n_shap=shap_rows_explained,
                n_background=shap_background,
                nsamples=shap_nsamples,
                device=device,
            )
            rows.append(ks)
            print(
                f"[latency] kernelshap p={p:>3} n={n:>4} "
                f"fit={ks.seconds_fit:.2f}s query={ks.seconds_query:.2f}s "
                f"per_row={ks.seconds_per_state:.3f}s",
                flush=True,
            )

            for n_Q in n_Q_grid:
                oc = _time_oracle(
                    X, y,
                    n_Q=n_Q,
                    checkpoint=checkpoint,
                    device=device,
                    rng=rng,
                )
                rows.append(oc)
                print(
                    f"[latency] oracle     p={p:>3} n={n:>4} n_Q={n_Q:>3} "
                    f"fit={oc.seconds_fit:.2f}s query={oc.seconds_query:.2f}s "
                    f"per_state={oc.seconds_per_state:.3f}s",
                    flush=True,
                )

    _write_csv(rows, out_csv)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a training checkpoint with the value-head state dict.",
    )
    parser.add_argument(
        "--out", default="results/bench_attribution_latency.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Single point p=8, n=500, n_Q=10 — arc devel shape check.",
    )
    parser.add_argument("--p-grid", type=int, nargs="+", default=list(P_GRID))
    parser.add_argument("--n-grid", type=int, nargs="+", default=list(N_GRID))
    parser.add_argument("--nq-grid", type=int, nargs="+", default=list(N_Q_GRID))
    parser.add_argument("--shap-nsamples", type=int, default=512)
    parser.add_argument("--shap-background", type=int, default=100)
    parser.add_argument("--shap-rows-explained", type=int, default=10)
    args = parser.parse_args(argv)

    if not Path(args.checkpoint).exists():
        print(f"[bench_attribution_latency] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    if args.smoke:
        args.p_grid = [8]
        args.n_grid = [500]
        args.nq_grid = [10]
        args.shap_rows_explained = 3

    out_csv = Path(args.out)
    rows = run_grid(
        checkpoint=args.checkpoint,
        out_csv=out_csv,
        p_grid=args.p_grid,
        n_grid=args.n_grid,
        n_Q_grid=args.nq_grid,
        shap_background=args.shap_background,
        shap_nsamples=args.shap_nsamples,
        shap_rows_explained=args.shap_rows_explained,
        device=args.device,
        seed=args.seed,
    )
    print(f"[bench_attribution_latency] wrote {len(rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
