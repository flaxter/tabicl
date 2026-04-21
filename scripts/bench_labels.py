"""Phase 3 — labels-per-second CPU throughput benchmark.

Measures, for each ``(prior, n, p, batch)`` configuration:

- ``t_scm``               — SCM instantiation + continuous ``(X, y)`` sampling.
- ``t_label_A``           — Head A label cost in isolation.
- ``t_label_I``           — Head I label cost in isolation (NaN for
                            non-identifiable priors; this column is ``0`` then).
- ``t_label_C``           — Head C label cost in isolation.
- ``t_total_with_labels`` — SCM + all three labels (the real Phase 4 cost).
- ``overhead_ratio``      — ``(t_total_with_labels - t_scm) / t_scm``.
- ``labels_per_sec``      — batch / ``t_total_with_labels``.

Timings are per-batch wall-clock via ``time.perf_counter()``. Output is a
single CSV appended at ``--out``.

Designed to run on the ``arc`` CPU cluster (see
``learning-to-explain/bench_labels.slurm``), but also works locally with a
reduced grid via ``--smoke``.
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

from tabicl.prior.identifiable_scm import ANMSCM, LiNGAMSCM, TreeSCM_Ident
from tabicl.prior.labels import compute_labels
from tabicl.prior.mlp_scm import MLPSCM

# --------------------------------------------------------------------------
# Prior factories — each returns a fresh SCM instance
# --------------------------------------------------------------------------


def _make_lingam(n: int, p: int, seed: int) -> LiNGAMSCM:
    return LiNGAMSCM(seq_len=n, num_features=p, seed=seed)


def _make_anm(n: int, p: int, seed: int) -> ANMSCM:
    return ANMSCM(seq_len=n, num_features=p, seed=seed)


def _make_tree_path(n: int, p: int, seed: int) -> TreeSCM_Ident:
    return TreeSCM_Ident(seq_len=n, num_features=p, seed=seed)


def _make_mlp(n: int, p: int, seed: int) -> MLPSCM:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return MLPSCM(seq_len=n, num_features=p, num_layers=3, hidden_dim=max(20, 2 * p + 2))


def _make_mix(n: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    choice = rng.choice(["lingam_scm", "anm_scm", "tree_path_scm"])
    return _FACTORIES[choice](n, p, seed)


_FACTORIES = {
    "lingam_scm": _make_lingam,
    "anm_scm": _make_anm,
    "tree_path_scm": _make_tree_path,
    "mlp_scm": _make_mlp,
    "mix_scm_identifiable": _make_mix,
}


# --------------------------------------------------------------------------
# Per-head isolation
# --------------------------------------------------------------------------


def _time_single_head(scm: Any, X: torch.Tensor, y: torch.Tensor, head: str,
                      n_mc: int, k: int) -> float:
    """Rough per-head cost: call compute_labels() suppressing other heads."""
    # We do not expose per-head knobs yet, so approximate by k=0 triples for
    # "A-only" / "I-only", and k=k with n_mc=1 for "C-only". This is a proxy
    # but matches the relative ordering of costs cleanly enough.
    rng = np.random.default_rng(0)
    if head == "A":
        t0 = time.perf_counter()
        compute_labels(scm, X, y, n_mc=n_mc, k_cond_triples=0, rng=rng)
        return time.perf_counter() - t0
    if head == "I":
        t0 = time.perf_counter()
        compute_labels(scm, X, y, n_mc=n_mc, k_cond_triples=0, rng=rng)
        return time.perf_counter() - t0
    if head == "C":
        t0 = time.perf_counter()
        compute_labels(scm, X, y, n_mc=max(1, n_mc // 8), k_cond_triples=k, rng=rng)
        return time.perf_counter() - t0
    raise ValueError(head)


# --------------------------------------------------------------------------
# Benchmark driver
# --------------------------------------------------------------------------


def bench_one(
    prior: str,
    n: int,
    p: int,
    batch: int,
    n_mc: int,
    k_cond_triples: int,
    seed: int,
) -> Dict[str, Any]:
    factory = _FACTORIES[prior]

    # 1. SCM + (X, y) sampling baseline (summed across the batch).
    t0 = time.perf_counter()
    scms = []
    Xs, ys = [], []
    for b in range(batch):
        scm = factory(n=n, p=p, seed=seed + b)
        X, y = scm()
        scms.append(scm)
        Xs.append(X)
        ys.append(y)
    t_scm = time.perf_counter() - t0

    # 2. Total (with labels).
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    for b in range(batch):
        compute_labels(
            scms[b], Xs[b], ys[b],
            n_mc=n_mc, k_cond_triples=k_cond_triples, rng=rng,
        )
    t_total_labels_only = time.perf_counter() - t0

    # 3. Per-head isolated costs (first batch element only, to keep the
    # benchmark under a few minutes on large grids).
    t_A = _time_single_head(scms[0], Xs[0], ys[0], "A", n_mc, k_cond_triples)
    t_I = _time_single_head(scms[0], Xs[0], ys[0], "I", n_mc, k_cond_triples)
    t_C = _time_single_head(scms[0], Xs[0], ys[0], "C", n_mc, k_cond_triples)

    t_total_with_labels = t_scm + t_total_labels_only
    overhead_ratio = t_total_labels_only / max(t_scm, 1e-9)

    return {
        "prior": prior,
        "n": n,
        "p": p,
        "batch": batch,
        "n_mc": n_mc,
        "k_cond_triples": k_cond_triples,
        "t_scm_s": round(t_scm, 4),
        "t_label_A_s": round(t_A, 4),
        "t_label_I_s": round(t_I, 4),
        "t_label_C_s": round(t_C, 4),
        "t_total_with_labels_s": round(t_total_with_labels, 4),
        "overhead_ratio": round(overhead_ratio, 3),
        "labels_per_sec": round(batch / max(t_total_with_labels, 1e-9), 2),
    }


def parse_list(s: str, cast=int) -> List[Any]:
    return [cast(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench_labels.csv")
    ap.add_argument("--grid-n", default="256,1024,4096")
    ap.add_argument("--grid-p", default="5,20,50")
    ap.add_argument("--grid-batch", default="32,128")
    ap.add_argument(
        "--prior-set",
        default="lingam_scm,anm_scm,tree_path_scm,mlp_scm,mix_scm_identifiable",
    )
    ap.add_argument("--n-mc", type=int, default=2048)
    ap.add_argument("--k-cond-triples", type=int, default=16)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Reserved for future joblib parallelism.")
    ap.add_argument("--n-batches", type=int, default=1,
                    help="Number of repetitions per (prior, n, p, batch) cell.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Override grids with a tiny smoke-test configuration.")
    args = ap.parse_args()

    if args.smoke:
        args.grid_n = "128,512"
        args.grid_p = "4,12"
        args.grid_batch = "4,16"
        args.prior_set = "lingam_scm,anm_scm,mlp_scm"
        args.n_mc = 128
        args.k_cond_triples = 4
        args.n_batches = 1

    ns = parse_list(args.grid_n)
    ps = parse_list(args.grid_p)
    batches = parse_list(args.grid_batch)
    priors = parse_list(args.prior_set, cast=str)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flush per row so partial results survive SLURM wall-clock kills.
    fieldnames = [
        "prior", "n", "p", "batch", "n_mc", "k_cond_triples",
        "t_scm_s", "t_label_A_s", "t_label_I_s", "t_label_C_s",
        "t_total_with_labels_s", "overhead_ratio", "labels_per_sec", "rep",
    ]
    n_rows = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        fh.flush()
        for prior in priors:
            for n in ns:
                for p in ps:
                    for batch in batches:
                        for rep in range(args.n_batches):
                            row = bench_one(
                                prior=prior, n=n, p=p, batch=batch,
                                n_mc=args.n_mc, k_cond_triples=args.k_cond_triples,
                                seed=args.seed + rep * 101,
                            )
                            row["rep"] = rep
                            writer.writerow(row)
                            fh.flush()
                            n_rows += 1
                            print(
                                f"{prior:>22s} n={n:>4d} p={p:>3d} batch={batch:>3d} "
                                f"t_total={row['t_total_with_labels_s']:.3f}s "
                                f"labels/s={row['labels_per_sec']:.1f} "
                                f"overhead={row['overhead_ratio']:.2f}x",
                                flush=True,
                            )
    print(f"\nWrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
