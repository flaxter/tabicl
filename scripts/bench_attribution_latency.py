"""Phase 6c — attribution latency: TabICLExplainer vs KernelSHAP.

Measures wall-clock cost of native attribution (one trunk forward pass
populating Heads A/I and caching embeddings for Head C) against the
KernelSHAP baseline on ``TabICLClassifier.predict_proba`` at several
sample budgets.

Attribution *quality* is out of scope — heads are randomly initialised.
The Phase 6c speedup heatmap only needs wall-clock numbers, and a
realised trained checkpoint is a separate deliverable (Task 7).

Grid:

- ``n_train`` ∈ {100, 500, 2000, 10000}
- ``p``       ∈ {5, 20, 100}
- ``n_explain`` fixed (default 10)
- KernelSHAP ``nsamples`` ∈ {256, 512, 1024}
- A hard per-KernelSHAP-call wall-clock cap (default 300 s) prevents a
  single big cell eating the whole SLURM job. Subsequent higher-budget
  calls in the same cell are skipped once the smallest budget times out.

CSV columns (flushed per row, db2b64d precedent from bench_labels.py):

``method, n_shap_samples, n_train, p, n_explain, wall_clock_s,
timed_out, notes``

Rows emitted per (n_train, p) cell:

- ``tabicl_explainer_fit`` — one ``expl.fit(X, y)`` with random-init heads
  (includes the base-estimator ``.fit`` that runs inside; separately
  captured as ``tabicl_base_fit`` so downstream analysis can subtract).
- ``tabicl_base_fit``      — ``base.fit(X, y)`` only (classification path).
- ``head_c_small|medium|large`` — ``expl.marginal_conditional_contributions(S)``
  for three conditioning-set sizes (on the already-fit explainer).
- ``kernel_shap``          — ``KernelExplainer.shap_values`` at each
  ``nsamples`` budget.

Usage
-----

Local smoke (two-cell tiny grid, < 2 min)::

    uv run python scripts/bench_attribution_latency.py \\
        --smoke --out /tmp/smoke.csv

Full run on arc (see ``learning-to-explain/bench_attribution_latency.slurm``)::

    uv run python scripts/bench_attribution_latency.py \\
        --out results/bench_attribution_latency.csv

Notes
-----

PLAN §6c nominally includes ``p=500`` and ``n_train=50000``; these are
excluded from the first-pass grid because KernelSHAP runtime on arc CPU
would be brutal. They are deliberately left for a later full sweep.
"""
from __future__ import annotations

import argparse
import csv
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.datasets import make_classification

from tabicl import TabICLClassifier, TabICLExplainer
from tabicl.model.heads import (
    ConditionalHead,
    InterventionalHead,
    ObservationalHead,
)


# --------------------------------------------------------------------------
# Timeout helper (SIGALRM — POSIX only; arc is Linux, so fine)
# --------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


@contextmanager
def _wallclock_cap(seconds: Optional[int]):
    """Raise ``_Timeout`` if the protected block runs longer than ``seconds``.

    ``seconds=None`` or ``<= 0`` disables the cap. Uses SIGALRM so a
    long-running C/NumPy loop in KernelSHAP will still be interrupted
    between Python bytecodes.
    """
    if not seconds or seconds <= 0:
        yield
        return
    prev = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


# --------------------------------------------------------------------------
# Head / data / base helpers
# --------------------------------------------------------------------------


def _fresh_heads(embed_dim: int) -> Dict[str, torch.nn.Module]:
    """Random-init attribution heads sized for ``embed_dim``.

    ``hidden_dim = embed_dim // 2`` matches ``tabicl/tests/test_explainer.py``
    and the Phase 2 default.
    """
    hidden_dim = embed_dim // 2
    # Deterministic init for reproducibility; weights aren't evaluated,
    # only timed.
    torch.manual_seed(0)
    return {
        "observational": ObservationalHead(embed_dim=embed_dim, hidden_dim=hidden_dim),
        "interventional": InterventionalHead(embed_dim=embed_dim, hidden_dim=hidden_dim),
        "conditional": ConditionalHead(embed_dim=embed_dim, hidden_dim=hidden_dim),
    }


def _make_data(n_train: int, p: int, n_explain: int, seed: int):
    """Synthetic binary classification — pure latency bench, no identifiability."""
    n_total = n_train + n_explain
    # n_informative capped at p so make_classification accepts the combo.
    n_informative = min(p, max(2, p // 2))
    X, y = make_classification(
        n_samples=n_total,
        n_features=p,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=2,
        random_state=seed,
    )
    X = X.astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:n_train + n_explain]


def _make_base() -> "TabICLClassifier":
    """A vanilla TabICLClassifier with the default released checkpoint."""
    return TabICLClassifier(n_estimators=1, random_state=0, verbose=False, device="cpu")


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


FIELDNAMES = [
    "method",
    "n_shap_samples",
    "n_train",
    "p",
    "n_explain",
    "wall_clock_s",
    "timed_out",
    "notes",
]


def _run_cell(
    n_train: int,
    p: int,
    n_explain: int,
    shap_nsamples: List[int],
    time_cap_s: int,
    seed: int,
    embed_dim: int,
    writer: csv.DictWriter,
    fh,
) -> None:
    """Run every method for one ``(n_train, p)`` cell and flush rows."""

    def emit(method: str, wall_clock_s: float, timed_out: bool,
             n_shap_samples: Optional[int] = None, notes: str = "") -> None:
        row = {
            "method": method,
            "n_shap_samples": "" if n_shap_samples is None else int(n_shap_samples),
            "n_train": n_train,
            "p": p,
            "n_explain": n_explain,
            "wall_clock_s": round(float(wall_clock_s), 4),
            "timed_out": "true" if timed_out else "false",
            "notes": notes,
        }
        writer.writerow(row)
        fh.flush()
        print(
            f"  {method:>22s}"
            f"{'' if n_shap_samples is None else f' ns={n_shap_samples}':>10s}"
            f"  t={wall_clock_s:>7.3f}s"
            f"{'  TIMED OUT' if timed_out else ''}"
            f"{('  [' + notes + ']') if notes else ''}",
            flush=True,
        )

    X_train, y_train, X_explain = _make_data(n_train, p, n_explain, seed=seed)

    # ------------------------------------------------------------------
    # 1. tabicl_base_fit — TabICLClassifier.fit in isolation (for later
    #    subtraction from the explainer-fit wall-clock).
    # ------------------------------------------------------------------
    base_for_shap = _make_base()
    t0 = time.perf_counter()
    base_for_shap.fit(X_train, y_train)
    t_base_fit = time.perf_counter() - t0
    emit("tabicl_base_fit", t_base_fit, timed_out=False)

    # Sanity-check embed_dim matches our hand-off value.
    actual_embed_dim = int(base_for_shap.model_config_["embed_dim"])
    if actual_embed_dim != embed_dim:
        emit(
            "warn_embed_dim_mismatch", 0.0, timed_out=False,
            notes=f"expected {embed_dim}, got {actual_embed_dim}",
        )
        embed_dim = actual_embed_dim  # trust reality over hand-off

    # ------------------------------------------------------------------
    # 2. tabicl_explainer_fit — fresh base + fresh heads, one pass.
    #    This refits the base (explainer .fit calls base.fit), so
    #    wall_clock_s includes the base fit captured separately above.
    # ------------------------------------------------------------------
    base_for_expl = _make_base()
    heads = _fresh_heads(embed_dim=embed_dim)
    expl = TabICLExplainer(base_estimator=base_for_expl, heads=heads)

    t0 = time.perf_counter()
    expl.fit(X_train, y_train)
    t_expl_fit = time.perf_counter() - t0
    emit(
        "tabicl_explainer_fit", t_expl_fit, timed_out=False,
        notes="includes base.fit",
    )

    # ------------------------------------------------------------------
    # 3. Head C query latency — three conditioning-set sizes.
    # ------------------------------------------------------------------
    small_S: List[int] = [0] if p >= 1 else []
    medium_S = list(range(min(p // 2, 10)))
    large_S = list(range(max(0, p - 2)))  # leave at least 2 features to score

    for label, S in [
        ("head_c_small", small_S),
        ("head_c_medium", medium_S),
        ("head_c_large", large_S),
    ]:
        t0 = time.perf_counter()
        _ = expl.marginal_conditional_contributions(S=S)
        t = time.perf_counter() - t0
        emit(label, t, timed_out=False, notes=f"|S|={len(S)}")

    # ------------------------------------------------------------------
    # 4. KernelSHAP baseline. If the smallest nsamples times out, skip
    #    the larger budgets — they will only be slower.
    # ------------------------------------------------------------------
    # Lazy import so the module loads without shap if someone runs
    # with --skip-shap.
    import shap

    background_size = min(100, X_train.shape[0])
    background = shap.sample(X_train, background_size, random_state=seed)

    skip_remaining = False
    for ns in sorted(shap_nsamples):
        if skip_remaining:
            emit(
                "kernel_shap", float("nan"), timed_out=True, n_shap_samples=ns,
                notes=f"skipped after smaller budget hit {time_cap_s}s cap",
            )
            continue

        ks = shap.KernelExplainer(base_for_shap.predict_proba, background)
        t0 = time.perf_counter()
        timed_out = False
        notes = ""
        try:
            with _wallclock_cap(time_cap_s):
                # silent=True suppresses shap's tqdm bar under SLURM.
                ks.shap_values(X_explain, nsamples=ns, silent=True)
        except _Timeout:
            timed_out = True
            notes = f"exceeded {time_cap_s}s cap"
            skip_remaining = True
        t = time.perf_counter() - t0
        emit("kernel_shap", t, timed_out, n_shap_samples=ns, notes=notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", default="results/bench_attribution_latency.csv")
    ap.add_argument("--grid-n", default="100,500,2000,10000")
    ap.add_argument("--grid-p", default="5,20,100")
    ap.add_argument("--n-explain", type=int, default=10)
    ap.add_argument("--shap-nsamples", default="256,512,1024")
    ap.add_argument(
        "--time-cap-s", type=int, default=300,
        help="Hard per-KernelSHAP-call wall-clock cap (seconds). "
             "The subsequent larger-budget calls in the same cell are "
             "skipped once the smallest budget trips the cap. 0 disables.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--embed-dim", type=int, default=128,
        help="Hand-off embed_dim used to size the random-init heads "
             "before the first fit. The default released TabICLClassifier "
             "checkpoint has embed_dim=128; mismatches are logged and the "
             "runtime value is used instead.",
    )
    ap.add_argument(
        "--smoke", action="store_true",
        help="Tiny two-cell grid for pre-SLURM sanity checks (<2 min).",
    )
    args = ap.parse_args()

    if args.smoke:
        args.grid_n = "100"
        args.grid_p = "5"
        args.shap_nsamples = "256"

    ns_list = [int(x) for x in args.grid_n.split(",") if x.strip()]
    ps_list = [int(x) for x in args.grid_p.split(",") if x.strip()]
    shap_nsamples = [int(x) for x in args.shap_nsamples.split(",") if x.strip()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"bench_attribution_latency: "
        f"grid n_train={ns_list} p={ps_list} "
        f"shap_nsamples={shap_nsamples} n_explain={args.n_explain} "
        f"time_cap_s={args.time_cap_s}",
        flush=True,
    )

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        fh.flush()

        for n_train in ns_list:
            for p in ps_list:
                print(f"\n=== cell n_train={n_train} p={p} ===", flush=True)
                _run_cell(
                    n_train=n_train,
                    p=p,
                    n_explain=args.n_explain,
                    shap_nsamples=shap_nsamples,
                    time_cap_s=args.time_cap_s,
                    seed=args.seed,
                    embed_dim=args.embed_dim,
                    writer=writer,
                    fh=fh,
                )

    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
