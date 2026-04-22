"""§11.3 qualitative quintet evaluator.

Five hand-designed DGPs — A (standalone), B (duplicate), C (pure
interaction), D (mediator chain), E (user-known context) — that each test a
specific qualitative pattern of ``r_{i|S} = sqrt(max(Delta_{i|S}, 0))``.
Preregistration §11.3 locks the pattern per panel and the binary
pass/fail rule. This module:

1. Samples the panel's DGP (continuous Y), discretises Y via quantile
   bins so the classifier trunk can consume it, fits a
   :class:`TabICLExplainer` with the supplied value head,
2. Queries ``conditional_predictive_values(S)`` at the canonical
   ``S`` states for that panel (e.g. ``empty``, ``{X1}``, ``{partner}``),
3. Runs the locked sign-of-rank-relation checks, emits one row per
   ``(panel, feature, S_state)`` to ``results/quintet_11_3.csv``, followed
   by a per-panel PASS/FAIL summary and a final ``pass_rate`` row.

Panel E re-uses the A-style focal-feature + interaction DGP with a proxy
and an interaction partner (Y = 0.4*X1 + X1*X3 + eps). This matches
``notes/preregistration.md`` §11.3 table row E — the three S states are
``empty``, ``{proxy}``, ``{partner}``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from tabicl import TabICLClassifier
from tabicl.sklearn.explainer import TabICLExplainer


NOISE = 0.15
DUPLICATE_NOISE = 0.10
MEDIATOR_NOISE = 0.15
CONTEXT_PROXY_NOISE = 0.15
CONTEXT_SIGNAL_WEIGHT = 0.40


# ---------------------------------------------------------------------------
# DGP samplers — continuous Y (discretised later for classifier training).
# ---------------------------------------------------------------------------


def _sample_A(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A: Y = X1 + eps; X2 independent noise."""
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = x1 + rng.normal(0.0, NOISE, n)
    return np.column_stack([x1, x2]), y


def _sample_B(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """B: X2 = X1 + tiny eps; Y = X1 + eps (duplicate feature)."""
    x1 = rng.normal(0.0, 1.0, n)
    x2 = x1 + rng.normal(0.0, DUPLICATE_NOISE, n)
    y = x1 + rng.normal(0.0, NOISE, n)
    return np.column_stack([x1, x2]), y


def _sample_C(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """C: Y = X1 * X2 + eps (pure interaction, independent centred X)."""
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = x1 * x2 + rng.normal(0.0, NOISE, n)
    return np.column_stack([x1, x2]), y


def _sample_D(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """D: X2 -> X1 -> Y (mediator chain). Features are (X1, X2)."""
    x2 = rng.normal(0.0, 1.0, n)
    x1 = x2 + rng.normal(0.0, MEDIATOR_NOISE, n)
    y = x1 + rng.normal(0.0, NOISE, n)
    return np.column_stack([x1, x2]), y


def _sample_E(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """E: Y = 0.4*X1 + X1*X3 + eps; X2 = X1 + proxy noise.

    Features are (X1_focal, X2_proxy, X3_partner). Exercises the
    §11.3 qualitative claim that ``Delta_{i|S}`` changes across S:
    revealing the partner *raises* X1's marginal value (interaction
    unlocked), while revealing the proxy *lowers* it (part of X1 is
    already explained).
    """
    x1 = rng.normal(0.0, 1.0, n)
    x2 = x1 + rng.normal(0.0, CONTEXT_PROXY_NOISE, n)
    x3 = rng.normal(0.0, 1.0, n)
    y = CONTEXT_SIGNAL_WEIGHT * x1 + x1 * x3 + rng.normal(0.0, NOISE, n)
    return np.column_stack([x1, x2, x3]), y


# ---------------------------------------------------------------------------
# Panel specs and the locked §11.3 pattern checks.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelSpec:
    panel: str
    feature_names: tuple[str, ...]
    sample: Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]]
    canonical_states: Dict[str, Tuple[int, ...]]
    checks: list[tuple[str, Callable[[Dict[str, np.ndarray]], bool]]] = field(
        default_factory=list
    )


def _panels() -> list[PanelSpec]:
    def A_checks(preds):
        # s_1 high, s_2 ~ 0  ==>  r_{X1|empty} > r_{X2|empty}
        # n_1 high, n_2 ~ 0  ==>  r_{X1|{X2}} > r_{X2|{X1}}
        c1 = preds["empty"][0] > preds["empty"][1]
        c2 = preds["{X2}"][0] > preds["{X1}"][1]
        return [
            ("s1 > s2", c1),
            ("n1 > n2", c2),
        ]

    def B_checks(preds):
        # Duplicate: s_1 and s_2 both non-trivial and similar, but
        # necessity collapses for each once the twin is known.
        e = preds["empty"]
        c1 = e[0] > 0.05 and e[1] > 0.05           # both useful standalone
        c2 = preds["{X2}"][0] < e[0] * 0.5          # X1 collapses once X2 known
        c3 = preds["{X1}"][1] < e[1] * 0.5          # X2 collapses once X1 known
        return [
            ("s1 > 0.05 and s2 > 0.05", c1),
            ("n1 < s1 / 2", c2),
            ("n2 < s2 / 2", c3),
        ]

    def C_checks(preds):
        # Pure interaction: s_1 ~ s_2 ~ 0; context reveals the interaction.
        e = preds["empty"]
        c1 = preds["{X2}"][0] > e[0] + 0.05         # r_{X1|{X2}} > r_{X1|empty}
        c2 = preds["{X1}"][1] > e[1] + 0.05         # r_{X2|{X1}} > r_{X2|empty}
        return [
            ("r_{X1|{X2}} > r_{X1|empty}", c1),
            ("r_{X2|{X1}} > r_{X2|empty}", c2),
        ]

    def D_checks(preds):
        # Mediator X2 -> X1 -> Y: upstream has standalone value,
        # necessity collapses once X1 is known.
        c1 = preds["empty"][1] > 0.05               # s_2 > 0
        c2 = preds["{X1}"][1] < preds["empty"][1]   # n_2 < s_2 once X1 known
        return [
            ("s2 > 0.05", c1),
            ("r_{X2|{X1}} < r_{X2|empty}", c2),
        ]

    def E_checks(preds):
        # Focal feature X1's marginal value increases when the interaction
        # partner is revealed, and decreases when the proxy is revealed.
        empty_v = preds["empty"][0]
        proxy_v = preds["{proxy}"][0]
        partner_v = preds["{partner}"][0]
        c1 = partner_v > empty_v                    # partner unlocks X1
        c2 = proxy_v < empty_v                      # proxy explains part of X1
        return [
            ("r_{X1|{partner}} > r_{X1|empty}", c1),
            ("r_{X1|{proxy}} < r_{X1|empty}", c2),
        ]

    return [
        PanelSpec(
            panel="A",
            feature_names=("X1", "X2"),
            sample=_sample_A,
            canonical_states={"empty": (), "{X1}": (0,), "{X2}": (1,)},
            checks=[("pattern", A_checks)],
        ),
        PanelSpec(
            panel="B",
            feature_names=("X1", "X2"),
            sample=_sample_B,
            canonical_states={"empty": (), "{X1}": (0,), "{X2}": (1,)},
            checks=[("pattern", B_checks)],
        ),
        PanelSpec(
            panel="C",
            feature_names=("X1", "X2"),
            sample=_sample_C,
            canonical_states={"empty": (), "{X1}": (0,), "{X2}": (1,)},
            checks=[("pattern", C_checks)],
        ),
        PanelSpec(
            panel="D",
            feature_names=("X1", "X2"),
            sample=_sample_D,
            canonical_states={"empty": (), "{X1}": (0,), "{X2}": (1,)},
            checks=[("pattern", D_checks)],
        ),
        PanelSpec(
            panel="E",
            feature_names=("X1_focal", "X2_proxy", "X3_partner"),
            sample=_sample_E,
            canonical_states={"empty": (), "{proxy}": (1,), "{partner}": (2,)},
            checks=[("pattern", E_checks)],
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _discretise(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin continuous Y into ``n_bins`` classes."""
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    labels = np.clip(np.searchsorted(edges, y) - 1, 0, n_bins - 1)
    return labels.astype(np.int64)


def run_panel(
    panel: PanelSpec,
    *,
    checkpoint_path: str | Path,
    n_rows: int,
    seed: int,
    n_bins: int,
    device: str,
) -> tuple[list[dict], list[tuple[str, bool]]]:
    rng = np.random.default_rng(seed)
    X, y = panel.sample(n_rows, rng)
    y_cls = _discretise(y, n_bins=n_bins)

    base = TabICLClassifier(device=device)
    explainer = TabICLExplainer(
        base_estimator=base,
        heads_checkpoint_path=str(checkpoint_path),
        device=device,
    )
    explainer.fit(X, y_cls)

    preds: Dict[str, np.ndarray] = {}
    for state_name, S in panel.canonical_states.items():
        preds[state_name] = np.asarray(
            explainer.conditional_predictive_values(list(S)), dtype=np.float64
        )

    rows: list[dict] = []
    for state_name, pred in preds.items():
        for idx, fname in enumerate(panel.feature_names):
            rows.append(
                {
                    "panel": panel.panel,
                    "feature": fname,
                    "S_state": state_name,
                    "predicted_rms": float(pred[idx]),
                }
            )

    check_results: list[tuple[str, bool]] = []
    for check_name, check_fn in panel.checks:
        for sub_name, passed in check_fn(preds):
            check_results.append((f"{check_name}: {sub_name}", bool(passed)))

    return rows, check_results


def write_csv(
    path: str | Path,
    all_rows: Sequence[dict],
    panel_results: Dict[str, list[tuple[str, bool]]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["panel", "feature", "S_state", "predicted_rms"]

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

        passed_panels = 0
        for panel_name, checks in panel_results.items():
            panel_pass = all(ok for _, ok in checks)
            if panel_pass:
                passed_panels += 1
            w.writerow(
                {
                    "panel": panel_name,
                    "feature": "PASS" if panel_pass else "FAIL",
                    "S_state": "; ".join(
                        f"{name}={int(ok)}" for name, ok in checks
                    ),
                    "predicted_rms": int(panel_pass),
                }
            )
        pass_rate = passed_panels / max(1, len(panel_results))
        w.writerow(
            {
                "panel": "all",
                "feature": "pass_rate",
                "S_state": f"{passed_panels}/{len(panel_results)}",
                "predicted_rms": pass_rate,
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--out", default="results/quintet_11_3.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--n_rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_bins", type=int, default=4,
                        help="Quantile bins for y discretisation.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of panels to run (e.g. A,C,E).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[quintet] checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    panels = _panels()
    if args.only:
        requested = {p.strip() for p in args.only.split(",") if p.strip()}
        panels = [p for p in panels if p.panel in requested]

    all_rows: list[dict] = []
    panel_results: Dict[str, list[tuple[str, bool]]] = {}
    for panel in panels:
        print(f"[quintet] running panel {panel.panel} ({len(panel.feature_names)} features)",
              flush=True)
        rows, checks = run_panel(
            panel,
            checkpoint_path=checkpoint_path,
            n_rows=args.n_rows,
            seed=args.seed,
            n_bins=args.n_bins,
            device=args.device,
        )
        all_rows.extend(rows)
        panel_results[panel.panel] = checks
        panel_pass = all(ok for _, ok in checks)
        print(
            f"[quintet] panel {panel.panel} {'PASS' if panel_pass else 'FAIL'}: "
            + "; ".join(f"{name}={int(ok)}" for name, ok in checks),
            flush=True,
        )

    write_csv(args.out, all_rows, panel_results)
    passed = sum(1 for checks in panel_results.values() if all(ok for _, ok in checks))
    total = len(panel_results)
    print(f"[quintet] summary: {passed}/{total} panels pass  pass_rate={passed/max(1,total):.3f}",
          flush=True)
    print(f"[quintet] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
