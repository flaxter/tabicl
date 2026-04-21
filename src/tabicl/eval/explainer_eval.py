"""End-to-end attribution-quality harness for the predictive value oracle.

A suite is a list of :class:`DatasetSpec` objects. Each spec bundles a
training ``(X, y)`` with a :class:`GroundTruth` mapping ``S -> RMS vector of
sqrt(Delta_{i|S})`` (NaN at positions ``i in S``).  For each dataset we call
an explainer-factory ``factory(X, y) -> Explainer``, query the explainer at
every ``S`` in ``value_by_state``, and compare its predictions to ground
truth.  Per-dataset metrics are packed into a :class:`DatasetScore` and
(optionally) written to CSV.

Metrics scored per dataset:

- ``spearman_value``, ``pearson_value`` — rank and linear correlation between
  predicted and true ``r_{i|S}``, averaged over conditioning states ``S``.
- ``mse_value``, ``mae_value`` — squared / absolute error in RMS units.
- ``top1_next_feature``, ``top3_next_feature`` — recall of the truth's
  best-next-feature in the explainer's top-``k`` ranking at each ``S``.
- ``spearman_sufficiency`` — Spearman on ``predictive_sufficiency_``
  vs ground-truth ``r_{i|emptyset}``.
- ``spearman_necessity`` — Spearman on ``predictive_necessity_`` vs
  ground-truth ``r_{i|[p]\\{i}}``.
- ``acquisition_auc`` — AUFC of cumulative true RMS along the predicted
  greedy path, normalised by the oracle-optimal path (both computed only
  over states present in ``value_by_state``; returns NaN otherwise).

Suite builders:

- :func:`build_in_distribution_suite` — draws datasets from the default
  prior mixture (``mix_scm``) and computes generator-oracle ground truth.
- :func:`build_held_out_prior_suite` — same functional, different prior
  subset, for cross-prior generalisation checks.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
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

from tabicl.eval.metrics import (
    pearson_per_dataset,
    spearman_per_dataset,
    topk_recall_per_dataset,
)


# ---------------------------------------------------------------------------
# Core datatypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    """Generator-oracle ground truth at a set of conditioning states.

    Parameters
    ----------
    value_by_state : Mapping[frozenset[int], np.ndarray]
        ``S -> vector of length p, RMS sqrt(Delta_{i|S})``. NaN at positions
        ``i in S`` (the query is undefined there).
    y_var : float, optional
        Total outcome variance ``Var(Y)`` on the oracle draw. Not used by any
        metric; reported in the CSV for normalisation diagnostics.
    """

    value_by_state: Mapping[FrozenSet[int], np.ndarray]
    y_var: Optional[float] = None


@dataclass(frozen=True)
class DatasetSpec:
    """One evaluation dataset: training context + ground truth."""

    name: str
    X: np.ndarray
    y: np.ndarray
    ground_truth: GroundTruth
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class DatasetScore:
    """Per-dataset metrics for one explainer on one dataset."""

    name: str
    n_features: int
    n_states: int
    # Pooled-over-states label fidelity
    spearman_value: float = float("nan")
    pearson_value: float = float("nan")
    mse_value: float = float("nan")
    mae_value: float = float("nan")
    top1_next_feature: float = float("nan")
    top3_next_feature: float = float("nan")
    # Endpoint-only label fidelity
    spearman_sufficiency: float = float("nan")
    spearman_necessity: float = float("nan")
    # Greedy-path acquisition
    acquisition_auc: float = float("nan")


Explainer = Any  # duck-typed protocol; see evaluate_explainer docstring
Factory = Callable[[np.ndarray, np.ndarray], Explainer]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _mask_to_valid(vec: np.ndarray) -> np.ndarray:
    return np.asarray(vec, dtype=np.float64)


def _pooled_value_metrics(
    preds_by_state: Mapping[FrozenSet[int], np.ndarray],
    ground_truth: GroundTruth,
) -> Tuple[float, float, float, float, float, float]:
    """Aggregate per-state Spearman/Pearson/MAE/MSE/top-k across ``S``."""
    spear = []
    pear = []
    sq = []
    ab = []
    top1 = []
    top3 = []
    for S, truth in ground_truth.value_by_state.items():
        if S not in preds_by_state:
            continue
        pred = _mask_to_valid(preds_by_state[S])
        truth = _mask_to_valid(truth)
        mask = np.isfinite(pred) & np.isfinite(truth)
        if mask.sum() >= 3:
            spear.append(spearman_per_dataset(pred, truth))
            pear.append(pearson_per_dataset(pred, truth))
        if mask.any():
            diff = pred[mask] - truth[mask]
            sq.append(float(np.mean(diff ** 2)))
            ab.append(float(np.mean(np.abs(diff))))
        # Top-k next-feature recall: rank by (signed) RMS on positions outside S.
        # `topk_recall_per_dataset` ranks by absolute value, which matches since
        # RMS is nonnegative.
        if mask.sum() >= 1:
            top1.append(topk_recall_per_dataset(pred, truth, k=1))
            if mask.sum() >= 3:
                top3.append(topk_recall_per_dataset(pred, truth, k=3))

    def _nanmean(xs: List[float]) -> float:
        arr = np.asarray(xs, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).any():
            return float("nan")
        return float(np.nanmean(arr))

    return (
        _nanmean(spear),
        _nanmean(pear),
        _nanmean(sq),
        _nanmean(ab),
        _nanmean(top1),
        _nanmean(top3),
    )


def _endpoint_metrics(
    explainer: Explainer,
    ground_truth: GroundTruth,
    p: int,
) -> Tuple[float, float]:
    """Spearman for predictive_sufficiency_ / predictive_necessity_ attrs."""
    suff = float("nan")
    empty = frozenset()
    if empty in ground_truth.value_by_state and hasattr(
        explainer, "predictive_sufficiency_"
    ):
        pred_suff = np.asarray(explainer.predictive_sufficiency_, dtype=np.float64)
        truth_suff = np.asarray(ground_truth.value_by_state[empty], dtype=np.float64)
        suff = spearman_per_dataset(pred_suff, truth_suff)

    nec = float("nan")
    loo_states = [frozenset(j for j in range(p) if j != i) for i in range(p)]
    all_present = all(s in ground_truth.value_by_state for s in loo_states)
    if all_present and hasattr(explainer, "predictive_necessity_"):
        pred_nec = np.asarray(explainer.predictive_necessity_, dtype=np.float64)
        truth_nec = np.full(p, np.nan, dtype=np.float64)
        for i, s in enumerate(loo_states):
            truth_nec[i] = ground_truth.value_by_state[s][i]
        nec = spearman_per_dataset(pred_nec, truth_nec)

    return suff, nec


def _acquisition_auc(
    explainer: Explainer,
    ground_truth: GroundTruth,
) -> float:
    """Normalised AUFC of cumulative true RMS along the predicted greedy path.

    Scoring path: ``S_0 = emptyset``; at step ``t`` the explainer selects
    ``i_t`` via ``greedy_predictive_path``; we credit the gain by ground
    truth's ``r_{i_t|S_t}``. Normalised by the oracle-optimal path that
    always selects ``argmax_i truth(S_t)[i]``. Returns NaN if any visited
    state is missing from ``value_by_state``.
    """
    if not hasattr(explainer, "greedy_predictive_path"):
        return float("nan")
    try:
        pred_path, _ = explainer.greedy_predictive_path()
    except Exception:
        return float("nan")
    if len(pred_path) == 0:
        return float("nan")

    vbs = ground_truth.value_by_state

    def _cumgain(path: Sequence[int]) -> Optional[float]:
        total = 0.0
        S: set = set()
        for step, i in enumerate(path):
            key = frozenset(S)
            if key not in vbs:
                return None
            v = vbs[key][int(i)]
            if not np.isfinite(v):
                return None
            total += float(v)
            S.add(int(i))
        return total

    pred_gain = _cumgain(pred_path)
    if pred_gain is None:
        return float("nan")

    # Oracle path: at each state, pick argmax of ground truth.
    oracle_path: List[int] = []
    S: set = set()
    while True:
        key = frozenset(S)
        if key not in vbs:
            return float("nan")
        row = np.asarray(vbs[key], dtype=np.float64).copy()
        for j in S:
            row[j] = -np.inf
        valid = np.isfinite(row)
        if not valid.any():
            break
        i_best = int(np.argmax(row))
        oracle_path.append(i_best)
        S.add(i_best)
        if len(oracle_path) >= len(pred_path):
            break

    oracle_gain = _cumgain(oracle_path)
    if oracle_gain is None or oracle_gain <= 0.0:
        return float("nan")
    return float(pred_gain / oracle_gain)


def _predict_by_state(
    explainer: Explainer,
    states: Iterable[FrozenSet[int]],
) -> Dict[FrozenSet[int], np.ndarray]:
    """Query ``conditional_predictive_values`` at every requested state."""
    out: Dict[FrozenSet[int], np.ndarray] = {}
    for S in states:
        out[S] = np.asarray(
            explainer.conditional_predictive_values(sorted(S)),
            dtype=np.float64,
        )
    return out


def score_one(
    explainer: Explainer,
    dataset: DatasetSpec,
) -> DatasetScore:
    """Compute every :class:`DatasetScore` metric for one (explainer, dataset)."""
    p = int(dataset.X.shape[1])
    preds = _predict_by_state(explainer, dataset.ground_truth.value_by_state.keys())
    spear, pear, mse, mae, t1, t3 = _pooled_value_metrics(preds, dataset.ground_truth)
    suff, nec = _endpoint_metrics(explainer, dataset.ground_truth, p)
    auc = _acquisition_auc(explainer, dataset.ground_truth)

    return DatasetScore(
        name=dataset.name,
        n_features=p,
        n_states=len(dataset.ground_truth.value_by_state),
        spearman_value=spear,
        pearson_value=pear,
        mse_value=mse,
        mae_value=mae,
        top1_next_feature=t1,
        top3_next_feature=t3,
        spearman_sufficiency=suff,
        spearman_necessity=nec,
        acquisition_auc=auc,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def evaluate_explainer(
    factory: Factory,
    suite: Sequence[DatasetSpec],
    *,
    out_csv: Optional[str | Path] = None,
) -> List[DatasetScore]:
    """Fit ``factory(X, y)`` per dataset, score metrics, optionally write CSV.

    Parameters
    ----------
    factory : callable
        ``(X, y) -> explainer`` where explainer supports
        ``conditional_predictive_values(S)`` and optionally
        ``predictive_sufficiency_``, ``predictive_necessity_``,
        ``greedy_predictive_path()``.
    suite : sequence of DatasetSpec
    out_csv : path-like, optional
        If set, write all DatasetScore rows to this CSV (creating parent
        directories).

    Returns
    -------
    list[DatasetScore]
    """
    scores: List[DatasetScore] = []
    for spec in suite:
        expl = factory(spec.X, spec.y)
        scores.append(score_one(expl, spec))

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(asdict(scores[0]).keys()) if scores else [
            f.name for f in DatasetScore.__dataclass_fields__.values()  # type: ignore[attr-defined]
        ]
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for s in scores:
                w.writerow(asdict(s))

    return scores


# ---------------------------------------------------------------------------
# Ground-truth computation (simulator-oracle plug-in via quantile-binning)
# ---------------------------------------------------------------------------


def _quantile_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.searchsorted(edges, x) - 1, 0, n_bins - 1)


def _binned_V(col_bins: np.ndarray, y: np.ndarray, S: Sequence[int], n_bins: int) -> float:
    S = np.asarray(list(S), dtype=int)
    if S.size == 0:
        return 0.0
    keys = col_bins[:, S]
    flat = np.zeros(keys.shape[0], dtype=np.int64)
    for col in range(keys.shape[1]):
        flat = flat * (n_bins + 1) + keys[:, col]
    order = np.argsort(flat)
    sf = flat[order]
    sy = y[order]
    split = np.where(np.diff(sf) != 0)[0] + 1
    groups = np.split(sy, split)
    means = np.array([float(np.mean(g)) for g in groups])
    counts = np.array([g.size for g in groups], dtype=np.float64)
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    overall = float(np.sum(means * counts) / total)
    return float(np.sum(counts * (means - overall) ** 2) / total)


def _ground_truth_from_oracle_draw(
    X_oracle: np.ndarray,
    y_oracle: np.ndarray,
    states: Sequence[FrozenSet[int]],
    *,
    n_bins: int = 10,
) -> GroundTruth:
    """Build a GroundTruth by quantile-binning a fresh (X, y) oracle draw.

    Matches the plug-in estimator used by :func:`tabicl.prior.labels.compute_value_queries`
    so the harness ground truth is consistent with the training labels.
    """
    p = int(X_oracle.shape[1])
    col_bins = np.stack(
        [_quantile_bin(X_oracle[:, j], n_bins=n_bins) for j in range(p)],
        axis=1,
    )
    by_state: Dict[FrozenSet[int], np.ndarray] = {}
    for S in states:
        V_S = _binned_V(col_bins, y_oracle, list(S), n_bins=n_bins)
        row = np.full(p, np.nan, dtype=np.float64)
        S_set = set(int(j) for j in S)
        for i in range(p):
            if i in S_set:
                continue
            V_Si = _binned_V(col_bins, y_oracle, list(S) + [int(i)], n_bins=n_bins)
            row[i] = float(np.sqrt(max(0.0, V_Si - V_S)))
        by_state[frozenset(S)] = row
    return GroundTruth(value_by_state=by_state, y_var=float(np.var(y_oracle)))


# ---------------------------------------------------------------------------
# State sampler for suites (matches preregistration §11.1 strata)
# ---------------------------------------------------------------------------


def canonical_states(
    p: int,
    *,
    rng: np.random.Generator,
    include_empty: bool = True,
    include_loo: bool = True,
    n_random_small: int = 3,
    n_random_medium: int = 2,
    n_random_near_full: int = 2,
) -> List[FrozenSet[int]]:
    """Return a stratified set of ``S`` states used by the in-distribution suite.

    Covers every |S|-stratum named in preregistration §11.1:
    ``|S|=0``, singleton, small, medium (~p/2), and near-full (p-2, p-1).
    Leave-one-out states (|S|=p-1, specific i) are included so
    ``predictive_necessity_`` is scorable.
    """
    states: List[FrozenSet[int]] = []
    if include_empty:
        states.append(frozenset())
    # Singleton (preregistration's "one" bucket)
    for _ in range(2):
        states.append(frozenset([int(rng.integers(0, p))]))
    # Small |S|
    for _ in range(n_random_small):
        k = int(rng.integers(2, min(5, p)))
        sel = rng.choice(p, size=min(k, p), replace=False).astype(int)
        states.append(frozenset(int(j) for j in sel))
    # Medium |S|
    med = max(1, p // 2)
    for _ in range(n_random_medium):
        sel = rng.choice(p, size=min(med, p), replace=False).astype(int)
        states.append(frozenset(int(j) for j in sel))
    # Near-full (p-2, p-1) via random choice of excluded features
    for _ in range(n_random_near_full):
        excl = 1 + int(rng.integers(0, 2))  # 1 or 2 excluded features
        excl = min(excl, p)
        exclude = rng.choice(p, size=excl, replace=False).astype(int)
        sel = np.setdiff1d(np.arange(p), exclude)
        states.append(frozenset(int(j) for j in sel))
    if include_loo:
        for i in range(p):
            states.append(frozenset(j for j in range(p) if j != i))
    # Deduplicate (different strata can collide on small p)
    seen: Dict[FrozenSet[int], None] = {}
    for s in states:
        seen[s] = None
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Suite builders
# ---------------------------------------------------------------------------


def _scm_factory(prior_type: str, *, num_features: int, seq_len: int, seed: int):
    """Instantiate one SCM with modest hyperparameters for evaluation use."""
    if prior_type == "mlp_scm":
        from tabicl.prior.mlp_scm import MLPSCM

        return MLPSCM(
            seq_len=seq_len,
            num_features=num_features,
            num_outputs=1,
            is_causal=True,
            y_is_effect=True,
            num_causes=max(1, num_features // 2),
            num_layers=3,
            hidden_dim=max(16, 4 * num_features),
            in_clique=False,
            sort_features=True,
            noise_std=0.1,
            device="cpu",
        )
    if prior_type == "tree_scm":
        from tabicl.prior.tree_scm import TreeSCM

        return TreeSCM(
            seq_len=seq_len,
            num_features=num_features,
            num_outputs=1,
            is_causal=True,
            y_is_effect=True,
            num_causes=max(1, num_features // 2),
            tree_model="xgboost",
            tree_n_estimators=32,
            tree_max_depth=4,
            noise_std=0.1,
            device="cpu",
        )
    raise ValueError(f"Unknown prior_type {prior_type!r}")


def _draw_dataset(
    prior_type: str,
    *,
    num_features: int,
    seq_len: int,
    seed: int,
    n_oracle: int,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, GroundTruth]:
    """Draw one dataset from an SCM and build its ground truth."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    scm = _scm_factory(prior_type, num_features=num_features, seq_len=seq_len, seed=seed)
    X_raw, y_raw = scm()
    X = X_raw.detach().cpu().numpy().astype(np.float64)
    y = y_raw.detach().cpu().numpy().astype(np.float64).reshape(-1)

    try:
        X_o, y_o = scm.simulate(n_samples=n_oracle, rng=rng)
    except Exception:
        X_o, y_o = scm.simulate(rng=rng)
    if hasattr(X_o, "cpu"):
        X_o = X_o.cpu().numpy()
    if hasattr(y_o, "cpu"):
        y_o = y_o.cpu().numpy()
    X_o = np.asarray(X_o, dtype=np.float64)
    y_o = np.asarray(y_o, dtype=np.float64).reshape(-1)

    p = int(X.shape[1])
    states = canonical_states(p, rng=rng)
    gt = _ground_truth_from_oracle_draw(X_o, y_o, states, n_bins=n_bins)
    return X, y, gt


def build_in_distribution_suite(
    n_datasets: int = 16,
    *,
    num_features: int = 8,
    seq_len: int = 500,
    seed: int = 0,
    n_oracle: int = 2000,
    n_bins: int = 10,
    prior_mix: Sequence[str] = ("mlp_scm", "tree_scm"),
) -> List[DatasetSpec]:
    """Evaluation suite drawn from the same prior mixture as training.

    Each dataset gets a stratified set of conditioning states covering every
    |S|-stratum named in preregistration §11.1.
    """
    rng = np.random.default_rng(seed)
    suite: List[DatasetSpec] = []
    for idx in range(n_datasets):
        prior_type = str(rng.choice(list(prior_mix)))
        X, y, gt = _draw_dataset(
            prior_type,
            num_features=num_features,
            seq_len=seq_len,
            seed=seed + idx + 1,
            n_oracle=n_oracle,
            n_bins=n_bins,
        )
        suite.append(
            DatasetSpec(
                name=f"{prior_type}_{idx:03d}",
                X=X,
                y=y,
                ground_truth=gt,
                meta={"prior_type": prior_type, "seed": seed + idx + 1},
            )
        )
    return suite


def build_held_out_prior_suite(
    n_datasets: int = 16,
    *,
    num_features: int = 8,
    seq_len: int = 500,
    seed: int = 1000,
    n_oracle: int = 2000,
    n_bins: int = 10,
) -> List[DatasetSpec]:
    """Cross-prior generalisation suite.

    Uses the prior not represented in the default training mixture heavy tail
    (``tree_scm`` alone) as a held-out check. Callers can pass a different
    mixture via :func:`build_in_distribution_suite` directly for more nuanced
    splits.
    """
    return build_in_distribution_suite(
        n_datasets,
        num_features=num_features,
        seq_len=seq_len,
        seed=seed,
        n_oracle=n_oracle,
        n_bins=n_bins,
        prior_mix=("tree_scm",),
    )


__all__ = [
    "GroundTruth",
    "DatasetSpec",
    "DatasetScore",
    "Factory",
    "Explainer",
    "canonical_states",
    "score_one",
    "evaluate_explainer",
    "build_in_distribution_suite",
    "build_held_out_prior_suite",
]
