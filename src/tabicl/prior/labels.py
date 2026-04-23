"""Conditional predictive value labels.

The central label is

    Delta_{i|S} = V(S union {i}) - V(S),   V(S) = Var(E[Y | X_S]),

the reduction in Bayes squared-error risk from revealing X_i once X_S is
known. Labels are computed from the generator/oracle, not from the finite
PFN context. The neural target is RMS-scale:

    r_{i|S} = sqrt(max(Delta_{i|S}, 0))        (units of Y).

Two label paths:

- ``V_gaussian(Sigma, y_idx, S)`` — exact closed form under joint
  Gaussianity. Used by unit-test fixtures and (optionally) a dedicated
  linear-Gaussian subprior. Exactness requires the joint (X, Y) to be
  Gaussian; it is **not** the Bayes value for non-Gaussian linear SCMs —
  only the best-linear-projection value. Use the simulator-oracle path for
  anything not jointly Gaussian.

- ``compute_value_queries(scm, X, y, params, rng)`` — simulator-oracle
  plug-in estimator of V(S) via quantile-binning the conditional mean. Used
  for the primary training priors (MLPSCM, TreeSCM).

Per-dataset label mixture (locked in preregistration §7.2) — 10 conditioning
states sampled per call:

    | bucket      | count | |S|               |
    |-------------|-------|-------------------|
    | empty       | 1     | 0                 |
    | singleton   | 2     | 1                 |
    | small       | 3     | uniform{2,3,4}    |
    | medium      | 2     | floor(p/2)        |
    | near_full   | 2     | uniform{p-2,p-1}  |

For each sampled S, targets are computed for all i in [p]; NaN for i in S.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Public datatypes
# ---------------------------------------------------------------------------


@dataclass
class ValueQuery:
    """One conditioning state S and its RMS predictive value targets.

    Attributes
    ----------
    S_mask : BoolTensor, shape (p,)
        True for features in the conditioning set S.
    targets : Tensor, shape (p,)
        RMS targets sqrt(max(Delta_{i|S}, 0)). NaN for positions i in S.
    raw_targets : Tensor, shape (p,) or None
        Variance-scale Delta_{i|S}. NaN for positions i in S. Optional.
    query_type : str
        One of {"empty", "singleton", "small", "medium", "near_full"}. Used
        for stratified loss diagnostics only.
    """

    S_mask: Tensor
    targets: Tensor
    raw_targets: Optional[Tensor] = None
    query_type: str = "random"


@dataclass(frozen=True)
class OracleContext:
    """One simulator-oracle draw prepared for repeated predictive-value queries."""

    X: np.ndarray
    y: np.ndarray
    col_bins: np.ndarray
    y_var: float


# ---------------------------------------------------------------------------
# Gaussian closed-form helpers (exact; test fixtures + future subprior)
# ---------------------------------------------------------------------------


def V_gaussian(Sigma: np.ndarray, y_idx: int, S: np.ndarray) -> float:
    """Return Var(E[Y | X_S]) for jointly Gaussian (X, Y).

    Uses a linear solve, not an explicit inverse. Ridge stabilisation is
    applied for near-singular Sigma_SS.

    Parameters
    ----------
    Sigma : ndarray, shape (p+1, p+1)
        Joint covariance of (X, Y) with Y at position ``y_idx``.
    y_idx : int
        Index of Y in the covariance matrix.
    S : ndarray of int, shape (|S|,)
        Feature indices.
    """
    if S.size == 0:
        return 0.0
    Sigma_SS = Sigma[np.ix_(S, S)]
    Sigma_yS = Sigma[y_idx, S]
    sol = np.linalg.solve(Sigma_SS + 1e-10 * np.eye(S.size), Sigma_yS)
    return float(Sigma_yS @ sol)


def delta_gaussian(Sigma: np.ndarray, y_idx: int, i: int, S: np.ndarray) -> float:
    """Return Delta_{i|S} = V(S union {i}) - V(S) for jointly Gaussian (X, Y).

    Returns NaN if ``i in S``; nonnegative otherwise (numerical floor at 0).
    """
    if int(i) in set(S.tolist()):
        return float("nan")
    S_plus = np.concatenate([S, np.array([int(i)])])
    gain = V_gaussian(Sigma, y_idx, S_plus) - V_gaussian(Sigma, y_idx, S)
    return float(max(0.0, gain))


# ---------------------------------------------------------------------------
# Conditioning-state sampler (locked mixture)
# ---------------------------------------------------------------------------


_DEFAULT_MIXTURE: Tuple[Tuple[str, int, str], ...] = (
    ("empty", 1, "zero"),
    ("singleton", 2, "one"),
    ("small", 3, "small"),
    ("medium", 2, "medium"),
    ("near_full", 2, "near_full"),
)
# Backup (thinned) mixture if label-cost smoketest forces it — 6 states.
_BACKUP_MIXTURE: Tuple[Tuple[str, int, str], ...] = (
    ("empty", 1, "zero"),
    ("singleton", 1, "one"),
    ("small", 2, "small"),
    ("medium", 1, "medium"),
    ("near_full", 1, "near_full"),
)
# Easy-strata mixture (REMEDY Phase 3): train only where labels are trustworthy.
# Medium and near_full are stripped because the cross-fitted direct-Delta
# estimator still has materially more variance on wide S; this isolates
# whether the method works where supervision is cleanest.
_EASY_MIXTURE: Tuple[Tuple[str, int, str], ...] = (
    ("empty", 1, "zero"),
    ("singleton", 2, "one"),
    ("small", 3, "small"),
)


def _draw_S(p: int, size_spec: str, rng: np.random.Generator) -> np.ndarray:
    """Sample a feature subset of the specified size-class from [p]."""
    if p <= 0:
        return np.zeros(0, dtype=int)
    if size_spec == "zero":
        size = 0
    elif size_spec == "one":
        size = 1
    elif size_spec == "small":
        # Uniform in {2, 3, 4}, truncated to feasible.
        size = int(rng.integers(2, min(5, max(3, p))))
    elif size_spec == "medium":
        size = p // 2
    elif size_spec == "near_full":
        # Uniform in {p-2, p-1}, truncated to non-negative.
        choices = [v for v in (p - 2, p - 1) if v >= 0]
        size = int(rng.choice(choices))
    else:
        raise ValueError(f"Unknown size_spec {size_spec!r}")
    size = max(0, min(size, p))
    if size == 0:
        return np.zeros(0, dtype=int)
    return rng.choice(p, size=size, replace=False).astype(int)


def sample_value_queries_meta(
    p: int,
    rng: np.random.Generator,
    mixture: str = "default",
) -> List[Tuple[np.ndarray, str]]:
    """Return a list of ``(S, query_type)`` pairs sampled from the mixture."""
    if mixture == "default":
        spec = _DEFAULT_MIXTURE
    elif mixture == "backup":
        spec = _BACKUP_MIXTURE
    elif mixture == "easy":
        spec = _EASY_MIXTURE
    else:
        raise ValueError(
            f"mixture must be 'default', 'backup', or 'easy'; got {mixture!r}"
        )
    out: List[Tuple[np.ndarray, str]] = []
    for query_type, count, size_spec in spec:
        for _ in range(count):
            S = _draw_S(p, size_spec, rng)
            out.append((S, query_type))
    return out


# ---------------------------------------------------------------------------
# Simulator-oracle V(S) via quantile-binning plug-in
# ---------------------------------------------------------------------------


def _quantile_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.searchsorted(edges, x) - 1, 0, n_bins - 1)


def _binned_V(col_bins: np.ndarray, y: np.ndarray, S: np.ndarray) -> float:
    """Plug-in estimator of Var(E[Y | X_S]) from pre-binned columns.

    Non-finite ``y`` rows are dropped before grouping. Group identities are
    formed with ``np.unique(..., axis=0)`` instead of an int64 flattening
    trick so high-dimensional states cannot overflow the key space.
    """
    if S.size == 0:
        return 0.0
    keys = np.asarray(col_bins[:, S])
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y)
    if not finite.any():
        return 0.0
    keys = keys[finite]
    y = y[finite]
    if y.size == 0:
        return 0.0

    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    counts = counts.astype(np.float64, copy=False)
    sums = np.bincount(inverse, weights=y, minlength=counts.size)
    means = sums / counts
    total = counts.sum()
    if total <= 0:
        return 0.0
    overall = float(np.sum(means * counts) / total)
    value = float(np.sum(counts * (means - overall) ** 2) / total)
    return max(0.0, value)


def build_oracle_context(
    scm: Any,
    p: int,
    *,
    n_oracle: int = 512,
    rng: Optional[np.random.Generator] = None,
    n_bins: int = 10,
) -> OracleContext:
    """Simulate and pre-bin one oracle sample for repeated queries over S."""
    rng = rng if rng is not None else np.random.default_rng()

    try:
        X_base, y_base = scm.simulate(n_samples=n_oracle, rng=rng)
    except RuntimeError:
        X_base, y_base = scm.simulate(rng=rng)

    if isinstance(X_base, torch.Tensor):
        X_base = X_base.cpu().numpy()
    if isinstance(y_base, torch.Tensor):
        y_base = y_base.cpu().numpy()

    X_base = np.asarray(X_base, dtype=np.float64)
    y_base = np.asarray(y_base, dtype=np.float64).reshape(-1)
    if X_base.ndim != 2 or X_base.shape[1] != p:
        raise ValueError(f"oracle sample must have shape (n, {p}); got {X_base.shape}")
    finite_y = y_base[np.isfinite(y_base)]
    if finite_y.size < 2:
        raise ValueError("oracle sample has fewer than 2 finite y values")

    col_bins = np.stack(
        [_quantile_bin(X_base[:, j], n_bins=n_bins) for j in range(p)], axis=1
    )
    return OracleContext(
        X=X_base,
        y=y_base,
        col_bins=col_bins,
        y_var=float(np.var(finite_y)),
    )


def V_of_subset(context: OracleContext, S: np.ndarray) -> float:
    """Plug-in estimate of ``V(S)`` from a prepared oracle draw."""
    return _binned_V(context.col_bins, context.y, np.asarray(S, dtype=int))


def delta_vector_for_S(context: OracleContext, S: np.ndarray) -> np.ndarray:
    """Vector of plug-in ``Delta_{i|S}`` estimates with NaN at ``i in S``."""
    S = np.asarray(S, dtype=int)
    p = int(context.X.shape[1])
    V_S = V_of_subset(context, S)
    out = np.full(p, np.nan, dtype=np.float64)
    S_set = set(int(j) for j in S.tolist())
    for i in range(p):
        if i in S_set:
            continue
        Si = np.concatenate([S, np.array([i], dtype=int)])
        out[i] = max(0.0, V_of_subset(context, Si) - V_S)
    return out


def delta_value(context: OracleContext, i: int, S: np.ndarray) -> float:
    """Single plug-in ``Delta_{i|S}`` entry from a prepared oracle draw."""
    return float(delta_vector_for_S(context, S)[int(i)])


# ---------------------------------------------------------------------------
# Cross-fitted direct-Delta estimator (REMEDY.md; replaces histogram plug-in)
# ---------------------------------------------------------------------------


def _standardize_by_train(
    Xtr: np.ndarray, Xte: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _knn_fit_predict(k: Optional[int] = None):
    """Build a ``fit_predict`` closure for :func:`_direct_delta_cf` (kNN)."""
    def fp(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
        from sklearn.neighbors import KNeighborsRegressor
        n_tr = Xtr.shape[0]
        kk = k if k is not None else max(1, int(np.ceil(np.sqrt(n_tr))))
        kk = min(kk, n_tr)
        model = KNeighborsRegressor(n_neighbors=kk)
        model.fit(Xtr, ytr)
        return model.predict(Xte)
    return fp


def _ridge_fit_predict(alpha: float = 1.0):
    """Build a ``fit_predict`` closure for :func:`_direct_delta_cf` (Ridge)."""
    def fp(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha)
        model.fit(Xtr, ytr)
        return model.predict(Xte)
    return fp


def _kernel_fit_predict(alpha: float = 1e-3, gamma: Optional[float] = None):
    """Build a ``fit_predict`` closure (KernelRidge, RBF, median-heuristic).

    If ``gamma`` is None, the bandwidth is set per fit from the median
    pairwise distance on a random subsample of ``Xtr`` (cap 256 rows for
    O(1) bandwidth cost, matches the smoketest kernel estimator).
    """
    def fp(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.metrics.pairwise import pairwise_distances
        g = gamma
        if g is None:
            n_tr = Xtr.shape[0]
            sub_n = min(256, n_tr)
            if sub_n < 2:
                g = 1.0
            else:
                sub = Xtr[:sub_n]
                D = pairwise_distances(sub)
                triu = D[np.triu_indices_from(D, k=1)]
                triu = triu[triu > 0]
                h = float(np.median(triu)) if triu.size else 1.0
                g = 1.0 / (2.0 * max(h * h, 1e-12))
        model = KernelRidge(alpha=alpha, kernel="rbf", gamma=g)
        model.fit(Xtr, ytr)
        return model.predict(Xte)
    return fp


def _direct_delta_cf(
    X: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    p: int,
    *,
    fit_predict,
    n_folds: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Cross-fitted direct estimator of Delta_{i|S} = E[(mu_{S+i} - mu_S)^2].

    Generic over the nuisance regressor family: ``fit_predict`` is a
    callable ``(Xtr_std, ytr, Xte_std) -> pred`` that handles one train/
    test fold. Train-fold standardization of active coordinates is applied
    upstream. For |S| = 0, mu_S uses the train-fold y-mean directly (no
    regressor fit). Nonnegative by construction; NaN for ``i in S``.

    Non-finite y rows (or rows with any non-finite X) are dropped before
    fold assignment. If fewer than ``max(2, n_folds)`` rows remain, returns
    0.0 for every valid candidate.
    """
    S = np.asarray(S, dtype=int)
    S_set = set(int(j) for j in S.tolist())
    rng = rng if rng is not None else np.random.default_rng()

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite]
    y = y[finite]
    n = X.shape[0]

    out = np.full(p, np.nan, dtype=np.float64)
    if n < max(2, n_folds):
        for i in range(p):
            if i not in S_set:
                out[i] = 0.0
        return out

    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=np.int64)
    for f, idx in enumerate(np.array_split(perm, n_folds)):
        fold_ids[idx] = f

    mu_S_pred = np.empty(n, dtype=np.float64)
    mu_Si_pred = np.full((n, p), np.nan, dtype=np.float64)

    for f in range(n_folds):
        te = fold_ids == f
        tr = ~te
        n_tr = int(tr.sum())
        if n_tr < 1:
            mu_S_pred[te] = 0.0
            continue
        Xtr, ytr = X[tr], y[tr]
        Xte = X[te]

        if S.size == 0:
            mu_S_pred[te] = float(ytr.mean())
        else:
            Xtr_S, Xte_S = _standardize_by_train(Xtr[:, S], Xte[:, S])
            mu_S_pred[te] = fit_predict(Xtr_S, ytr, Xte_S)

        for i in range(p):
            if i in S_set:
                continue
            feats = np.concatenate([S, np.array([i], dtype=int)])
            Xtr_Si, Xte_Si = _standardize_by_train(Xtr[:, feats], Xte[:, feats])
            mu_Si_pred[te, i] = fit_predict(Xtr_Si, ytr, Xte_Si)

    for i in range(p):
        if i in S_set:
            continue
        diff = mu_Si_pred[:, i] - mu_S_pred
        out[i] = float(np.mean(diff * diff))
    return out


def _direct_delta_cf_knn(
    X: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    p: int,
    *,
    n_folds: int = 5,
    k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Cross-fitted kNN direct estimator of Delta_{i|S} (REMEDY v1).

    Thin wrapper on :func:`_direct_delta_cf` with the kNN nuisance. Default
    ``k = ceil(sqrt(n_train))``.
    """
    return _direct_delta_cf(
        X, y, S, p,
        fit_predict=_knn_fit_predict(k),
        n_folds=n_folds, rng=rng,
    )


def _direct_delta_cf_ridge(
    X: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    p: int,
    *,
    n_folds: int = 5,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Cross-fitted ridge-regression direct estimator of Delta_{i|S}.

    Linear nuisance; fastest of the three variants, lower bias when the
    conditional mean is close to linear in the active coordinates, higher
    bias otherwise. Standard alpha=1.0 on standardized features.
    """
    return _direct_delta_cf(
        X, y, S, p,
        fit_predict=_ridge_fit_predict(alpha),
        n_folds=n_folds, rng=rng,
    )


def _direct_delta_cf_kernel(
    X: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    p: int,
    *,
    n_folds: int = 5,
    alpha: float = 1e-3,
    gamma: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Cross-fitted RBF kernel-ridge direct estimator of Delta_{i|S}.

    Nonparametric nuisance with median-heuristic bandwidth (if
    ``gamma=None``). Higher bias control than ridge, higher variance than
    kNN for wide S.
    """
    return _direct_delta_cf(
        X, y, S, p,
        fit_predict=_kernel_fit_predict(alpha, gamma),
        n_folds=n_folds, rng=rng,
    )


def delta_vector_for_S_direct_knn(
    context: OracleContext,
    S: np.ndarray,
    *,
    n_folds: int = 5,
    k: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Direct-Delta counterpart to :func:`delta_vector_for_S` (kNN nuisance)."""
    p = int(context.X.shape[1])
    return _direct_delta_cf_knn(
        context.X, context.y, np.asarray(S, dtype=int), p,
        n_folds=n_folds, k=k, rng=rng,
    )


def delta_vector_for_S_direct_ridge(
    context: OracleContext,
    S: np.ndarray,
    *,
    n_folds: int = 5,
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Direct-Delta counterpart with ridge-regression nuisance."""
    p = int(context.X.shape[1])
    return _direct_delta_cf_ridge(
        context.X, context.y, np.asarray(S, dtype=int), p,
        n_folds=n_folds, alpha=alpha, rng=rng,
    )


def delta_vector_for_S_direct_kernel(
    context: OracleContext,
    S: np.ndarray,
    *,
    n_folds: int = 5,
    alpha: float = 1e-3,
    gamma: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Direct-Delta counterpart with RBF kernel-ridge nuisance."""
    p = int(context.X.shape[1])
    return _direct_delta_cf_kernel(
        context.X, context.y, np.asarray(S, dtype=int), p,
        n_folds=n_folds, alpha=alpha, gamma=gamma, rng=rng,
    )


# ---------------------------------------------------------------------------
# Top-level label entry point
# ---------------------------------------------------------------------------


def compute_value_queries(
    scm: Any,
    X: Tensor,
    y: Tensor,
    params: Optional[Dict[str, Any]] = None,
    n_oracle: int = 512,
    rng: Optional[np.random.Generator] = None,
    mode: Optional[str] = None,
    mixture: str = "default",
    n_bins: int = 10,
    label_estimator: str = "histogram",
    label_knn_folds: int = 5,
    label_knn_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute predictive value query labels for one pre-``Reg2Cls`` dataset.

    Simulator-oracle plug-in estimator of V(S) via quantile-binning: one
    fresh-noise MC draw of (X, y) at size ``n_oracle`` is taken from the
    SCM; each feature column is quantile-binned once; V(S) is the
    weighted variance of the per-bin mean of y over the bin-tuple keyed by
    ``X[:, S]``.

    Parameters
    ----------
    scm : object
        SCM instance exposing ``simulate(n_samples, rng)``.
    X, y : Tensor
        The original training (X, y) from this dataset. Used only for
        feature-count; the oracle draws a fresh sample.
    params : dict, optional
        Generation parameters (unused here; reserved for subprior
        specialisation).
    n_oracle : int, default=512
        MC sample count for the oracle draw.
    rng : np.random.Generator, optional
    mode : {"nan"}, optional
        If ``"nan"``, return a single empty NaN query (fallback for priors
        with no usable ``simulate``).
    mixture : {"default", "backup"}
        Conditioning-state mixture (§7.2 locked default, thinned backup).
    n_bins : int
        Quantile bins per feature for the plug-in estimator.
    label_estimator : {"histogram", "direct_knn", "direct_ridge", "direct_kernel"}
        Which Delta_{i|S} estimator to use. ``"histogram"`` is the legacy
        quantile-binning plug-in of ``V(S) = Var(E[Y|X_S])`` and differences
        (§7.3). The ``"direct_*"`` variants all estimate the paper's direct
        identity E[(mu_{S+i}-mu_S)^2] with cross-fitted nuisance regressors
        (REMEDY.md); nonnegative by construction, no max(.,0) clip.
    label_knn_folds : int
        Cross-fitting folds for the direct-Delta variants.
    label_knn_k : int, optional
        Neighbor count for ``label_estimator="direct_knn"``. Default
        ``ceil(sqrt(n_train))``.

    Returns
    -------
    dict with keys:
      ``value_queries`` : list[ValueQuery]
      ``y_var_raw``     : float, total outcome variance (oracle draw)
      ``label_scale``   : str, ``"rms_y_units"``
    """
    rng = rng if rng is not None else np.random.default_rng()
    p = int(X.shape[-1])

    if p == 0 or mode == "nan" or not hasattr(scm, "simulate"):
        return _empty_value_labels(p)

    try:
        context = build_oracle_context(
            scm,
            p,
            n_oracle=n_oracle,
            rng=rng,
            n_bins=n_bins,
        )
    except ValueError:
        return _empty_value_labels(p)

    valid_estimators = ("histogram", "direct_knn", "direct_ridge", "direct_kernel")
    if label_estimator not in valid_estimators:
        raise ValueError(
            f"label_estimator must be one of {valid_estimators}; got {label_estimator!r}"
        )

    queries: List[ValueQuery] = []
    for S, query_type in sample_value_queries_meta(p, rng, mixture=mixture):
        if label_estimator == "histogram":
            raw = delta_vector_for_S(context, S)
        elif label_estimator == "direct_knn":
            raw = delta_vector_for_S_direct_knn(
                context, S,
                n_folds=label_knn_folds,
                k=label_knn_k,
                rng=rng,
            )
        elif label_estimator == "direct_ridge":
            raw = delta_vector_for_S_direct_ridge(
                context, S,
                n_folds=label_knn_folds,
                rng=rng,
            )
        else:  # direct_kernel
            raw = delta_vector_for_S_direct_kernel(
                context, S,
                n_folds=label_knn_folds,
                rng=rng,
            )
        targets = np.sqrt(np.clip(raw, a_min=0.0, a_max=None))
        S_mask = torch.zeros(p, dtype=torch.bool)
        if S.size > 0:
            S_mask[torch.as_tensor(S, dtype=torch.long)] = True
        queries.append(
            ValueQuery(
                S_mask=S_mask,
                targets=torch.as_tensor(targets, dtype=torch.float),
                raw_targets=torch.as_tensor(raw, dtype=torch.float),
                query_type=query_type,
            )
        )

    return {
        "value_queries": queries,
        "y_var_raw": context.y_var,
        "label_scale": "rms_y_units",
    }


def _empty_value_labels(p: int) -> Dict[str, Any]:
    """Return an empty label payload for degenerate datasets."""
    return {
        "value_queries": [],
        "y_var_raw": float("nan"),
        "label_scale": "rms_y_units",
    }
