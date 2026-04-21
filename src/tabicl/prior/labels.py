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

from dataclasses import dataclass, field
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
    spec = _DEFAULT_MIXTURE if mixture == "default" else _BACKUP_MIXTURE
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
    """Plug-in estimator of Var(E[Y | X_S]) from pre-binned columns."""
    if S.size == 0:
        return 0.0
    keys = col_bins[:, S]
    flat = np.zeros(keys.shape[0], dtype=np.int64)
    for col in range(keys.shape[1]):
        flat = flat * 10 + keys[:, col]
    order = np.argsort(flat)
    sorted_flat = flat[order]
    sorted_y = y[order]
    split = np.where(np.diff(sorted_flat) != 0)[0] + 1
    groups = np.split(sorted_y, split)
    means = np.array([float(np.mean(g)) for g in groups])
    counts = np.array([g.size for g in groups], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    overall = float(np.sum(means * counts) / total)
    return float(np.sum(counts * (means - overall) ** 2) / total)


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

    # Some SCMs (MLPSCM) have an XSampler with fixed per-column state that
    # doesn't respect n_samples; fall back to their native seq_len when
    # overriding would break the mixed-column sampler.
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
        return _empty_value_labels(p)

    y_var = float(np.var(y_base))

    col_bins = np.stack(
        [_quantile_bin(X_base[:, j], n_bins=n_bins) for j in range(p)], axis=1
    )

    queries: List[ValueQuery] = []
    all_feats = np.arange(p, dtype=int)
    for S, query_type in sample_value_queries_meta(p, rng, mixture=mixture):
        V_S = _binned_V(col_bins, y_base, S)
        targets = np.full(p, np.nan, dtype=np.float64)
        raw = np.full(p, np.nan, dtype=np.float64)
        S_set = set(int(j) for j in S.tolist())
        for i in all_feats:
            if int(i) in S_set:
                continue
            Si = np.concatenate([S, np.array([int(i)])])
            V_Si = _binned_V(col_bins, y_base, Si)
            delta = max(0.0, V_Si - V_S)
            raw[i] = delta
            targets[i] = float(np.sqrt(delta))
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
        "y_var_raw": y_var,
        "label_scale": "rms_y_units",
    }


def _empty_value_labels(p: int) -> Dict[str, Any]:
    """Return an empty label payload for degenerate datasets."""
    return {
        "value_queries": [],
        "y_var_raw": float("nan"),
        "label_scale": "rms_y_units",
    }
