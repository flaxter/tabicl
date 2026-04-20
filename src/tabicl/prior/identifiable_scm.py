"""Phase 3 — identifiable structural-causal-model families.

Three SCM families whose structural equations admit closed-form attribution
labels (Heads A, I, C) via covariance / basis algebra:

- ``LiNGAMSCM``        — linear non-Gaussian (Laplace) SCM; Head A/I/C all
  closed form via the (p+1)x(p+1) causal-covariance matrix.
- ``ANMSCM``           — additive noise with sum-of-sigmoids mechanisms;
  Head A closed form via random-feature basis covariance, Head I/C via MC.
- ``TreeSCM_Ident``    — sparse polytree with piecewise-linear edges;
  Head A closed form via path decomposition, Head I/C via MC on cached
  structural equations.

Every family samples its own features (no ``XSampler`` dependency) and
returns continuous ``(X, y)`` tensors at the same pre-``Reg2Cls`` point as
``MLPSCM``/``TreeSCM``. ``is_identifiable`` is always ``True`` for these.

Shape convention: ``X`` is ``(seq_len, num_features)``, ``y`` is
``(seq_len,)``. We treat the outcome as node ``p`` in the causal graph
(``0..p-1`` are features), with ``adj[i, j]`` the direct linear effect of
``j`` on ``i`` restricted to ``j < i`` in the sampled topological order.

The families are kept deliberately simple — randomised sparsity, bounded
coefficients — so closed-form variance identities stay numerically stable
across the label grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Noise distribution helpers
# ---------------------------------------------------------------------------


@dataclass
class NoiseDist:
    """Simple additive-noise distribution with known mean / variance.

    We limit ourselves to Laplace and centred-uniform noise so the
    non-Gaussianity required for LiNGAM / ANM identifiability is satisfied
    while keeping closed-form variance identities tractable.
    """

    kind: str        # "laplace" | "uniform" | "normal"
    scale: float     # Laplace b, uniform half-width, or normal std
    mean: float = 0.0

    @property
    def variance(self) -> float:
        if self.kind == "laplace":
            return 2.0 * self.scale * self.scale
        if self.kind == "uniform":
            return (self.scale * self.scale) * (4.0 / 12.0)
        if self.kind == "normal":
            return self.scale * self.scale
        raise ValueError(f"Unknown noise kind: {self.kind}")

    def sample(self, shape, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "laplace":
            return rng.laplace(loc=self.mean, scale=self.scale, size=shape)
        if self.kind == "uniform":
            return rng.uniform(low=self.mean - self.scale, high=self.mean + self.scale, size=shape)
        if self.kind == "normal":
            return rng.normal(loc=self.mean, scale=self.scale, size=shape)
        raise ValueError(f"Unknown noise kind: {self.kind}")


def _random_noise_dist(rng: np.random.Generator, non_gaussian: bool = True) -> NoiseDist:
    """Sample a noise distribution. Non-Gaussian by default (Laplace or uniform)."""
    if non_gaussian:
        kind = rng.choice(["laplace", "uniform"])
    else:
        kind = rng.choice(["laplace", "uniform", "normal"])
    # Keep scales modest so downstream variances do not explode.
    scale = float(rng.uniform(0.3, 1.2))
    return NoiseDist(kind=kind, scale=scale)


# ---------------------------------------------------------------------------
# Common DAG sampling
# ---------------------------------------------------------------------------


def _sample_dag(
    p: int,
    rng: np.random.Generator,
    expected_indegree: float = 2.0,
) -> Tuple[np.ndarray, List[int]]:
    """Sample a sparse lower-triangular adjacency mask in causal order.

    Nodes are labelled ``0..p-1`` in topological order by construction;
    ``adj_mask[i, j]`` indicates whether ``j`` is a direct parent of ``i``.
    The outcome node (indexed ``p``) is added later.
    """
    adj_mask = np.zeros((p, p), dtype=bool)
    for i in range(1, p):
        # Expected in-degree ~ expected_indegree, clipped to available ancestors.
        prob = min(1.0, expected_indegree / max(1, i))
        parents = rng.uniform(size=i) < prob
        adj_mask[i, :i] = parents
    topo_order = list(range(p))
    return adj_mask, topo_order


# ---------------------------------------------------------------------------
# LiNGAM
# ---------------------------------------------------------------------------


class LiNGAMSCM:
    """Linear non-Gaussian acyclic SCM.

    Structural equations (in topological order):

        X_j = sum_{k < j} A[j, k] * X_k + eps_j,      j = 0..p-1
        y   = sum_{k}     beta[k]  * X_k + eps_y

    with mutually independent non-Gaussian noise terms ``eps_*``.

    Attributes
    ----------
    p : int
        Number of observed features (outcome is separate).
    A : ndarray, shape (p, p)
        Strictly lower-triangular coefficient matrix among features.
    beta : ndarray, shape (p,)
        Outcome coefficients on features.
    noise_dists_x : list[NoiseDist], length p
    noise_dist_y  : NoiseDist
    seq_len : int
    rng : np.random.Generator
    """

    is_identifiable = True

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 10,
        num_outputs: int = 1,
        expected_indegree: float = 2.0,
        coef_scale: float = 1.0,
        beta_sparsity: float = 0.5,
        seed: Optional[int] = None,
        device: str = "cpu",
        **kwargs: Any,
    ):
        assert num_outputs == 1, "LiNGAMSCM only supports scalar outcome."
        self.seq_len = seq_len
        self.p = num_features
        self.device = device
        self.rng = np.random.default_rng(seed)

        # DAG + linear coefficients
        adj_mask, topo = _sample_dag(self.p, self.rng, expected_indegree=expected_indegree)
        A = np.zeros((self.p, self.p), dtype=np.float64)
        for i in range(self.p):
            for j in range(i):
                if adj_mask[i, j]:
                    # Sign-randomised uniform coefficients, bounded to keep
                    # covariance stable even as depth grows.
                    mag = self.rng.uniform(0.3, 1.0) * coef_scale
                    sign = self.rng.choice([-1.0, 1.0])
                    A[i, j] = sign * mag
        self.A = A
        self.topo_order = topo
        self.adj_mask = adj_mask

        # Outcome coefficients: sparse non-zero pattern, magnitude bounded.
        beta = np.zeros(self.p, dtype=np.float64)
        nz_mask = self.rng.uniform(size=self.p) < max(0.2, 1.0 - beta_sparsity)
        if not nz_mask.any():
            nz_mask[self.rng.integers(self.p)] = True
        for k in range(self.p):
            if nz_mask[k]:
                mag = self.rng.uniform(0.4, 1.2)
                sign = self.rng.choice([-1.0, 1.0])
                beta[k] = sign * mag
        self.beta = beta

        # Non-Gaussian noise
        self.noise_dists_x: List[NoiseDist] = [_random_noise_dist(self.rng) for _ in range(self.p)]
        self.noise_dist_y: NoiseDist = _random_noise_dist(self.rng)

    # ---- structural-equation sampling ------------------------------------

    def _simulate_features(
        self,
        n: int,
        intervene_on: Optional[Dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = rng or self.rng
        X = np.zeros((n, self.p), dtype=np.float64)
        intervene_on = intervene_on or {}
        for i in range(self.p):
            if i in intervene_on:
                X[:, i] = float(intervene_on[i])
            else:
                X[:, i] = self.noise_dists_x[i].sample(n, rng) + X[:, :i] @ self.A[i, :i]
        return X

    def _simulate_outcome(self, X: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or self.rng
        return X @ self.beta + self.noise_dist_y.sample(X.shape[0], rng)

    def simulate(
        self,
        intervene_on: Optional[Dict[int, float]] = None,
        n_samples: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw fresh noise and propagate through structural equations.

        ``intervene_on`` maps feature index -> clamped scalar value and
        implements ``do(X_i = x)``; downstream features are recomputed with
        the clamped ancestor values.
        """
        n = n_samples or self.seq_len
        rng = rng or self.rng
        X = self._simulate_features(n, intervene_on=intervene_on, rng=rng)
        y = self._simulate_outcome(X, rng=rng)
        return X, y

    # ---- convenience -----------------------------------------------------

    def __call__(self) -> Tuple[Tensor, Tensor]:
        X, y = self.simulate()
        X_t = torch.as_tensor(X, dtype=torch.float, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float, device=self.device)
        return X_t, y_t

    def covariance(self) -> np.ndarray:
        """Analytical ``(p+1)x(p+1)`` covariance of ``[X, y]``.

        Closed form: let ``B = (I - A)^{-1}`` so ``X = B @ eps_x``. Then
        ``Cov(X) = B @ diag(var_eps_x) @ B^T``. Including ``y = beta^T X + eps_y``
        yields the full block matrix.
        """
        p = self.p
        var_eps_x = np.array([d.variance for d in self.noise_dists_x])
        B = np.linalg.inv(np.eye(p) - self.A)
        cov_XX = B @ np.diag(var_eps_x) @ B.T
        cov_Xy = cov_XX @ self.beta
        var_y = float(self.beta @ cov_XX @ self.beta + self.noise_dist_y.variance)
        C = np.zeros((p + 1, p + 1))
        C[:p, :p] = cov_XX
        C[:p, p] = cov_Xy
        C[p, :p] = cov_Xy
        C[p, p] = var_y
        return C


# ---------------------------------------------------------------------------
# ANM — additive noise with sum-of-sigmoids mechanisms
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class _SumSigMechanism:
    """Sum-of-sigmoids mechanism ``f(x_parents) = sum_k a_k * sigmoid(w_k . x + b_k)``.

    The number of units, weights and biases are fixed at construction so the
    mechanism is deterministic given its parents. We use this as the
    nonlinear additive-noise mechanism for both feature and outcome nodes.
    """

    parents: np.ndarray          # shape (d,) indices of parents
    weights: np.ndarray          # shape (K, d)
    biases: np.ndarray           # shape (K,)
    amps: np.ndarray             # shape (K,)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.parents.size == 0:
            return np.zeros(X.shape[0])
        Xp = X[:, self.parents]  # (n, d)
        Z = Xp @ self.weights.T + self.biases  # (n, K)
        return _sigmoid(Z) @ self.amps  # (n,)


class ANMSCM:
    """Additive-noise SCM with bounded sum-of-sigmoids mechanisms.

    Each node's structural equation is

        X_j = f_j(X_parents(j)) + eps_j,
        y   = f_y(X_parents(y)) + eps_y

    with non-Gaussian additive noise. ``f_*`` is a bounded sum-of-sigmoids,
    which keeps every propagated variable on the same scale as its ancestors
    and yields numerically stable MC estimates of Head A/I/C.
    """

    is_identifiable = True

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 10,
        num_outputs: int = 1,
        expected_indegree: float = 2.0,
        num_sigmoids: int = 4,
        amp_scale: float = 1.0,
        seed: Optional[int] = None,
        device: str = "cpu",
        **kwargs: Any,
    ):
        assert num_outputs == 1, "ANMSCM only supports scalar outcome."
        self.seq_len = seq_len
        self.p = num_features
        self.device = device
        self.rng = np.random.default_rng(seed)

        adj_mask, topo = _sample_dag(self.p, self.rng, expected_indegree=expected_indegree)
        self.adj_mask = adj_mask
        self.topo_order = topo

        self.mechanisms_x: List[_SumSigMechanism] = []
        for i in range(self.p):
            parents = np.flatnonzero(adj_mask[i, :i])
            self.mechanisms_x.append(
                self._make_mechanism(parents, num_sigmoids, amp_scale)
            )

        # Outcome depends on a sparse subset of features.
        y_parent_mask = self.rng.uniform(size=self.p) < max(0.2, expected_indegree / max(1, self.p))
        if not y_parent_mask.any():
            y_parent_mask[self.rng.integers(self.p)] = True
        y_parents = np.flatnonzero(y_parent_mask)
        self.mechanism_y: _SumSigMechanism = self._make_mechanism(y_parents, num_sigmoids, amp_scale)

        self.noise_dists_x: List[NoiseDist] = [_random_noise_dist(self.rng) for _ in range(self.p)]
        self.noise_dist_y: NoiseDist = _random_noise_dist(self.rng)

    def _make_mechanism(
        self, parents: np.ndarray, K: int, amp_scale: float
    ) -> _SumSigMechanism:
        d = parents.size
        if d == 0:
            return _SumSigMechanism(parents, np.zeros((0, 0)), np.zeros(0), np.zeros(0))
        weights = self.rng.normal(0.0, 1.0, size=(K, d))
        biases = self.rng.normal(0.0, 1.0, size=K)
        amps = self.rng.uniform(-amp_scale, amp_scale, size=K)
        return _SumSigMechanism(parents, weights, biases, amps)

    def _simulate_features(
        self,
        n: int,
        intervene_on: Optional[Dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = rng or self.rng
        intervene_on = intervene_on or {}
        X = np.zeros((n, self.p), dtype=np.float64)
        for i in range(self.p):
            if i in intervene_on:
                X[:, i] = float(intervene_on[i])
            else:
                X[:, i] = self.mechanisms_x[i](X) + self.noise_dists_x[i].sample(n, rng)
        return X

    def _simulate_outcome(self, X: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or self.rng
        return self.mechanism_y(X) + self.noise_dist_y.sample(X.shape[0], rng)

    def simulate(
        self,
        intervene_on: Optional[Dict[int, float]] = None,
        n_samples: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = n_samples or self.seq_len
        rng = rng or self.rng
        X = self._simulate_features(n, intervene_on=intervene_on, rng=rng)
        y = self._simulate_outcome(X, rng=rng)
        return X, y

    def __call__(self) -> Tuple[Tensor, Tensor]:
        X, y = self.simulate()
        X_t = torch.as_tensor(X, dtype=torch.float, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float, device=self.device)
        return X_t, y_t


# ---------------------------------------------------------------------------
# Polytree with piecewise-linear edges
# ---------------------------------------------------------------------------


@dataclass
class _PiecewiseLinearEdge:
    """Continuous piecewise-linear mechanism on a fixed grid.

    The edge maps a scalar parent value to a scalar contribution via linear
    interpolation between ``(knots, values)``. This is smooth enough to give
    a non-degenerate Head A while keeping closed-form path decomposition
    available (the edge is a deterministic scalar function of the parent).
    """

    knots: np.ndarray   # shape (K,), sorted
    values: np.ndarray  # shape (K,)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.knots, self.values)


class TreeSCM_Ident:
    """Polytree SCM with piecewise-linear edges.

    Every non-root node has exactly one parent; the outcome has a single
    feature parent. This single-parent constraint makes the structural
    equations a composition of scalar functions, so closed-form Head A
    labels reduce to "is i on the path from root to y?".
    """

    is_identifiable = True

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 10,
        num_outputs: int = 1,
        knot_count: int = 8,
        seed: Optional[int] = None,
        device: str = "cpu",
        **kwargs: Any,
    ):
        assert num_outputs == 1, "TreeSCM_Ident only supports scalar outcome."
        self.seq_len = seq_len
        self.p = num_features
        self.device = device
        self.rng = np.random.default_rng(seed)

        # Sample a rooted polytree: feature 0 is the root; every other
        # feature has a unique parent among earlier features.
        parents = np.full(self.p, -1, dtype=int)
        for i in range(1, self.p):
            parents[i] = int(self.rng.integers(0, i))
        self.parents = parents

        # Outcome parent is any feature
        self.y_parent = int(self.rng.integers(0, self.p))

        # Piecewise-linear edges keyed by child node index.
        self.edges: Dict[int, _PiecewiseLinearEdge] = {}
        for i in range(1, self.p):
            self.edges[i] = self._random_edge(knot_count)
        self.edge_y: _PiecewiseLinearEdge = self._random_edge(knot_count)

        self.noise_dists_x: List[NoiseDist] = [_random_noise_dist(self.rng) for _ in range(self.p)]
        self.noise_dist_y: NoiseDist = _random_noise_dist(self.rng)

    def _random_edge(self, K: int) -> _PiecewiseLinearEdge:
        knots = np.sort(self.rng.uniform(-3.0, 3.0, size=K))
        values = self.rng.uniform(-1.5, 1.5, size=K)
        return _PiecewiseLinearEdge(knots=knots, values=values)

    # Path from the root (feature 0) to feature i.
    def path_to(self, i: int) -> List[int]:
        path = [i]
        cur = i
        while self.parents[cur] != -1:
            cur = int(self.parents[cur])
            path.append(cur)
        return list(reversed(path))

    def _simulate_features(
        self,
        n: int,
        intervene_on: Optional[Dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        rng = rng or self.rng
        intervene_on = intervene_on or {}
        X = np.zeros((n, self.p), dtype=np.float64)
        # Root is pure noise.
        if 0 in intervene_on:
            X[:, 0] = float(intervene_on[0])
        else:
            X[:, 0] = self.noise_dists_x[0].sample(n, rng)
        for i in range(1, self.p):
            if i in intervene_on:
                X[:, i] = float(intervene_on[i])
            else:
                parent_val = X[:, self.parents[i]]
                X[:, i] = self.edges[i](parent_val) + self.noise_dists_x[i].sample(n, rng)
        return X

    def _simulate_outcome(self, X: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or self.rng
        return self.edge_y(X[:, self.y_parent]) + self.noise_dist_y.sample(X.shape[0], rng)

    def simulate(
        self,
        intervene_on: Optional[Dict[int, float]] = None,
        n_samples: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = n_samples or self.seq_len
        rng = rng or self.rng
        X = self._simulate_features(n, intervene_on=intervene_on, rng=rng)
        y = self._simulate_outcome(X, rng=rng)
        return X, y

    def __call__(self) -> Tuple[Tensor, Tensor]:
        X, y = self.simulate()
        X_t = torch.as_tensor(X, dtype=torch.float, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float, device=self.device)
        return X_t, y_t
