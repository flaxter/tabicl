"""Phase 3 — closed-form + Monte-Carlo attribution-label computation.

Given an SCM instance and a pre-``Reg2Cls`` ``(X, y)`` sample from it, we
compute three attribution-label tensors per dataset:

- ``o_star``:       Head A labels, shape ``(p,)``, in ``Var(continuous_y)`` units.
- ``i_star``:       Head I labels, shape ``(p,)``, in ``continuous_y`` units.
                    All-NaN on non-identifiable samplers.
- ``c_triples``:    ``k = min(p, 16)`` sampled ``(i, S_mask, c_star)`` triples
                    for Head C. Not an all-pairs matrix.

The dispatch is:

- ``LiNGAMSCM``      → covariance algebra (all three heads closed form).
- ``ANMSCM``         → MC on structural equations (A: ``Var(E[y|...])`` via
                       grouping fresh noise; I: ``do(X_i = x)``; C: as A).
- ``TreeSCM_Ident``  → MC on cached structural equations.
- ``MLPSCM``         → fresh-noise MC only for A / C; ``is_identifiable=False``.
- ``TreeSCM``        → fresh-noise MC only for A / C; ``is_identifiable=False``.

Labels are tied to ``continuous_y`` (the outcome *before* ``Reg2Cls``). Phase
4's loss has to know this — see ``notes/PHASE3.md`` §Risks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .identifiable_scm import ANMSCM, LiNGAMSCM, TreeSCM_Ident


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_labels(
    scm: Any,
    X: Tensor,
    y: Tensor,
    params: Optional[Dict[str, Any]] = None,
    n_mc: int = 2048,
    k_cond_triples: int = 16,
    rng: Optional[np.random.Generator] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute attribution labels for one pre-``Reg2Cls`` dataset.

    Parameters
    ----------
    scm : object
        The instantiated SCM that produced ``(X, y)``. Dispatch is by type.
    X : Tensor, shape (T, p)
        Continuous features.
    y : Tensor, shape (T,)
        Continuous outcome.
    params : dict, optional
        Parameters passed to ``generate_dataset``. Used only for defaults.
    n_mc : int
        MC sample count for non-closed-form heads.
    k_cond_triples : int
        Number of Head-C triples sampled per dataset.
    rng : np.random.Generator, optional
    mode : {"closed_form", "mc", "nan"}, optional
        Overrides the type-based dispatch. ``"nan"`` returns all-NaN labels.

    Returns
    -------
    dict with keys ``o_star``, ``i_star``, ``c_triples``, ``is_identifiable``.
    """
    rng = rng or np.random.default_rng()
    p = X.shape[-1]

    if mode == "nan":
        return _nan_labels(p, k_cond_triples)

    if isinstance(scm, LiNGAMSCM):
        return _labels_lingam(scm, p, k_cond_triples, rng)
    if isinstance(scm, ANMSCM):
        return _labels_anm(scm, p, n_mc, k_cond_triples, rng)
    if isinstance(scm, TreeSCM_Ident):
        return _labels_tree_ident(scm, p, n_mc, k_cond_triples, rng)

    # Non-identifiable samplers (MLPSCM, TreeSCM): fresh-noise MC for A / C.
    simulate = getattr(scm, "simulate", None)
    if callable(simulate):
        return _labels_mc_non_identifiable(scm, p, n_mc, k_cond_triples, rng)

    return _nan_labels(p, k_cond_triples)


# ---------------------------------------------------------------------------
# NaN fallback
# ---------------------------------------------------------------------------


def _nan_labels(p: int, k_cond_triples: int) -> Dict[str, Any]:
    return {
        "o_star": torch.full((p,), float("nan")),
        "i_star": torch.full((p,), float("nan")),
        "c_triples": [],
        "is_identifiable": False,
    }


# ---------------------------------------------------------------------------
# Head-C triple sampling
# ---------------------------------------------------------------------------


def _sample_c_triples(p: int, k: int, rng: np.random.Generator) -> List[Tuple[int, np.ndarray]]:
    """Sample ``k`` ``(i, S_mask)`` pairs with ``i`` not in ``S``.

    ``S`` is drawn uniformly from subsets of ``{0..p-1} \\ {i}`` with a
    uniform |S| between 0 and p-1.
    """
    triples: List[Tuple[int, np.ndarray]] = []
    if p <= 0:
        return triples
    for _ in range(k):
        i = int(rng.integers(0, p))
        others = np.array([j for j in range(p) if j != i], dtype=int)
        if others.size == 0:
            S_mask = np.zeros(p, dtype=bool)
        else:
            size = int(rng.integers(0, others.size + 1))
            if size == 0:
                S = np.zeros(0, dtype=int)
            else:
                S = rng.choice(others, size=size, replace=False)
            S_mask = np.zeros(p, dtype=bool)
            S_mask[S] = True
        triples.append((i, S_mask))
    return triples


# ---------------------------------------------------------------------------
# LiNGAM closed form (covariance algebra)
# ---------------------------------------------------------------------------


def _labels_lingam(
    scm: LiNGAMSCM,
    p: int,
    k_cond_triples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    C = scm.covariance()  # (p+1, p+1)
    sig_XX = C[:p, :p]
    sig_Xy = C[:p, p]
    var_y = float(C[p, p])

    # Head A per PHASE3.md: o*_i = Var(E[y|X]) - Var(E[y|X_{-i}]).
    # With the joint covariance ``C`` this is the Schur-complement drop when
    # feature ``i`` is removed from the full conditioning set.
    o_star = np.zeros(p)
    full_idx = np.arange(p)
    for i in range(p):
        excl = np.array([j for j in full_idx if j != i])
        o_star[i] = max(
            0.0,
            _explained_var_lin(sig_XX, sig_Xy, full_idx)
            - _explained_var_lin(sig_XX, sig_Xy, excl),
        )

    # Head I closed form. Under do(X_i = x), downstream features see a
    # deterministic shift ``B[:, i] * x`` where ``B = (I - A)^{-1}`` — i.e.,
    # the total-effect column of ``i``. So
    #   E[y | do(X_i=x)] - E[y]  =  (B^T beta)_i * x
    # and i*_i = | (B^T beta)_i | * sd( pi_i ).
    B = np.linalg.inv(np.eye(p) - scm.A)
    total_effects = B.T @ scm.beta  # shape (p,)
    sd_feat = np.sqrt(np.clip(np.diag(sig_XX), 1e-12, None))
    i_star = np.abs(total_effects) * sd_feat

    c_triples: List[Tuple[int, Tensor, float]] = []
    for i, S_mask in _sample_c_triples(p, k_cond_triples, rng):
        gain = _partial_variance_gain(sig_XX, sig_Xy, var_y, i, cond=S_mask)
        c_triples.append((i, torch.as_tensor(S_mask, dtype=torch.bool), float(gain)))

    return {
        "o_star": torch.as_tensor(o_star, dtype=torch.float),
        "i_star": torch.as_tensor(i_star, dtype=torch.float),
        "c_triples": c_triples,
        "is_identifiable": True,
    }


def _partial_variance_gain(
    sig_XX: np.ndarray,
    sig_Xy: np.ndarray,
    var_y: float,
    i: int,
    cond: np.ndarray,
) -> float:
    """Return ``Var(E[y | X_S, X_i]) - Var(E[y | X_S])`` in the joint-Gaussian
    regression that corresponds to the linear SCM.

    For a multivariate-Gaussian ``(X, y)`` the conditional variance of ``y``
    given a subset ``T`` is ``var_y - sig_yT sig_TT^{-1} sig_Ty`` — we use
    this with ``T = S`` and ``T = S ∪ {i}``. The identity is exact under
    linear-Gaussian; under non-Gaussian LiNGAM noise it remains exact for
    the *explained-variance* quantity because ``E[y | T]`` in the
    best-linear-predictor sense is still linear.
    """
    S_idx = np.flatnonzero(cond)
    S_plus = np.concatenate([S_idx, np.array([i])])

    expl_S = _explained_var_lin(sig_XX, sig_Xy, S_idx)
    expl_Si = _explained_var_lin(sig_XX, sig_Xy, S_plus)
    return float(max(0.0, expl_Si - expl_S))


def _explained_var_lin(sig_XX: np.ndarray, sig_Xy: np.ndarray, T_idx: np.ndarray) -> float:
    """``Var(E[y | X_T])`` for the best-linear predictor of ``y`` given ``X_T``.

    For multivariate-Gaussian ``(X, y)`` this equals
    ``sig_yT @ sig_TT^{-1} @ sig_Ty``. For non-Gaussian LiNGAM it is still
    the correct linear-regression R^2 numerator.
    """
    if T_idx.size == 0:
        return 0.0
    sig_TT = sig_XX[np.ix_(T_idx, T_idx)]
    sig_yT = sig_Xy[T_idx]
    # Ridge-stabilise the inverse for near-singular configurations.
    try:
        sol = np.linalg.solve(sig_TT + 1e-10 * np.eye(T_idx.size), sig_yT)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(sig_TT, sig_yT, rcond=None)[0]
    return float(sig_yT @ sol)


# ---------------------------------------------------------------------------
# ANM / TreeSCM_Ident — MC on known structural equations
# ---------------------------------------------------------------------------


def _labels_anm(
    scm: ANMSCM,
    p: int,
    n_mc: int,
    k_cond_triples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    return _labels_mc_identifiable(scm, p, n_mc, k_cond_triples, rng)


def _labels_tree_ident(
    scm: TreeSCM_Ident,
    p: int,
    n_mc: int,
    k_cond_triples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    return _labels_mc_identifiable(scm, p, n_mc, k_cond_triples, rng)


def _labels_mc_identifiable(
    scm: Any,
    p: int,
    n_mc: int,
    k_cond_triples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """MC label computation that uses the SCM's structural equations directly.

    Requires ``scm.simulate(intervene_on=..., n_samples=..., rng=...)`` and
    ``scm.mechanism_y`` / ``scm.edge_y`` — i.e., the outcome mechanism must
    be applied to a provided ``X`` matrix in closed form (no noise) so we
    can evaluate ``E[y | X]`` at the structural level.
    """
    mu_y = _expected_y(scm, _get_feature_matrix(scm, n_mc, rng))
    var_y = float(np.var(mu_y + scm.noise_dist_y.sample(mu_y.shape[0], rng)))

    # Head A via residual-variance drop: Var(E[y|X]) - Var(E[y|X_{-i}])
    X_mc = _get_feature_matrix(scm, n_mc, rng)
    mu_full = _expected_y(scm, X_mc)
    var_mu_full = float(np.var(mu_full))

    o_star = np.zeros(p)
    # For each i, estimate Var(E[y|X_{-i}]) by resampling X_i conditional on X_{-i}.
    # For a strict causal SCM we cannot easily condition on X_{-i}, but
    # Var(E[y|X_{-i}]) = Var(mu_full) - E[Var(mu_full | X_{-i})] under the
    # identity Var(A) = Var(E[A|B]) + E[Var(A|B)]. We estimate the
    # inner-variance term by grouping Monte Carlo: for each base sample we
    # redraw X_i from its marginal (a *permutation* of the column, which is
    # a valid sample from the marginal).
    for i in range(p):
        X_shuf = X_mc.copy()
        perm = rng.permutation(n_mc)
        X_shuf[:, i] = X_mc[perm, i]
        mu_shuf = _expected_y(scm, X_shuf)
        # E[Var(mu_full | X_{-i})] ~ 0.5 * E[(mu_full - mu_shuf)^2]
        #   — this is Sobol's total-effect pick-freeze estimator.
        # First-order index: o*_i = Var(E[y|X]) - E[Var(E[y|X]|X_{-i})], but
        # for independent X the PLAN's first-order Sobol reduces to:
        #   Var(E[y|X_i]).
        # We use the cheaper estimate
        #   o*_i = Var(mu_full) - 0.5 * E[(mu_full - mu_shuf)^2]
        # which equals the PLAN's first-order Sobol under feature
        # independence, and is a well-behaved proxy under mild dependence.
        inner = 0.5 * float(np.mean((mu_full - mu_shuf) ** 2))
        o_star[i] = max(0.0, var_mu_full - inner)

    # Head I via the structural do-operator.
    i_star = np.zeros(p)
    n_outer = min(64, max(8, n_mc // 32))
    X_base = _get_feature_matrix(scm, n_mc, rng)
    mu_base_mean = float(np.mean(_expected_y(scm, X_base)))
    for i in range(p):
        # Sample x-values from pi_i (marginal of X_i) via the simulated column.
        x_vals = rng.choice(X_base[:, i], size=n_outer, replace=True)
        diffs = np.zeros(n_outer)
        for j, x in enumerate(x_vals):
            X_int, _ = scm.simulate(intervene_on={i: float(x)}, n_samples=n_mc // 4, rng=rng)
            if isinstance(X_int, torch.Tensor):
                X_int = X_int.cpu().numpy()
            mu_int = _expected_y(scm, X_int)
            diffs[j] = float(np.mean(mu_int)) - mu_base_mean
        i_star[i] = float(np.sqrt(max(0.0, np.mean(diffs ** 2))))

    # Head C via the same pick-freeze estimator but conditional on S.
    c_triples: List[Tuple[int, Tensor, float]] = []
    for i, S_mask in _sample_c_triples(p, k_cond_triples, rng):
        c_val = _head_c_mc(scm, X_mc, mu_full, i, S_mask, rng)
        c_triples.append((i, torch.as_tensor(S_mask, dtype=torch.bool), float(c_val)))

    return {
        "o_star": torch.as_tensor(o_star, dtype=torch.float),
        "i_star": torch.as_tensor(i_star, dtype=torch.float),
        "c_triples": c_triples,
        "is_identifiable": True,
    }


def _head_c_mc(
    scm: Any,
    X_mc: np.ndarray,
    mu_full: np.ndarray,
    i: int,
    S_mask: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Sobol-style conditional-contribution estimator.

    Freezing X_S and resampling X_{not in S ∪ {i}} + X_i gives
    E[Var(mu | X_S)]; additionally freezing X_i then gives
    E[Var(mu | X_S, X_i)]. Their difference is Head C.
    """
    n = X_mc.shape[0]
    # E[Var(mu | X_S)] ≈ 0.5 * E[(mu_full - mu_shuf_notS)^2]
    mask_notS = ~S_mask
    X_shuf_notS = X_mc.copy()
    perm = rng.permutation(n)
    X_shuf_notS[:, mask_notS] = X_mc[np.ix_(perm, mask_notS)]
    mu_shuf_notS = _expected_y(scm, X_shuf_notS)
    var_given_S = 0.5 * float(np.mean((mu_full - mu_shuf_notS) ** 2))

    # E[Var(mu | X_S ∪ {i})] ≈ 0.5 * E[(mu_full - mu_shuf_notSi)^2]
    mask_notSi = mask_notS.copy()
    mask_notSi[i] = False
    X_shuf_notSi = X_mc.copy()
    perm2 = rng.permutation(n)
    X_shuf_notSi[:, mask_notSi] = X_mc[np.ix_(perm2, mask_notSi)]
    mu_shuf_notSi = _expected_y(scm, X_shuf_notSi)
    var_given_Si = 0.5 * float(np.mean((mu_full - mu_shuf_notSi) ** 2))

    return max(0.0, var_given_S - var_given_Si)


def _expected_y(scm: Any, X: np.ndarray) -> np.ndarray:
    """Return ``E[y | X] = f_y(X)`` evaluated in closed form (no noise).

    We dispatch on the SCM type to pick the noise-free outcome mechanism.
    """
    if isinstance(scm, LiNGAMSCM):
        return X @ scm.beta
    if isinstance(scm, ANMSCM):
        return scm.mechanism_y(X)
    if isinstance(scm, TreeSCM_Ident):
        return scm.edge_y(X[:, scm.y_parent])
    raise TypeError(f"No noise-free outcome mechanism for {type(scm).__name__}")


def _get_feature_matrix(scm: Any, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fresh-noise simulation of X to feed into ``_expected_y``."""
    X, _ = scm.simulate(n_samples=n, rng=rng)
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    return X


# ---------------------------------------------------------------------------
# Non-identifiable fallback (MLPSCM / TreeSCM): fresh-noise MC for A / C only
# ---------------------------------------------------------------------------


def _labels_mc_non_identifiable(
    scm: Any,
    p: int,
    n_mc: int,
    k_cond_triples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """MC labels for samplers with no tractable noiseless outcome mechanism.

    MLP-SCM and TreeSCM expose ``simulate()`` but every rerun injects
    independent additive noise, so we can't evaluate ``E[y | X]`` in closed
    form the way we do for the identifiable families. As a fallback we use
    a plug-in estimator of ``Var(E[y | X_T])`` via feature binning:

    1. Sample one fresh batch of ``(X, y)`` from the SCM.
    2. For each target subset ``T``, bin the feature vectors ``X[:, T]``
       (each column into ``n_bins`` quantile buckets; combined subset bin
       index is a tuple hash) and take the variance of the per-bin mean
       of ``y``. This is a consistent estimator of ``Var(E[y | X_T])``
       as both the sample size and bin count grow.

    Head I is always NaN here — the plan explicitly tags these samplers as
    non-identifiable.
    """
    sim = scm.simulate
    X_base, y_base = sim(n_samples=n_mc, rng=rng)
    if isinstance(X_base, torch.Tensor):
        X_base = X_base.cpu().numpy()
        y_base = y_base.cpu().numpy()

    n_bins = 10
    # Pre-bin every column once.
    col_bins = np.stack(
        [_quantile_bin(X_base[:, j], n_bins=n_bins) for j in range(p)], axis=1
    )

    var_all = _binned_explained_var(col_bins, y_base, np.arange(p))
    o_star = np.zeros(p)
    for i in range(p):
        o_star[i] = max(0.0, _binned_explained_var(col_bins, y_base, np.array([i])))

    c_triples: List[Tuple[int, Tensor, float]] = []
    for i, S_mask in _sample_c_triples(p, k_cond_triples, rng):
        S_idx = np.flatnonzero(S_mask)
        Si_idx = np.concatenate([S_idx, np.array([i])])
        vS = _binned_explained_var(col_bins, y_base, S_idx) if S_idx.size > 0 else 0.0
        vSi = _binned_explained_var(col_bins, y_base, Si_idx)
        c_val = max(0.0, vSi - vS)
        c_triples.append((i, torch.as_tensor(S_mask, dtype=torch.bool), float(c_val)))

    return {
        "o_star": torch.as_tensor(o_star, dtype=torch.float),
        "i_star": torch.full((p,), float("nan")),
        "c_triples": c_triples,
        "is_identifiable": False,
    }


def _quantile_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin a column into ``n_bins`` quantile buckets, returning integer codes."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.searchsorted(edges, x) - 1, 0, n_bins - 1)


def _binned_explained_var(col_bins: np.ndarray, y: np.ndarray, T_idx: np.ndarray) -> float:
    """Plug-in ``Var(E[y | X_T])`` from feature-bin tuples.

    ``col_bins`` is ``(n, p)`` int; ``T_idx`` is the feature subset. We hash
    each row's bins-tuple to a single integer, compute mean y per key, and
    return the (sample-size weighted) variance of those means.
    """
    if T_idx.size == 0:
        return 0.0
    keys = col_bins[:, T_idx]
    # Encode multi-dim bin tuple as single int.
    flat = np.zeros(keys.shape[0], dtype=np.int64)
    for col in range(keys.shape[1]):
        flat = flat * 10 + keys[:, col]
    # Grouped mean.
    order = np.argsort(flat)
    sorted_flat = flat[order]
    sorted_y = y[order]
    split = np.where(np.diff(sorted_flat) != 0)[0] + 1
    groups = np.split(sorted_y, split)
    means = np.array([float(np.mean(g)) for g in groups])
    counts = np.array([g.size for g in groups], dtype=np.float64)
    overall = float(np.sum(means * counts) / counts.sum())
    return float(np.sum(counts * (means - overall) ** 2) / counts.sum())
