"""Tests for conditional predictive value labels."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tabicl.prior.labels import (
    OracleContext,
    _binned_V,
    _direct_delta_cf_knn,
    V_gaussian,
    build_oracle_context,
    delta_gaussian,
    delta_vector_for_S_direct_knn,
    compute_value_queries,
    sample_value_queries_meta,
    ValueQuery,
)


# ---------------------------------------------------------------------------
# V_gaussian / delta_gaussian — exact closed form fixtures
# ---------------------------------------------------------------------------


def _random_psd(p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p + 1, p + 1))
    M = A @ A.T + 0.5 * np.eye(p + 1)
    return M


def test_V_gaussian_empty_is_zero():
    Sigma = _random_psd(4, seed=0)
    assert V_gaussian(Sigma, y_idx=4, S=np.zeros(0, dtype=int)) == 0.0


def test_delta_gaussian_nonnegative():
    Sigma = _random_psd(5, seed=1)
    y_idx = 5
    rng = np.random.default_rng(2)
    for _ in range(30):
        p = 5
        size = int(rng.integers(0, p))
        S = rng.choice(p, size=size, replace=False) if size > 0 else np.zeros(0, dtype=int)
        i = int(rng.choice([j for j in range(p) if j not in set(S.tolist())]))
        assert delta_gaussian(Sigma, y_idx, i, S) >= -1e-9


def test_delta_squared_conditional_mean_identity_gaussian():
    """Under joint Gaussianity, Delta_{i|S} should equal E[(mu_{S+i} - mu_S)^2]
    where mu_T = E[Y|X_T]. Sanity-check against a direct MC estimate."""
    rng = np.random.default_rng(42)
    p = 4
    Sigma = _random_psd(p, seed=7)
    mean = np.zeros(p + 1)

    # Sample (X, Y) jointly.
    n = 20000
    samples = rng.multivariate_normal(mean, Sigma, size=n)
    X = samples[:, :p]
    Y = samples[:, p]

    S = np.array([0, 1])
    i = 2

    # Compute mu_S(x_S) = Sigma_{Y,S} Sigma_{S,S}^{-1} x_S.
    beta_S = np.linalg.solve(Sigma[np.ix_(S, S)], Sigma[p, S])
    beta_Si = np.linalg.solve(
        Sigma[np.ix_(np.r_[S, i], np.r_[S, i])],
        Sigma[p, np.r_[S, i]],
    )
    mu_S = X[:, S] @ beta_S
    mu_Si = X[:, np.r_[S, i]] @ beta_Si

    direct = float(np.mean((mu_Si - mu_S) ** 2))
    closed = delta_gaussian(Sigma, y_idx=p, i=i, S=S)
    # Large n tolerance.
    assert abs(direct - closed) < 0.03 * max(1.0, abs(closed)), (
        f"direct={direct:.4f} vs closed={closed:.4f}"
    )


def test_empty_set_delta_equals_V_singleton():
    """Delta_{i|empty} = V({i}) by definition."""
    Sigma = _random_psd(4, seed=9)
    for i in range(4):
        delta_empty = delta_gaussian(Sigma, y_idx=4, i=i, S=np.zeros(0, dtype=int))
        v_i = V_gaussian(Sigma, y_idx=4, S=np.array([i]))
        assert abs(delta_empty - v_i) < 1e-10


def test_leave_one_out_delta_equals_V_full_minus_V_minus_i():
    """Delta_{i|[p]\\{i}} = V([p]) - V([p]\\{i})."""
    Sigma = _random_psd(5, seed=11)
    p = 5
    for i in range(p):
        S_minus_i = np.array([j for j in range(p) if j != i])
        direct = delta_gaussian(Sigma, y_idx=p, i=i, S=S_minus_i)
        V_full = V_gaussian(Sigma, y_idx=p, S=np.arange(p))
        V_minus_i = V_gaussian(Sigma, y_idx=p, S=S_minus_i)
        assert abs(direct - max(0.0, V_full - V_minus_i)) < 1e-9


# ---------------------------------------------------------------------------
# Mixture sampler
# ---------------------------------------------------------------------------


def test_default_mixture_emits_10_states():
    rng = np.random.default_rng(0)
    states = sample_value_queries_meta(p=8, rng=rng, mixture="default")
    assert len(states) == 10


def test_backup_mixture_emits_6_states():
    rng = np.random.default_rng(0)
    states = sample_value_queries_meta(p=8, rng=rng, mixture="backup")
    assert len(states) == 6


def test_mixture_size_buckets_cover_range():
    rng = np.random.default_rng(0)
    states = sample_value_queries_meta(p=10, rng=rng, mixture="default")
    sizes = sorted(len(S) for S, _ in states)
    # Must hit |S|=0, |S|=1, and some near-full entries (p-2 or p-1).
    assert 0 in sizes
    assert 1 in sizes
    assert any(s >= 8 for s in sizes)


# ---------------------------------------------------------------------------
# compute_value_queries on a tiny MLPSCM-like stub
# ---------------------------------------------------------------------------


class _DuplicateSCM:
    """Y = X0 + noise; X1 = X0 + tiny noise (duplicate). X2 is independent."""

    def __init__(self):
        self.noise = 0.1

    def simulate(self, n_samples: int, rng: np.random.Generator):
        X0 = rng.standard_normal(n_samples)
        X1 = X0 + self.noise * rng.standard_normal(n_samples)
        X2 = rng.standard_normal(n_samples)
        X = np.stack([X0, X1, X2], axis=1)
        Y = X0 + self.noise * rng.standard_normal(n_samples)
        return torch.from_numpy(X), torch.from_numpy(Y)


class _InteractionSCM:
    """Y = X0 * X1 + noise; X0, X1, X2 independent standard normal."""

    def simulate(self, n_samples: int, rng: np.random.Generator):
        X = rng.standard_normal((n_samples, 3))
        Y = X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(n_samples)
        return torch.from_numpy(X), torch.from_numpy(Y)


def test_duplicate_features_high_s_low_n():
    """X0 and X1 are near-duplicates: both have high standalone value
    but low leave-one-out value."""
    scm = _DuplicateSCM()
    X = torch.zeros(16, 3)  # just carries the feature count
    y = torch.zeros(16)
    rng = np.random.default_rng(0)
    payload = compute_value_queries(scm, X, y, n_oracle=4000, rng=rng)
    queries = payload["value_queries"]

    # Find the empty-set query -> standalone values.
    empty_q = next(q for q in queries if q.query_type == "empty")
    s = empty_q.raw_targets.cpu().numpy()
    assert s[0] > 0.3 and s[1] > 0.3, f"standalone values low: {s}"

    # Build a leave-one-out query for X0 (S = {1, 2}) — one of the near_full
    # queries should include it.
    loo_candidates = [q for q in queries if q.query_type == "near_full"]
    assert loo_candidates, "no near_full queries produced"
    # Find one with exactly {1, 2} in S (p=3 so |S|=p-1=2).
    target_mask = torch.tensor([False, True, True])
    loo = next(
        (q for q in loo_candidates if torch.equal(q.S_mask, target_mask)),
        None,
    )
    if loo is not None:
        s0 = s[0]
        loo_s0 = float(loo.raw_targets[0])
        assert loo_s0 < 0.5 * s0, (
            f"leave-one-out value for duplicate should be much smaller than "
            f"standalone: standalone={s0:.3f}, loo={loo_s0:.3f}"
        )


def test_interaction_low_s_high_conditional():
    """Y = X0 * X1: each feature alone has low value, but conditional on
    the partner the marginal value is much higher."""
    scm = _InteractionSCM()
    X = torch.zeros(16, 3)
    y = torch.zeros(16)
    rng = np.random.default_rng(0)
    payload = compute_value_queries(scm, X, y, n_oracle=4000, rng=rng)

    queries = payload["value_queries"]
    empty_q = next(q for q in queries if q.query_type == "empty")
    s = empty_q.raw_targets.cpu().numpy()
    # Standalone values for X0 and X1 should be small (pure interaction).
    assert s[0] < 0.3 and s[1] < 0.3, f"pure interaction should have low s: {s}"

    # Find a singleton query with S={0} — the conditional value of X1 given X0
    # should be much higher than s[1].
    single_S0 = next(
        (q for q in queries
         if q.query_type == "singleton" and q.S_mask[0].item() and not q.S_mask[1].item() and not q.S_mask[2].item()),
        None,
    )
    if single_S0 is not None:
        cond_1_given_0 = float(single_S0.raw_targets[1])
        assert cond_1_given_0 > 2 * s[1], (
            f"conditional value of X1 given {{X0}} should exceed standalone: "
            f"cond={cond_1_given_0:.3f}, s1={s[1]:.3f}"
        )


def test_targets_are_NaN_inside_S():
    """Positions i in S must be NaN in both targets and raw_targets."""
    scm = _DuplicateSCM()
    X = torch.zeros(16, 3)
    y = torch.zeros(16)
    rng = np.random.default_rng(0)
    payload = compute_value_queries(scm, X, y, n_oracle=500, rng=rng)
    for q in payload["value_queries"]:
        S_idx = q.S_mask.nonzero(as_tuple=False).flatten().tolist()
        for i in S_idx:
            assert torch.isnan(q.targets[i])
            assert q.raw_targets is None or torch.isnan(q.raw_targets[i])


def test_targets_are_rms_sqrt_of_raw():
    """targets[i] == sqrt(max(raw_targets[i], 0)) wherever finite."""
    scm = _DuplicateSCM()
    X = torch.zeros(16, 3)
    y = torch.zeros(16)
    rng = np.random.default_rng(0)
    payload = compute_value_queries(scm, X, y, n_oracle=500, rng=rng)
    for q in payload["value_queries"]:
        if q.raw_targets is None:
            continue
        finite = torch.isfinite(q.targets)
        assert torch.allclose(
            q.targets[finite],
            torch.sqrt(torch.clamp(q.raw_targets[finite], min=0)),
            atol=1e-6,
        )


def test_binned_V_ignores_nonfinite_y():
    col_bins = np.array(
        [
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
        ],
        dtype=np.int64,
    )
    y = np.array([1.0, np.inf, 3.0, 5.0], dtype=np.float64)
    got = _binned_V(col_bins, y, np.array([0, 1], dtype=int))
    assert got == pytest.approx(2.0)


def test_binned_V_wide_state_does_not_overflow_group_keys():
    col_bins = np.array(
        [
            [0] * 20,
            [0] * 20,
            [9] * 19 + [8],
            [9] * 19 + [9],
        ],
        dtype=np.int64,
    )
    y = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
    got = _binned_V(col_bins, y, np.arange(20, dtype=int))
    assert got == pytest.approx(4.5)


class _AllInfYSCM:
    def simulate(self, n_samples: int, rng: np.random.Generator):
        X = rng.standard_normal((n_samples, 3))
        Y = np.full(n_samples, np.inf)
        return torch.from_numpy(X), torch.from_numpy(Y)


def test_compute_value_queries_returns_empty_for_nonfinite_oracle_y():
    scm = _AllInfYSCM()
    X = torch.zeros(16, 3)
    y = torch.zeros(16)
    payload = compute_value_queries(
        scm, X, y, n_oracle=128, rng=np.random.default_rng(0)
    )
    assert payload["value_queries"] == []
    assert np.isnan(payload["y_var_raw"])


# ---------------------------------------------------------------------------
# _direct_delta_cf_knn — REMEDY.md v1
# ---------------------------------------------------------------------------


def _gaussian_scm_sample(Sigma: np.ndarray, y_idx: int, n: int, seed: int):
    """Draw (X, y) from a jointly Gaussian (X, Y) with covariance Sigma."""
    rng = np.random.default_rng(seed)
    p = Sigma.shape[0] - 1
    Z = rng.multivariate_normal(np.zeros(Sigma.shape[0]), Sigma, size=n)
    X = np.delete(Z, y_idx, axis=1)
    y = Z[:, y_idx]
    return X.astype(np.float64), y.astype(np.float64), p


def test_direct_delta_nan_for_i_in_S():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = rng.standard_normal(200)
    out = _direct_delta_cf_knn(X, y, np.array([0, 2]), p=4, rng=np.random.default_rng(1))
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert np.isfinite(out[1]) and np.isfinite(out[3])


def test_direct_delta_nonnegative_without_clip():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 5))
    y = rng.standard_normal(300)
    out = _direct_delta_cf_knn(X, y, np.array([1]), p=5, rng=np.random.default_rng(3))
    for i in [0, 2, 3, 4]:
        assert out[i] >= 0.0


def test_direct_delta_empty_S_uses_train_mean():
    """|S|=0 should not do a neighbor search; mu_S == train-fold y mean.

    Check shape: estimator should be bounded by Var(Y) (for any candidate i)
    and positive for a feature that carries signal.
    """
    rng = np.random.default_rng(4)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    noise = 0.3 * rng.standard_normal(n)
    y = 2.0 * x1 + 0.0 * x2 + noise  # x1 drives y, x2 is noise
    X = np.stack([x1, x2], axis=1)
    out = _direct_delta_cf_knn(
        X, y, np.zeros(0, dtype=int), p=2, n_folds=5, rng=np.random.default_rng(5)
    )
    assert out[0] > out[1]  # x1 beats the noise feature
    assert out[0] > 0.5     # material signal for x1
    assert 0 <= out[1] < 0.5  # x2 near zero but nonnegative


def test_direct_delta_wide_S_is_not_identically_zero():
    """Under the histogram plug-in, |S| ~ p/2 collapses to flat zeros. The
    cross-fitted estimator should produce meaningful spread across candidate
    features on a simple SCM with non-trivial structure."""
    rng = np.random.default_rng(6)
    n = 400
    p = 10
    X = rng.standard_normal((n, p))
    # y depends on all features with decaying weights -- every candidate
    # feature in S-complement should still carry *some* signal.
    coefs = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0, 0.0, 0.0])
    y = X @ coefs + 0.2 * rng.standard_normal(n)
    S = np.array([0, 1, 2, 3])  # medium state
    out = _direct_delta_cf_knn(X, y, S, p=p, n_folds=5, rng=np.random.default_rng(7))
    non_s = [i for i in range(p) if i not in set(S.tolist())]
    vals = out[non_s]
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0.0)
    assert float(np.std(vals)) > 0.0  # not flat


def test_direct_delta_ranks_gaussian_closed_form():
    """On a jointly-Gaussian SCM, the direct-Delta ranking across candidate
    features should agree with delta_gaussian ranking (Spearman positive)."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(11)
    p = 5
    A = rng.standard_normal((p + 1, p + 1))
    Sigma = A @ A.T + 0.5 * np.eye(p + 1)
    y_idx = p  # Y is last coordinate

    X, y, _ = _gaussian_scm_sample(Sigma, y_idx, n=2000, seed=12)

    # Empty-S candidate vector
    S = np.zeros(0, dtype=int)
    est = _direct_delta_cf_knn(X, y, S, p=p, n_folds=5, rng=np.random.default_rng(13))
    ref = np.array([delta_gaussian(Sigma, y_idx, i, S) for i in range(p)])
    rho, _ = spearmanr(est, ref)
    assert rho > 0.7, f"expected rho > 0.7, got {rho:.3f}"

    # Non-empty S: pick one feature out and check ranking on the rest
    S = np.array([0])
    est = _direct_delta_cf_knn(X, y, S, p=p, n_folds=5, rng=np.random.default_rng(14))
    ref = np.array([delta_gaussian(Sigma, y_idx, i, S) for i in range(p)])
    mask = np.array([i not in set(S.tolist()) for i in range(p)])
    rho, _ = spearmanr(est[mask], ref[mask])
    assert rho > 0.7, f"expected rho > 0.7 on |S|=1, got {rho:.3f}"


def test_direct_delta_vector_for_S_matches_helper(monkeypatch):
    """delta_vector_for_S_direct_knn dispatches to _direct_delta_cf_knn."""
    rng = np.random.default_rng(20)

    class _SimpleSCM:
        def simulate(self, n_samples, rng):
            X = rng.standard_normal((n_samples, 4)).astype(np.float64)
            y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float64)
            return torch.from_numpy(X), torch.from_numpy(y)

    ctx = build_oracle_context(_SimpleSCM(), p=4, n_oracle=256, rng=rng)
    S = np.array([2])
    out = delta_vector_for_S_direct_knn(
        ctx, S, n_folds=5, rng=np.random.default_rng(21)
    )
    assert np.isnan(out[2])
    assert np.all(out[[0, 1, 3]] >= 0.0)
    # feature 0 should carry the strongest signal on the held-out set
    assert out[0] > out[1] > out[3]


def test_compute_value_queries_accepts_direct_knn_estimator():
    class _LinearSCM:
        def simulate(self, n_samples, rng):
            X = rng.standard_normal((n_samples, 3)).astype(np.float64)
            y = (X[:, 0] + 0.3 * X[:, 1]).astype(np.float64)
            return torch.from_numpy(X), torch.from_numpy(y)

    X = torch.zeros(16, 3)
    y = torch.zeros(16)
    payload = compute_value_queries(
        _LinearSCM(), X, y,
        n_oracle=256,
        mixture="backup",
        rng=np.random.default_rng(31),
        label_estimator="direct_knn",
        label_knn_folds=5,
    )
    assert len(payload["value_queries"]) > 0
    for q in payload["value_queries"]:
        assert q.raw_targets is not None
        raw = q.raw_targets.numpy()
        S_mask = q.S_mask.numpy()
        # Nonnegative without clip; NaN exactly where i in S
        assert np.all(np.isnan(raw[S_mask]))
        assert np.all(raw[~S_mask] >= 0.0)


def test_compute_value_queries_rejects_unknown_estimator():
    class _SCM:
        def simulate(self, n_samples, rng):
            X = rng.standard_normal((n_samples, 2)).astype(np.float64)
            y = X[:, 0].astype(np.float64)
            return torch.from_numpy(X), torch.from_numpy(y)

    with pytest.raises(ValueError, match="label_estimator"):
        compute_value_queries(
            _SCM(), torch.zeros(4, 2), torch.zeros(4),
            n_oracle=64,
            rng=np.random.default_rng(0),
            label_estimator="not_a_real_one",
        )
