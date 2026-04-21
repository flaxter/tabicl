"""Phase 6a — end-to-end attribution-quality harness.

Runs :class:`tabicl.TabICLExplainer` across a suite of datasets with
known ground-truth attribution labels (synthetic DGPs from the Phase 3
prior families or bespoke SCMs) and reports per-dataset and aggregate
attribution metrics for Heads A, I, C via the Phase 5 sklearn surface.

Companion to :mod:`tabicl.eval.eval_heads`:

- ``eval_heads`` measures **label fidelity** on the Phase 3 prior stream
  via batched trunk forward passes (Phase 6e).
- ``explainer_eval`` (this module) measures **end-to-end attribution
  quality** via the full Phase 5 sklearn surface (Phase 6a-d).

Notes
-----
Ground-truth labels are the same structural-equation quantities used to
train the heads (``tabicl.prior.labels.compute_labels``). For bespoke
suites (collider, identifiability-boundary) labels are computed
analytically from the joint-covariance formulation.

Head C scoring is on a fixed ``k_cond_triples`` set sampled by
``compute_labels``; an additional per-dataset "S = -i" Spearman check
cross-validates the A/C boundary equivalence from PLAN §Phase 4.

Attribution magnitudes shift under the base estimator's feature
normalisation — the ``mse_C`` / ``mae_C`` columns are recorded but
interpretation should lead with Spearman/Pearson/top-k.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch

from tabicl.eval.metrics import (
    HeadMetrics,
    aggregate_metrics,
    spearman_per_dataset,
)
from tabicl.prior.identifiable_scm import ANMSCM, LiNGAMSCM, NoiseDist, TreeSCM_Ident
from tabicl.prior.labels import compute_labels


# ---------------------------------------------------------------------------
# Dataclasses — result schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    """Per-dataset attribution ground truth."""

    observational: np.ndarray
    """Head A target of shape ``(p,)`` in ``Var(y)`` units."""

    interventional: Optional[np.ndarray]
    """Head I target of shape ``(p,)`` in y-units. ``None`` on non-identifiable DGPs."""

    conditional_triples: List[tuple]
    """Head C ground truth as ``[(i, S_frozenset, c_star), ...]``."""

    is_identifiable: bool


@dataclass(frozen=True)
class EvalCase:
    dataset_id: str
    X: np.ndarray
    y: np.ndarray
    ground_truth: GroundTruth


@dataclass(frozen=True)
class EvalSuite:
    name: str
    cases: Sequence[EvalCase]


@dataclass
class DatasetScore:
    """One CSV row per evaluated dataset."""

    suite: str
    dataset_id: str
    n: int
    p: int
    is_identifiable: bool
    spearman_A: float
    pearson_A: float
    top1_A: float
    top3_A: float
    top5_A: float
    spearman_I: float
    pearson_I: float
    top1_I: float
    top3_I: float
    top5_I: float
    mse_C: float
    mae_C: float
    n_triples_C: int
    spearman_C_at_minus_i: float
    fit_seconds: float


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------


def evaluate_explainer(
    explainer_factory: Callable[[], Any],
    suite: EvalSuite,
    *,
    out_csv: Optional[str | Path] = None,
    verbose: bool = False,
) -> List[DatasetScore]:
    """Fit a fresh ``TabICLExplainer`` on every case in ``suite`` and score it.

    The factory returns an **un-fitted** explainer per call so each run
    is independent. Use a closure to share a heads checkpoint:

        factory = lambda: TabICLExplainer(
            base_estimator=TabICLRegressor(n_estimators=4),
            heads_checkpoint_path=ckpt,
        )
        evaluate_explainer(factory, build_in_distribution_suite(50, seed=0))

    Returns the same ``DatasetScore`` list it writes to ``out_csv``.
    """
    scores: List[DatasetScore] = []
    for case in suite.cases:
        explainer = explainer_factory()
        t0 = time.perf_counter()
        explainer.fit(case.X, case.y)
        fit_seconds = time.perf_counter() - t0
        score = _score_one(suite.name, case, explainer, fit_seconds)
        scores.append(score)
        if verbose:
            print(
                f"[{suite.name}] {case.dataset_id}: "
                f"spearman_A={score.spearman_A:.3f} "
                f"spearman_I={score.spearman_I:.3f} "
                f"mse_C={score.mse_C:.4f} "
                f"({fit_seconds:.2f}s)"
            )
    if out_csv is not None:
        write_scores_csv(out_csv, scores)
    return scores


def _score_one(
    suite_name: str, case: EvalCase, explainer: Any, fit_seconds: float
) -> DatasetScore:
    p = int(case.X.shape[1])
    n = int(case.X.shape[0])
    gt = case.ground_truth

    pred_A = np.asarray(explainer.observational_relevance_, dtype=np.float64)
    target_A = np.asarray(gt.observational, dtype=np.float64)
    metrics_A = aggregate_metrics(pred_A[None, :], target_A[None, :])

    if gt.interventional is None:
        metrics_I = HeadMetrics(
            spearman=float("nan"),
            pearson=float("nan"),
            top1=float("nan"),
            top3=float("nan"),
            top5=float("nan"),
            n_valid=0,
        )
    else:
        pred_I = np.asarray(explainer.interventional_effects_, dtype=np.float64)
        target_I = np.asarray(gt.interventional, dtype=np.float64)
        metrics_I = aggregate_metrics(pred_I[None, :], target_I[None, :])

    mse_C, mae_C, n_triples_used = _score_head_c(explainer, gt.conditional_triples)
    sp_minus_i = _score_head_c_at_minus_i(explainer, gt, p)

    return DatasetScore(
        suite=suite_name,
        dataset_id=case.dataset_id,
        n=n,
        p=p,
        is_identifiable=bool(gt.is_identifiable),
        spearman_A=metrics_A.spearman,
        pearson_A=metrics_A.pearson,
        top1_A=metrics_A.top1,
        top3_A=metrics_A.top3,
        top5_A=metrics_A.top5,
        spearman_I=metrics_I.spearman,
        pearson_I=metrics_I.pearson,
        top1_I=metrics_I.top1,
        top3_I=metrics_I.top3,
        top5_I=metrics_I.top5,
        mse_C=mse_C,
        mae_C=mae_C,
        n_triples_C=n_triples_used,
        spearman_C_at_minus_i=sp_minus_i,
        fit_seconds=float(fit_seconds),
    )


def _score_head_c(
    explainer: Any, triples: Sequence[tuple]
) -> tuple[float, float, int]:
    preds: List[float] = []
    trues: List[float] = []
    for (i, S_frozen, c_star) in triples:
        vec = explainer.marginal_conditional_contributions(list(S_frozen))
        val = float(vec[int(i)])
        if not np.isfinite(val):
            continue
        preds.append(val)
        trues.append(float(c_star))
    if not preds:
        return float("nan"), float("nan"), 0
    diff = np.asarray(preds) - np.asarray(trues)
    return float((diff ** 2).mean()), float(np.abs(diff).mean()), len(preds)


def _score_head_c_at_minus_i(explainer: Any, gt: GroundTruth, p: int) -> float:
    if p < 3:
        return float("nan")
    pred_vec = np.full(p, np.nan, dtype=np.float64)
    all_idx = list(range(p))
    for i in range(p):
        S_minus_i = [j for j in all_idx if j != i]
        vec = explainer.marginal_conditional_contributions(S_minus_i)
        pred_vec[i] = float(vec[i])
    target = np.asarray(gt.observational, dtype=np.float64)
    return spearman_per_dataset(pred_vec, target)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


_CSV_COLS = list(DatasetScore.__dataclass_fields__.keys())
_AGGREGATE_COLS = _CSV_COLS[5:]  # skip suite/dataset_id/n/p/is_identifiable


def write_scores_csv(
    path: str | Path, scores: Sequence[DatasetScore]
) -> None:
    """Write per-dataset rows plus a trailing ``mean`` aggregate row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not scores:
        raise ValueError("No scores to write; did the eval loop produce any rows?")

    rows = [asdict(s) for s in scores]
    mean_row: dict = {
        "suite": rows[0]["suite"],
        "dataset_id": "mean",
        "n": int(sum(r["n"] for r in rows)),
        "p": "",
        "is_identifiable": "",
    }
    for col in _AGGREGATE_COLS:
        vals = [r[col] for r in rows if _finite_numeric(r[col])]
        mean_row[col] = float(np.mean(vals)) if vals else float("nan")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow(mean_row)


def _finite_numeric(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


# ---------------------------------------------------------------------------
# Ground-truth adapters
# ---------------------------------------------------------------------------


def _labels_dict_to_ground_truth(label_dict: dict) -> GroundTruth:
    """Translate ``tabicl.prior.labels.compute_labels`` output to ``GroundTruth``."""
    o_star = np.asarray(label_dict["o_star"].cpu().numpy(), dtype=np.float64)
    i_star_np = np.asarray(label_dict["i_star"].cpu().numpy(), dtype=np.float64)
    is_id = bool(label_dict["is_identifiable"])
    triples: List[tuple] = []
    for (i, S_mask, c_star) in label_dict["c_triples"]:
        S_np = S_mask.cpu().numpy() if hasattr(S_mask, "cpu") else np.asarray(S_mask)
        S_idx = tuple(int(j) for j in np.flatnonzero(S_np))
        triples.append((int(i), frozenset(S_idx), float(c_star)))
    interv: Optional[np.ndarray]
    interv = i_star_np if is_id and np.isfinite(i_star_np).any() else None
    return GroundTruth(
        observational=o_star,
        interventional=interv,
        conditional_triples=triples,
        is_identifiable=is_id,
    )


# ---------------------------------------------------------------------------
# Suite builders
# ---------------------------------------------------------------------------


def build_in_distribution_suite(
    n_datasets: int,
    seed: int,
    *,
    n_rows: int = 500,
    min_features: int = 5,
    max_features: int = 10,
    n_mc: int = 512,
    k_cond_triples: int = 8,
) -> EvalSuite:
    """In-distribution: LiNGAM / ANM / TreeSCM_Ident — same families as training.

    Rotates through the three identifiable SCM families in turn. All
    three heads have defined labels; ``is_identifiable=True``.
    """
    rng = np.random.default_rng(seed)
    cases: List[EvalCase] = []
    kinds = ("lingam", "anm", "tree_ident")
    for k in range(n_datasets):
        p = int(rng.integers(min_features, max_features + 1))
        kind = kinds[k % len(kinds)]
        scm_seed = int(rng.integers(0, 2**31 - 1))
        scm = _build_identifiable_scm(kind, seq_len=n_rows, p=p, seed=scm_seed)
        X_t, y_t = scm()
        labels = compute_labels(
            scm,
            X_t,
            y_t,
            n_mc=n_mc,
            k_cond_triples=k_cond_triples,
            rng=np.random.default_rng(scm_seed ^ 0xA5B4C3),
        )
        gt = _labels_dict_to_ground_truth(labels)
        cases.append(
            EvalCase(
                dataset_id=f"in_dist_{kind}_{k:04d}",
                X=X_t.cpu().numpy().astype(np.float32),
                y=y_t.cpu().numpy().astype(np.float32),
                ground_truth=gt,
            )
        )
    return EvalSuite(name="in_distribution", cases=cases)


def _build_identifiable_scm(
    kind: str, *, seq_len: int, p: int, seed: int
):
    if kind == "lingam":
        return LiNGAMSCM(seq_len=seq_len, num_features=p, seed=seed)
    if kind == "anm":
        return ANMSCM(seq_len=seq_len, num_features=p, seed=seed)
    if kind == "tree_ident":
        return TreeSCM_Ident(seq_len=seq_len, num_features=p, seed=seed)
    raise ValueError(f"Unknown identifiable-SCM kind: {kind!r}")


def build_held_out_prior_suite(
    n_datasets: int,
    seed: int,
    *,
    n_rows: int = 500,
    min_features: int = 5,
    max_features: int = 10,
    n_mc: int = 512,
    k_cond_triples: int = 8,
) -> EvalSuite:
    """Held-out prior: MLPSCM — non-identifiable, Head I ground-truth is ``None``.

    Exercises the "does this generalise beyond the identifiable subfamily
    the heads were trained on?" question. Head A and Head C targets are
    still well-defined (MC plug-in estimator from ``compute_labels``'s
    non-identifiable fallback path); Head I is honestly reported as
    unavailable per PLAN §6a "Held-out prior families".
    """
    # Local import — MLPSCM pulls in torch / sklearn at module load.
    from tabicl.prior.mlp_scm import MLPSCM

    rng = np.random.default_rng(seed)
    cases: List[EvalCase] = []
    for k in range(n_datasets):
        p = int(rng.integers(min_features, max_features + 1))
        scm_seed = int(rng.integers(0, 2**31 - 1))
        scm = MLPSCM(
            seq_len=n_rows,
            num_features=p,
            num_outputs=1,
            seed=scm_seed,
            device="cpu",
        )
        with torch.no_grad():
            X_t, y_t = scm()
        labels = compute_labels(
            scm,
            X_t,
            y_t,
            n_mc=n_mc,
            k_cond_triples=k_cond_triples,
            rng=np.random.default_rng(scm_seed ^ 0xB7C6D5),
        )
        gt = _labels_dict_to_ground_truth(labels)
        cases.append(
            EvalCase(
                dataset_id=f"held_out_mlp_{k:04d}",
                X=X_t.cpu().numpy().astype(np.float32),
                y=y_t.cpu().numpy().astype(np.float32),
                ground_truth=gt,
            )
        )
    return EvalSuite(name="held_out_prior", cases=cases)


def build_collider_suite(
    n_datasets: int,
    seed: int,
    *,
    n_rows: int = 500,
    n_parents: int = 2,
    n_independent: int = 1,
    n_downstream: int = 2,
    k_cond_triples: int = 8,
) -> EvalSuite:
    """Collider-rich: features downstream of y — **headline figure for §6.1**.

    Each DGP has ``n_parents`` true causes of ``y``, ``n_independent``
    noise features, and ``n_downstream`` features caused by ``y``
    (``X_d = alpha_d * y + eps_d``). Ground truth:

    - Head A (observational): large for parents and for downstream
      features (both carry info about ``y`` through the joint
      distribution); ≈0 for independent noise features.
    - Head I (interventional): ``|beta_i|`` for parents only; ``0`` for
      independent and downstream features (intervening on a descendant
      of y does not change y).

    The Head I contrast on downstream features is the paper's headline
    rhetorical move — SHAP-on-TabICL will inflate those features'
    relevance; Head I should zero them.
    """
    rng = np.random.default_rng(seed)
    cases: List[EvalCase] = []
    for k in range(n_datasets):
        case_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        beta = case_rng.uniform(0.5, 1.5, size=n_parents) * case_rng.choice(
            [-1.0, 1.0], size=n_parents
        )
        sigma_y = float(case_rng.uniform(0.3, 0.8))
        alpha = case_rng.uniform(0.5, 1.5, size=n_downstream) * case_rng.choice(
            [-1.0, 1.0], size=n_downstream
        )
        sigma_d = case_rng.uniform(0.2, 0.6, size=n_downstream)
        X, y = _simulate_collider_dgp(
            n_rows, beta, sigma_y, alpha, sigma_d, n_independent, case_rng
        )
        gt = _collider_ground_truth(
            beta=beta,
            sigma_y=sigma_y,
            alpha=alpha,
            sigma_d=sigma_d,
            n_independent=n_independent,
            k_cond_triples=k_cond_triples,
            rng=case_rng,
        )
        cases.append(
            EvalCase(
                dataset_id=f"collider_{k:04d}",
                X=X.astype(np.float32),
                y=y.astype(np.float32),
                ground_truth=gt,
            )
        )
    return EvalSuite(name="collider", cases=cases)


def _simulate_collider_dgp(
    n: int,
    beta: np.ndarray,
    sigma_y: float,
    alpha: np.ndarray,
    sigma_d: np.ndarray,
    n_independent: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_parents = beta.shape[0]
    n_downstream = alpha.shape[0]
    X_parents = rng.standard_normal((n, n_parents))
    X_indep = rng.standard_normal((n, n_independent))
    y = X_parents @ beta + rng.normal(scale=sigma_y, size=n)
    X_down = alpha[None, :] * y[:, None] + rng.normal(scale=sigma_d, size=(n, n_downstream))
    X = np.concatenate([X_parents, X_indep, X_down], axis=1)
    return X, y


def _collider_ground_truth(
    *,
    beta: np.ndarray,
    sigma_y: float,
    alpha: np.ndarray,
    sigma_d: np.ndarray,
    n_independent: int,
    k_cond_triples: int,
    rng: np.random.Generator,
) -> GroundTruth:
    """Analytical Heads A/I/C from the joint covariance of a y→X_d linear-Gaussian DGP."""
    n_parents = beta.shape[0]
    n_downstream = alpha.shape[0]
    p = n_parents + n_independent + n_downstream
    parent_idx = np.arange(n_parents)
    indep_idx = np.arange(n_parents, n_parents + n_independent)
    down_idx = np.arange(n_parents + n_independent, p)

    sig_XX, sig_Xy, var_y = _collider_joint_covariance(
        beta, sigma_y, alpha, sigma_d, n_independent
    )

    o_star = np.zeros(p, dtype=np.float64)
    full_idx = np.arange(p)
    expl_full = _explained_var_lin(sig_XX, sig_Xy, full_idx)
    for i in range(p):
        excl = np.array([j for j in full_idx if j != i])
        expl_excl = _explained_var_lin(sig_XX, sig_Xy, excl)
        o_star[i] = max(0.0, expl_full - expl_excl)

    i_star = np.zeros(p, dtype=np.float64)
    i_star[parent_idx] = np.abs(beta)  # Var(X_parent)=1
    # independent and downstream features already 0

    triples: List[tuple] = []
    for i, S_mask in _sample_c_masks(p, k_cond_triples, rng):
        S_idx = np.flatnonzero(S_mask)
        S_plus = np.concatenate([S_idx, np.array([i])])
        gain = max(
            0.0,
            _explained_var_lin(sig_XX, sig_Xy, S_plus)
            - _explained_var_lin(sig_XX, sig_Xy, S_idx),
        )
        triples.append((int(i), frozenset(int(j) for j in S_idx), float(gain)))

    return GroundTruth(
        observational=o_star,
        interventional=i_star,
        conditional_triples=triples,
        is_identifiable=True,
    )


def _collider_joint_covariance(
    beta: np.ndarray,
    sigma_y: float,
    alpha: np.ndarray,
    sigma_d: np.ndarray,
    n_independent: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Joint ``(Sigma_XX, Sigma_Xy, Var(y))`` for the collider DGP."""
    n_parents = beta.shape[0]
    n_downstream = alpha.shape[0]
    p = n_parents + n_independent + n_downstream
    var_y = float(beta @ beta + sigma_y ** 2)

    sig_XX = np.zeros((p, p), dtype=np.float64)
    sig_Xy = np.zeros(p, dtype=np.float64)

    parent_slc = slice(0, n_parents)
    indep_slc = slice(n_parents, n_parents + n_independent)
    down_slc = slice(n_parents + n_independent, p)

    sig_XX[parent_slc, parent_slc] = np.eye(n_parents)
    sig_XX[indep_slc, indep_slc] = np.eye(n_independent)

    down_cov = var_y * np.outer(alpha, alpha) + np.diag(sigma_d ** 2)
    sig_XX[down_slc, down_slc] = down_cov

    cov_parent_down = np.outer(beta, alpha)  # (n_parents, n_downstream)
    sig_XX[parent_slc, down_slc] = cov_parent_down
    sig_XX[down_slc, parent_slc] = cov_parent_down.T

    sig_Xy[parent_slc] = beta
    sig_Xy[indep_slc] = 0.0
    sig_Xy[down_slc] = alpha * var_y

    return sig_XX, sig_Xy, var_y


def _explained_var_lin(
    sig_XX: np.ndarray, sig_Xy: np.ndarray, T_idx: np.ndarray
) -> float:
    """``Var(E[y | X_T])`` for the best-linear predictor of y given X_T.

    Mirrors the private helper in ``tabicl.prior.labels`` — kept
    standalone here so this module doesn't reach into underscore-private
    API of the prior package.
    """
    if T_idx.size == 0:
        return 0.0
    sig_TT = sig_XX[np.ix_(T_idx, T_idx)]
    sig_yT = sig_Xy[T_idx]
    try:
        sol = np.linalg.solve(sig_TT + 1e-10 * np.eye(T_idx.size), sig_yT)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(sig_TT, sig_yT, rcond=None)[0]
    return float(sig_yT @ sol)


def _sample_c_masks(
    p: int, k: int, rng: np.random.Generator
) -> List[tuple[int, np.ndarray]]:
    """Sample ``k`` ``(i, S_mask)`` pairs with ``i`` not in ``S``."""
    out: List[tuple[int, np.ndarray]] = []
    if p <= 0:
        return out
    for _ in range(k):
        i = int(rng.integers(0, p))
        others = np.array([j for j in range(p) if j != i], dtype=int)
        S_mask = np.zeros(p, dtype=bool)
        if others.size > 0:
            size = int(rng.integers(0, others.size + 1))
            if size > 0:
                S = rng.choice(others, size=size, replace=False)
                S_mask[S] = True
        out.append((i, S_mask))
    return out


def build_id_boundary_suite(
    n_datasets: int,
    seed: int,
    *,
    n_rows: int = 500,
    min_features: int = 5,
    max_features: int = 10,
    n_mc: int = 512,
    k_cond_triples: int = 8,
) -> EvalSuite:
    """Identifiability boundary: half LiNGAM (identifiable), half symmetric
    linear-Gaussian (not identifiable for Head I).

    The paired suite supports the Honest Scoping figure from PLAN §6a:
    Head I should recover true effects on the identifiable half and
    degrade visibly on the non-identifiable half. Same-seed pairs are
    generated so per-DGP variance can be partitioned.
    """
    rng = np.random.default_rng(seed)
    cases: List[EvalCase] = []
    for k in range(n_datasets):
        p = int(rng.integers(min_features, max_features + 1))
        scm_seed = int(rng.integers(0, 2**31 - 1))
        identifiable = (k % 2 == 0)
        scm = _build_boundary_scm(
            identifiable=identifiable, seq_len=n_rows, p=p, seed=scm_seed
        )
        X_t, y_t = scm()
        labels = compute_labels(
            scm,
            X_t,
            y_t,
            n_mc=n_mc,
            k_cond_triples=k_cond_triples,
            rng=np.random.default_rng(scm_seed ^ 0xC5D6E7),
        )
        gt = _labels_dict_to_ground_truth(labels)
        # LiNGAMSCM always reports ``is_identifiable=True`` regardless of
        # noise kind, but our boundary construction breaks identifiability
        # by flipping the noise to Gaussian. Override the tag so downstream
        # aggregation can partition the two halves; the labels themselves
        # remain the true closed-form structural quantities.
        gt = GroundTruth(
            observational=gt.observational,
            interventional=gt.interventional,
            conditional_triples=gt.conditional_triples,
            is_identifiable=identifiable,
        )
        suffix = "id" if identifiable else "nonid"
        cases.append(
            EvalCase(
                dataset_id=f"boundary_{suffix}_{k:04d}",
                X=X_t.cpu().numpy().astype(np.float32),
                y=y_t.cpu().numpy().astype(np.float32),
                ground_truth=gt,
            )
        )
    return EvalSuite(name="id_boundary", cases=cases)


def _build_boundary_scm(
    *, identifiable: bool, seq_len: int, p: int, seed: int
) -> LiNGAMSCM:
    """Build a LiNGAM-style SCM; flip all noise to Gaussian to erase identifiability.

    Structurally identical adjacency + beta so the true structural
    effects are matched; only the *noise distribution* changes. Under
    symmetric Gaussian noise the DGP is observationally indistinguishable
    from its reversed-arrow counterparts, so Head I has no hope of
    recovering the structural coefficients from data alone — exactly the
    "non-identifiable half" the figure wants to show.
    """
    scm = LiNGAMSCM(seq_len=seq_len, num_features=p, seed=seed)
    if not identifiable:
        rng = np.random.default_rng(seed ^ 0xFACE)
        scm.noise_dists_x = [
            NoiseDist(kind="normal", scale=float(rng.uniform(0.3, 1.2)))
            for _ in range(p)
        ]
        scm.noise_dist_y = NoiseDist(
            kind="normal", scale=float(rng.uniform(0.3, 1.2))
        )
    return scm


def build_case_study_suite() -> EvalSuite:
    """Real-data case studies — placeholder for §6b.

    Birth-weight paradox, LaLonde observational-vs-experimental, and
    IHDP semi-synthetic. Data loaders and pre-registered ground truth
    land with Phase 6b; this harness already supports the resulting
    ``EvalCase`` shape.
    """
    raise NotImplementedError(
        "Case-study suite lives in Phase 6b (notes/preregistration.md). "
        "Implement loaders for CDC natality 2018, LaLonde Dehejia-Wahba, "
        "and IHDP response-surface A, then return an EvalSuite with one "
        "case per dataset."
    )
