"""Tests for PredictiveValueLoss (pred CE + value Huber)."""
from __future__ import annotations

from typing import List

import pytest
import torch

from tabicl.model.heads import ConditionalPredictiveValueHead
from tabicl.prior.labels import ValueQuery
from tabicl.train.multi_task_loss import LossBreakdown, PredictiveValueLoss


def _build_loss(embed_dim: int = 8, weighting: str = "uncertainty") -> PredictiveValueLoss:
    torch.manual_seed(0)
    head = ConditionalPredictiveValueHead(embed_dim=embed_dim)
    return PredictiveValueLoss(value_head=head, weighting=weighting)


def _make_labels(B: int, p: int, n_queries: int = 2) -> List[dict]:
    """Construct a batch of synthetic labels for smoke-testing."""
    labels: List[dict] = []
    for b in range(B):
        qs: List[ValueQuery] = []
        for q in range(n_queries):
            S_mask = torch.zeros(p, dtype=torch.bool)
            if q == 1:
                S_mask[0] = True  # singleton S = {0}
            targets = torch.full((p,), 0.5, dtype=torch.float)
            if q == 1:
                targets[0] = float("nan")  # NaN inside S
            qs.append(
                ValueQuery(
                    S_mask=S_mask,
                    targets=targets,
                    raw_targets=None,
                    query_type="empty" if q == 0 else "singleton",
                )
            )
        labels.append({"value_queries": qs, "y_var_raw": 1.0, "label_scale": "rms_y_units"})
    return labels


def test_forward_returns_finite_loss_and_breakdown():
    B, T_test, C, H, E = 2, 3, 4, 5, 8
    loss_fn = _build_loss(embed_dim=E)
    logits = torch.randn(B, T_test, C)
    y_true = torch.randint(0, C, (B, T_test))
    col_emb = torch.randn(B, H, E)
    d = torch.tensor([H, H])
    labels = _make_labels(B, p=H)
    total, bd = loss_fn(logits, y_true, col_emb, labels, d)
    assert torch.isfinite(total)
    assert isinstance(bd, LossBreakdown)
    assert bd.pred >= 0
    assert bd.value >= 0


def test_value_head_gets_gradients():
    B, T_test, C, H, E = 2, 2, 3, 4, 8
    loss_fn = _build_loss(embed_dim=E)
    logits = torch.randn(B, T_test, C, requires_grad=True)
    y_true = torch.randint(0, C, (B, T_test))
    col_emb = torch.randn(B, H, E, requires_grad=True)
    d = torch.tensor([H, H])
    labels = _make_labels(B, p=H)
    total, _ = loss_fn(logits, y_true, col_emb, labels, d)
    total.backward()
    # Every value-head parameter must have a non-None grad with finite values.
    for name, p in loss_fn.value_head.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_no_labels_gives_zero_value_loss():
    B, T_test, C, H, E = 1, 2, 3, 4, 8
    loss_fn = _build_loss(embed_dim=E)
    logits = torch.randn(B, T_test, C)
    y_true = torch.randint(0, C, (B, T_test))
    col_emb = torch.randn(B, H, E)
    d = torch.tensor([H])
    labels = [{"value_queries": [], "y_var_raw": 1.0, "label_scale": "rms_y_units"}]
    total, bd = loss_fn(logits, y_true, col_emb, labels, d)
    assert bd.value == 0.0


def test_nan_targets_excluded_from_loss():
    """Completely NaN targets should produce zero value-loss (nothing to fit)."""
    B, T_test, C, H, E = 1, 2, 3, 4, 8
    loss_fn = _build_loss(embed_dim=E)
    logits = torch.randn(B, T_test, C)
    y_true = torch.randint(0, C, (B, T_test))
    col_emb = torch.randn(B, H, E)
    d = torch.tensor([H])

    all_nan = torch.full((H,), float("nan"))
    query = ValueQuery(
        S_mask=torch.zeros(H, dtype=torch.bool),
        targets=all_nan,
        raw_targets=None,
        query_type="empty",
    )
    labels = [{"value_queries": [query], "y_var_raw": 1.0, "label_scale": "rms_y_units"}]
    _, bd = loss_fn(logits, y_true, col_emb, labels, d)
    assert bd.value == 0.0


def test_strata_diagnostics_populated():
    B, T_test, C, H, E = 2, 2, 3, 4, 8
    loss_fn = _build_loss(embed_dim=E)
    logits = torch.randn(B, T_test, C)
    y_true = torch.randint(0, C, (B, T_test))
    col_emb = torch.randn(B, H, E)
    d = torch.tensor([H, H])
    labels = _make_labels(B, p=H, n_queries=2)
    _, bd = loss_fn(logits, y_true, col_emb, labels, d)
    assert not (bd.value_empty != bd.value_empty), "value_empty stratum should be populated"
    assert not (bd.value_singleton != bd.value_singleton), "value_singleton stratum should be populated"


def test_manual_weighting_respects_lambdas():
    loss_fn = PredictiveValueLoss(
        value_head=ConditionalPredictiveValueHead(embed_dim=8),
        weighting="manual",
        lambdas={"pred": 0.0, "value": 1.0},
    )
    logits = torch.randn(1, 2, 3)
    y_true = torch.randint(0, 3, (1, 2))
    col_emb = torch.randn(1, 4, 8)
    d = torch.tensor([4])
    labels = _make_labels(1, p=4)
    total, bd = loss_fn(logits, y_true, col_emb, labels, d)
    # With lambda_pred=0, total == value * 1 (no regulariser under manual).
    assert abs(total.item() - bd.value) < 1e-5
