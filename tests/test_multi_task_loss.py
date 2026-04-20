"""Phase 4 — multi-task loss unit tests.

Covers the six invariants from the Phase 4 plan:
  1. Uncertainty weighting: grads flow everywhere, including log_sigma.
  2. is_identifiable mask zeros Head I contribution.
  3. NaN-padded labels past `d` are ignored.
  4. Varied per-sample c_triple counts are handled.
  5. Manual lambdas path applies supplied scalars.
  6. Consistency penalty couples Head A and Head C.

All tests run on CPU, seeded, total runtime < 20 s.
"""
from __future__ import annotations

import math

import pytest
import torch

from tabicl.model.heads import ConditionalHead, InterventionalHead, ObservationalHead
from tabicl.train.multi_task_loss import MultiTaskLoss


def _build_heads_and_loss(E: int = 16, **kwargs) -> MultiTaskLoss:
    torch.manual_seed(0)
    head_a = ObservationalHead(embed_dim=E)
    head_i = InterventionalHead(embed_dim=E)
    head_c = ConditionalHead(embed_dim=E)
    return MultiTaskLoss(head_a=head_a, head_i=head_i, head_c=head_c, **kwargs)


def _fake_batch(B: int = 3, H: int = 4, E: int = 16, T: int = 5, C: int = 3):
    torch.manual_seed(1)
    logits = torch.randn(B, T, C)
    y_true = torch.randint(0, C, (B, T))
    col_emb = torch.randn(B, H, E, requires_grad=True)
    o_star = torch.rand(B, H)
    i_star = torch.rand(B, H)
    is_id = torch.tensor([True, True, True])
    c_triples = [
        [(0, torch.tensor([False, True, False, False]), 0.1)],
        [(1, torch.tensor([True, False, False, False]), 0.2),
         (2, torch.tensor([True, True, False, False]), 0.05)],
        [(0, torch.tensor([False, False, True, False]), 0.15)],
    ]
    d = torch.tensor([H, H, H])
    return logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d


# ---------------------------------------------------------------------------
# Test 1 — uncertainty weighting end-to-end
# ---------------------------------------------------------------------------


def test_uncertainty_weighting_shapes_and_grad():
    """Loss is a scalar; backward() populates grads on trunk, heads, log_sigma."""
    loss_mod = _build_heads_and_loss(weighting="uncertainty")
    logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d = _fake_batch()

    total, bd = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d)
    assert total.ndim == 0
    assert math.isfinite(total.item())
    total.backward()

    # col_emb (proxy for trunk) gets a gradient.
    assert col_emb.grad is not None and torch.isfinite(col_emb.grad).all()

    # Every parameter under the loss module has a gradient (heads + log_sigma).
    for name, p in loss_mod.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name

    # Reasonable breakdown values.
    assert bd.pred > 0 and bd.A >= 0 and bd.I >= 0 and bd.C >= 0


# ---------------------------------------------------------------------------
# Test 2 — is_identifiable mask zeros Head I
# ---------------------------------------------------------------------------


def test_is_identifiable_mask_zeros_head_i():
    """With is_id all False, Head I loss is zero."""
    loss_mod = _build_heads_and_loss(weighting="manual",
                                     lambdas={"pred": 0.0, "obs": 0.0, "int": 1.0, "cond": 0.0})
    logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d = _fake_batch()
    is_id_all_false = torch.zeros(is_id.shape, dtype=torch.bool)

    total_false, bd_false = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id_all_false, c_triples, d)
    assert bd_false.I == 0.0

    # With is_id all True the Head I loss is positive (not coincidentally zero).
    total_true, bd_true = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d)
    assert bd_true.I > 0.0


# ---------------------------------------------------------------------------
# Test 3 — NaN-padded labels past `d` are ignored
# ---------------------------------------------------------------------------


def test_nan_padded_labels_ignored():
    """NaN values past the active-feature boundary don't poison the loss."""
    loss_mod = _build_heads_and_loss(weighting="uncertainty")
    logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d = _fake_batch()
    # Mark one sample as having only 2 active features; NaN-pad the rest.
    d = torch.tensor([2, 4, 4])
    o_star[0, 2:] = float("nan")
    i_star[0, 2:] = float("nan")

    total, bd = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d)
    assert math.isfinite(total.item())
    assert math.isfinite(bd.A) and math.isfinite(bd.I)


# ---------------------------------------------------------------------------
# Test 4 — per-sample triple counts can differ, including empty
# ---------------------------------------------------------------------------


def test_c_triples_varied_lengths():
    """Samples with 0 / 1 / k triples all handled cleanly."""
    loss_mod = _build_heads_and_loss(weighting="uncertainty")
    logits, y_true, col_emb, o_star, i_star, is_id, _c, d = _fake_batch()
    c_triples = [
        [],  # empty
        [(0, torch.tensor([False, True, False, False]), 0.1)],  # one
        [  # several
            (0, torch.tensor([False, True, False, False]), 0.1),
            (1, torch.tensor([True, False, False, False]), 0.2),
            (2, torch.tensor([True, True, False, False]), 0.05),
        ],
    ]
    total, bd = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d)
    assert math.isfinite(total.item())
    assert bd.C >= 0

    # All-empty triples: C loss is exactly 0.
    c_all_empty = [[], [], []]
    _, bd0 = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_all_empty, d)
    assert bd0.C == 0.0


# ---------------------------------------------------------------------------
# Test 5 — manual lambda override
# ---------------------------------------------------------------------------


def test_manual_lambdas_override():
    """'manual' weighting uses user-supplied lambdas and no log_sigma."""
    loss_mod = _build_heads_and_loss(
        weighting="manual",
        lambdas={"pred": 2.0, "obs": 0.5, "int": 0.25, "cond": 0.125},
    )
    assert not hasattr(loss_mod, "log_sigma2") or "log_sigma2" not in dict(loss_mod.named_parameters())

    logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d = _fake_batch()
    total, bd = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d)

    assert bd.w_pred == 2.0
    assert bd.w_A == 0.5
    assert bd.w_I == 0.25
    assert bd.w_C == 0.125

    # Total approximately equals the weighted sum (no uncertainty reg term).
    expected = 2.0 * bd.pred + 0.5 * bd.A + 0.25 * bd.I + 0.125 * bd.C
    assert math.isclose(total.item(), expected, rel_tol=1e-4, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# Test 6 — consistency penalty couples Head A and Head C
# ---------------------------------------------------------------------------


def test_consistency_penalty_when_enabled():
    """With cons_weight>0, perturbing head_c weights changes the total."""
    loss_mod = _build_heads_and_loss(
        weighting="manual",
        lambdas={"pred": 0.0, "obs": 0.0, "int": 0.0, "cond": 0.0},
        head_c_consistency_weight=1.0,
    )
    logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d = _fake_batch()

    total_before, bd_before = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d)
    assert bd_before.cons > 0

    with torch.no_grad():
        loss_mod.head_c.fc2.weight.add_(0.5)
    total_after, bd_after = loss_mod(logits, y_true, col_emb, o_star, i_star, is_id, c_triples, d)
    assert not math.isclose(total_before.item(), total_after.item(), rel_tol=1e-6)

    # Consistency penalty backprops into both head_a and head_c.
    loss_mod.zero_grad()
    total_after.backward()
    assert loss_mod.head_a.mlp.fc2.weight.grad is not None
    assert loss_mod.head_c.fc2.weight.grad is not None
