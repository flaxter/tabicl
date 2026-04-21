"""Tests for ConditionalPredictiveValueHead."""
from __future__ import annotations

import pytest
import torch

from tabicl.model.heads import ConditionalPredictiveValueHead


def _head(embed_dim: int = 8) -> ConditionalPredictiveValueHead:
    torch.manual_seed(0)
    return ConditionalPredictiveValueHead(embed_dim=embed_dim)


def test_output_shape_is_B_H():
    head = _head()
    B, H, E = 4, 6, 8
    emb = torch.randn(B, H, E)
    mask = torch.zeros(B, H, dtype=torch.bool)
    out = head(emb, mask)
    assert out.shape == (B, H)


def test_empty_S_is_valid():
    """All-False mask must run and produce finite output."""
    head = _head()
    emb = torch.randn(2, 5, 8)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    out = head(emb, mask)
    assert torch.isfinite(out).all()


def test_near_full_S_is_valid():
    """|S| = p - 1 (leave-one-out mask) must run and produce finite output."""
    head = _head()
    emb = torch.randn(3, 6, 8)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, 0] = False  # one "not-in-S" feature per row
    out = head(emb, mask)
    assert torch.isfinite(out).all()


def test_full_S_is_valid():
    """All-True mask must still run. The score at positions inside S is not
    semantically meaningful, but the head must not crash."""
    head = _head()
    emb = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    out = head(emb, mask)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_mask_dtype_must_be_bool():
    head = _head()
    emb = torch.randn(1, 3, 8)
    bad = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(TypeError):
        head(emb, bad)


def test_mask_shape_must_match():
    head = _head()
    emb = torch.randn(1, 3, 8)
    bad = torch.zeros(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        head(emb, bad)


def test_output_depends_on_mask():
    """Same embeddings + different masks => different scores (the head is
    not mask-blind). Tests fusion with e_S."""
    head = _head()
    emb = torch.randn(1, 4, 8)
    mask_empty = torch.zeros(1, 4, dtype=torch.bool)
    mask_singleton = torch.zeros(1, 4, dtype=torch.bool)
    mask_singleton[0, 0] = True
    out_a = head(emb, mask_empty)
    out_b = head(emb, mask_singleton)
    assert not torch.allclose(out_a, out_b)


def test_permutation_equivariance_joint():
    """Permuting columns of emb AND the mask simultaneously permutes the
    output identically. This holds because the sum-pool over S is permutation
    invariant and the per-column MLP is position-agnostic."""
    head = _head()
    torch.manual_seed(1)
    emb = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, False, True, False, False]], dtype=torch.bool)
    out = head(emb, mask)

    perm = torch.tensor([2, 0, 4, 1, 3])
    emb_p = emb[:, perm, :]
    mask_p = mask[:, perm]
    out_p = head(emb_p, mask_p)

    # out_p[j] should equal out[perm[j]]
    assert torch.allclose(out_p, out[:, perm], atol=1e-6)


def test_small_init_keeps_output_near_zero():
    """Untrained head's final linear has small-std init so outputs are tiny."""
    head = _head(embed_dim=16)
    emb = torch.randn(8, 10, 16)
    mask = torch.zeros(8, 10, dtype=torch.bool)
    out = head(emb, mask)
    assert out.abs().mean() < 0.1, f"untrained head mean |out| = {out.abs().mean():.3f}"
