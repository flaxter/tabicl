"""Covers the ``d``-aware feature-grouping path in ``ColEmbedding``.

Two regimes:

- uniform ``d``: every table has the same ``d[0]`` real features — fast path
  that truncates ``X`` before ``feature_grouping`` so no padding leaks into
  any real-feature group.
- non-uniform ``d``: per-``d``-value batched path; the scatter keeps each
  table's real-feature groups uncontaminated and zero-fills padding groups.

The oracle is a per-sample reference that truncates ``X`` to ``d[b]``
columns and runs ``ColEmbedding`` with ``d=None``. Real-feature positions
must match that oracle within ``atol=1e-5``.
"""
from __future__ import annotations

import torch

from tabicl.model.embedding import ColEmbedding


def _build_col_embedding(seed: int = 0) -> ColEmbedding:
    torch.manual_seed(seed)
    # affine=False matches the trunk default when feature_group is enabled
    # (TabICL.__init__: col_feature_group='same', col_affine=False).
    return ColEmbedding(
        embed_dim=16,
        num_blocks=2,
        nhead=4,
        dim_feedforward=32,
        num_inds=8,
        target_aware=True,
        max_classes=3,
        feature_group="same",
        feature_group_size=3,
        affine=False,
        reserve_cls_tokens=2,
    )


def _reference_per_sample(
    col: ColEmbedding, X: torch.Tensor, y_train: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    """Run ColEmbedding once per sample with X truncated to d[b] columns."""
    B, T, H = X.shape
    C = col.reserve_cls_tokens
    out = torch.zeros(B, T, H + C, col.embed_dim, dtype=X.dtype, device=X.device)
    for b in range(B):
        d_b = int(d[b].item())
        X_b = X[b : b + 1, :, :d_b]
        y_b = y_train[b : b + 1]
        emb_b = col._train_forward(X_b, y_b, d=None)  # (1, T, d_b+C, E)
        out[b, :, :C, :] = emb_b[0, :, :C, :]
        out[b, :, C : C + d_b, :] = emb_b[0, :, C : C + d_b, :]
    return out


def _toy_batch(B: int, T: int, H: int, train_size: int, seed: int = 42):
    torch.manual_seed(seed)
    X = torch.randn(B, T, H)
    y_train = torch.randint(0, 3, (B, train_size))
    return X, y_train


def test_fg_d_none_matches_legacy():
    """Passing d=None is equivalent to the no-d path."""
    col = _build_col_embedding()
    col.train()
    X, y_train = _toy_batch(B=3, T=20, H=6, train_size=14)

    out_none = col._train_forward(X, y_train, d=None)
    out_full = col._train_forward(X, y_train, d=torch.tensor([6, 6, 6]))

    # Real-feature positions (indices C..C+6) must match the legacy d=None path.
    C = col.reserve_cls_tokens
    assert torch.allclose(out_none[:, :, : C + 6, :], out_full[:, :, : C + 6, :], atol=1e-5)


def test_fg_d_uniform_matches_truncated_reference():
    """Uniform d: output at real-feature positions matches the per-sample oracle."""
    col = _build_col_embedding()
    col.train()
    X, y_train = _toy_batch(B=3, T=20, H=6, train_size=14)
    d = torch.tensor([4, 4, 4])

    with torch.no_grad():
        out = col._train_forward(X, y_train, d=d)
        ref = _reference_per_sample(col, X, y_train, d)

    C = col.reserve_cls_tokens
    assert torch.allclose(out[:, :, : C + 4, :], ref[:, :, : C + 4, :], atol=1e-5)
    # Padding-group positions should be zero.
    assert torch.all(out[:, :, C + 4 :, :] == 0)


def test_fg_d_nonuniform_matches_truncated_reference():
    """Non-uniform d: per-d-value scatter matches the per-sample oracle."""
    col = _build_col_embedding()
    col.train()
    X, y_train = _toy_batch(B=4, T=20, H=6, train_size=14)
    d = torch.tensor([3, 5, 3, 6])

    with torch.no_grad():
        out = col._train_forward(X, y_train, d=d)
        ref = _reference_per_sample(col, X, y_train, d)

    C = col.reserve_cls_tokens
    for b in range(len(d)):
        d_b = int(d[b].item())
        assert torch.allclose(
            out[b, :, : C + d_b, :], ref[b, :, : C + d_b, :], atol=1e-5
        ), f"mismatch at sample {b} (d={d_b})"
        assert torch.all(out[b, :, C + d_b :, :] == 0), (
            f"padding-group positions should be zero for sample {b} (d={d_b})"
        )
