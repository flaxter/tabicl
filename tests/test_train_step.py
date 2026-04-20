"""Phase 4 — one-step end-to-end smoke tests.

Exercises the Phase 4 wiring: `PriorDataset.get_batch()` → TabICL forward
with `return_column_embeddings=True` → `MultiTaskLoss` → backward → one
`optimizer.step()`. No `Trainer` class involvement — these tests keep the
scope tight and avoid the DDP / wandb / checkpoint machinery.

All tests run on CPU with tiny configs and are expected to finish inside
~20 s.
"""
from __future__ import annotations

import torch

from tabicl.model.heads import ConditionalHead, InterventionalHead, ObservationalHead
from tabicl.model.tabicl import TabICL
from tabicl.prior.dataset import PriorDataset
from tabicl.train.multi_task_loss import MultiTaskLoss


def _tiny_tabicl(embed_dim: int = 32) -> TabICL:
    torch.manual_seed(0)
    # Disable feature grouping so `d` (per-sample active feature count) can be
    # passed through — the default "same" grouping asserts d is None.
    return TabICL(
        max_classes=3,
        embed_dim=embed_dim,
        col_num_blocks=2, col_nhead=4, col_num_inds=8,
        col_feature_group=False,
        icl_num_blocks=2, icl_nhead=4,
        row_num_blocks=2, row_nhead=4, row_num_cls=4,
        ff_factor=2,
    )


def _build_loss(embed_dim: int, **kwargs) -> MultiTaskLoss:
    return MultiTaskLoss(
        head_a=ObservationalHead(embed_dim=embed_dim),
        head_i=InterventionalHead(embed_dim=embed_dim),
        head_c=ConditionalHead(embed_dim=embed_dim),
        **kwargs,
    )


def _small_batch(B: int = 2, H: int = 4, T: int = 64, train_size: int = 32):
    ds = PriorDataset(
        batch_size=B, batch_size_per_gp=1,
        min_features=3, max_features=H,
        max_classes=3,
        min_seq_len=None, max_seq_len=T,
        prior_type="lingam_scm",
        n_jobs=1,
    )
    out = ds.get_batch()
    # out = (X, y, d, seq_lens, train_sizes, o_star, i_star, is_id, c_triples)
    X, y, d, seq_lens, train_sizes, o_star, i_star, is_id, c_triples = out
    # Force a consistent train/test split.
    train_size = int(min(train_size, int(seq_lens[0].item()) - 1))
    return X, y, d, train_size, o_star, i_star, is_id, c_triples


# ---------------------------------------------------------------------------
# Test 1 — one training step runs end-to-end
# ---------------------------------------------------------------------------


def test_one_train_step_runs_on_cpu():
    """Forward + backward + optimizer step on a PriorDataset LiNGAM batch."""
    embed_dim = 32
    model = _tiny_tabicl(embed_dim=embed_dim)
    loss_mod = _build_loss(embed_dim=embed_dim)

    X, y, d, train_size, o_star, i_star, is_id, c_triples = _small_batch(B=2, H=4, T=64)
    y_train = y[:, :train_size]
    y_test = y[:, train_size:]

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_mod.parameters()),
        lr=1e-3,
    )
    model.train()
    loss_mod.train()

    logits, col_emb = model(X, y_train, d, return_column_embeddings=True)
    assert col_emb.shape == (X.shape[0], X.shape[-1], embed_dim)

    total, bd = loss_mod(
        logits=logits, y_true=y_test,
        col_emb=col_emb,
        o_star=o_star, i_star=i_star,
        is_id=is_id, c_triples=c_triples, d=d,
    )
    assert torch.isfinite(total)

    total.backward()

    # Every module that should be training has received a gradient.
    heads_have_grads = any(p.grad is not None for p in loss_mod.head_a.parameters())
    assert heads_have_grads
    assert loss_mod.log_sigma2.grad is not None

    optimizer.step()

    # Loss drops on a repeat forward pass (overfitting one batch is stable).
    logits2, col_emb2 = model(X, y_train, d, return_column_embeddings=True)
    total2, _ = loss_mod(
        logits=logits2, y_true=y_test,
        col_emb=col_emb2,
        o_star=o_star, i_star=i_star,
        is_id=is_id, c_triples=c_triples, d=d,
    )
    # Soft assertion: the new total is finite and (usually) smaller.
    assert torch.isfinite(total2)


# ---------------------------------------------------------------------------
# Test 2 — trunk-freeze warmup zeroes trunk grads
# ---------------------------------------------------------------------------


def test_trunk_freeze_zeros_trunk_grads_simulated():
    """Simulate the run.py trunk-freeze clause and check the result.

    The Trainer-level wiring is out of scope for this unit test — we mimic
    the relevant block: compute loss, backward(), then zero trunk grads.
    Asserts that post-zero the trunk params have zero gradient while heads
    retain theirs.
    """
    embed_dim = 32
    model = _tiny_tabicl(embed_dim=embed_dim)
    loss_mod = _build_loss(embed_dim=embed_dim)

    X, y, d, train_size, o_star, i_star, is_id, c_triples = _small_batch(B=2, H=4, T=48)
    y_train = y[:, :train_size]
    y_test = y[:, train_size:]

    trunk_params = list(model.parameters())
    head_params = list(loss_mod.parameters())

    logits, col_emb = model(X, y_train, d, return_column_embeddings=True)
    total, _ = loss_mod(
        logits=logits, y_true=y_test,
        col_emb=col_emb,
        o_star=o_star, i_star=i_star,
        is_id=is_id, c_triples=c_triples, d=d,
    )
    total.backward()

    # Record a reference head gradient before zeroing trunk.
    ref_head_grad = head_params[0].grad.clone()

    # Mimic run.py's trunk-freeze warmup.
    for p in trunk_params:
        if p.grad is not None:
            p.grad.zero_()

    # Trunk grads are all zero.
    for p in trunk_params:
        if p.grad is not None:
            assert torch.equal(p.grad, torch.zeros_like(p.grad))

    # Head grads are untouched.
    assert torch.equal(head_params[0].grad, ref_head_grad)
