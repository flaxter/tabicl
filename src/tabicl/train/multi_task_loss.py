"""Phase 4 — multi-task loss for TabICL with attribution heads.

Combines the existing cross-entropy classification loss with three Huber
losses on the Phase 2 attribution heads (A / I / C), using either a
learned-uncertainty weighting (Kendall, Gal, Cipolla 2018) or manually
specified lambdas.

The module holds the three head instances so it can re-use them
inside the loss forward — keeping all attribution gradients inside a
single `loss.backward()` call on the trainer side. It does *not* own
the trunk; callers feed pre-computed column embeddings in.

Labels come from ``PriorDataset.get_batch`` (Phase 3):

- ``o_star``, ``i_star``: ``(B, H)`` in ``Var(continuous_y)`` units,
  NaN-padded past each sample's active feature count.
- ``is_identifiable``: ``(B,)`` bool. Rows with False do not contribute
  to the Head I loss.
- ``c_triples``: ``list[list[(int, BoolTensor(H,), float)]]``. Up to 16
  subsampled ``(i, S_mask, c_star)`` per sample.

All paths are torch-only and CPU-safe. The module is a standard
``nn.Module`` — DDP, AMP, gradient accumulation work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.heads import ConditionalHead, InterventionalHead, ObservationalHead


# Index positions of the four tasks in the uncertainty-weighting parameter.
_IDX_PRED = 0
_IDX_A = 1
_IDX_I = 2
_IDX_C = 3


@dataclass
class LossBreakdown:
    """Per-task unscaled losses + scalar diagnostics for logging."""

    pred: float
    A: float
    I: float
    C: float
    cons: float
    # Weight used on each task (either lambdas or exp(-log_sigma^2)/2).
    w_pred: float
    w_A: float
    w_I: float
    w_C: float


class MultiTaskLoss(nn.Module):
    """Combined CE + Huber(A, I, C) + optional consistency loss.

    Parameters
    ----------
    head_a, head_i, head_c : nn.Module
        The Phase 2 attribution heads. Held as submodules so their
        parameters are included in `MultiTaskLoss.parameters()` and
        thus picked up by the outer optimizer when the caller registers
        this module on the trainer.
    weighting : {'uncertainty', 'manual'}
        'uncertainty' learns one `log_sigma^2` per task and combines
        losses via `0.5 * exp(-log_sigma^2) * L + 0.5 * log_sigma^2`
        (Kendall et al. 2018). 'manual' uses fixed lambdas.
    lambdas : dict, optional
        When ``weighting='manual'``: ``{'pred', 'obs', 'int', 'cond'}``
        keyed lambdas. Defaults to all-1.
    huber_delta : float
        Delta for the Huber loss on the three head outputs.
    head_c_consistency_weight : float
        Weight on the auxiliary penalty ``|o_hat_i - c_hat_{i|-i}|``.
        0 disables the term.
    """

    def __init__(
        self,
        head_a: ObservationalHead,
        head_i: InterventionalHead,
        head_c: ConditionalHead,
        weighting: str = "uncertainty",
        lambdas: Optional[Dict[str, float]] = None,
        huber_delta: float = 1.0,
        head_c_consistency_weight: float = 0.0,
    ):
        super().__init__()
        if weighting not in ("uncertainty", "manual"):
            raise ValueError(f"weighting must be 'uncertainty' or 'manual', got {weighting!r}")

        self.head_a = head_a
        self.head_i = head_i
        self.head_c = head_c
        self.weighting = weighting
        self.huber_delta = float(huber_delta)
        self.head_c_consistency_weight = float(head_c_consistency_weight)

        if weighting == "uncertainty":
            # log(sigma^2) per task, initialised at 0 -> each head starts at
            # weight 0.5. `log_sigma2` is learnable and included in the
            # outer optimizer's param group automatically.
            self.log_sigma2 = nn.Parameter(torch.zeros(4))
            self.lambdas = None
        else:
            lam = {"pred": 1.0, "obs": 1.0, "int": 1.0, "cond": 1.0}
            if lambdas:
                lam.update(lambdas)
            self.register_buffer(
                "manual_lambdas",
                torch.tensor(
                    [lam["pred"], lam["obs"], lam["int"], lam["cond"]],
                    dtype=torch.float,
                ),
            )
            self.lambdas = lam

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------

    def _weights(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the four scalar weights used to combine per-task losses."""
        if self.weighting == "uncertainty":
            half = 0.5
            w = half * torch.exp(-self.log_sigma2)
            return w[0], w[1], w[2], w[3]
        lam = self.manual_lambdas
        return lam[0], lam[1], lam[2], lam[3]

    def _uncertainty_reg(self) -> Tensor:
        """`0.5 * sum log_sigma2` — the Kendall et al. regulariser."""
        if self.weighting == "uncertainty":
            return 0.5 * self.log_sigma2.sum()
        return torch.zeros((), device=self.manual_lambdas.device)

    # ------------------------------------------------------------------
    # Task-level loss computations
    # ------------------------------------------------------------------

    def _huber_masked(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Per-element Huber, reduced as ``sum(mask*err) / max(1, sum(mask))``.

        ``target`` may contain NaN at positions where the mask is False —
        we replace those with zero before the Huber call so autograd does
        not see NaNs in the arithmetic path.
        """
        mask = mask.to(pred.dtype)
        target = torch.where(torch.isfinite(target), target, torch.zeros_like(target))
        err = F.huber_loss(pred, target, reduction="none", delta=self.huber_delta)
        denom = mask.sum().clamp(min=1.0)
        return (err * mask).sum() / denom

    def _head_a_loss(self, col_emb: Tensor, o_star: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        B, H, _ = col_emb.shape
        o_hat = self.head_a(col_emb)
        mask = _active_feature_mask(d, H) & torch.isfinite(o_star)
        return self._huber_masked(o_hat, o_star, mask), o_hat

    def _head_i_loss(
        self,
        col_emb: Tensor,
        i_star: Tensor,
        is_id: Tensor,
        d: Tensor,
    ) -> Tensor:
        B, H, _ = col_emb.shape
        i_hat = self.head_i(col_emb)
        feat_mask = _active_feature_mask(d, H) & torch.isfinite(i_star)
        # Broadcast identifiability mask over features.
        sample_mask = is_id.to(torch.bool).unsqueeze(-1).expand_as(feat_mask)
        mask = feat_mask & sample_mask
        return self._huber_masked(i_hat, i_star, mask)

    def _head_c_loss(
        self,
        col_emb: Tensor,
        c_triples: List[List[Tuple[int, Tensor, float]]],
        d: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Stack sample-level triples into a ``(M, H)`` batch for one Head C call.

        Returns the Huber loss and the per-triple predicted `c_hat` (or
        ``None`` when no triples are provided across the batch).
        """
        B, H, _ = col_emb.shape
        sample_idx: List[int] = []
        feat_idx: List[int] = []
        masks: List[Tensor] = []
        targets: List[float] = []
        for b, triples in enumerate(c_triples):
            for (i, S_mask, c_val) in triples:
                sample_idx.append(b)
                feat_idx.append(int(i))
                # Pad / truncate mask to the model's H. `get_batch` stores
                # masks on the per-sample feature count — which matches
                # `d[b]`. Pad with False past `d[b]` so the head sees the
                # same shape as the column-embedding axis.
                if S_mask.shape[0] >= H:
                    masks.append(S_mask[:H].to(torch.bool))
                else:
                    pad = torch.zeros(H - S_mask.shape[0], dtype=torch.bool, device=S_mask.device)
                    masks.append(torch.cat([S_mask.to(torch.bool), pad]))
                targets.append(float(c_val))

        if not sample_idx:
            zero = col_emb.new_zeros(())
            return zero, None

        sample_idx_t = torch.tensor(sample_idx, dtype=torch.long, device=col_emb.device)
        feat_idx_t = torch.tensor(feat_idx, dtype=torch.long, device=col_emb.device)
        mask_batch = torch.stack(masks, dim=0).to(col_emb.device)  # (M, H)
        target_t = torch.tensor(targets, dtype=col_emb.dtype, device=col_emb.device)

        emb_batch = col_emb[sample_idx_t]  # (M, H, E)
        c_hat_all = self.head_c(emb_batch, mask_batch)  # (M, H)
        c_hat = c_hat_all.gather(1, feat_idx_t.unsqueeze(-1)).squeeze(-1)  # (M,)

        err = F.huber_loss(c_hat, target_t, reduction="mean", delta=self.huber_delta)
        return err, c_hat

    def _consistency_loss(self, col_emb: Tensor, d: Tensor, o_hat: Tensor) -> Tensor:
        """Optional |o_hat_i - c_hat_{i|-i}| penalty (PLAN §Phase 4 option 3).

        Builds, for every active feature per sample, the mask with all other
        active features set to True; runs Head C once per sample; diffs
        against Head A.
        """
        B, H, _ = col_emb.shape
        # For each (b, i), build cond_mask that has every active feature
        # except i set to True. We stack across (b, i) and run Head C on
        # the resulting (M, H) batch.
        active = _active_feature_mask(d, H)  # (B, H)
        sample_idx: List[int] = []
        feat_idx: List[int] = []
        masks: List[Tensor] = []
        for b in range(B):
            db = int(d[b].item())
            for i in range(db):
                m = active[b].clone()
                m[i] = False
                sample_idx.append(b)
                feat_idx.append(i)
                masks.append(m)
        if not sample_idx:
            return col_emb.new_zeros(())
        sample_idx_t = torch.tensor(sample_idx, dtype=torch.long, device=col_emb.device)
        feat_idx_t = torch.tensor(feat_idx, dtype=torch.long, device=col_emb.device)
        mask_batch = torch.stack(masks, dim=0).to(col_emb.device)  # (M, H)
        emb_batch = col_emb[sample_idx_t]
        c_hat_minus_i = self.head_c(emb_batch, mask_batch).gather(
            1, feat_idx_t.unsqueeze(-1)
        ).squeeze(-1)
        o_hat_picked = o_hat[sample_idx_t, feat_idx_t]
        return F.l1_loss(c_hat_minus_i, o_hat_picked)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: Tensor,
        y_true: Tensor,
        col_emb: Tensor,
        o_star: Tensor,
        i_star: Tensor,
        is_id: Tensor,
        c_triples: List[List[Tuple[int, Tensor, float]]],
        d: Tensor,
    ) -> Tuple[Tensor, LossBreakdown]:
        """Return scalar total loss + unscaled per-task breakdown.

        Parameters
        ----------
        logits : Tensor, shape (B, T_test, C)
            Classification logits from `TabICL._train_forward`.
        y_true : Tensor, shape (B, T_test)
            Ground-truth class ids (long).
        col_emb : Tensor, shape (B, H, E)
            Per-feature trunk embeddings (from `return_column_embeddings=True`).
        o_star, i_star : Tensor, shape (B, H)
            Phase 3 attribution labels, NaN-padded past `d`.
        is_id : Tensor, shape (B,) bool
            Identifiability flag.
        c_triples : list[list[tuple]]
            Per-sample Head C triples; see module docstring.
        d : Tensor, shape (B,) long
            Active feature count per sample.
        """
        pred = logits.flatten(end_dim=-2)
        true = y_true.long().flatten()
        loss_pred = F.cross_entropy(pred, true)

        loss_A, o_hat = self._head_a_loss(col_emb, o_star, d)
        loss_I = self._head_i_loss(col_emb, i_star, is_id, d)
        loss_C, _c_hat = self._head_c_loss(col_emb, c_triples, d)

        w_pred, w_A, w_I, w_C = self._weights()
        reg = self._uncertainty_reg()

        total = (
            w_pred * loss_pred
            + w_A * loss_A
            + w_I * loss_I
            + w_C * loss_C
            + reg
        )

        loss_cons = col_emb.new_zeros(())
        if self.head_c_consistency_weight > 0:
            loss_cons = self._consistency_loss(col_emb, d, o_hat)
            total = total + self.head_c_consistency_weight * loss_cons

        breakdown = LossBreakdown(
            pred=float(loss_pred.detach().item()),
            A=float(loss_A.detach().item()),
            I=float(loss_I.detach().item()),
            C=float(loss_C.detach().item()),
            cons=float(loss_cons.detach().item()),
            w_pred=float(w_pred.detach().item()) if isinstance(w_pred, Tensor) else float(w_pred),
            w_A=float(w_A.detach().item()) if isinstance(w_A, Tensor) else float(w_A),
            w_I=float(w_I.detach().item()) if isinstance(w_I, Tensor) else float(w_I),
            w_C=float(w_C.detach().item()) if isinstance(w_C, Tensor) else float(w_C),
        )
        return total, breakdown


def _active_feature_mask(d: Tensor, H: int) -> Tensor:
    """Boolean ``(B, H)`` mask — True where the position is an active feature."""
    arange = torch.arange(H, device=d.device).unsqueeze(0)  # (1, H)
    return arange < d.unsqueeze(-1)
