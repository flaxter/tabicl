"""Prediction + conditional predictive value loss.

Combines the cross-entropy classification loss with a Huber loss on the
conditional predictive value head's output, using either learned-uncertainty
weighting (Kendall, Gal, Cipolla 2018) or manually specified lambdas.

The module holds the value head so its parameters are included in
``parameters()`` and picked up by the outer optimizer. It does not own the
trunk; callers feed pre-computed column embeddings in.

Labels come from ``PriorDataset.get_batch`` as a ``list[dict]`` of length
``B``, each dict carrying:

- ``value_queries`` : ``list[ValueQuery]`` — each query supplies one
  conditioning-set mask ``S_mask`` of shape ``(p_b,)`` plus RMS targets
  ``targets`` of shape ``(p_b,)``; targets are NaN for positions ``i in S``
  and for padded positions past ``d_b``.
- ``y_var_raw`` : float — outcome variance from the oracle draw (diagnostic).
- ``label_scale`` : str — always ``"rms_y_units"``.

Loss path:

- flatten ``(b, query)`` into a single query batch ``M``
- pad per-query masks + targets to the model's ``H`` (the trunk trims to
  max(d) in the batch; Phase 3 labels were emitted at ``num_features`` per
  dataset, which may be smaller than H)
- run the value head once on the stacked batch -> ``(M, H)`` predictions
- Huber over finite targets only (NaN mask)
- strata diagnostics by ``query_type`` (``empty / singleton / small /
  medium / near_full``)

All paths are torch-only and CPU-safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.heads import ConditionalPredictiveValueHead


@dataclass
class LossBreakdown:
    """Per-task unscaled losses + scalar diagnostics for logging."""

    pred: float
    value: float
    w_pred: float
    w_value: float
    # Optional per-stratum value-loss diagnostics (NaN if stratum empty).
    value_empty: float = float("nan")
    value_singleton: float = float("nan")
    value_small: float = float("nan")
    value_medium: float = float("nan")
    value_near_full: float = float("nan")


class PredictiveValueLoss(nn.Module):
    """Combined CE + Huber(value-oracle) with uncertainty or manual weighting.

    Parameters
    ----------
    value_head : ConditionalPredictiveValueHead
        The attribution head. Held as a submodule so its parameters appear
        in ``PredictiveValueLoss.parameters()``.
    weighting : {"uncertainty", "manual"}
        ``"uncertainty"`` learns one ``log_sigma^2`` per task and combines
        losses via ``0.5 * exp(-log_sigma^2) * L + 0.5 * log_sigma^2``
        (Kendall et al. 2018). ``"manual"`` uses fixed lambdas.
    lambdas : dict, optional
        When ``weighting="manual"``: ``{"pred", "value"}`` keyed lambdas.
        Defaults to all-1.
    huber_delta : float
        Delta for the Huber loss on the value-head output.
    """

    def __init__(
        self,
        value_head: ConditionalPredictiveValueHead,
        weighting: str = "uncertainty",
        lambdas: Optional[Dict[str, float]] = None,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        if weighting not in ("uncertainty", "manual"):
            raise ValueError(f"weighting must be 'uncertainty' or 'manual', got {weighting!r}")

        self.value_head = value_head
        self.weighting = weighting
        self.huber_delta = float(huber_delta)

        if weighting == "uncertainty":
            # log(sigma^2) per task, initialised at 0 so each task starts at
            # weight 0.5 and the regulariser contributes 0.
            self.log_sigma2 = nn.Parameter(torch.zeros(2))
            self.lambdas = None
        else:
            lam = {"pred": 1.0, "value": 1.0}
            if lambdas:
                lam.update(lambdas)
            self.register_buffer(
                "manual_lambdas",
                torch.tensor([lam["pred"], lam["value"]], dtype=torch.float),
            )
            self.lambdas = lam

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------

    def _weights(self) -> Tuple[Tensor, Tensor]:
        if self.weighting == "uncertainty":
            w = 0.5 * torch.exp(-self.log_sigma2)
            return w[0], w[1]
        lam = self.manual_lambdas
        return lam[0], lam[1]

    def _uncertainty_reg(self) -> Tensor:
        if self.weighting == "uncertainty":
            return 0.5 * self.log_sigma2.sum()
        return torch.zeros((), device=self.manual_lambdas.device)

    # ------------------------------------------------------------------
    # Value-oracle loss
    # ------------------------------------------------------------------

    def _value_loss(
        self,
        col_emb: Tensor,
        labels: List[Dict[str, Any]],
        d: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Stack per-dataset queries into a single ``(M, H)`` batch.

        Returns the Huber loss and a per-stratum diagnostic dict. If no
        finite targets are found across the batch, returns a differentiable
        zero.
        """
        B, H, _ = col_emb.shape
        device = col_emb.device

        sample_idx: List[int] = []
        mask_rows: List[Tensor] = []
        target_rows: List[Tensor] = []
        query_types: List[str] = []

        for b, lab in enumerate(labels):
            for q in lab.get("value_queries", []):
                S_mask = q.S_mask
                targets = q.targets
                p = int(S_mask.shape[0])

                # Pad / truncate mask + targets to model's H.
                if p >= H:
                    mask_row = S_mask[:H].to(torch.bool)
                    tgt_row = targets[:H]
                else:
                    mask_pad = torch.zeros(H - p, dtype=torch.bool)
                    mask_row = torch.cat([S_mask.to(torch.bool), mask_pad])
                    tgt_pad = torch.full((H - p,), float("nan"), dtype=targets.dtype)
                    tgt_row = torch.cat([targets, tgt_pad])

                sample_idx.append(b)
                mask_rows.append(mask_row)
                target_rows.append(tgt_row)
                query_types.append(getattr(q, "query_type", "random"))

        if not sample_idx:
            return col_emb.new_zeros(()), {}

        sample_idx_t = torch.tensor(sample_idx, dtype=torch.long, device=device)
        mask_batch = torch.stack(mask_rows, dim=0).to(device)  # (M, H)
        target_batch = torch.stack(target_rows, dim=0).to(device=device, dtype=col_emb.dtype)

        emb_batch = col_emb[sample_idx_t]  # (M, H, E)
        pred = self.value_head(emb_batch, mask_batch)  # (M, H)

        # Also mask past-d positions per sample — those are padding in the
        # label emit path and should never contribute.
        active = _active_feature_mask(d[sample_idx_t], H)  # (M, H)
        # Finite-target mask additionally excludes positions i in S and
        # uncomputed entries (both emitted as NaN in labels).
        finite = torch.isfinite(target_batch)
        loss_mask = active & finite

        # Replace NaN in targets with zero for autograd safety, then huber.
        targets_safe = torch.where(finite, target_batch, torch.zeros_like(target_batch))
        err = F.huber_loss(pred, targets_safe, reduction="none", delta=self.huber_delta)
        denom = loss_mask.sum().clamp(min=1.0)
        loss = (err * loss_mask.to(err.dtype)).sum() / denom

        # Per-stratum diagnostics.
        strata: Dict[str, float] = {}
        for qt in ("empty", "singleton", "small", "medium", "near_full"):
            rows = [i for i, t in enumerate(query_types) if t == qt]
            if not rows:
                strata[qt] = float("nan")
                continue
            idx = torch.tensor(rows, dtype=torch.long, device=device)
            s_err = err.index_select(0, idx)
            s_mask = loss_mask.index_select(0, idx).to(err.dtype)
            s_denom = s_mask.sum().clamp(min=1.0).item()
            strata[qt] = float((s_err * s_mask).sum().item() / s_denom)

        return loss, strata

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: Tensor,
        y_true: Tensor,
        col_emb: Tensor,
        labels: List[Dict[str, Any]],
        d: Tensor,
    ) -> Tuple[Tensor, LossBreakdown]:
        """Return scalar total loss + unscaled per-task breakdown.

        Parameters
        ----------
        logits : Tensor, shape (B, T_test, C)
            Classification logits from ``TabICL._train_forward``.
        y_true : Tensor, shape (B, T_test)
            Ground-truth class ids (long).
        col_emb : Tensor, shape (B, H, E)
            Per-feature trunk embeddings.
        labels : list[dict]
            Per-dataset value-query labels; see module docstring.
        d : Tensor, shape (B,) long
            Active feature count per sample.
        """
        pred_logits = logits.flatten(end_dim=-2)
        true_flat = y_true.long().flatten()
        loss_pred = F.cross_entropy(pred_logits, true_flat)

        loss_value, strata = self._value_loss(col_emb, labels, d)

        w_pred, w_value = self._weights()
        reg = self._uncertainty_reg()

        total = w_pred * loss_pred + w_value * loss_value + reg

        breakdown = LossBreakdown(
            pred=float(loss_pred.detach().item()),
            value=float(loss_value.detach().item()),
            w_pred=float(w_pred.detach().item()) if isinstance(w_pred, Tensor) else float(w_pred),
            w_value=float(w_value.detach().item()) if isinstance(w_value, Tensor) else float(w_value),
            value_empty=strata.get("empty", float("nan")),
            value_singleton=strata.get("singleton", float("nan")),
            value_small=strata.get("small", float("nan")),
            value_medium=strata.get("medium", float("nan")),
            value_near_full=strata.get("near_full", float("nan")),
        )
        return total, breakdown


def _active_feature_mask(d: Tensor, H: int) -> Tensor:
    """Boolean ``(*, H)`` mask — True where the position is an active feature."""
    arange = torch.arange(H, device=d.device).unsqueeze(0)  # (1, H)
    if d.dim() == 0:
        d = d.unsqueeze(0)
    return arange < d.unsqueeze(-1)
