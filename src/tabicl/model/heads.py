"""Conditional predictive value head.

The head consumes per-column embeddings ``e`` of shape ``(B, H, E)`` returned
by ``TabICL.forward(..., return_column_embeddings=True)`` together with a
boolean conditioning-set mask ``cond_mask`` of shape ``(B, H)``, and emits a
score per feature estimating the RMS conditional predictive value

    r_{i|S} = sqrt(Delta_{i|S}),   Delta_{i|S} = V(S union {i}) - V(S),

where V(S) = Var(E[Y | X_S]). The head returns a length-H vector for every
batch element in one forward pass per information state ``S``. Callers mask
entries at positions ``i in S`` (the output there is architecturally defined
but semantically undefined — see the predictive oracle public API).

The trunk stays frozen for the first 1000 steps of training (PLAN Phase 4)
while the head stabilises. The final linear layer is initialised with small
weights so the untrained head produces near-zero outputs rather than random
offsets.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor


def _build_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r} (supported: 'gelu', 'relu')")


def _small_init_final_layer(linear: nn.Linear, std: float = 0.01) -> None:
    """Small-weight init for the output projection of the head."""
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class ConditionalPredictiveValueHead(nn.Module):
    """Estimate RMS conditional predictive value r_{i|S} = sqrt(Delta_{i|S}).

    The conditioning-set representation ``e_S`` is a sum-pool over the columns
    selected by the mask. Fusion MLP consumes ``concat(e_i, e_S)`` and emits a
    scalar per feature.

    Parameters
    ----------
    embed_dim : int
        Dimensionality ``E`` of per-column embeddings.
    hidden_dim : Optional[int], default=None
        Hidden-layer dimension for the fusion MLP. Defaults to ``embed_dim // 2``.
    activation : str, default="gelu"
        Activation between the two linear layers. ``"gelu"`` or ``"relu"``.

    Notes
    -----
    The head computes a score for every feature in one forward pass — the
    substrate for the sklearn API call ``conditional_predictive_values(S)``.
    Positions inside ``S`` are still computed (the architecture is
    position-symmetric) and should be masked to ``NaN`` by the caller.

    Empty ``S`` is valid: ``e_S = 0`` and the head reduces to a function of
    ``e_i`` alone. Near-full ``S`` (size ``p - 1`` or ``p - 2``) is also
    valid and is sampled explicitly during training to cover the
    leave-one-out endpoint.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim if hidden_dim is not None else max(1, embed_dim // 2)

        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.act = _build_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, 1)

        _small_init_final_layer(self.fc2)

    def forward(self, column_embeddings: Tensor, cond_mask: Tensor) -> Tensor:
        """Compute a conditional predictive value score for every feature.

        Parameters
        ----------
        column_embeddings : Tensor
            Per-column embeddings of shape ``(B, H, E)``.
        cond_mask : Tensor
            Boolean mask of shape ``(B, H)``. ``True`` marks features in the
            conditioning set ``S``; ``False`` marks features not in ``S``.
            An all-``False`` row corresponds to ``S = empty`` — ``e_S`` is
            then the zero vector.

        Returns
        -------
        Tensor
            Per-feature RMS scores of shape ``(B, H)``.
        """
        if column_embeddings.dim() != 3:
            raise ValueError(
                f"Expected column_embeddings of shape (B, H, E), got {tuple(column_embeddings.shape)}"
            )
        if cond_mask.dtype != torch.bool:
            raise TypeError(f"cond_mask must be torch.bool, got {cond_mask.dtype}")
        if cond_mask.shape != column_embeddings.shape[:2]:
            raise ValueError(
                f"cond_mask shape {tuple(cond_mask.shape)} must match "
                f"column_embeddings.shape[:2] {tuple(column_embeddings.shape[:2])}"
            )

        mask = cond_mask.to(column_embeddings.dtype).unsqueeze(-1)   # (B, H, 1)
        e_S = (column_embeddings * mask).sum(dim=1)                  # (B, E)

        H = column_embeddings.shape[1]
        e_S_expanded = e_S.unsqueeze(1).expand(-1, H, -1)            # (B, H, E)
        fused = torch.cat([column_embeddings, e_S_expanded], dim=-1) # (B, H, 2E)

        x = self.act(self.fc1(fused))                                # (B, H, hidden)
        return self.fc2(x).squeeze(-1)                               # (B, H)
