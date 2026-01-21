from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

from .common import MLP, AttnPool


@dataclass
class VariantEffectOutput:
    # Per-variant (or per-site) effect logits/scores
    delta_logits: torch.Tensor          # [B, K] or [B, L]
    calibrated_prob: Optional[torch.Tensor] = None
    aux: Optional[Dict] = None

    @property
    def delta_score(self) -> torch.Tensor:
        return self.delta_logits

    @property
    def impact_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.delta_logits)


class VariantEffectHead(nn.Module):
    """
    Stub for a variant-effect head.
    Assumes you have:
      - ref_repr: [B, L, D]
      - alt_repr: [B, L, D] (or same if you compute a delta embedding)
      - mask: [B, L] bool
    Produces delta logits (alt - ref) pooled or per-position depending on mode.
    """
    def __init__(
        self,
        d_model: Optional[int] = None,
        hidden: int = 1024,
        out_dim: int = 1,
        per_position: bool = False,
        dropout: float = 0.1,
        d_in: Optional[int] = None,
    ):
        super().__init__()
        if d_model is None:
            if d_in is None:
                raise ValueError("d_model or d_in must be provided")
            d_model = d_in
        self.per_position = per_position
        self.pool = AttnPool(d_model)
        self.mlp = MLP(d_model, hidden, out_dim, dropout=dropout)

    def forward(
        self,
        ref_repr: torch.Tensor,
        alt_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> VariantEffectOutput:
        delta = alt_repr - ref_repr
        if delta.dim() == 2:
            logits = self.mlp(delta)
            if logits.shape[-1] == 1:
                logits = logits[:, 0]
            return VariantEffectOutput(delta_logits=logits, aux={"mode": "vector"})
        if self.per_position:
            logits = self.mlp(delta)
            if logits.shape[-1] == 1:
                logits = logits[..., 0]
            return VariantEffectOutput(delta_logits=logits, aux={"mode": "per_position"})
        pooled = self.pool(delta, mask=mask)
        logits = self.mlp(pooled)
        if logits.shape[-1] == 1:
            logits = logits[:, 0]
        return VariantEffectOutput(delta_logits=logits, aux={"mode": "pooled"})
