from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

from .common import MLP, AttnPool


@dataclass
class SplicingOutput:
    # Example: delta PSI proxy logits
    delta_psi_logits: torch.Tensor       # [B] or [B, J]
    junction_logits: Optional[torch.Tensor] = None
    aux: Optional[Dict] = None

    @property
    def delta_psi(self) -> torch.Tensor:
        return self.delta_psi_logits

    @property
    def splice_impact_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.delta_psi_logits)


class SplicingHead(nn.Module):
    """
    Stub splicing head: predicts splicing-impact proxy from sequence representation.
    In a full system, you'd condition on exon/intron annotations and produce junction-level outputs.
    """
    def __init__(
        self,
        d_model: Optional[int] = None,
        hidden: int = 1024,
        out_dim: int = 1,
        dropout: float = 0.1,
        d_in: Optional[int] = None,
        junction_classes: Optional[int] = None,
    ):
        super().__init__()
        if d_model is None:
            if d_in is None:
                raise ValueError("d_model or d_in must be provided")
            d_model = d_in
        self.pool = AttnPool(d_model)
        self.mlp = MLP(d_model, hidden, out_dim, dropout=dropout)
        self.junction = nn.Linear(d_model, junction_classes) if junction_classes else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> SplicingOutput:
        base = x if x.dim() == 2 else self.pool(x, mask=mask)
        logits = self.mlp(base)
        if logits.shape[-1] == 1:
            logits = logits[:, 0]
        junction_logits = self.junction(base) if self.junction is not None else None
        return SplicingOutput(delta_psi_logits=logits, junction_logits=junction_logits, aux={"head": "splicing"})
