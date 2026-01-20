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
    aux: Optional[Dict] = None


class SplicingHead(nn.Module):
    """
    Stub splicing head: predicts splicing-impact proxy from sequence representation.
    In a full system, you'd condition on exon/intron annotations and produce junction-level outputs.
    """
    def __init__(self, d_model: int, hidden: int = 1024, out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pool = AttnPool(d_model)
        self.mlp = MLP(d_model, hidden, out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> SplicingOutput:
        pooled = self.pool(x, mask=mask)
        logits = self.mlp(pooled)
        if logits.shape[-1] == 1:
            logits = logits[:, 0]
        return SplicingOutput(delta_psi_logits=logits, aux={"head": "splicing"})
