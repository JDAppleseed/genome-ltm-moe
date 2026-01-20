from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

from .common import MLP, AttnPool


@dataclass
class PhasingOutput:
    phase_consistency_logit: torch.Tensor  # [B]
    aux: Optional[Dict] = None


class PhasingHead(nn.Module):
    """
    Stub phasing head: predicts a local phase consistency/confidence score.
    """
    def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pool = AttnPool(d_model)
        self.mlp = MLP(d_model, hidden, 1, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> PhasingOutput:
        pooled = self.pool(x, mask=mask)
        logit = self.mlp(pooled)[:, 0]
        return PhasingOutput(phase_consistency_logit=logit, aux={"head": "phasing"})
