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

    @property
    def phase_conf(self) -> torch.Tensor:
        return torch.sigmoid(self.phase_consistency_logit)

    @property
    def switch_prob(self) -> torch.Tensor:
        return 1.0 - self.phase_conf


class PhasingHead(nn.Module):
    """
    Stub phasing head: predicts a local phase consistency/confidence score.
    """
    def __init__(self, d_model: Optional[int] = None, hidden: int = 512, dropout: float = 0.1, d_in: Optional[int] = None):
        super().__init__()
        if d_model is None:
            if d_in is None:
                raise ValueError("d_model or d_in must be provided")
            d_model = d_in
        self.pool = AttnPool(d_model)
        self.mlp = MLP(d_model, hidden, 1, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> PhasingOutput:
        base = x if x.dim() == 2 else self.pool(x, mask=mask)
        logit = self.mlp(base)[:, 0]
        return PhasingOutput(phase_consistency_logit=logit, aux={"head": "phasing"})
