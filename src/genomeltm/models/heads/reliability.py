from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

from .common import MLP, AttnPool


@dataclass
class ReliabilityOutput:
    uncertainty_logit: torch.Tensor     # [B]
    conflict_logit: torch.Tensor        # [B]
    abstain_logit: torch.Tensor         # [B]
    aux: Optional[Dict] = None


class ReliabilityHead(nn.Module):
    """
    Stub reliability head:
      - uncertainty_logit: epistemic-ish proxy (learned)
      - conflict_logit: disagreement proxy (learned)
      - abstain_logit: gating for deferral / human-in-the-loop
    """
    def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.pool = AttnPool(d_model)
        self.unc = MLP(d_model, hidden, 1, dropout=dropout)
        self.cfl = MLP(d_model, hidden, 1, dropout=dropout)
        self.abs = MLP(d_model, hidden, 1, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> ReliabilityOutput:
        pooled = self.pool(x, mask=mask)
        u = self.unc(pooled)[:, 0]
        c = self.cfl(pooled)[:, 0]
        a = self.abs(pooled)[:, 0]
        return ReliabilityOutput(uncertainty_logit=u, conflict_logit=c, abstain_logit=a)
