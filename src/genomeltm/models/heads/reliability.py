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

    @property
    def abstain_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.abstain_logit)

    @property
    def conflict_prob(self) -> torch.Tensor:
        return torch.sigmoid(self.conflict_logit)


class ReliabilityHead(nn.Module):
    """
    Stub reliability head:
      - uncertainty_logit: epistemic-ish proxy (learned)
      - conflict_logit: disagreement proxy (learned)
      - abstain_logit: gating for deferral / human-in-the-loop
    """
    def __init__(self, d_model: Optional[int] = None, hidden: int = 512, dropout: float = 0.1, d_in: Optional[int] = None):
        super().__init__()
        if d_model is None:
            if d_in is None:
                raise ValueError("d_model or d_in must be provided")
            d_model = d_in
        self.pool = AttnPool(d_model)
        self.unc = MLP(d_model, hidden, 1, dropout=dropout)
        self.cfl = MLP(d_model, hidden, 1, dropout=dropout)
        self.abs = MLP(d_model, hidden, 1, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> ReliabilityOutput:
        base = x if x.dim() == 2 else self.pool(x, mask=mask)
        u = self.unc(base)[:, 0]
        c = self.cfl(base)[:, 0]
        a = self.abs(base)[:, 0]
        return ReliabilityOutput(uncertainty_logit=u, conflict_logit=c, abstain_logit=a)
