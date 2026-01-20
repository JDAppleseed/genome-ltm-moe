from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

from .common import MLP, AttnPool


@dataclass
class SVOutput:
    # Breakpoint confidence + coarse class logits
    breakpoint_logit: torch.Tensor      # [B]
    class_logits: torch.Tensor          # [B, C]
    aux: Optional[Dict] = None


class StructuralVariantHead(nn.Module):
    """
    Stub SV head. In reality you'd use paired-window representations around candidate breakpoints.
    """
    def __init__(self, d_model: int, n_classes: int = 6, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.pool = AttnPool(d_model)
        self.bp = MLP(d_model, hidden, 1, dropout=dropout)
        self.cls = MLP(d_model, hidden, n_classes, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> SVOutput:
        pooled = self.pool(x, mask=mask)
        bp = self.bp(pooled)[:, 0]
        cls = self.cls(pooled)
        return SVOutput(breakpoint_logit=bp, class_logits=cls, aux={"n_classes": cls.shape[-1]})
