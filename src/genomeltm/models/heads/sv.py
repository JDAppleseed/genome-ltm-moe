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
    size_logits: Optional[torch.Tensor] = None
    aux: Optional[Dict] = None

    @property
    def bp_conf(self) -> torch.Tensor:
        return torch.sigmoid(self.breakpoint_logit)

    @property
    def sv_type_logits(self) -> torch.Tensor:
        return self.class_logits

    @property
    def size_class_logits(self) -> torch.Tensor:
        if self.size_logits is None:
            return self.class_logits
        return self.size_logits


class StructuralVariantHead(nn.Module):
    """
    Stub SV head. In reality you'd use paired-window representations around candidate breakpoints.
    """
    def __init__(
        self,
        d_model: Optional[int] = None,
        n_classes: int = 6,
        hidden: int = 1024,
        dropout: float = 0.1,
        d_in: Optional[int] = None,
        sv_types: Optional[int] = None,
        size_classes: Optional[int] = None,
    ):
        super().__init__()
        if d_model is None:
            if d_in is None:
                raise ValueError("d_model or d_in must be provided")
            d_model = d_in
        self.pool = AttnPool(d_model)
        self.bp = MLP(d_model, hidden, 1, dropout=dropout)
        self.cls = MLP(d_model, hidden, sv_types or n_classes, dropout=dropout)
        self.size = MLP(d_model, hidden, size_classes, dropout=dropout) if size_classes else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> SVOutput:
        base = x if x.dim() == 2 else self.pool(x, mask=mask)
        bp = self.bp(base)[:, 0]
        cls = self.cls(base)
        size_logits = self.size(base) if self.size is not None else None
        return SVOutput(breakpoint_logit=bp, class_logits=cls, size_logits=size_logits, aux={"n_classes": cls.shape[-1]})
