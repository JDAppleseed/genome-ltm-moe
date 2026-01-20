from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn

from .common import MLP, AttnPool


@dataclass
class RegulatoryOutput:
    # Multi-track regulatory predictions (e.g., accessibility/TF binding proxies)
    track_logits: torch.Tensor          # [B, T]
    aux: Optional[Dict] = None


class RegulatoryHead(nn.Module):
    """
    Stub regulatory head: pooled representation -> multi-track outputs.
    Replace with windowed / multi-resolution heads later.
    """
    def __init__(self, d_model: int, n_tracks: int = 256, hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pool = AttnPool(d_model)
        self.mlp = MLP(d_model, hidden, n_tracks, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> RegulatoryOutput:
        pooled = self.pool(x, mask=mask)
        logits = self.mlp(pooled)  # [B,T]
        return RegulatoryOutput(track_logits=logits, aux={"n_tracks": logits.shape[-1]})
