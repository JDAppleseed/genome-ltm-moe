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

    @property
    def effect_score(self) -> torch.Tensor:
        return self.track_logits.mean(dim=-1)

    @property
    def track_pred(self) -> torch.Tensor:
        return self.track_logits


class RegulatoryHead(nn.Module):
    """
    Stub regulatory head: pooled representation -> multi-track outputs.
    Replace with windowed / multi-resolution heads later.
    """
    def __init__(
        self,
        d_model: Optional[int] = None,
        n_tracks: int = 256,
        hidden: int = 2048,
        dropout: float = 0.1,
        d_in: Optional[int] = None,
    ):
        super().__init__()
        if d_model is None:
            if d_in is None:
                raise ValueError("d_model or d_in must be provided")
            d_model = d_in
        self.pool = AttnPool(d_model)
        self.mlp = MLP(d_model, hidden, n_tracks, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> RegulatoryOutput:
        base = x if x.dim() == 2 else self.pool(x, mask=mask)
        logits = self.mlp(base)  # [B,T]
        return RegulatoryOutput(track_logits=logits, aux={"n_tracks": logits.shape[-1]})
