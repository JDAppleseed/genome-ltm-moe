from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn


FusionMode = Literal["mean", "gated", "cross_attention_fusion"]


@dataclass(frozen=True)
class FusionConfig:
    d_model: int
    mode: FusionMode = "mean"
    dropout: float = 0.0


class CrossTileFusion(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        if config.mode == "gated":
            self.gate = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.Sigmoid())
        elif config.mode == "cross_attention_fusion":
            self.attn = nn.MultiheadAttention(config.d_model, num_heads=4, batch_first=True)
        else:
            self.gate = None
            self.attn = None

    def forward(self, tiles: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tiles.dim() != 3:
            raise ValueError("tiles must be (batch, tiles, d_model)")
        if self.config.mode == "mean":
            return self.dropout(tiles.mean(dim=1))
        if self.config.mode == "gated":
            gates = self.gate(tiles)
            return self.dropout((tiles * gates).mean(dim=1))
        if self.config.mode == "cross_attention_fusion":
            if context is None:
                context = tiles
            fused, _ = self.attn(context, tiles, tiles)
            return self.dropout(fused.mean(dim=1))
        raise ValueError(f"Unknown fusion mode: {self.config.mode}")
