from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class PlatformEncoderConfig:
    num_platforms: int
    embedding_dim: int
    dropout: float = 0.0


class PlatformFeatureEncoder(nn.Module):
    def __init__(self, config: PlatformEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_platforms, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, platform_ids: torch.Tensor) -> torch.Tensor:
        if platform_ids.dim() != 1:
            raise ValueError("platform_ids must be a 1D tensor")
        embedded = self.embedding(platform_ids)
        return self.dropout(embedded)
