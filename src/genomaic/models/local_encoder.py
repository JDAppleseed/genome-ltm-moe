from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn


EncoderMode = Literal["ssm_stub", "hyena_stub", "long_conv_stub"]
PoolMode = Literal["mean", "max", "cls"]


@dataclass(frozen=True)
class LocalEncoderConfig:
    d_model: int
    n_layers: int
    dropout: float = 0.0
    mode: EncoderMode = "ssm_stub"
    pool: PoolMode = "mean"


class LocalEncoder(nn.Module):
    def __init__(self, config: LocalEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        layers = []
        for _ in range(config.n_layers):
            if config.mode == "ssm_stub":
                layers.append(nn.Linear(config.d_model, config.d_model))
                layers.append(nn.GELU())
            elif config.mode == "hyena_stub":
                layers.append(nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1))
                layers.append(nn.GELU())
            elif config.mode == "long_conv_stub":
                layers.append(nn.Conv1d(config.d_model, config.d_model, kernel_size=7, padding=3))
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown encoder mode: {config.mode}")
        self.layers = nn.ModuleList(layers)
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected input shape (batch, length, d_model)")
        if mask is not None and mask.shape[:2] != x.shape[:2]:
            raise ValueError("Mask must match batch and length dimensions")
        hidden = self.input_norm(x)
        hidden = self.dropout(hidden)

        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                hidden = layer(hidden.transpose(1, 2)).transpose(1, 2)
            else:
                hidden = layer(hidden)

        hidden = self.output_norm(hidden)
        return self._pool(hidden, mask=mask)

    def _pool(self, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.config.pool == "mean":
            if mask is None:
                return hidden.mean(dim=1)
            weights = mask.float()
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denom
        if self.config.pool == "max":
            if mask is None:
                return hidden.max(dim=1).values
            masked = hidden.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            return masked.max(dim=1).values
        if self.config.pool == "cls":
            return hidden[:, 0, :]
        raise ValueError(f"Unknown pool mode: {self.config.pool}")
