from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP block with GELU + dropout."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AttnPool(nn.Module):
    """
    Attention pooling over sequence dimension.
    x: [B, L, D], mask: [B, L] bool (True for valid)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.query, std=0.02)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # scores: [B, L]
        q = self.query.unsqueeze(0).unsqueeze(1)  # [1,1,D]
        k = self.proj(x)  # [B,L,D]
        scores = (k * q).sum(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B,L,1]
        pooled = (x * w).sum(dim=1)  # [B,D]
        return pooled


@dataclass
class HeadOutput:
    logits: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    aux: Optional[dict] = None
