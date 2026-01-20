from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from genomaic.models.moe_router import MoEBlock, MoERouting


class MoESequenceModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_experts: int, d_ff: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(4096, d_model)
        self.layers = nn.ModuleList([MoEBlock(d_model, n_experts, d_ff) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, List[MoERouting]]:
        batch, seq_len = ids.shape
        pos = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.emb(ids) + self.pos(pos)
        routings: List[MoERouting] = []
        for layer in self.layers:
            x, routing = layer(x)
            routings.append(routing)
        x = self.ln(x)
        logits = self.head(x)
        return logits, routings
