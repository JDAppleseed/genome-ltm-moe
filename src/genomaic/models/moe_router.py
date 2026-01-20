from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MoERouting:
    dispatch_mask: torch.Tensor
    combine_weights: torch.Tensor
    expert_counts: torch.Tensor


class Top2Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> MoERouting:
        logits = self.gate(x)
        scores = torch.softmax(logits, dim=-1)
        top2 = torch.topk(scores, k=2, dim=-1)
        idx = top2.indices
        val = top2.values

        dispatch = torch.zeros(*scores.shape, device=scores.device)
        dispatch.scatter_(dim=-1, index=idx, src=val)
        combine = dispatch
        expert_counts = dispatch.sum(dim=(0, 1))
        return MoERouting(dispatch_mask=dispatch, combine_weights=combine, expert_counts=expert_counts)


class MoEBlock(nn.Module):
    def __init__(self, d_model: int, num_experts: int, d_ff: int):
        super().__init__()
        self.router = Top2Router(d_model, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, MoERouting]:
        routing = self.router(x)
        outputs = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts):
            mask = routing.dispatch_mask[..., idx:idx + 1]
            if mask.sum() == 0:
                continue
            outputs = outputs + expert(x) * mask
        return outputs, routing
