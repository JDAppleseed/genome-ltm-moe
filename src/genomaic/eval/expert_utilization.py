from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class UtilizationSummary:
    total_tokens: int
    per_expert: Dict[int, int]


def summarize_utilization(expert_counts: torch.Tensor) -> UtilizationSummary:
    counts = expert_counts.detach().cpu().to(torch.int64)
    total = int(counts.sum().item())
    per_expert = {idx: int(counts[idx].item()) for idx in range(counts.numel())}
    return UtilizationSummary(total_tokens=total, per_expert=per_expert)
