from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ExpertUtilization:
    counts: torch.Tensor
    normalized: torch.Tensor
    overflow_rate: float


def summarize_expert_utilization(counts: torch.Tensor) -> ExpertUtilization:
    counts = counts.float()
    total = counts.sum().clamp(min=1.0)
    normalized = counts / total
    overflow = float((counts == 0).float().mean().item())
    return ExpertUtilization(counts=counts, normalized=normalized, overflow_rate=overflow)


def log_expert_utilization(metrics: ExpertUtilization) -> Dict[str, float]:
    log = {
        "expert_overflow_rate": metrics.overflow_rate,
    }
    for idx, value in enumerate(metrics.normalized.tolist()):
        log[f"expert_{idx}_fraction"] = value
    return log
