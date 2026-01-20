from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass
class RiskCoverageResult:
    coverage: torch.Tensor  # [N]
    risk: torch.Tensor      # [N]


def risk_coverage_curve(
    correct: torch.Tensor,            # [B] bool/int
    confidence: torch.Tensor,         # [B] float
    n_points: int = 50
) -> RiskCoverageResult:
    """
    Produces a risk-coverage curve by sweeping confidence thresholds.
    risk = 1 - accuracy among retained predictions.
    """
    correct = correct.bool()
    conf = confidence.float()

    thr = torch.linspace(conf.min(), conf.max(), n_points, device=conf.device)
    cov = []
    risk = []
    for t in thr:
        keep = conf >= t
        if keep.sum() == 0:
            cov.append(torch.tensor(0.0, device=conf.device))
            risk.append(torch.tensor(0.0, device=conf.device))
            continue
        acc = correct[keep].float().mean()
        cov.append(keep.float().mean())
        risk.append(1.0 - acc)
    return RiskCoverageResult(coverage=torch.stack(cov), risk=torch.stack(risk))
