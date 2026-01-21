from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass(frozen=True)
class SVEvidenceConfig:
    d_model: int
    num_classes: int = 3


class SVEvidenceHead(nn.Module):
    def __init__(self, config: SVEvidenceConfig) -> None:
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        self.confidence = nn.Linear(config.d_model, 1)

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        if fused.dim() != 2:
            raise ValueError("fused embedding must be (batch, d_model)")
        logits = self.classifier(fused)
        confidence = torch.sigmoid(self.confidence(fused)).squeeze(-1)
        return {"logits": logits, "confidence": confidence}
