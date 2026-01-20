from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from genomeltm.models.heads.task_bundle import TaskBundle
from genomeltm.utils.config import load_yaml


@dataclass
class BackboneOutputs:
    x: torch.Tensor
    mask: Optional[torch.Tensor] = None
    ref_x: Optional[torch.Tensor] = None
    alt_x: Optional[torch.Tensor] = None
    aux: Optional[Dict[str, Any]] = None


class GenomeLTMMoE(nn.Module):
    """
    Minimal GenomeLTM-MoE scaffold with task bundle integration.

    This module expects token/tile representations as input and forwards them to
    multi-task heads. It intentionally preserves backbone math by keeping the
    representation flow unchanged.
    """

    def __init__(
        self,
        d_model: int = 512,
        input_dim: Optional[int] = None,
        task_cfg_path: str | Path = "configs/task_heads.yaml",
        model_cfg_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.model_cfg: Dict[str, Any] = {}
        if model_cfg_path:
            self.model_cfg = load_yaml(model_cfg_path)
            d_model = self.model_cfg.get("backbone", {}).get("ssm_blocks", {}).get("d_model", d_model)

        self.d_model = d_model
        self.input_dim = input_dim or d_model
        self.input_proj = None
        if self.input_dim != self.d_model:
            self.input_proj = nn.Linear(self.input_dim, self.d_model)

        task_cfg = load_yaml(task_cfg_path)
        self.task_bundle = TaskBundle(d_model=self.d_model, cfg=task_cfg)

    def _project(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if self.input_proj is None:
            return x
        return self.input_proj(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ref_x: Optional[torch.Tensor] = None,
        alt_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x = self._project(x)
        ref_x = self._project(ref_x)
        alt_x = self._project(alt_x)

        backbone_outputs = BackboneOutputs(x=x, mask=mask, ref_x=ref_x, alt_x=alt_x)
        task_outputs = self.task_bundle(x, mask=mask, ref_x=ref_x, alt_x=alt_x)

        return {
            "backbone": backbone_outputs,
            "tasks": task_outputs,
        }
