from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def initialize_deepspeed(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    dataloader: Optional[Any],
    ds_config: Dict[str, Any],
    grad_accum: int = 1,
) -> Tuple[Any, Optional[Any], Optional[Any]]:
    import deepspeed

    engine, optimizer, _, dataloader = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        training_data=dataloader,
        config=ds_config,
        gradient_accumulation_steps=grad_accum,
    )
    return engine, optimizer, dataloader
