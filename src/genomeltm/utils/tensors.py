from __future__ import annotations

from typing import Any

import torch


def tensor_like(
    value: Any,
    like: torch.Tensor | None,
    *,
    default: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_device = device or (like.device if like is not None else None)
    resolved_dtype = dtype
    if resolved_dtype is None:
        if like is not None:
            resolved_dtype = like.dtype if like.is_floating_point() else torch.float32
        elif value is None:
            resolved_dtype = torch.float32

    if value is None:
        if like is not None:
            return torch.full_like(like, float(default), dtype=resolved_dtype, device=resolved_device)
        return torch.tensor(float(default), device=resolved_device, dtype=resolved_dtype)

    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=resolved_device, dtype=resolved_dtype)
    elif resolved_device is not None or resolved_dtype is not None:
        value = value.to(device=resolved_device, dtype=resolved_dtype)

    if like is not None and value.shape != like.shape:
        value = value.expand_as(like)

    return value
