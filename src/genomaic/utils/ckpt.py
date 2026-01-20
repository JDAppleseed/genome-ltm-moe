from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    epoch: int,
    dataloader_state: Optional[Dict[str, Any]] = None,
    engine: Optional[Any] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if engine is not None:
        engine.save_checkpoint(str(path.parent), tag=path.name)
        return
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "step": step,
        "epoch": epoch,
        "dataloader_state": dataloader_state,
        "rng_state": _rng_state(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    engine: Optional[Any] = None,
) -> Dict[str, Any]:
    path = Path(path)
    if engine is not None:
        _, client_state = engine.load_checkpoint(str(path.parent), tag=path.name)
        return client_state or {}
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if "rng_state" in payload:
        _set_rng_state(payload["rng_state"])
    return payload


def find_latest_checkpoint(path: str | Path) -> Optional[Path]:
    path = Path(path)
    if not path.exists():
        return None
    candidates = sorted(path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def safe_symlink_latest(path: str | Path, target: str | Path) -> None:
    path = Path(path)
    target = Path(target)
    if path.exists() or path.is_symlink():
        path.unlink()
    os.symlink(target, path)
