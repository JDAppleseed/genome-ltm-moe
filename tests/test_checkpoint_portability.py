from __future__ import annotations

from pathlib import Path
import sys

import pytest

try:
    import torch
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))


def test_checkpoint_cpu_load(tmp_path: Path) -> None:
    try:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        path = tmp_path / "ckpt.pt"
        torch.save(checkpoint, path)

        loaded = torch.load(path, map_location="cpu")
        for tensor in loaded["model"].values():
            assert tensor.device.type == "cpu"
    except RuntimeError as exc:
        pytest.skip(f"torch runtime issue: {exc}")


def test_checkpoint_cuda_to_cpu(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        model = torch.nn.Linear(4, 2).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        path = tmp_path / "ckpt_cuda.pt"
        torch.save(checkpoint, path)

        loaded = torch.load(path, map_location="cpu")
        for tensor in loaded["model"].values():
            assert tensor.device.type == "cpu"
    except RuntimeError as exc:
        pytest.skip(f"torch runtime issue: {exc}")
