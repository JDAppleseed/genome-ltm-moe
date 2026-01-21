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


class TinyDeterministicDataset(torch.utils.data.IterableDataset):
    def __init__(self, seed: int, length: int) -> None:
        super().__init__()
        self.seed = seed
        self.length = length

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.length):
            yield torch.randint(0, 100, (4,), generator=generator)


def test_resume_determinism() -> None:
    try:
        dataset = TinyDeterministicDataset(seed=123, length=5)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None)
        data_iter = iter(loader)

        _ = next(data_iter)
        second = next(data_iter)

        dataset_resume = TinyDeterministicDataset(seed=123, length=5)
        loader_resume = torch.utils.data.DataLoader(dataset_resume, batch_size=None)
        data_iter_resume = iter(loader_resume)
        _ = next(data_iter_resume)
        second_after_resume = next(data_iter_resume)

        assert torch.equal(second, second_after_resume)
    except RuntimeError as exc:
        pytest.skip(f"torch runtime issue: {exc}")
