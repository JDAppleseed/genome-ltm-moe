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

from genomaic.models.moe_router import MoERouter  # noqa: E402
from genomaic.eval.expert_utilization import summarize_utilization  # noqa: E402


def test_moe_router_utilization_stats() -> None:
    router = MoERouter(d_model=4, num_experts=3)
    x = torch.randn(2, 5, 4)

    routing, stats = router(x)
    summary = summarize_utilization(routing.expert_counts)

    assert stats.expert_utilization.numel() == 3
    assert summary.total_tokens == int(routing.expert_counts.sum().item())
