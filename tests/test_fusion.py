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

from genomaic.models.fusion import CrossTileFusion, FusionConfig  # noqa: E402


@pytest.mark.parametrize("mode", ["mean", "gated", "cross_attention_fusion"])
def test_fusion_output_shape(mode: str) -> None:
    config = FusionConfig(d_model=8, mode=mode, dropout=0.0)
    fusion = CrossTileFusion(config)

    tiles = torch.randn(2, 5, 8)
    output = fusion(tiles)

    assert output.shape == (2, 8)
