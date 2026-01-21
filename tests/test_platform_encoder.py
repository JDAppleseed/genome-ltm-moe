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

from genomaic.models.platform_encoder import PlatformEncoderConfig, PlatformFeatureEncoder  # noqa: E402


def test_platform_encoder_output_shape() -> None:
    config = PlatformEncoderConfig(num_platforms=3, embedding_dim=5, dropout=0.0)
    encoder = PlatformFeatureEncoder(config)

    output = encoder(torch.tensor([0, 1, 2]))

    assert output.shape == (3, 5)
