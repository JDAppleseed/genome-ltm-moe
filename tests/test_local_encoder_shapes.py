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

from genomaic.models.local_encoder import LocalEncoder, LocalEncoderConfig  # noqa: E402


@pytest.mark.parametrize("mode", ["ssm_stub", "hyena_stub", "long_conv_stub"])
def test_local_encoder_output_shape(mode: str) -> None:
    config = LocalEncoderConfig(d_model=8, n_layers=2, mode=mode, pool="mean")
    encoder = LocalEncoder(config)

    x = torch.randn(4, 10, 8)
    output = encoder(x)

    assert output.shape == (4, 8)


def test_local_encoder_cls_pool() -> None:
    config = LocalEncoderConfig(d_model=4, n_layers=1, mode="ssm_stub", pool="cls")
    encoder = LocalEncoder(config)

    x = torch.randn(2, 6, 4)
    output = encoder(x)

    assert output.shape == (2, 4)
