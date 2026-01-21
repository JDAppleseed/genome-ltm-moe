from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from genomaic.models.sv_evidence import SVEvidenceConfig, SVEvidenceHead  # noqa: E402
from genomaic.eval.sv_evidence_postproc import postprocess_sv_logits  # noqa: E402


def test_sv_evidence_head_shapes() -> None:
    head = SVEvidenceHead(SVEvidenceConfig(d_model=4, num_classes=2))
    fused = torch.randn(3, 4)

    outputs = head(fused)

    assert outputs["logits"].shape == (3, 2)
    assert outputs["confidence"].shape == (3,)


def test_sv_postprocess() -> None:
    logits = np.array([[0.1, 0.9], [2.0, 1.0]])
    result = postprocess_sv_logits(logits)

    assert result["predicted_labels"] == [1, 0]
