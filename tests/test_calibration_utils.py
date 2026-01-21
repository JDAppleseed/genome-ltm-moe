from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from genomaic.eval.calibration import compute_ece, risk_coverage  # noqa: E402


def test_compute_ece() -> None:
    confidences = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])

    ece, bins = compute_ece(confidences, correct, n_bins=2)

    assert ece >= 0.0
    assert bins


def test_risk_coverage() -> None:
    confidences = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 0, 1, 0])

    coverage, risk = risk_coverage(confidences, correct)

    assert coverage.shape == risk.shape
