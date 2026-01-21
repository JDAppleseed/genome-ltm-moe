from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from genomaic.utils.class_imbalance import inverse_frequency_weights, normalize_weights  # noqa: E402


def test_inverse_frequency_weights() -> None:
    weights = inverse_frequency_weights({0: 10, 1: 5})

    assert weights[1] > weights[0]


def test_normalize_weights() -> None:
    weights = normalize_weights({0: 2.0, 1: 1.0})

    assert abs(sum(weights.values()) - 1.0) < 1e-6
