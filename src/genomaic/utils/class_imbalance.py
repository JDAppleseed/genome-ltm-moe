from __future__ import annotations

from typing import Dict, List


def inverse_frequency_weights(counts: Dict[int, int]) -> Dict[int, float]:
    if not counts:
        raise ValueError("counts must be non-empty")
    total = sum(counts.values())
    return {label: total / max(1, count) for label, count in counts.items()}


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    if not weights:
        raise ValueError("weights must be non-empty")
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}
