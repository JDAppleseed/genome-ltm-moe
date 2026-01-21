from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class CalibrationBin:
    confidence: float
    accuracy: float
    count: int


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> Tuple[float, List[CalibrationBin]]:
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")
    bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        conf_bin = float(confidences[mask].mean())
        acc_bin = float(correct[mask].mean())
        ece += (count / len(confidences)) * abs(acc_bin - conf_bin)
        bins.append(CalibrationBin(confidence=conf_bin, accuracy=acc_bin, count=count))
    return ece, bins


def risk_coverage(confidences: np.ndarray, correct: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")
    order = np.argsort(-confidences)
    sorted_correct = correct[order]
    coverage = np.arange(1, len(confidences) + 1) / len(confidences)
    risk = 1.0 - np.cumsum(sorted_correct) / np.arange(1, len(confidences) + 1)
    return coverage, risk
