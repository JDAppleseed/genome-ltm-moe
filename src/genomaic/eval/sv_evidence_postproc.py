from __future__ import annotations

from typing import Dict, List

import numpy as np


def postprocess_sv_logits(logits: np.ndarray) -> Dict[str, List[int]]:
    if logits.ndim != 2:
        raise ValueError("logits must be 2D array")
    labels = np.argmax(logits, axis=1).tolist()
    return {"predicted_labels": labels}
