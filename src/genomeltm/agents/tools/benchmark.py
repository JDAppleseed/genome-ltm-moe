from __future__ import annotations

from typing import Any, Dict, List


def benchmark(config: Dict[str, Any], provenance: List[Any]) -> Dict[str, Any]:  # noqa: ARG001 - provenance used for tracing
    """Deterministic benchmark placeholder."""
    suite = config.get("suite", "baseline")
    return {"suite": suite, "status": "benchmarked"}
