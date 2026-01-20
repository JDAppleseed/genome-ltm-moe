from __future__ import annotations

from typing import Any, Dict, List


def report(config: Dict[str, Any], provenance: List[Any]) -> Dict[str, Any]:
    """Deterministic report placeholder."""
    report_format = config.get("format", "markdown")
    return {"format": report_format, "status": "reported", "provenance_steps": len(provenance)}
