from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def ingest(config: Dict[str, Any], provenance: List[Any]) -> Dict[str, Any]:  # noqa: ARG001 - provenance used for tracing
    """Deterministic ingest placeholder for dry-lab workflows."""
    manifest_path = config.get("manifest_path")
    if manifest_path:
        manifest_path = Path(manifest_path)
        return {"manifest_path": str(manifest_path), "status": "ingested"}
    return {"status": "no_manifest"}
