from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from genomeltm.agents.tools.benchmark import benchmark
from genomeltm.agents.tools.ingest import ingest
from genomeltm.agents.tools.report import report


@dataclass
class ExecutionRecord:
    task: str
    status: str
    artifacts: Dict[str, Any] = field(default_factory=dict)


class DataAccessPolicy:
    def __init__(self, allowed_paths: List[str]):
        self.allowed_paths = [Path(p).resolve() for p in allowed_paths]

    def validate_path(self, path: str) -> None:
        if not self.allowed_paths:
            return
        resolved = Path(path).resolve()
        if not any(str(resolved).startswith(str(allowed)) for allowed in self.allowed_paths):
            raise PermissionError(f"Path not allowed by policy: {path}")


class Executor:
    """Sequential executor with provenance tracking and data-access constraints."""

    def __init__(self, config: Dict):
        self.config = config
        policy_cfg = config.get("data_access", {})
        self.policy = DataAccessPolicy(policy_cfg.get("allowed_paths", []))
        self.registry: Dict[str, Callable[..., Dict[str, Any]]] = {
            "ingest": ingest,
            "analyze": ingest,
            "benchmark": benchmark,
            "report": report,
        }
        self.provenance: List[ExecutionRecord] = []

    def _validate(self, step_cfg: Dict[str, Any]) -> None:
        for path_key in ("input_path", "output_path", "manifest_path"):
            if path_key in step_cfg:
                self.policy.validate_path(str(step_cfg[path_key]))

    def execute(self, plan: List[Any]) -> List[ExecutionRecord]:
        for step in plan:
            tool = self.registry.get(step.name)
            if tool is None:
                self.provenance.append(ExecutionRecord(task=step.name, status="skipped"))
                continue
            self._validate(step.config)
            artifacts = tool(step.config, self.provenance)
            self.provenance.append(ExecutionRecord(task=step.name, status="completed", artifacts=artifacts))
        return self.provenance
