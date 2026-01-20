from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PlanStep:
    name: str
    config: Dict


class Planner:
    """Deterministic planner for dry-lab task decomposition."""

    def __init__(self, config: Dict):
        self.config = config
        self.tasks = config.get("planner", {}).get("tasks", ["ingest", "analyze", "benchmark", "report"])

    def plan(self, goal: str) -> List[PlanStep]:  # noqa: ARG002 - deterministic plan
        steps: List[PlanStep] = []
        tool_cfg = self.config.get("tools", {})
        for name in self.tasks:
            steps.append(PlanStep(name=name, config=tool_cfg.get(name, {})))
        return steps
