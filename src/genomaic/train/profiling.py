from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ThroughputStats:
    steps_per_sec: float
    tokens_per_sec: float
    data_time: float


def profile_step(
    step_start: float,
    step_end: float,
    tokens: int,
    data_time: float,
) -> ThroughputStats:
    elapsed = max(step_end - step_start, 1e-6)
    return ThroughputStats(
        steps_per_sec=1.0 / elapsed,
        tokens_per_sec=tokens / elapsed,
        data_time=data_time,
    )


def build_profiler(enabled: bool) -> Optional[torch.profiler.profile]:
    if not enabled:
        return None
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("runs/profiler"),
        record_shapes=True,
        with_stack=False,
    )


def now() -> float:
    return time.time()
