from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List


@dataclass(frozen=True)
class ShardEntry:
    path: str
    size: int
    modality: str


def load_manifest(path: str | Path) -> List[ShardEntry]:
    entries: List[ShardEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            entries.append(
                ShardEntry(
                    path=payload["path"],
                    size=int(payload.get("size", 0)),
                    modality=payload.get("modality", "unknown"),
                )
            )
    return entries


def iter_manifest(paths: Iterable[ShardEntry]) -> Iterator[ShardEntry]:
    for entry in paths:
        yield entry
