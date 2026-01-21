from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    modalities: List[str]

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Adapter name must be provided")


def available_adapters() -> Iterable[DatasetAdapter]:
    # TODO(phase=10, priority=medium, owner=?): Implement concrete dataset adapters for public sources.
    # Context: Phase 10 requires adapter scaffolding without downloads.
    # Acceptance: Adapters yield local paths and metadata for known public datasets.
    return [
        DatasetAdapter(name="thousand_genomes", modalities=["fastq"]),
        DatasetAdapter(name="synthetic_qc", modalities=["fastq"]),
    ]
