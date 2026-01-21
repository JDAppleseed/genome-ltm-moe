from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, TYPE_CHECKING

try:
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:  # pragma: no cover - handled in training entrypoints
    IterableDataset = object

if TYPE_CHECKING:
    import torch

from genomaic.data.manifest import ShardEntry


def _open_fastq(path: Path):
    if path.suffix.endswith("gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_fastq_records(path: Path) -> Iterator[str]:
    with _open_fastq(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            seq = handle.readline().strip()
            handle.readline()
            handle.readline()
            if seq:
                yield seq


class FastqStreamingDataset(IterableDataset):
    def __init__(self, shards: Iterable[ShardEntry], max_reads: Optional[int] = None):
        super().__init__()
        self.shards = list(shards)
        self.max_reads = max_reads

    def __iter__(self) -> Iterator[str]:
        count = 0
        for shard in self.shards:
            for seq in _iter_fastq_records(Path(shard.path)):
                yield seq
                count += 1
                if self.max_reads and count >= self.max_reads:
                    return


def encode_sequences(seqs: List[str], vocab: dict[str, int], max_len: int):
    import torch

    ids = torch.full((len(seqs), max_len), vocab["N"], dtype=torch.long)
    for i, seq in enumerate(seqs):
        for j, ch in enumerate(seq[:max_len]):
            ids[i, j] = vocab.get(ch.upper(), vocab["N"])
    return ids
