from __future__ import annotations

from typing import Iterable, List, Sequence

import hashlib

from genomaic.data.manifest import ShardEntry


def shard_by_rank(manifest: Sequence[ShardEntry], rank: int, world_size: int) -> List[ShardEntry]:
    if world_size <= 1:
        return list(manifest)
    return [entry for idx, entry in enumerate(manifest) if idx % world_size == rank]


def deterministic_order(entries: Iterable[ShardEntry], epoch: int, seed: int = 0) -> List[ShardEntry]:
    entries = list(entries)
    rng = (seed + epoch) % 9973

    def _stable_key(entry: ShardEntry) -> int:
        digest = hashlib.md5(entry.path.encode("utf-8")).hexdigest()
        return (int(digest, 16) + rng) % 1_000_000_007

    entries.sort(key=_stable_key)
    return entries


def epoch_seed(base_seed: int, epoch: int, rank: int) -> int:
    return (base_seed + epoch * 997 + rank * 101) % 1_000_000_007
