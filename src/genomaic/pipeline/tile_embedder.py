from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import logging
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileEmbedderConfig:
    embedding_dim: int
    seed: int = 1337


class TileEmbedder:
    """Stub tile embedder for deterministic embeddings."""

    def __init__(self, config: TileEmbedderConfig) -> None:
        self.config = config

    def embed_tile_indices(self, tile_indices: Sequence[int]) -> torch.Tensor:
        # TODO(phase=4, priority=high, owner=?): Replace stub embeddings with encoder forward pass.
        # Context: Phase 4 scaffolding only emits deterministic vectors for index build.
        # Acceptance: TileEmbedder consumes read/tile features and outputs learned embeddings.
        embeddings = []
        for tile_index in tile_indices:
            rng = np.random.default_rng(self.config.seed + int(tile_index))
            vector = rng.standard_normal(self.config.embedding_dim, dtype=np.float32)
            embeddings.append(vector)
        stacked = np.stack(embeddings, axis=0)
        return torch.from_numpy(stacked)

    def embed_shard(
        self,
        tile_indices: Sequence[int],
        shard_path: Path,
    ) -> None:
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = self.embed_tile_indices(tile_indices)
        np.savez_compressed(
            shard_path,
            tile_indices=np.array(tile_indices, dtype=np.int64),
            embeddings=embeddings.numpy(),
        )


def load_shard(shard_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    payload = np.load(shard_path)
    if "tile_indices" not in payload or "embeddings" not in payload:
        raise ValueError(f"Invalid shard format: {shard_path}")
    tile_indices = torch.from_numpy(payload["tile_indices"].astype(np.int64))
    embeddings = torch.from_numpy(payload["embeddings"].astype(np.float32))
    return tile_indices, embeddings


def shard_indices(tile_indices: Sequence[int], shard_size: int) -> List[List[int]]:
    return [list(tile_indices[i : i + shard_size]) for i in range(0, len(tile_indices), shard_size)]
