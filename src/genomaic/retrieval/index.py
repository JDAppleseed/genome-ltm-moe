from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TileCoordinate:
    contig: str
    tile_index: int


@dataclass(frozen=True)
class RetrievalResult:
    tile_index: int
    score: float


@dataclass(frozen=True)
class DeterministicRetrievalConfig:
    enabled: bool
    neighbor_tiles: int


@dataclass(frozen=True)
class LearnedRetrievalConfig:
    enabled: bool
    top_k: int


class StubANNIndex:
    """Simple numpy-backed ANN stub with cosine similarity."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors: Optional[np.ndarray] = None
        self._tile_indices: List[int] = []

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, vectors: np.ndarray, tile_indices: Sequence[int]) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError("vectors must be 2D with correct dimensionality")
        if len(tile_indices) != vectors.shape[0]:
            raise ValueError("tile_indices length must match vectors")
        if self._vectors is None:
            self._vectors = vectors.astype(np.float32, copy=True)
        else:
            self._vectors = np.concatenate([self._vectors, vectors.astype(np.float32, copy=True)], axis=0)
        self._tile_indices.extend(list(tile_indices))

    def build(self) -> None:
        if self._vectors is None:
            raise ValueError("No vectors added to index")
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._vectors = self._vectors / norms

    @classmethod
    def load(cls, path: str) -> "StubANNIndex":
        payload = np.load(path)
        dim = int(payload["dim"])
        index = cls(dim=dim)
        index._vectors = payload["vectors"].astype(np.float32)
        index._tile_indices = payload["tile_indices"].astype(np.int64).tolist()
        return index

    def query(self, vector: np.ndarray, top_k: int) -> List[RetrievalResult]:
        if self._vectors is None:
            raise ValueError("Index is empty; call add/build before query")
        if vector.ndim != 1 or vector.shape[0] != self._dim:
            raise ValueError("vector must be 1D with correct dimensionality")
        normalized = vector.astype(np.float32, copy=True)
        norm = np.linalg.norm(normalized)
        if norm == 0:
            norm = 1.0
        normalized = normalized / norm
        scores = self._vectors @ normalized
        top_k = min(top_k, scores.shape[0])
        indices = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked = sorted(indices, key=lambda idx: (-scores[idx], self._tile_indices[idx]))
        return [RetrievalResult(tile_index=self._tile_indices[idx], score=float(scores[idx])) for idx in ranked]


class CoordinateWindowIndex:
    """Deterministic neighbor lookup based on tile ordering per contig."""

    def __init__(self, tiles: Iterable[TileCoordinate]) -> None:
        contig_map: Dict[str, List[int]] = {}
        for tile in tiles:
            contig_map.setdefault(tile.contig, []).append(tile.tile_index)
        self._contig_map = {contig: sorted(indices) for contig, indices in contig_map.items()}
        self._position_map = {
            contig: {tile_index: position for position, tile_index in enumerate(indices)}
            for contig, indices in self._contig_map.items()
        }

    def query(self, query_tile: TileCoordinate, neighbor_tiles: int) -> List[RetrievalResult]:
        indices = self._contig_map.get(query_tile.contig, [])
        if not indices:
            return []
        position_map = self._position_map.get(query_tile.contig, {})
        if query_tile.tile_index not in position_map:
            raise ValueError("Query tile not present in coordinate index")
        position = position_map[query_tile.tile_index]
        start = max(position - neighbor_tiles, 0)
        end = min(position + neighbor_tiles + 1, len(indices))
        window = indices[start:end]
        return [RetrievalResult(tile_index=tile_index, score=1.0) for tile_index in window]


class CombinedRetriever:
    def __init__(
        self,
        coordinate_index: CoordinateWindowIndex,
        ann_index: Optional[StubANNIndex],
        deterministic: DeterministicRetrievalConfig,
        learned: LearnedRetrievalConfig,
    ) -> None:
        self._coordinate_index = coordinate_index
        self._ann_index = ann_index
        self._deterministic = deterministic
        self._learned = learned

    def retrieve(
        self,
        query_tile: TileCoordinate,
        query_embedding: Optional[np.ndarray],
    ) -> List[RetrievalResult]:
        results: Dict[int, RetrievalResult] = {}
        if self._deterministic.enabled:
            for result in self._coordinate_index.query(query_tile, self._deterministic.neighbor_tiles):
                results[result.tile_index] = result
        if self._learned.enabled:
            if query_embedding is None:
                raise ValueError("query_embedding required for learned retrieval")
            if self._ann_index is None:
                raise ValueError("ANN index is not available")
            for result in self._ann_index.query(query_embedding, self._learned.top_k):
                if result.tile_index not in results:
                    results[result.tile_index] = result
        return sorted(results.values(), key=lambda item: item.tile_index)

    def retrieve_with_escalation(
        self,
        query_tile: TileCoordinate,
        query_embedding: Optional[np.ndarray],
        learned_multiplier: int,
        deterministic_neighbor_add: int,
    ) -> List[RetrievalResult]:
        learned = LearnedRetrievalConfig(
            enabled=self._learned.enabled,
            top_k=self._learned.top_k * max(1, learned_multiplier),
        )
        deterministic = DeterministicRetrievalConfig(
            enabled=self._deterministic.enabled,
            neighbor_tiles=self._deterministic.neighbor_tiles + max(0, deterministic_neighbor_add),
        )
        retriever = CombinedRetriever(
            coordinate_index=self._coordinate_index,
            ann_index=self._ann_index,
            deterministic=deterministic,
            learned=learned,
        )
        return retriever.retrieve(query_tile=query_tile, query_embedding=query_embedding)
