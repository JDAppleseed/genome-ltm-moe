from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from genomaic.retrieval.index import (  # noqa: E402
    CombinedRetriever,
    CoordinateWindowIndex,
    DeterministicRetrievalConfig,
    LearnedRetrievalConfig,
    StubANNIndex,
    TileCoordinate,
)


def test_stub_ann_index_query() -> None:
    index = StubANNIndex(dim=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
    index.add(vectors, [0, 1, 2])
    index.build()

    results = index.query(np.array([1.0, 0.1], dtype=np.float32), top_k=2)

    assert [res.tile_index for res in results] == [0, 2]


def test_coordinate_window_index() -> None:
    tiles = [TileCoordinate(contig="chr1", tile_index=i) for i in range(5)]
    index = CoordinateWindowIndex(tiles)

    results = index.query(TileCoordinate(contig="chr1", tile_index=2), neighbor_tiles=1)

    assert [res.tile_index for res in results] == [1, 2, 3]


def test_combined_retriever_merge() -> None:
    tiles = [TileCoordinate(contig="chr1", tile_index=i) for i in range(4)]
    coordinate_index = CoordinateWindowIndex(tiles)
    ann_index = StubANNIndex(dim=2)
    ann_index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), [0, 3])
    ann_index.build()

    retriever = CombinedRetriever(
        coordinate_index=coordinate_index,
        ann_index=ann_index,
        deterministic=DeterministicRetrievalConfig(enabled=True, neighbor_tiles=1),
        learned=LearnedRetrievalConfig(enabled=True, top_k=1),
    )

    results = retriever.retrieve(TileCoordinate(contig="chr1", tile_index=1), np.array([1.0, 0.0], dtype=np.float32))

    assert [res.tile_index for res in results] == [0, 1, 2]
