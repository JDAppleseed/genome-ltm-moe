from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"torch unavailable: {exc}", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from genomaic.pipeline.tile_embedder import TileEmbedder, TileEmbedderConfig, load_shard, shard_indices  # noqa: E402
from scripts.data.build_tile_index import build_tile_index  # noqa: E402


def _write_tiles(path: Path, count: int) -> None:
    lines = [f'{{"contig": "chr1", "start": {i}, "end": {i + 10}, "tile_index": {i}}}' for i in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tile_embedder_determinism() -> None:
    embedder = TileEmbedder(TileEmbedderConfig(embedding_dim=4, seed=123))
    first = embedder.embed_tile_indices([0, 1, 2])
    second = embedder.embed_tile_indices([0, 1, 2])

    assert torch.allclose(first, second)


def test_build_tile_index_resume(tmp_path: Path) -> None:
    tiles_path = tmp_path / "tiles.jsonl"
    _write_tiles(tiles_path, 5)
    output_dir = tmp_path / "out"

    index_path = build_tile_index(
        tiles_path=tiles_path,
        output_dir=output_dir,
        embedding_dim=8,
        shard_size=2,
        seed=1,
        resume=False,
    )
    assert index_path.exists()

    shard_paths = sorted((output_dir / "shards").glob("*.npz"))
    assert len(shard_paths) == 3

    build_tile_index(
        tiles_path=tiles_path,
        output_dir=output_dir,
        embedding_dim=8,
        shard_size=2,
        seed=1,
        resume=True,
    )

    tile_ids, embeddings = load_shard(shard_paths[0])
    assert tile_ids.numel() == embeddings.shape[0]
