from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from genomaic.pipeline.tile_embedder import TileEmbedder, TileEmbedderConfig, load_shard, shard_indices
from genomaic.retrieval.index import StubANNIndex


LOGGER = logging.getLogger(__name__)


def _load_tile_indices(tiles_path: Path) -> List[int]:
    tile_indices = []
    for line in tiles_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tile_indices.append(int(record["tile_index"]))
    return tile_indices


def _save_index(index: StubANNIndex, output_path: Path) -> None:
    if index._vectors is None:
        raise ValueError("Index has no vectors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vectors=index._vectors,
        tile_indices=np.array(index._tile_indices, dtype=np.int64),
        dim=index.dim,
    )


def build_tile_index(
    tiles_path: Path,
    output_dir: Path,
    embedding_dim: int,
    shard_size: int,
    seed: int,
    resume: bool,
) -> Path:
    tile_indices = _load_tile_indices(tiles_path)
    shards = shard_indices(tile_indices, shard_size)
    embedder = TileEmbedder(TileEmbedderConfig(embedding_dim=embedding_dim, seed=seed))

    shard_paths = []
    for shard_id, shard_tiles in enumerate(shards):
        shard_path = output_dir / "shards" / f"shard_{shard_id:04d}.npz"
        shard_paths.append(shard_path)
        if resume and shard_path.exists():
            LOGGER.info("Skipping existing shard", extra={"shard": str(shard_path)})
            continue
        LOGGER.info("Embedding shard", extra={"shard": str(shard_path), "tiles": len(shard_tiles)})
        embedder.embed_shard(shard_tiles, shard_path)

    index = StubANNIndex(dim=embedding_dim)
    for shard_path in shard_paths:
        tile_ids, embeddings = load_shard(shard_path)
        index.add(embeddings.numpy(), tile_ids.tolist())
    index.build()
    index_path = output_dir / "tile_index.npz"
    _save_index(index, index_path)
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tile embeddings and a stub ANN index.")
    parser.add_argument("--tiles", type=Path, required=True, help="Path to tiles.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--shard-size", type=int, default=1024, help="Tiles per shard")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic seed")
    parser.add_argument("--resume", action="store_true", help="Resume from existing shards")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    LOGGER.info("Starting tile index build", extra={"tiles": str(args.tiles)})
    index_path = build_tile_index(
        tiles_path=args.tiles,
        output_dir=args.out_dir,
        embedding_dim=args.embedding_dim,
        shard_size=args.shard_size,
        seed=args.seed,
        resume=args.resume,
    )
    LOGGER.info("Tile index build complete", extra={"index_path": str(index_path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
