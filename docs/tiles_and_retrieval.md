# Tiles and retrieval

## Tile specification (v1)

GenomAIc partitions the genome into overlapping tiles to create a deterministic, retrievable memory
bank.

Default v1 spec:

- `tile_bp`: 24,000 bp
- `stride_bp`: 6,000 bp (75% overlap)

The tile config also supports optional multi-resolution levels (fine/medium/coarse). These are
available but disabled by default in `configs/tiles.yaml`.

## Tile build flow

1. Provide contig lengths as a `.tsv` (`contig<TAB>length`) or `.json`.
2. Run the tile builder to emit JSONL tiles and metadata.

Example:

```bash
python scripts/data/build_tiles.py --config configs/tiles.yaml --contigs data/contigs.tsv --out-dir data/tiles
```

Outputs:

- `tiles.jsonl` with `contig/start/end/tile_index`
- `tiles_metadata.json` with summary stats
- `multi_res/<level>/tiles.jsonl` when multi-res is enabled

## Retrieval overview

Retrieval uses two complementary paths:

- **Deterministic neighbors**: fixed window lookup around a query tile.
- **Learned retrieval**: embedding index lookup (stubbed in v1).

This allows effective context sizes in the 10M–100M range without quadratic attention.
