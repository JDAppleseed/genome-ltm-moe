from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml

from genomaic.data.tiles import TileConfig, iter_tiles, load_contigs, tile_config_from_dict, tiles_to_jsonl


LOGGER = logging.getLogger(__name__)


def _write_jsonl(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_tiles(output_dir: Path, config: TileConfig, contigs_path: Path) -> None:
    contigs = load_contigs(contigs_path)
    tile_lines = tiles_to_jsonl(iter_tiles(contigs, config.tiles))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "tiles.jsonl", tile_lines)
    _write_metadata(
        output_dir / "tiles_metadata.json",
        {
            "reference_build": config.tiles.reference_build,
            "tile_bp": config.tiles.tile_bp,
            "stride_bp": config.tiles.stride_bp,
            "num_tiles": len(tile_lines),
        },
    )

    if config.multi_res.enabled:
        for level in config.multi_res.levels:
            spec = level.to_spec(config.tiles)
            spec.validate()
            level_lines = tiles_to_jsonl(iter_tiles(contigs, spec))
            level_dir = output_dir / "multi_res" / level.name
            _write_jsonl(level_dir / "tiles.jsonl", level_lines)
            _write_metadata(
                level_dir / "tiles_metadata.json",
                {
                    "reference_build": spec.reference_build,
                    "tile_bp": spec.tile_bp,
                    "stride_bp": spec.stride_bp,
                    "num_tiles": len(level_lines),
                },
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build genomic tiles from contig lengths.")
    parser.add_argument("--config", type=Path, required=True, help="Path to tiles config YAML")
    parser.add_argument("--contigs", type=Path, required=True, help="Path to contigs .tsv or .json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for tiles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    LOGGER.info("Loading tile config")
    config_data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config_data, dict):
        raise ValueError("Tile config must be a YAML mapping")
    config = tile_config_from_dict(config_data)

    LOGGER.info(
        "Building tiles",
        extra={
            "tile_bp": config.tiles.tile_bp,
            "stride_bp": config.tiles.stride_bp,
            "multi_res": config.multi_res.enabled,
        },
    )
    _build_tiles(args.out_dir, config, args.contigs)
    LOGGER.info("Tile build complete", extra={"output_dir": str(args.out_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
