from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from genomaic.data.tiles import TileSpec, iter_tiles, tile_config_from_dict  # noqa: E402


def test_iter_tiles_basic() -> None:
    spec = TileSpec(
        tile_bp=10,
        stride_bp=4,
        reference_build="hg38",
        include_chrM=True,
        include_contigs=None,
        exclude_contigs=[],
    )
    contigs = {"chr1": 18}
    tiles = list(iter_tiles(contigs, spec))

    assert [(tile.start, tile.end) for tile in tiles] == [(0, 10), (4, 14), (8, 18)]


def test_iter_tiles_filters_contigs() -> None:
    spec = TileSpec(
        tile_bp=5,
        stride_bp=2,
        reference_build="hg38",
        include_chrM=False,
        include_contigs=["chr2"],
        exclude_contigs=["chr2"],
    )
    contigs = {"chr1": 10, "chr2": 10, "chrM": 10}
    tiles = list(iter_tiles(contigs, spec))

    assert tiles == []


def test_tile_config_from_dict_multi_res() -> None:
    config = {
        "tiles": {
            "tile_bp": 10,
            "stride_bp": 4,
            "reference_build": "hg38",
            "include_chrM": True,
            "include_contigs": None,
            "exclude_contigs": [],
            "multi_res": {
                "enabled": True,
                "levels": [
                    {"name": "fine", "tile_bp": 10, "stride_bp": 4},
                    {"name": "coarse", "tile_bp": 20, "stride_bp": 10},
                ],
            },
        }
    }
    tile_config = tile_config_from_dict(config)

    assert tile_config.multi_res.enabled is True
    assert [level.name for level in tile_config.multi_res.levels] == ["fine", "coarse"]
