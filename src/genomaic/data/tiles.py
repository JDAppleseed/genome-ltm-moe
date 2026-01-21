from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

import json


@dataclass(frozen=True)
class TileSpec:
    tile_bp: int
    stride_bp: int
    reference_build: str
    include_chrM: bool
    include_contigs: Optional[List[str]]
    exclude_contigs: List[str]

    def validate(self) -> None:
        if self.tile_bp <= 0 or self.stride_bp <= 0:
            raise ValueError("tile_bp and stride_bp must be positive integers")
        if self.tile_bp <= self.stride_bp:
            raise ValueError("tile_bp must be > stride_bp")


@dataclass(frozen=True)
class MultiResLevel:
    name: str
    tile_bp: int
    stride_bp: int

    def to_spec(self, base: TileSpec) -> TileSpec:
        return TileSpec(
            tile_bp=self.tile_bp,
            stride_bp=self.stride_bp,
            reference_build=base.reference_build,
            include_chrM=base.include_chrM,
            include_contigs=base.include_contigs,
            exclude_contigs=base.exclude_contigs,
        )


@dataclass(frozen=True)
class MultiResSpec:
    enabled: bool
    levels: List[MultiResLevel]


@dataclass(frozen=True)
class TileConfig:
    tiles: TileSpec
    multi_res: MultiResSpec


@dataclass(frozen=True)
class Tile:
    contig: str
    start: int
    end: int
    tile_index: int

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "contig": self.contig,
            "start": self.start,
            "end": self.end,
            "tile_index": self.tile_index,
        }


def load_contigs(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"Contigs file not found: {path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): int(v) for k, v in data.items()}
        if isinstance(data, list):
            result: Dict[str, int] = {}
            for item in data:
                if not isinstance(item, dict) or "name" not in item or "length" not in item:
                    raise ValueError("JSON contig list must contain name/length objects")
                result[str(item["name"])] = int(item["length"])
            return result
        raise ValueError("JSON contigs must be a mapping or list of objects")
    if path.suffix == ".tsv":
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, length = line.split("\t")
            result[name] = int(length)
        return result
    raise ValueError("Unsupported contigs format; use .json or .tsv")


def tile_config_from_dict(config: Mapping[str, object]) -> TileConfig:
    tiles = config.get("tiles")
    if not isinstance(tiles, Mapping):
        raise ValueError("Config must include tiles mapping")
    tile_spec = TileSpec(
        tile_bp=int(tiles["tile_bp"]),
        stride_bp=int(tiles["stride_bp"]),
        reference_build=str(tiles["reference_build"]),
        include_chrM=bool(tiles.get("include_chrM", True)),
        include_contigs=list(tiles["include_contigs"]) if tiles.get("include_contigs") is not None else None,
        exclude_contigs=list(tiles.get("exclude_contigs", [])),
    )
    tile_spec.validate()

    multi_res_raw = tiles.get("multi_res", {})
    if not isinstance(multi_res_raw, Mapping):
        raise ValueError("tiles.multi_res must be a mapping")
    levels_raw = multi_res_raw.get("levels", [])
    if not isinstance(levels_raw, list):
        raise ValueError("tiles.multi_res.levels must be a list")
    levels = [
        MultiResLevel(name=str(level["name"]), tile_bp=int(level["tile_bp"]), stride_bp=int(level["stride_bp"]))
        for level in levels_raw
    ]
    multi_res = MultiResSpec(enabled=bool(multi_res_raw.get("enabled", False)), levels=levels)
    return TileConfig(tiles=tile_spec, multi_res=multi_res)


def _filter_contigs(contigs: Mapping[str, int], spec: TileSpec) -> Dict[str, int]:
    result = dict(contigs)
    if spec.include_contigs is not None:
        result = {name: length for name, length in result.items() if name in set(spec.include_contigs)}
    if not spec.include_chrM:
        for name in ["chrM", "MT", "chrMT"]:
            result.pop(name, None)
    for name in spec.exclude_contigs:
        result.pop(name, None)
    return dict(sorted(result.items()))


def iter_tiles(contigs: Mapping[str, int], spec: TileSpec) -> Iterator[Tile]:
    filtered = _filter_contigs(contigs, spec)
    tile_index = 0
    for contig, length in filtered.items():
        start = 0
        while start < length:
            end = min(start + spec.tile_bp, length)
            yield Tile(contig=contig, start=start, end=end, tile_index=tile_index)
            tile_index += 1
            if end == length:
                break
            start += spec.stride_bp


def tiles_to_jsonl(tiles: Iterable[Tile]) -> List[str]:
    return [json.dumps(tile.to_dict(), sort_keys=True) for tile in tiles]
