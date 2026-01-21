from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from genomaic.data.manifest import ShardEntry


def build_manifest(paths: Iterable[Path], output_path: Path, modality: str) -> None:
    entries: List[ShardEntry] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        entries.append(ShardEntry(path=str(path), size=path.stat().st_size, modality=modality))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps({"path": entry.path, "size": entry.size, "modality": entry.modality}) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manifest JSONL from local shard paths.")
    parser.add_argument("paths", nargs="+", type=Path, help="Local shard paths")
    parser.add_argument("--out", type=Path, required=True, help="Manifest output path")
    parser.add_argument("--modality", type=str, default="fastq", help="Modality label")
    args = parser.parse_args()

    build_manifest(args.paths, args.out, args.modality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
