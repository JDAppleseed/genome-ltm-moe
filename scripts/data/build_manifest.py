from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FASTQ manifest JSONL")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--modality", type=str, default="fastq")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as handle:
        for path in data_dir.rglob("*.fastq*"):
            size = path.stat().st_size
            record = {"path": str(path), "size": size, "modality": args.modality}
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote manifest: {output}")


if __name__ == "__main__":
    main()
