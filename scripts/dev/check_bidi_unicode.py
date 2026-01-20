#!/usr/bin/env python
"""Fail if bidi control characters are present in tracked text files."""

from __future__ import annotations

import argparse
import pathlib
import sys

BIDI_RANGES = [
    (0x202A, 0x202E),  # embedding/override
    (0x2066, 0x2069),  # isolate markers
]


def has_bidi(text: str) -> bool:
    return any(any(start <= ord(ch) <= end for start, end in BIDI_RANGES) for ch in text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan for bidi control characters.")
    parser.add_argument("root", nargs="?", default=".", help="Root directory to scan")
    args = parser.parse_args()

    root = pathlib.Path(args.root)
    offenders: list[pathlib.Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if has_bidi(text):
            offenders.append(path)

    if offenders:
        print("Found bidi control characters in:")
        for path in offenders:
            print(f"- {path}")
        return 1

    print("No bidi control characters detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
