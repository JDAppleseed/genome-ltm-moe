"""Validate YAML configuration files for basic structural correctness."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import yaml


LOGGER = logging.getLogger(__name__)


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that raises on duplicate keys."""


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ValueError(f"Duplicate key '{key}' at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _iter_yaml_files(paths: Iterable[Path]) -> List[Path]:
    yaml_files: List[Path] = []
    for path in paths:
        if path.is_dir():
            yaml_files.extend(sorted(path.rglob("*.yaml")))
            yaml_files.extend(sorted(path.rglob("*.yml")))
        else:
            yaml_files.append(path)
    return yaml_files


def validate_yaml_file(path: Path) -> List[str]:
    """Validate a single YAML file, returning warnings."""
    LOGGER.debug("validating_config", extra={"path": str(path)})
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=UniqueKeyLoader)

    if data is None:
        raise ValueError("YAML file is empty")
    if not isinstance(data, dict):
        raise ValueError("Expected YAML mapping at top level")

    warnings: List[str] = []
    if "schema_version" not in data and "$schema" not in data:
        warnings.append("Config has no schema_version or $schema header")
    return warnings


def validate_configs(paths: Iterable[Path]) -> int:
    errors: List[str] = []
    warnings: List[str] = []
    yaml_files = _iter_yaml_files(paths)
    if not yaml_files:
        raise ValueError("No YAML files found to validate")

    for yaml_file in yaml_files:
        try:
            file_warnings = validate_yaml_file(yaml_file)
            for warning in file_warnings:
                warnings.append(f"{yaml_file}: {warning}")
        except Exception as exc:  # noqa: BLE001 - report all validation errors
            errors.append(f"{yaml_file}: {exc}")

    if warnings:
        for warning in warnings:
            LOGGER.warning(warning)

    if errors:
        for error in errors:
            LOGGER.error(error)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all configs in the configs/ directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.all:
        target_paths = [Path("configs")]
    else:
        target_paths = args.paths

    if not target_paths:
        raise SystemExit("Provide --all or at least one path to validate.")

    LOGGER.info("Starting config validation")
    return validate_configs(target_paths)


if __name__ == "__main__":
    raise SystemExit(main())
