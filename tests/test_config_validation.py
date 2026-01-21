from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from scripts.validate_configs import validate_yaml_file  # noqa: E402


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_validate_yaml_file_accepts_mapping(tmp_path: Path) -> None:
    file_path = tmp_path / "config.yaml"
    _write_yaml(file_path, "schema_version: 1\ntraining:\n  steps: 10\n")

    warnings = validate_yaml_file(file_path)

    assert warnings == []


def test_validate_yaml_file_warns_on_missing_schema(tmp_path: Path) -> None:
    file_path = tmp_path / "config.yaml"
    _write_yaml(file_path, "training:\n  steps: 10\n")

    warnings = validate_yaml_file(file_path)

    assert warnings == ["Config has no schema_version or $schema header"]


def test_validate_yaml_file_rejects_duplicates(tmp_path: Path) -> None:
    file_path = tmp_path / "config.yaml"
    _write_yaml(file_path, "training:\n  steps: 10\ntraining:\n  steps: 20\n")

    with pytest.raises(ValueError, match="Duplicate key"):
        validate_yaml_file(file_path)


def test_validate_yaml_file_requires_mapping(tmp_path: Path) -> None:
    file_path = tmp_path / "config.yaml"
    _write_yaml(file_path, "- item1\n- item2\n")

    with pytest.raises(ValueError, match="Expected YAML mapping"):
        validate_yaml_file(file_path)
