from __future__ import annotations

from pathlib import Path
import importlib
import importlib.abc
import sys
from types import ModuleType
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))


class _BlockTorchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object | None, target: ModuleType | None = None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch import blocked for test")
        return None


class _TorchBlocker:
    def __enter__(self) -> "_TorchBlocker":
        self._finder = _BlockTorchFinder()
        sys.meta_path.insert(0, self._finder)
        sys.modules.pop("torch", None)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sys.meta_path = [finder for finder in sys.meta_path if finder is not self._finder]
        sys.modules.pop("torch", None)


def _import_targets() -> Iterator[str]:
    yield "genomaic.data.tiles"
    yield "genomaic.retrieval.index"
    yield "genomaic.data.adapters"
    yield "genomaic.utils.class_imbalance"
    yield "genomaic.utils.config"
    yield "scripts.validate_configs"
    yield "scripts.data.build_tiles"
    yield "scripts.data.build_manifest"


def test_imports_without_torch() -> None:
    with _TorchBlocker():
        for target in _import_targets():
            importlib.import_module(target)
