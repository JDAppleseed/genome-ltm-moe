from __future__ import annotations

import importlib
import pathlib
import unittest

import yaml


class SmokeChecksTest(unittest.TestCase):
    def test_yaml_configs_load(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        agentic = repo_root / "configs" / "agentic.yaml"
        verifier = repo_root / "configs" / "verifier_policy.yaml"
        with agentic.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)
        with verifier.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)

    def test_import_critical_modules(self) -> None:
        modules = [
            "genomaic.data.manifest",
            "genomaic.data.sharding",
            "genomaic.train.ds_config",
            "genomaic.data.stream_fastq",
            "genomeltm.pipeline.verifier_loop",
            "genomeltm.eval.abstention",
        ]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
