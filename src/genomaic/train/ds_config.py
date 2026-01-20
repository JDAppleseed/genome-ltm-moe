from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from genomaic.utils.config import load_yaml


def load_ds_config(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_ds_config_from_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    training_cfg = cfg.get("training", {})
    ds_cfg_path = training_cfg.get("deepspeed", {}).get("config") or training_cfg.get(
        "deepspeed_config", "configs/deepspeed/zero2_moe.json"
    )
    ds_cfg = load_ds_config(ds_cfg_path)

    training_cfg = cfg.get("training", {})
    ds_cfg["gradient_accumulation_steps"] = int(training_cfg.get("grad_accum", ds_cfg.get("gradient_accumulation_steps", 1)))
    ds_cfg["train_batch_size"] = int(training_cfg.get("train_batch_size", ds_cfg.get("train_batch_size", 1)))
    ds_cfg["train_micro_batch_size_per_gpu"] = int(
        training_cfg.get("micro_batch_size", ds_cfg.get("train_micro_batch_size_per_gpu", 1))
    )

    precision_cfg = training_cfg.get("precision", {})
    if precision_cfg.get("bf16", False):
        ds_cfg["bf16"] = {"enabled": True}
        ds_cfg.pop("fp16", None)
    elif precision_cfg.get("fp16", False):
        ds_cfg["fp16"] = {"enabled": True}
        ds_cfg.pop("bf16", None)

    return ds_cfg


def validate_moe_zero_compat(ds_cfg: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    uses_moe = bool(cfg.get("moe", {}).get("enabled", False) or cfg.get("training", {}).get("moe", {}).get("enabled", False))
    if not uses_moe:
        return

    zero_cfg = ds_cfg.get("zero_optimization", {})
    stage = int(zero_cfg.get("stage", 0))
    if stage > 2:
        raise ValueError("MoE training defaults to ZeRO-2; ZeRO-3+MoE not validated in this stack.")


def load_training_yaml(path: str | Path) -> Dict[str, Any]:
    return load_yaml(path)
