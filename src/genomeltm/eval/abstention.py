from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from genomeltm.models.heads.risk_coverage import risk_coverage_curve
from genomeltm.utils.config import load_yaml


@dataclass
class AbstentionEvalResult:
    coverage: torch.Tensor
    accuracy: torch.Tensor
    risk: torch.Tensor
    confidence: torch.Tensor


def compute_confidence(payload: Dict[str, torch.Tensor]) -> torch.Tensor:
    def _get_tensor(
        payload: Dict[str, torch.Tensor],
        key: str,
        like: torch.Tensor | None = None,
        default: float = 0.0,
    ) -> torch.Tensor:
        value = payload.get(key)
        dtype = None
        if like is not None:
            dtype = like.dtype if like.is_floating_point() else torch.float32
        if value is None:
            if like is not None:
                return torch.full_like(like, float(default), dtype=dtype)
            return torch.tensor(float(default), dtype=torch.float32)
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, device=like.device if like is not None else None, dtype=dtype)
        if like is not None and value.shape != like.shape:
            value = value.expand_as(like)
        return value

    like = None
    for key in ("abstain_logit", "uncertainty_logit", "conflict_logit", "correct"):
        candidate = payload.get(key)
        if candidate is not None:
            like = candidate
            break
    abstain_logit = _get_tensor(payload, "abstain_logit", like=like, default=0.0)
    uncertainty_logit = _get_tensor(payload, "uncertainty_logit", like=like, default=0.0)
    conflict_logit = _get_tensor(payload, "conflict_logit", like=like, default=0.0)
    abstain_prob = torch.sigmoid(abstain_logit)
    uncertainty_prob = torch.sigmoid(uncertainty_logit)
    conflict_prob = torch.sigmoid(conflict_logit)
    combined = torch.stack([abstain_prob, uncertainty_prob, conflict_prob], dim=0)
    return 1.0 - combined.max(dim=0).values


def run_abstention_eval(payload: Dict[str, torch.Tensor], n_points: int = 50) -> AbstentionEvalResult:
    correct = payload["correct"].bool()
    confidence = compute_confidence(payload)
    curve = risk_coverage_curve(correct=correct, confidence=confidence, n_points=n_points)
    accuracy = 1.0 - curve.risk
    return AbstentionEvalResult(
        coverage=curve.coverage,
        accuracy=accuracy,
        risk=curve.risk,
        confidence=confidence,
    )


def _load_payload(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Expected a dict payload with correctness + reliability logits")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Abstention-aware evaluation")
    parser.add_argument("--config", type=str, default="configs/abstention_eval.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    input_path = Path(args.input or cfg.get("input_path", ""))
    if not input_path:
        raise ValueError("Provide input_path in config or --input")
    output_path = args.output or cfg.get("output_path")
    n_points = int(cfg.get("n_points", 50))

    payload = _load_payload(input_path)
    result = run_abstention_eval(payload, n_points=n_points)

    metrics = {
        "coverage": result.coverage,
        "accuracy": result.accuracy,
        "risk": result.risk,
        "confidence": result.confidence,
    }

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(metrics, output_file)
        print(f"Saved abstention metrics: {output_file}")
    else:
        print("Abstention evaluation complete (no output path provided).")


if __name__ == "__main__":
    main()
