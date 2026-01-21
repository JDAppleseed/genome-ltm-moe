from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from genomeltm.models.heads.risk_coverage import risk_coverage_curve
from genomeltm.utils.config import load_yaml
from genomeltm.utils.tensors import tensor_like


@dataclass
class AbstentionEvalResult:
    coverage: "torch.Tensor"
    accuracy: "torch.Tensor"
    risk: "torch.Tensor"
    confidence: "torch.Tensor"


def compute_confidence(payload: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    import torch
    like = None
    for key in ("abstain_logit", "uncertainty_logit", "conflict_logit", "correct"):
        candidate = payload.get(key)
        if candidate is not None:
            like = candidate
            break
    abstain_logit = tensor_like(payload.get("abstain_logit"), like, default=0.0)
    uncertainty_logit = tensor_like(payload.get("uncertainty_logit"), like, default=0.0)
    conflict_logit = tensor_like(payload.get("conflict_logit"), like, default=0.0)
    abstain_prob = torch.sigmoid(abstain_logit)
    uncertainty_prob = torch.sigmoid(uncertainty_logit)
    conflict_prob = torch.sigmoid(conflict_logit)
    combined = torch.stack([abstain_prob, uncertainty_prob, conflict_prob], dim=0)
    return 1.0 - combined.max(dim=0).values


def run_abstention_eval(payload: Dict[str, "torch.Tensor"], n_points: int = 50) -> AbstentionEvalResult:
    import torch
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


def _load_payload(path: Path) -> Dict[str, "torch.Tensor"]:
    import torch
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Expected a dict payload with correctness + reliability logits")
    return payload


def main(argv: list[str] | None = None) -> None:
    import torch
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
