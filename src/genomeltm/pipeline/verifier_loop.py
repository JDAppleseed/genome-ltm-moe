from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from genomeltm.utils.config import load_yaml
from genomeltm.utils.tensors import tensor_like


@dataclass
class VerifierPolicy:
    max_reruns: int = 2
    abstain_threshold: float = 0.5
    conflict_threshold: float = 0.5
    support_threshold: float = 0.5
    confidence_threshold: float = 0.5


@dataclass
class VerifierResult:
    posterior: "torch.Tensor"
    confidence: "torch.Tensor"
    escalate_mask: "torch.Tensor"
    abstained: "torch.Tensor"


def load_verifier_policy(path: str = "configs/verifier_policy.yaml") -> VerifierPolicy:
    cfg = load_yaml(path)
    policy = cfg.get("verifier", cfg)
    thresholds = policy.get("thresholds", {})
    return VerifierPolicy(
        max_reruns=int(policy.get("max_reruns", 2)),
        abstain_threshold=float(thresholds.get("abstain", 0.5)),
        conflict_threshold=float(thresholds.get("conflict", 0.5)),
        support_threshold=float(thresholds.get("support", 0.5)),
        confidence_threshold=float(thresholds.get("confidence", 0.5)),
    )


def _extract_reliability(expert_outputs: Any) -> Optional[Any]:
    if expert_outputs is None:
        return None
    if isinstance(expert_outputs, dict) and "reliability" in expert_outputs:
        return expert_outputs["reliability"]
    return getattr(expert_outputs, "reliability", None)


def verifier_loop(
    verifier: Callable[..., Any],
    tile_emb: "torch.Tensor",
    retrieved_ctx: Optional[Any],
    expert_outputs: Optional[Any],
    passes_max: Optional[int] = None,
    low_conf: float = 0.95,
    policy: Optional[VerifierPolicy] = None,
    policy_path: str = "configs/verifier_policy.yaml",
    rerun_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> VerifierResult:
    """
    Multi-pass refinement loop with abstention-aware gating.

    If abstain/conflict/uncertainty exceed thresholds, the loop re-runs encoding
    via `rerun_fn` (e.g., expand context or alternate routing). The loop is
    capped at policy.max_passes and can abstain definitively.
    """
    policy = policy or load_verifier_policy(policy_path)
    max_passes = min(policy.max_reruns, passes_max or policy.max_reruns)
    escalation_level = 0

    import torch

    post = None
    conf = None
    escalate = None
    abstained = None

    for _ in range(max_passes):
        post, conf, escalate = verifier(tile_emb, retrieved_ctx, expert_outputs)
        reliability = _extract_reliability(expert_outputs)

        if reliability is not None:
            abstain_logit = tensor_like(getattr(reliability, "abstain_logit", None), conf)
            conflict_logit = tensor_like(getattr(reliability, "conflict_logit", None), conf)
            uncertainty_logit = tensor_like(getattr(reliability, "uncertainty_logit", None), conf)
            abstain_prob = torch.sigmoid(abstain_logit)
            conflict_prob = torch.sigmoid(conflict_logit)
            uncertainty_prob = torch.sigmoid(uncertainty_logit)
            abstained = (
                (abstain_prob > policy.abstain_threshold)
                | (conflict_prob > policy.conflict_threshold)
                | (uncertainty_prob > policy.support_threshold)
            )
            escalate = abstained if escalate is None else (escalate | abstained)
        else:
            abstained = torch.zeros_like(conf, dtype=torch.bool)

        if (conf >= low_conf).all() and not abstained.any():
            break

        if rerun_fn is not None and escalate is not None and escalate.any():
            escalation_level += 1
            rerun = rerun_fn(
                tile_emb=tile_emb,
                retrieved_ctx=retrieved_ctx,
                expert_outputs=expert_outputs,
                escalate_mask=escalate,
                escalation_level=escalation_level,
            )
            tile_emb = rerun.get("tile_emb", tile_emb)
            retrieved_ctx = rerun.get("retrieved_ctx", retrieved_ctx)
            expert_outputs = rerun.get("expert_outputs", expert_outputs)

    if abstained is None:
        abstained = torch.zeros_like(conf, dtype=torch.bool)

    return VerifierResult(posterior=post, confidence=conf, escalate_mask=escalate, abstained=abstained)
