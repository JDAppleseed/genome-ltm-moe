#!/usr/bin/env python3
"""
Cost and wall-clock estimator for early GenomeLTM-MoE experiments.

This is a planning tool, not a promise: real throughput depends heavily on:
- active context (hard_active_tokens_max)
- SSM/Hyena kernel implementations
- MoE routing overhead + expert parallelism
- IO pipeline efficiency
- modality mix (Illumina vs PacBio vs ONT; raw signal optional)

Pricing reference:
- Lambda B200 SXM6 180GB listed at $4.99 per GPU-hour (plus tax).
  See https://lambda.ai/pricing or https://lambda.ai/instances
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class RunPlan:
    genomes: int = 100
    modalities: str = "ILLUMINA+PACBIO+ONT"  # informational
    tokens_per_genome_effective: float = 2.0e9
    # NOTE: "effective tokens" are after tiling/sampling; raw read tokens can be much larger.
    total_effective_tokens: Optional[float] = None

    params_dense: float = 3.0e9
    moe_experts: int = 16
    top_k: int = 2
    # MoE compute multiplier: approximates extra expert FLOPs vs dense.
    moe_multiplier: float = 1.6

    gpus: int = 256
    price_per_gpu_hour_usd: float = 4.99  # Lambda B200 reference
    sustained_pf_per_gpu: float = 0.10    # rough planning: 0.05–0.20 PF/GPU typical range
    # Increase for long-context overhead + routing + eval
    overhead_multiplier: float = 8.0

def estimate_flops(params: float, tokens: float, moe_mult: float) -> float:
    # Common heuristic: training FLOPs ≈ 6 * params * tokens
    return 6.0 * params * tokens * moe_mult

def estimate_time_hours(total_flops: float, gpus: int, sustained_pf_per_gpu: float, overhead: float) -> float:
    # sustained_pf_per_gpu in PFLOP/s => 1 PFLOP/s = 1e15 FLOP/s
    flops_per_second = gpus * sustained_pf_per_gpu * 1e15
    seconds = (total_flops / flops_per_second) * overhead
    return seconds / 3600.0

def estimate_cost_usd(hours: float, gpus: int, price_per_gpu_hour: float) -> float:
    return hours * gpus * price_per_gpu_hour

def main():
    plan = RunPlan()

    if plan.total_effective_tokens is None:
        plan.total_effective_tokens = plan.genomes * plan.tokens_per_genome_effective

    flops = estimate_flops(plan.params_dense, plan.total_effective_tokens, plan.moe_multiplier)
    hours = estimate_time_hours(flops, plan.gpus, plan.sustained_pf_per_gpu, plan.overhead_multiplier)
    cost = estimate_cost_usd(hours, plan.gpus, plan.price_per_gpu_hour_usd)

    print("=== GenomeLTM-MoE Cost/Time Estimate (Planning Heuristic) ===")
    print(f"Genomes: {plan.genomes}")
    print(f"Modalities: {plan.modalities}")
    print(f"Total effective tokens: {plan.total_effective_tokens:,.2e}")
    print(f"Dense params: {plan.params_dense:,.2e}")
    print(f"MoE experts: {plan.moe_experts}, top-k: {plan.top_k}, moe_multiplier: {plan.moe_multiplier}")
    print(f"GPUs: {plan.gpus}")
    print(f"Assumed sustained PFLOP/s per GPU: {plan.sustained_pf_per_gpu}")
    print(f"Overhead multiplier (long-context+MoE+IO+eval): {plan.overhead_multiplier}")
    print()
    print(f"Estimated total training FLOPs: {flops:,.2e}")
    print(f"Estimated wall-clock: {hours:,.1f} GPU-hours per GPU / {hours:,.1f} hours elapsed")
    print(f"Estimated cost @ ${plan.price_per_gpu_hour_usd}/GPU-hr: ${cost:,.0f} (plus tax/fees if applicable)")
    print()
    print("Notes:")
    print("- This assumes you are training on 'effective tokens' after tiling/sampling.")
    print("- If you attempt to train on raw-read tokens at full depth, token counts and IO explode.")
    print("- Tune sustained_pf_per_gpu and overhead_multiplier after your first profiling run.")

if __name__ == '__main__':
    main()
