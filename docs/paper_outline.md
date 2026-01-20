# Paper Outline (Repo-Mapped)

## Title
GenomeLTM-MoE: Evidence-Grounded, Calibrated Genome Interpretation from Raw Multi-Platform Reads with Verifier Loops

## Abstract
(1 paragraph summary + key results)

## 1. Introduction
- Motivation: false confidence in genomics; multi-tech reality
- Why calibrated abstention matters
- Contributions

Repo mapping:
- `docs/architecture.md`
- `docs/abstention_and_ambiguity.md`
- `docs/cross_tech_fusion.md`

## 2. Related Work
- Sequence-to-function models (Enformer-like, AlphaGenome-like)
- Variant calling (traditional + learned)
- Long-context models (SSM/Hyena) and MoE + verification paradigms

Repo mapping:
- `docs/references.md`
- `docs/sequencing_platforms.md`

## 3. System Overview
- FASTQ-first ingestion and noise modeling
- Hierarchical memory and effective 10M–100M context
- MoE experts + verifier pass
- Internal Hypothesis Graph (IHG) and VCF projection

Repo mapping:
- `docs/architecture.md`
- `schemas/internal_hypothesis_graph.schema.json`
- `docs/vcf_projection.md`

## 4. Model
4.1 Read Encoder  
4.2 Tile Encoder + Memory Retrieval  
4.3 MoE routing and expert specialization  
4.4 Verifier loop and escalation policy  
4.5 Task heads (variant effect, splicing, regulatory, SV, phasing, reliability)

Repo mapping:
- `docs/model_spec.md`
- `src/genomeltm/models/*`
- `src/genomeltm/models/heads/*`
- `configs/task_heads.yaml` (if you add it)

## 5. Training Curriculum
- Stage 1: denoising/corruption reversal
- Stage 2: tile consensus
- Stage 3: cross-tech alignment
- Stage 4: supervised multi-task heads
- Stage 5: verifier training + abstention
- Stage 6: end-to-end tuning

Repo mapping:
- `docs/training_curriculum.md`
- `scripts/train_stage*.py`
- `configs/training_defaults.yaml`

## 6. Evaluation
- Variant calling accuracy (PPV/TPR) with calibration
- SV breakpoint metrics and size stratification
- Cross-tech concordance gains
- Risk-coverage curves (accuracy vs abstention)
- Efficiency: throughput and cost per genome

Repo mapping:
- `docs/eval_suite.md`
- `src/genomeltm/eval/*`
- `scripts/eval_concordance.py`

## 7. Results
- Benchmark tables and key plots
- Ablations: no-verifier, no-MoE, single-tech only
- Failure cases and ambiguity behavior

Repo mapping:
- `docs/eval_suite.md`
- `src/genomeltm/eval/benchmarks.py`

## 8. Discussion
- When to abstain vs decide
- Implications for rare disease research and difficult loci
- Limitations and future work

Repo mapping:
- `docs/abstention_and_ambiguity.md`
- `docs/roadmap_and_milestones.md`

## 9. Ethics & Governance
- Research-only boundary
- Data governance and privacy
- Auditability and reproducibility

Repo mapping:
- `docs/data_access_and_ethics.md`
- `SECURITY.md`

## 10. Conclusion
- Summary of contributions and impact

## Appendices
- VCF projection schema and custom INFO/FORMAT fields
- Hyperparameters and compute budgets
- Additional ablations

Repo mapping:
- `docs/vcf_projection.md`
- `configs/*`
- `docs/compute_and_cost.md`
