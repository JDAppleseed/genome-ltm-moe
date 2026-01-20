# GenomAIc Paper Outline (Repo-Mapped)

## Title (working)
GenomAIc: Evidence-Grounded, Multi-Modal Genome Interpretation from Raw Reads with Calibrated Uncertainty

## Abstract
(1 paragraph summary; include: raw FASTQ-first, cross-tech fusion, MoE specialization, verifier + abstention, benchmarks)

## 1. Introduction
- Motivation: raw-read noise + conflicting evidence are under-modeled
- Why confidence + abstention matters
- Contributions

Repo anchors:
- docs/architecture.md
- docs/abstention_and_ambiguity.md
- src/genomeltm/models/heads/task_bundle.py

## 2. Related Work
- Sequence-to-function models (AlphaGenome/Enformer-style)
- Read-level modeling and variant calling pipelines
- MoE + long-context architectures in bio
- Agentic scientific copilots (CRISPR-GPT inspiration)

Repo anchors:
- docs/agentic_workflows_inspiration.md
- docs/cross_tech_fusion.md

## 3. System Overview
- Inputs: Illumina/ONT/PacBio FASTQ; optional aligned evidence
- Representation hierarchy: reads → tiles → long-range memory
- Multi-head task suite + verifier loop

Repo anchors:
- src/genomeltm/models/*
- src/genomeltm/pipeline/*
- configs/*.yaml

## 4. Model Architecture
- Backbone (SSM/Hyena or equivalent)
- Routing: MoE experts (repeat/SV/splice/error)
- Heads (variant, splicing, regulatory, SV, phasing, reliability)
- Calibration + abstention

Repo anchors:
- src/genomeltm/models/heads/*
- src/genomeltm/models/moe_router.py (if present)
- src/genomeltm/models/heads/risk_coverage.py

## 5. Training Curriculum
Stage 0: infra & synthetic
Stage 1: denoise self-supervision
Stage 2: consensus tiles
Stage 3: cross-tech alignment
Stage 4: functional multi-task supervision
Stage 5: verifier + abstention tuning
Stage 6: end-to-end finetune

Repo anchors:
- docs/training_curriculum.md
- scripts/train_stage*.py

## 6. Benchmarks & Evaluation
- Variant accuracy by class and region
- SV breakpoint tolerance & size-stratified recall
- Calibration (ECE, reliability diagrams)
- Risk-coverage curves; abstention utility
- Cross-tech concordance gains

Repo anchors:
- docs/eval_suite.md
- src/genomeltm/eval/*
- src/genomeltm/eval/abstention.py

## 7. Results
- Core benchmark tables
- Ablations: no MoE / no verifier / single-tech
- Scaling laws + compute
- Failure cases and uncertainty exemplars

Repo anchors:
- scripts/run_end_to_end.py
- docs/compute_and_cost.md

## 8. Discussion
- Where the approach helps most (rare disease, repeats, SV-heavy)
- Limitations & data governance
- Future work

Repo anchors:
- docs/data_access_and_ethics.md
- SECURITY.md

## 9. Methods (Reproducibility)
- Data processing details
- Training hyperparameters
- Implementation notes

Repo anchors:
- pyproject.toml / requirements.txt
- configs/*
- src/genomeltm/data/*

## 10. Ethics & Safety
- Privacy-first handling of human reads
- Confidence reporting and abstention
- Access control for sensitive workflows

Repo anchors:
- docs/data_access_and_ethics.md

## Appendices
- Additional ablations, metrics, datasets, pseudo-code
