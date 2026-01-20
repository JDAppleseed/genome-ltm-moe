# GenomAIc — GenomeLTM-MoE  
**Ultra-long-context, evidence-grounded genome interpretation from raw reads (research-only)**

> **Mission (research-only):**  
> Build a noise-aware, ultra-long-context genome foundation model that learns directly from **raw sequencing reads** (FASTQ + metadata; optional ONT raw signal) to support:
> - calibrated variant and structural evidence synthesis  
> - disease-mechanism hypothesis generation  
> - robust cross-platform concordance analysis (Illumina ↔ PacBio ↔ Oxford Nanopore)  
>
> **Not a gene-editing project.**  
> This repository does **not** provide operational instructions for genome synthesis, genome editing, or CRISPR execution.  
> All outputs are intended strictly for **dry-lab research interpretation**, reproducibility, and clinical-research interfacing.

---

## Overview

**GenomAIc (GenomeLTM-MoE)** is a research-grade AI system for **genome interpretation at scale**, designed to operate directly on raw sequencing data while explicitly modeling noise, uncertainty, and long-range biological context.

The project prioritizes:
- **Correctness over confidence**
- **Abstention when evidence is insufficient**
- **Ultra-long effective context (10M–100M+ tokens)**
- **Reproducibility and distributed scalability**
- **Dry-lab safety constraints**

GenomAIc is not a variant caller in the traditional sense, nor a generative genome model. It is an **evidence synthesis and interpretation system** that produces predictions *with calibrated uncertainty and explicit provenance*.

---

## Why Ultra-Long Context?

Many biologically meaningful signals are inherently long-range:
- regulatory interactions
- repeats and segmental duplications
- complex immune loci
- structural variation
- haplotype and phasing context

Classic attention scales quadratically and becomes impractical.  
GenomAIc instead relies on **state-space / long-convolution operators**, hierarchical memory, and retrieval to achieve **10M–100M+ effective context** without quadratic blowup.

---

## High-Level System Design

1. **Read-level denoising & error modeling**  
   Learn \( P(\text{true sequence} \mid \text{observed reads, quality, platform}) \)

2. **Hierarchical genome memory**  
   Read → tile → long-context memory with learned retrieval

3. **Mixture-of-Experts (MoE)**  
   Specialized experts for:
   - splicing
   - repeats / SVs
   - coding constraint
   - regulatory regions
   - platform-specific error modes

4. **Verifier loop (bounded, safe)**  
   Second-pass consistency and calibration check  
   Low-confidence regions are re-routed or abstained

5. **Standardized projection (VCF v4.3)**  
   Export structured outputs + uncertainty evidence for interoperability  
   Spec: https://samtools.github.io/hts-specs/VCFv4.3.pdf

---

## Supported Modalities

GenomAIc is **FASTQ-first** and explicitly supports cross-technology fusion:

- **Illumina short-read WGS** (paired-end FASTQ)
- **PacBio HiFi** (FASTQ / BAM; high-accuracy long reads)
- **Oxford Nanopore** (FASTQ; optional POD5/FAST5 signal)

The system is designed to **reconcile conflicting evidence across platforms** into a single consensus interpretation with confidence reporting.

---

## Accuracy & Confidence Philosophy

Claims like “98%+ correct” are **task-bounded**, not global.

Examples:
- high-confidence variant PPV
- calibrated genotype quality
- SV breakpoint tolerance

Every task supports:
- explicit confidence
- evidence tracing
- a first-class **abstain** option

False certainty is treated as a failure mode.

---

## What This Repository Contains

This repository is a **full research stack**, including:

- Long-context genome modeling scaffolds
- Mixture-of-Experts routing and evaluation
- Multi-task interpretation heads
- Reliability, uncertainty, and abstention modeling
- Bounded verifier loops
- Deterministic FASTQ streaming and sharding
- Distributed training (DeepSpeed + torchrun)
- MacBook-as-control-plane orchestration
- Dry-lab agentic workflows (planner/executor)
- Reproducible evaluation tooling
- Developer-friendly bootstrapping and smoke checks

---

## What GenomAIc Does *Not* Do

❌ Genome synthesis  
❌ Genome editing  
❌ CRISPR execution  
❌ Wet-lab automation  
❌ Autonomous biological intervention  

Any exploratory generative work must live in explicitly isolated modules and is out of scope for the core system, reverse engineering is possible and can be explored, adherence to proper ethics MUST be maintained.

## Getting Started

### 1. Bootstrap Environment
```bash
bash scripts/dev/bootstrap_env.sh
source .venv/bin/activate
````

2. Run Smoke Checks
````
scripts/dev/smoke_check.sh
````

These checks:
	•	compile Python
	•	validate imports
	•	scan for bidi Unicode
	•	run lightweight tests
	•	fail with clear guidance if deps are missing

⸻

3. CPU Smoke Training (No GPU)
````
python -m genomaic.train.run \
  --config configs/training_cpu_smoke.yaml
````

⸻

4. Distributed Training (Example)
````
scripts/launch/torchrun_local.sh \
  configs/training_scale.yaml
````

SLURM templates are provided under scripts/launch/.

⸻

Development Principles
	•	Research-first
	•	Explicit uncertainty
	•	No silent failure
	•	No hidden data movement
	•	Config-driven everything
	•	Safe by default

⸻

Ethics & Safety
	•	Human genomic data must be handled under appropriate IRB / DUA constraints
	•	No automated medical decision-making, a human MUST be involved during practical processes
	•	Abstention preferred over false certainty, false positives or negatives must be strictly avoided
	•	Clear separation between interpretation and action

See docs/data_access_and_ethics.md.

⸻

Key References
	•	Illumina Genomics Architecture v3:
https://www.illumina.com/downloads/illumina-genomics-architecture-v3-tech-note-m-gl-03657.html
	•	PacBio HiFi sequencing:
https://www.pacb.com/technology/hifi-sequencing/
	•	Oxford Nanopore accuracy & Kit 14:
https://nanoporetech.com/platform/accuracy
https://nanoporetech.com/document/kit-14-device-and-informatics
	•	VCF v4.3 specification:
https://samtools.github.io/hts-specs/VCFv4.3.pdf

⸻

License

Apache-2.0 (see LICENSE).
Use is further constrained by SECURITY.md and relevant ethics documentation to research interpretation only.
