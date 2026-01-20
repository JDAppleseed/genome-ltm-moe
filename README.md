# GenomeLTM-MoE (research-only): ultra-long-context genome interpretation from raw reads

> **Mission (research-only):** Build a noise-aware, ultra-long-context genome foundation model that learns directly from raw sequencing reads (FASTQ + metadata; optional ONT raw signal) to support:
> - calibrated variant/structural evidence synthesis
> - disease-mechanism hypothesis generation
> - robust cross-platform concordance analysis (Illumina short reads vs PacBio HiFi vs Oxford Nanopore)
>
> **Not a gene-editing project.** This repository does not provide operational instructions for genome modification. Outputs are intended for research interpretation, reproducibility, and clinical-research interfacing.

## Why ultra-long context?
Many biologically meaningful signals are long-range: regulatory interactions, repeats, segmental duplications, complex regions, and structural context.
Classic attention scales quadratically; we instead use **state-space / long-convolution operators** and hierarchical memory for **10M–100M+ effective context**.

## High-level design
1. **Read-level denoising & error modeling**: learn P(true sequence | observed reads, quality, platform).
2. **Hierarchical genome memory**: tile embeddings + learned retrieval (10M–100M effective context).
3. **MoE experts**: specialized evidence models (splice, repeats/SV, coding constraint, platform error modes, etc.).
4. **Verifier loop**: second-pass model checks consistency + calibration; routes low-confidence regions back to experts.
5. **VCF v4.3 projection**: export standardized calls + uncertainty evidence for interoperability. (VCF spec: https://samtools.github.io/hts-specs/VCFv4.3.pdf)

## Supported modalities
- Illumina short-read WGS (FASTQ, paired-end)
- PacBio HiFi (FASTQ / BAM; high accuracy long reads)
- Oxford Nanopore (FASTQ; optional POD5/FAST5 signal for basecalling-aware modeling)

We explicitly support *cross-technology reconciliation* to produce a single consensus interpretation and confidence report.

## Accuracy philosophy
“98%+ correct” is defined per bounded task (e.g., high-confidence variant PPV, calibrated genotype quality, SV breakpoint tolerance),
with an explicit **abstain** option and confidence/evidence tracing.

## Getting started
This repo currently contains:
- architecture + training design docs (`docs/`)
- configs (`configs/`)
- schemas (`schemas/`)
- a cost estimator for Lambda B200 pricing (`scripts/cost_estimator.py`)

## Reading references
- Illumina Genomics Architecture v3 (official download page): https://www.illumina.com/downloads/illumina-genomics-architecture-v3-tech-note-m-gl-03657.html
- PacBio HiFi accuracy overview: https://www.pacb.com/technology/hifi-sequencing/
- Oxford Nanopore accuracy & Kit 14 duplex notes: https://nanoporetech.com/platform/accuracy and https://nanoporetech.com/document/kit-14-device-and-informatics
- VCF v4.3 spec: https://samtools.github.io/hts-specs/VCFv4.3.pdf

## License
Apache-2.0 (see LICENSE). Use is restricted by `SECURITY.md` and `docs/08_*` to research interpretation only.
