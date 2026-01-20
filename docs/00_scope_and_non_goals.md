# Scope and non-goals

## In scope (research-only)
- Learning robust, calibrated genome evidence from raw reads (FASTQ; optional ONT raw signal)
- Cross-platform concordance: Illumina vs PacBio vs ONT
- Variant and structural evidence synthesis with uncertainty quantification
- Standardized export (VCF v4.3 / BCF) with rich evidence fields
- Disease-mechanism hypothesis generation (e.g., identifying likely causal variants, pathways, regulatory disruptions)

## Out of scope (non-goals)
- Operational guidance for genome editing or “trait insertion/removal”
- “Generate an organism/genome from a prompt”
- Clinical decision-making without validation (this repo is for research workflows)

## Safety boundary
We treat outputs as **hypotheses + evidence** intended for expert review and downstream validation.
All design encourages calibrated uncertainty, abstention, and reproducibility.
