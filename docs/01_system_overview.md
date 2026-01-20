# System overview

## Pipeline summary (conceptual)
Raw reads (FASTQ + quality + platform metadata)
  -> Read-level encoder (noise-aware denoising objective)
  -> Tiling + hierarchical memory (genome-scale)
  -> MoE experts (domain-specific evidence models)
  -> Verifier (consistency + calibration pass)
  -> Internal Hypothesis Graph (IHG): probabilistic evidence store
  -> VCF v4.3 projection for interoperability

## Key internal artifact: Internal Hypothesis Graph (IHG)
A structured representation of:
- candidate loci/events (SNV/indel/SV/repeat expansions)
- posterior probabilities, calibrated confidence
- evidence pointers to reads/tiles/platforms
- expert attributions and “disagreement signatures”

IHG is *not* a VCF; VCF is an export view.
