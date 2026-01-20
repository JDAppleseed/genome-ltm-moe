# Model architecture: GenomeLTM-MoE

## Design goals
- 10M–100M+ effective context (via hierarchy + memory + retrieval)
- Avoid quadratic attention at base resolution
- MoE decomposition + verifier loop for reliability
- Explicit uncertainty, abstention, and evidence tracing

## A) Read-level encoder (local)
Inputs per read:
- base tokens (A/C/G/T/N)
- quality embedding (Phred bins)
- platform embedding (Illumina/PacBio/ONT)
- positional embeddings within read (cycle index)

Backbone (v0):
- Conv stem (motif/error patterns)
- Short-range block: either small attention window OR small SSM/Hyena block
Outputs:
- per-base posterior logits
- read embedding summary
- error-mode embedding (learned latent)

## B) Tile encoder + hierarchical memory (mid + global)
We map reads into genomic tiles (1–8kb) using a *soft assignment* mechanism:
- early stage: overlap-based clustering / approximate mapping
- later stage: reference-aware anchoring (without collapsing uncertainty)

Tile backbone:
- SSM/Hyena-style blocks for long-range mixing with sub-quadratic scaling.
- Tile embeddings are stored in a memory table with keys for retrieval.

## C) MoE experts
Router inputs:
- tile embedding
- entropy/disagreement metrics across reads
- coverage signatures
- platform metadata

Experts (initial set):
- ErrorModeExpert (platform/systematic error)
- IndelHomopolymerExpert
- SpliceJunctionExpert
- CodingConstraintExpert
- RepeatSVExpert
- ImmuneComplexRegionExpert

Each expert returns:
- region-level evidence vector
- calibrated confidence
- evidence pointers and “why” features

## D) Verifier loop
A larger model consumes:
- expert outputs
- a targeted subset of raw reads
- long-range tile context via retrieval
Tasks:
- consistency checks (cross-expert, cross-tech)
- calibration correction (post-hoc or learned)
- uncertainty escalation (“send back for re-analysis”)
Stops when:
- confidence threshold met OR abstention is emitted.

## E) 10M–100M effective context
Achieved through:
- hierarchical tokenization (bases -> tiles -> supertiles)
- retrieval of relevant tiles/landmarks into the active context
- long-range SSM mixing on tile/supertile streams

Note: this is "effective context" (indexed + retrievable), not full in-graph attention.
