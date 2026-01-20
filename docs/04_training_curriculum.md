# Training curriculum

## Stage 1: Denoising / corruption reversal (read-level)
Objective: learn P(true | observed) under realistic corruption.
Corruptions:
- substitutions conditioned on k-mer context
- indels bursts (platform dependent)
- quality degradation patterns
Losses:
- per-base denoising loss
- uncertainty calibration auxiliary loss

## Stage 2: Consensus formation (tile-level)
Objective: learn to fuse reads into tile representations with conflict awareness.
Losses:
- agreement maximization among consistent reads
- disagreement detection (predict when tile is ambiguous)

## Stage 3: Cross-tech alignment
Objective: align Illumina/PacBio/ONT evidence into a shared latent space.
Losses:
- contrastive alignment between techs on the same sample (where available)
- per-tech adversarial “invariance” where appropriate (preserve real differences)

## Stage 4: Expert training
Train experts with weak+strong labels:
- splice junctions (annotated)
- coding effects (constraint proxies)
- SV labels where curated truth sets exist
- platform error labels from controlled experiments

## Stage 5: Verifier training + abstention
Train verifier to:
- detect inconsistencies
- calibrate confidence
- abstain when evidence insufficient
