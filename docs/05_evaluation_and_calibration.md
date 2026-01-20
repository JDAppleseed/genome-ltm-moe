# Evaluation and calibration

## Primary metrics (task-bound)
- SNV/indel: PPV/TPR at high-confidence thresholds; calibration error (ECE)
- SV: breakpoint tolerance metrics, size stratification
- Phasing/haplotypes: switch error rate (if truth available)
- Cross-tech concordance: agreement rate stratified by region type

## Reliability requirements
- Calibrated confidence with abstention
- Evidence tracing: per-call pointers to supporting reads/tiles and per-tech contributions
- Ancestry and platform robustness: evaluate across diverse cohorts and sequencing configurations

## What "98%+" means here
Not “98% of all biology.”
Instead:
- “98% PPV at confidence ≥ X” for defined call classes
- “ECE ≤ Y” on held-out truth sets
- explicit abstention with quantifiable coverage
