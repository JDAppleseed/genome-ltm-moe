# Cross-technology fusion (Illumina + PacBio + ONT)

We treat each sequencing technology as an independent noisy channel observing the same genome.

## Fusion approach (research)
For each candidate event (SNV/indel/SV):
- maintain per-tech support summaries (depth, posterior, error-mode scores)
- combine into a consensus posterior using a principled rule:
  - product-of-experts for concordant evidence
  - mixture-of-experts when one tech dominates (avoid overconfident fusion)
- preserve conflict as an explicit signal rather than smoothing it away

## Verifier integration
If cross-tech conflict is high:
- route to verifier
- verifier requests targeted re-analysis and may abstain
