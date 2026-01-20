# Data modalities and cross-technology reconciliation

## Modalities
### Illumina (short reads)
- strengths: low per-base error, high throughput, mature ecosystem
- weaknesses: repeats/SV resolution, phasing limits
- input: paired-end FASTQ + quality + run metadata

### PacBio HiFi (long reads)
- strengths: long reads + ~99.9% accuracy claims; good for SV and haplotypes
- input: HiFi FASTQ/BAM
- reference: PacBio HiFi overview (external)

### Oxford Nanopore (ONT)
- strengths: very long reads, direct methylation signals, rapid sequencing
- weaknesses: historically higher error rates; improving with duplex + better basecallers
- input: FASTQ; optional POD5/FAST5 for basecalling-aware modeling

## Cross-tech reconciliation strategy
We treat each technology as an independent noisy channel observing an underlying genome.
We perform:
1) technology-specific evidence extraction (experts trained per modality)
2) shared latent genome representation in IHG (technology-agnostic)
3) consensus fusion with conflict detection:
   - “concordant call”: multiple techs agree, high confidence
   - “discordant call”: tech disagreement triggers verifier + targeted re-analysis
4) final VCF projection includes per-technology evidence subfields.

## Why this matters
Combining technologies is often the best route to resolve:
- complex repeats
- SV breakpoints
- difficult regions (segmental duplications, immune loci)
- phasing/haplotype structure
