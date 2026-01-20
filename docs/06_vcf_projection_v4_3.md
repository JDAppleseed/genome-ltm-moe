# VCF v4.3 projection for GenomeLTM-MoE

## Why VCF?
VCF provides interoperability for:
- downstream interpretation pipelines
- benchmarking against truth sets
- sharing results with clinical-research collaborators

VCF structure (v4.3):
- meta-information lines (##)
- header line (#CHROM ...)
- tab-delimited records with REF/ALT/QUAL/FILTER/INFO/FORMAT + sample columns

VCF v4.3 spec: https://samtools.github.io/hts-specs/VCFv4.3.pdf

## Export philosophy
IHG (Internal Hypothesis Graph) is the native representation.
VCF is a *view* of IHG:
- each record corresponds to an IHG “event node”
- QUAL is derived from calibrated posterior
- INFO/FORMAT fields include evidence and uncertainty

## Key VCF v4.3 concepts used
- Structured header lines for INFO/FORMAT/FILTER definitions
- END tag for reference blocks / gVCF-like ranges (when emitting reference confidence)
- <*> symbolic allele convention for reference confidence blocks (gVCF convention described in spec)
- Consider BCF2 for scalable binary output (VCF text can be huge)

## What we add (custom INFO/FORMAT fields)
We define new tags with explicit Source/Version in the header.
Examples:
INFO:
- IHG_ID: stable internal node identifier
- POST: posterior probability of ALT (or event)
- UNC: uncertainty score (entropy / variance proxy)
- TECHSUP: per-technology support summary
- CONFLICT: boolean/score indicating cross-tech disagreement
- EVIDPTR: pointer(s) to internal evidence records

FORMAT (per sample):
- GT, GQ, DP (standard)
- PL (standard if computed)
- PSTRAND: strand-bias proxy from learned evidence
- PQ: learned per-sample quality scalar
- TECHDP: per-tech depths (Illumina/PacBio/ONT)

## Filters
- LOWCONF: below confidence threshold
- CONFLICT: cross-tech conflict not resolved
- LOWCOV: insufficient read depth or coverage
- COMPLEX: region complexity (repeats/immune locus) triggers caution

## Notes on performance
VCF spec notes that VCF text is large/slow; BCF2 is recommended for scale when feasible.
