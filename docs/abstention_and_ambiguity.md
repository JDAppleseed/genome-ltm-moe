# Abstention and Ambiguity

Some genome regions are intrinsically ambiguous given available evidence (coverage, repeats, segmental duplications).
The system must treat ambiguity as a *first-class* output.

## Ambiguity Expert
A dedicated module estimates:
- mapping ambiguity proxies
- repeat complexity signatures
- cross-tech disagreement likelihood
- posterior entropy / variance

## Abstention policy (research)
If confidence < threshold or ambiguity > threshold:
- do not force a single call
- emit an "ABSTAIN" state in the internal hypothesis graph (IHG)
- project to VCF with FILTER=LOWCONF or FILTER=COMPLEX/CONFLICT and include UNC/CONFLICT tags

## Why this matters
Abstention prevents "hallucinated certainty" and makes downstream medical research safer.
