# Protein design overview (AlphaFold-like in the safe sense)

This project can include a *separate* protein-design track that follows the common safe pattern:
**proposal → verification → ranking**, where "verification" uses structure prediction and confidence checks.

## Common research pipeline
1. Propose candidate backbones/structures (e.g., diffusion-style backbones).
2. Design sequences conditioned on backbone (ProteinMPNN-style autoregressive design).
3. Verify predicted structure and interaction plausibility (AlphaFold-like predictors), report confidence.
4. Rank with developability heuristics (stability proxies, aggregation risk proxies).

This repo provides only the *software architecture* boundary for the above; it does not provide wet-lab procedures.
