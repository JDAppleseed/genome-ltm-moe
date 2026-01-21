# Configs overview

This directory contains YAML configuration files used across training, evaluation, and
pipeline scripts.

## Expectations

- **Top-level mapping**: each YAML file should be a mapping (dictionary).
- **Schema header**: include either `schema_version` or `$schema` at the top level.
- **Determinism**: prefer explicit values (avoid implicit defaults).

## Config inventory (selected)

- `tiles.yaml`: tile spec and retrieval defaults
- `fusion.yaml`: cross-tile fusion settings
- `platform_metadata.yaml`: platform embedding config
- `sv_evidence.yaml`: SV evidence head config
- `training_phase1_selfsup.yaml`: phase 1 self-supervised scaffold
- `training_phase2_retrieval.yaml`: phase 2 retrieval scaffold
- `training_phase3_local_encoder.yaml`: phase 3 local encoder scaffold
- `verifier_policy.yaml`: verifier rerun policy

## Validation

Use the validator script to check structural correctness and detect duplicate keys:

```bash
python scripts/validate_configs.py --all
```

The validator enforces:

- YAML parses cleanly
- no duplicate keys at any mapping level
- top-level mapping present

Warnings are emitted for missing schema headers.
