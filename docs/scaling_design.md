# Scaling + Distributed Training Design

## Phase 0 Decisions
- **Canonical entrypoint:** `python -m genomaic.train.run --config <yaml>`
- **Source of truth for distributed config:** `configs/training_scale.yaml`
  - Includes DeepSpeed config path, torchrun args, and SLURM resources.
- **Default MoE ZeRO stage:** ZeRO-2 (`configs/deepspeed/zero2_moe.json`).
- **Dense baseline ZeRO stage:** ZeRO-3 (`configs/deepspeed/zero3_dense.json`).

## File Map
- **Distributed init:** `src/genomaic/train/dist.py`
- **DeepSpeed engine + config:**
  - `src/genomaic/train/deepspeed_engine.py`
  - `src/genomaic/train/ds_config.py`
  - `configs/deepspeed/*.json`
- **Training loop:** `src/genomaic/train/run.py`
- **Data sharding + streaming:**
  - `src/genomaic/data/sharding.py`
  - `src/genomaic/data/manifest.py`
  - `src/genomaic/data/stream_fastq.py`
- **Checkpointing:** `src/genomaic/utils/ckpt.py`
- **MoE routing + utilization:**
  - `src/genomaic/models/moe_router.py`
  - `src/genomaic/models/moe_sequence.py`
  - `src/genomaic/eval/expert_utilization.py`
- **Launch scripts:** `scripts/launch/*`
- **SLURM templates:** `scripts/launch/sbatch_template.sbatch`
- **Mac control-plane scripts:** `scripts/remote/*`

## Notes
- The laptop is only a control plane; data staging must happen near compute.
- Deterministic sharding uses `rank` + `epoch` seeds and stable shard ordering.
- Checkpoints include RNG state for reproducibility and resume.
- Run `python scripts/dev/check_bidi_unicode.py` to ensure no bidi control characters slip into repo text files.
- Local dev setup: `scripts/dev/bootstrap_env.sh` (creates venv + `pip install -e .[dev]`) then `scripts/dev/smoke_check.sh`.
- Editable installs support `import genomaic` + `import genomeltm` without setting `PYTHONPATH`.
- Remote env setup: `scripts/remote/ensure_env.sh` for repeatable cluster installs.
