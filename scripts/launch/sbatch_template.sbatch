#!/usr/bin/env bash
#SBATCH --job-name=genomaic-train
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

CONFIG=${1:-configs/training_scale.yaml}

mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module load cuda || echo "[warn] Could not load cuda module; ensure CUDA is available via container/modules."
else
  echo "[info] Environment modules not available; assuming CUDA provided by container/system."
fi

srun bash scripts/launch/torchrun_slurm.sh "${CONFIG}"
