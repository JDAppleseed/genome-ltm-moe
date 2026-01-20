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

module load cuda || true

srun bash scripts/launch/torchrun_slurm.sh "${CONFIG}"
