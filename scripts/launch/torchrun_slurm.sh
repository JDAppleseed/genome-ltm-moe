#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/training_scale.yaml}
NNODES=${NNODES:-${SLURM_NNODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}
if [[ -z \"${MASTER_ADDR:-}\" && -n \"${SLURM_NODELIST:-}\" ]]; then
  MASTER_ADDR=$(scontrol show hostnames \"${SLURM_NODELIST}\" | head -n 1)
fi
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  -m genomaic.train.run \
  --config "${CONFIG}"
