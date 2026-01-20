#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/training_scale.yaml}
NPROC=${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}

torchrun \
  --nproc_per_node "${NPROC}" \
  -m genomaic.train.run \
  --config "${CONFIG}"
