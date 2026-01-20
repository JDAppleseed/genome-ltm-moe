#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/training_scale.yaml}
CHECKPOINT_DIR=${2:-runs/ckpt}

LATEST=$(ls -t "${CHECKPOINT_DIR}"/*.pt 2>/dev/null | head -n 1 || true)
if [[ -z "${LATEST}" ]]; then
  echo "No checkpoint found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

python -m genomaic.train.run --config "${CONFIG}" --resume "${LATEST}"
