#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/training_scale.yaml}
CHECKPOINT_DIR=${2:-runs/ckpt}

# Linux cluster-friendly: sort by mtime without relying on ls expansion.
LATEST=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "*.pt" -type f -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d" " -f2-)
if [[ -z "${LATEST}" ]]; then
  echo "No checkpoint found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

python -m genomaic.train.run --config "${CONFIG}" --resume "${LATEST}"
