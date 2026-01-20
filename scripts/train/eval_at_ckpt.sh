#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?checkpoint path required}
OUTPUT=${2:-runs/eval}

python -c "import torch; data=torch.load('${CHECKPOINT}', map_location='cpu'); print('Loaded checkpoint', data.get('step'))" \
  | tee "${OUTPUT}/ckpt_eval.txt"
