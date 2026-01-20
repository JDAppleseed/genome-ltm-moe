#!/usr/bin/env bash
set -euo pipefail

HOST=${1:?ssh host required}
REMOTE_PATH=${2:?remote repo path required}
SBATCH_ARGS=${3:-configs/training_scale.yaml}

ssh "${HOST}" "cd ${REMOTE_PATH} && sbatch scripts/launch/sbatch_template.sbatch ${SBATCH_ARGS}"
