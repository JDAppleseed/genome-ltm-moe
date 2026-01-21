#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-}"
RUN_NAME="${2:-genomaic-run}"
OUTPUT_DIR="${3:-runs}"

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: $0 <config_path> [run_name] [output_dir]" >&2
  exit 1
fi

cat <<EOF
GCP Batch stub:
- Config: ${CONFIG_PATH}
- Run name: ${RUN_NAME}
- Output dir: ${OUTPUT_DIR}

TODO: Implement job submission with config-driven data paths and object storage checkpoints.
EOF
