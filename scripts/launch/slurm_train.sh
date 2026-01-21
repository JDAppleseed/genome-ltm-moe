#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "SLURM_SUBMIT_DIR is not set. Run within an sbatch context." >&2
  exit 1
fi

CONFIG_PATH="${1:-}"
RUN_NAME="${2:-genomaic-run}"
OUTPUT_DIR="${3:-runs}"

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: $0 <config_path> [run_name] [output_dir]" >&2
  exit 1
fi

SBATCH_TEMPLATE="${SLURM_SUBMIT_DIR}/scripts/launch/sbatch_template.sbatch"
if [[ ! -f "${SBATCH_TEMPLATE}" ]]; then
  echo "Missing sbatch template: ${SBATCH_TEMPLATE}" >&2
  exit 1
fi

sbatch \
  --job-name="${RUN_NAME}" \
  --output="${OUTPUT_DIR}/${RUN_NAME}/slurm-%j.out" \
  --error="${OUTPUT_DIR}/${RUN_NAME}/slurm-%j.err" \
  "${SBATCH_TEMPLATE}" \
  "${CONFIG_PATH}" \
  "${RUN_NAME}" \
  "${OUTPUT_DIR}"
