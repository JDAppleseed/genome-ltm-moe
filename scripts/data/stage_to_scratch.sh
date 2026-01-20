#!/usr/bin/env bash
set -euo pipefail

SRC=${1:?source path required}
DEST=${2:?destination path required}

if command -v rsync >/dev/null 2>&1; then
  rsync -av --progress "${SRC}" "${DEST}"
elif command -v rclone >/dev/null 2>&1; then
  rclone copy "${SRC}" "${DEST}"
elif command -v s5cmd >/dev/null 2>&1; then
  s5cmd cp "${SRC}" "${DEST}"
else
  echo "No staging tool found (rsync/rclone/s5cmd)." >&2
  exit 1
fi
