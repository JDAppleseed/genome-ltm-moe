#!/usr/bin/env bash
set -euo pipefail

HOST=${1:?ssh host required}
DEST=${2:?destination path required}

rsync -av --exclude '.git' --exclude 'data' --exclude 'runs' ./ "${HOST}:${DEST}"
