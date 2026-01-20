#!/usr/bin/env bash
set -euo pipefail

HOST=${1:?ssh host required}
REMOTE_PATH=${2:?remote artifacts path required}
LOCAL_DEST=${3:?local destination required}

rsync -av "${HOST}:${REMOTE_PATH}" "${LOCAL_DEST}"
