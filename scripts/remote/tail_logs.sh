#!/usr/bin/env bash
set -euo pipefail

HOST=${1:?ssh host required}
LOG_PATH=${2:?log path required}

ssh "${HOST}" "tail -f ${LOG_PATH}"
