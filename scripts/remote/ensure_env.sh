#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=${VENV_DIR:-.venv}

# shellcheck disable=SC1091
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/../dev/_venv_common.sh"

ensure_venv "${VENV_DIR}"
activate_venv "${VENV_DIR}"

python -m pip install --upgrade pip
install_editable "dev"

python - <<'PY'
try:
    import torch  # noqa: F401
    print("[info] torch available.")
except ModuleNotFoundError:
    print("[info] torch not installed. Install torch for training workloads.")
PY
