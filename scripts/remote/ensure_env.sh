#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=${VENV_DIR:-.venv}

if [[ ! -d "${VENV_DIR}" ]]; then
  python -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python - <<'PY'
try:
    import torch  # noqa: F401
    print("[info] torch available.")
except ModuleNotFoundError:
    print("[info] torch not installed. Install torch for training workloads.")
PY
