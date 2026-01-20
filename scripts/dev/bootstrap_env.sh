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

python -c "import sys, yaml; print(sys.executable); print('PyYAML', yaml.__version__)"

echo "Next steps:"
echo "  scripts/dev/smoke_check.sh"
echo "Activate: source .venv/bin/activate"
echo "Optional: install torch for training (see https://pytorch.org/get-started/locally/)"
