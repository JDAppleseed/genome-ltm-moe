#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=${VENV_DIR:-.venv}

# shellcheck disable=SC1091
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source "${SCRIPT_DIR}/_venv_common.sh"

ensure_venv "${VENV_DIR}"
activate_venv "${VENV_DIR}"

python -m pip install --upgrade pip
install_editable "dev"

python -c "import sys; import yaml; print(sys.executable); print('PyYAML', yaml.__version__)"

echo "Next steps:"
echo "  scripts/dev/smoke_check.sh"
echo "Activate: source .venv/bin/activate"
echo "Optional: install torch for training (see https://pytorch.org/get-started/locally/)"
