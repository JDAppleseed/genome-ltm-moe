#!/usr/bin/env bash

ensure_venv() {
  local venv_dir="${1:-${VENV_DIR:-.venv}}"

  if [[ ! -d "${venv_dir}" ]]; then
    python -m venv "${venv_dir}"
  fi
}

activate_venv() {
  local venv_dir="${1:-${VENV_DIR:-.venv}}"

  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"
}

install_editable() {
  local dev_extra="${1:-dev}"

  python -m pip install -e ".[${dev_extra}]"
}
