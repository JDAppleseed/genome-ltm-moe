#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[warn] Not in a virtualenv. If deps missing, run scripts/dev/bootstrap_env.sh and activate .venv"
fi

python -m compileall src scripts

python - <<'PY'
try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    print("PyYAML missing. Run scripts/dev/bootstrap_env.sh")
    raise SystemExit(1)
PY

python -c "import yaml; yaml.safe_load(open('configs/agentic.yaml'))"
python -c "import yaml; yaml.safe_load(open('configs/verifier_policy.yaml'))"
python scripts/dev/check_bidi_unicode.py

python - <<'PY'
import importlib.util
if importlib.util.find_spec("pytest") is None:
    print("pytest missing. Run scripts/dev/bootstrap_env.sh")
    raise SystemExit(1)
PY

pytest -q
