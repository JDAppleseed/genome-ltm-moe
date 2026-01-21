#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[warn] Not in a virtualenv. If deps missing, run scripts/dev/bootstrap_env.sh and activate .venv"
fi

python -m compileall src scripts

export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"

python - <<'PY'
try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    print("PyYAML missing. Run scripts/dev/bootstrap_env.sh")
    raise SystemExit(1)
PY

python scripts/dev/check_bidi_unicode.py

python - <<'PY'
import importlib.util
if importlib.util.find_spec("pytest") is None:
    print("pytest missing. Run scripts/dev/bootstrap_env.sh")
    raise SystemExit(1)
PY

pytest -q

echo "[info] Scenario 1: Tile generation determinism"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
printf "chr1\t100\n" > "${tmp_dir}/contigs.tsv"
python scripts/data/build_tiles.py --config configs/tiles.yaml --contigs "${tmp_dir}/contigs.tsv" --out-dir "${tmp_dir}/out1"
python scripts/data/build_tiles.py --config configs/tiles.yaml --contigs "${tmp_dir}/contigs.tsv" --out-dir "${tmp_dir}/out2"
sha1="$(sha256sum "${tmp_dir}/out1/tiles.jsonl" | awk '{print $1}')"
sha2="$(sha256sum "${tmp_dir}/out2/tiles.jsonl" | awk '{print $1}')"
if [[ "${sha1}" != "${sha2}" ]]; then
  echo "Tile generation is not deterministic" >&2
  exit 1
fi

echo "[info] Scenario 2: LocalEncoder forward (skip if torch missing)"
python - <<'PY'
import importlib.util
if importlib.util.find_spec("torch") is None:
    print("torch missing, skipping LocalEncoder scenario")
    raise SystemExit(0)
import torch
from genomaic.models.local_encoder import LocalEncoder, LocalEncoderConfig
encoder = LocalEncoder(LocalEncoderConfig(d_model=8, n_layers=2, mode="ssm_stub", pool="mean"))
x = torch.randn(2, 5, 8)
out = encoder(x)
assert out.shape == (2, 8)
PY

echo "[info] Scenario 3: Retrieval index round-trip"
python - <<'PY'
import numpy as np
from genomaic.retrieval.index import StubANNIndex
index = StubANNIndex(dim=2)
index.add(np.array([[1.0, 0.0]], dtype=np.float32), [0])
index.build()
results = index.query(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
assert results and results[0].tile_index == 0
PY
