#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m generative_idps train-coordinate \
  --input-pdb "${ROOT_DIR}/Implementation_codes/polyq_1000frames.pdb" \
  --output-dir "${ROOT_DIR}/outputs/polyq_demo" \
  --latent-dim 13 \
  --test-size 200 \
  --epochs 5 \
  --batch-size 40 \
  --sample-count 50 \
  --label polyq
