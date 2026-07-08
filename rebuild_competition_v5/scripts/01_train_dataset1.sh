#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_ARG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_ARG="--smoke"
fi
"$PYTHON_BIN" -m src.submit \
  ${SMOKE_ARG} \
  --data-dir data_A \
  --teacher-zip /home/ma-user/work/jittor/result_pairwise_w05.zip \
  --artifacts artifacts \
  --reports reports \
  --submission submission \
  --seed "${SEED:-2026}" \
  train-dataset --dataset dataset1

