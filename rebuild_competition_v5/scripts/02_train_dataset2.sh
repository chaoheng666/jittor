#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_ARG=""
JITTOR_ARG="--enable-jittor"
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_ARG="--smoke"
  JITTOR_ARG=""
fi
"$PYTHON_BIN" -m src.submit \
  ${SMOKE_ARG} \
  ${JITTOR_ARG} \
  --data-dir data_A \
  --teacher-zip /home/ma-user/work/jittor/result_pairwise_w05.zip \
  --artifacts artifacts \
  --reports reports \
  --submission submission \
  --seed "${SEED:-2026}" \
  --max-valid-events "${MAX_VALID_EVENTS:-30000}" \
  --jittor-train-rows "${JITTOR_TRAIN_ROWS:-80000}" \
  --jittor-epochs "${JITTOR_EPOCHS:-8}" \
  --jittor-hidden "${JITTOR_HIDDEN:-192}" \
  train-dataset --dataset dataset2

