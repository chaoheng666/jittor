#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts reports submission logs
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m src.v4_pipeline \
  --data-dir "${DATA_DIR:-data_A}" \
  --v3-root "${V3_ROOT:-/home/ma-user/work/jittor_rebuild_v3}" \
  --artifacts "${ARTIFACTS:-artifacts}" \
  --reports "${REPORTS:-reports}" \
  --submission "${SUBMISSION:-submission}" \
  --seed "${SEED:-2026}" \
  --workers "${WORKERS:-8}" \
  --train-rows "${TRAIN_ROWS:-90000}" \
  --valid-rows "${VALID_ROWS:-18000}" \
  --max-pool "${MAX_POOL:-600}" \
  build-hard-mining
