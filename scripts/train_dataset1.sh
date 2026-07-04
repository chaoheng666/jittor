#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" scripts/run_specialized_pipeline.py \
  --target dataset1 \
  --zero-other 1 \
  --data-dir "${DATA_DIR:-data_A}" \
  --artifact-root "${ARTIFACT_ROOT:-artifacts}" \
  --out-dir "${OUT_DIR:-submission_dataset1_probe}" \
  --zip "${ZIP_PATH:-result_best.zip}" \
  --report "${REPORT_PATH:-reports/dataset1_probe.json}" \
  --train "${TRAIN:-1}" \
  --predict "${PREDICT:-1}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --max-rows "${MAX_ROWS:-0}" \
  --cuda "${CUDA:-1}"

