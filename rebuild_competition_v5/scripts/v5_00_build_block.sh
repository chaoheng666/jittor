#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts reports submission logs
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m src.v5_block_pipeline \
  --data-dir "${DATA_DIR:-data_A}" \
  --v3-root "${V3_ROOT:-/home/ma-user/work/jittor_rebuild_v3}" \
  --artifacts "${ARTIFACTS:-artifacts}" \
  --reports "${REPORTS:-reports}" \
  --submission "${SUBMISSION:-submission}" \
  --seed "${SEED:-3026}" \
  --workers "${WORKERS:-12}" \
  --history-frac "${HISTORY_FRAC:-0.70}" \
  --train-rows "${TRAIN_ROWS:-500000}" \
  --valid-rows "${VALID_ROWS:-80000}" \
  --max-pool "${MAX_POOL:-700}" \
  --svd-dim "${SVD_DIM:-128}" \
  --fit-edge-limit "${FIT_EDGE_LIMIT:-0}" \
  --src-seq-len "${SRC_SEQ_LEN:-64}" \
  --dst-seq-len "${DST_SEQ_LEN:-64}" \
  build-block
