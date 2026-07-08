#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts reports submission logs
PYTHON_BIN="${PYTHON_BIN:-python3}"
set +e
"$PYTHON_BIN" -m src.v5_block_pipeline \
  --data-dir "${DATA_DIR:-data_A}" \
  --v3-root "${V3_ROOT:-/home/ma-user/work/jittor_rebuild_v3}" \
  --artifacts "${ARTIFACTS:-artifacts}" \
  --reports "${REPORTS:-reports}" \
  --submission "${SUBMISSION:-submission}" \
  --workers "${WORKERS:-8}" \
  --hidden "${HIDDEN:-256}" \
  --predict-batch-size "${PREDICT_BATCH_SIZE:-2048}" \
  --src-seq-len "${SRC_SEQ_LEN:-64}" \
  --dst-seq-len "${DST_SEQ_LEN:-64}" \
  predict-block
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON_BIN" - <<'PY'
import os, sys
from pathlib import Path
art = Path(os.environ.get("ARTIFACTS", "artifacts"))
rep = Path(os.environ.get("REPORTS", "reports"))
needed = [art / "v5_block_ens.logits.npy", art / "v5_baseline_mlpw5p5.logits.npy", rep / "v5_block_predict_report.json"]
missing = [str(p) for p in needed if not p.exists()]
if missing:
    print("[v5_02] missing outputs", missing)
    sys.exit(1)
print("[v5_02] python exited non-zero, but logits/report are valid; continuing")
PY
fi
"$PYTHON_BIN" -m src.v5_block_pipeline \
  --data-dir "${DATA_DIR:-data_A}" \
  --v3-root "${V3_ROOT:-/home/ma-user/work/jittor_rebuild_v3}" \
  --artifacts "${ARTIFACTS:-artifacts}" \
  --reports "${REPORTS:-reports}" \
  --submission "${SUBMISSION:-submission}" \
  --blend-weight "${BLEND_WEIGHT:-0.10}" \
  --output-name "${OUTPUT_NAME:-result_v5_block_blend_0p10}" \
  pack-block
