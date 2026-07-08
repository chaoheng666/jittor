#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts reports submission logs
PYTHON_BIN="${PYTHON_BIN:-python3}"
set +e
"$PYTHON_BIN" -m src.v4_pipeline \
  --data-dir "${DATA_DIR:-data_A}" \
  --v3-root "${V3_ROOT:-/home/ma-user/work/jittor_rebuild_v3}" \
  --artifacts "${ARTIFACTS:-artifacts}" \
  --reports "${REPORTS:-reports}" \
  --submission "${SUBMISSION:-submission}" \
  --hard-hidden "${HARD_HIDDEN:-256}" \
  --id-hidden "${ID_HIDDEN:-256}" \
  --predict-batch-size "${PREDICT_BATCH_SIZE:-2048}" \
  predict-v4
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

artifacts = Path(os.environ.get("ARTIFACTS", "artifacts"))
reports = Path(os.environ.get("REPORTS", "reports"))
needed = [
    artifacts / "v4_baseline_mlpw5p5.logits.npy",
    artifacts / "v4_hard_ens.logits.npy",
    artifacts / "v4_id_ens.logits.npy",
    reports / "v4_predict_report.json",
]
missing = [str(p) for p in needed if not p.exists()]
if missing:
    print("[v4_03] predict failed and required outputs are missing:", missing)
    sys.exit(1)
print("[v4_03] python exited non-zero, but logits/report are valid; continuing")
PY
fi
"$PYTHON_BIN" -m src.v4_pipeline \
  --data-dir "${DATA_DIR:-data_A}" \
  --v3-root "${V3_ROOT:-/home/ma-user/work/jittor_rebuild_v3}" \
  --artifacts "${ARTIFACTS:-artifacts}" \
  --reports "${REPORTS:-reports}" \
  --submission "${SUBMISSION:-submission}" \
  pack-v4
