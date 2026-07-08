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
  --seeds "${ID_SEEDS:-2031,2032}" \
  --id-hidden "${ID_HIDDEN:-256}" \
  --emb-dim "${EMB_DIM:-32}" \
  --epochs "${ID_EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr "${LR:-8e-4}" \
  train-id-ranker
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
from pathlib import Path

report = Path(os.environ.get("REPORTS", "reports")) / "v4_id_ranker_report.json"
if not report.exists():
    sys.exit(1)
data = json.loads(report.read_text())
ok = False
for model in data.get("models", []):
    ckpt = model.get("checkpoint")
    if model.get("status") == "trained" and ckpt and Path(ckpt).exists():
        ok = True
if not ok:
    sys.exit(1)
print("[v4_02] python exited non-zero, but trained checkpoint/report are valid; continuing")
PY
fi
