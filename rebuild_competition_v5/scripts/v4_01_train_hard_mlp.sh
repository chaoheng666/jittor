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
  --seeds "${HARD_SEEDS:-2027,2028,2029}" \
  --hard-hidden "${HARD_HIDDEN:-256}" \
  --epochs "${HARD_EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --lr "${LR:-8e-4}" \
  train-hard-mlp
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
import os
import sys
from pathlib import Path

report = Path(os.environ.get("REPORTS", "reports")) / "v4_hard_mlp_report.json"
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
print("[v4_01] python exited non-zero, but trained checkpoint/report are valid; continuing")
PY
fi
