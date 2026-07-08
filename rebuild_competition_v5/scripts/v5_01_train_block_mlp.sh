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
  --seeds "${SEEDS:-3101,3102,3103}" \
  --hidden "${HIDDEN:-256}" \
  --epochs "${EPOCHS:-8}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --lr "${LR:-8e-4}" \
  train-block-mlp
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then
  "$PYTHON_BIN" - <<'PY'
import json, os, sys
from pathlib import Path
report = Path(os.environ.get("REPORTS", "reports")) / "v5_block_mlp_report.json"
if not report.exists():
    sys.exit(1)
data = json.loads(report.read_text())
ok = any(m.get("status") == "trained" and Path(m.get("checkpoint", "")).exists() for m in data.get("models", []))
if not ok:
    sys.exit(1)
print("[v5_01] python exited non-zero, but trained checkpoint/report are valid; continuing")
PY
fi
