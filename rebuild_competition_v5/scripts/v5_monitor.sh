#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== process =="
if [[ -f logs/v5_run.pid ]]; then
  ps -fp "$(cat logs/v5_run.pid)" || true
fi
pgrep -af "src.v5_block_pipeline|v5_run_block" || true
echo "== npu =="
command -v npu-smi >/dev/null 2>&1 && npu-smi info || true
echo "== reports =="
ls -lh reports/*v5* 2>/dev/null || true
echo "== submissions =="
ls -lh submission/result_v5*.zip 2>/dev/null || true
echo "== log =="
if [[ -L logs/v5_run_latest.log || -f logs/v5_run_latest.log ]]; then
  tail -n "${TAIL_N:-100}" logs/v5_run_latest.log
else
  ls -lt logs | head || true
fi
