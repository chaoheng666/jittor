#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== process =="
if [[ -f logs/v4_run.pid ]]; then
  PID="$(cat logs/v4_run.pid)"
  ps -fp "$PID" || true
else
  pgrep -af "src.v4_pipeline|v4_run_fast" || true
fi
echo "== npu =="
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
else
  echo "npu-smi not found"
fi
echo "== memory =="
free -h || true
echo "== latest log =="
LOG="${1:-}"
if [[ -z "$LOG" && -L logs/v4_run_latest.log ]]; then
  LOG="logs/v4_run_latest.log"
fi
if [[ -n "$LOG" && -f "$LOG" ]]; then
  tail -n "${TAIL_N:-80}" "$LOG"
else
  ls -lt logs | head || true
fi
echo "== reports =="
ls -lh reports/*v4* 2>/dev/null || true
echo "== submissions =="
ls -lh submission/result_v4*.zip 2>/dev/null || true
