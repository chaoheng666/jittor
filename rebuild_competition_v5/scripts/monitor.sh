#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== time =="
date -Is
echo
echo "== process =="
if [[ -f logs/overnight.pid ]]; then
  PID="$(cat logs/overnight.pid)"
  ps -fp "${PID}" || true
else
  pgrep -af "src.submit|run_overnight" || true
fi
echo
echo "== accelerator =="
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
elif command -v ascend-smi >/dev/null 2>&1; then
  ascend-smi info || true
else
  echo "npu-smi/ascend-smi not found"
fi
echo
echo "== memory/cpu =="
free -h || true
top -b -n 1 | head -n 20 || true
echo
echo "== recent logs =="
ls -lh logs || true
LATEST_LOG="$(ls -t logs/overnight_*.log 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_LOG}" ]]; then
  echo "--- ${LATEST_LOG} ---"
  tail -n 80 "${LATEST_LOG}"
fi
echo
echo "== reports =="
python3 - <<'PY' || true
import json
from pathlib import Path
for name in ["dataset1_train_report.json", "dataset2_train_report.json", "dataset1_predict_report.json", "dataset2_predict_report.json", "ensemble_manifest.json", "pack_check.json"]:
    p = Path("reports") / name
    if not p.exists():
        continue
    obj = json.loads(p.read_text())
    print(f"-- {name} --")
    if "aggregate_mrr" in obj:
        print({"aggregate_mrr": obj.get("aggregate_mrr"), "by_set_mrr": obj.get("by_set_mrr"), "weights": obj.get("weights"), "jittor": obj.get("jittor", {}).get("status")})
    elif "top1_stats" in obj:
        print(obj["top1_stats"])
    elif "packages" in obj:
        for k, v in obj["packages"].items():
            print(k, {"zip": v.get("zip"), "alpha": v.get("alpha_dataset2"), "top1": v.get("actual_top1_change_vs_teacher_dataset2")})
    else:
        print(str(obj)[:1000])
PY
echo
echo "== zips =="
find submission -maxdepth 1 -name '*.zip' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort || true

