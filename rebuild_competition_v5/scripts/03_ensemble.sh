#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_ARG=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_ARG="--smoke"
fi
"$PYTHON_BIN" -m src.submit \
  ${SMOKE_ARG} \
  --data-dir data_A \
  --teacher-zip /home/ma-user/work/jittor/result_pairwise_w05.zip \
  --artifacts artifacts \
  --reports reports \
  --submission submission \
  --target-changes "${TARGET_CHANGES:-0.01,0.03,0.05,0.08,0.12}" \
  --package-prefix "${PACKAGE_PREFIX:-result_rebuild}" \
  ensemble

