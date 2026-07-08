#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m src.submit \
  --data-dir data_A \
  --teacher-zip /home/ma-user/work/jittor/result_pairwise_w05.zip \
  --artifacts artifacts \
  --reports reports \
  --submission submission \
  profile

