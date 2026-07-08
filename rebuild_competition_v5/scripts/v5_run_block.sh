#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts reports submission
echo "[v5] start $(date -Is)"
bash scripts/v5_00_build_block.sh
bash scripts/v5_01_train_block_mlp.sh
bash scripts/v5_02_predict_pack.sh
echo "[v5] finish $(date -Is)"
