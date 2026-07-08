#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts reports submission
echo "[v4] start $(date -Is)"
echo "[v4] data_dir=${DATA_DIR:-data_A} train_rows=${TRAIN_ROWS:-90000} valid_rows=${VALID_ROWS:-18000} workers=${WORKERS:-8}"
bash scripts/v4_00_build_hard_mining.sh
bash scripts/v4_01_train_hard_mlp.sh
bash scripts/v4_02_train_id_ranker.sh
bash scripts/v4_03_predict_ensemble.sh
echo "[v4] finish $(date -Is)"
