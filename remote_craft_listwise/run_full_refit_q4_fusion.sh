#!/usr/bin/env bash
set -euo pipefail

source /home/ma-user/anaconda3/bin/activate Jittor-1.3.11.0

ROOT=/home/ma-user/work/craft_baseline_listwise_20260709
DATA_DIR=/home/ma-user/work/jittor_rebuild_v5/data_A
cd "$ROOT"

mkdir -p logs reports artifacts saved_models outputs/dataset1 outputs/dataset2

export cache_name="${CACHE_NAME:-craft_listwise_ce_20260709_bg}"
export JT_SYNC="${JT_SYNC:-0}"

RUN_NAME="${RUN_NAME:-full_q4_fusion_n64_e${FULL_EPOCHS:-2}}"
START_TS=$(date +%s)

echo "START_TS=$START_TS"
echo "START_TIME=$(date '+%F %T %z')"
echo "HOST=$(hostname)"
echo "PYTHON=$(which python)"
echo "cache_name=$cache_name"
echo "JT_SYNC=$JT_SYNC"
echo "RUN_NAME=$RUN_NAME"

if [ -f outputs/dataset2/dataset2_result.csv ] && [ ! -f outputs/dataset2/dataset2_result_quick_q4_fusion_backup.csv ]; then
  cp outputs/dataset2/dataset2_result.csv outputs/dataset2/dataset2_result_quick_q4_fusion_backup.csv
  echo "backed_up_quick_dataset2=outputs/dataset2/dataset2_result_quick_q4_fusion_backup.csv"
fi

python -u make_zero_dataset.py \
  --data_dir "$DATA_DIR" \
  --dataset dataset1 \
  --output_dir outputs \
  --report_dir reports

python -u dataset2_profile.py \
  --data_dir "$DATA_DIR" \
  --dataset dataset2 \
  --output reports/dataset2_profile_train_only_full_refit.json

echo "[full-refit] dataset2 q4 fusion"
python -u main.py \
  --mode refit \
  --predict \
  --dataset dataset2 \
  --data_dir "$DATA_DIR" \
  --save_dir saved_models \
  --output_dir outputs \
  --artifact_dir artifacts \
  --report_dir reports \
  --run_name "$RUN_NAME" \
  --epochs "${FULL_EPOCHS:-2}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
  --early_stop 1 \
  --num_neighbors "${NUM_NEIGHBORS:-64}" \
  --max_train_events 0 \
  --block_size "${BLOCK_SIZE:-50000}" \
  --src_history_neg_quota 4 \
  --score_mode fusion \
  --rebuild_cache

END_TS=$(date +%s)
echo "END_TS=$END_TS"
echo "END_TIME=$(date '+%F %T %z')"
echo "ELAPSED_SECONDS=$((END_TS-START_TS))"
echo "DONE"
