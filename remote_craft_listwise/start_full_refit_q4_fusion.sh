#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/craft_baseline_listwise_20260709
chmod +x run_full_refit_q4_fusion.sh
rm -f logs/full_refit.pid logs/full_refit_latest.log

export CACHE_NAME=craft_listwise_ce_20260709_bg
export JT_SYNC=0
export FULL_EPOCHS=2
export BATCH_SIZE=128
export EVAL_BATCH_SIZE=128
export BLOCK_SIZE=50000
export NUM_NEIGHBORS=64
export RUN_NAME=full_q4_fusion_n64_e2

nohup bash run_full_refit_q4_fusion.sh > logs/full_refit_latest.log 2>&1 &
echo "$!" > logs/full_refit.pid
echo "started_pid=$(cat logs/full_refit.pid)"
