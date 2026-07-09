#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/craft_baseline_listwise_20260709
chmod +x run_listwise_workflow.sh
rm -f logs/workflow.pid logs/workflow_latest.log

export CACHE_NAME=craft_listwise_ce_20260709_bg
export JT_SYNC=0
export SMOKE_MAX_TRAIN_EVENTS=128
export SMOKE_MAX_VAL_EVENTS=64
export SMOKE_BATCH_SIZE=8
export SMOKE_EVAL_BATCH_SIZE=8
export VALID_EPOCHS=1
export EARLY_STOP=1
export MAX_TRAIN_EVENTS=10000
export MAX_VAL_EVENTS=5000
export FINAL_EPOCHS=1
export FINAL_MAX_TRAIN_EVENTS=50000
export BATCH_SIZE=64
export EVAL_BATCH_SIZE=64
export BLOCK_SIZE=20000

nohup bash run_listwise_workflow.sh > logs/workflow_latest.log 2>&1 &
echo "$!" > logs/workflow.pid
echo "started_pid=$(cat logs/workflow.pid)"
