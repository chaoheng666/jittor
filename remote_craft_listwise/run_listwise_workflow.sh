#!/usr/bin/env bash
set -euo pipefail

source /home/ma-user/anaconda3/bin/activate Jittor-1.3.11.0

ROOT=/home/ma-user/work/craft_baseline_listwise_20260709
DATA_DIR=/home/ma-user/work/jittor_rebuild_v5/data_A
cd "$ROOT"

mkdir -p logs reports artifacts saved_models outputs

export cache_name="${CACHE_NAME:-craft_listwise_ce_20260709}"
export JT_SYNC="${JT_SYNC:-0}"

START_TS=$(date +%s)
echo "START_TS=$START_TS"
echo "START_TIME=$(date '+%F %T %z')"
echo "HOST=$(hostname)"
echo "PYTHON=$(which python)"
echo "cache_name=$cache_name"
echo "JT_SYNC=$JT_SYNC"

python -u dataset2_profile.py \
  --data_dir "$DATA_DIR" \
  --dataset dataset2 \
  --output reports/dataset2_profile_train_only.json

echo "[smoke] quota=0"
JT_SYNC=1 python -u main.py \
  --mode validate \
  --dataset dataset2 \
  --data_dir "$DATA_DIR" \
  --save_dir saved_models \
  --output_dir outputs \
  --artifact_dir artifacts \
  --report_dir reports \
  --run_name smoke_q0 \
  --epochs 1 \
  --batch_size "${SMOKE_BATCH_SIZE:-16}" \
  --eval_batch_size "${SMOKE_EVAL_BATCH_SIZE:-16}" \
  --early_stop 1 \
  --num_neighbors 64 \
  --max_train_events "${SMOKE_MAX_TRAIN_EVENTS:-256}" \
  --max_val_events "${SMOKE_MAX_VAL_EVENTS:-128}" \
  --block_size 50000 \
  --src_history_neg_quota 0 \
  --rebuild_cache \
  --sync_each_batch

echo "[validate] quota=0"
python -u main.py \
  --mode validate \
  --dataset dataset2 \
  --data_dir "$DATA_DIR" \
  --save_dir saved_models \
  --output_dir outputs \
  --artifact_dir artifacts \
  --report_dir reports \
  --run_name valid_q0_n64 \
  --epochs "${VALID_EPOCHS:-6}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
  --early_stop "${EARLY_STOP:-2}" \
  --num_neighbors 64 \
  --max_train_events "${MAX_TRAIN_EVENTS:-200000}" \
  --max_val_events "${MAX_VAL_EVENTS:-50000}" \
  --block_size "${BLOCK_SIZE:-50000}" \
  --src_history_neg_quota 0 \
  --rebuild_cache

echo "[validate] quota=4"
python -u main.py \
  --mode validate \
  --dataset dataset2 \
  --data_dir "$DATA_DIR" \
  --save_dir saved_models \
  --output_dir outputs \
  --artifact_dir artifacts \
  --report_dir reports \
  --run_name valid_q4_n64 \
  --epochs "${VALID_EPOCHS:-6}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
  --early_stop "${EARLY_STOP:-2}" \
  --num_neighbors 64 \
  --max_train_events "${MAX_TRAIN_EVENTS:-200000}" \
  --max_val_events "${MAX_VAL_EVENTS:-50000}" \
  --block_size "${BLOCK_SIZE:-50000}" \
  --src_history_neg_quota 4 \
  --rebuild_cache

BEST_RUN=$(python - <<'PY'
import json
from pathlib import Path
candidates = []
for name in ["valid_q0_n64", "valid_q4_n64"]:
    p = Path("reports") / f"{name}_report.json"
    data = json.loads(p.read_text())
    train = data.get("train", {})
    score = train.get("best_metric", -1)
    score_mode = train.get("best_score_mode", "craft")
    if score_mode not in {"craft", "fusion"}:
        score_mode = "craft"
    candidates.append((score, name, data["args"]["src_history_neg_quota"], score_mode))
candidates.sort(reverse=True)
score, name, quota, score_mode = candidates[0]
Path("reports/selected_config.json").write_text(json.dumps({
    "selected_run": name,
    "best_metric": score,
    "score_mode": score_mode,
    "src_history_neg_quota": quota,
    "num_neighbors": 64
}, indent=2, sort_keys=True) + "\n")
print(name)
PY
)

BEST_QUOTA=$(python - <<'PY'
import json
print(json.loads(open("reports/selected_config.json").read())["src_history_neg_quota"])
PY
)

BEST_SCORE_MODE=$(python - <<'PY'
import json
print(json.loads(open("reports/selected_config.json").read())["score_mode"])
PY
)

echo "[selected] run=$BEST_RUN quota=$BEST_QUOTA score_mode=$BEST_SCORE_MODE"

echo "[refit] all train + predict"
python -u main.py \
  --mode refit \
  --predict \
  --dataset dataset2 \
  --data_dir "$DATA_DIR" \
  --save_dir saved_models \
  --output_dir outputs \
  --artifact_dir artifacts \
  --report_dir reports \
  --run_name final_n64_q${BEST_QUOTA} \
  --epochs "${FINAL_EPOCHS:-6}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-128}" \
  --early_stop 1 \
  --num_neighbors 64 \
  --max_train_events "${FINAL_MAX_TRAIN_EVENTS:-0}" \
  --block_size "${BLOCK_SIZE:-50000}" \
  --src_history_neg_quota "$BEST_QUOTA" \
  --score_mode "${SCORE_MODE:-$BEST_SCORE_MODE}" \
  --rebuild_cache

END_TS=$(date +%s)
echo "END_TS=$END_TS"
echo "END_TIME=$(date '+%F %T %z')"
echo "ELAPSED_SECONDS=$((END_TS-START_TS))"
echo "DONE"
