#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
ACTION="${ACTION:-all}"

DATA_DIR="${DATA_DIR:-data_A}"
BASELINE_ROOT="${BASELINE_ROOT:-/home/ma-user/work/baseline_artifacts}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
REPORTS="${REPORTS:-reports}"
SUBMISSION="${SUBMISSION:-submission}"

SEED="${SEED:-3026}"
WORKERS="${WORKERS:-12}"
HISTORY_FRAC="${HISTORY_FRAC:-0.70}"
TRAIN_ROWS="${TRAIN_ROWS:-500000}"
VALID_ROWS="${VALID_ROWS:-80000}"
MAX_POOL="${MAX_POOL:-700}"
SVD_DIM="${SVD_DIM:-128}"
FIT_EDGE_LIMIT="${FIT_EDGE_LIMIT:-0}"
SRC_SEQ_LEN="${SRC_SEQ_LEN:-64}"
DST_SEQ_LEN="${DST_SEQ_LEN:-64}"
SEEDS="${SEEDS:-3101,3102,3103}"
HIDDEN="${HIDDEN:-256}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-2048}"
LR="${LR:-8e-4}"
REUSE_BASELINE_FEATURES="${REUSE_BASELINE_FEATURES:-0}"
BLEND_WEIGHT="${BLEND_WEIGHT:-0.10}"
OUTPUT_NAME="${OUTPUT_NAME:-result_final_blend_0p10}"
SWEEP_BLENDS="${SWEEP_BLENDS:-0.02,0.05,0.10,0.20,0.35,1.00}"

mkdir -p "$ARTIFACTS" "$REPORTS" "$SUBMISSION" logs

run_final() {
  "$PYTHON_BIN" -m src.final_pipeline \
    --data-dir "$DATA_DIR" \
    --baseline-root "$BASELINE_ROOT" \
    --artifacts "$ARTIFACTS" \
    --reports "$REPORTS" \
    --submission "$SUBMISSION" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --history-frac "$HISTORY_FRAC" \
    --train-rows "$TRAIN_ROWS" \
    --valid-rows "$VALID_ROWS" \
    --max-pool "$MAX_POOL" \
    --svd-dim "$SVD_DIM" \
    --fit-edge-limit "$FIT_EDGE_LIMIT" \
    --src-seq-len "$SRC_SEQ_LEN" \
    --dst-seq-len "$DST_SEQ_LEN" \
    --seeds "$SEEDS" \
    --hidden "$HIDDEN" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --predict-batch-size "$PREDICT_BATCH_SIZE" \
    --lr "$LR" \
    --reuse-baseline-features "$REUSE_BASELINE_FEATURES" \
    --blend-weight "$BLEND_WEIGHT" \
    --output-name "$OUTPUT_NAME" \
    --sweep-blends "$SWEEP_BLENDS" \
    "$1"
}

echo "[final] action=$ACTION start=$(date -Is)"
case "$ACTION" in
  all|build|train|predict|package|package-sweep)
    run_final "$ACTION"
    ;;
  *)
    echo "Unknown ACTION=$ACTION" >&2
    echo "Use one of: all, build, train, predict, package, package-sweep" >&2
    exit 2
    ;;
esac
echo "[final] action=$ACTION finish=$(date -Is)"
