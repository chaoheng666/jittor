#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" scripts/run_specialized_pipeline.py \
  --target all \
  --zero-other 0 \
  --data-dir "${DATA_DIR:-data_A}" \
  --artifact-root "${ARTIFACT_ROOT:-artifacts}" \
  --out-dir "${OUT_DIR:-submission_specialized}" \
  --zip "${ZIP_PATH:-result_best.zip}" \
  --report "${REPORT_PATH:-reports/specialized_pipeline.json}" \
  --train "${TRAIN:-1}" \
  --predict "${PREDICT:-1}" \
  --final-train "${FINAL_TRAIN:-1}" \
  --cuda "${CUDA:-1}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --max-rows "${MAX_ROWS:-0}" \
  --seed "${SEED:-2026}" \
  --d2-softmax-mode "${D2_SOFTMAX_MODE:-sampled}" \
  --d2-neg-count "${D2_NEG_COUNT:-4096}" \
  --d2-seq-len "${D2_SEQ_LEN:-80}" \
  --d2-emb-dim "${D2_EMB_DIM:-96}" \
  --d2-hidden-dim "${D2_HIDDEN_DIM:-192}" \
  --d2-dropout "${D2_DROPOUT:-0.1}" \
  --d2-epochs "${D2_EPOCHS:-6}" \
  --d2-batch-size "${D2_BATCH_SIZE:-512}" \
  --d2-lr "${D2_LR:-0.001}" \
  --d2-weight-decay "${D2_WEIGHT_DECAY:-0.000001}" \
  --d2-bpr-weight "${D2_BPR_WEIGHT:-0.05}" \
  --d2-fusion-model-weight "${D2_FUSION_MODEL_WEIGHT:-1.0}" \
  --d2-fusion-rule-weight "${D2_FUSION_RULE_WEIGHT:-0.25}" \
  --d2-valid-max-events "${D2_VALID_MAX_EVENTS:-20000}" \
  --d2-validate-before-final "${D2_VALIDATE_BEFORE_FINAL:-1}"
