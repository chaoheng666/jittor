#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-48}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-48}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-48}"

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
  --batch-size "${BATCH_SIZE:-1024}" \
  --max-rows "${MAX_ROWS:-0}" \
  --seed "${SEED:-2026}" \
  --d2-model-type "${D2_MODEL_TYPE:-temporal}" \
  --d2-feature-model-kind "${D2_FEATURE_MODEL_KIND:-jittor_mlp}" \
  --d2-feature-max-train-events "${D2_FEATURE_MAX_TRAIN_EVENTS:-120000}" \
  --d2-feature-auto-weight "${D2_FEATURE_AUTO_WEIGHT:-1}" \
  --d2-listwise-neg-count "${D2_LISTWISE_NEG_COUNT:-99}" \
  --d2-listwise-max-train-events "${D2_LISTWISE_MAX_TRAIN_EVENTS:-160000}" \
  --d2-listwise-hidden-dim "${D2_LISTWISE_HIDDEN_DIM:-160}" \
  --d2-listwise-epochs "${D2_LISTWISE_EPOCHS:-10}" \
  --d2-listwise-batch-size "${D2_LISTWISE_BATCH_SIZE:-4096}" \
  --d2-listwise-lr "${D2_LISTWISE_LR:-0.001}" \
  --d2-listwise-margin-weight "${D2_LISTWISE_MARGIN_WEIGHT:-0.05}" \
  --d2-listwise-new-pair-only "${D2_LISTWISE_NEW_PAIR_ONLY:-1}" \
  --d2-pairwise-neg-count "${D2_PAIRWISE_NEG_COUNT:-8}" \
  --d2-pairwise-max-train-events "${D2_PAIRWISE_MAX_TRAIN_EVENTS:-400000}" \
  --d2-softmax-mode "${D2_SOFTMAX_MODE:-sampled}" \
  --d2-temporal-max-train-events "${D2_TEMPORAL_MAX_TRAIN_EVENTS:-0}" \
  --d2-neg-count "${D2_NEG_COUNT:-4096}" \
  --d2-seq-len "${D2_SEQ_LEN:-160}" \
  --d2-emb-dim "${D2_EMB_DIM:-128}" \
  --d2-hidden-dim "${D2_HIDDEN_DIM:-256}" \
  --d2-dropout "${D2_DROPOUT:-0.1}" \
  --d2-epochs "${D2_EPOCHS:-10}" \
  --d2-batch-size "${D2_BATCH_SIZE:-2048}" \
  --d2-lr "${D2_LR:-0.001}" \
  --d2-weight-decay "${D2_WEIGHT_DECAY:-0.000001}" \
  --d2-bpr-weight "${D2_BPR_WEIGHT:-0.05}" \
  --d2-all-dst-weight "${D2_ALL_DST_WEIGHT:-0.20}" \
  --d2-hard-negative-count "${D2_HARD_NEGATIVE_COUNT:-512}" \
  --d2-sampled-correction "${D2_SAMPLED_CORRECTION:-1}" \
  --d2-rerank-neg-count "${D2_RERANK_NEG_COUNT:-99}" \
  --d2-rerank-weight "${D2_RERANK_WEIGHT:-1.00}" \
  --d2-fusion-model-weight "${D2_FUSION_MODEL_WEIGHT:-0.05}" \
  --d2-fusion-rule-weight "${D2_FUSION_RULE_WEIGHT:-1.0}" \
  --d2-include-test-vocab "${D2_INCLUDE_TEST_VOCAB:-1}" \
  --d2-unknown-policy "${D2_UNKNOWN_POLICY:-neutral}" \
  --d2-unknown-score "${D2_UNKNOWN_SCORE:-0.0}" \
  --d2-unknown-margin "${D2_UNKNOWN_MARGIN:-0.0}" \
  --d2-cold-prior-weight "${D2_COLD_PRIOR_WEIGHT:-0.0}" \
  --d2-valid-max-events "${D2_VALID_MAX_EVENTS:-20000}" \
  --d2-validate-before-final "${D2_VALIDATE_BEFORE_FINAL:-0}"
