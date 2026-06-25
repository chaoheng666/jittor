#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data_A}"
VALID_ROOT="${VALID_ROOT:-validation_luxury}"
MODEL_ROOT="${MODEL_ROOT:-luxury_models}"
SCORE_DIR="${SCORE_DIR:-luxury_scores}"
OUT_DIR="${OUT_DIR:-submission_luxury}"
ZIP_PATH="${ZIP_PATH:-result_luxury.zip}"

MAX_VALID="${MAX_VALID:-0}"
VALID_RATIO="${VALID_RATIO:-0.2}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-50}"
HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-100}"
HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-1000}"
HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-300}"

MLP_SEEDS="${MLP_SEEDS:-2026,2027,2028}"
MLP_HIDDEN_DIMS="${MLP_HIDDEN_DIMS:-64,128,256}"
MLP_WEIGHTS="${MLP_WEIGHTS:-0.1,0.2,0.3}"
MLP_EPOCHS="${MLP_EPOCHS:-10}"

SEQ_SEEDS="${SEQ_SEEDS:-2026,2027}"
SEQ_LENS="${SEQ_LENS:-30,50}"
SEQ_GAMMAS="${SEQ_GAMMAS:-0.1,0.2,0.3}"
SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-128}"
SEQ_EPOCHS="${SEQ_EPOCHS:-10}"

RUN_CRAFT="${RUN_CRAFT:-1}"
CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-30,50}"
CRAFT_HIDDEN_SIZES="${CRAFT_HIDDEN_SIZES:-64,128}"
CRAFT_EPOCHS="${CRAFT_EPOCHS:-6}"

BATCH_SIZE="${BATCH_SIZE:-256}"
CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-200}"
GPU_COUNT="${GPU_COUNT:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
USE_CUDA="${USE_CUDA:-1}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv_jittor}"

cd "$(dirname "$0")"

has_data() {
  [ -f "$DATA_DIR/dataset1/train.csv" ] && \
  [ -f "$DATA_DIR/dataset1/test.csv" ] && \
  [ -f "$DATA_DIR/dataset2/train.csv" ] && \
  [ -f "$DATA_DIR/dataset2/test.csv" ]
}

if ! has_data; then
  echo "data files are missing under $DATA_DIR"
  echo "run run_jittor_ranker.sh once, or set DATA_DIR to the downloaded data directory"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

PYTHON_BIN="python3"
if [ "$USE_VENV" = "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
fi

if ! $PYTHON_BIN - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("jittor") else 1)
PY
then
  "$PYTHON_BIN" -m pip install -U pip
  "$PYTHON_BIN" -m pip install -U jittor
fi

CUDA_ARG=""
if [ "$USE_CUDA" = "1" ]; then
  CUDA_ARG="--cuda"
fi

mkdir -p "$MODEL_ROOT" "$SCORE_DIR"

echo "building luxury validation sets"
for mode in mixed recent-heavy popular-heavy transition-heavy; do
  "$PYTHON_BIN" scripts/valid_builder.py \
    --data-dir "$DATA_DIR" \
    --out-dir "${VALID_ROOT}_${mode}" \
    --valid-ratio "$VALID_RATIO" \
    --max-valid "$MAX_VALID" \
    --hard-recent-limit "$HARD_RECENT_LIMIT" \
    --hard-transition-limit "$HARD_TRANSITION_LIMIT" \
    --hard-popular-limit "$HARD_POPULAR_LIMIT" \
    --hard-popular-sample "$HARD_POPULAR_SAMPLE" \
    --valid-mode "$mode"
done

MAIN_VALID_DIR="${VALID_ROOT}_mixed"
job_idx=0

run_gpu_job() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 5
  done
  local gpu=$((job_idx % GPU_COUNT))
  job_idx=$((job_idx + 1))
  if [ "$USE_CUDA" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$@" &
  else
    "$@" &
  fi
}

echo "training residual MLP models"
IFS=',' read -r -a mlp_hidden_dims <<< "$MLP_HIDDEN_DIMS"
IFS=',' read -r -a mlp_weights <<< "$MLP_WEIGHTS"
for hidden in "${mlp_hidden_dims[@]}"; do
  for weight in "${mlp_weights[@]}"; do
    tag="mlp_h${hidden}_w${weight//./p}"
    run_gpu_job "$PYTHON_BIN" scripts/train_ranker.py \
      --valid-dir "$MAIN_VALID_DIR" \
      --model-dir "$MODEL_ROOT/$tag" \
      --epochs "$MLP_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --hidden-dim "$hidden" \
      --mlp-weight "$weight" \
      --eval-ratio "$EVAL_RATIO" \
      --max-rows "$MAX_VALID" \
      --seed-list "$MLP_SEEDS" \
      $CUDA_ARG
  done
done
wait

echo "training sequence residual models"
IFS=',' read -r -a seq_lens <<< "$SEQ_LENS"
IFS=',' read -r -a seq_gammas <<< "$SEQ_GAMMAS"
for seq_len in "${seq_lens[@]}"; do
  for gamma in "${seq_gammas[@]}"; do
    tag="seq_l${seq_len}_g${gamma//./p}"
    run_gpu_job "$PYTHON_BIN" scripts/train_seq_ranker.py \
      --valid-dir "$MAIN_VALID_DIR" \
      --model-dir "$MODEL_ROOT/$tag" \
      --epochs "$SEQ_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --seq-len "$seq_len" \
      --hidden-dim "$SEQ_HIDDEN_DIM" \
      --gamma "$gamma" \
      --eval-ratio "$EVAL_RATIO" \
      --max-rows "$MAX_VALID" \
      --seed-list "$SEQ_SEEDS" \
      $CUDA_ARG
  done
done
wait

if [ "$RUN_CRAFT" = "1" ]; then
  echo "training CRAFT models"
  IFS=',' read -r -a craft_neighbors <<< "$CRAFT_NEIGHBORS"
  IFS=',' read -r -a craft_hidden_sizes <<< "$CRAFT_HIDDEN_SIZES"
  for neighbors in "${craft_neighbors[@]}"; do
    for hidden in "${craft_hidden_sizes[@]}"; do
      tag="craft_n${neighbors}_h${hidden}"
      run_gpu_job "$PYTHON_BIN" scripts/train_craft_ranker.py \
        --data-dir "$DATA_DIR" \
        --valid-dir "$MAIN_VALID_DIR" \
        --model-dir "$MODEL_ROOT/$tag" \
        --score-dir "$SCORE_DIR" \
        --run-name "$tag" \
        --epochs "$CRAFT_EPOCHS" \
        --batch-size "$CRAFT_BATCH_SIZE" \
        --num-neighbors "$neighbors" \
        --hidden-size "$hidden" \
        --max-rows "$MAX_VALID" \
        $CUDA_ARG
    done
  done
  wait
else
  echo "skip CRAFT models because RUN_CRAFT=$RUN_CRAFT"
fi

echo "searching ensemble weights"
"$PYTHON_BIN" scripts/search_ensemble.py \
  --valid-dir "$MAIN_VALID_DIR" \
  --model-root "$MODEL_ROOT" \
  --score-dir "$SCORE_DIR" \
  --out "$MODEL_ROOT/ensemble_weights.json" \
  --max-rows "$MAX_VALID"

echo "predicting luxury ensemble"
"$PYTHON_BIN" scripts/predict_luxury_ensemble.py \
  --data-dir "$DATA_DIR" \
  --weights "$MODEL_ROOT/ensemble_weights.json" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH"

echo "done: $ZIP_PATH"
