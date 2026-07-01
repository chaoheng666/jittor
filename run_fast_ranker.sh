#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data_A}"
DATA_URL="${DATA_URL:-https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads}"
FOLD_ROOT="${FOLD_ROOT:-rank_folds_fast}"
CACHE_DIR="${CACHE_DIR:-feature_cache_fast}"
MODEL_ROOT="${MODEL_ROOT:-fast_models}"
SCORE_ROOT="${SCORE_ROOT:-fast_scores}"
OUT_DIR="${OUT_DIR:-submission_fast}"
ZIP_PATH="${ZIP_PATH:-result_fast.zip}"

SEED="${SEED:-2026}"
MAX_VALID="${MAX_VALID:-150000}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
FUSE_RULE="${FUSE_RULE:-1.0}"
HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-100}"
HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-300}"
HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-3000}"
HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-350}"
COLD_FRACTION="${COLD_FRACTION:--1}"
MAX_COLD_POOL="${MAX_COLD_POOL:-3000000}"

USE_LGBM="${USE_LGBM:-1}"
LGBM_MAX_ROWS="${LGBM_MAX_ROWS:-120000}"
LGBM_THREADS="${LGBM_THREADS:-0}"
REQUIRE_LGBM_BETTER="${REQUIRE_LGBM_BETTER:-1}"
MIN_TOP1_DIFF="${MIN_TOP1_DIFF:-0.15}"

RUN_TGNN="${RUN_TGNN:-1}"
TGNN_EPOCHS="${TGNN_EPOCHS:-8}"
TGNN_BATCH_SIZE="${TGNN_BATCH_SIZE:-128}"
TGNN_NODE_EMB_DIM="${TGNN_NODE_EMB_DIM:-128}"
TGNN_TIME_EMB_DIM="${TGNN_TIME_EMB_DIM:-32}"
TGNN_HIDDEN_DIM="${TGNN_HIDDEN_DIM:-192}"
TGNN_SRC_NEIGHBORS="${TGNN_SRC_NEIGHBORS:-50}"
TGNN_CAND_NEIGHBORS="${TGNN_CAND_NEIGHBORS:-30}"
TGNN_SECOND_HOP="${TGNN_SECOND_HOP:-20}"
TGNN_HARD_NEGATIVES="${TGNN_HARD_NEGATIVES:-30}"
TGNN_BPR_WEIGHT="${TGNN_BPR_WEIGHT:-0.1}"
TGNN_DROPOUT="${TGNN_DROPOUT:-0.1}"
TGNN_MAX_ROWS="${TGNN_MAX_ROWS:-$LGBM_MAX_ROWS}"

RUN_MLP="${RUN_MLP:-1}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-256}"
MLP_WEIGHT="${MLP_WEIGHT:-0.2}"
MLP_EPOCHS="${MLP_EPOCHS:-10}"
HARD_NEGATIVES="${HARD_NEGATIVES:-30}"
BPR_WEIGHT="${BPR_WEIGHT:-0.1}"

RUN_SEQ="${RUN_SEQ:-1}"
SEQ_LEN="${SEQ_LEN:-200}"
SEQ_GAMMA="${SEQ_GAMMA:-0.25}"
SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-256}"
SEQ_EPOCHS="${SEQ_EPOCHS:-12}"

BATCH_SIZE="${BATCH_SIZE:-512}"
SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-256}"
FEATURE_WORKERS="${FEATURE_WORKERS:-48}"
FEATURE_CHUNK_SIZE="${FEATURE_CHUNK_SIZE:-512}"
GPU_COUNT="${GPU_COUNT:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
USE_CUDA="${USE_CUDA:-1}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv_jittor}"

cd "$(dirname "$0")"

has_data() {
  if [ ! -d "$DATA_DIR" ]; then
    return 1
  fi
  for dataset_path in "$DATA_DIR"/*; do
    if [ -d "$dataset_path" ] && [ -f "$dataset_path/train.csv" ] && [ -f "$dataset_path/test.csv" ]; then
      return 0
    fi
  done
  return 1
}

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

if ! has_data; then
  mkdir -p "$DOWNLOAD_DIR"
  DATA_ARCHIVE="$DOWNLOAD_DIR/data_A_download"
  if [ ! -f "$DATA_ARCHIVE" ]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L "$DATA_URL" -o "$DATA_ARCHIVE"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$DATA_ARCHIVE" "$DATA_URL"
    else
      echo "curl or wget is required to download data"
      exit 1
    fi
  fi
  rm -rf "$DOWNLOAD_DIR/extracted"
  mkdir -p "$DOWNLOAD_DIR/extracted"
  if "$PYTHON_BIN" - <<PY
import zipfile
raise SystemExit(0 if zipfile.is_zipfile("$DATA_ARCHIVE") else 1)
PY
  then
    "$PYTHON_BIN" - <<PY
import zipfile
zipfile.ZipFile("$DATA_ARCHIVE").extractall("$DOWNLOAD_DIR/extracted")
PY
  else
    tar -xf "$DATA_ARCHIVE" -C "$DOWNLOAD_DIR/extracted"
  fi
  FOUND_DATA="$($PYTHON_BIN - <<PY
from pathlib import Path
root = Path("$DOWNLOAD_DIR/extracted")
def has_dataset_children(path):
    return any(
        p.is_dir() and (p / "train.csv").exists() and (p / "test.csv").exists()
        for p in path.iterdir()
    )
for p in [root] + list(root.rglob("*")):
    if p.is_dir() and has_dataset_children(p):
        print(p)
        break
PY
)"
  if [ -z "$FOUND_DATA" ]; then
    echo "downloaded archive does not contain dataset directories with train.csv and test.csv"
    exit 1
  fi
  rm -rf "$DATA_DIR"
  mv "$FOUND_DATA" "$DATA_DIR"
fi

if ! $PYTHON_BIN - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("jittor") else 1)
PY
then
  "$PYTHON_BIN" -m pip install -U pip
  "$PYTHON_BIN" -m pip install -U jittor
fi

if [ "$USE_LGBM" = "1" ]; then
  if ! $PYTHON_BIN - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("lightgbm") else 1)
PY
  then
    "$PYTHON_BIN" -m pip install -U lightgbm
  fi
fi

CUDA_ARG=""
if [ "$USE_CUDA" = "1" ]; then
  CUDA_ARG="--cuda"
fi

mkdir -p "$MODEL_ROOT" "$SCORE_ROOT"

echo "building rank folds: $FOLD_ROOT"
"$PYTHON_BIN" scripts/build_rank_folds.py \
  --data-dir "$DATA_DIR" \
  --fold-root "$FOLD_ROOT" \
  --max-valid "$MAX_VALID" \
  --hard-recent-limit "$HARD_RECENT_LIMIT" \
  --hard-transition-limit "$HARD_TRANSITION_LIMIT" \
  --hard-popular-limit "$HARD_POPULAR_LIMIT" \
  --hard-popular-sample "$HARD_POPULAR_SAMPLE" \
  --valid-mode test-prior \
  --cold-fraction "$COLD_FRACTION" \
  --max-cold-pool "$MAX_COLD_POOL" \
  --seed "$SEED"

echo "building fold feature caches under $CACHE_DIR"
"$PYTHON_BIN" scripts/build_feature_cache.py \
  --data-dir "$DATA_DIR" \
  --fold-root "$FOLD_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --seq-lens "$SEQ_LEN" \
  --workers "$FEATURE_WORKERS" \
  --chunk-size "$FEATURE_CHUNK_SIZE"

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

for fold_dir in "$FOLD_ROOT"/fold*; do
  [ -d "$fold_dir" ] || continue
  fold="$(basename "$fold_dir")"
  fold_cache="$CACHE_DIR/$fold"
  fold_model="$MODEL_ROOT/$fold"
  fold_scores="$SCORE_ROOT/$fold"
  mkdir -p "$fold_model" "$fold_scores"

  if [ "$USE_LGBM" = "1" ]; then
    echo "training LightGBM ranker for $fold"
    "$PYTHON_BIN" scripts/train_lgbm_ranker.py \
      --cache-dir "$fold_cache" \
      --model-dir "$fold_model/lgbm" \
      --score-dir "$fold_scores" \
      --max-rows "$LGBM_MAX_ROWS" \
      --eval-ratio "$EVAL_RATIO" \
      --seed "$SEED" \
      --num-threads "$LGBM_THREADS"
  fi

  DATASETS="$($PYTHON_BIN - <<PY
from pathlib import Path
root = Path("$fold_dir")
print(" ".join(sorted(p.name for p in root.iterdir() if p.is_dir() and (p/"valid.csv").exists())))
PY
)"

  if [ "$RUN_TGNN" = "1" ]; then
    echo "training temporal GNN rankers for $fold"
    for dataset in $DATASETS; do
      run_gpu_job "$PYTHON_BIN" scripts/train_tgnn_ranker.py \
        --data-dir "$DATA_DIR" \
        --valid-dir "$fold_dir" \
        --cache-dir "$fold_cache" \
        --dataset "$dataset" \
        --model-dir "$fold_model/tgnn" \
        --score-dir "$fold_scores" \
        --epochs "$TGNN_EPOCHS" \
        --batch-size "$TGNN_BATCH_SIZE" \
        --node-emb-dim "$TGNN_NODE_EMB_DIM" \
        --time-emb-dim "$TGNN_TIME_EMB_DIM" \
        --hidden-dim "$TGNN_HIDDEN_DIM" \
        --dropout "$TGNN_DROPOUT" \
        --src-neighbors "$TGNN_SRC_NEIGHBORS" \
        --cand-neighbors "$TGNN_CAND_NEIGHBORS" \
        --second-hop "$TGNN_SECOND_HOP" \
        --hard-negatives "$TGNN_HARD_NEGATIVES" \
        --bpr-weight "$TGNN_BPR_WEIGHT" \
        --eval-ratio "$EVAL_RATIO" \
        --max-rows "$TGNN_MAX_ROWS" \
        --seed "$SEED" \
        $CUDA_ARG
    done
    wait
  fi

  if [ "$RUN_MLP" = "1" ]; then
    echo "training MLP residual models for $fold"
    for dataset in $DATASETS; do
      run_gpu_job "$PYTHON_BIN" scripts/train_ranker.py \
        --valid-dir "$fold_dir" \
        --cache-dir "$fold_cache" \
        --dataset "$dataset" \
        --model-dir "$fold_model/mlp" \
        --epochs "$MLP_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --hidden-dim "$MLP_HIDDEN_DIM" \
        --fuse-rule "$FUSE_RULE" \
        --mlp-weight "$MLP_WEIGHT" \
        --eval-ratio "$EVAL_RATIO" \
        --hard-negatives "$HARD_NEGATIVES" \
        --bpr-weight "$BPR_WEIGHT" \
        --max-rows "$LGBM_MAX_ROWS" \
        --seed "$SEED" \
        $CUDA_ARG
    done
    wait
  fi

  if [ "$RUN_SEQ" = "1" ]; then
    echo "training sequence residual models for $fold"
    for dataset in $DATASETS; do
      run_gpu_job "$PYTHON_BIN" scripts/train_seq_ranker.py \
        --valid-dir "$fold_dir" \
        --cache-dir "$fold_cache" \
        --dataset "$dataset" \
        --model-dir "$fold_model/seq" \
        --epochs "$SEQ_EPOCHS" \
        --batch-size "$SEQ_BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        --hidden-dim "$SEQ_HIDDEN_DIM" \
        --fuse-rule "$FUSE_RULE" \
        --gamma "$SEQ_GAMMA" \
        --eval-ratio "$EVAL_RATIO" \
        --hard-negatives "$HARD_NEGATIVES" \
        --bpr-weight "$BPR_WEIGHT" \
        --max-rows "$LGBM_MAX_ROWS" \
        --seed "$SEED" \
        $CUDA_ARG
    done
    wait
  fi

  echo "searching ensemble weights for $fold"
  "$PYTHON_BIN" scripts/search_ensemble.py \
    --valid-dir "$fold_dir" \
    --cache-dir "$fold_cache" \
    --model-root "$fold_model" \
    --score-dir "$fold_scores" \
    --out "$fold_model/ensemble_weights.json" \
    --max-rows "$LGBM_MAX_ROWS"
done

echo "writing ablation summary"
REQUIRE_ARG=""
if [ "$USE_LGBM" = "1" ] && [ "$REQUIRE_LGBM_BETTER" = "1" ]; then
  REQUIRE_ARG="--require-lgbm-better"
fi
"$PYTHON_BIN" scripts/run_ablation.py \
  --cache-root "$CACHE_DIR" \
  --score-root "$SCORE_ROOT" \
  --weights-root "$MODEL_ROOT" \
  --out "$MODEL_ROOT/ablation_summary.csv" \
  --max-rows "$LGBM_MAX_ROWS" \
  $REQUIRE_ARG

echo "predicting fold-averaged ensemble"
PREDICT_MIN_TOP1_DIFF="$MIN_TOP1_DIFF"
if [ "$USE_LGBM" != "1" ]; then
  PREDICT_MIN_TOP1_DIFF="0"
fi
"$PYTHON_BIN" scripts/predict_fold_ensemble.py \
  --data-dir "$DATA_DIR" \
  --cache-root "$CACHE_DIR" \
  --weights-root "$MODEL_ROOT" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH" \
  --min-top1-diff "$PREDICT_MIN_TOP1_DIFF"

echo "done: $ZIP_PATH"
