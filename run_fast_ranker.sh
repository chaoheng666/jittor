#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data_A}"
DATA_URL="${DATA_URL:-https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads}"
VALID_ROOT="${VALID_ROOT:-validation_fast}"
VALID_MODE="${VALID_MODE:-test-prior}"
VALID_DIR="${VALID_DIR:-${VALID_ROOT}_${VALID_MODE}}"
CACHE_DIR="${CACHE_DIR:-feature_cache_fast}"
MODEL_ROOT="${MODEL_ROOT:-fast_models}"
OUT_DIR="${OUT_DIR:-submission_fast}"
ZIP_PATH="${ZIP_PATH:-result_fast.zip}"

SEED="${SEED:-2026}"
MAX_VALID="${MAX_VALID:-0}"
VALID_RATIO="${VALID_RATIO:-0.2}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
FUSE_RULE="${FUSE_RULE:-1.0}"
HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-100}"
HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-300}"
HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-3000}"
HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-350}"
COLD_FRACTION="${COLD_FRACTION:--1}"
MAX_COLD_POOL="${MAX_COLD_POOL:-3000000}"

MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-128}"
MLP_WEIGHT="${MLP_WEIGHT:-0.2}"
MLP_EPOCHS="${MLP_EPOCHS:-10}"
HARD_NEGATIVES="${HARD_NEGATIVES:-30}"

SEQ_LEN="${SEQ_LEN:-100}"
SEQ_GAMMA="${SEQ_GAMMA:-0.25}"
SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-192}"
SEQ_EPOCHS="${SEQ_EPOCHS:-10}"

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

CUDA_ARG=""
if [ "$USE_CUDA" = "1" ]; then
  CUDA_ARG="--cuda"
fi

mkdir -p "$MODEL_ROOT"

echo "building validation: $VALID_DIR"
"$PYTHON_BIN" scripts/valid_builder.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$VALID_DIR" \
  --valid-ratio "$VALID_RATIO" \
  --max-valid "$MAX_VALID" \
  --hard-recent-limit "$HARD_RECENT_LIMIT" \
  --hard-transition-limit "$HARD_TRANSITION_LIMIT" \
  --hard-popular-limit "$HARD_POPULAR_LIMIT" \
  --hard-popular-sample "$HARD_POPULAR_SAMPLE" \
  --valid-mode "$VALID_MODE" \
  --cold-fraction "$COLD_FRACTION" \
  --max-cold-pool "$MAX_COLD_POOL" \
  --seed "$SEED"

echo "building feature cache with $FEATURE_WORKERS workers: $CACHE_DIR"
"$PYTHON_BIN" scripts/build_feature_cache.py \
  --data-dir "$DATA_DIR" \
  --valid-dir "$VALID_DIR" \
  --cache-dir "$CACHE_DIR" \
  --seq-lens "$SEQ_LEN" \
  --workers "$FEATURE_WORKERS" \
  --chunk-size "$FEATURE_CHUNK_SIZE"

DATASETS="$($PYTHON_BIN - <<PY
from pathlib import Path
root = Path("$VALID_DIR")
print(" ".join(sorted(p.name for p in root.iterdir() if p.is_dir() and (p/"valid.csv").exists())))
PY
)"

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

echo "training one-parameter MLP and Seq models in parallel"
for dataset in $DATASETS; do
  run_gpu_job "$PYTHON_BIN" scripts/train_ranker.py \
    --valid-dir "$VALID_DIR" \
    --cache-dir "$CACHE_DIR" \
    --dataset "$dataset" \
    --model-dir "$MODEL_ROOT/mlp" \
    --epochs "$MLP_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$MLP_HIDDEN_DIM" \
    --fuse-rule "$FUSE_RULE" \
    --mlp-weight "$MLP_WEIGHT" \
    --eval-ratio "$EVAL_RATIO" \
    --hard-negatives "$HARD_NEGATIVES" \
    --max-rows "$MAX_VALID" \
    --seed "$SEED" \
    $CUDA_ARG

  run_gpu_job "$PYTHON_BIN" scripts/train_seq_ranker.py \
    --valid-dir "$VALID_DIR" \
    --cache-dir "$CACHE_DIR" \
    --dataset "$dataset" \
    --model-dir "$MODEL_ROOT/seq" \
    --epochs "$SEQ_EPOCHS" \
    --batch-size "$SEQ_BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --hidden-dim "$SEQ_HIDDEN_DIM" \
    --fuse-rule "$FUSE_RULE" \
    --gamma "$SEQ_GAMMA" \
    --eval-ratio "$EVAL_RATIO" \
    --hard-negatives "$HARD_NEGATIVES" \
    --max-rows "$MAX_VALID" \
    --seed "$SEED" \
    $CUDA_ARG
done
wait

echo "searching cached ensemble weights"
"$PYTHON_BIN" scripts/search_ensemble.py \
  --valid-dir "$VALID_DIR" \
  --cache-dir "$CACHE_DIR" \
  --model-root "$MODEL_ROOT" \
  --out "$MODEL_ROOT/ensemble_weights.json" \
  --max-rows "$MAX_VALID"

echo "predicting from cached test features"
"$PYTHON_BIN" scripts/predict_luxury_ensemble.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --weights "$MODEL_ROOT/ensemble_weights.json" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH"

echo "done: $ZIP_PATH"
