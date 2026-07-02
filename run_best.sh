#!/usr/bin/env bash
set -euo pipefail

# Single shell entrypoint.
# Trains CRAFT-style temporal candidate rerankers, selects by validation MRR,
# and scores only the official 100 candidates.

DATA_DIR="${DATA_DIR:-data_A}"
DATA_URL="${DATA_URL:-https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads}"
MODEL_ROOT="${MODEL_ROOT:-competition_models_best}"
OUT_DIR="${OUT_DIR:-submission_best}"
ZIP_PATH="${ZIP_PATH:-result_best.zip}"
CONFIG_PATH="${CONFIG_PATH:-$MODEL_ROOT/craft_rerank_config.json}"

PRESET="${PRESET:-final}"
DATASET="${DATASET:-all}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
HISTORY_RATIO="${HISTORY_RATIO:-0.8}"
MIN_VALIDATION_GAIN="${MIN_VALIDATION_GAIN:-0.0005}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.15}"

CRAFT_SEEDS="${CRAFT_SEEDS:-2026}"
CRAFT_EPOCHS="${CRAFT_EPOCHS:-8}"
CRAFT_SAMPLE_EDGES="${CRAFT_SAMPLE_EDGES:-300000}"
CRAFT_CONFIGS="${CRAFT_CONFIGS:-128:64:60:0.0005:32,128:64:90:0.0005:32,160:64:60:0.0004:48,160:80:90:0.0004:48,192:80:120:0.0003:48,96:48:60:0.0007:24,128:80:120:0.0004:32,192:64:90:0.0003:64}"

BATCH_SIZE="${BATCH_SIZE:-256}"
GPU_COUNT="${GPU_COUNT:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
USE_CUDA="${USE_CUDA:-1}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv_jittor}"

if [ "$PRESET" = "quick" ]; then
  CRAFT_EPOCHS="${QUICK_EPOCHS:-1}"
  CRAFT_SAMPLE_EDGES="${QUICK_SAMPLE_EDGES:-2000}"
  CRAFT_CONFIGS="${QUICK_CONFIGS:-32:16:30:0.001:4}"
  BATCH_SIZE="${QUICK_BATCH_SIZE:-128}"
  MAX_PARALLEL="${QUICK_MAX_PARALLEL:-1}"
  GPU_COUNT="${QUICK_GPU_COUNT:-1}"
fi

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

echo "python: $($PYTHON_BIN --version)"

if ! has_data; then
  mkdir -p "$DOWNLOAD_DIR"
  DATA_ARCHIVE="$DOWNLOAD_DIR/data_A_download"
  if [ ! -f "$DATA_ARCHIVE" ]; then
    echo "data not found, downloading from $DATA_URL"
    if command -v curl >/dev/null 2>&1; then
      curl -L "$DATA_URL" -o "$DATA_ARCHIVE"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$DATA_ARCHIVE" "$DATA_URL"
    else
      echo "curl or wget is required to download data"
      exit 1
    fi
  else
    echo "using cached data archive: $DATA_ARCHIVE"
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

if ! has_data; then
  echo "data files are still missing under $DATA_DIR"
  exit 1
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

echo "training CRAFT rerankers preset=$PRESET dataset=$DATASET"
IFS=',' read -r -a configs <<< "$CRAFT_CONFIGS"
for config in "${configs[@]}"; do
  IFS=':' read -r hidden embed history lr negatives <<< "$config"
  tag="craft_h${hidden}_e${embed}_l${history}_n${negatives}_lr${lr//./p}"
  run_gpu_job "$PYTHON_BIN" scripts/train_edge_ranker.py \
    --data-dir "$DATA_DIR" \
    --model-dir "$MODEL_ROOT/$tag" \
    --dataset "$DATASET" \
    --epochs "$CRAFT_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$hidden" \
    --embed-dim "$embed" \
    --history-len "$history" \
    --lr "$lr" \
    --pairwise-weight "$PAIRWISE_WEIGHT" \
    --negatives "$negatives" \
    --sample-edges "$CRAFT_SAMPLE_EDGES" \
    --history-ratio "$HISTORY_RATIO" \
    --eval-ratio "$EVAL_RATIO" \
    --seed-list "$CRAFT_SEEDS" \
    $CUDA_ARG
done
wait

echo "selecting best validation-MRR model"
"$PYTHON_BIN" scripts/select_edge_model.py \
  --data-dir "$DATA_DIR" \
  --model-root "$MODEL_ROOT" \
  --out "$CONFIG_PATH" \
  --dataset "$DATASET" \
  --min-validation-gain "$MIN_VALIDATION_GAIN"

echo "predicting official candidates"
"$PYTHON_BIN" scripts/predict_edge_intensity.py \
  --data-dir "$DATA_DIR" \
  --config "$CONFIG_PATH" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH"

echo "checking zip"
"$PYTHON_BIN" - <<PY
import csv, zipfile
path = "$ZIP_PATH"
with zipfile.ZipFile(path) as zf:
    for name in sorted(zf.namelist()):
        rows = 0
        with zf.open(name) as f:
            for line in f:
                row = next(csv.reader([line.decode("utf-8")]))
                vals = [float(x) for x in row]
                if len(vals) != 100:
                    raise SystemExit(f"{name}: expected 100 columns, got {len(vals)}")
                if rows == 0:
                    print(name, len(vals), min(vals), max(vals), sum(vals))
                rows += 1
        print(name, "rows", rows)
PY

echo "done: $ZIP_PATH"
