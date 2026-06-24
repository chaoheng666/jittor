#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data_A}"
DATA_URL="${DATA_URL:-https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads}"
VALID_DIR="${VALID_DIR:-validation}"
MODEL_DIR="${MODEL_DIR:-models}"
OUT_DIR="${OUT_DIR:-submission_mlp}"
ZIP_PATH="${ZIP_PATH:-result_mlp.zip}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
MAX_VALID="${MAX_VALID:-0}"
FUSE_RULE="${FUSE_RULE:-1.0}"
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

PYTHON_BIN="python3"
if [ "$USE_VENV" = "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "creating virtual env: $VENV_DIR"
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

  echo "extracting data"
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
for p in [root] + list(root.rglob("*")):
    if p.is_dir() and (p / "dataset1/train.csv").exists() and (p / "dataset2/train.csv").exists():
        print(p)
        break
PY
)"
  if [ -z "$FOUND_DATA" ]; then
    echo "downloaded archive does not contain dataset1/dataset2 train files"
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
  echo "installing jittor"
  "$PYTHON_BIN" -m pip install -U pip
  "$PYTHON_BIN" -m pip install -U jittor
fi

echo "checking jittor"
"$PYTHON_BIN" - <<'PY'
import jittor as jt
print("jittor", jt.__version__)
PY

echo "building validation samples"
"$PYTHON_BIN" scripts/valid_builder.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$VALID_DIR" \
  --max-valid "$MAX_VALID"

CUDA_ARG=""
if [ "$USE_CUDA" = "1" ]; then
  CUDA_ARG="--cuda"
fi

echo "training jittor mlp ranker"
"$PYTHON_BIN" scripts/train_ranker.py \
  --valid-dir "$VALID_DIR" \
  --model-dir "$MODEL_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --fuse-rule "$FUSE_RULE" \
  $CUDA_ARG

echo "predicting final submission"
"$PYTHON_BIN" scripts/predict_ranker.py \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH" \
  --mode fuse \
  $CUDA_ARG

echo "done: $ZIP_PATH"
