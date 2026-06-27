#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data_A}"
DATA_URL="${DATA_URL:-https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads}"
VALID_ROOT="${VALID_ROOT:-validation_competition}"
VALID_MODE="${VALID_MODE:-test-prior}"
MODEL_ROOT="${MODEL_ROOT:-competition_models}"
SCORE_DIR="${SCORE_DIR:-competition_scores}"
OUT_DIR="${OUT_DIR:-submission_competition}"
ZIP_PATH="${ZIP_PATH:-result.zip}"

MAX_VALID="${MAX_VALID:-0}"
VALID_RATIO="${VALID_RATIO:-0.2}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
FUSE_RULE="${FUSE_RULE:-1.0}"
HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-80}"
HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-200}"
HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-2000}"
HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-300}"
COLD_FRACTION="${COLD_FRACTION:--1}"
MAX_COLD_POOL="${MAX_COLD_POOL:-2000000}"

MLP_SEEDS="${MLP_SEEDS:-2026,2027,2028}"
MLP_HIDDEN_DIMS="${MLP_HIDDEN_DIMS:-64,128,256}"
MLP_WEIGHTS="${MLP_WEIGHTS:-0.1,0.2,0.35}"
MLP_EPOCHS="${MLP_EPOCHS:-12}"

SEQ_SEEDS="${SEQ_SEEDS:-2026,2027}"
SEQ_LENS="${SEQ_LENS:-30,50,100}"
SEQ_GAMMAS="${SEQ_GAMMAS:-0.1,0.2,0.35}"
SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-192}"
SEQ_EPOCHS="${SEQ_EPOCHS:-12}"

RUN_CRAFT="${RUN_CRAFT:-1}"
CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-30,50}"
CRAFT_HIDDEN_SIZES="${CRAFT_HIDDEN_SIZES:-64,128}"
CRAFT_EPOCHS="${CRAFT_EPOCHS:-8}"

BATCH_SIZE="${BATCH_SIZE:-256}"
SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-192}"
CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-200}"
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

mkdir -p "$MODEL_ROOT" "$SCORE_DIR"

MAIN_VALID_DIR="${VALID_ROOT}_${VALID_MODE}"
echo "building validation: $MAIN_VALID_DIR"
"$PYTHON_BIN" scripts/valid_builder.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$MAIN_VALID_DIR" \
  --valid-ratio "$VALID_RATIO" \
  --max-valid "$MAX_VALID" \
  --hard-recent-limit "$HARD_RECENT_LIMIT" \
  --hard-transition-limit "$HARD_TRANSITION_LIMIT" \
  --hard-popular-limit "$HARD_POPULAR_LIMIT" \
  --hard-popular-sample "$HARD_POPULAR_SAMPLE" \
  --valid-mode "$VALID_MODE" \
  --cold-fraction "$COLD_FRACTION" \
  --max-cold-pool "$MAX_COLD_POOL"

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
      --fuse-rule "$FUSE_RULE" \
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
      --batch-size "$SEQ_BATCH_SIZE" \
      --seq-len "$seq_len" \
      --hidden-dim "$SEQ_HIDDEN_DIM" \
      --fuse-rule "$FUSE_RULE" \
      --gamma "$gamma" \
      --eval-ratio "$EVAL_RATIO" \
      --max-rows "$MAX_VALID" \
      --seed-list "$SEQ_SEEDS" \
      $CUDA_ARG
  done
done
wait

CRAFT_AVAILABLE="$($PYTHON_BIN - <<'PY'
try:
    from jittor_geometric.data import TemporalData
    from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
    from jittor_geometric.nn.models.craft import CRAFT
except Exception:
    print("0")
else:
    print("1")
PY
)"
if [ "$RUN_CRAFT" = "1" ] && [ "$CRAFT_AVAILABLE" = "1" ]; then
  echo "training CRAFT temporal graph models"
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
elif [ "$RUN_CRAFT" = "1" ]; then
  echo "skip CRAFT: jittor_geometric is not installed in this environment"
else
  echo "skip CRAFT because RUN_CRAFT=$RUN_CRAFT"
fi

echo "searching ensemble weights"
"$PYTHON_BIN" scripts/search_ensemble.py \
  --valid-dir "$MAIN_VALID_DIR" \
  --model-root "$MODEL_ROOT" \
  --score-dir "$SCORE_DIR" \
  --out "$MODEL_ROOT/ensemble_weights.json" \
  --max-rows "$MAX_VALID"

echo "predicting final ensemble"
"$PYTHON_BIN" scripts/predict_luxury_ensemble.py \
  --data-dir "$DATA_DIR" \
  --weights "$MODEL_ROOT/ensemble_weights.json" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH"

echo "checking zip"
"$PYTHON_BIN" - <<PY
import csv, zipfile
path = "$ZIP_PATH"
with zipfile.ZipFile(path) as zf:
    for name in sorted(zf.namelist()):
        with zf.open(name) as f:
            row = next(csv.reader(line.decode("utf-8") for line in f))
        vals = [float(x) for x in row]
        print(name, len(vals), min(vals), max(vals), sum(vals))
PY

echo "done: $ZIP_PATH"
