#!/usr/bin/env bash
set -euo pipefail

# Single best-effect entrypoint.
# Runs the full submission pipeline with the broadest current search:
# rule ranking, edge residual MLP sweeps, optional CRAFT, ensemble search,
# prediction, and result zip packing.

DATA_DIR="${DATA_DIR:-data_A}"
DATA_URL="${DATA_URL:-https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads}"
VALID_ROOT="${VALID_ROOT:-validation_best}"
VALID_MODE="${VALID_MODE:-test-row}"
MODEL_ROOT="${MODEL_ROOT:-competition_models_best}"
SCORE_DIR="${SCORE_DIR:-competition_scores_best}"
OUT_DIR="${OUT_DIR:-submission_best}"
ZIP_PATH="${ZIP_PATH:-result_best.zip}"

MAX_VALID="${MAX_VALID:-0}"
VALID_RATIO="${VALID_RATIO:-0.2}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
FUSE_RULE="${FUSE_RULE:-0.95}"
HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-80}"
HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-300}"
HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-3000}"
HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-350}"
COLD_FRACTION="${COLD_FRACTION:--1}"
MAX_COLD_POOL="${MAX_COLD_POOL:-3000000}"
WEIGHT_GRID="${WEIGHT_GRID:-0.01,0.02,0.05,0.08,0.1,0.15,0.2}"
SEARCH_START="${SEARCH_START:-rule}"
MIN_ADD_GAIN="${MIN_ADD_GAIN:-0.0005}"
MAX_TOP1_DIFF="${MAX_TOP1_DIFF:-0.35}"
RESIDUALIZE_AGAINST_RULE="${RESIDUALIZE_AGAINST_RULE:-1}"

EDGE_SEEDS="${EDGE_SEEDS:-2026,2027}"
EDGE_HIDDEN_DIMS="${EDGE_HIDDEN_DIMS:-64,128,256}"
EDGE_GAMMAS="${EDGE_GAMMAS:-0.05,0.08,0.15,0.25,0.35}"
EDGE_EPOCHS="${EDGE_EPOCHS:-10}"
EDGE_NEGATIVES="${EDGE_NEGATIVES:-10}"
EDGE_SAMPLE_EDGES="${EDGE_SAMPLE_EDGES:-250000}"

RUN_CRAFT="${RUN_CRAFT:-1}"
CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-20,30,50,80}"
CRAFT_HIDDEN_SIZES="${CRAFT_HIDDEN_SIZES:-64,128}"
CRAFT_EPOCHS="${CRAFT_EPOCHS:-10}"

BATCH_SIZE="${BATCH_SIZE:-256}"
CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-140}"
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

echo "building evaluation validation: $MAIN_VALID_DIR"
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

echo "training edge residual models"
IFS=',' read -r -a edge_hidden_dims <<< "$EDGE_HIDDEN_DIMS"
IFS=',' read -r -a edge_gammas <<< "$EDGE_GAMMAS"
for hidden in "${edge_hidden_dims[@]}"; do
  for gamma in "${edge_gammas[@]}"; do
    tag="edge_h${hidden}_g${gamma//./p}"
    run_gpu_job "$PYTHON_BIN" scripts/train_edge_ranker.py \
      --data-dir "$DATA_DIR" \
      --model-dir "$MODEL_ROOT/$tag" \
      --epochs "$EDGE_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --hidden-dim "$hidden" \
      --fuse-rule "$FUSE_RULE" \
      --gamma "$gamma" \
      --negatives "$EDGE_NEGATIVES" \
      --sample-edges "$EDGE_SAMPLE_EDGES" \
      --eval-ratio "$EVAL_RATIO" \
      --seed-list "$EDGE_SEEDS" \
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
RESIDUAL_ARG=""
if [ "$RESIDUALIZE_AGAINST_RULE" != "1" ]; then
  RESIDUAL_ARG="--no-residualize-against-rule"
fi
"$PYTHON_BIN" scripts/search_ensemble.py \
  --valid-dir "$MAIN_VALID_DIR" \
  --model-root "$MODEL_ROOT" \
  --score-dir "$SCORE_DIR" \
  --out "$MODEL_ROOT/ensemble_weights.json" \
  --max-rows "$MAX_VALID" \
  --weight-grid "$WEIGHT_GRID" \
  --start "$SEARCH_START" \
  --min-add-gain "$MIN_ADD_GAIN" \
  --max-top1-diff "$MAX_TOP1_DIFF" \
  $RESIDUAL_ARG

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
