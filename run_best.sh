#!/usr/bin/env bash
set -euo pipefail

# Full fusion entrypoint.
# Builds data diagnostics, statistical base intensity, optional Jittor deep
# components, large-pool/time-replay validation, sanity-gated fusion, and the
# final official-candidate submission zip.

DATA_DIR="${DATA_DIR:-data_A}"
DATA_URL="${DATA_URL:-https://cloud.tsinghua.edu.cn/f/6a9569def9044d49bb96/?dl=1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-downloads}"
MODEL_ROOT="${MODEL_ROOT:-models_v2}"
REPORT_DIR="${REPORT_DIR:-reports}"
OUT_DIR="${OUT_DIR:-submission_best}"
ZIP_PATH="${ZIP_PATH:-result_best.zip}"

HISTORY_RATIO="${HISTORY_RATIO:-0.8}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
FUSE_RULE="${FUSE_RULE:-0.95}"
REQUIRE_LEARNED="${REQUIRE_LEARNED:-1}"

RUN_LEGACY="${RUN_LEGACY:-1}"
RUN_SEQ="${RUN_SEQ:-1}"
RUN_CRAFT="${RUN_CRAFT:-1}"
INSTALL_JITTOR="${INSTALL_JITTOR:-1}"

EDGE_SEEDS="${EDGE_SEEDS:-2026,2027}"
EDGE_HIDDEN_DIMS="${EDGE_HIDDEN_DIMS:-64,128,256}"
EDGE_GAMMAS="${EDGE_GAMMAS:-0.05,0.08,0.15,0.25,0.35}"
EDGE_EPOCHS="${EDGE_EPOCHS:-10}"
EDGE_NEGATIVES="${EDGE_NEGATIVES:-10}"
EDGE_NEGATIVE_MODE="${EDGE_NEGATIVE_MODE:-mixed}"
EDGE_SAMPLE_EDGES="${EDGE_SAMPLE_EDGES:-250000}"
LEGACY_VALIDATE_TOP_K="${LEGACY_VALIDATE_TOP_K:-3}"

SEQ_EPOCHS="${SEQ_EPOCHS:-3}"
SEQ_NEG_PER_POS="${SEQ_NEG_PER_POS:-256}"
SEQ_SAMPLE_EDGES="${SEQ_SAMPLE_EDGES:-100000}"
CRAFT_EPOCHS="${CRAFT_EPOCHS:-2}"
CRAFT_NEG_PER_POS="${CRAFT_NEG_PER_POS:-64}"
CRAFT_SAMPLE_EDGES="${CRAFT_SAMPLE_EDGES:-30000}"

VAL_MAX_EDGES="${VAL_MAX_EDGES:-2000}"
VAL_POOL_SIZE="${VAL_POOL_SIZE:-500}"
VAL_CANDIDATE_MODE="${VAL_CANDIDATE_MODE:-mixed}"
REPLAY_BLOCKS="${REPLAY_BLOCKS:-3}"
REPLAY_MAX_EVENTS="${REPLAY_MAX_EVENTS:-500}"
REPLAY_POOL_SIZE="${REPLAY_POOL_SIZE:-500}"
REPLAY_CANDIDATE_MODE="${REPLAY_CANDIDATE_MODE:-mixed}"
SANITY_MAX_ROWS="${SANITY_MAX_ROWS:-5000}"
PREDICT_MAX_ROWS="${PREDICT_MAX_ROWS:-0}"

BATCH_SIZE="${BATCH_SIZE:-256}"
GPU_COUNT="${GPU_COUNT:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
TOTAL_CPU_THREADS="${TOTAL_CPU_THREADS:-48}"
CPU_THREADS_PER_JOB="${CPU_THREADS_PER_JOB:-}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-}"
CPU_WORKERS="${CPU_WORKERS:-2}"
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

has_jittor() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("jittor") else 1)
PY
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "python3 or python not found"
    exit 1
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "configured PYTHON_BIN not found: $PYTHON_BIN"
  exit 1
fi

if [ "$USE_VENV" = "1" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
fi

echo "python: $($PYTHON_BIN --version)"

if [ "$GPU_COUNT" -lt 1 ]; then
  GPU_COUNT=1
fi
if [ "$MAX_PARALLEL" -lt 1 ]; then
  MAX_PARALLEL=1
fi
if [ "$TOTAL_CPU_THREADS" -lt 1 ]; then
  TOTAL_CPU_THREADS=1
fi
if [ "$CPU_WORKERS" -lt 1 ]; then
  CPU_WORKERS=1
fi
if [ -z "$CPU_THREADS_PER_JOB" ]; then
  CPU_THREADS_PER_JOB=$(( (TOTAL_CPU_THREADS + MAX_PARALLEL - 1) / MAX_PARALLEL ))
fi
if [ "$CPU_THREADS_PER_JOB" -lt 1 ]; then
  CPU_THREADS_PER_JOB=1
fi
if [ -z "$CPU_THREADS_PER_WORKER" ]; then
  CPU_THREADS_PER_WORKER=$(( (TOTAL_CPU_THREADS + CPU_WORKERS - 1) / CPU_WORKERS ))
fi
if [ "$CPU_THREADS_PER_WORKER" -lt 1 ]; then
  CPU_THREADS_PER_WORKER=1
fi
echo "resource plan: gpu_count=$GPU_COUNT max_parallel=$MAX_PARALLEL cpu_threads_per_job=$CPU_THREADS_PER_JOB cpu_workers=$CPU_WORKERS cpu_threads_per_worker=$CPU_THREADS_PER_WORKER"

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

if ! has_jittor && [ "$INSTALL_JITTOR" = "1" ] && { [ "$RUN_LEGACY" = "1" ] || [ "$RUN_SEQ" = "1" ] || [ "$RUN_CRAFT" = "1" ]; }; then
  "$PYTHON_BIN" -m pip install -U pip
  "$PYTHON_BIN" -m pip install -U jittor
fi

HAS_JITTOR=0
if has_jittor; then
  HAS_JITTOR=1
fi

CUDA_ARG=""
if [ "$USE_CUDA" = "1" ]; then
  CUDA_ARG="--cuda"
fi

mkdir -p "$MODEL_ROOT" "$REPORT_DIR"
GPU_LOCK_DIR="$MODEL_ROOT/.gpu_locks_$$"
rm -rf "$GPU_LOCK_DIR"
mkdir -p "$GPU_LOCK_DIR"
trap 'rm -rf "$GPU_LOCK_DIR"' EXIT

echo "analyzing data distribution"
"$PYTHON_BIN" scripts/analyze_data_distribution.py \
  --data-dir "$DATA_DIR" \
  --history-ratio "$HISTORY_RATIO" \
  --out-dir "$REPORT_DIR/data_stats"

echo "building base intensity artifacts"
"$PYTHON_BIN" scripts/train_base_intensity.py \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_ROOT/base_intensity"

DATASETS="$($PYTHON_BIN - <<PY
from pathlib import Path
from src.data_loader import find_dataset_dirs
print(" ".join(p.name for p in find_dataset_dirs("$DATA_DIR")))
PY
)"

if { [ "$RUN_LEGACY" = "1" ] || [ "$RUN_SEQ" = "1" ] || [ "$RUN_CRAFT" = "1" ]; } && [ -z "$DATASETS" ]; then
  echo "no datasets found under $DATA_DIR"
  exit 1
fi

JOB_PIDS=()

wait_for_jobs() {
  local status=0
  local pid
  for pid in "${JOB_PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  JOB_PIDS=()
  return "$status"
}

run_gpu_job() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 5
  done

  if [ "$USE_CUDA" != "1" ]; then
    OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
    "$@" &
    JOB_PIDS+=("$!")
    return
  fi

  while true; do
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
      local lock_dir="$GPU_LOCK_DIR/gpu_${gpu}.lock"
      if mkdir "$lock_dir" 2>/dev/null; then
        (
          trap 'rm -rf "$lock_dir"' EXIT
          CUDA_VISIBLE_DEVICES="$gpu" \
          OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" \
          OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" \
          MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" \
          NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB" \
          "$@"
        ) &
        JOB_PIDS+=("$!")
        return
      fi
    done
    sleep 5
  done
}

if [ "$RUN_LEGACY" = "1" ] && [ "$HAS_JITTOR" = "1" ]; then
  echo "training legacy edge MLP models"
  IFS=',' read -r -a edge_hidden_dims <<< "$EDGE_HIDDEN_DIMS"
  IFS=',' read -r -a edge_gammas <<< "$EDGE_GAMMAS"
  for hidden in "${edge_hidden_dims[@]}"; do
    for gamma in "${edge_gammas[@]}"; do
      tag="edge_h${hidden}_g${gamma//./p}"
      for dataset in $DATASETS; do
        run_gpu_job "$PYTHON_BIN" scripts/train_edge_ranker.py \
          --data-dir "$DATA_DIR" \
          --model-dir "$MODEL_ROOT/legacy/$tag" \
          --dataset "$dataset" \
          --epochs "$EDGE_EPOCHS" \
          --batch-size "$BATCH_SIZE" \
          --hidden-dim "$hidden" \
          --fuse-rule "$FUSE_RULE" \
          --gamma "$gamma" \
          --negatives "$EDGE_NEGATIVES" \
          --negative-mode "$EDGE_NEGATIVE_MODE" \
          --sample-edges "$EDGE_SAMPLE_EDGES" \
          --history-ratio "$HISTORY_RATIO" \
          --eval-ratio "$EVAL_RATIO" \
          --seed-list "$EDGE_SEEDS" \
          $CUDA_ARG
      done
    done
  done
  wait_for_jobs
elif [ "$RUN_LEGACY" = "1" ]; then
  echo "skip legacy edge MLP: jittor is not available"
fi

if [ "$RUN_SEQ" = "1" ]; then
  echo "training seq_nextdst component"
  for dataset in $DATASETS; do
    run_gpu_job "$PYTHON_BIN" scripts/train_nextdst.py \
      --data-dir "$DATA_DIR" \
      --model-dir "$MODEL_ROOT/seq" \
      --dataset "$dataset" \
      --history-ratio "$HISTORY_RATIO" \
      --sample-edges "$SEQ_SAMPLE_EDGES" \
      --neg-per-pos "$SEQ_NEG_PER_POS" \
      --epochs "$SEQ_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      $CUDA_ARG
  done
fi

if [ "$RUN_CRAFT" = "1" ]; then
  echo "training craft residual component"
  for dataset in $DATASETS; do
    run_gpu_job "$PYTHON_BIN" scripts/train_craft_residual.py \
      --data-dir "$DATA_DIR" \
      --model-dir "$MODEL_ROOT/craft" \
      --dataset "$dataset" \
      --history-ratio "$HISTORY_RATIO" \
      --sample-edges "$CRAFT_SAMPLE_EDGES" \
      --neg-per-pos "$CRAFT_NEG_PER_POS" \
      --epochs "$CRAFT_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      $CUDA_ARG
  done
fi

if [ "$RUN_SEQ" = "1" ] || [ "$RUN_CRAFT" = "1" ]; then
  wait_for_jobs
fi

echo "running large-pool validation"
OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
"$PYTHON_BIN" scripts/validate_large_pool.py \
  --data-dir "$DATA_DIR" \
  --model-root "$MODEL_ROOT" \
  --out "$REPORT_DIR/val_large_pool.csv" \
  --history-ratio "$HISTORY_RATIO" \
  --max-eval-edges "$VAL_MAX_EDGES" \
  --pool-size "$VAL_POOL_SIZE" \
  --candidate-mode "$VAL_CANDIDATE_MODE" \
  --legacy-top-k "$LEGACY_VALIDATE_TOP_K" \
  --workers "$CPU_WORKERS" \
  --batch-size "$BATCH_SIZE"

echo "running time-replay validation"
OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
"$PYTHON_BIN" scripts/time_replay_eval.py \
  --data-dir "$DATA_DIR" \
  --model-root "$MODEL_ROOT" \
  --out "$REPORT_DIR/time_replay.csv" \
  --summary-out "$REPORT_DIR/time_replay_summary.csv" \
  --blocks "$REPLAY_BLOCKS" \
  --max-block-events "$REPLAY_MAX_EVENTS" \
  --pool-size "$REPLAY_POOL_SIZE" \
  --candidate-mode "$REPLAY_CANDIDATE_MODE" \
  --legacy-top-k "$LEGACY_VALIDATE_TOP_K" \
  --workers "$CPU_WORKERS" \
  --batch-size "$BATCH_SIZE"

CONFIG_PATH="$MODEL_ROOT/fusion_config.json"

echo "selecting fusion"
REQUIRE_LEARNED_ARG=""
if [ "$REQUIRE_LEARNED" = "1" ]; then
  REQUIRE_LEARNED_ARG="--require-learned"
fi
"$PYTHON_BIN" scripts/select_fusion.py \
  --data-dir "$DATA_DIR" \
  --model-root "$MODEL_ROOT" \
  --validation "$REPORT_DIR/val_large_pool.csv" \
  --time-replay "$REPORT_DIR/time_replay_summary.csv" \
  --out "$CONFIG_PATH" \
  --legacy-top-k "$LEGACY_VALIDATE_TOP_K" \
  $REQUIRE_LEARNED_ARG

echo "running official-candidate sanity"
OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
"$PYTHON_BIN" scripts/official_candidate_sanity.py \
  --data-dir "$DATA_DIR" \
  --config "$CONFIG_PATH" \
  --out "$REPORT_DIR/official_candidate_sanity.json" \
  --adjusted-config "$CONFIG_PATH" \
  --max-rows "$SANITY_MAX_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$CPU_WORKERS" \
  $REQUIRE_LEARNED_ARG

echo "predicting official candidates"
OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER" \
"$PYTHON_BIN" scripts/predict_fusion.py \
  --data-dir "$DATA_DIR" \
  --config "$CONFIG_PATH" \
  --out-dir "$OUT_DIR" \
  --zip "$ZIP_PATH" \
  --batch-size "$BATCH_SIZE" \
  --workers "$CPU_WORKERS" \
  --max-rows "$PREDICT_MAX_ROWS" \
  --report "$REPORT_DIR/export_check.json"

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
