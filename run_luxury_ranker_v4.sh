#!/usr/bin/env bash
set -euo pipefail

# Fast robust ablation: no CRAFT, capped validation, good for quick online probes.
export ZIP_PATH="${ZIP_PATH:-result_v4_fast.zip}"
export VALID_ROOT="${VALID_ROOT:-validation_v4_fast}"
export MODEL_ROOT="${MODEL_ROOT:-competition_models_v4_fast}"
export SCORE_DIR="${SCORE_DIR:-competition_scores_v4_fast}"
export OUT_DIR="${OUT_DIR:-submission_v4_fast}"

export VALID_MODE="${VALID_MODE:-test-prior}"
export MAX_VALID="${MAX_VALID:-150000}"
export MAX_COLD_POOL="${MAX_COLD_POOL:-1000000}"
export FUSE_RULE="${FUSE_RULE:-1.0}"

export HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-80}"
export HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-250}"
export HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-2500}"
export HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-300}"

export MLP_SEEDS="${MLP_SEEDS:-2026,2027}"
export MLP_HIDDEN_DIMS="${MLP_HIDDEN_DIMS:-64,128,256}"
export MLP_WEIGHTS="${MLP_WEIGHTS:-0.1,0.2,0.35}"
export MLP_EPOCHS="${MLP_EPOCHS:-10}"

export SEQ_SEEDS="${SEQ_SEEDS:-2026,2027}"
export SEQ_LENS="${SEQ_LENS:-30,50,100}"
export SEQ_GAMMAS="${SEQ_GAMMAS:-0.1,0.2,0.35}"
export SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-192}"
export SEQ_EPOCHS="${SEQ_EPOCHS:-10}"

export RUN_CRAFT="${RUN_CRAFT:-0}"

export BATCH_SIZE="${BATCH_SIZE:-256}"
export SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-192}"
export GPU_COUNT="${GPU_COUNT:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export USE_CUDA="${USE_CUDA:-1}"

exec bash "$(dirname "$0")/run_luxury_ranker.sh"
