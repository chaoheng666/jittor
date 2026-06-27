#!/usr/bin/env bash
set -euo pipefail

# Best-first full ensemble: balanced for dataset1 repeated edges and dataset2 new links.
export ZIP_PATH="${ZIP_PATH:-result_v1.zip}"
export VALID_ROOT="${VALID_ROOT:-validation_v1}"
export MODEL_ROOT="${MODEL_ROOT:-competition_models_v1}"
export SCORE_DIR="${SCORE_DIR:-competition_scores_v1}"
export OUT_DIR="${OUT_DIR:-submission_v1}"

export VALID_MODE="${VALID_MODE:-test-prior}"
export MAX_VALID="${MAX_VALID:-0}"
export MAX_COLD_POOL="${MAX_COLD_POOL:-3000000}"
export FUSE_RULE="${FUSE_RULE:-1.0}"

export HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-100}"
export HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-300}"
export HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-3000}"
export HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-350}"

export MLP_SEEDS="${MLP_SEEDS:-2026,2027,2028}"
export MLP_HIDDEN_DIMS="${MLP_HIDDEN_DIMS:-64,128,256}"
export MLP_WEIGHTS="${MLP_WEIGHTS:-0.08,0.15,0.25,0.4}"
export MLP_EPOCHS="${MLP_EPOCHS:-14}"

export SEQ_SEEDS="${SEQ_SEEDS:-2026,2027}"
export SEQ_LENS="${SEQ_LENS:-30,50,100}"
export SEQ_GAMMAS="${SEQ_GAMMAS:-0.08,0.15,0.25,0.4}"
export SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-192}"
export SEQ_EPOCHS="${SEQ_EPOCHS:-14}"

export RUN_CRAFT="${RUN_CRAFT:-1}"
export CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-30,50}"
export CRAFT_HIDDEN_SIZES="${CRAFT_HIDDEN_SIZES:-64,128}"
export CRAFT_EPOCHS="${CRAFT_EPOCHS:-8}"

export BATCH_SIZE="${BATCH_SIZE:-256}"
export SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-192}"
export CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-200}"
export GPU_COUNT="${GPU_COUNT:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export USE_CUDA="${USE_CUDA:-1}"

exec bash "$(dirname "$0")/run_luxury_ranker.sh"
