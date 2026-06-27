#!/usr/bin/env bash
set -euo pipefail

# New-link / bipartite aggressive: more transition, popularity, and larger residual room.
export ZIP_PATH="${ZIP_PATH:-result_v2_newlink.zip}"
export VALID_ROOT="${VALID_ROOT:-validation_v2_newlink}"
export MODEL_ROOT="${MODEL_ROOT:-competition_models_v2_newlink}"
export SCORE_DIR="${SCORE_DIR:-competition_scores_v2_newlink}"
export OUT_DIR="${OUT_DIR:-submission_v2_newlink}"

export VALID_MODE="${VALID_MODE:-test-prior}"
export MAX_VALID="${MAX_VALID:-0}"
export MAX_COLD_POOL="${MAX_COLD_POOL:-3000000}"
export FUSE_RULE="${FUSE_RULE:-0.85}"

export HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-60}"
export HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-450}"
export HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-5000}"
export HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-500}"

export MLP_SEEDS="${MLP_SEEDS:-2026,2027,2028}"
export MLP_HIDDEN_DIMS="${MLP_HIDDEN_DIMS:-128,256}"
export MLP_WEIGHTS="${MLP_WEIGHTS:-0.2,0.35,0.5}"
export MLP_EPOCHS="${MLP_EPOCHS:-14}"

export SEQ_SEEDS="${SEQ_SEEDS:-2026,2027,2028}"
export SEQ_LENS="${SEQ_LENS:-50,100,150}"
export SEQ_GAMMAS="${SEQ_GAMMAS:-0.2,0.35,0.5}"
export SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-256}"
export SEQ_EPOCHS="${SEQ_EPOCHS:-14}"

export RUN_CRAFT="${RUN_CRAFT:-1}"
export CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-50,80}"
export CRAFT_HIDDEN_SIZES="${CRAFT_HIDDEN_SIZES:-64,128}"
export CRAFT_EPOCHS="${CRAFT_EPOCHS:-8}"

export BATCH_SIZE="${BATCH_SIZE:-256}"
export SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-128}"
export CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-160}"
export GPU_COUNT="${GPU_COUNT:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export USE_CUDA="${USE_CUDA:-1}"

exec bash "$(dirname "$0")/run_luxury_ranker.sh"
