#!/usr/bin/env bash
set -euo pipefail

# Repeat-edge conservative: stronger rule prior, smaller residuals, recent-history hard negatives.
export ZIP_PATH="${ZIP_PATH:-result_v3_repeat.zip}"
export VALID_ROOT="${VALID_ROOT:-validation_v3_repeat}"
export MODEL_ROOT="${MODEL_ROOT:-competition_models_v3_repeat}"
export SCORE_DIR="${SCORE_DIR:-competition_scores_v3_repeat}"
export OUT_DIR="${OUT_DIR:-submission_v3_repeat}"

export VALID_MODE="${VALID_MODE:-test-prior}"
export MAX_VALID="${MAX_VALID:-0}"
export MAX_COLD_POOL="${MAX_COLD_POOL:-2000000}"
export FUSE_RULE="${FUSE_RULE:-1.25}"

export HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-150}"
export HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-160}"
export HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-1500}"
export HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-250}"

export MLP_SEEDS="${MLP_SEEDS:-2026,2027,2028}"
export MLP_HIDDEN_DIMS="${MLP_HIDDEN_DIMS:-64,128,256}"
export MLP_WEIGHTS="${MLP_WEIGHTS:-0.05,0.1,0.2}"
export MLP_EPOCHS="${MLP_EPOCHS:-12}"

export SEQ_SEEDS="${SEQ_SEEDS:-2026,2027}"
export SEQ_LENS="${SEQ_LENS:-20,30,50}"
export SEQ_GAMMAS="${SEQ_GAMMAS:-0.05,0.1,0.2}"
export SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-160}"
export SEQ_EPOCHS="${SEQ_EPOCHS:-12}"

export RUN_CRAFT="${RUN_CRAFT:-1}"
export CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-20,30,50}"
export CRAFT_HIDDEN_SIZES="${CRAFT_HIDDEN_SIZES:-64}"
export CRAFT_EPOCHS="${CRAFT_EPOCHS:-8}"

export BATCH_SIZE="${BATCH_SIZE:-256}"
export SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-192}"
export CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-220}"
export GPU_COUNT="${GPU_COUNT:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export USE_CUDA="${USE_CUDA:-1}"

exec bash "$(dirname "$0")/run_luxury_ranker.sh"
