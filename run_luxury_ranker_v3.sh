#!/usr/bin/env bash
set -euo pipefail

# Fast repeat-edge conservative version: stronger rule prior and shorter source history.
export ZIP_PATH="${ZIP_PATH:-result_v3_repeat_fast.zip}"
export VALID_ROOT="${VALID_ROOT:-validation_v3_repeat_fast}"
export CACHE_DIR="${CACHE_DIR:-feature_cache_v3_repeat_fast}"
export MODEL_ROOT="${MODEL_ROOT:-fast_models_v3_repeat}"
export SCORE_DIR="${SCORE_DIR:-fast_scores_v3_repeat}"
export OUT_DIR="${OUT_DIR:-submission_v3_repeat_fast}"

export SEED="${SEED:-2026}"
export FEATURE_WORKERS="${FEATURE_WORKERS:-48}"
export FUSE_RULE="${FUSE_RULE:-1.25}"
export HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-150}"
export HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-160}"
export HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-1500}"
export HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-250}"

export MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-128}"
export MLP_WEIGHT="${MLP_WEIGHT:-0.1}"
export MLP_EPOCHS="${MLP_EPOCHS:-10}"

export SEQ_LEN="${SEQ_LEN:-50}"
export SEQ_GAMMA="${SEQ_GAMMA:-0.1}"
export SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-160}"
export SEQ_EPOCHS="${SEQ_EPOCHS:-10}"

export RUN_CRAFT="${RUN_CRAFT:-1}"
export CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-30}"
export CRAFT_HIDDEN_SIZE="${CRAFT_HIDDEN_SIZE:-64}"
export CRAFT_EPOCHS="${CRAFT_EPOCHS:-6}"

export BATCH_SIZE="${BATCH_SIZE:-512}"
export SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-256}"
export CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-240}"
export GPU_COUNT="${GPU_COUNT:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export USE_CUDA="${USE_CUDA:-1}"

exec bash "$(dirname "$0")/run_fast_ranker.sh"
