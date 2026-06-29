#!/usr/bin/env bash
set -euo pipefail

# Fast CRAFT-heavy version: one CRAFT parameter, stronger graph component, no sweep.
export ZIP_PATH="${ZIP_PATH:-result_v5_craft_fast.zip}"
export VALID_ROOT="${VALID_ROOT:-validation_v5_craft_fast}"
export CACHE_DIR="${CACHE_DIR:-feature_cache_v5_craft_fast}"
export MODEL_ROOT="${MODEL_ROOT:-fast_models_v5_craft}"
export SCORE_DIR="${SCORE_DIR:-fast_scores_v5_craft}"
export OUT_DIR="${OUT_DIR:-submission_v5_craft_fast}"

export SEED="${SEED:-2026}"
export FEATURE_WORKERS="${FEATURE_WORKERS:-48}"
export FUSE_RULE="${FUSE_RULE:-0.95}"
export HARD_RECENT_LIMIT="${HARD_RECENT_LIMIT:-80}"
export HARD_TRANSITION_LIMIT="${HARD_TRANSITION_LIMIT:-300}"
export HARD_POPULAR_LIMIT="${HARD_POPULAR_LIMIT:-3000}"
export HARD_POPULAR_SAMPLE="${HARD_POPULAR_SAMPLE:-350}"

export MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-128}"
export MLP_WEIGHT="${MLP_WEIGHT:-0.2}"
export MLP_EPOCHS="${MLP_EPOCHS:-8}"

export SEQ_LEN="${SEQ_LEN:-100}"
export SEQ_GAMMA="${SEQ_GAMMA:-0.25}"
export SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-192}"
export SEQ_EPOCHS="${SEQ_EPOCHS:-8}"

export RUN_CRAFT="${RUN_CRAFT:-1}"
export CRAFT_NEIGHBORS="${CRAFT_NEIGHBORS:-80}"
export CRAFT_HIDDEN_SIZE="${CRAFT_HIDDEN_SIZE:-128}"
export CRAFT_EPOCHS="${CRAFT_EPOCHS:-10}"

export BATCH_SIZE="${BATCH_SIZE:-512}"
export SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-256}"
export CRAFT_BATCH_SIZE="${CRAFT_BATCH_SIZE:-160}"
export GPU_COUNT="${GPU_COUNT:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export USE_CUDA="${USE_CUDA:-1}"

exec bash "$(dirname "$0")/run_fast_ranker.sh"
