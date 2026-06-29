#!/usr/bin/env bash
set -euo pipefail

# Fast no-CRAFT probe: one parameter set, capped validation by default.
export ZIP_PATH="${ZIP_PATH:-result_v4_nocraft_fast.zip}"
export VALID_ROOT="${VALID_ROOT:-validation_v4_nocraft_fast}"
export CACHE_DIR="${CACHE_DIR:-feature_cache_v4_nocraft_fast}"
export MODEL_ROOT="${MODEL_ROOT:-fast_models_v4_nocraft}"
export SCORE_DIR="${SCORE_DIR:-fast_scores_v4_nocraft}"
export OUT_DIR="${OUT_DIR:-submission_v4_nocraft_fast}"

export SEED="${SEED:-2026}"
export MAX_VALID="${MAX_VALID:-150000}"
export FEATURE_WORKERS="${FEATURE_WORKERS:-48}"
export FUSE_RULE="${FUSE_RULE:-1.0}"

export MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-128}"
export MLP_WEIGHT="${MLP_WEIGHT:-0.2}"
export MLP_EPOCHS="${MLP_EPOCHS:-8}"

export SEQ_LEN="${SEQ_LEN:-100}"
export SEQ_GAMMA="${SEQ_GAMMA:-0.25}"
export SEQ_HIDDEN_DIM="${SEQ_HIDDEN_DIM:-192}"
export SEQ_EPOCHS="${SEQ_EPOCHS:-8}"

export RUN_CRAFT="${RUN_CRAFT:-0}"

export BATCH_SIZE="${BATCH_SIZE:-512}"
export SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-256}"
export GPU_COUNT="${GPU_COUNT:-8}"
export MAX_PARALLEL="${MAX_PARALLEL:-8}"
export USE_CUDA="${USE_CUDA:-1}"

exec bash "$(dirname "$0")/run_fast_ranker.sh"
