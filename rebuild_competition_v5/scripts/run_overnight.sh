#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs reports artifacts submission
PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_ARG=""
JITTOR_ARG="--enable-jittor"
if [[ "${SMOKE:-0}" == "1" ]]; then
  SMOKE_ARG="--smoke"
  JITTOR_ARG=""
fi

BASE_ARGS=(
  ${SMOKE_ARG}
  --data-dir data_A
  --teacher-zip /home/ma-user/work/jittor/result_pairwise_w05.zip
  --artifacts artifacts
  --reports reports
  --submission submission
  --seed "${SEED:-2026}"
)

echo "[run_overnight] start $(date -Is) smoke=${SMOKE:-0}"
"$PYTHON_BIN" -m src.submit "${BASE_ARGS[@]}" profile
"$PYTHON_BIN" -m src.submit "${BASE_ARGS[@]}" train-dataset --dataset dataset1
"$PYTHON_BIN" -m src.submit "${BASE_ARGS[@]}" ${JITTOR_ARG} \
  --max-valid-events "${MAX_VALID_EVENTS:-30000}" \
  --jittor-train-rows "${JITTOR_TRAIN_ROWS:-80000}" \
  --jittor-epochs "${JITTOR_EPOCHS:-8}" \
  --jittor-hidden "${JITTOR_HIDDEN:-192}" \
  train-dataset --dataset dataset2
"$PYTHON_BIN" -m src.submit "${BASE_ARGS[@]}" predict-dataset --dataset dataset1
"$PYTHON_BIN" -m src.submit "${BASE_ARGS[@]}" predict-dataset --dataset dataset2
"$PYTHON_BIN" -m src.submit "${BASE_ARGS[@]}" \
  --target-changes "${TARGET_CHANGES:-0.01,0.03,0.05,0.08,0.12}" \
  --package-prefix "${PACKAGE_PREFIX:-result_rebuild}" \
  ensemble
"$PYTHON_BIN" -m src.submit "${BASE_ARGS[@]}" pack-check
echo "[run_overnight] finish $(date -Is)"

