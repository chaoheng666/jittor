# Jittor Future-Edge Fusion Ranker

This repository predicts future temporal edge intensity and ranks the 100
official candidates for each `(src, time)` test row.

The current pipeline uses a sanity-gated fusion system:

- `base_intensity_v3`: statistical future-edge intensity from repeat edges,
  recency, destination popularity, source transitions, and temporal CN/AA/RA.
- `manual_rule`: the original robust rule prior.
- `edge_mlp_legacy`: the previous Jittor MLP residual baseline, kept as a
  fallback and A/B comparison.
- `seq_nextdst`: optional Jittor next-destination sequence tower.
- `craft_residual`: optional Jittor target-aware residual with `zero_id`
  cold-node handling.

Deep components are disabled when they are missing or fail validation. When a
component is valid but weaker than the robust base/rule baselines, it is kept
only at a reduced weight instead of being hard-zeroed.

## Run

```bash
bash run_best.sh
```

Default output:

```text
result_best.zip
  dataset1.csv
  dataset2.csv
```

## Pipeline

1. Check or download `data_A`.
2. Analyze data distribution into `reports/data_stats/`.
3. Build `base_intensity_v3` artifacts.
4. Train optional Jittor deep components and legacy MLP if Jittor is available.
5. Run large-pool validation and time-replay validation.
6. Select `models_v2/fusion_config.json`.
7. Run official-candidate sanity and automatically reduce risky deep weights.
8. Score official candidates, softmax each row, and pack `result_best.zip`.

## Common Overrides

- `RUN_LEGACY=0`: skip legacy edge MLP training.
- `RUN_SEQ=0`: skip next-destination training.
- `RUN_CRAFT=0`: skip CRAFT residual training.
- `INSTALL_JITTOR=0`: do not install Jittor automatically.
- `VAL_MAX_EDGES=2000`, `VAL_POOL_SIZE=500`: broad mixed-pool validation
  budget for the future-edge intensity objective.
- `VAL_CANDIDATE_MODE=mixed`: keep the main validation aligned with scoring
  arbitrary `(src, dst, time)` points. `test-prior` remains available only as
  an optional diagnostic.
- `REPLAY_POOL_SIZE=500`, `REPLAY_CANDIDATE_MODE=mixed`: time-replay validation
  budget and sampling mode. Components with failed replay blocks are reduced or
  disabled during fusion selection.
- `SANITY_MAX_ROWS=5000`: official-candidate sanity sample size; set `0` for full.
- `PREDICT_MAX_ROWS=0`: final prediction row limit; keep `0` for real
  submissions, set a small value for shell smoke tests.
- `EDGE_NEGATIVE_MODE=mixed`: legacy MLP negative sampler mode.
- `LEGACY_VALIDATE_TOP_K=3`: validate the top legacy MLP candidates selected
  by training metadata, then choose among those by mixed validation metrics.
- `REQUIRE_LEARNED=1`: fail the full run if every learned component has zero
  usable fusion weight for any dataset, including after official-candidate
  sanity adjustments. Set `0` only for local smoke tests or intentional
  rule/base-only probes.
- `GPU_COUNT=8`, `MAX_PARALLEL=8`, `TOTAL_CPU_THREADS=48`: default resource
  plan for an 8-GPU / 48-core server. GPU jobs use per-GPU locks, legacy MLP
  jobs are split by hyperparameter and dataset, and each GPU job receives
  `CPU_THREADS_PER_JOB=ceil(TOTAL_CPU_THREADS / MAX_PARALLEL)` by default.
- `CPU_WORKERS=2`: run dataset-level CPU stages such as validation, sanity,
  and prediction in parallel. Each worker receives
  `CPU_THREADS_PER_WORKER=ceil(TOTAL_CPU_THREADS / CPU_WORKERS)` by default.
  The default worker count is `2` because the provided data has two datasets.
- `USE_CUDA=0`, `USE_VENV=0`: run in the current CPU Python environment.

Fast local probe without Jittor:

```bash
USE_CUDA=0 USE_VENV=0 INSTALL_JITTOR=0 RUN_LEGACY=0 \
REQUIRE_LEARNED=0 \
SEQ_SAMPLE_EDGES=100 CRAFT_SAMPLE_EDGES=100 \
VAL_MAX_EDGES=20 VAL_POOL_SIZE=50 \
REPLAY_BLOCKS=3 REPLAY_MAX_EVENTS=10 REPLAY_POOL_SIZE=50 \
SANITY_MAX_ROWS=100 PREDICT_MAX_ROWS=100 bash run_best.sh
```

Quick submission check:

```bash
python - <<'PY'
import csv, zipfile
with zipfile.ZipFile("result_best.zip") as zf:
    for name in sorted(zf.namelist()):
        with zf.open(name) as f:
            row = next(csv.reader(line.decode("utf-8") for line in f))
        vals = [float(x) for x in row]
        print(name, len(vals), min(vals), max(vals), sum(vals))
PY
```
