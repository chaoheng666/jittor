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

Deep components receive non-zero fusion weights only when large-pool and
time-replay validation do not show a regression. Official test candidates are
used only for sanity checks and final export.

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
- `VAL_MAX_EDGES=2000`, `VAL_POOL_SIZE=2000`: large-pool validation budget.
- `SANITY_MAX_ROWS=5000`: official-candidate sanity sample size; set `0` for full.
- `EDGE_NEGATIVE_MODE=mixed`: legacy MLP negative sampler mode.
- `USE_CUDA=0`, `USE_VENV=0`: run in the current CPU Python environment.

Fast local probe without Jittor:

```bash
USE_CUDA=0 USE_VENV=0 INSTALL_JITTOR=0 RUN_LEGACY=0 \
SEQ_SAMPLE_EDGES=100 CRAFT_SAMPLE_EDGES=100 \
VAL_MAX_EDGES=20 VAL_POOL_SIZE=50 \
REPLAY_BLOCKS=3 REPLAY_MAX_EVENTS=10 REPLAY_POOL_SIZE=50 \
SANITY_MAX_ROWS=100 bash run_best.sh
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
