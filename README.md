# Jittor Future-Edge Intensity Ranker

This repository predicts the strength of future temporal edges and ranks the
100 official candidates for each `(src, time)` test row.

The current pipeline no longer builds a synthetic validation candidate set. It
learns from real future edges with hard negatives, selects the best model by
future-edge pairwise accuracy, and then scores the official candidates.

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
2. Train edge-intensity MLP sweeps from real temporal edges.
3. Select the best model per dataset using future-edge pairwise accuracy.
4. Score the 100 official candidates for each test row.
5. Softmax each row and pack `result_best.zip`.

## Common Overrides

- `EDGE_HIDDEN_DIMS=64,128,256`: MLP hidden-size sweep.
- `EDGE_GAMMAS=0.05,0.08,0.15,0.25,0.35`: residual strength sweep.
- `EDGE_SEEDS=2026,2027`: training seeds.
- `EDGE_NEGATIVES=10`: hard negatives per real future edge.
- `EDGE_SAMPLE_EDGES=250000`: cap supervision edges per dataset per model; set `0` for all.
- `MIN_FUTURE_GAIN=0.0005`: minimum gain over rule-only before selecting an MLP.
- `USE_CUDA=0`: run on CPU.
- `USE_VENV=0`: use the current Python environment.

Fast local probe:

```bash
USE_CUDA=0 USE_VENV=0 MAX_PARALLEL=1 GPU_COUNT=1 \
EDGE_HIDDEN_DIMS=8 EDGE_GAMMAS=0.05 EDGE_SEEDS=2026 \
EDGE_EPOCHS=1 EDGE_NEGATIVES=2 EDGE_SAMPLE_EDGES=2000 \
bash run_best.sh
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
