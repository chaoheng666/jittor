# Jittor Temporal Graph Re-ranker

This repository reranks the 100 provided candidates for each `(src, time)` test row.

The official pipeline now uses:

- rule ranking as the stable base scorer;
- `test-row` validation only for evaluation and ensemble selection;
- edge-level residual MLP training from raw temporal edges only;
- optional CRAFT temporal graph score caches when `jittor_geometric` is installed.

The old 100-candidate supervised MLP/sequence training path has been removed.

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

The script downloads `data_A` if it is missing, builds `test-row` validation, trains a broad edge-residual model sweep, runs the widest CRAFT sweep when `jittor_geometric` is available, searches a rule-anchored ensemble, and packs the submission.

## Common Overrides

- `VALID_MODE=test-row`: validation reuses real test candidate-row shape for evaluation only.
- `EDGE_HIDDEN_DIMS=64,128,256`: hidden sizes for edge residual MLPs.
- `EDGE_GAMMAS=0.05,0.08,0.15,0.25,0.35`: residual strength sweep.
- `EDGE_NEGATIVES=10`: negatives sampled per positive edge.
- `EDGE_SAMPLE_EDGES=250000`: cap supervision edges per dataset per model; set `0` for all.
- `RUN_CRAFT=0`: skip optional CRAFT training.
- `MAX_VALID=150000`: cap validation rows for faster local probing.

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
