# Jittor Temporal Link Reranker

This repository predicts future temporal links for the competition format:
each test row provides `(src, time)` and 100 candidate destinations, and the
submission returns one probability row per test row.

The codebase is intentionally split by dataset behavior:

- `dataset1`: repeat-heavy scene. Uses the robust statistical/rule ranker.
- `dataset2`: new-link scene. Uses the `split` column and trains a Jittor
  temporal recommender for next-destination ranking.

## Entrypoints

```bash
bash scripts/train_dataset1.sh
bash scripts/train_dataset2.sh
bash scripts/train_all.sh
```

- `train_dataset1.sh`: trains/predicts dataset1 and writes dataset2 as all-zero
  rows for single-scene score probing.
- `train_dataset2.sh`: trains/predicts dataset2 and writes dataset1 as all-zero
  rows for single-scene score probing.
- `train_all.sh`: normal submission path; predicts both datasets.

Default output:

```text
result_best.zip
  dataset1.csv
  dataset2.csv
```

## File Layout

```text
scripts/
  run_specialized_pipeline.py
  train_dataset1.sh
  train_dataset2.sh
  train_all.sh

src/
  common/        # CSV/zip/report helpers
  dataset1/      # repeat/rule ranker
  dataset2/      # Jittor temporal recommender
  data_loader.py
  metrics.py
  feature_builder.py
  rule_ranker_v2.py
  base_intensity_v3.py
```

Generated outputs are ignored by git:

```text
artifacts*/
reports/
submission*/
*.zip
```

## Dataset1

Dataset1 keeps the conservative `base_intensity_v3 + manual_rule` stack. This
scene has a high repeat-edge signal, so the pipeline avoids slow residual deep
components that previously collapsed back to rule scoring.

## Dataset2

Dataset2 is treated as a new-link temporal recommendation problem.

When the `split` column exists:

- `split=0` is the local training window.
- `split=1` is the local validation window.
- final submission training can retrain on all training rows.

The Jittor model learns source embeddings, destination embeddings, a compact
source-history state, and time-gap features. The main objective is destination
softmax over all known destinations or a large sampled approximation. A small
BPR term is used only as an auxiliary ranking loss.

Split validation reports bucketed MRR for:

- repeated pairs;
- new pairs;
- cold destinations that are not in the training destination vocabulary.

Prediction fuses the Jittor model score with the dataset2 rule/statistical
score, so repeated-pair signals, destination popularity, source recent
preference, and cold-destination downweighting remain active.

Useful overrides:

```bash
D2_SOFTMAX_MODE=sampled|full
D2_NEG_COUNT=4096
D2_EPOCHS=6
D2_BATCH_SIZE=512
D2_BPR_WEIGHT=0.05
D2_FUSION_MODEL_WEIGHT=1.0
D2_FUSION_RULE_WEIGHT=0.25
D2_VALIDATE_BEFORE_FINAL=0|1
FINAL_TRAIN=0|1
```

Example:

```bash
D2_SOFTMAX_MODE=full D2_EPOCHS=4 bash scripts/train_dataset2.sh
```

## Local Checks

```bash
python -m compileall src scripts
python scripts/run_specialized_pipeline.py --target dataset1 --zero-other 1 --max-rows 10 --cuda 0
```
