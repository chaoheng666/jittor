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
  dataset2_split_eval.py
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
source-history state, recent-destination sequence features, position weights,
and time-gap features. The main objective is destination softmax over all known
destinations or a corrected large sampled approximation. A small BPR term is
used only as an auxiliary ranking loss.

Split validation reports bucketed MRR for:

- repeated pairs;
- new pairs;
- cold destinations that are not in the training destination vocabulary.
- cold sources and sources with no usable history.

Prediction fuses the Jittor model score with the dataset2 rule/statistical
score, so destination popularity and source recent preference remain active.
The default rule weight is deliberately small on dataset2 because the local
held-out positives are new pairs; repeated-pair rules are kept as weak evidence,
not as the dominant signal.

Useful overrides:

```bash
D2_SOFTMAX_MODE=sampled|full
D2_NEG_COUNT=4096
D2_EPOCHS=6
D2_BATCH_SIZE=2048
D2_BPR_WEIGHT=0.05
D2_HARD_NEGATIVE_COUNT=512
D2_SAMPLED_CORRECTION=1
D2_RERANK_NEG_COUNT=64
D2_RERANK_WEIGHT=0.10
D2_FUSION_MODEL_WEIGHT=1.0
D2_FUSION_RULE_WEIGHT=0.10
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

## Dataset2 Split Evaluation

Run the split evaluator before trusting any dataset2 model change:

```bash
python scripts/dataset2_split_eval.py \
  --modes pop,recent,rule_only \
  --eval-sets pseudo100,hard-pseudo100 \
  --max-events 20000
```

After a dataset2 artifact exists, compare all scoring paths:

```bash
python scripts/dataset2_split_eval.py \
  --modes rule_only,model_only,fusion,pop,recent \
  --eval-sets all-dst,pseudo100,hard-pseudo100 \
  --cuda
```

The evaluator writes bucketed MRR for `overall`, `repeated`, `new_pair`,
`cold_dst`, `cold_src`, and `no_history_src`. Validation is frozen to `split=0`
history; skipped model samples are counted in the `overall` denominator with
MRR 0. Full `rule_only` over `all-dst` is exact but slow because it scores every
known destination. `train_dataset2.sh` keeps validation-before-final on by
default for probing; `train_all.sh` turns it off by default to avoid training
dataset2 twice during the final packaging run.
