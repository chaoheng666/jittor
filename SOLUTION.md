# Solution Notes

## Core Objective

For each test row `(src, time, candidate_1..candidate_100)`, the model scores
the 100 candidate destinations, normalizes the row to probabilities, and writes
the CSV required by the judge. Since the judge uses MRR, ranking quality is the
main objective; probability calibration is secondary.

## Why the Pipeline Is Split

Dataset1 and dataset2 have different structure:

- Dataset1 is repeat-heavy. Historical pair and recency rules are strong.
- Dataset2 is mostly a future new-pair task. Its `split` column provides a
  clean chronological train/validation boundary.

A single generic selector pushed both scenes back toward the same rule score and
made slow deep components expensive without improving ranking. The current
project therefore uses dataset-specific training paths.

## Current Data Flow

```text
data_A/
  dataset1/
  dataset2/

scripts/train_dataset1.sh
  -> train dataset1 rule/base artifact
  -> predict dataset1
  -> write dataset2 all-zero probe rows
  -> pack result_best.zip

scripts/train_dataset2.sh
  -> train dataset2 temporal recommender
  -> predict dataset2
  -> write dataset1 all-zero probe rows
  -> pack result_best.zip

scripts/train_all.sh
  -> train/predict both datasets
  -> pack official result_best.zip
```

## Dataset1 Method

Dataset1 uses `base_intensity_v3 + manual_rule` with a cold-destination penalty.
This preserves the high repeat-edge score and avoids unused deep residual
components.

## Dataset2 Method

Dataset2 uses a Jittor temporal recommender:

```text
score(src, dst, time) =
  dot(user_state(src, time), dst_embedding)
  + dst_bias
  + candidate_mlp(user_state, dst_embedding)   # candidate mode
```

`user_state` combines:

- source id embedding;
- recent destination-history sequence;
- position/recency weights over the sequence;
- per-position time-gap encoding;
- global time-gap/history-count features.

Training:

- validation mode: train on `split=0`, validate on `split=1`;
- final mode: retrain on all training rows;
- objective: full known-destination softmax or corrected large sampled softmax;
- sampled negatives: random known destinations plus hard negatives from source
  recent destinations and global popular destinations;
- auxiliary loss: small BPR term against the hardest sampled negative in the
  batch candidate set.
- auxiliary rerank loss: a small per-row candidate set trains the candidate MLP
  used by the final 100-candidate predictor.

Validation reports held-out MRR on frozen `split=0` history. The split evaluator
supports all known destinations, pseudo100, and hard-pseudo100 candidate pools,
split into `overall`, `repeated`, `new_pair`, `cold_dst`, `cold_src`, and
`no_history_src` buckets. Cold or no-history rows that the model cannot score
are counted explicitly with MRR 0 instead of being silently dropped. This checks
whether the model is learning next-destination ranking, not only beating a
handcrafted negative sampler.

The standalone split evaluator is:

```bash
python scripts/dataset2_split_eval.py \
  --modes rule_only,model_only,fusion,pop,recent \
  --eval-sets all-dst,pseudo100,hard-pseudo100
```

`rule_only` can run without Jittor. `model_only` and `fusion` require a trained
dataset2 artifact under `artifacts/dataset2`.

Prediction does not use the deep model alone. The final dataset2 score is:

```text
zscore(jittor_model_score) * D2_FUSION_MODEL_WEIGHT
+ zscore(rule_stat_score) * D2_FUSION_RULE_WEIGHT
```

The rule/statistical score keeps destination popularity and source recent
preference in the final ranker. It uses a lightweight dataset2 rule scorer at
prediction time to avoid millions of expensive feature-builder calls. On
dataset2 the default rule weight is small because the held-out positives are new
pairs; repeated-pair evidence should not dominate the neural new-link score.

## Performance Decisions

- Old zero-weight deep components were removed with the old generic stack.
- Probe all-zero CSVs are written row-by-row to avoid large zero matrices.
- Dataset predictions are written in probability chunks, so the pipeline does
  not keep the full test score matrix in memory.
- Dataset2 sampled softmax uses shared batch negatives and configurable
  `D2_NEG_COUNT`/`D2_BATCH_SIZE` to fit 16G GPUs.
- `D2_HARD_NEGATIVE_COUNT` adds rule-style hard negatives without making them
  the primary objective.
- `D2_SAMPLED_CORRECTION=1` subtracts an estimated sampling log-probability in
  the sampled-softmax logits.
- `D2_RERANK_NEG_COUNT` and `D2_RERANK_WEIGHT` train the candidate MLP without
  forcing the main sampled softmax to materialize per-row thousands of
  candidates.
- BPR is auxiliary only and defaults to `D2_BPR_WEIGHT=0.05`.
- Dataset2 rule fusion defaults to `D2_FUSION_RULE_WEIGHT=0.10`.
- Full softmax remains available through `D2_SOFTMAX_MODE=full`.
