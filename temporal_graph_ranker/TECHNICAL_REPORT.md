# Temporal Graph Ranker: Design Report

## Goal

Each test row supplies a source node, a query time, and 100 candidate
destinations.  The task is not global link generation; it is candidate
re-ranking.  The only meaningful target is the position of the true
destination among those 100 candidates, measured by MRR (mean reciprocal
rank).  The project writes a valid probability distribution after ranking.

## Data Findings

| Property | dataset1 | dataset2 |
| --- | ---: | ---: |
| Training edges | 690,848 | 2,261,283 |
| Source nodes | 22,093 | 12,708 |
| Destination nodes | 23,012 | 50,640 |
| Repeated-edge fraction | 72.6% | 2.2% |
| Test time follows training | yes | yes |

The two data sets need different inductive biases.  `dataset1` has many
repeated source-destination interactions, so pair history and recent exact
matches are valuable.  `dataset2` has very few repeated pairs, so memorizing a
previous pair is unreliable; recent destination popularity, temporal trend,
low-rank graph structure, source interest profiles, and destination transition
statistics matter more.

## Stable Rule Stage

`src/stable_stage.py` builds a `GraphFeatureModel` independently for each data
set.  It uses only the supplied training CSV and the supplied test candidate
IDs.  The model produces 21 features for every source/candidate pair:

1. `rule`: a dataset-specific hand-designed combination of the signals below.
2. `pop`, `recent_pop`, `trend`: all-time, recent-window, and recent-versus-old
   destination activity.
3. `recency`: how recently the candidate destination appeared before the query.
4. `src_recent_exact`, `pair_log`: source history and repeated-pair evidence.
5. `dst_known`, `degree_cap`, `candidate_seen_in_test`: candidate support and
   test-candidate distribution statistics.
6. `svd`, `profile`: low-rank graph similarity and a recency-weighted source
   interest profile.
7. `transition`: a decayed destination-to-destination transition score from
   each source sequence.
8. Eight reciprocal-rank versions of the strongest signals.

Rule weights are tuned on time-respecting validation sets: synthetic hard
negatives, real test-candidate templates with a held-out positive injected, and
a low-popularity variant.  Weight search caches normalized feature tensors and
performs coordinate updates, rather than recomputing all 21 row z-scores for
every candidate weight.

The stable dataset2 score is:

```text
stable_logits = zscore(feature_logits) + 5.5 * zscore(stable_mlp_logits)
```

`stable_mlp_logits` comes from a Jittor candidate MLP trained on the same
21 features.  The `prepare` action deliberately omits this NPU stage, so it is
safe to run beside a previous NPU job.  Full `all` includes it.

## Context Candidate Ranker

`src/context_stage.py` adds 14 features to the stable 21, for 35 in total.
They encode source sequence mean/max/last similarity, recent destination
audience similarity, audience size, exact recent audience membership, rank
versions of sequence signals, source activity, recent-history coverage, and
three source-activity interaction features.

The ranker is a residual, three-hidden-layer Jittor MLP:

```text
35 -> 384 -> 384 -> 384 -> 1
score = nonlinear_tower + 0.15 * linear_skip
```

It is trained with candidate-set cross entropy: each training row has exactly
100 candidates and the label is the true destination index.  Two training
sources are mixed:

1. 400,000 hard lists, whose negatives are high-scoring under the stable rule.
2. 400,000 test-template lists, where a held-out real positive is inserted into
   a real test row from the same source whenever possible.

The template examples are important: they reproduce the online candidate
frequency and known/unknown destination mix, instead of relying solely on
random negative sampling.  Validation combines 60,000 hard lists and 100,000
template lists.

The default ranker uses three seeds (`3101,3102,3103`), batch size 4096,
hidden width 384, ten epochs, Adam with learning rate `8e-4`, and weight decay
`1e-5`.  Their row-z-scored logits are averaged.

## Submission Fusion

`dataset1.csv` is produced by the stable rule stage.  Dataset2 uses

```text
submission_logits =
    (1 - blend_weight) * zscore(stable_logits)
  + blend_weight * zscore(context_logits)
```

`blend_weight=0.10` is the conservative primary submission.  The script also
packages `0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.35, 1.00`, so the final
submission can be selected after online feedback without retraining.

## Server Execution Design

The server has 192 physical ARM cores, eight NUMA nodes, 1.5 TiB RAM, and one
Ascend 910B.  The pipeline treats them as separate resources:

- CPU feature workers use POSIX `fork` and share the fitted read-only graph
  model copy-on-write.  This avoids serializing one large graph pickle per
  worker.
- Feature tensors are built with up to 96 workers, pinned by default to cores
  64-191.  `numactl --interleave=all` distributes allocations across NUMA
  nodes.
- BLAS is restricted to one thread during feature workers. Stable and context
  graphs use 128-dimensional sparse SVD, with a bounded 32-thread BLAS pool
  through `threadpoolctl` when the numerical backend supports it.
- Training uses the Ascend device with batch size 4096.  The environment's
  Jittor ACL backend cannot backpropagate through `Embedding`, so the design
  intentionally uses robust precomputed graph and sequence features rather
  than an unverified embedding lookup path.

This separation keeps the high-parallel CPU stages fast and prevents a hidden
BLAS oversubscription from turning 96 workers into thousands of runnable
threads.
