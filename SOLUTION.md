# Main Branch Solution Notes

## Goal

The main branch is a conservative fusion ranker for the official 100-candidate
future-edge task. It keeps the strong rule/statistical baseline as the anchor
and adds Jittor residual components only when validation and sanity checks allow
them.

## Pipeline

`run_best.sh` is the entrypoint:

```text
data check/download
  -> data diagnostics
  -> base intensity artifacts
  -> legacy edge MLP grid
  -> seq_nextdst and craft_residual training
  -> mixed large-pool validation
  -> mixed time replay
  -> fusion config selection
  -> official-candidate sanity adjustment
  -> prediction and result_best.zip
```

## Validation

Validation defaults to a broad mixed negative pool with 500 candidates per row.
This keeps the main gate aligned with the intended future-edge intensity
objective: every `(src, dst, time)` should receive a meaningful score, not only
destinations from a constructed 100-candidate row.

`test-prior` candidate sampling still exists as an optional diagnostic and as a
sanity lens for the official candidate distribution, but it is not the default
model-selection gate.

Deep components are handled as follows:

- missing or validation-failed components get weight `0`;
- valid but weaker learned components are reduced instead of hard-zeroed;
- learned components with failed time-replay blocks are reduced or disabled;
- sanity failures reduce risky learned components and increase cold penalty
  when predicted unseen top candidates are too frequent.

The full `run_best.sh` path defaults to `REQUIRE_LEARNED=1`. If every learned
component collapses to zero usable weight for a dataset, the run fails instead
of silently producing a rule/base-only submission. The same check is applied
after official-candidate sanity adjustments.

## Resource Plan

The default script targets an 8-GPU, 48-core server:

```text
GPU_COUNT=8
MAX_PARALLEL=8
TOTAL_CPU_THREADS=48
CPU_THREADS_PER_JOB=ceil(TOTAL_CPU_THREADS / MAX_PARALLEL)
CPU_THREADS_PER_WORKER=ceil(TOTAL_CPU_THREADS / CPU_WORKERS)
```

GPU jobs use lock directories so one logical job occupies one visible GPU until
it exits. This avoids the previous round-robin issue where a new job could be
placed on a still-busy GPU while another GPU was idle.

The legacy MLP grid is split by hyperparameter and dataset, then scheduled
across the 8 GPUs. `seq_nextdst` and `craft_residual` are also split by dataset
and launched as independent GPU jobs, so the script avoids serially training
both datasets in one process when there is free GPU capacity.

CPU-only stages use `CPU_WORKERS=2` by default to process the two datasets in
parallel during validation, sanity checks, and prediction. Each worker gets an
explicit BLAS/Jittor thread budget so the 48-core server is used deliberately
without uncontrolled oversubscription.

## Current Caveat

The main branch is still a rule-first fusion system, not a fully calibrated
probabilistic intensity model. Mixed-pool and time-replay validation reduce the
risk of fitting one local candidate distribution, but they are still proxies for
the hidden judge. Learned components should only be trusted when they survive
both broad validation and time stability checks.
