# Temporal Graph Ranker

An independent Jittor project for the dynamic graph link-prediction track.  It
reads only `data_A/dataset1` and `data_A/dataset2`, produces both CSV files,
and packages competition-ready ZIP submissions.

The project has one entry script:

```bash
bash scripts/run_pipeline.sh
```

`ACTION=all` executes the complete workflow.  The command creates the stable
rule baseline, trains the candidate ranker on Ascend, predicts both datasets,
and emits several submission blends.

```bash
ACTION=all bash scripts/run_pipeline.sh
```

When another task is already using the NPU, start only the CPU-heavy preparation
stage.  It is pinned to CPU cores 64-191 by default, leaving 64 physical cores
available to the existing task and does not modify any other project directory.

```bash
ACTION=prepare bash scripts/run_pipeline.sh
```

After the NPU is free, run the remaining stages without repeating the CPU preparation:

```bash
ACTION=neural bash scripts/run_pipeline.sh
```

The useful output paths are:

```text
submission/temporal_ranker_blend_0p10.zip
submission/temporal_ranker_blend_0p02.zip
submission/temporal_ranker_blend_0p05.zip
submission/temporal_ranker_blend_0p08.zip
submission/temporal_ranker_blend_0p12.zip
submission/temporal_ranker_blend_0p15.zip
submission/temporal_ranker_blend_0p20.zip
submission/temporal_ranker_blend_0p35.zip
submission/temporal_ranker_pure.zip
```

The design and parameter rationale are in `TECHNICAL_REPORT.md`.
