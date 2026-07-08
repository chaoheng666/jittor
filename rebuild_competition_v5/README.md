# Jittor Rebuild v5

Final competition pipeline for the best known submission:

- Package: `result_v5_block_blend_0p10.zip`
- Public score: `1.2829`
- Dataset1: copied from the v3 `mlpw_5p5` package
- Dataset2: `90%` v3 MLP baseline logits + `10%` v5 block MLP logits after row z-score normalization

## Run

The v5 pipeline depends on the v3 rebuild outputs for the dataset1 CSV and the dataset2 baseline logits. On the competition server this is expected at:

```bash
/home/ma-user/work/jittor_rebuild_v3
```

Run v5 from this directory:

```bash
bash scripts/v5_run_block.sh
```

The default final output is:

```bash
submission/result_v5_block_blend_0p10.zip
```

## Steps

```bash
bash scripts/v5_00_build_block.sh
bash scripts/v5_01_train_block_mlp.sh
bash scripts/v5_02_predict_pack.sh
```

`v5_02_predict_pack.sh` only packages the best known variant by default. To reproduce a different blend without editing code:

```bash
BLEND_WEIGHT=0.05 OUTPUT_NAME=result_v5_block_blend_0p05 bash scripts/v5_02_predict_pack.sh
```

## Notes

- The trained block MLP uses seeds `3101,3102,3103`.
- The default blend weight is intentionally fixed at `0.10` because that is the highest known online score from the v5 sweep.
- Generated artifacts, logs, CSV files, and zip packages are intentionally ignored by git.
