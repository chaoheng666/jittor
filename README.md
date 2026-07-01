# Jittor Temporal Graph Re-Ranker

赛道一候选内精排方案。测试集已给出每个 `(src, time)` 的 100 个候选 `dst`，本仓库只负责重排候选，不做全量召回。

## 当前结论

- `rule` 仍是最稳的主干。线上分数相同的版本，本质上都只选中了规则组件。
- 旧版 MLP/Seq 组件被忽略的主要原因是训练和推理都把 `rule_score` 融进模型分数，信号高度共线；如果内部 holdout 没超过规则，还会直接退化成规则分数副本。
- CRAFT 在本地验证上看起来更高，但线上掉分，说明当前合成验证集对动态图组件不可靠；默认链路已移除 CRAFT。
- 当前优化方向是 `rule + neural residual`：规则负责强先验，MLP/Seq 只作为残差修正规则排序错误。

## 一键运行

Linux 服务器运行：

```bash
bash run_fast_ranker.sh
```

默认输出：

```text
result_fast.zip
  dataset1.csv
  dataset2.csv
```

8 卡运行建议：

```bash
GPU_COUNT=8 MAX_PARALLEL=8 USE_CUDA=1 bash run_fast_ranker.sh
```

内存紧张时限制验证样本：

```bash
MAX_VALID=150000 GPU_COUNT=8 MAX_PARALLEL=8 USE_CUDA=1 bash run_fast_ranker.sh
```

## 常用参数

- `MAX_VALID=0`：使用全部验证样本；内存不足时设为 `100000` 或 `150000`。
- `VALID_MODE=test-prior`：默认，按测试候选 cold 比例构造验证集。
- `HARD_NEGATIVES=30`：训练时只让 label 和规则高分负样本参与主要梯度，降低 easy negatives 对残差模型的稀释。
- `MLP_HIDDEN_DIM=128`、`MLP_WEIGHT=0.2`：残差 MLP 配置。
- `SEQ_LEN=100`、`SEQ_GAMMA=0.25`、`SEQ_HIDDEN_DIM=192`：source 历史序列残差配置。
- `FEATURE_WORKERS=48`：Linux 上构建特征缓存的并行 worker 数；Windows 会自动退回单进程。

## 手动流程

```bash
python scripts/valid_builder.py --data-dir data_A --out-dir validation_fast_test-prior --valid-mode test-prior --max-valid 150000
python scripts/build_feature_cache.py --data-dir data_A --valid-dir validation_fast_test-prior --cache-dir feature_cache_fast --seq-lens 100 --workers 48
python scripts/train_ranker.py --valid-dir validation_fast_test-prior --cache-dir feature_cache_fast --model-dir fast_models/mlp --hidden-dim 128 --mlp-weight 0.2 --hard-negatives 30 --cuda
python scripts/train_seq_ranker.py --valid-dir validation_fast_test-prior --cache-dir feature_cache_fast --model-dir fast_models/seq --seq-len 100 --gamma 0.25 --hidden-dim 192 --hard-negatives 30 --cuda
python scripts/search_ensemble.py --valid-dir validation_fast_test-prior --cache-dir feature_cache_fast --model-root fast_models --out fast_models/ensemble_weights.json
python scripts/predict_luxury_ensemble.py --data-dir data_A --cache-dir feature_cache_fast --weights fast_models/ensemble_weights.json --zip result_fast.zip
```

提交前快速检查：

```bash
python - <<'PY'
import csv, zipfile
with zipfile.ZipFile("result_fast.zip") as zf:
    for name in sorted(zf.namelist()):
        with zf.open(name) as f:
            row = next(csv.reader(line.decode("utf-8") for line in f))
        vals = [float(x) for x in row]
        print(name, len(vals), min(vals), max(vals), sum(vals))
PY
```
