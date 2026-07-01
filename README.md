# Jittor Temporal Graph Re-Ranker

赛道一候选内精排方案。测试集已经给出每个 `(src, time)` 的 100 个候选 `dst`，本项目只做候选重排，不做全量召回。

## 当前冲分方案

当前主线已经从 `rule + neural residual` 升级为：

```text
3 折时间验证
  + 扩展时序/二阶/候选先验特征
  + LightGBM LambdaRank 主排序
  + Jittor Temporal GNN 学习动态图邻域传播
  + Jittor MLP/Seq residual 补充
  + fold ensemble 平均
```

核心判断：

- `dataset1` 高重复边，历史 pair 和 recency 是强先验。
- `dataset2` 二部图新链接，测试候选 cold 比例高，需要显式建模 dst 热度、转移、共现和候选分布。
- LightGBM LambdaRank 直接学习每行 100 个候选的排序，比让神经网络从规则分数上做小残差更适合当前任务。
- Temporal GNN 从 query time 之前的 source/candidate 历史邻域采样消息，显式学习随时间演化的图连接模式，而不是手写 `A->B->C` 规则。
- Jittor 模型保留为补充信号：MLP residual 和候选感知 time-aware sequence residual。

## 一键训练

Linux 服务器运行：

```bash
GPU_COUNT=8 MAX_PARALLEL=8 USE_CUDA=1 USE_LGBM=1 bash run_fast_ranker.sh
```

默认输出：

```text
result_fast.zip
  dataset1.csv
  dataset2.csv
```

同时会生成：

```text
rank_folds_fast/        # fold0 last10%, fold1 last15%, fold2 last20%
feature_cache_fast/     # 每折 valid/test 特征缓存
fast_scores/            # LightGBM valid/test score
fast_models/            # LGBM/Jittor 模型、每折 ensemble 权重、ablation_summary.csv
submission_fast/        # 解压后的提交 csv
```

## 默认参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `MAX_VALID` | `150000` | 每个 fold 每个 dataset 的验证 query 上限 |
| `LGBM_MAX_ROWS` | `120000` | LightGBM/Jittor 实际训练和融合搜索 query 上限 |
| `SEQ_LEN` | `200` | source 历史序列长度 |
| `SEQ_HIDDEN_DIM` | `256` | Seq residual hidden size |
| `SEQ_EPOCHS` | `12` | Seq residual epochs |
| `MLP_HIDDEN_DIM` | `256` | MLP residual hidden size |
| `MLP_EPOCHS` | `10` | MLP residual epochs |
| `HARD_NEGATIVES` | `30` | hard-negative CE/BPR 使用的规则高分负样本数 |
| `BPR_WEIGHT` | `0.1` | Jittor residual 的 BPR 辅助 loss 权重 |
| `FEATURE_WORKERS` | `48` | Linux 特征构建 worker 数 |
| `REQUIRE_LGBM_BETTER` | `1` | LightGBM 在每折每个 dataset 上必须超过 rule，否则停止 |
| `MIN_TOP1_DIFF` | `0.15` | 最终提交与 rule-only 的 top1 差异比例下限 |
| `RUN_TGNN` | `1` | 是否训练 Jittor Temporal GNN ranker |
| `TGNN_EPOCHS` | `8` | TGNN epochs |
| `TGNN_BATCH_SIZE` | `128` | TGNN batch size |
| `TGNN_NODE_EMB_DIM` | `128` | TGNN node embedding 维度 |
| `TGNN_HIDDEN_DIM` | `192` | TGNN hidden size |
| `TGNN_SRC_NEIGHBORS` | `50` | source 历史邻居采样数 |
| `TGNN_CAND_NEIGHBORS` | `30` | 每个 candidate 历史邻居采样数 |
| `TGNN_SECOND_HOP` | `20` | source 二跳 temporal ego graph 采样数 |

LightGBM 按数据集自动使用不同参数：

- `dataset1`：`num_leaves=31`、`learning_rate=0.03`、`min_data_in_leaf=200`、`num_boost_round=2500`。
- `dataset2`：`num_leaves=63`、`learning_rate=0.02`、`min_data_in_leaf=300`、`num_boost_round=3000`。

## 手动流程

```bash
python scripts/build_rank_folds.py --data-dir data_A --fold-root rank_folds_fast --max-valid 150000
python scripts/build_feature_cache.py --data-dir data_A --fold-root rank_folds_fast --cache-dir feature_cache_fast --seq-lens 200 --workers 48

python scripts/train_lgbm_ranker.py --cache-dir feature_cache_fast/fold0 --model-dir fast_models/fold0/lgbm --score-dir fast_scores/fold0 --max-rows 120000
python scripts/train_tgnn_ranker.py --data-dir data_A --valid-dir rank_folds_fast/fold0 --cache-dir feature_cache_fast/fold0 --model-dir fast_models/fold0/tgnn --score-dir fast_scores/fold0 --max-rows 120000 --cuda
python scripts/train_ranker.py --valid-dir rank_folds_fast/fold0 --cache-dir feature_cache_fast/fold0 --model-dir fast_models/fold0/mlp --hidden-dim 256 --max-rows 120000 --cuda
python scripts/train_seq_ranker.py --valid-dir rank_folds_fast/fold0 --cache-dir feature_cache_fast/fold0 --model-dir fast_models/fold0/seq --seq-len 200 --hidden-dim 256 --max-rows 120000 --cuda

python scripts/search_ensemble.py --valid-dir rank_folds_fast/fold0 --cache-dir feature_cache_fast/fold0 --model-root fast_models/fold0 --score-dir fast_scores/fold0 --out fast_models/fold0/ensemble_weights.json --max-rows 120000
python scripts/run_ablation.py --cache-root feature_cache_fast --score-root fast_scores --weights-root fast_models --out fast_models/ablation_summary.csv --max-rows 120000
python scripts/predict_fold_ensemble.py --data-dir data_A --cache-root feature_cache_fast --weights-root fast_models --zip result_fast.zip
```

实际一键脚本会对 `fold0/fold1/fold2` 全部执行上述训练与融合。

只验证 TGNN 链路时可以跑小样本 smoke：

```bash
MAX_VALID=1000 TGNN_MAX_ROWS=1000 RUN_TGNN=1 USE_LGBM=0 RUN_MLP=0 RUN_SEQ=0 bash run_fast_ranker.sh
```

## 提交检查

```bash
python - <<'PY'
import csv, zipfile
with zipfile.ZipFile("result_fast.zip") as zf:
    for name in sorted(zf.namelist()):
        rows = 0
        with zf.open(name) as f:
            for line in f:
                vals = [float(x) for x in line.decode("utf-8").strip().split(",")]
                assert len(vals) == 100
                assert all(0.0 <= x <= 1.0 for x in vals)
                assert abs(sum(vals) - 1.0) < 1e-4
                rows += 1
        print(name, rows)
PY
```
