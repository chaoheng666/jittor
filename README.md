# Jittor Temporal Graph Re-Ranker

这是赛道一“基于图学习的动态推荐任务”的候选内精排方案。测试集已经给出每个 `(src, time)` 的 100 个候选 `dst`，所以本仓库不做全量召回，专注把这 100 个候选排准。

## 关键改动

- `dataset1` 按高重复边场景处理：强化历史 pair、最近交互、pair recency、source 最近序列。
- `dataset2` 按二部图新链接场景处理：历史 pair 会降权，训练中未出现的 cold dst 会强惩罚，强化 dst 新近热度、item-to-item 转移和滑窗共现。
- 验证集默认使用 `test-prior` 候选构造：从测试候选中估计 cold dst 比例，让本地验证更接近线上候选分布。
- 融合不再固定从规则模型开始，而是在验证集上选择单体最强组件作为起点，再贪心加入 MLP、序列模型和可选 CRAFT 动态图模型。

## 一键训练并生成提交

Linux 服务器运行：

```bash
bash run_luxury_ranker.sh
```

默认输出：

```text
result.zip
  dataset1.csv
  dataset2.csv
```

如果 `data_A/dataset1` 和 `data_A/dataset2` 不存在，脚本会自动从题目链接下载并解压。脚本会自动安装 `jittor`；如果环境里已有 `jittor_geometric`，会额外训练 CRAFT 动态图模型，否则自动跳过 CRAFT，不影响 MLP/序列融合。

## 推荐正式配置

8 卡 V100 直接用默认配置即可：

```bash
GPU_COUNT=8 MAX_PARALLEL=8 USE_CUDA=1 bash run_luxury_ranker.sh
```

也可以直接跑预设版本，按优先级从 `v1` 开始：

```bash
bash run_luxury_ranker_v1.sh
bash run_luxury_ranker_v2.sh
bash run_luxury_ranker_v3.sh
bash run_luxury_ranker_v4.sh
bash run_luxury_ranker_v5.sh
```

版本用途：

| 脚本 | 输出 | 用途 |
| --- | --- | --- |
| `run_luxury_ranker_v1.sh` | `result_v1.zip` | 最推荐，平衡重复边和新链接，全量 ensemble。 |
| `run_luxury_ranker_v2.sh` | `result_v2_newlink.zip` | 偏二部图新链接，转移/热度/长序列更强。 |
| `run_luxury_ranker_v3.sh` | `result_v3_repeat.zip` | 偏重复边保守，规则权重更高、残差更小。 |
| `run_luxury_ranker_v4.sh` | `result_v4_fast.zip` | 快速稳健版，不跑 CRAFT，验证集默认 15 万。 |
| `run_luxury_ranker_v5.sh` | `result_v5_craft.zip` | 动态图偏重，扩大 CRAFT sweep。 |

如果服务器内存紧张，把本地验证样本截断到 12 万到 18 万：

```bash
MAX_VALID=150000 GPU_COUNT=8 MAX_PARALLEL=8 bash run_luxury_ranker.sh
```

如果已经安装官方 baseline 的 JittorGeometric 依赖，保留：

```bash
RUN_CRAFT=1 bash run_luxury_ranker.sh
```

如果没有这个依赖或 CRAFT 不稳定：

```bash
RUN_CRAFT=0 bash run_luxury_ranker.sh
```

## 常用参数

- `MAX_VALID=0`：使用全部验证样本；内存不够时设为 `100000`、`150000`。
- `VALID_MODE=test-prior`：默认，按测试候选 cold 比例构造验证集。
- `MAX_COLD_POOL=2000000`：流式保留的 cold 候选池上限，B 榜内存紧张时可调小。
- `MLP_HIDDEN_DIMS=64,128,256`、`MLP_WEIGHTS=0.1,0.2,0.35`：并行训练多组残差 MLP。
- `SEQ_LENS=30,50,100`、`SEQ_GAMMAS=0.1,0.2,0.35`：并行训练 source 历史序列残差模型。
- `RUN_CRAFT=1`：有 JittorGeometric 时加入动态图 CRAFT 分数缓存。

## 手动流程

```bash
python scripts/valid_builder.py --data-dir data_A --out-dir validation_competition_test-prior --valid-mode test-prior --max-valid 150000
python scripts/train_ranker.py --valid-dir validation_competition_test-prior --model-dir competition_models/mlp_h128_w0p2 --hidden-dim 128 --mlp-weight 0.2 --seed-list 2026,2027 --cuda
python scripts/train_seq_ranker.py --valid-dir validation_competition_test-prior --model-dir competition_models/seq_l100_g0p2 --seq-len 100 --gamma 0.2 --hidden-dim 192 --seed-list 2026,2027 --cuda
python scripts/search_ensemble.py --valid-dir validation_competition_test-prior --model-root competition_models --score-dir competition_scores --out competition_models/ensemble_weights.json
python scripts/predict_luxury_ensemble.py --data-dir data_A --weights competition_models/ensemble_weights.json --zip result.zip
```

提交前快速检查：

```bash
python - <<'PY'
import csv, zipfile
with zipfile.ZipFile("result.zip") as zf:
    for name in sorted(zf.namelist()):
        with zf.open(name) as f:
            row = next(csv.reader(line.decode("utf-8") for line in f))
        vals = [float(x) for x in row]
        print(name, len(vals), min(vals), max(vals), sum(vals))
PY
```
