# 豪华版训练指南

## 目标

豪华版使用多模型融合冲分：

```text
规则分 + 残差 MLP + 序列残差模型 + CRAFT 分数缓存
```

最终不会盲目平均。`scripts/search_ensemble.py` 会在验证集上按 MRR 贪心搜索权重，加入后降分的模型权重自动为 0。

## 一键训练

服务器推荐命令：

```bash
bash run_luxury_ranker.sh
```

快速冒烟：

```bash
MAX_VALID=5000 RUN_CRAFT=0 MLP_HIDDEN_DIMS=64 MLP_WEIGHTS=0.2 SEQ_LENS=30 SEQ_GAMMAS=0.2 bash run_luxury_ranker.sh
```

正式冲分：

```bash
MAX_VALID=0 RUN_CRAFT=1 bash run_luxury_ranker.sh
```

输出：

```text
result_luxury.zip
```

## 服务器资源使用

默认按 8 张 GPU 并行跑不同模型和 seed，不做 DDP。

```bash
GPU_COUNT=8 MAX_PARALLEL=8 bash run_luxury_ranker.sh
```

如果显存紧张：

```bash
MAX_PARALLEL=4 BATCH_SIZE=128 CRAFT_BATCH_SIZE=100 bash run_luxury_ranker.sh
```

## 关键参数

### 验证构造

`VALID_ROOT`

豪华验证目录前缀，默认 `validation_luxury`。脚本会生成：

```text
validation_luxury_mixed
validation_luxury_recent-heavy
validation_luxury_popular-heavy
validation_luxury_transition-heavy
```

当前训练和融合默认使用 `mixed`，其他目录用于人工复核泛化稳定性。

`MAX_VALID`

验证样本数上限，默认 `0` 不限制。

```text
冒烟: 5000
快速调参: 50000
正式: 0
```

`HARD_RECENT_LIMIT`

source 最近交互候选数量，默认 `50`。

建议：

```text
20, 50, 80
```

`HARD_TRANSITION_LIMIT`

last_dst 转移候选数量，默认 `100`。

建议：

```text
50, 100, 200
```

`HARD_POPULAR_SAMPLE`

热门 dst hard negative 数量，默认 `300`。

建议：

```text
100, 200, 300
```

### 残差 MLP

`MLP_SEEDS`

MLP 多 seed，默认 `2026,2027,2028`。

`MLP_HIDDEN_DIMS`

MLP 隐层维度，默认 `64,128,256`。

`MLP_WEIGHTS`

MLP 残差权重，默认 `0.1,0.2,0.3`。

建议先看融合器日志。如果大部分 MLP 都被 drop，说明残差过强或验证构造不匹配，优先保留 `0.1`。

### 序列残差模型

`SEQ_SEEDS`

序列模型多 seed，默认 `2026,2027`。

`SEQ_LENS`

source 历史序列长度，默认 `30,50`。

`SEQ_GAMMAS`

序列残差权重，默认 `0.1,0.2,0.3`。

`SEQ_HIDDEN_DIM`

序列模型 hidden 维度，默认 `128`。V100 16G 上先不要超过 256。

### CRAFT

`RUN_CRAFT`

是否训练 CRAFT，默认 `1`。

如果没有安装 `jittor_geometric`：

```bash
RUN_CRAFT=0 bash run_luxury_ranker.sh
```

`CRAFT_NEIGHBORS`

CRAFT 历史邻居数量，默认 `30,50`。

`CRAFT_HIDDEN_SIZES`

CRAFT hidden size，默认 `64,128`。

`CRAFT_EPOCHS`

CRAFT 训练轮数，默认 `6`。正式冲分可以试 `10`。

## 推荐实验顺序

第一步，冒烟：

```bash
MAX_VALID=5000 RUN_CRAFT=0 MLP_HIDDEN_DIMS=64 MLP_WEIGHTS=0.2 SEQ_LENS=30 SEQ_GAMMAS=0.2 bash run_luxury_ranker.sh
```

确认能生成 `result_luxury.zip`。

第二步，不跑 CRAFT，先看 MLP + Seq：

```bash
RUN_CRAFT=0 MAX_VALID=50000 bash run_luxury_ranker.sh
```

重点看 `search_ensemble.py` 日志：

```text
component=... mrr=...
add ... weight=... mrr=...
drop ...
```

第三步，开启 CRAFT：

```bash
RUN_CRAFT=1 MAX_VALID=50000 bash run_luxury_ranker.sh
```

如果 CRAFT 经常被 drop，就降低 CRAFT 训练组合：

```bash
CRAFT_NEIGHBORS=30 CRAFT_HIDDEN_SIZES=64 RUN_CRAFT=1 bash run_luxury_ranker.sh
```

第四步，正式全量：

```bash
MAX_VALID=0 RUN_CRAFT=1 MAX_PARALLEL=8 bash run_luxury_ranker.sh
```

## 单独训练命令

训练残差 MLP：

```bash
python scripts/train_ranker.py --valid-dir validation_luxury_mixed --model-dir luxury_models/mlp_h128_w02 --hidden-dim 128 --mlp-weight 0.2 --seed-list 2026,2027 --cuda
```

训练序列模型：

```bash
python scripts/train_seq_ranker.py --valid-dir validation_luxury_mixed --model-dir luxury_models/seq_l50_g02 --seq-len 50 --gamma 0.2 --seed-list 2026,2027 --cuda
```

训练 CRAFT：

```bash
python scripts/train_craft_ranker.py --data-dir data_A --valid-dir validation_luxury_mixed --model-dir luxury_models/craft_n30_h64 --score-dir luxury_scores --num-neighbors 30 --hidden-size 64 --cuda
```

搜索融合：

```bash
python scripts/search_ensemble.py --valid-dir validation_luxury_mixed --model-root luxury_models --score-dir luxury_scores --out luxury_models/ensemble_weights.json
```

生成提交：

```bash
python scripts/predict_luxury_ensemble.py --data-dir data_A --weights luxury_models/ensemble_weights.json --zip result_luxury.zip
```

## 结果检查

```bash
python - <<'PY'
import csv, zipfile
path = "result_luxury.zip"
with zipfile.ZipFile(path) as zf:
    for name in zf.namelist():
        with zf.open(name) as f:
            row = next(csv.reader(line.decode("utf-8") for line in f))
            vals = [float(x) for x in row]
            print(name, len(vals), min(vals), max(vals), sum(vals))
PY
```

要求：

```text
每行 100 个值
每个值在 [0,1]
每行概率和接近 1
```

## 实验记录模板

```text
实验名:
命令:
dataset1 rule_mrr:
dataset1 ensemble_mrr:
dataset2 rule_mrr:
dataset2 ensemble_mrr:
线上分数:
被 add 的模型:
被 drop 的模型:
备注:
```

