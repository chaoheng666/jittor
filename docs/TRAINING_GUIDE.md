# 训练与调参指南

## 当前模型

当前方案不是纯 MLP 排序，而是规则主导的残差排序：

```text
final_score = rule_score * FUSE_RULE + tanh(mlp_score) * MLP_WEIGHT
```

原因是这个赛题重复边比例高，历史记忆和近期行为很强。规则分先保证基础排序，MLP 只做小幅修正，避免模型噪声把规则能排对的候选打乱。

训练日志里重点看：

```text
rule_mrr=... mlp_mrr=... fused_mrr=...
```

只有 `fused_mrr > rule_mrr` 时，预测阶段才会使用 MLP 残差；否则自动退回规则分。

## 一键训练

默认训练：

```bash
bash run_jittor_ranker.sh
```

常用推荐配置：

```bash
EPOCHS=8 BATCH_SIZE=256 HIDDEN_DIM=64 MLP_WEIGHT=0.2 bash run_jittor_ranker.sh
```

生成文件：

```text
result_mlp.zip
```

## 参数说明

### 数据与输出

`DATA_DIR`

训练和测试数据目录，默认 `data_A`。

`VALID_DIR`

本地验证样本目录，默认 `validation`。

`MODEL_DIR`

模型保存目录，默认 `models`。

`OUT_DIR`

预测 csv 输出目录，默认 `submission_mlp`。

`ZIP_PATH`

最终提交 zip 文件名，默认 `result_mlp.zip`。

### 验证样本构造

`VALID_RATIO`

只影响 `dataset1` 的时间切分比例，默认 `0.2`，表示最后 20% 时间边作为验证边。

建议尝试：

```text
0.1, 0.15, 0.2
```

`MAX_VALID`

最多构造多少条验证样本，默认 `0` 表示不限制。快速试验可以设小一点。

建议：

```text
快速试验：MAX_VALID=50000
正式训练：MAX_VALID=0
```

`HARD_RECENT_LIMIT`

每个 source 保留多少个最近交互 dst，用于 hard negative 和序列特征，默认 `20`。

建议尝试：

```text
20, 30, 50
```

`HARD_TRANSITION_LIMIT`

从 last_dst 的转移统计里取多少个高频 dst 作为 hard negative，默认 `50`。

建议尝试：

```text
20, 50, 100
```

`HARD_POPULAR_LIMIT`

统计多少个全局热门 dst，默认 `500`。

建议尝试：

```text
300, 500, 1000
```

`HARD_POPULAR_SAMPLE`

每条验证样本最多加入多少个热门 dst 候选，默认 `200`。因为最终只需要 100 个候选，这个值越大，热门负样本越容易排进候选池。

建议尝试：

```text
100, 200, 300
```

### 模型训练

`EPOCHS`

训练轮数，默认 `8`。

建议尝试：

```text
6, 8, 12
```

`BATCH_SIZE`

训练 batch size，默认 `256`。

建议尝试：

```text
128, 256, 512
```

显存不够就降到 `128`。

`HIDDEN_DIM`

MLP 隐层维度，默认 `64`。

建议尝试：

```text
32, 64, 128
```

当前特征维度不高，优先 `64`，不要一开始就加大模型。

`FUSE_RULE`

规则分权重，默认 `1.0`。越大越依赖规则。

建议尝试：

```text
0.8, 1.0, 1.2
```

`MLP_WEIGHT`

MLP 残差权重，默认 `0.2`。越大模型修正越强，也越容易覆盖规则。

建议优先调这个参数：

```text
0.0, 0.1, 0.2, 0.3, 0.5
```

其中 `0.0` 等价于只看规则分，可以作为强基线。

`EVAL_RATIO`

训练脚本内部从 `valid.csv` 后面切多少比例做模型选择，默认 `0.2`。

建议尝试：

```text
0.15, 0.2, 0.3
```

`USE_CUDA`

是否使用 GPU，默认 `1`。

没有 GPU 时：

```bash
USE_CUDA=0 bash run_jittor_ranker.sh
```

`USE_VENV`

是否自动创建虚拟环境，默认 `1`。

如果服务器已有可用环境：

```bash
USE_VENV=0 bash run_jittor_ranker.sh
```

## 推荐调参顺序

第一步，跑规则基线：

```bash
MLP_WEIGHT=0.0 ZIP_PATH=result_rule.zip bash run_jittor_ranker.sh
```

记录每个 dataset 的 `rule_mrr`。

第二步，只调残差强度：

```bash
MLP_WEIGHT=0.1 ZIP_PATH=result_w01.zip bash run_jittor_ranker.sh
MLP_WEIGHT=0.2 ZIP_PATH=result_w02.zip bash run_jittor_ranker.sh
MLP_WEIGHT=0.3 ZIP_PATH=result_w03.zip bash run_jittor_ranker.sh
```

优先选择 `fused_mrr - rule_mrr` 最大且两个 dataset 都不退化的配置。

第三步，调规则权重：

```bash
FUSE_RULE=0.8 MLP_WEIGHT=0.2 ZIP_PATH=result_f08_w02.zip bash run_jittor_ranker.sh
FUSE_RULE=1.0 MLP_WEIGHT=0.2 ZIP_PATH=result_f10_w02.zip bash run_jittor_ranker.sh
FUSE_RULE=1.2 MLP_WEIGHT=0.2 ZIP_PATH=result_f12_w02.zip bash run_jittor_ranker.sh
```

如果 `dataset1` 过度依赖重复边，通常更大的 `FUSE_RULE` 会更稳；如果 `dataset2` 热门候选多，可能需要稍低的 `FUSE_RULE` 给 MLP 修正空间。

第四步，调 hard negative：

```bash
HARD_RECENT_LIMIT=30 HARD_TRANSITION_LIMIT=50 HARD_POPULAR_SAMPLE=200 ZIP_PATH=result_hard_a.zip bash run_jittor_ranker.sh
HARD_RECENT_LIMIT=50 HARD_TRANSITION_LIMIT=100 HARD_POPULAR_SAMPLE=100 ZIP_PATH=result_hard_b.zip bash run_jittor_ranker.sh
HARD_RECENT_LIMIT=20 HARD_TRANSITION_LIMIT=20 HARD_POPULAR_SAMPLE=300 ZIP_PATH=result_hard_c.zip bash run_jittor_ranker.sh
```

hard negative 越难，本地验证越接近真实测试，但训练也可能更保守。看 `fused_mrr` 是否稳定超过 `rule_mrr`。

第五步，最后再调模型大小和训练轮数：

```bash
HIDDEN_DIM=32 EPOCHS=8 ZIP_PATH=result_dim32.zip bash run_jittor_ranker.sh
HIDDEN_DIM=64 EPOCHS=12 ZIP_PATH=result_dim64_e12.zip bash run_jittor_ranker.sh
HIDDEN_DIM=128 EPOCHS=8 ZIP_PATH=result_dim128.zip bash run_jittor_ranker.sh
```

如果 `mlp_mrr` 高但 `fused_mrr` 不高，说明 MLP 和规则冲突，优先降低 `MLP_WEIGHT`。

如果 `rule_mrr` 高但 `fused_mrr` 低，说明残差在破坏规则，使用 `MLP_WEIGHT=0.0` 或 `0.1`。

如果 `rule_mrr` 和 `fused_mrr` 都低，优先调 hard negative 和规则权重，不要先加大模型。

## 实验记录模板

建议每次记录：

```text
实验名:
命令:
dataset1 rule_mrr:
dataset1 fused_mrr:
dataset2 rule_mrr:
dataset2 fused_mrr:
线上分数:
备注:
```

示例：

```text
实验名: f10_w02_default
命令: FUSE_RULE=1.0 MLP_WEIGHT=0.2 ZIP_PATH=result_f10_w02.zip bash run_jittor_ranker.sh
dataset1 rule_mrr:
dataset1 fused_mrr:
dataset2 rule_mrr:
dataset2 fused_mrr:
线上分数:
备注:
```

## 提交前检查

确认 zip 结构：

```text
result.zip
  dataset1.csv
  dataset2.csv
```

确认每行 100 个概率，且与 `test.csv` 候选顺序一致。

快速检查：

```bash
python - <<'PY'
import csv, zipfile
path = "result_mlp.zip"
with zipfile.ZipFile(path) as zf:
    for name in zf.namelist():
        with zf.open(name) as f:
            row = next(csv.reader(line.decode("utf-8") for line in f))
            print(name, len(row), sum(float(x) for x in row))
PY
```

