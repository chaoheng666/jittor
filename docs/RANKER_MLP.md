# Jittor MLP Ranker

本项目把任务建模成“时序图上的下一跳候选排序”。测试集已经为每个 `(src, time)` 给出 100 个候选 `dst`，所以当前模型不做全量节点召回，而是在这 100 个候选内部学习排序。

## 1. 当前是否可以直接训练

可以训练，但需要先满足：

- 已有数据目录：`data_A/dataset1/train.csv`、`data_A/dataset1/test.csv`、`data_A/dataset2/train.csv`、`data_A/dataset2/test.csv`
- 当前 Python 环境已安装 `jittor`
- 已用 `scripts/valid_builder.py` 生成本地验证样本

当前机器如果缺少 `data_A` 或 `jittor`，直接运行 `scripts/train_ranker.py` 会失败。Linux 服务器上推荐直接执行：

```bash
bash run_jittor_ranker.sh
```

该脚本会自动检查数据、下载数据、安装 Jittor、构造验证集、训练模型并生成提交 zip。

## 2. 手动训练流程

构造本地验证样本：

```powershell
python scripts/valid_builder.py --data-dir data_A --out-dir validation --max-valid 50000
```

训练 CPU 版：

```powershell
python scripts/train_ranker.py --valid-dir validation --model-dir models --epochs 8 --batch-size 256
```

训练 GPU 版：

```powershell
python scripts/train_ranker.py --valid-dir validation --model-dir models --epochs 8 --batch-size 256 --cuda
```

输出模型：

- `models/dataset1_jt_ranker.pkl`
- `models/dataset2_jt_ranker.pkl`

## 3. 当前建模思路

每一行训练样本包含：

```text
src, time, label, c1, c2, ..., c100
```

其中 `label` 是真实 `dst` 在 100 个候选里的位置。模型对每个候选 `(src, dst, time)` 构造特征并输出分数，目标是让真实 `dst` 的分数尽量排在最前。

整体流程：

```text
历史边 (src, dst, time)
        ↓
FeatureBuilder 构造候选特征
        ↓
RuleRankerV2 生成规则分
        ↓
MLP 学习候选打分
        ↓
fused_score = mlp_score + rule_score * fuse_rule
        ↓
按 fused_score 输出 100 个候选概率
```

## 4. 当前特征

候选特征主要包括：

- `src-dst` 关系：是否历史连接过、历史次数、近期次数、最近一次交互时间、预测时刻距离上次交互的时间差
- `src` 行为：总交互次数、最近 5/10/20 次是否出现候选、不同 `dst` 数、重复连接偏好
- `dst` 热度：全局出现次数、近期出现次数、热度趋势、最近出现时间、被多少不同 `src` 连接过
- 序列转移：`src` 上一次连接的 `dst` 到当前候选 `dst` 的转移次数
- 规则分：`RuleRankerV2` 根据历史频次、近期性、热度和转移关系得到的强规则分

## 5. 评估和提交

训练时日志会同时输出：

- `mlp_mrr`：只看 MLP 分数的 MRR
- `fused_mrr`：MLP 分数和规则分融合后的 MRR

默认保存 `fused_mrr` 最好的 epoch。

生成提交：

```powershell
python scripts/predict_ranker.py --data-dir data_A --model-dir models --out-dir submission_mlp --zip result_mlp.zip --mode fuse
```

只生成纯 MLP 结果：

```powershell
python scripts/predict_ranker.py --data-dir data_A --model-dir models --out-dir submission_mlp_only --zip result_mlp_only.zip --mode mlp
```

如果融合不稳定，可以小范围调整规则分融合权重：

```powershell
python scripts/train_ranker.py --valid-dir validation --model-dir models --epochs 8 --fuse-rule 0.5
python scripts/train_ranker.py --valid-dir validation --model-dir models --epochs 8 --fuse-rule 1.5
```
