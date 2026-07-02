# 真实未来边强度预测方案说明

## 1. 任务重新定义

这个任务的测试集给定了每一行的 `(src, time)` 和 100 个候选 `dst`。提交时只需要给这 100 个候选输出概率。表面上看，这是一个候选内排序问题；但真正应该学习的对象不是“某个构造出来的 100 候选验证集”，而是动态图中一条边在未来发生的强度。

当前方案把核心目标定义为：

```text
score(src, dst, time) ≈ P(edge src -> dst happens at time | history before time)
```

也就是说，模型应该回答的问题是：

```text
在当前历史图下，src 在这个时间更可能连向哪个 dst？
```

测试时官方给出的 100 个候选只是最后的约束。我们不会从全图召回节点，而是在这 100 个候选上分别计算边强度，然后按强度排序并 softmax。

## 2. 为什么不再构造伪验证集

旧方案会构造一个尽量贴近 test 行形态的验证集，比如复用真实测试候选模板，再把历史中的未来正边插入进去。这种方式可以作为 sanity check，但它有根本缺陷：

- 候选集仍然是人为构造的，不是真实隐藏答案的生成机制。
- 负样本分布再怎么模拟，也不能保证与线上候选完全一致。
- 如果用这个验证集做模型选择，模型可能会拟合“构造方法”，而不是拟合真实边发生规律。
- 线下 MRR 的提升可能来自候选构造偏差，而不一定代表真实未来边预测能力更强。

因此现在删除这条路线，不再运行 `valid_builder`、不再依赖 `valid.csv`、不再用构造候选 MRR 做集成选择。

新的验证和选择依据是：真实未来边 pairwise ranking。模型训练时只把真实发生的未来边作为正样本，再采样没有发生的候选边作为负样本，优化真实未来边分数高于负边分数。

这并不是说负样本采样可以完全消失。历史数据只告诉我们哪些边发生了，不会显式列出所有未发生边。负采样仍然必要，但它只用于训练“真实边高于未发生边”的排序边界，不再用于构造假的 100 候选验证行。

## 3. 整体流程

唯一入口是：

```bash
bash run_best.sh
```

完整流程为：

```text
原始 train/test
  -> 从真实时序边训练 edge-intensity MLP sweep
  -> 用未来边 pairwise accuracy 选择每个数据集最优模型
  -> 如果 MLP 没有超过规则基线，回退 rule-only
  -> 对官方 test 的 100 candidates 计算边强度
  -> row z-score
  -> softmax
  -> 写出 result_best.zip
```

默认输出：

```text
result_best.zip
  dataset1.csv
  dataset2.csv
```

## 4. 核心信号设计

边强度由四类信号共同描述。

### 4.1 Pair Hazard

Pair hazard 估计历史上出现过的 `(src, dst)` 是否容易再次发生。

这类信号尤其适合 `dataset1` 这种重复边较强的场景。重要特征包括：

- `(src, dst)` 是否出现过。
- pair 历史出现次数。
- pair 近期出现次数。
- pair 最近一次出现距离当前时间的 recency。
- pair 时间间隔。
- `dst` 是否出现在 `src` 最近 5、10、20、50 次交互中。
- `src` 是否刚刚访问过同一个 `dst`。
- `src` 的重复访问比例。

这类特征解决的问题是：如果某个源点有明显重复连接习惯，模型应该把历史高频、近期刚发生过、时间间隔合理的 pair 打高分。

### 4.2 Source Sequence Transition

Source sequence transition 估计 `src` 当前状态下更可能转向哪个 `dst`。

它关注 `src` 最近访问序列，而不是只记住固定 pair。重要特征包括：

- 上一个交互目标到当前候选的转移次数。
- 近期多个历史目标到当前候选的转移累积分数。
- 滑窗共现分数。
- 转移命中次数。
- 共现命中次数。
- 反向转移和反向共现信号。

这类信号更适合新链接或二部图场景。即使 `(src, dst)` 过去没有出现过，只要 `src` 最近访问过的节点经常转向该 `dst`，候选也应该被提高。

### 4.3 Destination Prior

Destination prior 估计某个 `dst` 在当前时间是否整体容易被连到。

重要特征包括：

- `dst` 全局热度。
- `dst` 近期热度。
- 更近时间窗口中的热度。
- `dst` 热度趋势。
- `dst` 最近出现时间。
- `dst` 被多少不同 `src` 连过。
- `dst` 是否是训练中未出现过的 cold node。

这类信号解决热门目标、新鲜目标和 cold candidate 的问题。对 `dataset2`，目的节点热度、趋势和新鲜度通常比固定 pair 记忆更重要。

### 4.4 Neural Residual

规则分数覆盖了大量稳定时序规律，但手工权重无法穷尽所有非线性组合。因此当前方案保留 edge MLP 作为残差模型。

MLP 的输入是完整候选特征向量，其中包含规则分数 `rule_score`。MLP 不单独接管最终排序，而是作为规则强度上的补偿项：

```text
edge_score = rule_score * fuse_rule + tanh(mlp_score) * gamma
```

这样设计有三个好处：

- 规则模型仍然是稳定主干。
- 神经网络只学习规则遗漏的残差信号。
- `gamma` 限制残差幅度，降低过拟合风险。

## 5. 训练流程

训练脚本是：

```text
scripts/train_edge_ranker.py
```

每个数据集的训练步骤如下。

### 5.1 时间切分

读取原始 `train.csv`，按 `time` 排序。前段作为历史图，后段作为未来监督边。

```text
history_edges      = earlier edges
supervision_edges  = later real edges
```

历史图只提供过去信息；未来边只作为正样本监督。这保证训练目标是预测真实未来边，而不是拟合构造候选。

### 5.2 正样本

正样本全部来自真实未来边：

```text
(src, true_dst, time)
```

这些边是真实发生的交互，是模型应该打高分的对象。

### 5.3 Hard Negatives

负样本是没有作为该条未来正边发生的候选 `dst`。为了让训练有实际区分能力，优先采 hard negatives：

- `src` 最近交互过的 `dst`。
- 上一个 `dst` 的高频转移目标。
- 上一个 `dst` 的滑窗共现目标。
- 全局热门 `dst`。
- 随机历史 `dst` 补齐。

这些负样本比纯随机节点更难，也更接近测试候选中的强干扰项。

### 5.4 Pairwise Ranking Objective

对每条正边和多个负边构造 pair：

```text
positive = (src, true_dst, time)
negative = (src, sampled_dst, time)
```

模型优化目标：

```text
score(positive) > score(negative)
```

训练中的融合差值为：

```text
fused_diff =
    (rule_pos - rule_neg) * fuse_rule
  + (mlp_pos - mlp_neg) * gamma
```

如果 `fused_diff > 0`，说明真实未来边排在负边前面。

### 5.5 Future-Edge Metrics

训练 metadata 中保存真实未来边评估指标：

- `future_rule_acc`：规则分数让真实未来边高于负边的比例。
- `future_residual_acc`：MLP 残差单独让真实未来边高于负边的比例。
- `future_fused_acc`：规则 + MLP 残差融合后真实未来边高于负边的比例。
- `future_eval_loss`：未来边 pairwise loss。
- `train_pairs`：训练 pair 数。
- `eval_pairs`：评估 pair 数。

这些指标直接来自真实未来边，不依赖任何构造的 100 候选验证行。

## 6. 模型选择

模型选择脚本是：

```text
scripts/select_edge_model.py
```

`run_best.sh` 会训练多个 edge MLP 配置：

```bash
EDGE_HIDDEN_DIMS=64,128,256
EDGE_GAMMAS=0.05,0.08,0.15,0.25,0.35
EDGE_SEEDS=2026,2027
```

训练完成后，选择脚本会扫描所有：

```text
{dataset}_edge_ranker.pkl
```

对每个数据集选择 `future_fused_acc` 最高的模型。

如果最佳 MLP 相对规则基线的提升小于：

```bash
MIN_FUTURE_GAIN=0.0005
```

则该数据集回退到 rule-only。这是一个保守策略：如果神经残差不能在真实未来边评估中稳定超过规则模型，就不让它影响提交结果。

选择结果写入：

```text
competition_models_best/edge_intensity_config.json
```

配置示例：

```json
{
  "mode": "edge_intensity",
  "datasets": {
    "dataset1": {
      "selection_metric": "future_fused_acc",
      "components": [
        {
          "name": "edge_h128_g0p15/seed_2026",
          "type": "edge_mlp",
          "path": "competition_models_best/edge_h128_g0p15/seed_2026/dataset1_edge_ranker.pkl",
          "weight": 1.0
        }
      ]
    }
  }
}
```

## 7. 最终预测

预测脚本是：

```text
scripts/predict_edge_intensity.py
```

预测时对每个测试行：

```text
src, time, c1, c2, ..., c100
```

分别计算：

```text
score(src, c1, time)
score(src, c2, time)
...
score(src, c100, time)
```

如果数据集选择 rule-only，则直接使用规则边强度。

如果数据集选择 edge MLP，则使用：

```text
rule_score * fuse_rule + tanh(mlp_score) * gamma
```

之后对每行 100 个候选分数做 row z-score，再 softmax：

```text
prob_i = exp(score_i) / sum_j exp(score_j)
```

这样输出满足提交格式：每行 100 个概率，概率和约等于 1。

## 8. 当前文件职责

### `run_best.sh`

唯一入口。负责数据检查、Jittor 环境检查、并行训练 edge MLP sweep、选择最优模型、预测并打包。

### `scripts/train_edge_ranker.py`

训练真实未来边强度 MLP。使用真实未来边作为正样本，hard negatives 作为负样本，优化 pairwise ranking。

### `scripts/select_edge_model.py`

根据 `future_fused_acc` 选择每个数据集最优模型。若 MLP 没有超过规则基线，则回退 rule-only。

### `scripts/predict_edge_intensity.py`

读取模型选择配置，对官方测试候选计算边强度，输出提交 zip。

### `src/feature_builder.py`

构造 pair hazard、source sequence transition、destination prior 等时序边特征。

### `src/rule_ranker_v2.py`

规则边强度 scorer。`dataset1` 使用更偏重复边的权重，`dataset2` 使用更偏新链接和目的节点先验的权重。

### `src/jt_ranker.py`

定义 MLP、特征顺序、标准化、模型保存和加载。

### `src/edge_scoring.py`

封装最终预测需要的通用打分逻辑：训练边读取、测试行读取、规则打分、MLP 打分、row z-score 和 softmax。

### `src/data_loader.py`

封装数据集目录发现、训练边读取和测试行读取。

## 9. 已删除的旧路径

当前代码不再包含以下旧方案：

- 构造伪 100 候选验证集。
- 基于构造验证集的 MRR 集成搜索。
- CRAFT 分数缓存路线。
- 旧的 `predict_luxury_ensemble.py`。

这些路径被删除，是为了避免系统继续围绕人为构造的验证候选优化。现在的训练、选择和预测都围绕真实未来边强度展开。

## 10. 关键参数

### MLP 搜索空间

```bash
EDGE_HIDDEN_DIMS=64,128,256
EDGE_GAMMAS=0.05,0.08,0.15,0.25,0.35
EDGE_SEEDS=2026,2027
```

### 训练规模

```bash
EDGE_EPOCHS=10
EDGE_NEGATIVES=10
EDGE_SAMPLE_EDGES=250000
```

`EDGE_SAMPLE_EDGES=0` 表示使用全部监督边。默认限制样本数是为了控制训练时间。

### 选择阈值

```bash
MIN_FUTURE_GAIN=0.0005
```

只有当最优 MLP 的 `future_fused_acc` 至少超过规则基线这个阈值，才选择 MLP；否则 rule-only。

### 快速调试

```bash
USE_CUDA=0 USE_VENV=0 MAX_PARALLEL=1 GPU_COUNT=1 \
EDGE_HIDDEN_DIMS=8 EDGE_GAMMAS=0.05 EDGE_SEEDS=2026 \
EDGE_EPOCHS=1 EDGE_NEGATIVES=2 EDGE_SAMPLE_EDGES=2000 \
bash run_best.sh
```

## 11. 方案优点

第一，训练目标更直接。模型优化真实未来边高于负边，而不是优化构造候选集上的表现。

第二，保留了强规则先验。重复边、转移、共现、热度、趋势、cold 惩罚这些稳定信号仍然是主干。

第三，神经网络风险受控。MLP 只是残差项，且需要在真实未来边评估中超过规则模型才会被选中。

第四，测试阶段自然适配官方候选。模型本身估计边强度，官方 100 candidates 只是最后的排序集合。

第五，代码路径更简单。删除了伪验证构造、CRAFT 缓存和集成搜索后，正式流程只剩训练、选择、预测三个核心步骤。

## 12. 注意事项

负样本采样仍然会影响训练质量。它无法完全代表所有未发生边，但当前 hard negative 策略优先选择近期、转移、共现和热门节点，比随机负样本更接近真实困难候选。

`dataset1` 和 `dataset2` 的图结构差异很大。规则权重仍然按数据集区分，不能假设所有数据集共享同一套边生成机制。

如果机器资源有限，可以先用快速调试配置检查流程，再跑默认完整配置生成最终提交。

## 13. 总结

当前方案从“拟合构造验证候选”改成“预测真实未来边强度”。训练用真实未来边作为正样本，hard negatives 提供排序边界；规则模型提供稳定时序先验，MLP 学习受控残差；模型选择只看真实未来边 pairwise accuracy；最终只在官方给出的 100 个候选上比较边强度并输出概率。
