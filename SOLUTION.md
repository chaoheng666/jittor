# 动态推荐候选重排序方案说明

## 1. 任务理解与问题分析

本项目解决的是动态图推荐中的候选内精排问题。测试数据已经为每个查询行给定了一个 `(src, time)` 和 100 个候选 `dst`，提交文件只需要对这 100 个候选分别输出概率。因此，本方案不做全量召回，也不试图从所有节点中搜索目标节点，而是专注于：在给定的 100 个候选内部，把真实目标排到尽可能靠前的位置。

评价指标本质上围绕排序质量展开，代码中使用 `MRR` 作为本地验证指标。对于每一行，如果真实候选排第 1，贡献为 `1.0`；排第 2，贡献为 `0.5`；排第 k，贡献为 `1/k`。这意味着模型最重要的目标不是让所有候选概率都绝对校准，而是让正样本相对其他 99 个候选的排序尽量靠前，尤其要尽量争取 Top1。

从数据形态看，这个问题有几个关键难点。

第一，训练数据是历史交互边，测试数据是候选行。训练集中天然只有正边，没有测试时那种固定 100 候选的监督格式。如果直接随机构造负样本训练，负样本分布很可能和真实测试候选分布不一致，导致本地验证看起来很好，线上表现却不稳定。

第二，不同数据集的行为模式差异明显。`dataset1` 更像高重复边场景，很多 `src -> dst` 会在历史中重复出现，因此历史 pair 记忆、最近交互、pair recency 等信号非常强。`dataset2` 更像二部图或新链接预测场景，历史 pair 重复不一定可靠，反而目的节点热度、近期趋势、item-to-item 转移、滑窗共现等信号更关键。

第三，候选中可能混有冷门或训练中未出现的 `dst`。这类候选在真实测试分布中可能大量存在。如果验证集没有模拟这种 cold candidate 分布，模型容易错误高估训练内热门模式或错误处理未见节点。

第四，神经模型容易过拟合本地负采样。候选重排序里，手工时序规则通常很稳，但神经模型可以补充非线性关系。如果完全让神经网络接管排序，它可能在验证构造上学到伪规律，造成线上波动。因此当前方案采用“规则强锚点 + 神经残差补偿”的结构，而不是端到端盲训一个大模型。

## 2. 总体解决方案

当前仓库只保留一个正式入口：

```bash
bash run_best.sh
```

该脚本执行完整训练与预测流程，最终输出：

```text
result_best.zip
  dataset1.csv
  dataset2.csv
```

整体方案可以概括为：

```text
原始 train/test
  -> 构造贴近 test 行形态的验证集
  -> 训练强规则 ranker 作为稳定基线
  -> 从原始时序边训练 edge-level residual MLP
  -> 可选训练 CRAFT 动态图模型并缓存分数
  -> 在验证集上搜索保守集成权重
  -> 对 test 的 100 候选输出 softmax 概率
  -> 打包 result_best.zip
```

方案的核心不是用单个复杂模型替代所有逻辑，而是把不同信号分工清楚：

- 规则模型负责稳定、可解释、泛化性强的排序主干。
- Edge residual MLP 负责学习规则没有覆盖到的非线性残差信号。
- CRAFT 动态图模型负责补充基于邻居序列和时间演化的表示能力。
- 集成搜索负责只接纳真正能提升验证 MRR 的组件，并限制排序漂移。

最终预测时，每个组件都会产生一个 `num_rows x 100` 的分数矩阵。所有分数按行标准化后加权相加，再对每一行做 softmax，得到提交所需的 100 个概率。

## 3. 核心训练流程

### 3.1 数据准备

入口脚本首先检查 `data_A` 是否存在，并确认其下有类似如下结构：

```text
data_A/
  dataset1/
    train.csv
    test.csv
  dataset2/
    train.csv
    test.csv
```

如果数据缺失，`run_best.sh` 会尝试从脚本中配置的 `DATA_URL` 下载并解压。

随后脚本会检查 Python 环境。如果启用默认的 `USE_VENV=1`，会在当前目录创建或复用 `.venv_jittor`，并确保安装 `jittor`。如果环境里有 `jittor_geometric`，则会额外启用 CRAFT 动态图模型；否则自动跳过 CRAFT，不影响规则模型和 edge MLP 的训练。

### 3.2 验证集构造

验证集由 `scripts/valid_builder.py` 构造。默认模式是：

```bash
VALID_MODE=test-row
```

该模式的核心思想是：验证集的 100 候选结构要尽量接近真实测试集，而不是简单随机负采样。

具体做法如下：

1. 从原始 `train.csv` 中划分训练边和验证正边。
2. 如果 `train.csv` 中存在 `split` 列，则优先使用该列划分。
3. 如果没有可用 `split` 列，则按时间排序，使用后段边作为验证正样本。
4. 对每条验证正边 `(src, dst, time)`，从真实 `test.csv` 的候选行中采样一个 100 候选模板。
5. 如果正样本已经在模板中，直接使用其位置作为 label。
6. 如果正样本不在模板中，则把模板中的一个候选替换为正样本，且尽量保持 cold/known 属性一致。

这样做的好处是，本地验证不只是“随机挑 99 个负样本”，而是让候选分布、冷启动比例、热门候选干扰更接近真实测试。这对候选内重排序非常关键，因为模型最终面对的不是全量节点，而是官方给定的 100 个候选。

输出结构大致为：

```text
validation_best_test-row/
  dataset1/
    train.csv
    valid.csv
  dataset2/
    train.csv
    valid.csv
```

其中 `valid.csv` 的字段为：

```text
src,time,label,c1,c2,...,c100
```

`label` 表示真实目标在 `c1` 到 `c100` 中的位置。

### 3.3 规则模型训练与打分

规则模型由 `src/rule_ranker_v2.py` 实现，底层特征由 `src/feature_builder.py` 构造。

规则模型不需要梯度训练，而是基于历史边统计时序特征，并对每个候选打一个可解释分数。主要特征包括：

- 历史 pair 是否出现过。
- `(src, dst)` pair 出现次数。
- pair 最近出现次数。
- pair 最近一次出现距离当前时间的 recency。
- `src` 最近 5、10、20、50 次交互中是否出现过候选 `dst`。
- 候选 `dst` 的全局热度。
- 候选 `dst` 的近期热度。
- 候选 `dst` 的热度趋势。
- 候选 `dst` 最近一次出现时间。
- `src` 活跃度。
- `src` 历史不同 `dst` 数量。
- `src` 重复访问比例。
- 上一个交互目标到当前候选的转移统计。
- 近期序列中多个历史目标到当前候选的转移统计。
- 滑窗共现统计。
- 反向转移和反向共现统计。
- 候选是否是训练中未出现过的 cold dst。

规则权重分为两套：

1. `REPEAT_EDGE_WEIGHTS`：适合重复边场景，主要用于 `dataset1`。
2. `NEW_LINK_WEIGHTS`：适合新链接/二部图场景，主要用于 `dataset2`。

`dataset1` 中，历史重复 pair、最近交互、pair recency 的权重更高，因为真实目标往往和过去行为强相关。

`dataset2` 中，历史 pair 会被降权，目的节点热度、近期趋势、转移、共现等更重要，因为它更像预测一个新的 `src -> dst` 链接。

规则模型是整个系统的稳定基线。后续 MLP 和 CRAFT 都不是直接替代规则模型，而是在规则模型基础上做补充。

### 3.4 Edge Residual MLP 训练

Edge residual MLP 由 `scripts/train_edge_ranker.py` 训练，模型结构在 `src/jt_ranker.py` 中。

旧方案曾经存在直接对 100 候选行训练 MLP 或序列模型的路径，但当前代码已经移除。现在 MLP 只从原始时序边学习，训练目标更加贴近动态图边预测本身。

训练过程如下：

1. 读取每个数据集的原始 `train.csv`。
2. 按时间排序。
3. 用前 `history_ratio` 部分作为历史上下文。
4. 用后段边作为监督正样本。
5. 对每条正边 `(src, positive_dst, time)`，采样多个 hard negatives。
6. 对正负候选分别构造同一套候选特征。
7. 训练 MLP，使融合后的正样本分数高于负样本。

负样本不是纯随机采样，而是优先从更难的候选中选：

- 当前 `src` 最近交互过的 `dst`。
- 上一个 `dst` 之后常出现的转移目标。
- 与上一个 `dst` 在滑窗中共现的目标。
- 全局热门 `dst`。
- 如果 hard negatives 不足，再从历史出现过的 `dst` 中随机补齐。

这种 hard negative 策略更接近真实 test 候选，因为官方候选里通常也会包含热门、近期、相似或容易混淆的候选。

MLP 输入不是原始 node id embedding，而是一组手工统计特征加规则分数：

```text
[bias, pair features, src recent features, dst popularity features,
 transition/cooc features, rule_score]
```

训练目标不是简单让 MLP 单独判断正负，而是让它学习规则分数之外的残差。核心公式是：

```text
fused_diff = rule_diff * fuse_rule + mlp_diff * gamma
```

其中：

- `rule_diff` 是正样本规则分数减负样本规则分数。
- `mlp_diff` 是正样本 MLP 输出减负样本 MLP 输出。
- `fuse_rule` 控制规则主干强度。
- `gamma` 控制神经残差幅度。

这意味着即使 MLP 训练出一些不稳定信号，也会被 `gamma` 限制在一个较小范围内。模型只有在确实能补充规则模型时，才会影响最终排序。

`run_best.sh` 当前默认训练较宽的 MLP sweep：

```bash
EDGE_HIDDEN_DIMS=64,128,256
EDGE_GAMMAS=0.05,0.08,0.15,0.25,0.35
EDGE_SEEDS=2026,2027
EDGE_NEGATIVES=10
EDGE_SAMPLE_EDGES=250000
EDGE_EPOCHS=10
```

这会形成多个不同容量、不同残差强度、不同随机种子的模型。后续不是全部盲目加入，而是交给集成搜索筛选。

### 3.5 可选 CRAFT 动态图训练

CRAFT 训练由 `scripts/train_craft_ranker.py` 完成。

如果环境中安装了 `jittor_geometric`，入口脚本会启用 CRAFT；如果没有安装，则输出提示并自动跳过。

CRAFT 的作用是提供另一类动态图信号。规则模型和 edge MLP 都主要依赖显式统计特征，而 CRAFT 会基于历史邻居序列、交互时间和动态邻居采样学习表示。它可以捕捉一部分手工规则不容易表达的时序图结构。

训练逻辑大致如下：

1. 从验证目录读取训练边。
2. 构造 `TemporalData`。
3. 使用 `TemporalDataLoader` 进行动态图批训练。
4. 对每个 batch 采样历史邻居序列。
5. 使用正样本和负样本训练 BPR 排序损失。
6. 每轮在 `valid.csv` 的 100 候选行上计算 MRR。
7. 保存验证 MRR 最好的 CRAFT 权重。
8. 把 valid/test 的 CRAFT 分数缓存为 `.npy`。

默认 CRAFT sweep：

```bash
CRAFT_NEIGHBORS=20,30,50,80
CRAFT_HIDDEN_SIZES=64,128
CRAFT_EPOCHS=10
```

CRAFT 不是必需组件。它能跑时参与集成，不能跑时系统仍然可以依赖规则模型和 edge residual MLP 完成提交。

### 3.6 集成搜索

集成搜索由 `scripts/search_ensemble.py` 完成。

该步骤是当前方案稳定性的关键。不是训练出多少模型就全部平均，而是在验证集上逐个判断组件是否值得加入。

搜索流程如下：

1. 先计算规则模型在验证集上的 MRR。
2. 从模型目录中发现所有 edge MLP。
3. 从分数目录中发现所有 CRAFT valid score cache。
4. 对每个组件计算单体 MRR。
5. 对每个组件分数做 row z-score 标准化。
6. 默认把非规则组件 residualize against rule，即减去规则分数的 z-score。
7. 从规则模型作为初始 ensemble。
8. 按组件单体 MRR 从高到低尝试加入。
9. 在预设 `WEIGHT_GRID` 中搜索加入权重。
10. 如果加入后 MRR 提升超过 `MIN_ADD_GAIN`，且 Top1 相对规则模型变化率不超过 `MAX_TOP1_DIFF`，才接受该组件。

row z-score 的意义是：每一行的 100 个候选只比较相对分数，不让不同组件的绝对分值尺度影响集成。

residualize against rule 的意义是：如果 MLP 或 CRAFT 已经学到了和规则模型高度相同的排序信号，就不重复放大；只让它们贡献和规则不同的部分。

Top1 drift 限制的意义是：规则模型通常已经很强，如果一个组件让大量行的 Top1 都变掉，即使局部验证有小提升，也可能说明它过于激进。`MAX_TOP1_DIFF=0.35` 用来控制这种风险。

最终输出：

```text
competition_models_best/
  ensemble_weights.json
```

该文件记录每个数据集最终采用哪些组件、每个组件的权重、是否 residualize。

### 3.7 最终预测与打包

最终预测由 `scripts/predict_luxury_ensemble.py` 完成。

预测时，对每个数据集：

1. 读取原始训练边。
2. 读取测试行 `(src, time, candidates)`。
3. 按 `ensemble_weights.json` 逐个加载组件。
4. 对规则组件重新构造规则分数。
5. 对 edge MLP 加载模型并构造候选特征。
6. 对 CRAFT 读取 test `.npy` 缓存。
7. 对每个组件分数做 row z-score。
8. 对 residualized 组件减去规则 z-score。
9. 按权重相加得到总分。
10. 对每行 100 个候选做 softmax。
11. 写出提交 CSV。
12. 打包 zip。

提交文件中每一行有 100 个概率，概率和约等于 1。

## 4. 当前脚本职责

### 4.1 `run_best.sh`

唯一正式入口。它负责串起完整流程：

- 检查或下载数据。
- 创建/复用虚拟环境。
- 安装 `jittor`。
- 构造验证集。
- 并行训练多个 edge residual MLP。
- 检测并可选训练 CRAFT。
- 搜索集成权重。
- 预测最终提交。
- 检查 zip 中每个 CSV 的行概率基本情况。

如果只想正式跑一次结果，使用这个脚本即可。

### 4.2 `scripts/valid_builder.py`

负责构造本地验证集。核心是 `test-row` 模式，让验证候选结构更贴近真实测试候选结构。

它还支持其他候选构造模式，例如 `test-prior`、`recent-heavy`、`popular-heavy`、`transition-heavy`、`mixed`，但当前最推荐的是 `test-row`。

### 4.3 `scripts/train_edge_ranker.py`

负责训练 edge-level residual MLP。它只使用原始时序边和 hard negatives，不直接从验证候选行中学习。

训练结果是每个数据集一个 `.pkl`，同时保存特征均值、标准差、特征名、残差强度等 metadata。

### 4.4 `scripts/train_craft_ranker.py`

负责训练可选 CRAFT 动态图模型。如果 `jittor_geometric` 不可用，该脚本不会被入口脚本调用。

它会保存模型，并额外保存 valid/test 分数缓存，供集成阶段直接读取。

### 4.5 `scripts/search_ensemble.py`

负责自动搜索规则、MLP、CRAFT 的集成权重。

它只接纳能在验证集上带来足够 MRR 提升的组件，并通过 Top1 drift 限制避免过激排序变化。

### 4.6 `scripts/predict_luxury_ensemble.py`

负责读取最终集成配置，对测试集做预测并打包提交。

### 4.7 `scripts/evaluate_mrr.py`

辅助脚本，用于单独评估规则模型在某个验证目录上的 MRR。不是正式入口必需步骤，但适合调试规则权重时使用。

### 4.8 `src/feature_builder.py`

负责从历史边中统计候选特征，是规则模型和 MLP 的共同特征基础。

### 4.9 `src/rule_ranker_v2.py`

负责规则排序。包含重复边场景和新链接场景两套权重。

### 4.10 `src/jt_ranker.py`

定义候选特征顺序、MLP ranker、特征标准化、模型保存与加载。

### 4.11 `src/luxury_scoring.py`

封装验证/测试读取、MRR、row z-score、softmax、规则打分、MLP 打分等通用 scoring 逻辑。

### 4.12 `src/data_loader.py`

封装数据集目录发现、训练边读取、测试行读取等基础数据访问逻辑。

## 5. 核心创新点

### 5.1 将任务明确建模为候选内重排序

很多动态图推荐方案会默认做全量节点预测，但本任务测试集已经给出 100 个候选。当前方案把问题聚焦为候选内排序，避免把资源浪费在全量召回上。

这样可以把工程重点放到：

- 如何模拟真实候选分布。
- 如何在 100 个候选内稳定排序。
- 如何提升 MRR 和 Top1。
- 如何避免候选分布偏差造成线下线上不一致。

### 5.2 使用 test-row 验证构造降低分布偏差

验证集不是简单随机负采样，而是尽量复用真实测试候选行。这样做直接解决了候选分布偏差问题。

如果验证负样本太简单，模型会学到“区分正边和随机节点”；但线上需要区分的是“正边和 99 个官方候选”。这两个问题难度和分布都不同。

`test-row` 验证让本地指标更接近真实提交环境。

### 5.3 规则模型作为强先验

规则模型覆盖了动态图推荐中非常稳定的信号：

- 重复交互。
- 最近行为。
- 目的节点热度。
- 近期趋势。
- 时序转移。
- 滑窗共现。
- 冷启动惩罚。

这些信号可解释、稳定、训练成本低，尤其适合候选内精排。

### 5.4 针对不同数据集采用不同归纳偏置

当前代码显式区分 `dataset1` 和 `dataset2`：

- `dataset1` 强化重复边记忆。
- `dataset2` 强化新链接、热度、趋势、转移、共现。

这不是简单调参，而是根据数据结构差异设置不同归纳偏置。对于动态图任务，这一点很重要，因为不同图的边生成机制可能完全不同。

### 5.5 神经模型只学习残差

Edge MLP 不是单独输出最终排序，而是被限制为规则分数上的补偿项：

```text
final_score = rule_score * fuse_rule + neural_residual * gamma
```

这种设计有三个好处：

1. 保留规则模型的稳定性。
2. 让神经模型专注学习规则遗漏的非线性组合。
3. 通过 `gamma` 控制过拟合风险。

### 5.6 使用 hard negative 训练真实排序能力

负样本优先来自近期、转移、共现和热门目标。这些都是容易和正样本混淆的候选。

相比随机负样本，hard negative 更接近真实 100 候选的困难程度，能让 MLP 学到更有用的排序边界。

### 5.7 多模型 sweep 后保守集成

`run_best.sh` 会训练多个 hidden size、gamma、seed 的 MLP，也会尝试多个 CRAFT 配置。但最终不是全部平均，而是由 `search_ensemble.py` 根据验证 MRR 筛选。

保守集成策略包括：

- 从规则模型开始。
- 组件先 row z-score。
- 非规则组件默认减去规则分数，学习残差。
- 只有 MRR 提升超过阈值才加入。
- 控制 Top1 变化率，避免过激模型破坏强基线。

这种方式比简单平均更稳，也更适合比赛提交。

### 5.8 CRAFT 作为可插拔动态图增强

CRAFT 能跑时增加动态图神经表示能力；不能跑时系统仍然完整可用。

这让方案兼顾了效果上限和环境鲁棒性。

## 6. 关键参数说明

`run_best.sh` 中最重要的参数如下。

### 6.1 验证相关

```bash
VALID_MODE=test-row
VALID_RATIO=0.2
MAX_VALID=0
```

`MAX_VALID=0` 表示使用全部验证样本。如果机器内存或时间不够，可以设为 `150000` 做快速验证。

### 6.2 规则与残差强度

```bash
FUSE_RULE=0.95
EDGE_GAMMAS=0.05,0.08,0.15,0.25,0.35
```

`FUSE_RULE` 是规则主干权重。`EDGE_GAMMAS` 是 MLP 残差强度 sweep。较小 gamma 更保守，较大 gamma 更激进，最终由集成搜索决定是否采用。

### 6.3 Hard negative

```bash
EDGE_NEGATIVES=10
HARD_RECENT_LIMIT=80
HARD_TRANSITION_LIMIT=300
HARD_POPULAR_LIMIT=3000
HARD_POPULAR_SAMPLE=350
```

这些参数控制验证候选构造和训练负样本中近期、转移、热门候选的覆盖范围。

### 6.4 Edge MLP sweep

```bash
EDGE_HIDDEN_DIMS=64,128,256
EDGE_SEEDS=2026,2027
EDGE_EPOCHS=10
EDGE_SAMPLE_EDGES=250000
```

这会形成多个 MLP 版本，提高搜索空间。`EDGE_SAMPLE_EDGES` 控制每个模型最多采样多少监督边，避免训练时间过长。

### 6.5 CRAFT sweep

```bash
RUN_CRAFT=1
CRAFT_NEIGHBORS=20,30,50,80
CRAFT_HIDDEN_SIZES=64,128
CRAFT_EPOCHS=10
```

如果 `jittor_geometric` 不可用，CRAFT 会自动跳过。

### 6.6 集成搜索

```bash
WEIGHT_GRID=0.01,0.02,0.05,0.08,0.1,0.15,0.2
MIN_ADD_GAIN=0.0005
MAX_TOP1_DIFF=0.35
RESIDUALIZE_AGAINST_RULE=1
```

`MIN_ADD_GAIN` 防止加入微弱或偶然提升的组件。`MAX_TOP1_DIFF` 防止最终排序相对规则模型变化过大。

## 7. 输出文件说明

运行后主要产物包括：

```text
validation_best_test-row/
  dataset1/train.csv
  dataset1/valid.csv
  dataset2/train.csv
  dataset2/valid.csv

competition_models_best/
  edge_h.../
    seed_2026/
    seed_2027/
  craft_n.../
  ensemble_weights.json

competition_scores_best/
  dataset1_*_valid.npy
  dataset1_*_test.npy
  dataset2_*_valid.npy
  dataset2_*_test.npy

submission_best/
  dataset1.csv
  dataset2.csv

result_best.zip
```

其中最重要的是 `result_best.zip`，这是最终提交文件。

## 8. 当前方案解决了什么问题

第一，解决了训练和测试格式不一致的问题。通过 `test-row` 验证构造，本地验证更接近真实候选分布。

第二，解决了不同数据集模式差异的问题。规则模型对重复边和新链接两类场景分别设计权重。

第三，解决了神经模型不稳定的问题。MLP 只学习残差，并且通过 gamma、集成筛选和 Top1 drift 控制风险。

第四，解决了随机负样本过简单的问题。Hard negative 让训练更接近真实候选内排序难度。

第五，解决了多模型融合容易过拟合的问题。集成搜索只加入有足够验证收益的组件，并默认 residualize against rule。

第六，解决了环境依赖不确定的问题。CRAFT 是可选增强项，不是硬依赖。

## 9. 风险与注意事项

第一，`run_best.sh` 是效果优先配置，训练时间会比较长。如果没有多卡 GPU，建议先设置：

```bash
MAX_VALID=150000 RUN_CRAFT=0 bash run_best.sh
```

做快速验证，但正式提交仍建议跑默认完整配置。

第二，CRAFT 依赖 `jittor_geometric`。如果环境没有该依赖，脚本会跳过 CRAFT，最终效果可能低于完整配置。

第三，验证集虽然尽量模拟测试候选，但仍然不能保证和线上完全一致。因此保守集成和规则锚定是必要的。

第四，`dataset1` 和 `dataset2` 的规律差异很大，不建议用完全相同的规则权重解释所有数据集。

第五，`MAX_VALID=0` 会使用全部验证数据，最稳但也最耗时。机器资源不足时可以临时截断。

## 10. 一句话总结

当前方案是一个面向动态图候选重排序的稳健集成系统：用贴近真实测试行的验证集校准方向，以规则模型作为强先验，用 edge-level MLP 学习受控残差，必要时加入 CRAFT 动态图表示，最后通过保守集成搜索生成最终提交。它的设计目标不是追求单模型复杂度，而是在候选分布、数据集差异、过拟合风险和线上稳定性之间取得更好的平衡。
