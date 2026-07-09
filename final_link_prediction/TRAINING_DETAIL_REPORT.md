# 最终方案精细训练报告

本文档面向完全不了解本工程的人，逐步说明最终方案从数据读取、特征构造、模型训练、预测到打包提交的全过程。报告对应当前项目目录 `final_link_prediction`，主入口是 `scripts/train_final.sh` 和 `python -m src.final_pipeline`。

## 1. 总体目标

比赛任务是链接预测。每条测试样本给定：

```text
src, time, candidate_dst_1, candidate_dst_2, ..., candidate_dst_100
```

模型需要给这 100 个候选目标各输出一个概率。真实目标排得越靠前，得分越高。

最终提交包包含两个文件：

- `dataset1.csv`
- `dataset2.csv`

最终线上已知最好结果：

- 提交包：`result_final_blend_0p10.zip`
- 公开分：`1.2829`
- 核心策略：`90%` 稳定基线 + `10%` 上下文增强排序模型

## 2. 入口和默认参数

唯一推荐入口：

```bash
bash scripts/train_final.sh
```

默认执行完整流程：

```bash
ACTION=all bash scripts/train_final.sh
```

也可以分阶段执行：

```bash
ACTION=build bash scripts/train_final.sh
ACTION=train bash scripts/train_final.sh
ACTION=predict bash scripts/train_final.sh
ACTION=package bash scripts/train_final.sh
ACTION=package-sweep bash scripts/train_final.sh
```

默认参数如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `DATA_DIR` | `data_A` | 原始数据目录，包含 `dataset1` 和 `dataset2` |
| `BASELINE_ROOT` | `baseline_artifacts` | 稳定基线产物目录；默认由本项目自动生成 |
| `ARTIFACTS` | `artifacts` | 中间模型、特征、logits 保存目录 |
| `REPORTS` | `reports` | JSON 报告保存目录 |
| `SUBMISSION` | `submission` | CSV 和 zip 提交包保存目录 |
| `SEED` | `3026` | 主随机种子 |
| `WORKERS` | `12` | 多进程构造特征和困难候选的进程数 |
| `HISTORY_FRAC` | `0.70` | 构造训练样本时，把官方训练 split 的前 70% 当历史上下文 |
| `TRAIN_ROWS` | `500000` | 最多抽取 50 万条上下文排序训练样本 |
| `VALID_ROWS` | `80000` | 最多抽取 8 万条验证样本 |
| `MAX_POOL` | `700` | 每个正样本先构造最多 700 个候选池，再筛成 100 个候选 |
| `SVD_DIM` | `128` | 图 SVD embedding 维度 |
| `FIT_EDGE_LIMIT` | `0` | 调试用截断；0 表示不截断 |
| `SRC_SEQ_LEN` | `64` | 源对象近期行为序列长度 |
| `DST_SEQ_LEN` | `64` | 目标对象近期受众序列长度 |
| `SEEDS` | `3101,3102,3103` | 训练 3 个上下文排序 MLP 的随机种子 |
| `HIDDEN` | `256` | MLP 隐藏层宽度 |
| `EPOCHS` | `8` | 每个 MLP 训练 8 轮 |
| `BATCH_SIZE` | `512` | 训练 batch size |
| `PREDICT_BATCH_SIZE` | `2048` | 预测 batch size |
| `LR` | `8e-4` | Adam 学习率 |
| `REUSE_BASELINE_FEATURES` | `0` | 预测时是否复用基线特征分片 |
| `BLEND_WEIGHT` | `0.10` | 上下文增强模型融合权重 |
| `OUTPUT_NAME` | `result_final_blend_0p10` | 默认输出提交名 |
| `SWEEP_BLENDS` | `0.02,0.05,0.10,0.20,0.35,1.00` | 批量打包多个融合比例 |

## 3. 数据来源

### 3.1 原始训练数据

路径：

```text
data_A/dataset2/train.csv
```

读取字段：

- `src`：源对象 ID。
- `dst`：真实连接目标 ID。
- `time`：时间戳或时间顺序字段。
- `split`：可选字段。存在时优先使用官方 split。

读取逻辑：

1. 使用 `csv.DictReader` 读取。
2. 要求至少有 `src,dst,time` 三列。
3. 转成整数。
4. 按 `(time, src, dst)` 排序。

### 3.2 原始测试数据

路径：

```text
data_A/dataset2/test.csv
```

读取格式：

- 第 1 列：`src`
- 第 2 列：`time`
- 后 100 列：候选 `dst`

代码要求每行正好 102 列。读取后保存成：

```text
TestRow(src, time, candidates)
```

其中 `candidates` 是长度为 100 的目标 ID 元组。

### 3.3 稳定基线产物

稳定基线现在已经合并进本项目，不再要求外部目录或旧项目预先存在。默认目录：

```text
baseline_artifacts
```

生成命令：

```bash
ACTION=baseline bash scripts/train_final.sh
```

完整训练命令 `ACTION=all bash scripts/train_final.sh` 会默认先执行稳定基线阶段，再执行上下文增强排序阶段。如果已经有可复用的 `baseline_artifacts`，可以用：

```bash
BUILD_BASELINE=0 ACTION=all bash scripts/train_final.sh
```

稳定基线阶段会从原始 `data_A/dataset1` 和 `data_A/dataset2` 直接生成这些文件：

```text
baseline_artifacts/reports/dataset1_train_report.json
baseline_artifacts/reports/dataset1_predict_report.json
baseline_artifacts/reports/dataset2_train_report.json
baseline_artifacts/reports/dataset2_predict_report.json
baseline_artifacts/artifacts/dataset2_predict_shards/feature_logits_part_*.npy
baseline_artifacts/artifacts/dataset2_predict_shards/mlp_logits_part_*.npy
baseline_artifacts/artifacts/dataset2_predict_shards/features_part_*.npy
baseline_artifacts/submission_mlp_peak/result_rebuild_mlpw_5p5/dataset1.csv
```

这些产物的作用：

1. `dataset1.csv`：最终提交里的 dataset1 结果，由项目内部稳定规则模型直接生成。
2. `dataset2_train_report.json`：保存 dataset2 稳定规则特征权重，用于后续困难候选挖掘和规则 logits 计算。
3. `feature_logits_part_*.npy`：dataset2 测试集上稳定规则模型的原始分数。
4. `mlp_logits_part_*.npy`：dataset2 测试集上稳定基线 MLP 的原始分数。
5. `features_part_*.npy`：dataset2 测试集 21 维基础图特征，可用于复用或检查。

最终基线分数计算方式：

```text
baseline_logits = zscore(feature_logits) + 5.5 * zscore(mlp_logits)
```

其中 `5.5` 是固定的 `BASELINE_MLP_WEIGHT`。它只控制最终上下文模型融合前的稳定基线强度；稳定基线阶段本身会独立训练规则权重和 MLP。

### 3.4 稳定基线阶段具体做什么

稳定基线阶段由 `src/stable_baseline.py` 实现，入口是：

```bash
ACTION=baseline bash scripts/train_final.sh
```

它不是调用根目录的其他项目，也不是读取旧模型；它在 `final_link_prediction` 内部完成以下工作：

1. 训练 dataset1 稳定规则模型。
2. 预测 dataset1，并写出最终提交要用的 `dataset1.csv`。
3. 训练 dataset2 稳定规则权重。
4. 训练 dataset2 稳定基线 MLP。
5. 预测 dataset2，并保存规则 logits、MLP logits 和 21 维基础特征分片。

dataset1 的结果来源：

```text
baseline_artifacts/submission_mlp_peak/result_rebuild_mlpw_5p5/dataset1.csv
```

这个文件现在由本项目直接生成。它使用 `GraphFeatureModel("dataset1")` 构造基础图特征，再通过本地验证集搜索规则特征权重，最后对测试集打分并 softmax 成概率。

dataset2 的稳定基线包含两部分：

- 规则模型：`GraphFeatureModel("dataset2") + search_weights_multi()` 搜出来的规则特征权重。
- MLP 模型：使用稳定规则特征构造的验证样本训练一个候选内 MLP。

稳定基线默认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `STABLE_SEED` | `2026` | 稳定基线随机种子 |
| `STABLE_SVD_DIM` | `160` | 稳定基线 SVD embedding 维度 |
| `STABLE_RECENT_LIMIT` | `160` | 每个源对象保留最近历史目标数 |
| `STABLE_TRANSITION_WINDOW` | `16` | 转移统计窗口 |
| `STABLE_TRANSITION_TOPK` | `384` | 每个历史目标保留的转移候选数 |
| `STABLE_MAX_VALID_EVENTS` | `30000` | 规则权重搜索最多使用的验证事件数 |
| `STABLE_SEARCH_ROUNDS` | `5` | 规则权重坐标搜索轮数 |
| `TRAIN_STABLE_MLP` | `1` | 是否训练 dataset2 稳定基线 MLP |
| `STABLE_MLP_TRAIN_ROWS` | `80000` | 稳定基线 MLP 最多训练样本数 |
| `STABLE_MLP_HIDDEN` | `192` | 稳定基线 MLP 隐藏层宽度 |
| `STABLE_MLP_EPOCHS` | `8` | 稳定基线 MLP 训练轮数 |
| `STABLE_MLP_BATCH_SIZE` | `256` | 稳定基线 MLP batch size |
| `STABLE_MLP_LR` | `8e-4` | 稳定基线 MLP Adam 学习率 |
| `STABLE_PREDICT_WORKERS` | `4` | 稳定基线测试特征分片进程数 |
| `STABLE_PREDICT_BATCH_SIZE` | `16384` | 稳定基线预测 batch size |

## 4. 数据切分

函数：`split_edges(dataset_dir, final_train=False, prefer_official=True)`

默认优先使用官方 split：

- `split == "0"`：训练 split，记作 `split0`
- `split != "0"`：验证 split，记作 `split1`

如果没有官方 split，才按时间尾部切分：

- 前 85%：训练
- 后 15%：验证

最终上下文排序训练阶段继续把 `split0` 拆成两段：

```text
history_edges = split0 前 70%
train_pool = split0 后 30%
```

原因：

- `history_edges` 用来拟合图统计和上下文，模拟“只知道过去”的状态。
- `train_pool` 里的边作为正样本，训练模型预测后续可能出现的连接。

默认抽样：

- 从 `train_pool` 随机抽最多 `500000` 条训练边。
- 从 `split1` 随机抽最多 `80000` 条验证边。
- 随机种子为 `3026`。

## 5. 图特征模型 GraphFeatureModel

图特征模型是最终方案的基础特征生成器。它把历史边转成统计特征、embedding 特征和转移特征。

初始化参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `dataset` | `dataset2` | 当前任务数据集名 |
| `svd_dim` | `128` | SVD embedding 维度 |
| `recent_limit` | `160` | 每个源对象保留最近 160 个历史目标 |
| `transition_window` | `16` | 统计目标转移时看最近 16 个历史目标 |
| `transition_topk` | `256` | 每个历史目标最多保留 256 个转移目标 |
| `seed` | 传入参数 | SVD 随机种子 |

### 5.1 训练时构造了三个图模型

最终流程里会拟合 3 个 `GraphFeatureModel`：

| 模型文件 | 使用数据 | 作用 |
| --- | --- | --- |
| `artifacts/models/dataset2_block_history_model.pkl` | `history_edges` | 构造训练样本特征，避免训练样本看到未来 |
| `artifacts/models/dataset2_split0_valid_model.pkl` | 完整 `split0` | 构造验证样本特征 |
| `artifacts/models/dataset2_all_train_final_model.pkl` | 完整 `train.csv` | 构造测试集最终预测特征 |

三个模型的 seed 分别是：

- `3026`
- `3027`
- `3028`

### 5.2 图统计

`GraphFeatureModel.fit()` 会按时间排序所有历史边，然后统计：

- `src_recent[src]`：某个源对象最近连接过的目标，最多 160 个。
- `src_count[src]`：源对象出现次数。
- `dst_count[dst]`：目标对象出现次数。
- `dst_recent_count[dst]`：目标对象在最近时间段出现次数。
- `dst_mid_count[dst]`：目标对象在中间时间段出现次数。
- `dst_older_count[dst]`：目标对象在较早时间段出现次数。
- `dst_last_time[dst]`：目标对象最近一次出现时间。
- `pair_count[(src, dst)]`：源目标组合历史出现次数。
- `test_candidate_count[dst]`：测试候选集中目标出现次数。

时间段划分：

- `mid_cut`：历史时间的 60 分位点。
- `recent_cut`：历史时间的 82 分位点。

即：

- `time >= recent_cut` 计入 recent。
- `mid_cut <= time < recent_cut` 计入 mid。
- `time < mid_cut` 计入 older。

### 5.3 SVD embedding

构造一个 `src x dst` 稀疏矩阵：

```text
matrix[src_id, dst_id] = log1p(1.0 + 2.8 * time_norm)
```

其中：

```text
time_norm = (time - time_min) / max(time_max - time_min, 1.0)
```

含义：

- 越新的边权重越高。
- 用 `log1p` 压缩极端值。

然后使用：

```text
TruncatedSVD(n_components=128, n_iter=7, random_state=seed)
```

得到：

- `src_emb`：源对象 embedding。
- `dst_emb`：目标对象 embedding。

两者都做 L2 归一化，方便后续用点积表示相似度。

### 5.4 转移统计

对每个源对象按时间排序历史目标序列。对于当前目标 `dst`，向前看最近 `16` 个历史目标 `prev_dst`，统计：

```text
raw[prev_dst][dst] += 1 / sqrt(rank)
```

其中 `rank=1` 表示最近一个历史目标，越近权重越大。

最终转移分数做归一化：

```text
transition[prev_dst][dst] =
    log1p(raw_count) / sqrt(log1p(count(prev_dst)) * log1p(count(dst)))
```

每个 `prev_dst` 只保留 top 256 个转移目标。

## 6. 基础特征构造

每个样本有 100 个候选目标。`GraphFeatureModel.feature_row()` 会为每个候选构造 21 个基础特征。

最终特征张量形状：

```text
rows x 100 x 21
```

### 6.1 21 个基础特征

| 序号 | 特征名 | 构造方式 | 直观含义 |
| --- | --- | --- | --- |
| 1 | `rule` | 多个统计特征的手工加权组合 | 人工规则总分 |
| 2 | `pop` | `log1p(dst_count) / max_log_dst` | 目标总体热度 |
| 3 | `recent_pop` | `log1p(dst_recent_count) / max_log_recent` | 目标近期热度 |
| 4 | `trend` | `log1p(recent) - log1p(mid + older * 0.25)` 后截断到 `[-5,5]` 再除以 5 | 近期趋势是否上升 |
| 5 | `recency` | `1 / (1 + max(time - last_time, 0) / time_scale)` | 目标最近出现得有多近 |
| 6 | `src_recent_exact` | 如果候选在源对象近期历史中，取 `1/sqrt(rank)` | 源对象是否最近连过该目标 |
| 7 | `pair_log` | `log1p(pair_count(src,dst)) / max_log_pair` | 这个源目标组合历史出现强度 |
| 8 | `dst_known` | 历史中出现过为 1，否则为 0 | 候选是否为已知目标 |
| 9 | `degree_cap` | `min(pop, 0.72)` | 截断后的热度，避免过度偏向超热门目标 |
| 10 | `candidate_seen_in_test` | `log1p(test_candidate_count) / max_log_test_candidate` | 候选在测试候选集中出现频率 |
| 11 | `svd` | `dot(src_emb, dst_emb)` | 源和目标的图 embedding 相似度 |
| 12 | `profile` | 源对象历史目标 embedding 加权平均后，与候选目标 embedding 点积 | 源对象兴趣画像匹配度 |
| 13 | `transition` | 最近历史目标到候选目标的转移分数加权和 | 行为序列转移可能性 |
| 14 | `rank_rule` | `1 / rank(rule)` | rule 在本行候选内的倒数排名 |
| 15 | `rank_pop` | `1 / rank(pop)` | 热度候选内排名 |
| 16 | `rank_recent_pop` | `1 / rank(recent_pop)` | 近期热度候选内排名 |
| 17 | `rank_pair` | `1 / rank(pair_log)` | 历史组合强度候选内排名 |
| 18 | `rank_recency` | `1 / rank(recency)` | 最近出现候选内排名 |
| 19 | `rank_svd` | `1 / rank(svd)` | embedding 相似度候选内排名 |
| 20 | `rank_profile` | `1 / rank(profile)` | 兴趣画像候选内排名 |
| 21 | `rank_transition` | `1 / rank(transition)` | 转移分数候选内排名 |

### 6.2 dataset2 的 rule 公式

最终方案主要处理 `dataset2`。`dataset2` 的规则分为：

```text
rule =
    1.00 * degree_cap
  + 1.15 * recent_pop
  + 0.50 * trend
  + 0.65 * recency
  + 0.15 * profile
  + 0.10 * transition
  - 0.28 * pair_log
  - 0.08 * src_recent_exact
  - 0.80 * (1.0 - dst_known)
```

解释：

- 鼓励近期热门、趋势上升、最近出现、画像匹配、序列转移强的候选。
- 惩罚未知目标。
- 对 `pair_log` 和 `src_recent_exact` 给负权重，是因为 dataset2 的线上分布里，简单重复历史连接不一定可靠，过度押重复会有风险。

## 7. 困难候选构造

训练上下文排序模型时，不能只拿简单随机负例。最终方案为每个正样本先构造一个最多 700 个候选的池子，再从中挑出 100 个候选。

函数：

```text
_candidate_pool()
_build_hard_worker()
_hard_lists_from_edges()
```

### 7.1 候选池来源

每条正样本 `(src, pos_dst, time)` 的候选池初始包含真实目标 `pos_dst`，然后按顺序加入：

1. `src` 最近连接过的目标，最多 120 个。
2. `src` 最近 24 个历史目标的转移 top 目标，每个历史目标最多加 40 个。
3. 近期热门目标 `recent_hot`，最多 160 个。
4. 总体热门目标 `hot`，最多 160 个。
5. 低热度目标 `low_pop` 随机采样，最多 120 个。
6. 如果仍不足 `max_pool`，从历史出现过的目标 `known_dst` 里随机补齐。

去重策略：

- 每个候选池维护 `seen` 集合。
- 已出现的目标不会重复加入。

默认最多保留：

```text
max_pool = 700
```

### 7.2 从候选池筛成 100 个候选

对候选池先计算 21 个基础特征，再用稳定基线的规则权重打分：

```text
pool_score = score_feature_tensor(pool_features, baseline_weights)
```

筛选方式：

1. 按 `pool_score` 从高到低排序。
2. 排除真实目标。
3. 取前 99 个最高分错误候选作为困难负例。
4. 加回真实目标，共 100 个。
5. 随机打乱这 100 个候选。
6. 记录真实目标所在下标，作为训练 label。

这样构造出来的负例不是随机的，而是“稳定基线也容易混淆的候选”。这会让 MLP 学到更细的排序差异。

### 7.3 训练集和验证集产物

生成文件：

```text
artifacts/final_train.npz
artifacts/final_valid.npz
```

每个 `.npz` 包含：

- `features`：形状 `N x 100 x 29`，float16 保存。
- `src_ids`：每条样本的源对象 ID。
- `dst_ids`：每条样本的 100 个候选目标 ID。
- `labels`：真实目标在 100 个候选中的位置。

为什么是 29 维：

```text
21 个基础图特征 + 8 个上下文增强特征 = 29
```

## 8. 上下文增强特征

函数：

```text
SequenceContext
_append_sequence_features()
```

这一步在 21 个基础特征后追加 8 个上下文特征。

### 8.1 目标受众上下文

`SequenceContext` 会对历史边按时间排序，统计每个目标最近吸引过哪些源对象：

```text
dst_recent_src[dst] = 最近连接过 dst 的 src 列表，最多 64 个
```

然后对每个 `dst`，取这些源对象的 `src_emb` 平均，得到：

```text
audience_mean[dst]
```

并记录：

```text
audience_count[dst] = log1p(最近受众数量)
```

`audience_count` 之后会除以全局最大值归一化。

### 8.2 源对象近期行为上下文

对每条样本的 `src`，取它最近最多 64 个历史目标：

```text
recent = src_recent[src][-64:]
```

把这些历史目标的 `dst_emb` 取出，形成源对象近期行为序列。

### 8.3 8 个追加特征

| 序号 | 追加特征 | 构造方式 | 含义 |
| --- | --- | --- | --- |
| 1 | `seq_mean` | 候选 `dst_emb` 与源历史目标 embedding 均值点积 | 候选是否符合源对象整体近期兴趣 |
| 2 | `seq_max` | 候选 `dst_emb` 与源历史目标 embedding 的最大点积 | 候选是否和某个最近行为特别相似 |
| 3 | `seq_last` | 候选 `dst_emb` 与源最后一次历史目标 embedding 点积 | 候选是否延续最近一次行为 |
| 4 | `aud_dot` | 目标受众均值 embedding 与当前 `src_emb` 点积 | 当前源对象是否像该目标最近受众 |
| 5 | `aud_count` | `log1p(目标最近受众数) / max_audience_count` | 目标近期吸引力强弱 |
| 6 | `recent_src_hit` | 当前 `src` 是否在目标最近受众列表里 | 当前源对象是否最近连接过或接近该目标受众 |
| 7 | `rank_seq_max` | `1 / rank(seq_max)` | `seq_max` 的候选内排名 |
| 8 | `rank_aud_dot` | `1 / rank(aud_dot)` | `aud_dot` 的候选内排名 |

追加后特征形状：

```text
N x 100 x 29
```

## 9. 上下文排序 MLP

函数：

```text
train_context_ranker()
_train_feature_mlp()
```

### 9.1 输入

训练输入：

```text
artifacts/final_train.npz
```

验证输入：

```text
artifacts/final_valid.npz
```

使用字段：

- `features`：`N x 100 x 29`
- `labels`：真实目标下标，范围 0 到 99

训练前把 float16 特征转为 float32。

### 9.2 特征标准化

对训练集所有候选特征计算均值和标准差：

```text
mean = train_x.reshape(-1, feature_dim).mean(axis=0)
std = train_x.reshape(-1, feature_dim).std(axis=0)
std < 1e-6 的维度设为 1.0
```

然后对训练和验证特征做：

```text
x_norm = (x - mean) / std
```

标准化参数保存到：

```text
artifacts/final_mlp_seed{seed}/feature_norm.npz
```

### 9.3 模型结构

每个候选共享同一个 MLP。输入张量形状：

```text
batch x 100 x 29
```

模型先 reshape：

```text
(batch * 100) x 29
```

然后通过三层线性网络：

```text
Linear(29, 256)
Relu()
Linear(256, 256)
Relu()
Linear(256, 1)
```

最后 reshape 回：

```text
batch x 100
```

输出是每个候选的 logits。

### 9.4 损失函数

使用候选内交叉熵：

```text
cross_entropy_loss(logits, labels)
```

含义：

- 每条样本有 100 个候选。
- `labels` 是真实目标在 100 个候选中的位置。
- 模型学习把真实目标的 logit 拉高。

代码中保留了 margin/bpr 参数入口，但最终实际只使用交叉熵。这样更适配 Jittor/Ascend 图编译，减少不稳定算子。

### 9.5 优化器和训练参数

每个模型使用：

```text
Adam(lr=8e-4, weight_decay=1e-5)
```

默认训练：

- 隐藏层：`256`
- epoch：`8`
- batch size：`512`
- 种子：`3101, 3102, 3103`

每个 epoch 后在验证集上计算：

```text
tie_aware_mrr(pred, valid_y)
```

其中 tie-aware 表示如果多个候选分数完全相同，会按并列名次的平均位置处理。

### 9.6 三个模型的保存位置

三个随机种子分别输出：

```text
artifacts/final_mlp_seed3101/feature_mlp.pkl
artifacts/final_mlp_seed3101/feature_norm.npz

artifacts/final_mlp_seed3102/feature_mlp.pkl
artifacts/final_mlp_seed3102/feature_norm.npz

artifacts/final_mlp_seed3103/feature_mlp.pkl
artifacts/final_mlp_seed3103/feature_norm.npz
```

训练报告保存：

```text
reports/final_model_report.json
```

报告中每个模型记录：

- `status`
- `seed`
- `checkpoint`
- `norm`
- 每个 epoch 的 `loss`
- 每个 epoch 的 `valid_mrr`

## 10. 预测流程

函数：

```text
predict_context_ranker()
```

### 10.1 测试集特征

预测读取：

```text
data_A/dataset2/test.csv
reports/final_build_report.json
artifacts/models/dataset2_all_train_final_model.pkl
```

默认 `REUSE_BASELINE_FEATURES=0`，因此会用最终 all-train 图模型重新构造测试基础特征：

```text
artifacts/final_test_features/features_part_*.npy
```

然后使用稳定基线权重计算规则 logits：

```text
feature_logits = score_feature_tensor(features, baseline_weights)
```

同时仍从稳定基线目录读取 baseline MLP logits：

```text
artifacts/dataset2_predict_shards/mlp_logits_part_*.npy
```

如果 `REUSE_BASELINE_FEATURES=1`，则测试基础特征、规则 logits、MLP logits 都直接从稳定基线目录读取。

### 10.2 追加上下文特征

预测阶段用完整训练集构造最终上下文：

```text
final_ctx = SequenceContext(final_model, all_train_edges, dst_seq_len=64)
```

然后给测试集追加 8 个上下文特征，得到：

```text
test_rows x 100 x 29
```

### 10.3 三模型 ensemble

依次加载：

```text
artifacts/final_mlp_seed3101/feature_mlp.pkl
artifacts/final_mlp_seed3102/feature_mlp.pkl
artifacts/final_mlp_seed3103/feature_mlp.pkl
```

每个模型先用自己的 `feature_norm.npz` 标准化特征，再输出 logits。

最终上下文增强 logits：

```text
context_logits = mean([
    zscore(logits_seed3101),
    zscore(logits_seed3102),
    zscore(logits_seed3103)
])
```

保存：

```text
artifacts/final_context_ensemble.logits.npy
```

稳定基线 logits 保存：

```text
artifacts/final_baseline.logits.npy
```

预测报告保存：

```text
reports/final_predict_report.json
```

包含：

- 使用了几个训练成功的模型。
- logits 形状。
- 是否复用基线特征。
- 上下文模型和基线模型的 top1 改变比例。
- 基线 top1 的目标覆盖、已知目标比例、热度统计。
- 上下文模型 top1 的目标覆盖、已知目标比例、热度统计。

## 11. 最终融合和提交打包

函数：

```text
package_final_result()
```

输入：

```text
artifacts/final_baseline.logits.npy
artifacts/final_context_ensemble.logits.npy
baseline_root 下的 dataset1.csv
```

融合公式：

```text
final_logits =
    zscore(baseline_logits) * (1.0 - blend_weight)
  + zscore(context_logits) * blend_weight
```

默认：

```text
blend_weight = 0.10
```

也就是：

```text
final_logits = 90% 稳定基线 + 10% 上下文增强排序模型
```

然后：

1. 对 `final_logits` 做 softmax，得到每行 100 个概率。
2. 写出 `dataset2.csv`。
3. 从稳定基线目录复制 `dataset1.csv`。
4. 压缩为 zip。

默认输出：

```text
submission/result_final_blend_0p10/dataset1.csv
submission/result_final_blend_0p10/dataset2.csv
submission/result_final_blend_0p10.zip
```

打包报告：

```text
reports/final_pack_manifest.json
```

其中 `0.10` 融合会记录：

```text
public_score: 1.2829
```

## 12. 每一步产物汇总

### build 阶段

命令：

```bash
ACTION=build bash scripts/train_final.sh
```

产物：

```text
artifacts/models/dataset2_block_history_model.pkl
artifacts/models/dataset2_split0_valid_model.pkl
artifacts/models/dataset2_all_train_final_model.pkl
artifacts/block_hard/train_part_*.npz
artifacts/block_hard/valid_part_*.npz
artifacts/final_train.npz
artifacts/final_valid.npz
reports/final_build_report.json
```

做了什么：

1. 读取 dataset2 训练和测试数据。
2. 按官方 split 切出 `split0` 和 `split1`。
3. 把 `split0` 前 70% 作为历史，后 30% 作为训练候选池。
4. 拟合训练历史图模型。
5. 拟合验证图模型。
6. 用稳定基线权重构造困难候选。
7. 给训练和验证样本追加 8 个上下文特征。
8. 保存 29 维训练/验证特征。
9. 用完整训练集拟合最终测试图模型。

### train 阶段

命令：

```bash
ACTION=train bash scripts/train_final.sh
```

产物：

```text
artifacts/final_mlp_seed3101/feature_mlp.pkl
artifacts/final_mlp_seed3101/feature_norm.npz
artifacts/final_mlp_seed3102/feature_mlp.pkl
artifacts/final_mlp_seed3102/feature_norm.npz
artifacts/final_mlp_seed3103/feature_mlp.pkl
artifacts/final_mlp_seed3103/feature_norm.npz
reports/final_model_report.json
```

做了什么：

1. 读取 `final_train.npz` 和 `final_valid.npz`。
2. 对训练特征计算均值和标准差。
3. 训练 3 个结构相同、随机种子不同的 MLP。
4. 每轮记录训练 loss 和验证 MRR。
5. 保存模型和标准化参数。

### predict 阶段

命令：

```bash
ACTION=predict bash scripts/train_final.sh
```

产物：

```text
artifacts/final_test_features/features_part_*.npy
artifacts/final_context_ensemble.logits.npy
artifacts/final_baseline.logits.npy
reports/final_predict_report.json
```

做了什么：

1. 读取测试集。
2. 用最终 all-train 图模型构造测试基础特征。
3. 追加 8 个上下文特征。
4. 加载 3 个 MLP 分别预测。
5. 对 3 份 logits 做行内 z-score 后平均。
6. 保存上下文增强 logits 和稳定基线 logits。

### package 阶段

命令：

```bash
ACTION=package bash scripts/train_final.sh
```

产物：

```text
submission/result_final_blend_0p10/dataset1.csv
submission/result_final_blend_0p10/dataset2.csv
submission/result_final_blend_0p10.zip
reports/final_pack_manifest.json
```

做了什么：

1. 加载稳定基线 logits。
2. 加载上下文增强 logits。
3. 按 90/10 融合。
4. softmax 转概率。
5. 校验 CSV 每行 100 列、概率和接近 1。
6. 打包成比赛提交 zip。

## 13. 为什么最终选择 0.10 融合

纯上下文增强模型会改变更多第一名候选，风险更高。最终公开分最高的已知版本是：

```text
稳定基线 90% + 上下文增强排序模型 10%
```

这个比例的含义：

- 保留稳定基线的大部分判断。
- 允许上下文模型在少数候选上修正排序。
- 控制 top1 change，减少最后一次提交的不确定性。

因此当前主线输出固定为：

```text
result_final_blend_0p10.zip
```

## 14. 复现实验时最容易出错的地方

1. 默认 `BASELINE_ROOT=baseline_artifacts` 会由本项目生成；只有在 `BUILD_BASELINE=0` 时，才要求该目录已经存在。
2. `data_A/dataset2/train.csv` 和 `test.csv` 必须完整。
3. 如果 `REUSE_BASELINE_FEATURES=1`，基线特征分片必须和当前测试集顺序完全一致。
4. 如果只保留最终 zip，不能反推出其他 blend，因为 softmax 后原始 logits 信息已经丢失。
5. `0.20`、`0.35`、`1.00` 这类比例更激进，可能提升，也可能下降。
6. 训练依赖 Jittor 和加速器环境，本地 Windows 只适合做代码检查，不适合完整训练。

## 15. 用一句话复述最终训练策略

先用稳定基线筛出难负例，再用图统计、SVD embedding、源对象行为序列和目标受众上下文训练 3 个候选内排序 MLP，预测时把 3 个 MLP 的上下文增强 logits 平均，最后只以 10% 权重融入稳定基线，生成已知公开分 `1.2829` 的最终提交。
