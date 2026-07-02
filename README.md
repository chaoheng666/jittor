# Jittor CRAFT-Rerank

本项目用于赛道一动态推荐任务：对官方测试集中每个 `(src, time)` 的 100 个候选 `dst` 做未来交互概率排序，并输出提交 zip。

项目保留一个 shell 训练入口：

```bash
bash run_best.sh
```

默认输出：

```text
result_best.zip
  dataset1.csv
  dataset2.csv
```

## 方法

当前方案是 CRAFT 风格候选重排器，而不是点式边分类器：

- 候选 `dst` embedding 对 `src` 最近历史序列做 candidate-to-history cross-attention。
- 显式加入 repeat/memory、source transition/cooc、destination prior。
- 加入 TNCN-lite 结构特征：temporal CN、recent CN、AA、RA、PA、2-hop overlap。
- 使用 query-aware gate 融合 CRAFT、repeat、structure、rule 四个分支。
- 训练目标是 `1 positive + K hard negatives` 的 listwise softmax，辅以 pairwise ranking loss。
- 模型选择按验证 MRR，并拆分 seen/unseen MRR；如果神经模型没有稳定超过规则基线，则自动回退 rule-only。

## 常用运行

8 卡正式跑：

```bash
PRESET=final GPU_COUNT=8 MAX_PARALLEL=8 USE_CUDA=1 bash run_best.sh
```

CPU 冒烟测试：

```bash
PRESET=quick USE_CUDA=0 USE_VENV=0 bash run_best.sh
```

只跑某个数据集：

```bash
DATASET=dataset2 bash run_best.sh
```

## 关键参数

- `CRAFT_CONFIGS`：模型搜索配置，格式为 `hidden:embed:history_len:lr:negatives`，多个配置用逗号分隔。
- `CRAFT_SEEDS`：训练 seed 列表。
- `CRAFT_EPOCHS`：每个配置训练轮数。
- `CRAFT_SAMPLE_EDGES`：每个配置最多采样多少真实未来边；`0` 表示使用全部监督边。
- `BATCH_SIZE`：Jittor batch size。
- `MIN_VALIDATION_GAIN`：神经模型相对规则 MRR 的最低提升门槛。

## 文件职责

- `run_best.sh`：唯一 shell 训练入口，负责环境、并行训练、选择、预测和 zip 检查。
- `scripts/train_edge_ranker.py`：训练 CRAFT-Rerank listwise 模型。
- `scripts/select_edge_model.py`：按验证 MRR 选择模型，并在 eval cache 对齐时尝试 top-2 融合。
- `scripts/predict_edge_intensity.py`：对官方 100 候选打分并生成提交。
- `src/feature_builder.py`：动态图缓存、repeat/transition/prior/structure/query 特征。
- `src/jt_ranker.py`：候选特征构造、CRAFT-Rerank 模型、保存加载。
- `src/rule_ranker_v2.py`：规则基线和低收益回退。
