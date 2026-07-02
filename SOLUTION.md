# CRAFT-Rerank 动态推荐方案

## 核心判断

赛题最终评测是每个测试查询的 100 候选重排序，目标是 MRR。因此当前项目不再把任务当作点式边分类，也不再只训练“真实边高于随机负边”的 MLP 残差，而是直接训练候选列表内的排序模型。

本地数据诊断结果支持场景化路由：

- `dataset1`：非二部图，重复边额外占比高，repeat/memory 信号强。
- `dataset2`：严格二部图，重复边占比低，更依赖 source 最近序列、目标热度趋势和 item transition/cooc。

最终方案保留规则基线作为稳定下限，用 CRAFT 风格主干学习候选与 source 最近行为的匹配关系，并加入 repeat 和 TNCN-lite 结构分支。

## 模型结构

每个查询输入为：

```text
src, time, [candidate_1 ... candidate_100]
```

训练时每条真实未来边构造成：

```text
1 positive + K hard negatives
```

模型包含四个分支：

1. `craft_score`：候选 `dst` embedding 对 `src` 最近 `L` 个历史目标做 cross-attention，使用节点 embedding、位置编码和历史时间差编码。
2. `repeat_score`：学习重复边、近期访问、pair count、pair recency 等 seen-heavy 信号。
3. `structure_score`：学习 temporal CN、recent CN、AA、RA、PA、2-hop overlap 等 TNCN-lite 结构信号。
4. `rule_score`：吸收手工规则基线，作为稳定残差和自动回退依据。

四个分支通过 query-aware gate 融合。gate 的输入是 query 级统计量，包括 source 活跃度、repeat rate、候选历史命中率、候选 cold ratio、候选平均热度、是否二部图等。

## 特征

特征只来自训练图和时间，不使用外部数据。

核心候选特征：

- pair hazard：是否出现过、历史次数、近期次数、最近一次距离当前时间、是否在最近 5/10/20/50 次交互中。
- source sequence：last transition、recent transition、window cooc、reverse transition/cooc。
- destination prior：全局热度、近期热度、趋势、最近活跃时间、unique source 数、cold 标记。
- structure：temporal CN、recent CN、AA、RA、PA、2-hop overlap、shared recent neighbor。

核心序列输入：

- source 最近 `L` 个历史目标节点。
- 每个历史目标相对当前查询时间的 log time delta。
- 新近优先的位置编码。

## 训练与选择

训练脚本是：

```bash
bash run_best.sh
```

内部流程：

```text
读取 train/test
  -> 按时间切分 history/supervision
  -> 从真实未来边构造 listwise 样本
  -> hard negative 采样
  -> 训练 CRAFT-Rerank
  -> 验证集计算 MRR / hit@1 / seen MRR / unseen MRR
  -> 选择验证 MRR 最高模型
  -> 如 top-2 eval cache 对齐，尝试加权融合
  -> 若神经模型未超过规则基线阈值，回退 rule-only
  -> 对官方 100 候选预测并 softmax
  -> 打包 result_best.zip
```

主损失为 listwise softmax，辅助损失为 pairwise ranking。这样训练目标与最终 MRR@100 更一致。

## 资源策略

默认 `final` preset 会跑 8 个高价值配置，适合 48 核 CPU + 8 卡 16G GPU 的服务器：

```bash
PRESET=final GPU_COUNT=8 MAX_PARALLEL=8 USE_CUDA=1 bash run_best.sh
```

默认配置不做大规模无意义网格搜索，而是围绕最关键的三个超参展开：

- history length：60/90/120
- embedding dim：48/64/80
- negatives：24/32/48/64

如果需要快速确认环境和提交格式：

```bash
PRESET=quick USE_CUDA=0 USE_VENV=0 bash run_best.sh
```

## 输出

```text
competition_models_best/
  craft_*/seed_*/dataset*_edge_ranker.pkl
  craft_*/seed_*/dataset*_edge_ranker_eval.npz
  craft_rerank_config.json

submission_best/
  dataset1.csv
  dataset2.csv

result_best.zip
```

`result_best.zip` 中每行 100 个概率，概率和约等于 1。

## 回退策略

规则分支不是临时兜底，而是正式方案的一部分。选择阶段会比较：

```text
validation_mrr - rule_mrr
```

如果提升小于 `MIN_VALIDATION_GAIN`，该数据集直接使用 rule-only，避免神经模型在弱验证收益下污染提交。
