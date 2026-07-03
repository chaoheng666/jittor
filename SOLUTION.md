# 完整深模融合方案说明

## 1. 目标

本仓库不再只按 `future_fused_acc` 选择单个 `rule + edge_mlp` 模型，而是把未来边预测实现为一个可回退的融合系统：

```text
score(src, dst, time) ~= future-edge intensity under history before time
```

官方测试集给出的 100 个候选只在最终导出和 sanity 阶段使用；训练和主验证围绕真实未来边与大池负样本展开。

## 2. 核心组件

- `base_intensity_v3`：主干统计模型，融合 pair repeat、recency、destination popularity、source transition、co-occurrence、temporal CN/AA/RA 和 cold penalty。
- `manual_rule`：保留旧规则层，作为稳定先验和回退项。
- `edge_mlp_legacy`：保留旧 MLP residual，新增 `--negative-mode legacy|mixed|proposal`；fusion 默认只在 large-pool 有收益时启用。
- `seq_nextdst`：Jittor embedding sequence tower，学习 `src` 最近交互序列下的 next destination。
- `craft_residual`：Jittor target-aware residual，默认 `cold_policy=zero_id`，只允许小权重参与重排。

如果当前环境没有 Jittor，`train_nextdst.py` 和 `train_craft_residual.py` 会写出 disabled artifact；base、validation、sanity 和 fusion 仍可运行。

## 3. 验证协议

主验证是 `scripts/validate_large_pool.py`：

```text
positive future edge + mixed proposal negatives -> MRR / Hit@10
```

负样本来自四类 proposal：

- source-local hard negatives
- global/recent popular destinations
- random seen destinations
- random cold/all-space destinations

`scripts/time_replay_eval.py` 按时间块推进历史，检查组件是否依赖固定时间段热度。`scripts/official_candidate_sanity.py` 只做提交前诊断，检查 top1/top5 unseen 比例、rule top1 agreement 和概率导出风险。

## 4. 融合与回退

`scripts/select_fusion.py` 输出：

```text
models_v2/fusion_config.json
```

每个 dataset 配置包含：

- `components`
- `temperature`
- `cold_penalty`
- `fallback_policy`
- `validation_metrics`
- `sanity_metrics`

默认权重：

```text
dataset1: base=0.55, seq=0.15, craft<=0.05, legacy=0.10, rule=0.15
dataset2: base=0.45, seq=0.30, craft<=0.05, legacy=0.05, rule=0.20, cold_penalty=0.20
```

实际启用规则更保守：deep/legacy 组件必须在 large-pool 上不低于 `base_intensity_v3` 和 `manual_rule`，否则权重置零；seq 若 time-replay 下降超过 `0.002`，权重降到 `0.10` 或更低；sanity 失败时关闭 CRAFT、压低 seq，并增加 cold penalty。

## 5. 入口

唯一入口仍是：

```bash
bash run_best.sh
```

完整流程：

```text
analyze
  -> train_base
  -> train_legacy / train_seq / train_craft
  -> validate_large_pool
  -> time_replay_eval
  -> select_fusion
  -> official_candidate_sanity
  -> predict_fusion
  -> result_best.zip
```

## 6. 产物

```text
reports/data_stats/
reports/val_large_pool.csv
reports/time_replay.csv
reports/official_candidate_sanity.json
reports/export_check.json
models_v2/fusion_config.json
result_best.zip
```

模型和报告产物都在 `.gitignore` 中，不会进入版本库。
