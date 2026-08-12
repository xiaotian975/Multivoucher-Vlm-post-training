# Phase08 Loss Ablation Baseline Partial 归档（20260812_5gpu_ablation_r3）

## 当前口径

本归档只包含第一个候选 `dpo_v2_baseline` 的训练结果。服务器主流程尚未进入该候选的 Train decode dev 或 sample500 推理评测，因此本目录目前没有业务指标 error cases。为避免打断服务器正在训练的第二个候选，本轮只拉取已经可用的训练审计、pair report 和训练日志，并生成训练曲线图。

## 训练结果

| variant | status | loss_type | lambda_sft | global_step | loss | preference_loss | sft_nll_loss | preference_margin | holdout_pair_accuracy | holdout_preference_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dpo_v2_baseline | training_done_eval_not_run | dpo | 0.1000 | 40.0000 | 0.0121 | 0.0121 | 0.0000 | 44.0543 | 1.0000 | 25.7227 |

## Pair 构造摘要

- Train-only: `True`
- 输入 Train cases: `30000`
- train pairs: `3000`
- holdout pairs: `300`
- decode dev rows: `152`
- Train/Holdout/Decode case overlap: `0`, `0`, `0`

## 图表

- [baseline_loss_curves.png](figures/baseline_loss_curves.png)
- [baseline_preference_margin.png](figures/baseline_preference_margin.png)
- [pair_type_counts.png](figures/pair_type_counts.png)

## 缺失产物

- `error cases`: 尚未生成，因为 `dpo_v2_baseline` 还没有跑 Train decode dev/sample500 推理评测。
- sample500 metrics: 尚未生成。
- 错误迁移分析: 尚未生成。
- adapter 权重存在于服务器 `outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_baseline_ablation/`，但未复制进 Git 归档。
