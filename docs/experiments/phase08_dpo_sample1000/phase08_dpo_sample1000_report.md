# Phase 08 DPO Sample1000 实验报告（2026-08-11）

## 实验设置

- 阶段：Phase 08 的 DPO 子任务。
- 基座模型：`Qwen3-VL-8B-Instruct`。
- 初始 adapter：Phase 07 SFT adapter，`outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/`。
- DPO 输出 adapter：`outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1/`。
- 训练配置：`configs/train/dpo_qwen3vl_8b.yaml`。
- 训练数据：`data/mv_audit/dpo_main/pairs_train.jsonl`，来自 MV-Train。
- 本次运行：`max_samples=1000`，`require_existing_images=true`。
- 因缺少可用图片而跳过的 pair：`324`。

## Reward 审计

| Item | Value |
| --- | ---: |
| examples | 1000 |
| chosen mean reward | 1.000000 |
| rejected mean reward | 0.460112 |
| mean reward gap | 0.539888 |
| positive reward gap rate | 0.844000 |
| rejected JSON valid rate | 0.896000 |
| rejected high-risk miss rate | 0.155000 |
| rejected hallucination penalty | 0.015506 |

## 训练动态

| Item | First Step | Last Step |
| --- | ---: | ---: |
| global_step | 1.0 | 63.0 |
| loss | 0.6877865195274353 | 0.0005680027534253895 |
| chosen_logp | -0.0008737894240766764 | -0.007150840014219284 |
| rejected_logp | -50.337337493896484 | -128.99093627929688 |
| preference_margin | 0.10750198364257812 | 74.73100280761719 |

## 图表

- [DPO loss 曲线](figures/dpo_loss_curve.png)
- [Preference margin 曲线](figures/dpo_preference_margin.png)
- [Chosen/Rejected logp 对比](figures/dpo_logp_comparison.png)

## 阶段结论

DPO sample1000 已完成并生成 adapter。训练 loss 从 `0.6877865195274353` 下降到 `0.0005680027534253895`，preference margin 从 `0.10750198364257812` 上升到 `74.73100280761719`，说明训练过程已经把 chosen 与 rejected 的偏好差距明显拉开。

这还不是 Phase 08 的最终业务指标结论。下一步需要用 DPO adapter 作为 `M3 SFT + DPO` 做 sample500 抽样推理评测，并与 Phase 07 的 M2 SFT 结果比较 High-risk Miss Rate、Hallucination Rate、Evidence Support Rate 和 False Manual Review Rate。GRPO 暂缓启动。
