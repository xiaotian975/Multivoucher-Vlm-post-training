# Phase 08 M3v2 Sample500 评测报告（2026-08-12）

## 实验定位

本轮实验是在 DPO v1 业务失败之后执行的保守 DPO v2 修正实验。目标不是启动 GRPO，而是验证 Train-only、业务加权、保护型 pair 和较保守 DPO 强度能否得到更稳定的 `M3v2 = M2 SFT + DPO v2 adapter`。

## 数据与训练约束

- DPO v2 pair 来源：`data/mv_audit/raw_cases/main/train_cases.jsonl`，严格 Train-only。
- Pair 构造输出：`3000` train pairs、`300` holdout pairs、`152` train decode dev rows。
- Case-level 划分：train/holdout/decode-dev overlap 均为 `0`。
- 过滤：启用 existing-images 与 evidence/bbox 校验，`skipped_missing_images=18021`。
- Pair 类型：hard rejected、high-risk miss、protective、normal calibration。
- 权重：`severity_weight * hardness_weight * reliability_weight`，`min=0.75`，`max=3.0`，均值约 `2.506`。

## DPO v2 训练摘要

- 训练样本数：`3000` pairs。
- 最终 global step：`80`。
- chosen mean reward：`1.0`。
- rejected mean reward：`0.05691666666666736`。
- mean reward gap：`0.9430833333333613`。
- positive reward gap rate：`1.0`。
- 最终训练 loss：`0.003977533429861069`。
- 最终 preference margin：`55.59977722167969`。
- 最终 holdout pair accuracy：`1.0`。

训练曲线见：`figures/dpo_v2_training_curves.png`。

## Train Decode Dev

Train decode dev 只用于训练域小样本监控，不作为 test 结论：

| model_id | split | total_cases | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m3v2_dpo | train_decode_dev | 20 | 0.750 | 0.750 | 0.300 | 0.637 | 8 |

## M3v2 Sample500 结果

| split | total_cases | schema_compliance | field_em | risk_type_macro_f1 | audit_accuracy | high_risk_miss_rate | evidence_support_rate | hallucination_rate | bbox_acc_relaxed | error_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 500 | 0.816 | 0.815 | 0.781 | 0.734 | 0.284 | 0.731 | 0.000 | 0.719 | 186 |
| test_robust | 500 | 0.804 | 0.803 | 0.801 | 0.740 | 0.279 | 0.722 | 0.002 | 0.718 | 165 |
| test_unseen_template | 500 | 0.860 | 0.859 | 0.747 | 0.732 | 0.288 | 0.767 | 0.000 | 0.756 | 183 |
| test_hard_negative | 500 | 1.000 | 1.000 | 0.636 | 0.852 | 0.168 | 0.961 | 0.001 | 0.949 | 131 |

## M2 / M3 / M3v2 平均指标对比

| model_id | schema_compliance | field_em | risk_type_macro_f1 | audit_accuracy | high_risk_miss_rate | evidence_support_rate | hallucination_rate | bbox_acc_relaxed | error_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_sft | 0.877 | 0.876 | 0.743 | 0.773 | 0.243 | 0.804 | 0.001 | 0.795 | 164.5 |
| m3_dpo | 0.870 | 0.869 | 0.753 | 0.668 | 0.237 | 0.799 | 0.001 | 0.790 | 211.0 |
| m3v2_dpo | 0.870 | 0.869 | 0.742 | 0.764 | 0.255 | 0.795 | 0.001 | 0.786 | 166.3 |

关键变化：

- 相比 M3 v1，M3v2 的 Audit Accuracy 从 `0.668` 回升到 `0.764`，error cases 从 `211.0` 降到 `166.3`。
- 相比 M2，M3v2 的 Audit Accuracy 从 `0.773` 降到 `0.764`，下降约 `0.009`，基本贴近“最多下降 0.01”的保守门槛。
- 但 High-risk Miss Rate 从 M2 的 `0.243` 升到 M3v2 的 `0.255`，没有达到“至少下降 0.03”的目标，方向仍不理想。
- Evidence Support Rate 从 `0.804` 到 `0.795`，下降约 `0.008`，在可接受边界内。

## 错误迁移

| split | both_correct | both_wrong | m2_correct_m3v2_wrong | m2_wrong_m3v2_correct | audit_accuracy_delta | high_risk_miss_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 363 | 124 | 9 | 4 | -0.010 | 0.014 |
| test_robust | 363 | 122 | 8 | 7 | -0.002 | 0.003 |
| test_unseen_template | 358 | 121 | 13 | 8 | -0.010 | 0.014 |
| test_hard_negative | 411 | 52 | 22 | 15 | -0.014 | 0.016 |

错误迁移说明：M3v2 显著修复了 DPO v1 对总体审计准确率的破坏，但没有解决高风险漏检，部分 split 的 high-risk miss delta 仍为正。错误迁移脚本仅使用冻结评测输出做诊断，不得把 Test/sample500 case 回流到后续 DPO 训练数据。

## 结论

DPO v2 是一次“部分修复但未达最终目标”的实验：

- 成功点：相比 DPO v1，M3v2 明显恢复了 Audit Accuracy，error cases 接近 M2，说明 Train-only、protective pairs 和保守训练强度确实减少了负迁移。
- 失败点：High-risk Miss Rate 没有下降，反而略高于 M2，说明当前 DPO v2 pair/reward 仍没有把“高风险不放行”变成稳定业务能力。
- 当前不建议进入 GRPO/M4。更合理的下一步是继续做 DPO v3 或 SFT 数据增强，围绕 high-risk miss 构造更强但不过度破坏审计边界的 Train-only hard cases。

## 归档文件

- `metrics_summary.csv`
- `metrics_by_model.csv`
- `m2_m3_m3v2_split_metrics.csv`
- `m2_m3v2_transition_summary.csv`
- `dpo_v2/dpo_v2_reward_audit.json`
- `dpo_v2/pair_report.json`
- `error_cases/*.jsonl`
- `error_migration/*.json/.csv`
- `figures/*.png`
