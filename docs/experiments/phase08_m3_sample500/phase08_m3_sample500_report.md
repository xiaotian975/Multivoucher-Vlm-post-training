# Phase 08 M3 Sample500 评测报告（2026-08-11）

## 实验设置

- 评测对象：`M3 = Qwen3-VL-8B-Instruct + Phase 07 SFT adapter + DPO adapter`。
- DPO adapter：`outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1/`。
- 测试规模：`4 splits × 500 = 2000` 条。
- 推理方式：8 卡 shard 数据并行；每个 split 拆成 2 个 shard，完成后合并为标准 prediction JSONL。
- 服务器输出：`outputs/predictions/phase08_m3_sample500/` 与 `outputs/eval_reports/phase08_m3_sample500/`。
- 本地归档：`docs/experiments/phase08_m3_sample500/`。

## 合并校验

四个 split 最终均为 `500/500`。`merge_summary.json` 中四个 split 的 `missing=0`，`duplicates_seen_before_dedup=0`。全量 predictions 未进入 Git，仅保留服务器输出路径和可进 Git 的汇总材料。

## M3 指标

| Split | JSON Validity | Schema Compliance | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.808 | 0.650 | 0.264 | 0.727 | 0.002 | 227 |
| test_robust | 1.000 | 0.804 | 0.652 | 0.264 | 0.724 | 0.002 | 203 |
| test_unseen_template | 1.000 | 0.868 | 0.634 | 0.276 | 0.778 | 0.001 | 231 |
| test_hard_negative | 1.000 | 1.000 | 0.738 | 0.145 | 0.966 | 0.001 | 183 |

## M2 vs M3 平均对比

| Model | Schema Compliance | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | Error Cases Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| m2_sft | 0.877 | 0.773 | 0.243 | 0.804 | 0.001 | 164.500 |
| m3_dpo | 0.870 | 0.668 | 0.237 | 0.799 | 0.001 | 211.000 |
| delta M3-M2 | -0.007 | -0.105 | -0.005 | -0.005 | 0.001 | 46.500 |

## 图表

- [M2/M3 平均指标对比](figures/m2_vs_m3_average_metrics.png)
- [M3 各 split 指标](figures/m3_split_metrics.png)
- [M3 error cases 数量](figures/m3_error_cases_by_split.png)

## 结论

M3 的 JSON Validity 保持为 `1.000`，Hallucination Rate 仍然很低，hard negative 的 Evidence Support Rate 也维持在高位。但与 M2 sample500 相比，M3 平均 Audit Accuracy 从 `0.773` 降至 `0.668`，平均 error cases 从 `164.5` 增至 `211.0`。High-risk Miss Rate 从 `0.243` 变为 `0.237`，只有很小改善，不足以抵消审计准确率下降。

因此，DPO sample1000 虽然训练过程收敛，但 M3 业务指标不支持直接宣称 Phase 08 成功。当前更合理的结论是：DPO 偏好数据或 reward 构造需要复盘，GRPO 不应基于当前结果盲目扩大运行。服务器上的 GRPO 产物只有 `examples=1/global_step=1` 的 smoke 级别结果，不作为正式 M4 结果写入。
