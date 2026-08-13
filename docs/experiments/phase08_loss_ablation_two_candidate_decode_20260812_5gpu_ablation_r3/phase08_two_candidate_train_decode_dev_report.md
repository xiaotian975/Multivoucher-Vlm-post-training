# Phase08 Two-candidate Train Decode Dev 对比（20260812_5gpu_ablation_r3）

## 实验口径

本轮按省钱策略截停完整 5 候选 ablation：`dpo_v2_baseline` 与 `auxdpo_v2_strong` 完成训练后，不再继续 `auxdpo_v2_stronger`、`ipo_v1`、`ipo_aux_v1`。本报告只比较前两个候选在 Train-only decode dev 上的表现，不包含 sample500 测试结论。

## Train Decode Dev 指标

| variant | total_cases | json_validity | schema_compliance | field_em | risk_type_macro_f1 | audit_accuracy | high_risk_miss_rate | false_manual_review_rate | evidence_support_rate | hallucination_rate | evidence_bbox_accuracy_relaxed | error_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dpo_v2_baseline | 152.0 | 1.0000 | 0.8684 | 0.8678 | 0.9012 | 0.8355 | 0.2299 | 0.0000 | 0.8098 | 0.0000 | 0.7983 | 44.0 |
| auxdpo_v2_strong | 152.0 | 1.0000 | 0.8684 | 0.8678 | 0.9012 | 0.8355 | 0.2299 | 0.0000 | 0.8098 | 0.0000 | 0.7983 | 42.0 |

## 关键结论

- 两个候选的 `audit_accuracy`、`high_risk_miss_rate`、`evidence_support_rate` 基本一致。
- `auxdpo_v2_strong` 的 `error_cases` 较 baseline 少 2 个，但不足以证明业务指标显著改善。
- 该结果仅用于决定是否继续扩大到 sample500，不是正式 sample500 测试结论。

## 图表

- [train_decode_core_metrics.png](figures/train_decode_core_metrics.png)
- [train_decode_validity_metrics.png](figures/train_decode_validity_metrics.png)
- [train_decode_error_cases.png](figures/train_decode_error_cases.png)
