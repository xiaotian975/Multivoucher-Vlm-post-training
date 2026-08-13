
## Phase08 Two-candidate Train Decode Dev 对比（20260812_5gpu_ablation_r3）

为降低服务器成本，本轮在 `dpo_v2_baseline` 与 `auxdpo_v2_strong` 两个候选完成训练后截停后续 3 个候选，只做 Train-only decode dev 对比。完整归档见 `docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/phase08_two_candidate_train_decode_dev_report.md`。

| variant | total_cases | json_validity | schema_compliance | field_em | risk_type_macro_f1 | audit_accuracy | high_risk_miss_rate | false_manual_review_rate | evidence_support_rate | hallucination_rate | evidence_bbox_accuracy_relaxed | error_cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dpo_v2_baseline | 152.0 | 1.0000 | 0.8684 | 0.8678 | 0.9012 | 0.8355 | 0.2299 | 0.0000 | 0.8098 | 0.0000 | 0.7983 | 44.0 |
| auxdpo_v2_strong | 152.0 | 1.0000 | 0.8684 | 0.8678 | 0.9012 | 0.8355 | 0.2299 | 0.0000 | 0.8098 | 0.0000 | 0.7983 | 42.0 |

该结果不是 sample500 测试结论，只用于决定是否值得继续扩大到 sample500。
