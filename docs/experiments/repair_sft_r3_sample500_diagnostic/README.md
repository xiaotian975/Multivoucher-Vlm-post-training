# repair_sft_r3 sample500 历史口径诊断

## 实验定位

- 本轮只补跑 `repair_sft_r3` 在历史 `sample500` 四 split 上的推理和评测。
- 该结果仅用于诊断和补表，不用于训练、调参、checkpoint 选择或 final holdout 重试。
- DPO V3 checkpoint-15 未运行。
- final_holdout_v1 已消耗且失败：Audit Accuracy=0.7160, High-risk Miss=0.3152, Schema=0.8460, Error Cases=454。

## R3 split 指标

| split | total_cases | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test_clean | 500 | 1.0 | 0.642 | 0.536 | 0.46956521739130436 | 0.5975 | 295 |
| test_robust | 500 | 1.0 | 0.672 | 0.582 | 0.4662756598240469 | 0.6194444444444445 | 283 |
| test_unseen_template | 500 | 1.0 | 0.644 | 0.542 | 0.49002849002849 | 0.5987208008898777 | 280 |
| test_hard_negative | 500 | 1.0 | 0.95 | 0.77 | 0.26077097505668934 | 0.9047741935483871 | 197 |

## 历史均值对比

| model_id | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |
| --- | --- | --- | --- | --- | --- | --- |
| m2_sft | 1.0 | 0.8765000000000001 | 0.7735 | 0.2426641881460555 | 0.8035245025336795 | 164.5 |
| m3_dpo | 1.0 | 0.87 | 0.6685 | 0.2372939319042304 | 0.7987088122605364 | 211.0 |
| m3v2_dpo | 1.0 | 0.87 | 0.7645 | 0.25455002191127213 | 0.795188326535657 | 166.25 |
| repair_sft_r3 | 1.0000 | 0.7270 | 0.6075 | 0.4217 | 0.6801 | 263.7500 |

## Delta vs 历史模型

| baseline_model_id | candidate_model_id | audit_accuracy_delta | high_risk_miss_rate_delta | evidence_support_rate_delta | schema_compliance_delta | error_cases_delta |
| --- | --- | --- | --- | --- | --- | --- |
| m2_sft | repair_sft_r3 | -0.1660 | 0.1790 | -0.1234 | -0.1495 | 99.2500 |
| m3_dpo | repair_sft_r3 | -0.0610 | 0.1844 | -0.1186 | -0.1430 | 52.7500 |
| m3v2_dpo | repair_sft_r3 | -0.1570 | 0.1671 | -0.1151 | -0.1430 | 97.5000 |

## 诊断结论

- 相对 M2 历史 sample500 baseline，`repair_sft_r3` 的 Audit Accuracy 方向为：退化。
- sample500 是 reporting-only 历史 benchmark；本轮结果只用于诊断，不能抵消已消耗 final_holdout_v1 的失败结论。
- 后续若继续改进，应建立新的开发/验证闭环和新的 final holdout v2，不能把本轮 sample500 error cases 回流训练或选择。
