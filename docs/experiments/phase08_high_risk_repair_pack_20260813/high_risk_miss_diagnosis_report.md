# Phase08 High-risk Miss Diagnosis and Repair Pack

## Conclusion

- DPO v2 two-candidate Train decode dev did not move core business metrics: baseline and AuxDPO have the same Audit Accuracy and High-risk Miss Rate.
- The repair direction is therefore data/contract focused: separate schema failures from real business misses, then reinforce high-risk non-pass cases from MV-Train only.
- This pack is a candidate set for a low-cost SFT/rule-constrained validation run; it is not evidence for sample500 generalization yet.

## Metric Snapshot
| model_id | scope | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m2_sft | sample500_mean | 1.0000 | 0.8765 | 0.7735 | 0.2427 | 0.8035 | 164.5000 |
| m3_dpo | sample500_mean | 1.0000 | 0.8700 | 0.6685 | 0.2373 | 0.7987 | 211.0000 |
| m3v2_dpo | sample500_mean | 1.0000 | 0.8700 | 0.7645 | 0.2546 | 0.7952 | 166.2500 |
| dpo_v2_baseline | train_decode_dev | 1.0000 | 0.8684 | 0.8355 | 0.2299 | 0.8098 | 44.0000 |
| auxdpo_v2_strong | train_decode_dev | 1.0000 | 0.8684 | 0.8355 | 0.2299 | 0.8098 | 42.0000 |

## Error Source Summary
| source | split | error_cases | high_risk_miss | reject_recommendation_errors | top_issues | top_anomaly_families | top_problem_classes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2 sample500 | test_clean | 173 | 44 | 122 | bbox_strict_error:89; schema_invalid:84; business_metrics_zeroed:84; high_risk_miss:44; audit_mismatch:44 | date_mismatch:50; duplicate_in_batch:50; order_id_inconsistent:33; amount_abnormal:13; merchant_mismatch:8 | schema_contract_failure:84; evidence_not_trustworthy:45; model_missed_high_risk:44 |
| M3 sample500 | test_clean | 227 | 30 | 171 | schema_invalid:96; business_metrics_zeroed:96; bbox_strict_error:93; audit_mismatch:79; unsupported_evidence:32 | amount_abnormal:54; date_mismatch:50; duplicate_in_batch:50; order_id_inconsistent:32; person_mismatch:13 | schema_contract_failure:96; evidence_not_trustworthy:52; recognized_risk_but_decision_released:49; model_missed_high_risk:30 |
| M3v2 sample500 | test_clean | 186 | 41 | 129 | bbox_strict_error:94; schema_invalid:92; business_metrics_zeroed:92; high_risk_miss:41; audit_mismatch:41 | date_mismatch:50; duplicate_in_batch:50; order_id_inconsistent:34; amount_abnormal:15; person_mismatch:11 | schema_contract_failure:92; evidence_not_trustworthy:53; model_missed_high_risk:41 |
| M2 sample500 | test_robust | 176 | 34 | 123 | schema_invalid:95; business_metrics_zeroed:95; bbox_strict_error:81; unsupported_evidence:39; high_risk_miss:34 | duplicate_in_batch:50; date_mismatch:50; order_id_inconsistent:31; evidence_or_document_insufficient:12; person_mismatch:11 | schema_contract_failure:95; evidence_not_trustworthy:47; model_missed_high_risk:34 |
| M3 sample500 | test_robust | 203 | 27 | 157 | schema_invalid:98; business_metrics_zeroed:98; audit_mismatch:76; bbox_strict_error:61; unsupported_evidence:32 | duplicate_in_batch:50; date_mismatch:50; amount_abnormal:49; order_id_inconsistent:30; person_mismatch:7 | schema_contract_failure:98; recognized_risk_but_decision_released:49; evidence_not_trustworthy:29; model_missed_high_risk:27 |
| M3v2 sample500 | test_robust | 165 | 32 | 113 | schema_invalid:98; business_metrics_zeroed:98; bbox_strict_error:67; unsupported_evidence:36; high_risk_miss:32 | duplicate_in_batch:50; date_mismatch:50; order_id_inconsistent:30; amount_abnormal:9; person_mismatch:8 | schema_contract_failure:98; evidence_not_trustworthy:35; model_missed_high_risk:32 |
| M2 sample500 | test_unseen_template | 176 | 61 | 124 | bbox_strict_error:108; schema_invalid:68; business_metrics_zeroed:68; unsupported_evidence:62; high_risk_miss:61 | date_mismatch:50; duplicate_in_batch:49; order_id_inconsistent:33; amount_abnormal:13; evidence_or_document_insufficient:10 | schema_contract_failure:68; model_missed_high_risk:61; evidence_not_trustworthy:47 |
| M3 sample500 | test_unseen_template | 231 | 64 | 173 | bbox_strict_error:119; audit_mismatch:117; unsupported_evidence:68; schema_invalid:66; business_metrics_zeroed:66 | amount_abnormal:60; date_mismatch:50; duplicate_in_batch:50; order_id_inconsistent:33; evidence_or_document_insufficient:12 | schema_contract_failure:66; model_missed_high_risk:64; recognized_risk_but_decision_released:53; evidence_not_trustworthy:48 |
| M3v2 sample500 | test_unseen_template | 183 | 64 | 127 | bbox_strict_error:113; schema_invalid:70; business_metrics_zeroed:70; unsupported_evidence:65; high_risk_miss:64 | date_mismatch:50; duplicate_in_batch:50; order_id_inconsistent:35; evidence_or_document_insufficient:12; person_mismatch:10 | schema_contract_failure:70; model_missed_high_risk:64; evidence_not_trustworthy:49 |
| M2 sample500 | test_hard_negative | 133 | 67 | 123 | bbox_strict_error:133; unsupported_evidence:69; high_risk_miss:67; audit_mismatch:67; hallucination:2 | order_id_inconsistent:75; person_mismatch:20; amount_abnormal:19; merchant_mismatch:19 | model_missed_high_risk:67; evidence_not_trustworthy:66 |
| M3 sample500 | test_hard_negative | 183 | 64 | 175 | audit_mismatch:131; bbox_strict_error:128; unsupported_evidence:67; high_risk_miss:64; hallucination:3 | order_id_inconsistent:74; amount_abnormal:74; merchant_mismatch:18; person_mismatch:17 | recognized_risk_but_decision_released:67; model_missed_high_risk:64; evidence_not_trustworthy:52 |
| M3v2 sample500 | test_hard_negative | 131 | 74 | 122 | bbox_strict_error:131; unsupported_evidence:76; high_risk_miss:74; audit_mismatch:74; hallucination:2 | order_id_inconsistent:80; person_mismatch:20; amount_abnormal:18; merchant_mismatch:13 | model_missed_high_risk:74; evidence_not_trustworthy:57 |
| dpo_v2_baseline train_decode_dev | train_decode_dev | 44 | 5 | 28 | bbox_strict_error:24; schema_invalid:20; business_metrics_zeroed:20; high_risk_miss:5; audit_mismatch:5 | duplicate_in_batch:11; date_mismatch:9; person_mismatch:6; evidence_or_document_insufficient:5; clean_or_low_risk:5 | schema_contract_failure:20; evidence_not_trustworthy:19; model_missed_high_risk:5 |
| auxdpo_v2_strong train_decode_dev | train_decode_dev | 42 | 5 | 27 | bbox_strict_error:22; schema_invalid:20; business_metrics_zeroed:20; high_risk_miss:5; audit_mismatch:5 | duplicate_in_batch:11; date_mismatch:9; evidence_or_document_insufficient:5; order_id_inconsistent:5; person_mismatch:5 | schema_contract_failure:20; evidence_not_trustworthy:17; model_missed_high_risk:5 |

## Aggregate Error Mechanisms
### Issue tags
- bbox_strict_error: 1263
- audit_mismatch: 830
- schema_invalid: 807
- business_metrics_zeroed: 807
- unsupported_evidence: 641
- high_risk_miss: 612
- hallucination: 27

### Anomaly families
- order_id_inconsistent: 530
- duplicate_in_batch: 471
- date_mismatch: 468
- amount_abnormal: 349
- person_mismatch: 152
- merchant_mismatch: 122
- evidence_or_document_insufficient: 95
- clean_or_low_risk: 66

### Problem classes
- schema_contract_failure: 807
- evidence_not_trustworthy: 616
- model_missed_high_risk: 612
- recognized_risk_but_decision_released: 218

## Representative High-risk Cases
| source | split | case_id | issues | problem_class | anomaly_family | truth_risk_level | pred_risk_level | truth_audit_result | pred_audit_result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 sample500 | test_clean | MV_MAIN_002394 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | order_id_inconsistent | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_007044 | bbox_strict_error | evidence_not_trustworthy | merchant_mismatch | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_016846 | bbox_strict_error | evidence_not_trustworthy | amount_abnormal | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_031194 | bbox_strict_error | evidence_not_trustworthy | merchant_mismatch | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_012930 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | duplicate_in_batch | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_004195 | bbox_strict_error | evidence_not_trustworthy | amount_abnormal | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_032244 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | order_id_inconsistent | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_002264 | bbox_strict_error | evidence_not_trustworthy | order_id_inconsistent | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_004962 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | order_id_inconsistent | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_032497 | bbox_strict_error | evidence_not_trustworthy | person_mismatch | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_000812 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | order_id_inconsistent | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_021287 | bbox_strict_error | evidence_not_trustworthy | amount_abnormal | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_018569 | bbox_strict_error | evidence_not_trustworthy | amount_abnormal | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_023310 | bbox_strict_error | evidence_not_trustworthy | evidence_or_document_insufficient | high | high | missing_info | missing_info |
| M2 sample500 | test_clean | MV_MAIN_011505 | bbox_strict_error | evidence_not_trustworthy | clean_or_low_risk | low | low | pass | pass |
| M2 sample500 | test_clean | MV_MAIN_028463 | bbox_strict_error | evidence_not_trustworthy | amount_abnormal | medium | medium | manual_review | manual_review |
| M2 sample500 | test_clean | MV_MAIN_005115 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | order_id_inconsistent | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_009620 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | order_id_inconsistent | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_026704 | bbox_strict_error | evidence_not_trustworthy | person_mismatch | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_025291 | bbox_strict_error | evidence_not_trustworthy | order_id_inconsistent | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_014041 | bbox_strict_error | evidence_not_trustworthy | clean_or_low_risk | low | low | pass | pass |
| M2 sample500 | test_clean | MV_MAIN_004013 | bbox_strict_error | evidence_not_trustworthy | merchant_mismatch | high | high | reject_recommendation | reject_recommendation |
| M2 sample500 | test_clean | MV_MAIN_000741 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | duplicate_in_batch | high | low | reject_recommendation | pass |
| M2 sample500 | test_clean | MV_MAIN_028610 | high_risk_miss|audit_mismatch|unsupported_evidence|bbox_strict_error | model_missed_high_risk | date_mismatch | high | low | reject_recommendation | pass |

## Repair Strategy

- Schema contract failures: strengthen output-format constraints before interpreting business metrics.
- Model missed high risk: add MV-Train high-risk non-pass SFT samples with complete evidence and bbox.
- Recognized risk but released decision: reinforce anomaly-to-risk-to-audit mapping, especially reject_recommendation.
- Evidence not trustworthy: keep only candidates with source_doc_type, evidence text, and bbox references.

## Repair Pack

- selected_candidates: 120
- candidate_pool_size: 14315
- selected_family_counts: {"amount_abnormal": 18, "order_id_inconsistent": 17, "evidence_or_document_insufficient": 17, "date_mismatch": 17, "person_mismatch": 17, "merchant_mismatch": 17, "duplicate_in_batch": 17}

| case_id | anomaly_family | score | evidence_count | repair_strategy |
| --- | --- | --- | --- | --- |
| MV_MAIN_000008 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000032 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000059 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000072 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000081 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000097 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000122 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000136 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000137 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000156 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000162 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000183 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000237 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000272 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000283 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000285 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000288 | amount_abnormal | 17 | 6 | SFT reinforce amount cross-check: invoice/payment/order/reimbursement amounts must agree before pass. |
| MV_MAIN_000003 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000093 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000114 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000129 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000167 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000176 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000186 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000235 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000241 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000274 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000291 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000343 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |
| MV_MAIN_000344 | order_id_inconsistent | 17 | 8 | SFT reinforce order-id and order-document consistency with explicit reject/manual-review evidence. |

## Leakage Check

- selected_overlap_counts: {"dpo_v2_holdout": 0, "train_decode_dev": 0, "sample500": 0}
- Source policy: MV-Train high-risk non-pass only; exclude DPO holdout, train_decode_dev, and sample500 case ids.

## Low-cost Validation Gate

- JSON Validity must remain 1.0.
- Audit Accuracy must not fall below M2, or may drop by at most 0.01.
- High-risk Miss Rate must improve by at least 0.03 versus M2.
- Evidence Support Rate may drop by at most 0.01.
- If error cases improve but High-risk Miss Rate does not, stop the training line and write Phase08 as a DPO negative result.
