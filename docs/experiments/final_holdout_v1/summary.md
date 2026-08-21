# Final Holdout v1 Summary

This final holdout has been consumed. Results must not be used for further training or reward tuning.

## Aggregate

| total_cases | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 1.0000 | 0.8460 | 0.7160 | 0.3152 | 0.8358 | 454 |

## Split Metrics

| split | total_cases | json_validity | schema_compliance | audit_accuracy | high_risk_miss_rate | evidence_support_rate | error_cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 250 | 1.0000 | 0.8200 | 0.7040 | 0.3392 | 0.8236 | 122 |
| test_robust | 250 | 1.0000 | 0.8560 | 0.7360 | 0.2840 | 0.8598 | 111 |
| test_unseen_template | 250 | 1.0000 | 0.7640 | 0.6440 | 0.3920 | 0.7586 | 133 |
| test_hard_negative | 250 | 1.0000 | 0.9440 | 0.7800 | 0.2455 | 0.9013 | 88 |

## Error Attribution

Machine-readable attribution is written to `error_attribution_summary.json`.

| problem_class | cases |
| --- | ---: |
| evidence_not_trustworthy | 287 |
| model_missed_high_risk | 229 |
| schema_contract_failure | 154 |
| decision_or_risk_mismatch | 130 |
