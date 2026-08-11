# Phase 08 DPO 诊断与优化方案（2026-08-11）

## 结论

DPO sample1000 的训练过程本身收敛，但 M3 sample500 业务评测不符合预期。M3 相比 M2 的 High-risk Miss Rate 只从 `0.243` 小幅降到 `0.237`，但 Audit Accuracy 从 `0.773` 降到 `0.668`，error cases 平均从 `164.5` 增到 `211.0`。因此当前不应继续扩大 GRPO，应先复盘 DPO 偏好数据、reward 构造和训练强度。

## 关键证据

| Metric | M2 SFT | M3 SFT+DPO | Delta M3-M2 |
| --- | ---: | ---: | ---: |
| Audit Accuracy | 0.773 | 0.668 | -0.105 |
| High-risk Miss Rate | 0.243 | 0.237 | -0.005 |
| Evidence Support Rate | 0.804 | 0.799 | -0.005 |
| Error Cases Avg | 164.500 | 211.000 | +46.500 |

DPO reward audit 显示 chosen mean reward 为 `1.000`、rejected mean reward 为 `0.460`，训练后 preference margin 从 `0.108` 增至 `74.731`，loss 降到 `0.000568`。这说明模型很快学会区分当前 chosen/rejected pair，但这种偏好没有稳定转化为业务指标提升。

## 错误迁移

| Split | M2 Correct -> M3 Wrong | M2 Wrong -> M3 Correct | Both Wrong | Both Correct |
| --- | ---: | ---: | ---: | ---: |
| test_clean | 65 | 11 | 162 | 262 |
| test_robust | 57 | 30 | 146 | 267 |
| test_unseen_template | 76 | 21 | 155 | 248 |
| test_hard_negative | 90 | 40 | 93 | 277 |
| Total | 288 | 102 | 556 | 1054 |

M3 新增错误 `288` 个，只修复 M2 错误 `102` 个，净增错误 `186` 个。`test_hard_negative` 的新增错误最多，说明 DPO 对困难负例的审计边界造成了明显扰动。

## 问题类型变化

| Issue | Delta M3-M2 |
| --- | ---: |
| audit_mismatch | +197 |
| schema_invalid | +13 |
| business_metrics_zeroed | +13 |
| hallucination | +2 |
| high_risk_miss | -21 |
| unsupported_evidence | -15 |
| bbox_strict_error | -10 |

最核心的问题不是 JSON 合法性或证据框，而是 `audit_mismatch` 大幅增加。DPO 确实略微减少了 `high_risk_miss`，但收益远小于审计决策错误增加带来的损失。

## 高风险新增错误集中类型

`M2 correct -> M3 wrong` 主要集中在高风险 `reject_recommendation` 样本：

| Pattern | Count |
| --- | ---: |
| test_hard_negative / high / reject_recommendation / amount_mismatch | 57 |
| test_robust / high / reject_recommendation / amount_mismatch | 24 |
| test_unseen_template / high / reject_recommendation / amount_mismatch | 23 |
| test_unseen_template / high / reject_recommendation / over_reimbursement | 22 |
| test_clean / high / reject_recommendation / amount_mismatch | 21 |
| test_clean / high / reject_recommendation / over_reimbursement | 20 |
| test_robust / high / reject_recommendation / over_reimbursement | 18 |
| test_hard_negative / high / reject_recommendation / order_id_mismatch | 14 |

这说明当前 DPO 可能破坏了 SFT 已学到的高风险拒绝边界，尤其是金额类异常、超额报销和订单号不一致。

## 优化建议

1. 暂停 GRPO，不把当前 `examples=1/global_step=1` 的 smoke 产物作为正式 M4 结果。
2. 重构 DPO pair 数据，只从 Train 构造，不从 Val/Test 反向调参。
3. 增加 hard rejected：结构合法、证据看似完整，但 `audit_result` 或 `risk_level` 错误的 rejected。
4. 增加保护型 pair：保留 M2 在高风险金额类、超额报销、订单号不一致上的正确拒绝行为。
5. 降低 DPO 训练强度：更小 learning rate、更少 step、更低 beta 或 early stopping，避免 preference margin 过度拉大。
6. 加入 DPO holdout pair 监控，不以训练 loss 接近 0 作为成功标准。

## 下一轮验收门槛

新 DPO 版本至少应满足：

- Audit Accuracy 不低于 M2，或下降不超过 `0.01`。
- High-risk Miss Rate 相比 M2 至少下降 `0.03`。
- Evidence Support Rate 不低于 M2，或下降不超过 `0.01`。
- Hallucination Rate 继续保持低位。
- error cases 平均数量不高于 M2。

如果新 DPO 仍无法同时改善 High-risk Miss Rate 和 Audit Accuracy，应停止 DPO 扩大训练，把 Phase08 结论写成 DPO negative result / reward-data mismatch，并把后续重点转向偏好数据质量和审计规则。

## 相关材料

- [错误迁移统计](transition_summary.csv)
- [逐 case 失败分析](dpo_failure_analysis.csv)
- [按风险/审计/异常类型聚合](failure_by_type.csv)
- [问题类型变化](issue_delta_summary.csv)
- [错误迁移图](figures/dpo_transition_counts.png)
- [M3-M2 指标差异图](figures/m3_minus_m2_metric_delta.png)
- [Issue count shift 图](figures/issue_count_shift.png)
