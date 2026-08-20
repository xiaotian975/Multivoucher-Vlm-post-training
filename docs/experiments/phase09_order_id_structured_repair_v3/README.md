# Phase09 Order-ID Structured Repair SFT v3

## Decision

当前不直接进入正式 GRPO。Prompt guard R2 在 `train_decode_dev` 上保持 JSON/schema 合法，但没有修复 order_id 高风险漏判：24 条错误中 18 条是 bbox-only，6 条是 missing_order_id_evidence，其中旧 5 条 order_id miss 仍未修复。因此当前残留问题不是“识别风险但决策放行”，而是模型没有稳定读出并表达两处 order_id 证据。

`rl_recommended=false`，当前状态为 `NOT_READY_FOR_RL`。GRPO 只能作为后续 smoke/pilot 展示，不作为当前业务修复手段。

## Prompt Guard R2 Snapshot

- total_cases: 152
- json_validity: 1.0
- schema_compliance: 1.0
- audit_accuracy: 0.9605263157894737
- high_risk_miss_rate: 0.06896551724137931
- evidence_support_rate: 0.988527724665392
- error_cases: 24

错误归因产物：`outputs/analysis/phase09_order_id_structured_repair_v3/prompt_guard_error_attribution.md`。

## Repair SFT v3 Mix

`Order-ID Structured Repair v3` 不改变对外 JSON schema，只强化训练目标格式：

- order_id mismatch 的 reason 显式写成“订单截图订单号 A 与报销申请单订单号 B 不一致”。
- evidence 前两条固定为 `source_doc_type=order, field=order_id` 与 `source_doc_type=reimbursement_form, field=order_id`。
- risk/audit 强制保持 `high/reject_recommendation`。

Mix 构成：

- 240 条 R1/R2 carryover，降低遗忘风险。
- 120 条 Train-only order_id mismatch structured repair。
- 120 条 Train-only calibration，其中 80 条 low/pass，40 条 medium/manual_review。
- 排除 DPO holdout、train_decode_dev、sample500/Test；`overlap_with_excluded_count=0`。

主要产物：

- `docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix.jsonl`
- `docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix_manifest.json`
- `configs/train/high_risk_repair_sft_v3_order_id_structured_from_r2_qwen3vl_8b_server.yaml`
- `scripts/12_run_order_id_repair_sft_v3_server.sh`

## Next Server Run

低成本顺序：

1. 上传 v3 mix/config/script 和相关 schema guard 代码。
2. 先跑 dry-run。
3. 只有 dry-run 通过且用户再次确认 `ALLOW_TRAINING=1`，才启动 Repair SFT v3 小训练。
4. 只跑 `train_decode_dev` 152 条 fast gate。
5. 拉回 metrics/errors/predictions，然后关机。

## Acceptance Criteria

- `json_validity = 1.0`
- `schema_compliance = 1.0`
- High-risk Miss 低于当前 prompt_guard R2 的 `0.0690`
- 旧 5 条 order_id miss 至少修复 3 条
- Evidence Support 下降不超过 `0.01`

如果 v3 后仍然是 missing_order_id_evidence，停止扩展 SFT，进入报告整理；如果残留错误转成 recognized_risk_but_decision_released，再做 GRPO reward smoke 和 compatibility dry-run。