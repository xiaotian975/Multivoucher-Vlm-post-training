# Phase 03: 异常注入、风险规则和数据划分

## 阶段目标

把正常 base cases 改造成包含异常的审计样本，并用统一 `risk_rule_engine.py` 生成唯一 `risk_level` 和 `audit_result`。同时完成 case 级数据划分，防止图片级混拆和数据泄漏。

## 允许修改范围

- `src/mv_audit/data_gen/anomaly_injector.py`
- `src/mv_audit/data_gen/risk_rule_engine.py`
- `src/mv_audit/data_gen/split_builder.py`
- `configs/data_gen/debug.yaml`
- `configs/data_gen/main.yaml`
- `scripts/01_generate_cases.sh`
- `scripts/01_inject_anomalies.sh`
- `scripts/01_split_cases.sh`
- `data/mv_audit/raw_cases/`

## 禁止事项

- 不渲染图片。
- 不记录 bbox。
- 不构造 SFT/DPO/GRPO 数据。
- 不训练模型。
- 不在多个脚本中重复实现风险规则。

## 输入

- `data/mv_audit/raw_cases/base_cases_debug.jsonl`
- 异常比例配置。
- 全局异常类型、risk rule 和 audit priority。

## 输出

- `data/mv_audit/raw_cases/all_cases_with_anomaly_debug.jsonl`
- `data/mv_audit/raw_cases/train_cases.jsonl`
- `data/mv_audit/raw_cases/val_in_template_cases.jsonl`
- `data/mv_audit/raw_cases/val_unseen_template_cases.jsonl`
- `data/mv_audit/raw_cases/test_clean_cases.jsonl`
- `data/mv_audit/raw_cases/test_robust_cases.jsonl`
- `data/mv_audit/raw_cases/test_unseen_template_cases.jsonl`
- `data/mv_audit/raw_cases/test_hard_negative_cases.jsonl`
- 异常、风险等级和审核结果统计报告。

支持 `primary_anomaly_type`：

- `none`
- `amount_mismatch`
- `over_reimbursement`
- `date_mismatch`
- `merchant_mismatch`
- `applicant_mismatch`
- `order_id_mismatch`
- `missing_document`
- `duplicate_in_batch`
- `unreadable_image`

## 测试方式

- 统计每类 `primary_anomaly_type` 数量。
- 统计每类 `risk_level` 数量。
- 统计每类 `audit_result` 数量。
- 随机抽样 20 条异常样本人工检查标签。
- 检查 Train/Val/Test 无重复 `case_id`。

## 完成定义

- `risk_level` 和 `audit_result` 全部由 `risk_rule_engine.py` 统一生成。
- 多异常样本取最高风险等级。
- `missing_document` 的 `audit_result` 优先为 `missing_info`。
- 高风险且 `evidence_sufficient=true` 输出 `reject_recommendation`。
- 高风险且 `evidence_sufficient=false` 输出 `manual_review`。
- `unreadable_image` 在 metadata 中记录不可读凭证、不可读字段和 `core_field_unreadable`。
- `amount_mismatch` 与 `over_reimbursement` 遵守更具体异常优先原则。

## 下一阶段依赖

phase 04 依赖本阶段输出的 split cases、`documents` 列表、`metadata.template_group`、`metadata.unreadable_doc_type`、`metadata.unreadable_fields` 和重复凭证 metadata。
