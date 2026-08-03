# Phase 02: Case Schema、字典与底层交易真值表

## 阶段目标

生成正常的结构化交易真值表。每个 base case 包含金额、人员、商户、日期、订单号、支付流水号、费用类型、材料列表和初始标签。此阶段是后续异常注入、图片渲染、bbox 标注、训练数据构造和评测的基础。

## 允许修改范围

- `configs/schema/case_schema.json`
- `configs/data_gen/debug.yaml`
- `configs/data_gen/main.yaml`
- `data/mv_audit/dictionaries/`
- `src/mv_audit/data_gen/generate_base_cases.py`
- `src/mv_audit/data_gen/case_validator.py`
- `scripts/01_generate_base_cases.sh`
- `data/mv_audit/raw_cases/base_cases_debug.jsonl`

## 禁止事项

- 不注入异常。
- 不实现 risk rule engine。
- 不做数据划分。
- 不渲染图片。
- 不构造 SFT/DPO/GRPO 数据。
- 不训练模型。

## 输入

- `configs/data_gen/debug.yaml`
- `configs/data_gen/main.yaml`
- `data/mv_audit/dictionaries/names.json`
- `data/mv_audit/dictionaries/merchants.json`
- `data/mv_audit/dictionaries/expense_types.json`
- `data/mv_audit/dictionaries/cities.json`

调试规模可以先使用名字 200 个、商户 100 个、费用类型 8 类、城市 20 个。主实验再扩展到名字 2,000 个、商户 1,000 个、费用类型 8 到 12 类、城市 50 个。

## 输出

`case_schema.json` 至少包含：

- `case_id`
- `applicant`
- `payer`
- `order_user`
- `merchant_canonical`
- `invoice_merchant`
- `payment_merchant`
- `order_merchant`
- `merchant_aliases`
- `expense_type`
- `invoice_amount`
- `payment_amount`
- `reimbursement_amount`
- `order_amount`
- `tax_amount`
- `order_date`
- `payment_date`
- `invoice_date`
- `application_date`
- `invoice_id`
- `order_id`
- `payment_id`
- `documents`
- `primary_anomaly_type`
- `anomaly_types`
- `evidence_sufficient`
- `risk_level`
- `audit_result`
- `metadata`

正常 base case 默认：

```json
{
  "primary_anomaly_type": "none",
  "anomaly_types": [],
  "risk_level": "low",
  "audit_result": "pass",
  "evidence_sufficient": true,
  "documents": ["invoice", "payment", "reimbursement_form", "order"]
}
```

## 测试方式

- 运行 `bash scripts/01_generate_base_cases.sh`。
- 校验输出 JSONL 数量正确。
- 校验所有 case 通过 `case_schema.json`。
- 抽样 10 条，检查金额、日期、商户、人员、订单号合理。

## 完成定义

- `case_id` 不重复。
- 金额为两位小数字符串。
- 日期为 `YYYY-MM-DD`。
- 正常样本四类金额一致，人员一致，日期满足 `order_date <= payment_date <= invoice_date <= application_date`。
- 所有正常样本为 `low + pass`。

## 下一阶段依赖

phase 03 依赖本阶段输出的正常 base cases、稳定 case schema 和基础校验函数。
