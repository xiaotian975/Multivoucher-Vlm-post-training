# Phase 05: SFT、DPO、GRPO 数据格式构造

## 阶段目标

把原始 case、图片路径和 bbox annotation 转换成 LoRA-SFT、DPO 和 GRPO 所需的数据格式。SFT、DPO、GRPO 不是三套独立数据源，必须从 MV-Train 派生。Val/Test 不参与 DPO 构造和 GRPO reward 调参。

## 允许修改范围

- `src/mv_audit/converters/`
- `configs/schema/output_schema.json`
- `scripts/03_build_train_data.sh`
- `data/mv_audit/sft/`
- `data/mv_audit/dpo/`
- `data/mv_audit/grpo/`

## 禁止事项

- 不训练模型。
- 不修改图片渲染和 bbox 生成逻辑。
- 不从 Val/Test 构造 DPO pairs。
- 不从 Val/Test 构造 GRPO prompts。
- 不为 `missing_document` 的缺失材料构造伪 bbox。
- 不为 `unreadable_image` 的不可读字段构造确定 value 和 bbox。

## 输入

- `data/mv_audit/raw_cases/train_cases.jsonl`
- `data/mv_audit/annotations/field_bboxes_train.jsonl`
- `data/mv_audit/images/train/`
- 全局 output schema 约束。

## 输出

- `src/mv_audit/converters/build_sft_data.py`
- `src/mv_audit/converters/build_dpo_pairs.py`
- `src/mv_audit/converters/build_grpo_prompts.py`
- `configs/schema/output_schema.json`
- `data/mv_audit/sft/train.jsonl`
- `data/mv_audit/sft/val.jsonl`
- `data/mv_audit/dpo/pairs_train.jsonl`
- `data/mv_audit/grpo/prompts_train.jsonl`

SFT 数据包括：

- 完整审核 SFT：60%。
- 字段抽取和证据定位 SFT：25%。
- 单项一致性判断 SFT：15%。

DPO rejected type 至少覆盖：

- `high_risk_pass`
- `wrong_anomaly_type`
- `wrong_risk_level`
- `wrong_evidence_source`
- `wrong_bbox`
- `unsupported_reason`
- `json_invalid`
- `hallucinated_field`
- `unreadable_but_guess`
- `missing_doc_but_pass`

GRPO prompts 优先选择：

- high risk
- `over_reimbursement`
- `merchant_mismatch`
- `applicant_mismatch`
- `order_id_mismatch`
- `missing_document`
- `unreadable_image`
- hard negative

## 测试方式

- 抽样验证 SFT assistant answer 是合法 JSON，并符合 `output_schema.json`。
- 抽样验证 DPO chosen 和 rejected 是同一个 prompt 下的两个回答。
- 抽样验证 GRPO ground truth 包含 reward function 所需字段。
- 专门抽查 `missing_document` 和 `unreadable_image`。

## 完成定义

- SFT、DPO、GRPO 数据均可从 MV-Train 可复现生成。
- 模型输出 schema 不包含 `primary_anomaly_type` 和 `evidence_sufficient`。
- evidence 从 bbox records 自动构造。
- 图片顺序可随机打乱，但 prompt 和 evidence 的 `source_image_id`、`source_doc_type` 对齐。
- 不存在 Val/Test 泄漏到训练格式数据中的情况。

## 下一阶段依赖

phase 06 依赖 `output_schema.json`、SFT/DPO/GRPO ground truth、bbox annotation 和模型输出格式定义。

## 待确认问题

- DPO rejected 中人工手写少量高质量反例是否纳入第一版执行，需要确认。
