# Phase 06: JSON Parser、BBox Evaluator 和基础评测

## 阶段目标

在训练前先建立可信评测系统。评测模块也是 phase 08 reward function 的基础，因此必须先于 SFT 训练完成并通过 fake predictions 验证。

## 允许修改范围

- `src/mv_audit/evaluation/`
- `scripts/make_fake_predictions.py`
- `scripts/08_evaluate.sh`
- `outputs/eval_reports/`

## 禁止事项

- 不训练模型。
- 不修改训练数据构造逻辑。
- 不把严重错误 JSON 过度自动修复成正确输出。
- 不让业务指标绕过 JSON/schema 失败。

## 输入

- `configs/schema/output_schema.json`
- ground truth JSONL。
- bbox annotations。
- fake predictions 或模型 raw output。

## 输出

- `src/mv_audit/evaluation/json_parser.py`
- `src/mv_audit/evaluation/bbox_evaluator.py`
- `src/mv_audit/evaluation/field_metrics.py`
- `src/mv_audit/evaluation/consistency_metrics.py`
- `src/mv_audit/evaluation/audit_metrics.py`
- `src/mv_audit/evaluation/evidence_metrics.py`
- `src/mv_audit/evaluation/hallucination_metrics.py`
- `src/mv_audit/evaluation/evaluate_all.py`
- `scripts/make_fake_predictions.py`

核心指标至少包含：

- `json_validity`
- `schema_compliance`
- `field_em`
- `risk_type_macro_f1`
- `audit_accuracy`
- `high_risk_miss_rate`
- `false_manual_review_rate`
- `evidence_support_rate`
- `hallucination_rate`
- `evidence_value_accuracy`
- `evidence_source_accuracy`
- `evidence_bbox_accuracy_strict`
- `evidence_bbox_accuracy_relaxed`

## 测试方式

- 用 `scripts/make_fake_predictions.py` 生成完美预测。
- 用 `scripts/make_fake_predictions.py` 生成故意错误预测。
- 运行 `evaluate_all.py` 分别评测两份预测。
- 完美预测指标应接近 1。
- 故意错误预测应触发 high-risk miss、hallucination、unsupported conclusion 和 bbox 错误。

## 完成定义

- parser 能处理 markdown `json` code fence。
- parser 能从多余自然语言中抽取最外层 JSON。
- JSON 不合法时，该样本 `json_validity=0`、`schema_compliance=0`，业务指标记 0，并记录 parse error。
- bbox strict 使用 IoU >= 0.5。
- bbox relaxed 使用 IoU >= 0.3 或中心点落入真实框邻域。
- `missing_document` 不要求缺失材料 bbox，但必须由 `document_complete=false` 和 reason 支持。
- `unreadable_image` 不要求不可读字段 bbox，但必须由 uncertainty 支持。
- 幻觉检测能识别不存在图片、错误凭证类型、缺失材料 bbox 和不可读字段编造。

## 下一阶段依赖

phase 07 依赖本阶段的 parser、schema validation、评测指标、error cases 导出和 fake prediction 验证结果。
