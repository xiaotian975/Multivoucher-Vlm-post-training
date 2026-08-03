# MultiVoucher-Audit Global Contracts

## 数据对象边界

每个样本以 case 为基本单位。一个 case 包含多张凭证图片、结构化 ground truth、字段级证据、文档级状态和 case-level 审核标签。

`primary_anomaly_type` 是数据生成分布控制字段。`anomaly_types` 是模型训练和评测使用的多标签异常列表。

`evidence_sufficient` 只属于 ground truth metadata，不属于模型输出 schema。它用于标签生成和评测判断，模型需要通过 `evidence`、`uncertainty`、`risk_level` 和 `audit_result` 体现证据是否充分。

## 模型输出 Schema

模型最终输出必须是单个 Evidence-Grounded JSON 对象，顶层字段固定为：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `case_id` | string | 当前报销 case ID |
| `field_extraction` | object | 从各凭证抽取的业务字段 |
| `consistency_check` | object | 审核检查项集合 |
| `anomaly_types` | array[string] | 多标签异常类型 |
| `risk_level` | string | 唯一风险等级 |
| `audit_result` | string | 唯一审核建议 |
| `reason` | string | 支持结论的简明解释 |
| `evidence` | array[object] | 值证据、来源证据、位置证据和文本证据 |
| `uncertainty` | object | 不确定字段和是否需要人工复核 |

模型不得输出：

- `primary_anomaly_type`
- `evidence_sufficient`

## Field Extraction

`field_extraction` 至少覆盖：

| 字段组 | 字段 |
| --- | --- |
| 金额 | `invoice_amount`、`payment_amount`、`reimbursement_amount`、`order_amount`、`tax_amount` |
| 商户 | `invoice_merchant`、`payment_merchant`、`order_merchant`、`merchant` |
| 人员 | `applicant`、`payer`、`order_user` |
| 日期 | `invoice_date`、`payment_date`、`order_date`、`application_date` |
| ID | `invoice_id`、`order_id`、`payment_id` |
| 类型 | `expense_type` |

金额保存为两位小数字符串。日期保存为 `YYYY-MM-DD`。商户字段同时保留不同凭证中的原始商户文本和归一化商户名。

## Consistency Check

`consistency_check` 不是狭义字段一致性集合，而是审核检查项集合，至少包含：

| 字段 | 含义 |
| --- | --- |
| `amount_consistent` | 发票、支付、报销、订单金额是否一致或合理 |
| `merchant_consistent` | 发票、支付、订单商户是否一致或属于可接受别名 |
| `person_consistent` | 申请人、支付人、订单用户是否一致 |
| `date_reasonable` | 订单、支付、发票、申请日期顺序是否合理 |
| `order_id_consistent` | 订单截图与报销申请单订单号是否一致 |
| `payment_id_present` | 支付流水号是否存在且格式合理 |
| `document_complete` | 发票、支付截图、报销单、订单截图是否完整 |
| `duplicate_in_batch` | 当前输入材料内部是否存在重复凭证 |

## Anomaly Types

允许异常类型固定为：

| 异常类型 | 含义 |
| --- | --- |
| `amount_mismatch` | 发票金额、支付金额、报销金额或订单金额不一致，但不属于超额报销 |
| `over_reimbursement` | 报销金额高于实际支付金额或发票金额 |
| `date_mismatch` | 订单日期、支付日期、发票日期、报销申请日期不合理 |
| `merchant_mismatch` | 发票商户与支付截图或订单截图商户不一致 |
| `applicant_mismatch` | 报销申请人与支付人或订单用户不一致 |
| `order_id_mismatch` | 订单号不一致 |
| `missing_document` | 缺少发票、支付截图、报销申请单或订单截图 |
| `duplicate_in_batch` | 当前输入材料中存在重复发票或重复订单截图 |
| `unreadable_image` | 图片模糊、遮挡、压缩严重，无法可靠识别 |

`amount_mismatch` 与 `over_reimbursement` 采用更具体异常优先原则。如果报销金额高于支付金额或发票金额，主异常设为 `over_reimbursement`，默认 `anomaly_types` 只标记 `["over_reimbursement"]`。只有存在独立额外金额冲突时，才允许两者同时出现。

## Risk Rule

`risk_level` 枚举固定为：

- `low`
- `medium`
- `high`

金额差异定义：

```text
amount_delta =
max(invoice_amount, payment_amount, reimbursement_amount, order_amount)
-
min(invoice_amount, payment_amount, reimbursement_amount, order_amount)

amount_delta_ratio = amount_delta / max(min_positive_amount, 1)
```

风险等级由程序规则生成。一个 case 只能有一个唯一 `risk_level`。如果同时触发多条规则，采用最高风险等级。

| 条件 | 风险等级 |
| --- | --- |
| 无异常，材料完整，图片清晰 | `low` |
| 普通金额不一致且差异 <= 100 元且比例 <= 5% | `medium` |
| 普通金额不一致且差异 > 100 元或比例 > 5% | `high` |
| 报销金额高于支付金额或发票金额，差异 <= 100 元且比例 <= 5% | `medium` |
| 报销金额高于支付金额或发票金额，差异 > 100 元或比例 > 5% | `high` |
| 支付日期与发票日期间隔超过 90 天 | `medium` |
| 发票日期早于订单日期超过 30 天 | `medium` |
| 报销申请日期早于支付日期 | `high` |
| 商户明显不一致 | `high` |
| 申请人、支付人、订单用户不一致 | `high` |
| 订单号不一致 | `high` |
| 缺少订单截图 | `medium` |
| 缺少发票、支付截图或报销申请单 | `high` |
| 当前 case 内存在重复发票或重复订单截图 | `high` |
| 不可读区域影响非核心字段 | `medium` |
| 不可读区域影响金额、商户、人员、订单号等核心字段 | `high` |

## Audit Result Rule

`audit_result` 枚举固定为：

- `pass`
- `manual_review`
- `missing_info`
- `reject_recommendation`

审核结果由程序化优先级规则生成。一个 case 只能有一个唯一 `audit_result`。

优先级固定为：

```text
missing_info > reject_recommendation > manual_review > pass
```

| 条件 | 审核结果 |
| --- | --- |
| 缺少发票、支付截图、报销申请单或订单截图 | `missing_info` |
| 高风险且 `evidence_sufficient=true` | `reject_recommendation` |
| 高风险且 `evidence_sufficient=false` | `manual_review` |
| 中风险异常、轻微金额差异、日期间隔异常、非核心字段不可读 | `manual_review` |
| 无异常，材料完整，图片清晰 | `pass` |

`risk_level` 和 `audit_result` 不是一一对应关系。high risk 可以对应 `reject_recommendation`、`missing_info` 或 `manual_review`。

## Evidence And BBox

每条 evidence 至少包含：

| 字段 | 说明 |
| --- | --- |
| `source_image_id` | 证据来自哪张输入图片 |
| `source_doc_type` | 证据来自哪类凭证 |
| `field` | 证据对应字段 |
| `value` | 字段值 |
| `bbox` | 归一化 bbox |
| `evidence_text` | 证据周围可读文本 |

模型输出 bbox 使用 `[x1, y1, x2, y2]`，坐标范围为 0 到 1000。渲染阶段同时保存绝对像素坐标 `bbox_abs` 和归一化坐标 `bbox_norm`。评测阶段把模型输出的归一化 bbox 转换回绝对坐标后计算 IoU。

严格 bbox 匹配使用 IoU >= 0.5。宽松 bbox 匹配使用 IoU >= 0.3 或预测框中心点落入真实框邻域。

## Missing And Unreadable Constraints

`missing_document`：

- 不要求缺失材料本身有 bbox。
- 不允许为缺失材料伪造 bbox。
- 输出中必须设置 `consistency_check.document_complete=false`。
- `reason` 必须说明缺少哪类材料。
- `audit_result` 优先为 `missing_info`。

`unreadable_image`：

- 不要求不可读字段有正确 bbox。
- 不允许为不可读字段编造确定值和 bbox。
- 输出中必须设置 `uncertainty.has_uncertain_fields=true`。
- `uncertain_fields` 必须列出无法可靠识别的字段。
- 如果核心字段不可读，通常应转人工复核。

## Data Split And Leakage

数据划分以 case 为单位，不以图片为单位。同一个 case 的发票、支付截图、报销单和订单截图不能拆到不同 split。

MV-SFT、MV-DPO 和 MV-GRPO 都只能从 MV-Train 派生。MV-Val 和 MV-Test 只用于验证和最终评测，不参与：

- DPO 数据构造。
- GRPO reward 调参。
- 错误样本回流训练。
- few-shot 示例选择。
- prompt 或规则调参时的反复窥探。

模板、商户、订单号和异常组合都应尽量隔离。`MV-Val-UnseenTemplate` 与 `MV-Test-UnseenTemplate` 的模板不能重叠。
