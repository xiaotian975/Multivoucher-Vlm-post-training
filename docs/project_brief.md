# MultiVoucher-Audit Project Brief

## 项目目标

MultiVoucher-Audit 是一个面向企业费用报销一致性审计的多模态后训练实验项目。项目基于 `Qwen3-VL-8B-Instruct`，让模型处理同一个报销 case 下的多张视觉凭证，并输出可验证、可评测、带证据约束的审核结论。

项目核心不是做普通发票 OCR，而是构建一个多图凭证一致性审计闭环。模型需要同时完成字段抽取、跨图一致性校验、异常类型识别、风险等级预测、审核建议生成、Evidence-Grounded JSON 输出、bbox 位置证据输出，以及在证据不足或图像不可读时主动转人工复核。

## 任务边界

第一版业务范围限定为企业费用报销一致性审计。标准完整 case 包含四类凭证：

| 凭证类型 | 主要信息 |
| --- | --- |
| `invoice` | 发票号码、开票日期、销售方、项目、金额、税额、价税合计 |
| `payment` | 支付金额、支付时间、收款方、付款人、支付流水号 |
| `reimbursement_form` | 申请人、费用类型、报销金额、申请日期、事由、订单号 |
| `order` | 订单号、商品或服务、商户、订单金额、订单用户、下单时间 |

模型需要比较金额、商户、人员、日期、订单号、支付流水号存在性、材料完整性、批内重复和图片可读性。`duplicate_in_batch` 只表示当前输入材料内部存在重复凭证，不涉及历史数据库查重。

## 核心输入输出

输入是一组同一报销 case 的多张凭证图片，以及提示词中给出的图片编号和凭证类型。图片顺序可以变化，但 prompt 必须明确每张图片的 `source_image_id` 和 `source_doc_type`，以便模型输出证据引用。

输出是统一的 Evidence-Grounded JSON，必须包含：

- `case_id`
- `field_extraction`
- `consistency_check`
- `anomaly_types`
- `risk_level`
- `audit_result`
- `reason`
- `evidence`
- `uncertainty`

模型不得输出 `primary_anomaly_type` 和 `evidence_sufficient`。这两个字段只属于 ground truth metadata 或数据生成控制信息。

## 训练链路

第一版完整训练链路如下：

```text
Qwen3-VL-8B-Instruct
        ↓
Zero-shot / Few-shot Baseline
        ↓
LoRA-SFT
        ↓
DPO
        ↓
小规模 GRPO
```

`LoRA-SFT` 负责让模型学习任务格式、字段抽取、跨图一致性判断和结构化输出。`DPO` 负责纠正风险偏好和证据偏好，降低高风险放行、无证据结论和不可读图片时编造行为。小规模 `GRPO` 使用 rule-based reward 强化 JSON 合法、高风险不放行、证据正确和不确定转人工等关键行为。

调试规模可以先跑通 2,000 到 5,000 cases、每类凭证 3 到 5 个模板、4 类核心异常和 LoRA-SFT 评测闭环。主实验再扩展到 30,000 cases、9 类异常、64 套模板、DPO 和小规模 GRPO。

## 第一版不做什么

第一版明确不做：

- 不做 IPO。
- 不使用 RLVR 表述。
- 不做历史发票重复报销检测、历史库查重、RAG 或外部检索系统。
- 不覆盖采购付款、售后理赔、物流单审核、合同审核等非企业费用报销一致性场景。
- 不让模型直接执行生产审批或自动驳回；模型输出 `reject_recommendation`，最终仍由人工确认。
- 不把验证集或测试集用于 DPO 数据构造、GRPO reward 调参或错误样本回流训练。
- 不在文档阶段实现训练、数据生成、图片渲染或评测代码。

## 待确认问题

- phase 04 使用的本地中文字体文件路径和授权方式需要后续确认。
- phase 01 的模型下载源优先级需要按机器环境确认：Hugging Face 或 ModelScope。
- 主实验规模是否严格采用 30,000/3,000/8,000 cases，还是长期保留 debug 规模后再扩容，需要后续确认。
- DPO rejected 中“人工手写少量高质量反例”的比例和是否真的人工参与，需要后续确认。
