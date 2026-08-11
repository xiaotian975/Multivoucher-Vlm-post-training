# MultiVoucher-Audit

更新时间：2026-08-11
本文是一份完整的工程项目报告，集中说明 MultiVoucher-Audit 的背景、任务定义、数据集、方法路线、创新点、Phase 00 到 Phase 08 的工程状态、训练状态、验证状态、测试状态、超参数、实验结果和失败原因。

## 0. 阅读说明

本文按项目工程文档中的阶段编号整理：`Phase 00` 到 `Phase 08`。这是一组编号阶段，不等同于“模型训练第 0/1/2/3 阶段”。其中：

- `Phase 00-06` 主要是工程、数据、渲染、训练数据格式和评测系统准备，不产生真实模型训练指标。
- `Phase 07` 是真实 LoRA-SFT 训练、验证 loss、M0/M1/M2 抽样推理与业务评测。
- `Phase 08` 已完成 DPO sample1000 训练和 M3 sample500 推理评测，但未完成正式 GRPO、M4 评测和完整 M2/M3/M4 对比。
- 服务器已经关闭。本文只基于本地已归档到 Git 的可信材料和项目文档，不复制模型权重、不复制全量 predictions、不复制原始训练日志。

## 1. 项目前后背景与研究动机

企业费用报销审核通常不是单张发票识别问题，而是多张凭证之间的一致性审计问题。一个报销 case 可能同时包含发票、支付截图、报销申请单和订单截图。真实审核中，系统或人工不仅要读出单张图片上的字段，还要判断多张图片之间金额是否一致、商户是否一致、申请人和支付人是否一致、订单号是否一致、日期是否合理、材料是否缺失、图片是否可读，以及是否存在批内重复凭证。

传统 OCR 或单图 VLM demo 通常只能解决“读出图片文字”或“回答图片问题”的一部分需求，很难完整覆盖以下业务要求：

- 多图联合理解：同一个 case 下需要同时处理 2 到 4 张凭证图片。
- 结构化输出：模型必须输出固定 schema 的 Evidence-Grounded JSON，而不是自由文本。
- 证据可追溯：审计结论必须带 `source_image_id`、`source_doc_type`、字段值、bbox 和 `evidence_text`。
- 风险可评测：输出必须能被程序自动计算 JSON Validity、Schema Compliance、Field EM、Audit Accuracy、High-risk Miss Rate、Evidence Support Rate、Hallucination Rate 等指标。
- 不确定性处理：材料缺失、核心字段不可读或证据不足时，模型应转人工复核或补材料，而不是编造确定结论。

因此，本项目把任务定义为“多凭证费用报销一致性审计”的 VLM 后训练实验，而不是普通 OCR、发票分类或单图问答。

### 1.1 为什么选择企业费用报销一致性审计

选择企业费用报销一致性审计作为 VLM 后训练场景，主要基于以下考虑：

| 原因 | 说明 |
| --- | --- |
| 业务价值明确 | 报销审核是企业财务、内控和合规中的高频场景，错误放行可能造成资金损失，过度拦截又会降低员工报销效率。 |
| 天然多模态 | 报销审核不仅看文本字段，还要看凭证图片、截图版式、字段位置、材料是否缺失、图片是否可读。 |
| 天然多图推理 | 单个 case 往往包含发票、支付截图、报销单和订单截图，需要跨图片比较金额、商户、人员、日期和订单号。 |
| 有清晰评价指标 | 字段抽取、JSON schema、风险等级、审核建议、证据支持、高风险漏检、幻觉和 bbox 都可以程序化评测。 |
| 适合构造可控合成数据 | 真实企业报销数据涉及隐私和合规，合成数据可以规避隐私问题，同时保留字段真值、异常标签和证据位置。 |
| 适合展示完整后训练流程 | 该任务可以完整覆盖数据构造、图片渲染、SFT、DPO、reward、评测和错误分析，适合作为 VLM post-training 工程项目。 |

这个场景比普通发票 OCR 更能体现 VLM 后训练价值：OCR 只要求“读出来”，而费用报销一致性审计要求模型“读出来、比对清楚、判断风险、给出证据、知道什么时候不能确定”。

### 1.2 传统方法通常怎么做

传统企业报销审核系统通常由以下几类方法组合而成：

| 传统方法 | 常见做法 | 优点 | 局限 |
| --- | --- | --- | --- |
| 人工审核 | 财务人员逐张查看发票、支付截图、报销单和订单截图 | 灵活、能处理复杂异常 | 成本高、效率低、标准不一致、难以规模化 |
| OCR + 表单抽取 | 用 OCR 识别图片文字，再用模板或规则抽取字段 | 对固定版式发票较有效 | 对截图、不同模板、模糊遮挡、跨图一致性弱 |
| 规则引擎 | 用固定规则判断金额差、日期间隔、材料缺失等 | 可解释、易审计 | 依赖字段抽取质量，难覆盖复杂组合异常 |
| RPA/流程自动化 | 自动下载附件、录入系统、触发审批流 | 能节省重复操作 | 不真正理解凭证内容，容易把错误字段传递到后续流程 |
| 传统机器学习分类器 | 基于 OCR 字段或人工特征训练异常分类模型 | 对结构化字段可建模 | 对图片证据、版式、不可读字段和跨图语义理解有限 |
| 财务 SaaS 内置校验 | 根据发票验真、预算、额度、重复提交等做校验 | 工程成熟、适合生产流程 | 通常依赖外部系统和结构化数据，难评价模型是否“看图有证据” |

简化来看，传统路线通常是：

```text
凭证图片
-> OCR
-> 字段抽取
-> 规则引擎/人工审核
-> 审批结论
```

这种路线更像“先把图片变成字段，再用规则判断”，而不是端到端地让模型基于多张图片和证据完成审计。

### 1.3 传统方法长期积累的问题

传统方法在企业费用报销一致性审计中长期存在一些难点：

| 问题 | 具体表现 |
| --- | --- |
| 多图关系难建模 | 发票、支付、报销单、订单之间的金额、商户、人员和订单号关系分散在多张图片中，传统 OCR pipeline 容易只做单图抽取。 |
| 模板和版式脆弱 | 一旦凭证截图来源、字体、排版、压缩方式变化，基于模板的字段抽取容易失效。 |
| 证据链不完整 | 传统系统可能给出“异常/通过”，但很难同时输出字段值、来源图片、bbox 和证据文本。 |
| 缺材料和不可读难处理 | 模糊、遮挡、缺少订单或缺少支付截图时，系统容易继续给出确定结论，或者只能粗暴转人工。 |
| 高风险漏检与误拦截难平衡 | 规则太松会漏放高风险，规则太严会制造大量人工复核。 |
| 数据隐私导致训练样本难积累 | 真实报销数据包含人员、商户、金额和企业流程信息，难以公开或大规模标注。 |
| 评测闭环不完整 | 很多系统只看 OCR 准确率或审批通过率，缺少对 JSON 合法性、证据支持、幻觉和高风险漏检的统一指标。 |
| 错误难归因 | 出错时难判断是 OCR 错、字段映射错、规则错、跨图推理错，还是证据不足。 |

这些问题的共同点是：传统方法通常把“看图”“抽字段”“做判断”“给证据”“处理不确定性”拆成多个松耦合模块，每个模块局部可控，但整体审计行为很难用统一目标训练和评测。

### 1.4 本项目采用的方法

本项目采用的是“合成数据 + 多图 VLM + LoRA-SFT/DPO/GRPO 后训练 + Evidence-Grounded JSON 评测”的路线：

```text
可控合成 case
-> 异常注入与风险规则
-> 多凭证图片渲染与 bbox 标注
-> 构造 SFT / DPO / GRPO 数据
-> Qwen3-VL-8B-Instruct 后训练
-> Evidence-Grounded JSON 输出
-> 多维业务指标评测与 error cases 诊断
```

具体做法：

| 模块 | 本项目做法 |
| --- | --- |
| 数据来源 | 用程序生成可控的企业报销 case，避免真实隐私数据问题。 |
| 图片生成 | 将结构化 case 渲染成发票、支付截图、报销单、订单截图，并记录字段 bbox。 |
| 异常构造 | 注入金额不一致、超额报销、日期异常、商户异常、人员异常、订单号异常、缺材料、批内重复和不可读图片。 |
| 风险标签 | 用统一 risk rule engine 生成唯一 `risk_level` 和 `audit_result`。 |
| 模型训练 | 用 LoRA-SFT 学习结构化审计输出，用 DPO 尝试优化风险偏好和证据偏好，GRPO 作为后续 planned reward route。 |
| 输出形式 | 强制输出 Evidence-Grounded JSON，包含字段、审计结论、证据、bbox 和不确定性。 |
| 评测方式 | 同时评测 schema、字段、审计、证据、bbox、幻觉、高风险漏检和错误样本。 |

### 1.5 相比传统方法的创新

相比传统 OCR + 规则引擎，本项目的创新主要体现在：

| 对比维度 | 传统方法 | 本项目方法 | 创新点 |
| --- | --- | --- | --- |
| 输入粒度 | 单张凭证或 OCR 后字段 | 一个 case 下多张凭证图片 | 从单图识别转为多图 case-level 审计 |
| 输出形式 | 字段表、规则命中、审批结果 | Evidence-Grounded JSON | 结构化结论和证据链同时输出 |
| 证据定位 | 通常缺少 bbox 级证据 | 输出 `source_image_id`、`source_doc_type`、bbox、evidence text | 审计结果可追溯、可评测 |
| 数据来源 | 依赖真实业务数据或人工标注 | 可控合成 case + 渲染图片 + 自动 bbox | 避开隐私限制，同时保留真值 |
| 训练方式 | 规则或传统分类器为主 | Qwen3-VL + LoRA-SFT + DPO/GRPO 设计 | 把 VLM 后训练引入多凭证审计 |
| 评测方式 | OCR 准确率、人工抽检、审批率 | JSON/schema/field/audit/evidence/bbox/hallucination 多指标 | 能定位模型到底错在哪里 |
| 不确定性 | 多依赖人工兜底 | schema 中显式要求 uncertainty | 让模型学习“不确定时不编造” |

更准确地说，本项目不是替代所有传统财务系统，而是在传统系统最薄弱的“多图语义理解、证据定位和可评测审计输出”环节引入 VLM 后训练能力。

### 1.6 本方法解决了传统问题中的哪些

| 传统问题 | 本项目是否解决 | 如何解决 | 当前证据 |
| --- | --- | --- | --- |
| 多图关系难建模 | 部分解决 | 以 case 为单位输入多张图片，让模型学习跨图字段一致性 | M2 Audit Accuracy 平均约 `0.774`，显著高于 M0/M1 |
| 模板和版式脆弱 | 部分解决 | 构造多模板、robust 和 unseen template 测试集 | M2 在 `test_unseen_template` 上 JSON Validity 为 `1.000`，Schema Compliance 约 `0.864` |
| 证据链不完整 | 明显改善 | 输出 evidence、bbox、source image 和 source doc type | M2 Evidence Support Rate 平均约 `0.804` |
| 评测闭环不完整 | 明显解决 | 建立 JSON/schema/field/audit/evidence/bbox/hallucination 指标 | Phase 07/08 均已生成 metrics summary 和 error cases |
| 数据隐私和标注困难 | 工程上解决 | 使用合成数据自动生成字段真值、风险标签和 bbox | main 配置为 `41000` cases，完整规模约 16 万级图片 |
| 缺材料/不可读处理 | 部分解决 | output schema 设计 `uncertainty`，risk/audit rule 包含 `missing_info` 和 `manual_review` | 训练和评测链路已覆盖，但仍需更细业务指标分析 |
| 高风险漏检 | 尚未充分解决 | SFT 有改善，DPO 尝试进一步降低 | M2 High-risk Miss Rate 仍约 `0.243`，M3 只降到约 `0.237` |
| DPO 偏好优化可靠性 | 未解决，已暴露问题 | 用 M2/M3 对比发现 reward-data mismatch | M3 Audit Accuracy 从 `0.773` 降到 `0.668` |

因此，当前项目已经证明：VLM + LoRA-SFT 可以显著提升多凭证审计的结构化输出和证据支持能力；但也暴露出：简单 DPO pair 和过强偏好训练不一定能改善业务指标，甚至可能破坏 SFT 已学到的审计边界。

## 2. 任务定义

### 2.1 输入

模型输入是一组同一报销 case 的多张凭证图片，以及 prompt 中明确给出的图片编号和凭证类型。标准完整 case 包含四类凭证：

| 凭证类型 | 主要信息 | 作用 |
| --- | --- | --- |
| `invoice` | 发票号码、开票日期、销售方、项目、金额、税额、价税合计 | 判断发票金额、商户和日期 |
| `payment` | 支付金额、支付时间、收款方、付款人、支付流水号 | 判断实际支付金额、支付人和收款方 |
| `reimbursement_form` | 申请人、费用类型、报销金额、申请日期、事由、订单号 | 判断报销请求本身是否合理 |
| `order` | 订单号、商品或服务、商户、订单金额、订单用户、下单时间 | 判断订单与报销/支付/发票是否一致 |

### 2.2 输出

模型必须输出单个 Evidence-Grounded JSON 对象，顶层字段固定为：

| 字段 | 含义 |
| --- | --- |
| `case_id` | 当前报销 case ID |
| `field_extraction` | 从多张凭证中抽取出的金额、商户、人员、日期、ID、费用类型等字段 |
| `consistency_check` | 金额、商户、人员、日期、订单号、支付流水号、材料完整性和批内重复检查 |
| `anomaly_types` | 多标签异常类型 |
| `risk_level` | 唯一风险等级：`low`、`medium`、`high` |
| `audit_result` | 唯一审核建议：`pass`、`manual_review`、`missing_info`、`reject_recommendation` |
| `reason` | 支持审核结论的简明解释 |
| `evidence` | 字段值、来源图片、来源凭证、bbox 和证据文本 |
| `uncertainty` | 不确定字段和是否需要人工复核 |

模型不得输出 `primary_anomaly_type` 和 `evidence_sufficient`。这两个字段只属于数据生成和 ground truth metadata，不属于模型最终输出。

### 2.3 需要模型学会的能力

| 能力 | 说明 | 对应评测指标 |
| --- | --- | --- |
| JSON 格式遵守 | 输出必须是可解析 JSON，且符合 schema | JSON Validity、Schema Compliance |
| 字段抽取 | 抽取金额、商户、人员、日期、订单号、支付流水号等 | Field EM |
| 跨图一致性审计 | 判断不同凭证间字段是否一致或合理 | Risk Type Macro-F1、Audit Accuracy |
| 高风险不放行 | 高风险样本不能错误输出 `pass` 或错误降级 | High-risk Miss Rate |
| 证据定位 | 给出支持结论的图片、字段、bbox 和文本 | Evidence Support Rate、BBox Accuracy |
| 幻觉抑制 | 不引用不存在图片、不为缺失/不可读字段编造证据 | Hallucination Rate |
| 不确定性处理 | 缺材料或不可读时转补材料/人工复核 | False Manual Review Rate、uncertainty 相关检查 |

### 2.4 第一版明确不做

- 不做历史库查重、RAG 或外部检索。
- 不做采购付款、售后理赔、物流单、合同审核等其他业务场景。
- 不让模型直接执行生产审批；`reject_recommendation` 只是拒绝建议，最终仍应由人工确认。
- 不把 Val/Test 用于 DPO 数据构造、GRPO reward 调参或错误样本回流训练。

## 3. 数据集说明

### 3.1 数据集名称与性质

本项目的数据集可以称为 `MultiVoucher-Audit`。它是一个面向多凭证费用报销审计的合成数据集，而不是来自真实企业财务系统的原始隐私数据。

数据集采用程序化合成路线：

```text
字典和规则
-> 正常结构化 base cases
-> 异常注入
-> 风险等级和审核结果规则生成
-> case-level split
-> 四类凭证图片渲染
-> bbox 标注记录
-> SFT/DPO/GRPO 训练格式转换
-> 评测 ground truth
```

合成数据的价值在于：每个 case 都有可控的字段真值、异常真值、风险标签、审计标签和 bbox 证据，因此可以自动评测模型是否真的按照证据完成审计。局限在于：模板、语言风格、视觉噪声、商户/人员分布仍可能与真实企业数据存在域差距。

### 3.2 主实验数据规模

`configs/data_gen/main.yaml` 中的主实验配置如下：

| Split | Case 数 |
| --- | ---: |
| train | 30000 |
| val_in_template | 2000 |
| val_unseen_template | 1000 |
| test_clean | 2000 |
| test_robust | 2000 |
| test_unseen_template | 2000 |
| test_hard_negative | 2000 |
| 合计 | 41000 |

一个完整 case 最多包含 4 类凭证图片，因此完整 main 规模对应约 16 万级图片。实际训练时，由于服务器上存在一部分图片未完整上传或生成完成，本次 LoRA-SFT 使用的是 existing-images 子集：训练 `21682` 条，验证 `1138` 条。

### 3.3 异常类型与分布

主实验配置中的异常分布：

| `primary_anomaly_type` | 比例 | 含义 |
| --- | ---: | --- |
| `none` | 0.30 | 正常样本 |
| `amount_mismatch` | 0.12 | 发票/支付/报销/订单金额不一致 |
| `over_reimbursement` | 0.08 | 报销金额高于实际支付或发票金额 |
| `date_mismatch` | 0.08 | 日期顺序或间隔异常 |
| `merchant_mismatch` | 0.08 | 发票、支付、订单商户不一致 |
| `applicant_mismatch` | 0.08 | 申请人、支付人、订单用户不一致 |
| `order_id_mismatch` | 0.08 | 订单号不一致 |
| `missing_document` | 0.08 | 缺少必要凭证 |
| `duplicate_in_batch` | 0.05 | 当前输入材料内部有重复凭证 |
| `unreadable_image` | 0.05 | 图片模糊、遮挡、压缩严重或核心字段不可读 |

### 3.4 训练数据形态

| 数据形态 | 作用 | 来源 |
| --- | --- | --- |
| SFT JSONL | 教模型按固定 schema 输出完整审计 JSON | MV-Train |
| DPO pairs | 用 chosen/rejected 偏好样本纠正风险偏好和证据偏好 | MV-Train |
| GRPO prompts | 用 rule-based reward 做小规模强化 | MV-Train |
| eval sets | 评测 M0/M1/M2/M3/M4 的业务指标 | MV-Val/MV-Test |
| error cases | 保存模型失败样本，供诊断和后续报告分析 | 评测输出 |

## 4. 方法路线与预期目标

### 4.1 方法路线

第一版完整训练路线设计为：

```text
Qwen3-VL-8B-Instruct
-> M0 zero-shot baseline
-> M1 few-shot baseline
-> M2 LoRA-SFT
-> M3 LoRA-SFT + DPO
-> M4 LoRA-SFT + DPO + GRPO
```

各阶段作用：

| 方法 | 预期作用 |
| --- | --- |
| M0 zero-shot | 检查基座模型在无任务训练时的自然能力 |
| M1 few-shot | 检查少量示例能否让模型遵守 schema 和任务格式 |
| LoRA-SFT | 学会输出格式、字段抽取、跨图一致性判断、证据引用和基础审计规则 |
| DPO | 降低高风险放行、无证据结论、不可读图片编造等偏好问题 |
| GRPO | 用 rule-based reward 强化 JSON 合法、高风险不放行、证据正确和不确定转人工 |

### 4.2 预期目标

理想情况下，M2 相比 M0/M1 应显著提升：

- JSON Validity
- Schema Compliance
- Field EM
- Audit Accuracy
- Evidence Support Rate
- Evidence BBox Accuracy

M3/M4 相比 M2 应进一步改善：

- High-risk Miss Rate 下降。
- Hallucination Rate 不上升。
- Evidence Support Rate 上升或至少不下降。
- False Manual Review Rate 不明显恶化。
- Audit Accuracy 不能明显低于 M2。

### 4.3 当前实际结果与预期差距

当前结果是“半成功、半失败”：

- M2 符合预期：LoRA-SFT 显著提升格式、字段、证据和审计能力。
- M3 不符合预期：DPO 训练过程收敛，但业务指标没有提升，Audit Accuracy 反而明显下降。
- M4 未完成：GRPO 只有 smoke 级别产物，不能作为正式实验结果。

## 5. 创新点与可包装贡献

当前项目可以包装的创新点主要在“任务设计、数据构造、评测闭环和后训练诊断”，而不是单纯提出新的模型结构。

| 创新点 | 说明 |
| --- | --- |
| 多凭证一致性审计任务 | 不是单图 OCR，而是同一报销 case 下多张凭证的跨图一致性审核 |
| Evidence-Grounded JSON 输出 | 模型输出必须同时包含结构化字段、审计结论、证据来源、bbox 和不确定性 |
| 可控合成数据闭环 | 从字段真值、异常注入、风险规则到图像渲染和 bbox 标注均可程序化生成 |
| case-level split 防泄漏 | 同一 case 的多张图片不会拆到不同 split，避免图片级泄漏 |
| 多维业务指标评测 | 同时评估 JSON/schema、字段、风险、审核建议、证据、bbox、幻觉和高风险漏检 |
| 后训练负结果诊断 | 不只报告 DPO loss，而是用 M2/M3 sample500 对比证明 DPO 出现 reward-data mismatch |
| 可扩展到 DPO v2/GRPO | 当前失败诊断已经指出 hard rejected、保护型 pair、holdout 和保守 DPO 的下一步方向 |

需要注意：如果面向 HR 或项目展示，“创新点”应表达为工程和实验闭环创新，而不是声称提出了全新的 DPO/GRPO 算法。

## 6. 当前核心结论

LoRA-SFT 的 M2 明显成功：模型已经能稳定输出合法 JSON，并显著提升字段抽取、审计准确率和证据支持率。但 DPO 后的 M3 没有达到预期：DPO loss 虽然收敛、preference margin 很大，但业务评测中 Audit Accuracy 从 M2 的约 `0.773` 降到 M3 的约 `0.668`，High-risk Miss Rate 只从 `0.243` 小幅降到 `0.237`，因此当前 DPO 应判定为 negative result，不建议继续扩大 GRPO。

## 7. 项目目标与模型编号

MultiVoucher-Audit 是一个多凭证视觉语言模型后训练实验项目，目标是让 VLM 基于发票、付款凭证、报销单、订单等多张凭证图片完成结构化审计输出，包括字段抽取、跨凭证一致性判断、风险等级、审计结论和证据定位。

核心闭环如下：

```text
case schema
-> base cases
-> anomaly injection
-> case-level split
-> voucher rendering + bbox annotations
-> SFT/DPO/GRPO data
-> evaluation
-> LoRA-SFT / DPO / GRPO post-training experiments
```

模型编号：

| 编号 | 方法 | 当前状态 |
| --- | --- | --- |
| M0 | `Qwen3-VL-8B-Instruct` zero-shot | Phase 07 sample500 已评测 |
| M1 | `Qwen3-VL-8B-Instruct` few-shot | Phase 07 sample500 已评测 |
| M2 | `Qwen3-VL-8B-Instruct + LoRA-SFT` | SFT 已训练，val loss 已验证，sample500 已评测 |
| M3 | `Qwen3-VL-8B-Instruct + LoRA-SFT + DPO` | DPO 已训练，sample500 已评测，但业务指标失败 |
| M4 | `Qwen3-VL-Instruct + LoRA-SFT + DPO + GRPO` | 未正式完成，仅有 GRPO smoke 级别产物 |

### 7.1 目录结构

| 路径 | 内容 |
| --- | --- |
| `docs/` | 项目 brief、全局契约、执行路线、phase 文档和实验归档 |
| `configs/` | 数据生成、模型、schema、训练和评测配置 |
| `src/mv_audit/` | Python 包，包含数据生成、渲染、扰动、转换、评测、推理和训练模块 |
| `data/mv_audit/` | 字典、raw cases、图片、bbox annotations、SFT/DPO/GRPO 数据 |
| `scripts/` | 环境准备、数据生成、图片渲染、训练数据构造、训练和评估入口 |
| `outputs/` | 日志、评估报告、bbox 可视化样本、预测和 checkpoint 输出位置 |
| `models/` | 本地模型权重目录，约定为 `models/Qwen3-VL-8B-Instruct` |

### 7.2 快速运行入口

安装为 editable package：

```bash
pip install -e .
```

运行最小 debug pipeline：

```bash
bash scripts/run_debug_pipeline.sh
```

Windows PowerShell 可以运行：

```powershell
.\scripts\run_debug_pipeline.ps1
```

生成 debug cases、注入异常并划分 split：

```bash
bash scripts/01_generate_cases.sh
```

渲染 debug 图片并生成 bbox 标注和可视化样本：

```bash
bash scripts/02_render_images.sh
```

构造 debug 训练数据并运行 fake prediction 评估：

```bash
bash scripts/03_build_train_data.sh
bash scripts/08_evaluate.sh
```

main 规模数据入口：

```bash
bash scripts/01_generate_main_cases.sh
bash scripts/02_render_main_images.sh
bash scripts/03_build_main_train_data.sh
```

### 7.3 Qwen3-VL Smoke Test

准备环境和模型：

```bash
bash scripts/00_prepare_env.sh
bash scripts/00_download_qwen3vl.sh
```

需要走 ModelScope 时：

```bash
USE_MODELSCOPE=1 bash scripts/00_download_qwen3vl.sh
```

单图 smoke test：

```bash
python -m mv_audit.inference.qwen3vl_smoke_test \
  --config configs/model/qwen3vl_8b.yaml \
  --image examples/test_invoice.png \
  --output outputs/logs/qwen3vl_smoke_test.log
```

多图 smoke test：

```bash
python -m mv_audit.inference.qwen3vl_multi_image_test \
  --config configs/model/qwen3vl_8b.yaml \
  --images examples/invoice.png examples/payment.png examples/reimbursement_form.png examples/order.png \
  --output outputs/logs/qwen3vl_multi_image_test.log
```

smoke test 需要本地或可下载的模型权重、足够显存，以及实际存在的测试图片路径。README 中的 `examples/*.png` 只是命令示例，运行前需要替换成真实图片路径。

## 8. Phase 00-08 总状态

| Phase | 目标 | 当前状态 | 训练状态 | 验证/测试状态 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 00 | 项目骨架、目录、基础工具 | 已完成工程基础 | 不涉及训练 | Python 包结构和基础工具用于后续阶段 | 无真实模型指标 |
| 01 | Qwen3-VL 单图/多图推理验证 | 已实现模型配置、下载脚本和 smoke test 入口 | 不训练 | smoke test 用于验证模型加载和多图输入 | 不评价业务准确率 |
| 02 | case schema、字典、正常交易真值表 | 已建立数据生成基础 | 不训练 | schema、金额、日期、枚举等校验 | 为异常注入准备 |
| 03 | 异常注入、risk rule engine、case split | 已定义 10 类 `primary_anomaly_type` 和风险规则 | 不训练 | 检查异常分布、risk_level、audit_result、case_id 无重复 | Train/Val/Test 按 case 划分 |
| 04 | 凭证图片渲染、bbox 记录、视觉扰动 | 已建立渲染、bbox、扰动模块 | 不训练 | bbox 可视化抽检是验收重点 | 坐标采用 0-1000 归一化 |
| 05 | SFT/DPO/GRPO 数据格式构造 | 已有 SFT、DPO、GRPO converter | 不训练 | 验证 SFT JSON、DPO 同 prompt、GRPO reward 字段 | Val/Test 不进 DPO/GRPO |
| 06 | JSON parser、bbox evaluator、基础评测 | 已有评测模块和 fake prediction 验证 | 不训练 | 完美预测/故意错误预测用于验证 evaluator 行为 | fake 指标不等于真实模型结果 |
| 07 | LoRA-SFT、M0/M1/M2 推理与评测 | 已完成 sample500 评测并归档 | M2 SFT 已完成 | val loss 已记录；M0/M1/M2 四个 split 均已测试 | SFT 明显有效，但高风险漏检仍偏高 |
| 08 | DPO、小规模 GRPO、M2/M3/M4 对比 | DPO 与 M3 sample500 已完成；GRPO/M4 未完成 | DPO 已完成；GRPO 未正式完成 | M3 sample500 已测试；M4 未测试 | DPO 业务指标失败 |

## 9. 数据与切分

### 9.1 主实验规划

项目 brief 中的主实验目标规模是 `30,000/3,000/8,000 cases`，分别对应 Train/Val/Test 级别的数据规模规划。主实验包含 9 类异常、64 套模板、DPO 和小规模 GRPO。

### 9.2 本次 SFT 实际训练数据

本次服务器训练使用的是 existing-images 子集，因此少于早期规划中的完整 main 规模：

| 数据 | 路径 | 条数 | 用途 |
| --- | --- | ---: | --- |
| SFT train | `data/mv_audit/sft_main/train_existing_images.jsonl` | 21682 | LoRA-SFT 训练 |
| SFT val | `data/mv_audit/sft_main/val_existing_images.jsonl` | 1138 | LoRA-SFT loss 验证 |

### 9.3 Phase 07/08 抽样评测数据

Phase 07 和 Phase 08 的业务评测采用 sample500 方案：

- 4 个测试 split：`test_clean`、`test_robust`、`test_unseen_template`、`test_hard_negative`。
- 每个 split 抽样 500 条。
- Phase 07：`M0/M1/M2 × 4 splits × 500 = 6000` 条预测。
- Phase 08 M3：`M3 × 4 splits × 500 = 2000` 条预测。
- 抽样 ground truth 路径：`data/mv_audit/eval_sets_phase07_sample500/`。
- 抽样 manifest 路径：`data/mv_audit/eval_sets_phase07_sample500/manifests/`。

### 9.4 数据边界

重要约束：

- MV-SFT、MV-DPO、MV-GRPO 都只能从 MV-Train 派生。
- MV-Val 和 MV-Test 只用于验证和最终评测。
- Val/Test 不参与 DPO 数据构造、GRPO reward 调参、错误样本回流、few-shot 示例选择或 prompt 规则调参。

## 10. Phase 07：LoRA-SFT 训练与验证

### 10.1 SFT 配置

| 项目 | 值 |
| --- | --- |
| 基座模型 | `Qwen3-VL-8B-Instruct` |
| 训练配置 | `configs/train/sft_lora_qwen3vl_8b_phase07_server.yaml` |
| SFT adapter | `outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/` |
| train file | `data/mv_audit/sft_main/train_existing_images.jsonl` |
| val file | `data/mv_audit/sft_main/val_existing_images.jsonl` |
| output schema | `configs/schema/output_schema.json` |
| ground truth dir | `data/mv_audit/eval_sets_main` |

核心训练超参数：

| 超参数 | 值 |
| --- | ---: |
| `num_train_epochs` | 1 |
| `learning_rate` | `1.0e-4` |
| `per_device_train_batch_size` | 1 |
| `per_device_eval_batch_size` | 1 |
| `gradient_accumulation_steps` | 16 |
| `bf16` | true |
| `fp16` | false |
| `gradient_checkpointing` | true |
| `logging_steps` | 10 |
| `save_steps` | 500 |
| `eval_steps` | 500 |
| `eval_strategy` | steps |
| `save_total_limit` | 2 |

LoRA 超参数：

| 项目 | 值 |
| --- | --- |
| `r` | 16 |
| `alpha` | 32 |
| `dropout` | 0.05 |
| target modules | `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` |

推理相关配置：

| 项目 | 值 |
| --- | ---: |
| `max_new_tokens` | 1024 |
| `temperature` | 0.0 |
| `top_p` | 0.9 |
| `image_max_pixels` | 262144 |
| `flush_every` | 1 |
| `few_shot_count` | 2 |

### 10.2 SFT 训练结果

| 指标 | 值 |
| --- | ---: |
| 最终 `global_step` | 1356 |
| 最终 epoch | 1.0 |
| 训练耗时 | 67437.9456s，约 18.73h |
| 最终 `train_loss` | 0.004982630206586825 |
| checkpoint | `checkpoint-1000`、`checkpoint-1356`、最终 adapter 均存在 |

### 10.3 SFT 验证集 loss

| 记录 | epoch | eval_loss |
| --- | ---: | ---: |
| eval 1 | 0.37 | 0.00028587671113200486 |
| eval 2 | 0.74 | 0.00010228458268102258 |

解释：SFT 已完成训练，并且做过 validation loss eval。但 validation loss 只说明模型在 teacher-forcing 条件下对验证数据的拟合情况，不能替代业务指标评测；因此仍需要 Phase 07 的 M0/M1/M2 sample500 测试。

## 11. Phase 07：M0/M1/M2 Sample500 测试结果

### 11.1 评测设置

| 项目 | 值 |
| --- | --- |
| 评测目录 | `docs/experiments/phase07_sample500/` |
| 服务器 predictions | `outputs/predictions/phase07_sample500/` |
| 服务器 reports | `outputs/eval_reports/phase07_sample500/` |
| 评测规模 | `M0/M1/M2 × 4 splits × 500 = 6000` |
| 模型 | `m0_zero_shot`、`m1_few_shot`、`m2_sft` |
| splits | `test_clean`、`test_robust`、`test_unseen_template`、`test_hard_negative` |

归档材料：

- `docs/experiments/phase07_sample500/metrics_summary.csv`
- `docs/experiments/phase07_sample500/metrics_by_model.csv`
- `docs/experiments/phase07_sample500/error_cases/`
- `docs/experiments/phase07_sample500/figures/`
- `docs/experiments/phase07_sample500/phase07_sample500_report.md`

### 11.2 M0/M1/M2 平均指标

| Model | JSON Validity | Schema Compliance | Field EM | Risk Type Macro-F1 | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | BBox Acc Relaxed | Error Cases Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 zero-shot | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 500.0 |
| M1 few-shot | 1.000 | 0.116 | 0.114 | 0.022 | 0.079 | 0.999 | 0.025 | 0.575 | 0.010 | 500.0 |
| M2 LoRA-SFT | 1.000 | 0.877 | 0.876 | 0.743 | 0.774 | 0.243 | 0.804 | 0.001 | 0.795 | 164.5 |

### 11.3 M2 分 split 指标

| Split | JSON Validity | Schema Compliance | Field EM | Risk Type Macro-F1 | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | BBox Acc Relaxed | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.832 | 0.831 | 0.780 | 0.744 | 0.270 | 0.747 | 0.000 | 0.737 | 173 |
| test_robust | 1.000 | 0.810 | 0.809 | 0.798 | 0.742 | 0.276 | 0.727 | 0.002 | 0.723 | 176 |
| test_unseen_template | 1.000 | 0.864 | 0.863 | 0.754 | 0.742 | 0.274 | 0.775 | 0.000 | 0.764 | 176 |
| test_hard_negative | 1.000 | 1.000 | 1.000 | 0.639 | 0.866 | 0.152 | 0.965 | 0.001 | 0.954 | 133 |

### 11.4 Phase 07 结论

SFT 是有效的：

- M2 的 JSON Validity 在四个 split 上都是 `1.000`。
- M2 的 Schema Compliance 从 M0/M1 的极低水平提升到平均 `0.877`。
- M2 的 Field EM 达到平均 `0.876`。
- M2 的 Audit Accuracy 达到平均 `0.774`。
- M2 的 Evidence Support Rate 达到平均 `0.804`。
- M2 的 Hallucination Rate 平均仅约 `0.001`。

但 SFT 仍有明显短板：

- M2 的 High-risk Miss Rate 平均仍为 `0.243`。
- `test_clean/test_robust/test_unseen_template` 上高风险漏检率约 `0.270-0.276`。
- 这说明模型虽然学会了格式、字段和证据，但“高风险不放行”的业务边界仍不够稳定。

因此 Phase 07 支持进入 Phase 08 做 DPO 或 reward-based 优化，但后续优化必须以业务指标为准，不能只看训练 loss。

## 12. Phase 08：DPO Sample1000 训练

### 12.1 DPO 配置

| 项目 | 值 |
| --- | --- |
| 基座模型 | `Qwen3-VL-8B-Instruct` |
| 初始 adapter | `outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/` |
| DPO 输出 adapter | `outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1/` |
| 训练配置 | `configs/train/dpo_qwen3vl_8b.yaml` |
| 训练数据 | `data/mv_audit/dpo_main/pairs_train.jsonl` |
| 数据来源 | MV-Train |
| `require_existing_images` | true |
| 本次运行 | `max_samples=1000` |
| 因缺少可用图片跳过 | 324 pairs |

核心 DPO 超参数：

| 超参数 | 值 |
| --- | ---: |
| `learning_rate` | `5.0e-6` |
| `num_train_epochs` | 1 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 16 |
| `gradient_checkpointing` | true |
| `beta` | 0.1 |
| `bf16` | true |
| `logging_steps` | 10 |
| `save_steps` | 500 |
| `save_total_limit` | 2 |

### 12.2 DPO Reward 审计

| Metric | Value |
| --- | ---: |
| examples | 1000 |
| skipped_missing_images | 324 |
| global_step | 63 |
| history_len | 63 |
| chosen_mean_reward | 1.000000 |
| rejected_mean_reward | 0.460112 |
| mean_reward_gap | 0.539888 |
| positive_reward_gap_rate | 0.844000 |
| rejected_json_valid_rate | 0.896000 |
| rejected_high_risk_miss_rate | 0.155000 |
| rejected_hallucination_penalty | 0.015506 |

### 12.3 DPO 训练动态

| Item | First Step | Last Step |
| --- | ---: | ---: |
| global_step | 1.0 | 63.0 |
| loss | 0.6877865195274353 | 0.0005680027534253895 |
| chosen_logp | -0.0008737894240766764 | -0.007150840014219284 |
| rejected_logp | -50.337337493896484 | -128.99093627929688 |
| preference_margin | 0.10750198364257812 | 74.73100280761719 |

DPO 训练过程本身看起来“收敛”：loss 很低，chosen/rejected 的 preference margin 被拉得很大。但这只能说明模型学会了当前 DPO pairs 的偏好区分，不能说明真实业务指标提升。

## 13. Phase 08：M3 Sample500 推理与测试

### 13.1 评测设置

| 项目 | 值 |
| --- | --- |
| M3 定义 | `Qwen3-VL-8B-Instruct + Phase 07 SFT adapter + DPO adapter` |
| 评测配置 | `configs/train/phase08_m3_sample500_server.yaml` |
| SFT adapter | `outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/` |
| DPO adapter | `outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1/` |
| predictions | `outputs/predictions/phase08_m3_sample500/` |
| eval reports | `outputs/eval_reports/phase08_m3_sample500/` |
| 本地归档 | `docs/experiments/phase08_m3_sample500/` |
| 推理方式 | 8 卡 shard 数据并行 |

M3 merge 校验结果：

| Split | Rows | Duplicates Before Dedup | Missing |
| --- | ---: | ---: | ---: |
| test_clean | 500 | 0 | 0 |
| test_robust | 500 | 0 | 0 |
| test_unseen_template | 500 | 0 | 0 |
| test_hard_negative | 500 | 0 | 0 |

### 13.2 M3 分 split 指标

| Split | JSON Validity | Schema Compliance | Field EM | Risk Type Macro-F1 | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | BBox Acc Relaxed | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.808 | 0.807 | 0.806 | 0.650 | 0.264 | 0.727 | 0.002 | 0.716 | 227 |
| test_robust | 1.000 | 0.804 | 0.803 | 0.815 | 0.652 | 0.264 | 0.724 | 0.002 | 0.722 | 203 |
| test_unseen_template | 1.000 | 0.868 | 0.867 | 0.750 | 0.634 | 0.276 | 0.778 | 0.001 | 0.766 | 231 |
| test_hard_negative | 1.000 | 1.000 | 1.000 | 0.641 | 0.738 | 0.145 | 0.966 | 0.001 | 0.957 | 183 |

### 13.3 M2 与 M3 平均指标对比

| Model | JSON Validity | Schema Compliance | Field EM | Risk Type Macro-F1 | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | BBox Acc Relaxed | Error Cases Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 LoRA-SFT | 1.000 | 0.877 | 0.876 | 0.743 | 0.773 | 0.243 | 0.804 | 0.001 | 0.795 | 164.5 |
| M3 SFT+DPO | 1.000 | 0.870 | 0.869 | 0.753 | 0.668 | 0.237 | 0.799 | 0.001 | 0.790 | 211.0 |
| Delta M3-M2 | 0.000 | -0.007 | -0.007 | +0.010 | -0.105 | -0.005 | -0.005 | +0.001 | -0.004 | +46.5 |

### 13.4 Phase 08 当前验收状态

| 子任务 | 状态 | 说明 |
| --- | --- | --- |
| reward function | 已完成 | 有测试和 DPO reward audit |
| DPO 训练 | 已完成 | sample1000 已训练出 adapter |
| M3 推理评测 | 已完成 | 四个 split 均 500/500 |
| GRPO 正式训练 | 未完成 | 仅有 `examples=1/global_step=1` 的 smoke 级别产物 |
| M4 推理评测 | 未完成 | 没有正式 M4 sample500 |
| M2/M3/M4 完整对比 | 未完成 | 目前只有 M2/M3 对比 |

## 14. DPO 失败诊断

### 14.1 总体判断

DPO 不符合预期。它在训练集偏好目标上收敛，但没有转化为业务指标提升：

- Audit Accuracy 从 `0.773` 降到 `0.668`，下降约 `0.105`。
- High-risk Miss Rate 从 `0.243` 降到 `0.237`，只改善约 `0.005`。
- Evidence Support Rate 从 `0.804` 降到 `0.799`，略降。
- Error Cases Avg 从 `164.5` 增到 `211.0`，明显变差。

### 14.2 错误迁移

| Split | M2 Correct -> M3 Wrong | M2 Wrong -> M3 Correct | Both Wrong | Both Correct |
| --- | ---: | ---: | ---: | ---: |
| test_clean | 65 | 11 | 162 | 262 |
| test_robust | 57 | 30 | 146 | 267 |
| test_unseen_template | 76 | 21 | 155 | 248 |
| test_hard_negative | 90 | 40 | 93 | 277 |
| Total | 288 | 102 | 556 | 1054 |

解释：

- M3 新增错误 `288` 个。
- M3 修复 M2 错误 `102` 个。
- 净增错误 `186` 个。
- `test_hard_negative` 中 `M2 correct -> M3 wrong` 最多，为 `90` 个，说明 DPO 对困难负例的审计边界扰动最大。

### 14.3 问题类型变化

| Issue | Delta M3-M2 |
| --- | ---: |
| audit_mismatch | +197 |
| schema_invalid | +13 |
| business_metrics_zeroed | +13 |
| hallucination | +2 |
| high_risk_miss | -21 |
| unsupported_evidence | -15 |
| bbox_strict_error | -10 |

核心解释：

- `audit_mismatch` 大幅增加，是 DPO 失败的主因。
- `high_risk_miss` 确实减少了 21 次，但收益远远不够抵消 `audit_mismatch +197`。
- `unsupported_evidence` 和 `bbox_strict_error` 有小幅改善，说明 DPO 可能对证据相关行为有一点帮助，但没有守住审计决策边界。

### 14.4 高风险新增错误集中类型

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

这说明当前 DPO 可能破坏了 SFT 已学到的高风险拒绝边界，尤其是：

- `amount_mismatch`
- `over_reimbursement`
- `order_id_mismatch`

## 15. 可能失败原因

### 15.1 DPO pairs 可能太容易

DPO reward audit 中：

- chosen mean reward = `1.000`
- rejected mean reward = `0.460`
- preference margin 最终达到 `74.731`
- loss 最终降到 `0.000568`

这说明 chosen/rejected 的差异可能过于容易被模型捕捉。模型可以快速学会“当前 pair 中哪个更像 chosen”，但这种区分不一定等价于真实测试集上的审计能力提升。

### 15.2 DPO 优化目标可能过窄

High-risk Miss Rate 只从 `0.243` 降到 `0.237`，改善约 `0.005`。如果 DPO 的目标是降低高风险放行，这个收益太小，不足以证明偏好数据有效。

### 15.3 DPO 可能造成审计边界负迁移

M3 在四个 split 的 Audit Accuracy 全部下降。这不像单一 split 偶然波动，更像 DPO adapter 对整体审计决策边界造成了系统性扰动。

尤其是高风险 `reject_recommendation` 中的金额类异常和订单号不一致，M2 原本能做对的一部分样本在 M3 中被改错，说明 DPO 可能没有保护 SFT 已经学到的正确能力。

### 15.4 训练强度可能偏大

本次 DPO 虽然只跑了 sample1000，但 margin 被拉到非常大，loss 接近 0。这可能意味着训练信号已经过强，模型被推向过度偏好的局部行为，而不是温和修正高风险漏检。

### 15.5 缺少 holdout pair 监控

当前 DPO 主要记录训练动态和 reward audit，缺少独立的 Train 内 holdout pair 监控。如果只看训练 loss 或 preference margin，容易误判为成功。

## 16. 后续改进建议

### 16.1 暂停 GRPO/M4

不建议基于当前 M3 结果继续扩大 GRPO。原因：

- DPO 业务指标已经失败。
- GRPO 如果从不稳定的 M3 出发，可能会放大错误审计边界。
- 当前 GRPO 只有 smoke 级别产物，不应写成正式实验结果。

### 16.2 重构 DPO v2 pairs

DPO v2 应只从 MV-Train 构造，不从 Val/Test 反向调参。建议加入：

- hard rejected：结构合法、证据看似完整，但 `audit_result` 或 `risk_level` 错误。
- high-risk miss 强化 pair：高风险样本错误放行、错误降级为中低风险。
- 保护型 pair：保护 M2 已经做对的高风险拒绝样本，尤其是 `amount_mismatch`、`over_reimbursement`、`order_id_mismatch`。
- 重复控制：限制同一 case 和同一 rejected type 的过高占比，避免模型过拟合单一偏好模式。

### 16.3 降低 DPO 训练强度

建议新 DPO 使用更保守配置：

- 更小 learning rate，例如 `1e-6` 或 `2e-6`。
- 更低 beta，例如 `0.05`。
- 更少 step 或 early stopping。
- 加入 Train 内 holdout pairs。
- 不以 loss 接近 0 作为成功标准，而以 holdout preference 和 sample500 业务指标为准。

### 16.4 新 DPO 验收标准

新 DPO 版本至少应满足：

- Audit Accuracy 不低于 M2，或下降不超过 `0.01`。
- High-risk Miss Rate 相比 M2 至少下降 `0.03`。
- Evidence Support Rate 不低于 M2，或下降不超过 `0.01`。
- Hallucination Rate 保持低位，不明显上升。
- Error Cases Avg 不高于 M2。

如果新 DPO 仍无法同时改善 High-risk Miss Rate 和 Audit Accuracy，应停止继续堆 DPO/GRPO，把 Phase08 结论写成 `DPO negative result / reward-data mismatch`，后续重点转向偏好数据质量、审计规则和训练样本构造。

## 17. 工程完成情况、未完成任务与修改方向

### 17.1 已完成任务

| 类别 | 已完成内容 | 证据或产物 |
| --- | --- | --- |
| 项目定义 | 明确企业费用报销多凭证一致性审计任务 | `docs/project_brief.md`、`docs/global_contracts.md` |
| 工程骨架 | 建立 Python 包、配置目录、脚本入口和 phase 文档 | `src/mv_audit/`、`configs/`、`scripts/`、`docs/phases/` |
| 数据 schema | 定义 case schema 和模型输出 schema | `configs/schema/case_schema.json`、`configs/schema/output_schema.json` |
| 数据生成 | 支持正常 case 生成、异常注入、风险规则和 case-level split | `src/mv_audit/data_gen/` |
| 图片渲染 | 支持四类凭证渲染、bbox 记录、视觉扰动和 bbox 可视化 | `src/mv_audit/rendering/`、`src/mv_audit/perturbation/` |
| 训练数据构造 | 支持 SFT、DPO、GRPO 格式转换 | `src/mv_audit/converters/` |
| 评测系统 | 支持 JSON/schema/字段/审计/证据/bbox/幻觉等指标 | `src/mv_audit/evaluation/` |
| SFT 训练 | 完成 `Qwen3-VL-8B-Instruct` LoRA-SFT | `outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/` |
| SFT 验证 | 完成 validation loss eval | `eval_loss=0.0002858767`、`0.0001022846` |
| Phase 07 测试 | 完成 M0/M1/M2 sample500 评测 | `docs/experiments/phase07_sample500/` |
| DPO 训练 | 完成 DPO sample1000 adapter | `outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1/` |
| M3 测试 | 完成 M3 sample500 评测和 M2/M3 对比 | `docs/experiments/phase08_m3_sample500/` |
| 失败诊断 | 完成 DPO negative result 分析 | `docs/experiments/phase08_dpo_diagnosis/` |

### 17.2 未完成任务

| 类别 | 未完成内容 | 当前建议 |
| --- | --- | --- |
| 全量 Phase 07 | 当前正式报告采用 sample500，不是完整 24000 条全量评测 | 如果预算允许，后续可补全量或扩大抽样 |
| 数据完整备份 | 全量 predictions、checkpoint、原始日志仍在服务器侧，不在 Git | 释放服务器前需重新开机打包备份 |
| DPO v2 | 当前 DPO v1 业务失败，尚未重构 pair 数据 | 优先做 Train-only DPO v2 |
| GRPO/M4 | GRPO 只有 smoke 产物，M4 未正式训练和评测 | 当前先暂停，不建议继续烧钱 |
| 最终实验报告 | 尚无 M2/M3/M4 完整最终对比 | 等 DPO v2 或 M4 有结果后再补 |
| 真实数据验证 | 当前是合成数据闭环，没有真实企业数据外部验证 | 后续可做小规模真实/半真实样例测试 |

### 17.3 修改方向优先级

| 优先级 | 方向 | 具体动作 | 目标 |
| --- | --- | --- | --- |
| P0 | 备份服务器大文件 | 如需释放服务器，先备份 SFT/DPO adapter、predictions、logs、runtime | 保证实验可复现 |
| P1 | DPO 失败归因深化 | 继续分析 `M2 correct -> M3 wrong` 的具体输出差异 | 找出 DPO 破坏审计边界的机制 |
| P1 | DPO v2 pair 重构 | 增加 hard rejected、保护型 pair、high-risk miss 强化 pair | 降低高风险漏检且不损伤 Audit Accuracy |
| P1 | DPO 训练降强度 | 降低 lr/beta/step，加入 early stopping 和 Train holdout | 防止 preference margin 过度拉大 |
| P2 | SFT 数据增强 | 针对高风险金额、超额报销、订单号不一致补充样本 | 提升 M2 本身的高风险稳定性 |
| P2 | 规则/后处理辅助 | 对高风险放行、缺材料、不可读字段加轻量校验 | 提供保底安全约束 |
| P3 | GRPO 重启 | 仅在 DPO v2 或 SFT+规则后处理稳定后再考虑 | 避免从失败 M3 放大错误 |

### 17.4 下一步最推荐路线

最推荐的下一步不是继续训练 GRPO，而是先做一个成本较低、可解释的 DPO v2 迭代：

```text
M2/M3 error transition analysis
-> Train-only DPO v2 pair policy
-> small DPO v2 training with conservative hyperparameters
-> M3v2 sample500 evaluation
-> compare M2 / M3 / M3v2
```

如果 M3v2 仍不能同时守住 Audit Accuracy 和降低 High-risk Miss Rate，应停止 DPO 路线，把项目结论写成“当前偏好数据和 reward 构造与业务指标存在错配”，转向 SFT 数据增强和规则约束。

## 18. 可供后续分析的问题清单

后续可以把本文发给 ChatGPT，并重点问：

1. 为什么 DPO loss 收敛但 M3 业务指标下降？
2. 当前 DPO pair 构造中最可能的问题是什么？
3. 如何设计更难、更贴近业务指标的 rejected answers？
4. 如何构造保护型 pairs，避免破坏 M2 已经学会的高风险拒绝能力？
5. DPO 的 `learning_rate`、`beta`、训练步数、early stopping 应如何调整？
6. 是否应该继续 DPO，还是转向 SFT 数据增强、规则后处理或 reward 重新设计？
7. 在不使用 Val/Test 泄漏的前提下，如何用 Train-only holdout 监控 DPO 是否过拟合？

## 19. 关键本地归档路径

Phase 07：

- `docs/experiments/phase07_sample500/phase07_sample500_report.md`
- `docs/experiments/phase07_sample500/metrics_summary.csv`
- `docs/experiments/phase07_sample500/metrics_by_model.csv`
- `docs/experiments/phase07_sample500/error_cases/`
- `docs/experiments/phase07_sample500/figures/`

Phase 08 DPO：

- `docs/experiments/phase08_dpo_sample1000/phase08_dpo_sample1000_report.md`
- `docs/experiments/phase08_dpo_sample1000/dpo_summary.csv`
- `docs/experiments/phase08_dpo_sample1000/training_history.csv`
- `docs/experiments/phase08_dpo_sample1000/dpo_reward_audit.json`
- `docs/experiments/phase08_dpo_sample1000/figures/`

Phase 08 M3：

- `docs/experiments/phase08_m3_sample500/phase08_m3_sample500_report.md`
- `docs/experiments/phase08_m3_sample500/metrics_summary.csv`
- `docs/experiments/phase08_m3_sample500/metrics_by_model.csv`
- `docs/experiments/phase08_m3_sample500/m2_m3_comparison.csv`
- `docs/experiments/phase08_m3_sample500/merge_summary.json`
- `docs/experiments/phase08_m3_sample500/error_cases/`
- `docs/experiments/phase08_m3_sample500/figures/`

DPO 诊断：

- `docs/experiments/phase08_dpo_diagnosis/phase08_dpo_diagnosis_report.md`
- `docs/experiments/phase08_dpo_diagnosis/dpo_failure_analysis.csv`
- `docs/experiments/phase08_dpo_diagnosis/failure_by_type.csv`
- `docs/experiments/phase08_dpo_diagnosis/issue_delta_summary.csv`
- `docs/experiments/phase08_dpo_diagnosis/transition_summary.csv`
- `docs/experiments/phase08_dpo_diagnosis/figures/`

## 20. 服务器产物状态

以下产物没有进入 Git，只保留在服务器侧路径中。服务器已关闭，如果后续释放实例或删除数据盘，需要先开机备份：

- SFT adapter：`outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/`
- DPO adapter：`outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1/`
- Phase 07 predictions：`outputs/predictions/phase07_sample500/`
- Phase 08 M3 predictions：`outputs/predictions/phase08_m3_sample500/`
- 训练日志和 runtime shard 日志：`outputs/logs/`、`outputs/runtime/`
- 数据与 manifest：`data/mv_audit/`

这些大文件不适合直接提交到 Git。当前 Git 中只归档了指标、报告、图表、error cases 和 manifest 摘要。
