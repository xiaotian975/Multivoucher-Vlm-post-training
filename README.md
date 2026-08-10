# MultiVoucher-Audit

MultiVoucher-Audit 是一个面向企业费用报销一致性审计的多图 VLM 后训练项目，基座模型为 `Qwen3-VL-8B-Instruct`。它不是单张发票 OCR demo，而是把一个完整报销 case 中的发票、支付截图、报销申请单、订单截图一起交给模型，让模型做跨图字段抽取、一致性校验、异常识别和风险判断。

项目最终希望模型输出统一的 Evidence-Grounded JSON：既给出 `risk_level` 和 `audit_result`，也给出支持结论的字段值、图片来源、bbox 位置证据和不确定性说明。对于缺材料、图片不可读或证据不足的情况，模型不应编造确定结论，而应转向 `missing_info` 或 `manual_review`。

这个仓库目前主要用于构建可控的合成数据、图片渲染、训练数据格式、评测工具和后续 LoRA-SFT/DPO/GRPO 实验闭环。新人可以把它理解为一个“多凭证审计任务的实验工厂”：先生成有标签的报销 case，再渲染成多张凭证图片，随后构造训练样本并评测模型是否真的根据证据做审计。

## 项目要解决什么

一个标准 case 包含四类凭证：

| 凭证类型 | 作用 |
| --- | --- |
| `invoice` | 提供发票号码、日期、销售方、项目、金额、税额、价税合计 |
| `payment` | 提供支付金额、支付时间、收款方、付款人、支付流水号 |
| `reimbursement_form` | 提供申请人、费用类型、报销金额、申请日期、事由、订单号 |
| `order` | 提供订单号、商品或服务、商户、订单金额、订单用户、下单时间 |

模型需要完成的核心判断包括：

- 字段抽取：从不同图片中读出金额、商户、人员、日期、订单号、支付流水号等字段。
- 跨图一致性校验：比较发票、支付、报销单、订单中的金额、商户、人员、日期和订单号是否一致或合理。
- 异常识别：识别 `amount_mismatch`、`over_reimbursement`、`date_mismatch`、`merchant_mismatch`、`applicant_mismatch`、`order_id_mismatch`、`missing_document`、`duplicate_in_batch`、`unreadable_image` 等异常。
- 风险与审核建议：输出唯一 `risk_level` 和唯一 `audit_result`，例如 `pass`、`manual_review`、`missing_info`、`reject_recommendation`。
- 证据定位：输出字段值对应的 `source_image_id`、`source_doc_type`、`bbox` 和 `evidence_text`。
- 不确定性处理：当核心字段不可读或材料不完整时，明确列出不确定字段，避免无证据结论。

## 当前仓库状态

当前仓库已经不只是 phase 00 骨架，已经具备从合成 case 到训练数据和评测工具的主体链路。总体状态如下：

- 已有工程骨架、配置目录、脚本入口、基础 IO/config/logging 工具，以及阶段文档约束。
- 已有 `case_schema.json`、`output_schema.json`、字典文件、debug/main 两套数据生成配置。
- 已有正常 case 生成、异常注入、风险规则引擎和 case 级数据划分。
- 已有四类凭证图片渲染、bbox 记录、bbox 可视化和视觉扰动相关模块。
- 已有 SFT、DPO、GRPO 数据格式转换工具，以及对应的数据输出目录。
- 已有 JSON parser、bbox/字段/一致性/证据/幻觉等评估模块，并有 fake prediction 评估报告用于验证评测器行为。
- 已有 `Qwen3-VL-8B-Instruct` 配置、模型下载脚本、单图和多图 smoke test；本地存在 `models/Qwen3-VL-8B-Instruct` 目录。
- 尚未完成真实 LoRA-SFT、DPO、GRPO 训练代码，也尚未形成 M0/M1/M2/M3/M4 的真实模型指标对比。

推荐先读：

- `docs/project_brief.md`：项目目标、任务边界和第一版不做什么。
- `docs/global_contracts.md`：schema、risk rule、bbox、数据泄漏等全局约束。
- `docs/execution_roadmap.md`：phase 00 到 phase 08 的工程路线。

## 数据与实验链路

项目主链路可以概括为：

```text
case schema
  -> base cases
  -> anomaly injection
  -> case-level splits
  -> voucher rendering + bbox annotations
  -> SFT/DPO/GRPO data
  -> evaluation
  -> LoRA-SFT/DPO/GRPO experiments
```

其中前半段负责生成“有真值的多图报销审计样本”，后半段负责让 VLM 学会输出结构化审计结论，并用统一评测指标检查模型是否真的抽对字段、遵守 JSON schema、引用正确证据、减少高风险漏放和幻觉。

## 目录结构怎么读

| 路径 | 内容 |
| --- | --- |
| `docs/` | 项目 brief、全局契约、执行路线和各 phase 说明 |
| `configs/` | 数据生成、模型、schema、训练和评测配置 |
| `src/mv_audit/` | 项目 Python 包，包含数据生成、渲染、转换、评测、推理、训练占位和工具模块 |
| `data/mv_audit/` | 字典、raw cases、图片、bbox annotations、SFT/DPO/GRPO 数据 |
| `scripts/` | 常用流水线脚本：环境准备、数据生成、图片渲染、训练数据构造、评估 |
| `outputs/` | 日志、评估报告、bbox 可视化样本、预测和 checkpoint 输出位置 |
| `models/` | 本地模型权重目录，当前约定为 `models/Qwen3-VL-8B-Instruct` |

## 快速开始

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

构造 debug 训练数据：

```bash
bash scripts/03_build_train_data.sh
```

运行 fake prediction 评估，用来检查评测模块是否能区分完美预测和故意错误预测：

```bash
bash scripts/08_evaluate.sh
```

如果要使用 main 规模数据，对应入口是：

```bash
bash scripts/01_generate_main_cases.sh
bash scripts/02_render_main_images.sh
bash scripts/03_build_main_train_data.sh
```

## Qwen3-VL Smoke Test

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

注意：smoke test 需要本地或可下载的模型权重、足够显存，以及实际存在的测试图片路径。当前仓库有 `configs/model/qwen3vl_8b.yaml` 和本地模型目录约定，但 README 中的 `examples/*.png` 只是命令示例，运行前需要替换成真实图片路径。

## 关键约束

全局规则以 `docs/global_contracts.md` 为准。实现、训练和评测时尤其要注意：

- 模型最终输出必须是单个 Evidence-Grounded JSON 对象。
- 模型输出不得包含 `primary_anomaly_type` 和 `evidence_sufficient`；这两个字段只属于 ground truth metadata 或数据生成控制信息。
- bbox 使用 `[x1, y1, x2, y2]`，坐标范围归一化到 0 到 1000。
- 数据划分以 case 为单位，同一个 case 的多张图片不能拆到不同 split。
- MV-Val 和 MV-Test 只用于验证和最终评测，不能参与 DPO 数据构造、GRPO reward 调参、错误样本回流、few-shot 示例选择或 prompt 规则调参。
- `duplicate_in_batch` 只表示当前输入材料内部存在重复凭证，不表示历史库查重。
- 模型不直接执行生产审批；`reject_recommendation` 只是拒绝建议，最终仍应由人工确认。

## 未来工作

第一版后续重点在 phase 07 和 phase 08：

- 实现 LoRA-SFT 训练、批量推理和 M0/M1/M2 baseline 对比。
- 报告 JSON Validity、Schema Compliance、Audit Accuracy、High-risk Miss Rate、Evidence Support Rate、bbox 相关指标等核心结果。
- 在 SFT 能稳定输出合法 JSON 后，实现 DPO，用于纠正风险偏好和证据偏好。
- 实现小规模 GRPO 和 rule-based reward，强化高风险不放行、证据正确、JSON 合法和不确定转人工。
- 形成 M2/M3/M4 指标对比、错误样本分析和最终实验报告。

## 实验进展补充（2026-08-10）

本节记录服务器侧已经完成的 SFT 训练摘要。这里的结论只覆盖 LoRA-SFT 的训练过程、checkpoint 和验证集 loss，不等同于 Phase 07 的最终业务指标评测；JSON Validity、Schema Compliance、Audit Accuracy、High-risk Miss Rate、Evidence Support Rate 等指标仍以后续 `phase07_sample500` 评测报告为准。

### SFT 训练摘要

- 基座模型：`Qwen3-VL-8B-Instruct`。
- 训练配置：`configs/train/sft_lora_qwen3vl_8b_phase07_server.yaml`。
- SFT adapter：`outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/`。
- 训练数据：`data/mv_audit/sft_main/train_existing_images.jsonl`，`21682` 条。
- 验证数据：`data/mv_audit/sft_main/val_existing_images.jsonl`，`1138` 条。
- 本次使用的是 existing-images 子集，因此样本数少于早期规划中的完整 main 规模。

### 核心训练参数

- `num_train_epochs=1`
- `learning_rate=1e-4`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=16`
- `bf16=true`
- `gradient_checkpointing=true`
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`
- LoRA target modules：`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`

### 训练与验证结果

- 最终训练步数：`global_step=1356`。
- 最终 epoch：`1.0`。
- 训练耗时：`67437.9456s`，约 `18.73h`。
- 最终 `train_loss=0.004982630206586825`。
- 验证记录 1：`eval_loss=0.00028587671113200486`，`epoch=0.37`。
- 验证记录 2：`eval_loss=0.00010228458268102258`，`epoch=0.74`。
- checkpoint：`checkpoint-1000`、`checkpoint-1356` 和最终 adapter 均已生成。

### 当前解释口径

- SFT 已完成训练，并且在验证集上做过 loss eval。
- Phase 07 的 `M0/M1/M2` 抽样推理和业务指标评测仍在独立的 `phase07_sample500` 输出目录中推进。
- 当前不记录 DPO/GRPO 结果；这部分属于 Phase 08，需等 Phase 07 的 SFT 输出稳定性和业务指标确认后再推进。
