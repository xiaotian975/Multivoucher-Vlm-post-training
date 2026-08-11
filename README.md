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

## Phase 07 抽样评测结果（2026-08-11）

Phase 07 sample500 已完成并归档。服务器侧完成 `M0/M1/M2 × 4 splits × 500`，共 `6000/6000` 条预测；完整评测结果已整理到 `docs/experiments/phase07_sample500/`。全量 predictions、checkpoint 和训练日志仍保留在服务器 `outputs/` 中，不进入 Git 仓库。

完整报告和图表：

- [Phase 07 Sample500 抽样评测报告](docs/experiments/phase07_sample500/phase07_sample500_report.md)
- [完整 12 行指标表](docs/experiments/phase07_sample500/metrics_summary.csv)
- [按模型聚合指标表](docs/experiments/phase07_sample500/metrics_by_model.csv)
- [模型平均指标图](docs/experiments/phase07_sample500/figures/model_average_metrics.png)
- [M2 分测试集指标图](docs/experiments/phase07_sample500/figures/m2_split_metrics.png)
- [错误样本数量图](docs/experiments/phase07_sample500/figures/error_cases_by_split.png)

M2 LoRA-SFT 核心结果如下：

| Split | JSON Validity | Schema Compliance | Field EM | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.832 | 0.831 | 0.744 | 0.270 | 0.747 |
| test_robust | 1.000 | 0.810 | 0.809 | 0.742 | 0.276 | 0.727 |
| test_unseen_template | 1.000 | 0.864 | 0.863 | 0.742 | 0.274 | 0.775 |
| test_hard_negative | 1.000 | 1.000 | 0.9998 | 0.866 | 0.152 | 0.965 |

按 4 个 split 求平均后，模型对比如下：

| Model | Schema Compliance | Field EM | Risk Type Macro-F1 | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 zero-shot | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| M1 few-shot | 0.115 | 0.114 | 0.022 | 0.079 | 0.999 | 0.025 | 0.575 |
| M2 LoRA-SFT | 0.877 | 0.876 | 0.743 | 0.774 | 0.243 | 0.804 | 0.001 |

结论：M2 相比 M0/M1 显著提升了结构化输出、字段抽取、审计准确率和证据支持率，JSON Validity 在四个测试集上均为 `1.000`。但 M2 的高风险漏检率仍约为 `0.15-0.28`，说明后续 Phase 08 的 DPO/GRPO 仍有明确价值。

## Phase 08 DPO 训练结果（2026-08-11）

Phase 08 已完成 DPO 子任务的 sample1000 训练与结果归档。本节只记录 `M3 = SFT + DPO` adapter 的训练过程和 reward 审计结果；M3 sample500 业务指标评测结果见下一节。GRPO 目前只有 smoke 级别产物，不作为正式结果写入。

完整报告和图表：

- [Phase 08 DPO Sample1000 实验报告](docs/experiments/phase08_dpo_sample1000/phase08_dpo_sample1000_report.md)
- [DPO 汇总指标](docs/experiments/phase08_dpo_sample1000/dpo_summary.csv)
- [DPO 训练历史](docs/experiments/phase08_dpo_sample1000/training_history.csv)
- [DPO loss 曲线](docs/experiments/phase08_dpo_sample1000/figures/dpo_loss_curve.png)
- [Preference margin 曲线](docs/experiments/phase08_dpo_sample1000/figures/dpo_preference_margin.png)
- [Chosen/Rejected logp 对比](docs/experiments/phase08_dpo_sample1000/figures/dpo_logp_comparison.png)

### DPO 实验设置

- 初始模型：`Qwen3-VL-8B-Instruct` + Phase 07 SFT adapter。
- SFT adapter：`outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/`。
- DPO adapter：`outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1/`。
- 训练配置：`configs/train/dpo_qwen3vl_8b.yaml`。
- 训练数据：`data/mv_audit/dpo_main/pairs_train.jsonl`，来自 MV-Train。
- 本次运行：`max_samples=1000`，`require_existing_images=true`。
- 因缺少可用图片而跳过的 pair：`324`。

### DPO 核心结果

| Metric | Value |
| --- | ---: |
| examples | 1000 |
| global_step | 63 |
| final loss | 0.000568 |
| chosen mean reward | 1.000000 |
| rejected mean reward | 0.460112 |
| mean reward gap | 0.539888 |
| positive reward gap rate | 0.844000 |
| rejected JSON valid rate | 0.896000 |
| rejected high-risk miss rate | 0.155000 |
| rejected hallucination penalty | 0.015506 |

训练动态显示，loss 从 `0.6877865195274353` 下降到 `0.0005680027534253895`，preference margin 从 `0.10750198364257812` 上升到 `74.73100280761719`，说明 DPO 训练已经明显拉开 chosen 与 rejected 的偏好差距。

### 当前 Phase 08 状态

- 已完成：reward function、DPO 训练代码、DPO sample1000 后台训练、DPO adapter、DPO reward 审计、训练曲线和实验归档。
- 已完成：用 DPO adapter 做 `M3 SFT + DPO` 的 sample500 抽样推理评测，结果见下一节。
- 未完成：正式 GRPO 训练、M4 推理评测、M2/M3/M4 最终业务指标对比。
- 当前判断：DPO 训练过程收敛，但 M3 业务指标未明显优于 M2，因此 Phase 08 不能直接判定为成功。

## Phase 08 M3 Sample500 评测结果（2026-08-11）

Phase 08 的 `M3 = SFT + DPO` sample500 推理评测已完成并归档。服务器侧采用 8 卡 shard 数据并行完成剩余推理，四个 split 均合并为 `500/500`；`merge_summary.json` 显示 `missing=0`、`duplicates_seen_before_dedup=0`。全量 predictions、checkpoint 和原始日志仍保留在服务器 `outputs/` 中，不进入 Git。

完整报告和图表：

- [Phase 08 M3 Sample500 评测报告](docs/experiments/phase08_m3_sample500/phase08_m3_sample500_report.md)
- [M3 完整指标表](docs/experiments/phase08_m3_sample500/metrics_summary.csv)
- [M2/M3 平均对比表](docs/experiments/phase08_m3_sample500/metrics_by_model.csv)
- [M2/M3 逐 split 对比表](docs/experiments/phase08_m3_sample500/m2_m3_comparison.csv)
- [M2/M3 平均指标图](docs/experiments/phase08_m3_sample500/figures/m2_vs_m3_average_metrics.png)
- [M3 分 split 指标图](docs/experiments/phase08_m3_sample500/figures/m3_split_metrics.png)
- [M3 error cases 数量图](docs/experiments/phase08_m3_sample500/figures/m3_error_cases_by_split.png)

M3 核心结果如下：

| Split | JSON Validity | Schema Compliance | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.808 | 0.650 | 0.264 | 0.727 | 0.002 | 227 |
| test_robust | 1.000 | 0.804 | 0.652 | 0.264 | 0.724 | 0.002 | 203 |
| test_unseen_template | 1.000 | 0.868 | 0.634 | 0.276 | 0.778 | 0.001 | 231 |
| test_hard_negative | 1.000 | 1.000 | 0.738 | 0.145 | 0.966 | 0.001 | 183 |

按四个 split 求平均后，M2 与 M3 对比如下：

| Model | Schema Compliance | Audit Accuracy | High-risk Miss Rate | Evidence Support Rate | Hallucination Rate | Error Cases Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 LoRA-SFT | 0.877 | 0.773 | 0.243 | 0.804 | 0.001 | 164.500 |
| M3 SFT+DPO | 0.870 | 0.668 | 0.237 | 0.799 | 0.001 | 211.000 |
| Delta M3-M2 | -0.007 | -0.105 | -0.005 | -0.005 | 0.001 | 46.500 |

结论：DPO 后的 M3 没有明显优于 M2。M3 的 JSON Validity 仍为 `1.000`，High-risk Miss Rate 平均只从 `0.243` 小幅降到 `0.237`，但 Audit Accuracy 从 `0.773` 降到 `0.668`，error cases 平均数量也增加。因此当前不能把 Phase 08 直接判定为成功，后续需要复盘 DPO 偏好数据或 reward 构造。

当前 Phase 08 验收状态：

- DPO 子任务：已完成。
- M3 sample500 推理评测：已完成。
- GRPO/M4：未正式完成。服务器上仅存在 `examples=1/global_step=1` 的 GRPO smoke 级别产物，不作为正式 M4 结果写入。
- Phase 08 整体：未完全验收完成；若下一步是 DPO/GRPO 调整分析，依赖已满足；若要进入最终实验报告或最终 demo，还需要正式 GRPO、M4 sample500 和 M2/M3/M4 对比。

## Phase 08 DPO 诊断与优化方向（2026-08-11）

基于 M2/M3 sample500 对比，当前 DPO 结果应视为 negative result，而不是可直接进入 GRPO 的成功结果。完整诊断见 [Phase 08 DPO 诊断与优化方案](docs/experiments/phase08_dpo_diagnosis/phase08_dpo_diagnosis_report.md)。

核心判断：

- DPO 训练过程收敛，但业务指标不符合预期。
- M3 相比 M2 的平均 Audit Accuracy 从 `0.773` 降到 `0.668`。
- High-risk Miss Rate 只从 `0.243` 小幅降到 `0.237`，改善不足。
- Evidence Support Rate 从 `0.804` 小幅降到 `0.799`。
- error cases 平均从 `164.5` 增到 `211.0`。
- 错误迁移统计显示：`M2 correct -> M3 wrong` 为 `288` 个，`M2 wrong -> M3 correct` 为 `102` 个，净增错误 `186` 个。

主要问题：

- `audit_mismatch` 在 M3 中大幅增加，四个 split 合计比 M2 多 `197` 次。
- DPO 略微减少 `high_risk_miss`，但只减少 `21` 次，收益不足以抵消审计决策错误增加。
- 新增错误主要集中在高风险 `reject_recommendation` 样本，尤其是 `amount_mismatch`、`over_reimbursement` 和 `order_id_mismatch`。

后续方向：

- 暂停 GRPO，不扩大当前 GRPO smoke 产物。
- 只从 Train 重构 DPO pairs，不从 Val/Test 反向调参。
- 增加 hard rejected 和保护型 pairs，重点保护 M2 已经做对的高风险拒绝样本。
- 降低 DPO 训练强度，尝试更小 learning rate、更少 step、更低 beta 或 early stopping。
- 新 DPO 版本必须同时守住 Audit Accuracy 和 High-risk Miss Rate；否则应把 Phase08 结论写成 DPO negative result / reward-data mismatch。
