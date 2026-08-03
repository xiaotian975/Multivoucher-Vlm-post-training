# Phase 07: LoRA-SFT、推理和 Baseline 评测

## 阶段目标

完成 M0、M1、M2 三个模型的对比，并验证 LoRA-SFT 是否提升 JSON 合法率、字段抽取、审核准确率、证据支持率和高风险不放行能力。

模型编号：

| 编号 | 方法 |
| --- | --- |
| M0 | `Qwen3-VL-8B-Instruct` zero-shot |
| M1 | `Qwen3-VL-8B-Instruct` few-shot |
| M2 | `Qwen3-VL-8B-Instruct + LoRA-SFT` |

## 允许修改范围

- `src/mv_audit/training/train_sft.py`
- `src/mv_audit/inference/batch_inference.py`
- `configs/train/sft_lora_qwen3vl_8b.yaml`
- `scripts/04_train_sft.sh`
- `scripts/07_run_inference.sh`
- `scripts/08_evaluate.sh`
- `outputs/checkpoints/sft/`
- `outputs/predictions/m0_zero_shot/`
- `outputs/predictions/m1_few_shot/`
- `outputs/predictions/m2_sft/`
- `outputs/eval_reports/`

## 禁止事项

- 不做 DPO。
- 不做 GRPO。
- 不修改 SFT 数据生成逻辑，除非 phase 05 验收发现数据格式错误。
- 不从 test set 选择 few-shot 示例。
- 不只报告 loss，不报告业务指标。

## 输入

- `data/mv_audit/sft/train.jsonl`
- `data/mv_audit/sft/val.jsonl`
- 各测试集 eval files。
- `configs/schema/output_schema.json`
- `Qwen3-VL-8B-Instruct` 基座模型。
- phase 06 评测系统。

## 输出

- `src/mv_audit/training/train_sft.py`
- `src/mv_audit/inference/batch_inference.py`
- `configs/train/sft_lora_qwen3vl_8b.yaml`
- SFT checkpoint。
- M0/M1/M2 predictions。
- `metrics_summary.csv`。
- error cases 导出。

SFT 默认配置：

- `learning_rate: 1e-4`
- `num_train_epochs: 2`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 16`
- `bf16: true`
- `gradient_checkpointing: true`
- `lora_r: 16`
- `lora_alpha: 32`
- `lora_dropout: 0.05`
- LoRA target modules: `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`

## 测试方式

- 运行 M0 zero-shot 推理。
- 运行 M1 few-shot 推理，示例只能来自训练集。
- 训练 M2 LoRA-SFT。
- 对 `test_clean`、`test_robust`、`test_unseen_template`、`test_hard_negative` 分别推理和评测。
- 导出各模型、各测试集指标表。

## 完成定义

至少报告：

- JSON Validity
- Schema Compliance
- Field EM
- Risk Type Macro-F1
- Audit Accuracy
- High-risk Miss Rate
- False Manual Review Rate
- Evidence Support Rate
- Hallucination Rate
- Evidence BBox Accuracy Relaxed

如果 SFT 后 JSON Validity 没有明显上升，需要回查 SFT 数据格式、assistant loss mask 和 prompt 模板。如果 High-risk Miss Rate 仍很高，进入 phase 08 通过 DPO 和 reward 继续增强。

## 下一阶段依赖

phase 08 依赖本阶段的 SFT checkpoint、M2 评测结果、错误样本分类和稳定 JSON 输出能力。
