# Phase 08: DPO、小规模 GRPO 和 Reward Function

## 阶段目标

在 LoRA-SFT 已能稳定输出合法 JSON 后，完成 DPO、小规模 GRPO 和 rule-based reward。DPO 目标是纠正风险偏好和证据偏好；GRPO 目标是进一步强化高风险不放行、证据正确、JSON 合法和不确定转人工。

## 允许修改范围

- `src/mv_audit/training/train_dpo.py`
- `src/mv_audit/training/train_grpo.py`
- `src/mv_audit/training/reward_function.py`
- `configs/train/dpo_qwen3vl_8b.yaml`
- `configs/train/grpo_qwen3vl_8b.yaml`
- `scripts/05_train_dpo.sh`
- `scripts/06_train_grpo.sh`
- `tests/test_reward_function.py` 或 `scripts/test_reward_function.py`
- `outputs/checkpoints/dpo/`
- `outputs/checkpoints/grpo/`
- `outputs/eval_reports/`

## 禁止事项

- 不在 SFT JSON 输出不稳定时强行进入 GRPO。
- 不修改 SFT 数据生成逻辑。
- 不从 Val/Test 构造 DPO 或 GRPO 训练数据。
- 不把 reward 调参建立在 Test 反复窥探上。
- 不让模型学习简单过度保守策略而不监控 False Manual Review Rate。

## 输入

- `outputs/checkpoints/sft/`
- `data/mv_audit/dpo/pairs_train.jsonl`
- `data/mv_audit/grpo/prompts_train.jsonl`
- phase 06 parser 和评测指标。
- phase 07 M2 评测报告和错误样本。

## 输出

- `src/mv_audit/training/train_dpo.py`
- `src/mv_audit/training/train_grpo.py`
- `src/mv_audit/training/reward_function.py`
- `configs/train/dpo_qwen3vl_8b.yaml`
- `configs/train/grpo_qwen3vl_8b.yaml`
- DPO checkpoint。
- GRPO checkpoint。
- reward 单元测试结果。
- M2/M3/M4 对比评测报告。

Reward function 输出：

```json
{
  "reward": 0.0,
  "details": {
    "r_field": 0.0,
    "r_consistency": 0.0,
    "r_anomaly": 0.0,
    "r_audit": 0.0,
    "r_evidence": 0.0,
    "r_json": 0.0,
    "r_uncertainty": 0.0,
    "p_hallucination": 0.0,
    "p_high_risk_miss": 0.0
  }
}
```

Reward 规则：

- JSON 不合法，直接 `reward=-1`。
- ground truth `risk_level=high` 且模型 `audit_result=pass`，直接 `reward=-1`。
- 否则按加权项计算并 clip 到 `[-1, 1]`。

```text
R_raw =
+0.15*r_field
+0.15*r_consistency
+0.20*r_anomaly
+0.15*r_audit
+0.15*r_evidence
+0.10*r_json
+0.10*r_uncertainty
-0.20*p_hallucination
-0.40*p_high_risk_miss

reward = clip(R_raw, -1, 1)
```

## 测试方式

Reward 单元测试至少覆盖：

- 完美答案 reward 接近 1。
- JSON 非法 reward = -1。
- high-risk pass reward = -1。
- 缺失材料但输出 pass 触发强惩罚。
- `unreadable_image` 编造字段触发 hallucination 或 uncertainty penalty。
- 证据来源错误降低 `r_evidence`。

训练后分别评测：

- M2 SFT
- M3 SFT + DPO
- M4 SFT + DPO + GRPO

## 完成定义

- DPO 能从 SFT checkpoint 初始化，reference model 为冻结 SFT。
- GRPO 能从 DPO checkpoint 初始化，reference model 为冻结 DPO。
- GRPO group advantage 使用组内 reward 标准化。
- 训练日志记录平均 reward、JSON valid rate、high-risk miss rate 和 hallucination penalty。
- DPO/GRPO 后 High-risk Miss Rate 和 Hallucination Rate 下降。
- Evidence Support Rate 上升。
- False Manual Review Rate 没有明显恶化；若明显上升，需要调整 uncertainty reward 或 high-risk penalty 权重。

## 下一阶段依赖

phase 08 完成后进入实验报告、错误分析、可视化 demo 或后续扩展，不再属于第一版核心工程阶段。
