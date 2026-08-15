# MultiVoucher-Audit 后训练闭环 Codex 执行规范 v4

> 文档定位：本文件是 MultiVoucher-Audit 后训练闭环的需求与执行规范。后续 Repair、RL、Final Holdout 等操作应优先以本文为准；若本文与现有 evaluator 或已归档事实冲突，应记录冲突并优先保持现有 evaluator 行为。

## 0. Codex 的任务定义

Codex 的目标不是：

> 尽可能多地实现 SFT、DPO、GRPO。

而是：

> 在不破坏现有 MultiVoucher-Audit 实验资产、评测边界和 M2 baseline 的前提下，建立一个可复现、可中止、可审计的 High-risk Repair → Error Diagnosis → Optional Online RL 实验闭环。

最终路线：

\[
\boxed{
M2
\rightarrow
Validation\ Baseline
\rightarrow
Repair\ R1
\rightarrow
Paired\ Diagnosis
\rightarrow
Repair\ R2\ (optional)
\rightarrow
Decision\ Bottleneck\ Verification
\rightarrow
Online\ RL\ (optional)
\rightarrow
Frozen\ Final\ Holdout
}
\]

---

## 1. Codex 必须遵守的五条最高优先级规则

### RULE-1：不得破坏已有实验

禁止覆盖：

```text
M2 checkpoint
M3 checkpoint
M3v2 checkpoint
sample500 predictions
已有 metrics
已有 docs/experiments/
原始 train/val/test JSONL
```

所有新实验只能：

```text
创建新目录
创建新配置
创建新 checkpoint
创建新 manifest
```

---

### RULE-2：优先复用现有代码

尤其：

```text
src/mv_audit/evaluation/
src/mv_audit/inference/
src/mv_audit/training/
```

禁止重新写：

```text
第二套 JSON parser
第二套 Audit Accuracy
第二套 High-risk Miss
第二套 Evidence evaluator
```

README 当前已经指定统一 evaluator：

```text
json_parser.py
audit_metrics.py
evidence_metrics.py
bbox_evaluator.py
evaluate_all.py
```


---

### RULE-3：测试集不能参与开发

从现在开始：

```text
sample500
```

不再承担新的：

```text
模型选择
reward 调参
Repair R1/R2 决策
是否进入 RL 的判断
```

这些全部使用 Validation。

---

### RULE-4：昂贵操作默认禁止

以下动作只有显式 flag 才允许：

```text
正式 Repair training
正式 GRPO training
完整 sample500
final_holdout
大规模图像重新渲染
```

例如：

```bash
ALLOW_TRAINING=1
ALLOW_RL=1
ALLOW_FINAL_HOLDOUT=1
```

缺少 flag：

```text
必须 dry-run / audit-only
```

---

### RULE-5：任何阶段失败都不能自动跳到下一算法

例如：

```text
Repair R1 failed
```

禁止自动：

```text
→ GRPO
```

必须：

```text
→ diagnosis
→ archive
→ STOP
```

---

## 2. 实验状态机

新增：

```text
outputs/runtime/post_training_v4/state.json
```

示例：

```json
{
  "phase": "REPAIR_PREP",
  "status": "READY",
  "last_completed_step": "BUILD_REPAIR_DEV",
  "next_allowed_step": "RUN_M2_REPAIR_DEV",
  "final_holdout_locked": true,
  "final_holdout_consumed": false
}
```

合法 Phase：

```text
REPO_AUDIT
DATA_BOUNDARY
BASELINE
REPAIR_PREP
REPAIR_R1
REPAIR_R2
REPAIR_DIAGNOSIS
RL_DECISION
RL_PREP
RL_COMPAT
RL_REWARD_SMOKE
RL_PILOT
RL_FINAL
FINAL_EVAL
DONE
FAILED
```

Codex 每完成一个阶段：

```text
更新 state.json
生成 run_manifest
```

---

## 3. Phase A：Repository Audit

### 3.1 第一轮只读文件

Codex 必须先阅读：

```text
README.md
docs/code_inventory.md

src/mv_audit/evaluation/evaluate_all.py
src/mv_audit/evaluation/audit_metrics.py
src/mv_audit/evaluation/evidence_metrics.py
src/mv_audit/evaluation/json_parser.py
src/mv_audit/evaluation/bbox_evaluator.py

src/mv_audit/inference/batch_inference.py

src/mv_audit/training/train_sft.py
src/mv_audit/training/train_dpo.py

src/mv_audit/analysis/high_risk_repair_pack.py
src/mv_audit/converters/build_high_risk_repair_sft_mix.py

scripts/11_run_high_risk_repair_sft_r1_server.sh
```

如果文件不存在：

```text
记录 missing
不得自行假设实现
```

---

## 4. Repository Audit 输出

必须生成：

```text
docs/experiments/phase09_repair_v4/
    repository_audit.md
    env_snapshot.txt
```

`repository_audit.md` 至少回答：

```text
M2 checkpoint 在哪里？
M2 adapter 如何加载？
当前 inference prompt 在哪里定义？
generation config 从哪里读取？
High-risk Miss 在哪里实现？
Evidence Support 在哪里实现？
Hallucination 在哪里实现？
False Manual Review 在哪里实现？
已有 Repair 数据在哪里？
已有 Train Decode Dev 在哪里？
```

---

## 5. 环境快照

运行：

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import peft; print(peft.__version__)"
python -c "import trl; print(trl.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
```

另外：

```bash
nvidia-smi
```

保存：

```text
env_snapshot.txt
```

此阶段：

```text
禁止 pip install -U
```

---

## 6. Phase B：重新定义数据角色

正式定义：

```text
Train
    → 模型参数更新

Validation
    → 模型选择/调参/error diagnosis

Historical Benchmark
    → sample500，只用于历史结果比较

Final Holdout
    → 最终冻结后的唯一最终测试
```

---

## 7. Validation 不再从 MV-Train 优先构造

README 中已有：

```text
val_in_template      = 2000
val_unseen_template  = 1000
```

同时实际 SFT validation existing-images 为 1138 条。

因此 Codex 必须首先检查：

```text
val_in_template 有多少 case 图片完整
val_unseen_template 有多少 case 图片完整
```

输出：

```text
validation_inventory.json
```

---

## 8. Validation Inventory 格式

```json
{
  "val_in_template": {
    "total_cases": 2000,
    "complete_images": 0,
    "missing_images": 0
  },
  "val_unseen_template": {
    "total_cases": 1000,
    "complete_images": 0,
    "missing_images": 0
  },
  "existing_sft_val_cases": 1138
}
```

数字必须实际扫描产生。

禁止直接把 README 数字当磁盘真实状态。

---

## 9. Validation 分割

目标建立：

```text
repair_dev_v1
rl_dev_v1
```

且：

\[
repair\_dev\cap rl\_dev=\varnothing
\]

---

## 10. Validation 默认目标规模

如果有效 validation cases ≥ 1000：

```text
repair_dev_v1 = 500
rl_dev_v1     = 500
```

如果 ≥ 1600：

```text
repair_dev_v1 = 800
rl_dev_v1     = 800
```

如果有效 case < 600：

```text
STOP
```

并报告：

```text
INSUFFICIENT_VALIDATION_DATA
```

不能静默改成 Train 数据。

---

## 11. Validation 分层

两套 dev 都尽量保持：

```text
val_in_template : val_unseen_template
≈
2 : 1
```

与原 validation 规模近似。

同时尽量按：

```text
risk_level
audit_result
anomaly_type
number_of_images
template
```

做 stratified split。

---

## 12. 分层不能强行制造不存在类别

例如：

```text
high / missing_info
```

如果真实 validation 中不存在：

```text
记录 count=0
```

不能人工复制样本。

---

## 13. Validation Seed 固定

使用：

```text
seed = 20260815
```

生成：

```text
data/mv_audit/dev/repair_dev_v1.jsonl
data/mv_audit/dev/rl_dev_v1.jsonl
```

同时：

```text
repair_dev_v1_manifest.json
rl_dev_v1_manifest.json
```

---

## 14. Manifest 必须记录 SHA256

至少：

```json
{
  "seed": 20260815,
  "num_cases": 500,
  "case_ids_sha256": "...",
  "dataset_sha256": "...",
  "source_splits": {},
  "risk_distribution": {},
  "audit_distribution": {},
  "anomaly_distribution": {}
}
```

---

## 15. Train Decode Dev 的新定位

现有：

```text
train_decode_dev
≈ 152
```

仍然保留。

但只用于：

```text
fast smoke
catastrophic regression detection
```

不用于正式模型选择。

---

## 16. Final Holdout

建立：

```text
final_holdout_v1
```

来源：

```text
test_clean
test_robust
test_unseen_template
test_hard_negative
```

历史 sample500 已经使用的 case 全部排除。

---

## 17. Used Case Registry

新增：

```text
docs/experiments/data_boundary/
    used_case_registry.json
```

Codex 扫描：

```text
historical predictions
sample500 manifests
diagnosis files
DPO evaluation outputs
repair packs
```

收集所有：

```text
test case_id
```

形成：

```text
used_test_case_ids
```

---

## 18. Final Holdout 默认规模

每 split：

```text
250
```

总：

\[
4\times250=1000
\]

如果服务器预算后续允许，可在**锁定前**改为：

```text
500 × 4
```

一旦 manifest 创建：

```text
不得再改变规模
```

---

## 19. Final Holdout 锁

生成：

```text
final_holdout_v1.jsonl
final_holdout_v1_manifest.json
FINAL_HOLDOUT_LOCKED
```

`FINAL_HOLDOUT_LOCKED` 内容：

```text
dataset_sha256
case_ids_sha256
creation_timestamp
git_commit
```

---

## 20. Final Holdout 防误运行

最终脚本：

```text
scripts/20_run_final_holdout.sh
```

必须要求：

```bash
ALLOW_FINAL_HOLDOUT=YES_I_UNDERSTAND
```

否则：

```bash
exit 2
```

---

## 21. Final Holdout 运行一次规则

第一次正式生成预测后写：

```text
FINAL_HOLDOUT_CONSUMED
```

后续：

```text
不同模型配置不得重新用于模型选择
```

允许同一 model/config 因技术中断进行 resume。

---

## 22. Phase C：冻结 Evaluation Contract

新增：

```text
configs/eval/audit_eval_frozen_v1.yaml
```

内容不能凭空创建。

Codex 必须：

```text
读取现有 M2/sample500 inference config
```

然后复制其已验证参数。

---

## 23. Frozen Evaluation 包括

```text
processor
prompt
system instruction
chat template
image ordering
max_new_tokens
do_sample
temperature
top_p
JSON schema
JSON parser
evaluator
```

所有 Repair/RL 模型最终 inference 使用同一 contract。

---

## 24. Evaluation Fingerprint

每个评测目录生成：

```text
evaluation_manifest.json
```

包括：

```json
{
  "model": "",
  "base_model": "",
  "adapter": "",
  "adapter_sha256": "",
  "dataset_sha256": "",
  "eval_config_sha256": "",
  "prompt_sha256": "",
  "schema_sha256": "",
  "git_commit": "",
  "transformers_version": "",
  "seed": 42
}
```

---

## 25. Phase D：建立 M2 Validation Baseline

运行：

```text
M2 → train_decode_dev
M2 → repair_dev_v1
M2 → rl_dev_v1
```

其中：

```text
train_decode_dev
```

主要用于后续 Repair fast smoke。

`repair_dev_v1`：

```text
Repair model selection
```

`rl_dev_v1`：

```text
RL decision + RL model selection
```

---

## 26. M2 Baseline 目录

```text
docs/experiments/phase09_repair_v4/baselines/

m2_train_decode_dev/
m2_repair_dev_v1/
m2_rl_dev_v1/
```

每个目录：

```text
metrics.json
predictions.jsonl
error_cases.jsonl
evaluation_manifest.json
```

---

## 27. Phase E：抽取统一 Case-level Scorer

这是整个后续方案的关键软件改造。

目标：

```python
score_case(
    prediction,
    ground_truth,
) -> CaseScore
```

---

## 28. CaseScore 建议结构

```python
@dataclass
class CaseScore:
    json_valid: bool
    schema_valid: bool

    field_score: float
    anomaly_score: float

    audit_correct: bool
    risk_correct: bool

    high_risk_miss: bool
    false_manual_review: bool
    false_escalation: bool

    evidence_support: float
    hallucination: float
    bbox_score: float | None
```

---

## 29. Case Scorer 的实现原则

必须：

```text
从现有 evaluator 抽取
```

而不是：

```text
重写已有指标
```

要求：

```text
evaluate_all()
```

调用的新 Case Scorer 与旧指标结果保持一致。

---

## 30. Regression Test

新增：

```text
tests/test_case_scorer_regression.py
```

随机抽取已有历史预测，例如：

```text
50～100 cases
```

验证：

```text
新 case scorer 聚合结果
≈
现有 evaluate_all 结果
```

允许浮点误差：

```text
1e-8 / 1e-6
```

视现有实现选择。

如果不一致：

```text
STOP
```

不能继续 Reward 实现。

---

## 31. 新增 False Escalation

这是新增 guardrail，不冒充旧指标。

定义第一版严格限定：

\[
FalseEscalation =
I(y_{gt}=pass \land y_{pred}\neq pass)
\]

即 Ground Truth：

```text
pass
```

但模型输出：

```text
manual_review
reject_recommendation
missing_info
```

均视为 unnecessary escalation。

---

## 32. 为什么先采用这个简单定义

不要一开始建立：

```text
pass < manual_review < missing_info < reject
```

因为：

```text
missing_info
manual_review
reject
```

并不天然构成严格线性严重程度。

第一版只保护：

```text
正常 pass case
```

足够解决“全部判高风险”的 reward hacking。

---

## 33. Error Attribution 输出设计

不再：

```json
{"error_source":"decision_error"}
```

改为：

```json
{
  "error_tags": [
    "perception_error",
    "decision_error"
  ],
  "primary_error_source": "perception_error",
  "decision_candidate": false
}
```

---

## 34. Error Tag 判定优先级

建议：

```text
1. output_contract_error
2. perception_error
3. consistency_error
4. decision_error
5. evidence_error
```

---

## 35. output_contract_error

满足：

```text
JSON invalid
OR
schema invalid
```

---

## 36. perception_error

当：

```text
关键 field extraction 错误
```

例如：

```text
amount
merchant
person
date
order_id
```

当前第一版可以利用：

```text
field_score < 1.0
```

打 `perception_error`。

---

## 37. consistency_error

满足：

```text
field 基本正确
```

但：

```text
anomaly_types
或
consistency_check
```

错误。

---

## 38. decision_error

满足：

```text
JSON/schema valid
field 基本正确
anomaly 基本正确
```

但是：

```text
audit_result wrong
或
high_risk_miss
```

---

## 39. evidence_error

满足：

```text
Audit Result correct
```

但：

```text
Evidence Support 不足
```

---

## 40. Decision Candidate 的严格定义

这是决定 RL 是否合理的核心。

建议默认：

```text
json_valid == True
schema_valid == True
field_score >= 0.95
anomaly_score >= 0.90
```

并且：

```text
audit_correct == False
OR
high_risk_miss == True
```

则：

```text
decision_candidate = true
```

阈值必须放：

```text
configs/analysis/error_attribution_v1.yaml
```

不能写死。

---

## 41. Phase F：Audit 现有 Repair 数据

当前：

```text
120 repair
+
120 calibration
```

继续作为 R1，不推翻已有工作。

---

## 42. Repair Audit 输出

生成：

```text
repair_distribution_report.json
repair_distribution_report.md
```

---

## 43. 必须检查

```text
total rows
unique case_ids
duplicate case_ids
missing image paths
invalid JSON answers
schema-invalid answers
anomaly distribution
risk distribution
audit_result distribution
template distribution
image-count distribution
```

以及：

```text
repair vs calibration distribution
```

---

## 44. Repair R1 Hard Failure

任何：

```text
missing images > 0
duplicate IDs > 0
invalid ground truth > 0
train/dev overlap > 0
test overlap > 0
```

则：

```text
REPAIR_DATA_INVALID
```

禁止训练。

---

## 45. Calibration 分类

Calibration 样本检查是否覆盖：

```text
low/pass
medium/manual_review
high/reject
high/missing_info
high/manual_review
```

只报告真实存在的组合。

---

## 46. Calibration 语义

其目的定义为：

\[
OldCapabilityReplay
\]

需要保护：

```text
正常 pass
正确 high-risk
manual_review
missing_info
Evidence
```

而不是简单：

```text
normal samples
```

---

## 47. Phase G：Repair R1 模型加载

从：

```text
Qwen3-VL Base
+
M2 adapter
```

继续训练。

PEFT 官方 `PeftModel.from_pretrained(..., is_trainable=True)` 就是继续训练已有 adapter 的标准接口。

---

## 48. 推荐加载语义

```python
base_model = load_qwen3vl(...)

model = PeftModel.from_pretrained(
    base_model,
    M2_ADAPTER_PATH,
    is_trainable=True,
)
```

禁止：

```python
model = get_peft_model(base_model, new_lora_config)
```

因为那会创建新的随机 LoRA，而不是继续 M2。

---

## 49. 加载后的 Assertions

必须检查：

```python
assert trainable_params > 0
assert base_trainable_params == 0
assert lora_trainable_params > 0
```

同时记录：

```text
trainable_params
total_params
trainable_ratio
target_modules
adapter_name
```

---

## 50. M2 Adapter Integrity

训练前计算：

```text
M2 adapter SHA256
```

训练后重新检查原始 M2：

```text
SHA256 必须完全一致
```

证明没有覆盖 M2。

---

## 51. Repair R1 输出目录

```text
outputs/checkpoints/sft/repair_sft_r1/
```

禁止：

```text
outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/
```

直接写入。

---

## 52. Repair R1 Optimizer Update 设计

不要机械固定：

```text
gradient_accumulation=16
```

真正约束：

\[
updates=
\left\lceil
\frac{N}
{world\_size\times batch\times grad\_acc}
\right\rceil
\times epochs
\]

R1 目标：

```text
30～60 optimizer steps
```

---

## 53. 240 样本建议

优先：

```text
world_size = 1
batch = 1
grad_acc = 4
epochs = 1
```

则约：

\[
60\ updates
\]

如果显存/稳定性需要：

```text
grad_acc=8
```

则：

\[
30\ updates
\]

---

## 54. 禁止使用 5 GPU DDP 跑 240 条 Repair

除非明确有理由。

否则很可能把：

```text
optimizer step
```

压缩到极少数量。

Repair R1 优先单卡。

---

## 55. Repair R1 起始参数

```yaml
learning_rate: 1.0e-5
epochs: 1

per_device_train_batch_size: 1
gradient_accumulation_steps: 4

bf16: true
gradient_checkpointing: true

seed: 42
```

这是：

```text
engineering starting point
```

不是已验证 optimum。

---

## 56. Repair R1 Dry Run

首先：

```text
8～16 samples
1～2 optimizer steps
```

必须验证：

```text
forward
loss finite
backward
LoRA grad non-zero
optimizer step
save
reload
inference
```

---

## 57. Repair R1 正式训练允许条件

必须存在：

```text
REPAIR_DATA_AUDIT_PASS
M2_ADAPTER_INTEGRITY_PASS
REPAIR_DRY_RUN_PASS
```

并：

```bash
ALLOW_TRAINING=1
```

---

## 58. Phase H：Repair Fast Gate

先：

```text
M2 vs R1
```

在：

```text
train_decode_dev
```

上比较。

这里只排查 catastrophic failure。

---

## 59. Fast Gate 默认阈值

放入：

```text
configs/gates/repair_fast_gate_v1.yaml
```

例如：

```yaml
json_validity_delta_min: -0.01
schema_delta_min: -0.03
audit_accuracy_delta_min: -0.03
high_risk_miss_delta_max: 0.03
```

含义：

> 只要出现明显崩塌，就不浪费更大 validation inference。

---

## 60. Fast Gate 不负责宣告成功

即使：

```text
152 cases
```

结果很好：

```text
也只能继续 Main Gate
```

不能直接进入 R2/RL/final。

---

## 61. Repair Main Gate

正式比较：

```text
M2
vs
Repair R1
```

在：

```text
repair_dev_v1
```

上完成。

---

## 62. Repair Main Gate 指标

至少：

```text
JSON Validity
Schema
Field Score
Anomaly Score
Audit Accuracy
High-risk Miss
False Manual Review
False Escalation
Evidence Support
Hallucination
BBox
```

---

## 63. 所有指标同时输出 count

High-risk Miss：

```text
13 / 70 = 0.1857
```

而不仅：

```text
0.1857
```

False Escalation：

```text
5 / 160
```

同理。

---

## 64. Repair Candidate 选择采用约束式，而非加权总分

不要创建：

```text
0.5*Accuracy - 2*HRM + ...
```

来选模型。

正确规则：

### 先判断 candidate eligible：

```text
Audit Δ >= -0.01
Evidence Δ >= -0.01
False Escalation Δ <= +0.01
JSON 无明显下降
```

然后在 eligible candidate 中：

```text
优先 High-risk Miss 最低
```

再 tie-break：

```text
Audit Accuracy 更高
```

---

## 65. R1 Success

默认：

\[
HRM_{M2}-HRM_{R1}\ge0.03
\]

同时 guardrails 通过。

则：

```text
R1_SUCCESS
```

---

## 66. R1 Partial

如果：

\[
0.01
\le
HRM_{M2}-HRM_{R1}
<
0.03
\]

且 guardrails 通过：

```text
R1_PARTIAL
```

进入：

```text
R2 candidate generation
```

---

## 67. R1 Fail

如果：

```text
HRM improvement < 0.01
```

或者：

```text
Audit regression > 0.01
Evidence regression > 0.01
False Escalation increase > 0.01
```

则：

```text
R1_FAIL
```

进入诊断。

这些阈值都写 config，可在**训练前**修改，训练结束后不能为了让模型过 gate 再修改。

---

## 68. Borderline Result

如果结果距离阈值：

```text
<= 0.01
```

例如：

```text
HRM improvement = 0.029
```

建议：

```text
不要根据 0.001 差异做强结论
```

运行：

```text
paired bootstrap
```

辅助判断。

---

## 69. Paired Bootstrap

默认：

```text
5000 resamples
seed = 42
```

计算：

```text
Δ Audit Accuracy
Δ HRM
Δ Evidence
```

95% CI。

用途：

```text
判断结果稳定性
```

不要求强制统计显著性。

---

## 70. Paired Migration

输出：

```text
repair_transitions.csv
```

至少：

```text
case_id
ground_truth_audit
ground_truth_risk
anomaly_types

m2_audit_correct
r1_audit_correct

m2_hrm
r1_hrm

m2_false_escalation
r1_false_escalation

m2_evidence
r1_evidence

error_tags
```

---

## 71. 必须报告四种迁移

```text
Correct → Correct
Correct → Wrong
Wrong → Correct
Wrong → Wrong
```

以及：

```text
HR Miss → Correct
Correct → HR Miss
HR Miss → HR Miss
```

---

## 72. Phase I：Repair R2

只有：

```text
R1_PARTIAL
```

才自动推荐 R2。

不是：

```text
R1_FAIL → R2
```

---

## 73. R2 Training Data 来源

只能：

```text
MV-Train
```

但优先使用：

```text
没有进入原 R1 的真实 M2 rollout errors
```

---

## 74. 建立 Repair Mining Pool

Codex 先找：

```text
MV-Train case_ids
-
R1 case_ids
```

并检查：

```text
图片完整
```

如果可用 Train case 不足，可以报告：

```text
NEED_RENDERING
```

不能未经允许批量重新渲染几万张图片。

---

## 75. R2 Hard Case Mining

在：

```text
repair_mining_pool
```

上运行 M2 inference。

提取：

```text
真实 M2 errors
```

优先：

```text
High-risk Miss
Decision Candidate
Consistency Error
```

---

## 76. R2 默认数据规模

先：

```text
500 hard cases
+
500 replay/calibration
=
1000
```

如果 hard case 不足：

```text
使用真实可用数量
```

不人为复制。

---

## 77. R2 Sampling Distribution

建议限制：

```text
单 anomaly family <= 35%
```

只作为 soft target。

如果真实错误高度集中：

```text
可以超过
```

但必须记录：

```text
why_balance_not_possible
```

---

## 78. R2 训练仍然从 M2 开始还是 R1 开始？

这是一个必须固定的实验选择。

推荐：

```text
M2 → R2 dataset
```

重新训练：

```text
repair_sft_r2
```

而不是：

```text
R1 → 再训练 R2
```

理由：

> 避免 R2 的结果同时混入 R1 小数据训练造成的历史漂移，更容易比较 R1 vs R2 数据策略。

因此：

```text
R1 和 R2 都是 M2 的分支
```

---

## 79. Repair 模型结构

```text
                ┌── Repair R1
M2 ─────────────┤
                └── Repair R2
```

而不是：

```text
M2 → R1 → R2
```

---

## 80. Best Repair Selection

比较：

```text
M2
R1
R2
```

都在：

```text
repair_dev_v1
```

上。

使用第 64 节 constraint-first selection。

锁定：

```text
best_repair
```

---

## 81. Best Repair Manifest

生成：

```text
BEST_REPAIR_LOCK.json
```

包含：

```text
checkpoint
checkpoint SHA256
training data SHA256
config SHA256
repair_dev metrics
selection rule
```

---

## 82. Phase J：独立 RL Decision

Best Repair 选完后：

```text
第一次
```

在：

```text
rl_dev_v1
```

运行。

---

## 83. 为什么 RL Dev 独立

因为：

```text
repair_dev_v1
```

已经被用于：

```text
R1/R2 选择
```

如果继续用它决定：

```text
reward weights
GRPO hyperparameters
```

会进一步开发集过拟合。

所以：

```text
repair_dev
```

与：

```text
rl_dev
```

分离。

---

## 84. 在 RL Dev 上做 Error Attribution

统计剩余：

```text
High-risk Miss
```

分别属于：

```text
output_contract
perception
consistency
decision
evidence
mixed
```

---

## 85. RL Decision Heuristic

生成：

```json
{
  "decision_candidate_ratio": 0.0,
  "perception_error_ratio": 0.0,
  "consistency_error_ratio": 0.0,
  "rl_recommended": false,
  "reason": ""
}
```

---

## 86. 推荐 RL 的默认启发式

满足：

```text
Decision Candidate 是最大类别
```

并且：

```text
decision_candidate_ratio >= 0.40
```

同时：

```text
perception_error_ratio < 0.35
```

则：

```text
rl_recommended = true
```

这些不是理论定理。

只是：

```text
Codex 自动生成建议
```

---

## 87. Codex 不能根据 heuristic 自动启动 RL

即使：

```text
rl_recommended=true
```

仍只生成：

```text
READY_FOR_RL
```

正式训练要求：

```bash
ALLOW_RL=1
```

---

## 88. Phase K：RL 数据构造

RL training：

```text
只能来自 Train
```

RL evaluation：

```text
rl_dev_v1
```

---

## 89. GRPO Training Dataset 默认构成

建议：

```text
60% hard / decision-focused
40% calibration/replay
```

其中 hard 部分：

```text
High-risk Miss
Decision Candidate
hard consistency cases
```

Calibration：

```text
normal/pass
manual_review
missing_info
already-correct high-risk
```

---

## 90. 第一版规模

```text
grpo_train_v1 = 500～1000
```

先不做 3000+。

因为每个 prompt 会进行多次 rollout。

---

## 91. Disk Dataset Schema

```json
{
  "case_id": "MV_MAIN_xxx",
  "prompt": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "image"},
        {
          "type": "text",
          "text": "..."
        }
      ]
    }
  ],
  "image_paths": [
    "a.png",
    "b.png"
  ],
  "ground_truth": {}
}
```

---

## 92. Trainer Dataset 转换

TRL 当前 VLM GRPO 要求样本含 `prompt` 和 `image`/`images`，其中图片由 PIL Image 或 PIL Image 列表传入，并由 trainer 的 image processor 处理。

因此 loader 必须：

```text
image_paths
→
PIL.Image
→
images
```

不能指望 TRL 自动读自定义字符串路径。

---

## 93. Qwen3-VL Prompt 处理

必须使用：

```text
AutoProcessor
+
现有 Qwen3-VL chat template
```

Qwen3-VL 官方示例本身使用 `AutoProcessor` 并通过多模态 message content 提供图片。

Codex 不要手写：

```text
<image>
```

token 拼接逻辑，除非现有项目就是这样实现且经验证。

---

## 94. GRPO Dataset Test

新增：

```text
tests/test_grpo_dataset.py
```

随机 8 cases：

```text
load
→ image open
→ processor
→ chat template
→ tokenize
```

必须全部通过。

---

## 95. Phase L：Composite Reward

只实现一个主函数：

```python
business_reward_func(...)
```

不要实现 7 个相互独立、重复解析 JSON 的 reward functions。

TRL 确实支持多个 reward function 并将 reward 聚合，但本项目多个业务指标高度关联；单一 Composite Reward 更容易共享 parser/evaluator 并实现 parsing gate。

---

## 96. Reward 计算顺序

严格：

```text
1. parse
2. schema
3. semantic score
4. business penalty
5. logging
6. return reward
```

---

## 97. Invalid JSON

如果：

```text
json_valid == false
```

立即：

\[
R=-1.5
\]

然后：

```text
不继续 semantic scoring
```

---

## 98. Schema Reward

如果 JSON 合法：

\[
R_{schema}
=
\begin{cases}
1 & schema\ valid\\
0 & otherwise
\end{cases}
\]

---

## 99. Field Reward

必须来自 case-level scorer：

\[
R_{field}\in[0,1]
\]

建议：

```text
正确字段数 / 可评测字段数
```

如果旧 evaluator 有自己的标准：

```text
必须复用旧标准
```

---

## 100. Anomaly Reward

采用：

\[
F1(A_{pred},A_{gt})
\]

特殊情况：

```text
两者都为空/none → 1
只有一个为空 → 0
```

如果项目原 evaluator 的逻辑不同：

```text
以项目 evaluator 为准
```

---

## 101. Audit Reward

\[
R_{audit}
=
I(audit_{pred}=audit_{gt})
\]

---

## 102. Evidence Reward

直接使用：

```text
per-case Evidence Support
```

归一化：

\[
[0,1]
\]

---

## 103. Core Reward

\[
R_{quality}
=
0.20R_{schema}
+
0.20R_{field}
+
0.20R_{anomaly}
+
0.25R_{audit}
+
0.15R_{evidence}
\]

所以：

\[
0\le R_{quality}\le1
\]

---

## 104. 第一版不单独 Reward Risk Level

删除：

```text
R_risk
```

避免：

```text
risk reward
+
audit reward
+
HRM penalty
```

三次重复编码相近信号。

Risk 继续作为：

```text
评测指标
诊断字段
```

---

## 105. High-risk Miss Penalty

必须由：

```text
shared CaseScore
```

返回：

```text
high_risk_miss
```

不能重新实现。

\[
P_{HRM}=I(high\_risk\_miss)
\]

---

## 106. False Escalation Penalty

\[
P_{FE}
=
I(audit_{gt}=pass\land audit_{pred}\neq pass)
\]

---

## 107. Hallucination Penalty

\[
P_H=
I(hallucination>0)
\]

第一版采用 binary penalty。

---

## 108. Reward v1

\[
\boxed{
R=
R_{quality}
-
1.0P_{HRM}
-
0.25P_{FE}
-
0.25P_H
}
\]

自然范围：

\[
[-1.5,1.0]
\]

不额外 clip。

---

## 109. Reward 配置

```text
configs/reward/audit_reward_v1.yaml
```

```yaml
invalid_json_reward: -1.5

quality:
  schema: 0.20
  field: 0.20
  anomaly: 0.20
  audit: 0.25
  evidence: 0.15

penalty:
  high_risk_miss: 1.0
  false_escalation: 0.25
  hallucination: 0.25
```

---

## 110. Reward 必须验证权重和

启动时：

```python
assert abs(sum(quality_weights.values()) - 1.0) < 1e-8
```

---

## 111. Reward Golden Tests

必须建立至少：

```text
8 cases
```

### T1 Perfect normal

```text
pass
→ 高 reward
```

### T2 Normal → reject

必须：

```text
false_escalation=1
```

### T3 High/reject correct

```text
无 HRM
```

### T4 High → pass

```text
HRM penalty
```

### T5 missing_document → missing_info

如果 GT 就是 missing_info：

```text
不得被 HRM penalty
```

### T6 unreadable → manual_review

如果 GT 为 manual_review：

```text
不得被错误 HRM penalty
```

### T7 Invalid JSON

```text
reward=-1.5
```

### T8 Correct audit + hallucinated evidence

Reward：

```text
低于完全正确版本
```

---

## 112. Reward Monotonicity Tests

还要 assert：

```text
perfect > hallucination
perfect > HRM
perfect > false escalation
valid correct > invalid JSON
```

---

## 113. Reward Logging

记录：

```text
reward_total
reward_quality

reward_schema
reward_field
reward_anomaly
reward_audit
reward_evidence

penalty_hrm
penalty_false_escalation
penalty_hallucination
```

以及：

```text
json_invalid_rate
audit_action_distribution
high_risk_prediction_rate
completion_length
```

---

## 114. TRL Custom Reward Columns

当前 TRL 对 custom reward 如果需要 `ground_truth`、`case_id` 等额外 dataset columns，应保持 `remove_unused_columns=False`；当前文档默认也是 `False`，但项目配置仍应显式 pin。

---

## 115. Phase M：GRPO 环境兼容性

当前 TRL 已支持 VLM GRPO；官方 VLM tested list 明确列了 Qwen2-VL/Qwen2.5-VL 等，并提醒不是所有 VLM 都保证兼容。当前文档其它 supported-model 列表又已经出现 Qwen3-VL，因此对 **Qwen3-VL + 本项目多图 + 已训练 PEFT adapter** 做本地 compatibility smoke 仍然必要。

---

## 116. RL 模型加载

不要：

```text
Repair adapter
+
new PEFT config
```

再次 wrap。

推荐：

```python
base = load_base()

model = PeftModel.from_pretrained(
    base,
    BEST_REPAIR_PATH,
    is_trainable=True,
)

trainer = GRPOTrainer(
    model=model,
    peft_config=None,
    ...
)
```

TRL 当前 `GRPOTrainer` 接受 `PeftModel` 作为 policy model。

---

## 117. Compatibility Smoke 数据

```text
16 cases
```

即可。

不要一开始 256。

---

## 118. Compatibility Smoke Steps

```text
max_steps=2
```

只验证工程链路。

---

## 119. Compatibility Checklist

必须验证：

```text
Base load
Repair adapter load
trainable params > 0
multi-image loading
processor
chat template
rollout
completion parse
ground_truth forwarding
reward
backward
LoRA gradient
optimizer step
checkpoint save
checkpoint reload
post-reload inference
```

---

## 120. Parameter-change Assertion

Smoke 前：

```text
保存一组 LoRA tensor snapshot
```

Smoke 后：

```text
至少一个 trainable LoRA tensor 发生变化
```

否则：

```text
TRAINING_NO_EFFECT
```

---

## 121. Compatibility Smoke 禁止 vLLM

第一轮：

```yaml
use_vllm: false
```

减少变量。

---

## 122. Phase N：Reward Smoke

兼容通过后：

```text
128 cases
```

如果运行成本可接受：

```text
256
```

---

## 123. Reward Smoke Training

建议：

```text
max_steps = 10
```

目的不是提升 benchmark。

而是：

```text
观察 reward signal
```

---

## 124. num_generations

起始：

```yaml
num_generations: 4
```

当前 TRL 默认值可能不是 4，因此必须显式 pin；文档目前默认 `8`，并要求 effective batch size 能被 `num_generations` 整除。

---

## 125. Batch Divisibility Assertion

启动前：

\[
B_{effective}
=
world\_size
\times
per\_device\_batch
\times
grad\_acc
\]

要求：

\[
B_{effective}\bmod num\_generations=0
\]

否则：

```text
CONFIG_INVALID
```

---

## 126. 单 GPU 示例

如果：

```text
world_size=1
batch=1
num_generations=4
```

则：

```text
grad_acc
```

至少选择能使 effective batch 成为 4 的配置。

例如：

```text
grad_acc=4
```

---

## 127. Reward Smoke 输出

生成：

```text
reward_smoke_stats.json
```

包括：

```text
mean reward
std reward
min
max

group reward std mean
zero variance group ratio

json invalid rate
HRM rate
false escalation rate
hallucination rate

pass rate
manual_review rate
missing_info rate
reject rate

completion p50
completion p90
completion p95
completion p99
```

---

## 128. Zero Variance

定义：

```text
group std < 1e-6
```

为：

```text
zero_variance_group
```

---

## 129. Zero Variance Warning

如果：

```text
zero_variance_group_ratio > 0.50
```

标记：

```text
REWARD_SIGNAL_WEAK
```

先诊断。

---

## 130. Zero Variance 诊断顺序

检查：

```text
temperature
rollout diversity
dataset difficulty
reward granularity
```

而不是：

```text
增加训练 steps
```

---

## 131. Reward Hacking Detection

Reward smoke 期间必须检查：

```text
reject rate 是否突然上升
manual_review rate 是否突然上升
输出长度是否异常变化
Evidence 数量是否异常增加
Hallucination 是否上升
JSON 是否恶化
```

---

## 132. Phase O：选择 Loss Type

不要预先宣称：

```text
一定 Dr.GRPO
```

当前 TRL 的 `GRPOConfig` 默认 `loss_type="dapo"`；`grpo`、`dr_grpo`、`dapo` 都是支持的 loss，其中 vanilla `grpo` 存在文档明确指出的 length bias，而 `dr_grpo` 和 `dapo` 使用不同的归一化方式缓解该问题。

---

## 133. 第一版默认

保持：

```yaml
loss_type: dapo
```

因为：

```text
减少额外实验变量
```

---

## 134. Dr.GRPO 触发条件

Reward Smoke 统计：

```text
completion length variation
reward-length correlation
truncation
```

如果：

```text
completion length 差异很大
```

或者：

```text
存在明显 reward-length correlation
```

再建立：

```text
dapo vs dr_grpo
```

小规模对照。

---

## 135. scale_rewards

当前 TRL 默认 group scaling，并支持：

```text
"group"
"batch"
false
```

文档指出 Dr.GRPO 主张不按 group std 进行 reward scaling。

本项目第一版显式：

```yaml
scale_rewards: false
```

因为业务 Reward 已经有固定业务尺度。

---

## 136. beta

不要依赖 TRL 默认。

第一版明确：

```yaml
beta: 0.0
```

如果后续出现明显 policy drift：

```text
再考虑小 KL constraint
```

不要一开始同时调：

```text
reward weights
beta
loss type
temperature
```

---

## 137. RL Learning Rate

第一版建议只定义一个保守 candidate：

```yaml
learning_rate: 1.0e-6
```

属于：

```text
initial engineering choice
```

不是论文结论。

如果：

```text
LoRA 参数确实有梯度
但 50 steps 几乎没有 policy/reward 变化
```

才允许建立第二 candidate，例如：

```text
3e-6
```

不做大规模 sweep。

---

## 138. Completion Length

先统计 Best Repair deterministic + sampled outputs。

设置：

```text
max_completion_length
```

使：

```text
>= p99 + safety margin
```

---

## 139. Truncation Gate

如果：

```text
truncation_rate > 1%
```

重新检查 max completion length。

---

## 140. GRPO Pilot

Reward Smoke 通过后：

```text
max_steps=50
```

作为第一轮真正 RL pilot。

---

## 141. Pilot Checkpoints

建议：

```text
step 25
step 50
```

两个 checkpoint。

不要每 2 steps 跑完整 dev。

---

## 142. Pilot Evaluation

在：

```text
rl_dev_v1
```

比较：

```text
Best Repair
RL-step25
RL-step50
```

---

## 143. RL Candidate Eligibility

必须：

```text
Audit Δ >= -0.01
Evidence Δ >= -0.01
False Escalation Δ <= +0.01
JSON 无明显下降
Hallucination 无明显恶化
```

---

## 144. Eligible Candidate 排序

只在 eligible 内：

```text
High-risk Miss 最低
```

tie：

```text
Audit Accuracy 更高
```

---

## 145. Pilot Fail

如果 step25 和 step50 都：

```text
HRM 未改善
```

或者出现：

```text
明显 over-escalation
```

则：

```text
RL_PILOT_FAIL
```

停止 RL。

---

## 146. Pilot Success

如果：

```text
HRM 改善
+
所有 guardrail 合格
```

才考虑：

```text
更多 RL steps
```

---

## 147. Full RL 不预先规定 500/1000 steps

先根据 50-step pilot：

```text
reward curve
dev curve
policy drift
```

决定。

原则：

```text
最少够用
```

而不是：

```text
烧固定 steps
```

---

## 148. Minimum Ablation

如果 RL 成功，仅做：

```text
A = Best Repair

B = RL without HRM penalty

C = RL Full Reward
```

目的：

> 验证 HRM 业务 penalty 是否真正贡献改善。

---

## 149. 不把 DAPO/Dr.GRPO 对比强制列为 Ablation

只有发现：

```text
length-related problem
```

才做。

节约资源。

---

## 150. Phase P：最终模型冻结

在：

```text
repair_dev
+
rl_dev
```

完成所有决策。

然后建立：

```text
FINAL_MODEL_LOCK.json
```

---

## 151. FINAL_MODEL_LOCK

包含：

```json
{
  "selected_model": "...",
  "adapter_sha256": "...",
  "base_model": "...",
  "training_config_sha256": "...",
  "reward_config_sha256": "...",
  "evaluation_config_sha256": "...",
  "selection_finished": true
}
```

---

## 152. Final Holdout 前置检查

`scripts/20_run_final_holdout.sh` 必须 assert：

```text
FINAL_HOLDOUT_LOCKED exists
FINAL_MODEL_LOCK exists
ALLOW_FINAL_HOLDOUT set
```

---

## 153. Final Holdout 模型

预先允许比较：

```text
M2
Best Repair
Final RL（若存在）
```

这三个模型的身份在打开 final results 前已经固定。

---

## 154. Final Holdout 结果不能改变训练

运行后：

```text
禁止继续训练
```

如果 RL final 表现反而差：

```text
如实报告
```

不能：

```text
回去调 reward 再测
```

---

## 155. Final Result 目录

```text
docs/experiments/final_holdout_v1/

manifest/
metrics/
predictions/
figures/
paired_transitions/
summary.md
```

---

## 156. Codex 必须使用的统一 Run Manifest

所有阶段：

```json
{
  "run_id": "",
  "phase": "",
  "start_time": "",
  "end_time": "",

  "git_commit_before": "",
  "git_commit_after": "",

  "input_files": [],
  "output_files": [],

  "dataset_hash": "",
  "config_hash": "",
  "checkpoint_hash": "",

  "commands": [],

  "dry_run": true,
  "success": false,

  "gate_result": "",
  "failure_reason": ""
}
```

---

## 157. Sentinel 文件

建议使用：

```text
READY
RUNNING
FAILED
PASSED
READY_TO_ARCHIVE
```

延续现有服务器工程习惯。

---

## 158. 原子写入

运行状态：

```text
先写 temp
再 rename
```

避免 SSH/进程崩溃后留下：

```text
半个 JSON
```

---

## 159. Training Resume

Repair/RL checkpoint 必须支持：

```text
resume_from_checkpoint
```

但：

```text
resume 不得改变 config
```

如果 config hash 改变：

```text
必须新 run_id
```

---

## 160. Dry-run 不得污染正式目录

Dry-run：

```text
outputs/runtime/dry_run/<run_id>/
```

正式：

```text
outputs/runtime/<phase>/<run_id>/
```

---

## 161. Codex 建议新增模块

```text
src/mv_audit/analysis/
    dataset_registry.py
    build_validation_splits.py
    build_final_holdout.py
    leakage_check.py

    error_attribution.py
    paired_migration.py
    bootstrap_metrics.py

src/mv_audit/evaluation/
    case_scorer.py

src/mv_audit/rewards/
    __init__.py
    audit_reward.py

src/mv_audit/training/
    train_grpo.py

src/mv_audit/data/
    grpo_dataset.py
```

---

## 162. Tests

```text
tests/
    test_case_scorer_regression.py
    test_dataset_boundary.py
    test_error_attribution.py
    test_repair_dataset.py
    test_audit_reward.py
    test_grpo_dataset.py
    test_grpo_config.py
```

---

## 163. test_dataset_boundary.py 必测

```text
Train vs Validation overlap = 0
Repair train vs Repair dev = 0
GRPO train vs RL dev = 0
sample500 vs final holdout = 0
used_test vs final holdout = 0
repair vs final holdout = 0
GRPO vs final holdout = 0
```

---

## 164. test_grpo_config.py 必测

```text
num_generations divisibility
max_completion_length > 0
reward weights sum = 1
remove_unused_columns == false
loss_type 属于允许集合
```

---

## 165. Preflight Command

建议建立：

```bash
python -m mv_audit.tools.preflight \
    --phase repair_r1 \
    --config ...
```

或等价脚本。

---

## 166. Repair Preflight 输出

```text
[PASS] dataset exists
[PASS] images exist
[PASS] no leakage
[PASS] M2 adapter exists
[PASS] M2 hash recorded
[PASS] trainable adapter load
[PASS] optimizer updates = 60
[PASS] output dir is new

READY_FOR_REPAIR_R1
```

---

## 167. GRPO Preflight

```text
[PASS] Best Repair locked
[PASS] RL recommended
[PASS] dataset valid
[PASS] images valid
[PASS] reward tests pass
[PASS] batch divisible by num_generations
[PASS] completion length config valid
[PASS] Qwen3-VL compatibility smoke required/passed

READY_FOR_RL
```

---

## 168. Final Holdout Preflight

```text
[PASS] final dataset locked
[PASS] final model locked
[PASS] model selection complete
[PASS] no overlap
[PASS] ALLOW_FINAL_HOLDOUT

READY_FOR_FINAL
```

---

## 169. Codex 第一次执行的精确范围

第一次不要让 Codex：

```text
训练 Repair
写完整 GRPO
运行 final
```

第一轮任务只做：

```text
A. Repository Audit
B. Data Boundary
C. Validation Split
D. Final Holdout Lock
E. Case Scorer Refactor
F. M2 Validation Baseline 支持
G. Repair Dataset Audit 支持
H. Preflight
I. Tests
```

---

## 170. 第一轮 Codex 不允许做

```text
正式 Repair R1 training
正式 Repair R2
GRPO
改 Reward
sample500
final_holdout inference
pip upgrade
```

---

## 171. 第一轮完成标准

Codex 必须最终提供：

```text
1. 变更文件列表
2. 为什么修改这些文件
3. 测试执行结果
4. Validation inventory
5. repair_dev manifest
6. rl_dev manifest
7. final_holdout manifest
8. leakage report
9. Repair 120+120 distribution report
10. M2 baseline 执行命令
11. Repair R1 preflight 命令
12. 尚未执行的昂贵操作清单
```

---

## 172. 第二轮 Codex 范围

第一轮验证无误后：

```text
1. 补跑 M2 validation baseline
2. Repair R1 dry-run
3. Repair R1 正式训练
4. Fast gate
5. Main gate
6. Paired migration
7. Error attribution
8. 自动生成 R1 decision report
```

停止。

---

## 173. R1 Decision Report

生成：

```text
repair_r1_decision.json
```

例如：

```json
{
  "status": "PARTIAL",
  "m2_high_risk_miss": 0.24,
  "r1_high_risk_miss": 0.22,
  "delta_hrm": -0.02,

  "delta_audit_accuracy": -0.002,
  "delta_evidence": -0.003,
  "delta_false_escalation": 0.001,

  "gate": {
    "accuracy": true,
    "evidence": true,
    "false_escalation": true
  },

  "recommended_next_step": "REPAIR_R2"
}
```

---

## 174. 第三轮 Codex 范围

只根据 R1 Decision：

### SUCCESS

```text
锁 Best Repair
→ RL Decision
```

### PARTIAL

```text
构造 R2
→ R2 training/evaluation
```

### FAIL

```text
Diagnosis only
→ STOP
```

---

## 175. RL Decision Report

最终生成：

```text
rl_decision.json
```

```json
{
  "best_repair": "",
  "residual_hrm_cases": 0,

  "error_distribution": {
    "perception": 0,
    "consistency": 0,
    "decision": 0,
    "mixed": 0
  },

  "decision_candidate_ratio": 0.0,

  "rl_recommended": false,

  "reason": ""
}
```

---

## 176. 第四轮 Codex

只有：

```text
rl_recommended=true
+
用户允许
```

才：

```text
GRPO dataset
Reward
Compatibility Smoke
Reward Smoke
Pilot
```

---

## 177. 不要求 Codex 一次完成整条路线

正确工作方式：

```text
实现一阶段
→ 生成 artifact
→ gate
→ 再执行下一阶段
```

而不是：

```text
写一个巨型脚本
一晚上自动从 M2 跑到 RL
```

---

## 178. 服务器费用控制

每个阶段生成：

```text
estimated_workload.json
```

不需要预测人民币价格，只报告：

```text
cases
images
optimizer steps
rollouts
num_generations
approx generated completions
```

例如：

\[
1000\ prompts\times4=4000\ rollouts
\]

这样在启动 RL 前能直观看成本。

---

## 179. Progress Logging

建议每：

```text
5～10 optimizer steps
```

记录一次训练 summary。

不要每个 token/样本大量打印。

---

## 180. Error Sample 保存限制

开发阶段：

```text
保存代表性 error sample
```

但不需要所有图像复制进 Git。

只存：

```text
case_id
paths
prediction
GT
score
error tags
```

---

## 181. Git 边界

继续遵守 README：

Git 保留：

```text
code
configs
small manifests
metrics
figures
reports
```

不保留：

```text
model weights
full image datasets
huge predictions
```

---

## 182. 最终 README 更新规则

任何实验：

```text
只在 PASSED + archived
```

后写入 README。

Dry run：

```text
不得描述成模型效果
```

Smoke：

```text
不得描述成正式实验
```

---

## 183. DPO 的最终定位

README 最终继续保留：

```text
DPO v1 negative result
DPO v2 partial recovery / HRM failure
```

不删除失败实验。

DPO 已经证明训练 pair 分离成功并不意味着实际审计 KPI 改善。

---

## 184. 最终方法叙事

最终项目应该能够形成：

```text
Qwen3-VL Base
    ↓
LoRA-SFT
    ↓
M2
    ↓
DPO Preference Alignment
    ↓
发现 Proxy Objective Misalignment
    ↓
Paired Error Migration
    ↓
Targeted High-risk Repair
    ↓
Residual Error Attribution
    ↓
Capability Bottleneck?
    ├── Yes → More supervised/data repair
    └── No
         ↓
Decision Bottleneck?
         ├── No → Stop
         └── Yes
              ↓
      Verifiable-reward Online RL
              ↓
        Final Holdout
```

---

## 185. 给 Codex 的第一轮任务文本

### TASK

实现 MultiVoucher-Audit 后训练闭环的 **Phase A–G preparation only**。

本轮禁止任何正式训练和 final evaluation。

### 目标

完成：

```text
repository audit
validation inventory
repair_dev_v1 / rl_dev_v1
used-case registry
final_holdout_v1 lock
dataset leakage checks
shared case-level scorer
case scorer regression tests
error attribution skeleton
repair data distribution audit
Repair R1 preflight
```

### 执行前

必须阅读：

```text
README.md
docs/code_inventory.md
existing evaluation modules
existing inference modules
existing SFT/DPO training modules
existing high-risk repair modules/scripts
```

### 禁止

```text
不要覆盖已有文件
不要改变 M2 checkpoint
不要启动 Repair training
不要启动 GRPO
不要跑 final_holdout inference
不要升级依赖
不要重写第二套 evaluator
```

### 数据约束

```text
Repair/RL training → Train
Repair/RL model selection → Validation
Historical sample500 → 不再用于模型选择
Final Holdout → unused Test cases
```

### Validation

优先从：

```text
val_in_template
val_unseen_template
```

中构造互斥的：

```text
repair_dev_v1
rl_dev_v1
```

必须扫描实际可用图片后决定最终规模。

### Final Holdout

排除所有历史使用过的 test case。

生成：

```text
manifest
SHA256
lock sentinel
```

但禁止 inference。

### Evaluation

从现有 evaluator 抽取：

```python
score_case()
```

并通过 regression test 证明聚合结果与旧 evaluator 一致。

### Repair Audit

检查现有：

```text
120 repair
120 calibration
```

输出分布、图片存在性、JSON/schema 和 leakage report。

### Preflight

能够在不训练的情况下判断：

```text
M2 adapter 是否可继续训练
预计 optimizer update 数
output directory 是否安全
所有数据是否有效
```

### 本轮完成后输出

必须报告：

```text
files changed
tests run
tests passed/failed

repair_dev size
rl_dev size
final_holdout size

all overlap counts

repair sample distribution
calibration sample distribution

M2 baseline commands
Repair dry-run command
Repair formal-run command

expensive operations NOT executed
```

如果发现任何旧实现与本文方案冲突：

```text
不要静默修改业务定义
```

而是：

```text
记录冲突
优先保持现有 evaluator 行为
说明需要修改方案还是代码
```

### END TASK
