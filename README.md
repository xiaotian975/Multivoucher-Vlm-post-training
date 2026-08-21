# MultiVoucher-Audit：多凭证报销审计 VLM 后训练

更新时间：2026-08-21

> 一个从数据合成、LoRA-SFT、结构化错误修复、Model-Mined DPO、业务评测到模型归档的完整 VLM 后训练项目。基座模型为 `Qwen3-VL-8B-Instruct`，输入发票、支付截图、报销申请单和订单截图，输出可校验、可定位、可追责的 Evidence-Grounded JSON。

## 先看这里：项目结论

### 招聘者一分钟摘要

这个项目解决的不是单图 OCR，而是多张业务凭证之间的字段抽取、跨图一致性判断、风险分级和证据定位。项目从零构造了带字段、异常、风险、审核动作和 bbox 真值的合成数据，完成 `LoRA-SFT -> Repair SFT -> DPO` 的完整后训练链，并建立 JSON/schema、证据、风险漏检和错误迁移等分层门禁。

| 项目成果 | 可审计结果 |
| --- | --- |
| 原生模型到可用结构化审计 | M0 zero-shot 的 Audit Accuracy 为 `0.0000`，M2 LoRA-SFT 在历史 sample500 上达到 `0.7735`，JSON Validity 为 `1.0000` |
| 定位并修复高风险漏检 | Structured Repair SFT v3 在 152 条 Train-only `train_decode_dev` 上达到 Audit `96.71%`、HRM `5.75%`、Evidence Support `98.76%` |
| 完成偏好对齐研究 | Model-Mined DPO v3 在独立 probe 上使 task reward 提升 `+0.167`、Order-ID Pair Rate 提升 `+11.1pp` |
| 建立模型治理结论 | SFT v3 在 final_holdout_v1 与 sample500 诊断中均退化，最终标记 `FINAL_HOLDOUT_FAILED / NOT_DEPLOYED`；DPO checkpoint-15 为 `ALIGNMENT_RESEARCH_CANDIDATE`，因 full gate 未通过而不替代 SFT |
| 完成可复现归档 | M2 -> R1 -> R2 -> R3 -> DPO weak-40 -> DPO strong-15 六级 adapter 全部在本地通过 SHA256、配置和 safetensors 头部校验 |

“Production candidate”只表示曾通过 Train-only 开发门禁；后续 final_holdout_v1 已消耗且失败，模型没有部署。历史 sample500、Train-only train_decode_dev 与 final_holdout_v1 是不同 benchmark，本文始终分表报告。

### 我在项目中负责什么

- 将报销审核拆成多图字段抽取、跨图一致性、风险规则、审核动作、证据闭环和结构契约六类问题。
- 设计合成数据生成、异常注入、case-level split、图片渲染和 bbox 真值链路。
- 审计并迭代 LoRA-SFT、Repair SFT、DPO/IPO/GRPO 代码，设置 dry-run、reward smoke、fast gate 和停止条件。
- 针对 DPO loss 与业务 KPI 背离，完成 easy-pair 饱和、sequence margin 虚高和决策边界漂移诊断。
- 设计 model-error-mined hard pairs、assistant-token mean log-prob、alignment probe/full gate 两级选择。
- 完成五卡 DDP、模型切分、case sharding、checkpoint probe 并行和远端归档自动化。
- 使用 AI 辅助代码草拟、排错和文档整理；需求边界、代码审计、实验选择、指标核验和候选决策由人工控制，并由可复现产物验证。

### 阅读路线

| 读者 | 建议入口 |
| --- | --- |
| 招聘者/面试官 | 本节、[结果总表](#实验结果总表)、[创新点](#项目级创新点)、[工程复盘](#问题原因与解决方案) |
| 初学者 | [完整训练谱系](#完整后训练谱系)、[多卡实现](#五卡训练模型并行与任务并行)、[指标说明](#指标来源公式与业务含义) |
| 复现实验 | [服务器手册](#服务器训练推理评测与归档)、[代码地图](#代码结构与输入输出)、[Phase 10 报告](docs/experiments/phase10_model_error_mined_dpo_v3/README.md) |
| 查模型 | [归档审计](docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive_audit.md)、[模型选择](docs/experiments/phase10_model_error_mined_dpo_v3/model_selection.json) |

## 项目背景与任务定义

企业费用审核需要同时核对发票、支付截图、报销申请单和订单截图。模型接收一个 case 下的 2 到 4 张图片，输出固定 schema JSON：

```text
images + case_id
  -> field_extraction
  -> consistency_check
  -> anomaly_types
  -> risk_level
  -> audit_result
  -> reason
  -> evidence[{source_image_id, source_doc_type, field, value, bbox}]
  -> uncertainty
```

任务难点有四层：视觉层要读出小字号订单号和金额；关系层要判断同一字段在不同凭证是否一致；决策层要把异常映射为风险和审核动作；接口层要保证 JSON/schema 始终可被程序消费。只优化某一层，常会出现“异常识别正确但审核放行”或“语义正确但 schema 非法”。

## 技术栈与系统架构

- 基座：`Qwen3-VL-8B-Instruct`
- 参数高效训练：PEFT LoRA，target modules 为 `q/k/v/o/gate/up/down_proj`
- 训练：PyTorch、Transformers、BF16、gradient checkpointing、DDP、`device_map=balanced`
- 对齐：DPO、weighted preference pair、SFT auxiliary loss、mean-token log-prob
- 数据与评测：JSONL、JSON Schema Draft 2020-12、case-level split、bbox/证据评分
- 工程：Bash、PowerShell watcher、pytest、SHA256、draw.io、可恢复 shard merge

![完整后训练架构](docs/experiments/phase10_model_error_mined_dpo_v3/figures/post_training_pipeline.png)

可编辑源文件：[post_training_pipeline.drawio](docs/experiments/phase10_model_error_mined_dpo_v3/figures/post_training_pipeline.drawio)

## 项目级创新点

以下是项目级工程创新，不把通用 DPO、LoRA 或 Accuracy 包装为原创算法：

1. **多凭证证据闭环**：审核结论必须回指图片、凭证类型、字段、值和 bbox。
2. **结构化 Repair SFT**：把错误归因拆成 schema、证据、感知和决策，只对真实残留错误构造小规模 repair mix。
3. **Order-ID 双侧证据约束**：订单号不一致时，target 和 verifier 同时要求两类凭证中的两个不同订单号。
4. **Model-Error-Mined Hard Pairs**：从当前 SFT 模型真实生成错误中挖 rejected，避免规则篡改 easy rejected 过早饱和。
5. **Mean-token DPO**：以 assistant token 平均 log-prob 计算偏好，降低长 JSON 累加 margin 失真。
6. **Probe/full-gate 分层选择**：先检查目标偏好是否学到，再以 152 条 full gate 防止局部收益掩盖业务负迁移。
7. **可审计候选治理**：adapter、配置、日志、predictions、metrics、errors 和 SHA256 构成完整谱系。

## 完整后训练谱系

### 总时间线

```text
业务 schema 与风险规则
  -> 合成 case / 异常注入 / 图片与 bbox
  -> M0 zero-shot / M1 few-shot
  -> M2 LoRA-SFT
  -> DPO v1 负迁移
  -> DPO v2 保护性修正
  -> Repair SFT R1 / R2
  -> schema guard 与 order-id 错误归因
  -> Structured Repair SFT R3
  -> 240 x 4 model mining
  -> 120 hard train pairs + 24 case-disjoint probe
  -> DPO v3 weak-40 / strong checkpoint-15
  -> probe gate / 152-case full gate
  -> production 与 research 候选分流归档
```

每一步都遵循“先定位、再 dry-run、再训练、再 fast gate、最后决定是否继续”的顺序。sample500/Test 没有回流到 Phase 9/10 的训练、pair 挖掘或 reward 调参。

### 各阶段的输入、输出和决策

| 阶段 | 出发点与输入 | 训练/执行 | 主要输出 | 结果与决策 |
| --- | --- | --- | --- | --- |
| 数据生成 | 业务词典、case schema、异常规则 | 生成 case，case-level split，渲染图片与 bbox | raw cases、annotations、images、SFT/DPO/GRPO JSONL | 建立可控真值和隐私友好的训练底座 |
| M0/M1 | 原始 Qwen3-VL；zero/few-shot prompt | 仅推理 | sample500 predictions/metrics | Audit `0.0000/0.0785`，prompt 不足以建立任务 |
| M2 SFT | 21,682 train + 1,138 val | LoRA-SFT，1 epoch | `qwen3vl_8b_lora_existing_epoch1` | sample500 Audit `0.7735`，成为历史业务基线 |
| DPO v1 | 1,000 条规则式 easy rejected | DPO | M3 adapter、history、sample500 | loss 很低且 margin 极高，Audit 降到 `0.6685`，判为业务负迁移 |
| DPO v2 | 3,000 train + 300 holdout；hard/protective/calibration pairs | weighted DPO + SFT auxiliary | M3v2 与 ablation | Audit 恢复到 `0.7645`，HRM `0.2546`，仍不优于 M2 |
| Repair R1/R2 | 120 high-risk repair + 120 calibration，后续强化 order-id | 从 M2/R1 继续 LoRA-SFT | R1、R2 adapter 与 152 条 gate | schema 修复后确认残留核心是 order-id 证据读取 |
| Repair R3 | 480 条 Train-only structured mix | 从 R2 继续 1 epoch，五卡 DDP | R3 adapter、metrics/errors | Audit `0.9671`、HRM `0.0575`，选为开发候选 |
| Model mining | 未进入 repair/dev/holdout/test 的 240 条 Train-only case | 每条采样 4 个，共 960 completions | 894 个 schema 合法输出、pair audit | 从真实模型错误构造困难偏好，不再依赖简单篡改 |
| DPO v3 weak | 120 train pairs、24 probe，R3 policy/reference | 40 step，`beta=0.2`，mean-token | weak checkpoint-40、日志 | 获取可继续训练的弱对齐初始化 |
| DPO v3 strong | weak-40 继续训练 | 最多 20 step，5 step 一存；选择 checkpoint-15 | strong cp15、probe/full metrics | probe 正向，但 full gate 业务退化，归档为 research candidate |

### Structured Repair SFT v3

R3 不再只在 prompt 中提醒订单号，而是把 target 改成显式结构：

- `reason` 写明“订单截图订单号 A 与报销单订单号 B 不一致”。
- `evidence` 前置两条 order-id 证据，并要求不同来源、不同值。
- verifier 对 mismatch case 检查双侧证据，schema guard 只用于业务输出合法化。
- repair mix 保留 low/pass calibration 和 R1/R2 carryover，避免全量升级为高风险。

配置：[high_risk_repair_sft_v3_order_id_structured_from_r2_qwen3vl_8b_server.yaml](configs/train/high_risk_repair_sft_v3_order_id_structured_from_r2_qwen3vl_8b_server.yaml)

```bash
# dry-run 后，只有显式允许训练时启动
ALLOW_TRAINING=1 bash scripts/12_run_order_id_repair_sft_v3_server.sh
```

输入是 480 条 Train-only mix；输出是 R3 adapter、152 条 predictions、metrics 和 errors。实际训练完成 96 step，运行约 498 秒。

### Model-Mined DPO v3

DPO v1/v2 的核心问题不是“训练没收敛”，而是 pair 太容易。chosen 多为 M2 已见过的标准答案，rejected 多为规则篡改；holdout preference accuracy 很早达到 1.0，模型只学会区分人工坏答案。

DPO v3 改成：

1. 用 R3 对 240 条隔离的 Train-only case 每条采样 4 次。
2. 对原始输出做 JSON/schema、task reward、风险动作和 order-id evidence 审计。
3. 优先保留模型真实生成的困难 rejected，控制 pair 难度与类别比例。
4. 使用 assistant-token 平均 log-prob，避免长 JSON 仅因 token 多而产生巨大 margin。
5. high-risk/order-id pair 提高权重，并加入 `lambda_sft` 约束语言建模漂移。
6. checkpoint 先过 24 条 alignment probe，再对唯一候选运行 152 条 full gate。

核心配置：

| 阶段 | LR | beta | lambda_sft | grad accumulation | 上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| weak DPO | `1e-6` | `0.2` | `0.05` | 4 | 40 step |
| strong continuation | `5e-6` | `1.0` | `0.2` | 4 | 20 step |

入口：[scripts/13_run_model_mined_dpo_v3_server.sh](scripts/13_run_model_mined_dpo_v3_server.sh)、[scripts/16_run_model_error_mined_dpo_v3_strong_server.sh](scripts/16_run_model_error_mined_dpo_v3_strong_server.sh)、[scripts/17_resume_dpo_v3_strong_full_gate_server.sh](scripts/17_resume_dpo_v3_strong_full_gate_server.sh)。

## 实验结果总表

### 历史 sample500：M2、DPO v1/v2 与 R3 诊断

sample500 是四个 split、每个 500 条的历史 benchmark。表内数值为 split 平均；`repair_sft_r3` 是 final 失败后的诊断补跑，不用于重新选模或调参。

| 模型 | Audit Accuracy | High-risk Miss | Evidence Support | 结论 |
| --- | ---: | ---: | ---: | --- |
| M2 LoRA-SFT | **0.7735** | 0.2427 | **0.8035** | 历史业务 baseline |
| DPO v1 | 0.6685 | **0.2373** | 0.7987 | loss 收敛但 Audit 明显负迁移 |
| DPO v2 | 0.7645 | 0.2546 | 0.7952 | 恢复 Accuracy，HRM 未改善 |
| Structured Repair SFT v3 | 0.6075 | 0.4217 | 0.6801 | final 失败后诊断；相对 M2 明显退化 |

### Train-only train_decode_dev：仅比较 R3 与 DPO v3

| 模型 | JSON / Schema | Audit Accuracy | High-risk Miss | Evidence Support | Error cases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured Repair SFT v3 | 1.000 / 1.000 | **0.9671** | **0.0575** | 0.9876 | 23 |
| DPO v3 strong checkpoint-15 | 1.000 / 1.000 | 0.8684 | 0.1379 | **0.9904** | 40 |

### 2026-08-21 post-final 诊断归档

final_holdout_v1 已在远端完成并消耗，本次只归档已有结果和补跑 sample500 诊断，不训练、不调参、不重跑 final_holdout、不运行 DPO V3 checkpoint-15。

| Benchmark | 执行方式 | 本地归档 | Audit Accuracy | High-risk Miss | Evidence Support | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| final_holdout_v1 | 使用已有 4 split predictions 重新生成 metrics/errors/summary；不重新推理 | `docs/experiments/final_holdout_v1/`、`outputs/eval_reports/final_holdout_v1/`、`outputs/predictions/final_holdout_v1/` | 0.7160 | 0.3152 | 0.8358 | `FINAL_HOLDOUT_FAILED / NOT_DEPLOYED` |
| sample500 historical | `repair_sft_r3` 按历史 sample500 口径推理 4 x 500；不启用 DPO V3 checkpoint-15 | `docs/experiments/repair_sft_r3_sample500_diagnostic/`、`outputs/eval_reports/repair_sft_r3_sample500_diagnostic/`、`outputs/predictions/repair_sft_r3_sample500_diagnostic/` | 0.6075 | 0.4217 | 0.6801 | 相对 M2 baseline 明显退化 |

本次 sample500 使用配置 [repair_sft_r3_sample500_historical_server.yaml](configs/train/repair_sft_r3_sample500_historical_server.yaml) 和入口 [24_run_repair_sft_r3_sample500_diagnostic.sh](scripts/24_run_repair_sft_r3_sample500_diagnostic.sh)。机器可读补表见 [post_training_metrics.csv](docs/experiments/phase10_model_error_mined_dpo_v3/post_training_metrics.csv)，final 与 sample500 归档都包含 `artifact_manifest.json` 以记录本地文件 SHA256。sample500 与 final_holdout 结果均为 diagnosis/reporting-only，不得回流训练、reward tuning 或 checkpoint 选择。

two-stage supervised tuning: broad task SFT followed by targeted repair SFT

SFT两阶段监督微调：先用大规模任务 SFT 建立多凭证审计基础能力，再用小规模 Train-only Repair SFT 对高风险漏报和证据定位薄弱点进行定向修复。

Qwen3-VL base model → 主 LoRA-SFT → Repair SFT R1/R2/R3 精修 → 得到当前最好 SFT 候选 repair_sft_r3

Repair SFT 的有效性来自“先大规模 SFT 建立通用任务能力，再用小规模高置信错题样本进行定向行为修复”。它不是小数据从零训练出强模型，而是在已有 SFT 能力上做高信号纠偏。

### DPO v3 alignment probe

24 条 case-disjoint probe 中，checkpoint-15 相对 R3：

- mean task reward：`-0.5003 -> -0.3336`，提升 `0.1667`；
- Order-ID Pair Rate：`0 -> 0.1111`，提升 `11.11pp`；
- probe HRM：`0.7500 -> 0.6667`；
- JSON valid rate 保持 `1.0`，false escalation 保持 `0`。

但 full gate 中 baseline HRM 5 条、DPO HRM 12 条，修复 0 条、新增 7 条，因此局部对齐收益不足以支持业务替换。

![分 benchmark 结果](docs/experiments/phase10_model_error_mined_dpo_v3/figures/post_training_metrics_by_benchmark.png)

训练曲线、probe 曲线、四个案例和逐 case 错误迁移见 [Phase 10 完整报告](docs/experiments/phase10_model_error_mined_dpo_v3/README.md)。

## 五卡训练、模型并行与任务并行

![五卡执行拓扑](docs/experiments/phase10_model_error_mined_dpo_v3/figures/multi_gpu_execution_topology.png)

可编辑源文件：[multi_gpu_execution_topology.drawio](docs/experiments/phase10_model_error_mined_dpo_v3/figures/multi_gpu_execution_topology.drawio)

本项目实际用了四种并行方式。它们解决的问题不同，不能统称为“多卡训练”。

### 1. SFT v3：五进程 DDP

```text
torchrun --nproc_per_node=5
  rank0 -> GPU0: 完整 Qwen3-VL + LoRA
  rank1 -> GPU1: 完整 Qwen3-VL + LoRA
  ...
  rank4 -> GPU4: 完整 Qwen3-VL + LoRA
  NCCL all-reduce LoRA gradients
```

- 每张卡各有一份完整模型副本，五个进程读取不同 mini-batch。
- 只有 LoRA 参数参与更新：`43,646,976 / 8,810,770,672 = 0.4954%`。
- `per_device_train_batch_size=1`、`gradient_accumulation_steps=1`，全局 batch 为 5。
- 480 条样本完成 `480 / 5 = 96` step，这个 step 数和五个 rank 日志共同证明 DDP 确实工作。
- BF16 与 gradient checkpointing 降低激活显存；LoRA r=16、alpha=32、dropout=0.05。

DDP 的优势是数据吞吐高；代价是每卡都必须放下完整模型。若单卡连一份完整模型都放不下，增加 GPU 数并不会自动解决单卡 OOM。

### 2. DPO v2/v3：单进程模型切分

DPO 同时需要 policy 和 reference 两套模型。单卡 32GB 无法容纳两套 8B VLM 及前后向状态，因此采用一个 Python 进程配合 `device_map=auto/balanced`：

```text
one process
  policy layers    -> GPU0..GPU4
  reference layers -> GPU0..GPU4
  one batch flows through layer shards
```

这是模型分片，不是 DDP：GPU 不保存完整副本，也不处理五个独立 batch。它解决“放不下”，但 pipeline 上不同层的计算量、通信和等待不同，所以不能保证五张卡的 GPU-Util 完全均衡。显存占满只说明参数已放置，不能证明计算利用率高。

DPO v3 使用 `per_device_train_batch_size=1`、`gradient_accumulation_steps=4`、BF16、gradient checkpointing；weak 每 10 step 保存，strong 每 5 step 保存。训练上限和 checkpoint gate 防止为了追 loss 无限制烧算力。

### 3. 推理与 model mining：case sharding

240 x 4 model mining 和 152 条 fast gate 都按 case 切成 5 个 shard。每张 GPU 运行独立模型实例并处理不同 case，完成后由 [merge_inference_shards.py](tools/merge_inference_shards.py) 按 case id 确定性合并、检查重复和缺失。

这种方式是真正的任务并行，适合无梯度生成；某个 shard 中断时只需恢复该 shard，不必重跑全部样本。

### 4. Probe：checkpoint 级并行

10/20/30/40 或 strong 5/10/15/20 checkpoint 分配到不同 GPU 并行跑 24 条 probe。选择脚本读取每个 checkpoint 的同构指标，选出最早满足局部门禁的候选；只有这个候选再跑 152 条 full gate。

## 指标来源、公式与业务含义

指标分为通用模型指标、结构契约指标和项目自定义业务指标。通用指标没有被包装成原创；项目自定义部分用于把“模型看起来会答”转化为“业务可审计”。

记真值为 `y`、预测为 `ŷ`，`N` 为相应指标的有效样本数。

### 通用指标

| 指标 | 定义/公式 | 趋势 | 业务含义 | 代码 |
| --- | --- | --- | --- | --- |
| Accuracy | `正确样本数 / N` | 越高越好 | 用于 audit action 等离散预测 | [audit_metrics.py](src/mv_audit/evaluation/audit_metrics.py) |
| Exact Match | `1[y = ŷ]` 的平均 | 越高越好 | 字段值或 consistency 布尔值完全一致 | [field_metrics.py](src/mv_audit/evaluation/field_metrics.py)、[consistency_metrics.py](src/mv_audit/evaluation/consistency_metrics.py) |
| Macro-F1 | 各类别 F1 的算术平均 | 越高越好 | 不让数量较多的 low-risk 掩盖 high-risk | [audit_metrics.py](src/mv_audit/evaluation/audit_metrics.py) |
| IoU | `area(Bp ∩ Bg) / area(Bp ∪ Bg)` | 越高越好 | 证据框与真值框重叠程度 | [bbox_evaluator.py](src/mv_audit/evaluation/bbox_evaluator.py) |
| SFT loss | assistant target token 的交叉熵 | 越低通常越好 | 衡量 teacher-forcing 拟合，不等同业务准确率 | [train_sft.py](src/mv_audit/training/train_sft.py) |
| DPO loss | `-log sigmoid(beta * ((logπw-logπref,w) - (logπl-logπref,l)))` | 越低通常越好 | 衡量 chosen 相对 rejected 的偏好，不等同业务成功 | [train_dpo.py](src/mv_audit/training/train_dpo.py) |
| Preference accuracy | `count(margin > 0) / pair_count` | 越高越好 | pair 上 policy 是否更偏好 chosen |
| Preference margin | chosen/reference 与 rejected/reference 的 log-ratio 差 | 适度为佳 | 过大可能是 easy-pair 饱和；不同 log-prob 归一化版本不能横比 |

### 结构契约指标

| 指标 | 分母与判定 | 趋势 | 含义 |
| --- | --- | --- | --- |
| JSON Validity | 可被 JSON parser 解析的输出数 / 全部输出 | 越高越好 | 输出能否进入后续程序 |
| Schema Compliance | 通过 `configs/schema/output_schema.json` 的输出数 / 全部输出 | 越高越好 | 字段层级、类型、枚举和 required 是否符合契约 |

解析与 schema 校验见 [json_parser.py](src/mv_audit/evaluation/json_parser.py)。schema guard 是业务输出兜底；alignment reward 和原始质量审计使用 guard 前输出，避免后处理“替模型拿分”。

### 项目自定义业务指标

| 指标 | 精确定义与分母 | 趋势 | 为什么需要 |
| --- | --- | --- | --- |
| High-risk Miss, HRM | 在真值 `risk=high` 中，预测缺失、预测风险非 high 或 `audit_result=pass` 的比例 | 越低越好 | 直接衡量高风险漏放，是首要安全门禁 |
| False Manual Review | 真值 audit=pass 中预测 manual_review 的比例 | 越低越好 | 衡量正常单被无谓推给人工 |
| False Escalation | 真值 low/pass 中预测 high 或 reject 的比例 | 越低越好 | 防止靠“全部拒绝”虚假降低 HRM |
| Evidence Support | 预测 evidence 的 `(image_id, doc_type, field, value)` 四元组精确命中真值的比例 | 越高越好 | 结论是否有正确来源和正确值共同支撑 |
| Evidence Value | 预测 `(field, value)` 命中真值的比例 | 越高越好 | 单独观察值读取能力 |
| Evidence Source | 预测 `(image_id, doc_type, field)` 命中真值的比例 | 越高越好 | 单独观察证据来源定位 |
| Hallucination | 预测 evidence 中来源非法、字段在真值中不确定或四元组不被真值支持的比例 | 越低越好 | 检查看似完整但实际编造的证据 |
| Relaxed bbox | `IoU >= 0.3` 或预测框中心落入真值框 | 越高越好 | OCR 文本框边界略有偏移时仍认可定位正确 |
| Order-ID Pair Rate | order-id mismatch 真值中，预测同时给出订单截图和报销单两个不同且正确订单号的 case 比例 | 越高越好 | 防止只说“不一致”却不展示矛盾双方 |
| Error migration | baseline/DPO 错误集合的 fixed、persistent、introduced 数量 | fixed 多、introduced 少 | 判断对齐到底修了谁，又伤了谁 |

Evidence 与 hallucination 见 [evidence_metrics.py](src/mv_audit/evaluation/evidence_metrics.py)、[hallucination_metrics.py](src/mv_audit/evaluation/hallucination_metrics.py)；HRM 与汇总见 [case_scorer.py](src/mv_audit/evaluation/case_scorer.py)、[evaluate_all.py](src/mv_audit/evaluation/evaluate_all.py)；DPO v3 的 order-id 和迁移审计见 [model_mined_dpo_v3.py](tools/model_mined_dpo_v3.py)、[compare_dpo_v3_results.py](tools/compare_dpo_v3_results.py)。

### 自定义 task reward

task reward 是偏好 pair 审计和 probe 的项目评分，不等同于 Audit Accuracy：

```text
R = +0.15 field
    +0.15 consistency
    +0.20 anomaly
    +0.15 audit
    +0.15 evidence
    +0.10 JSON
    +0.10 uncertainty
    +0.25 order-id pair
    -0.20 hallucination
    -0.40 high-risk miss
    -0.50 missing order-id pair
    -0.40 false escalation
```

最终 reward 会裁剪；非法 JSON/schema 或 high-risk pass 直接强惩罚为 `-1`。实现见 [reward_function.py](src/mv_audit/training/reward_function.py) 及 model-mining 工具。reward 的作用是表达训练偏好，最终候选仍由独立业务指标决定。

## 问题、原因与解决方案

| 问题 | 根因 | 诊断证据 | 解决方案与结果 |
| --- | --- | --- | --- |
| 训练样本缺图 | 数据路径与服务器实际图片资产不一致 | converter 报缺失图片，训练无法可靠启动 | 构造 existing-images 子集并在 dry-run 统计缺失，不从 Test 补数据 |
| `field_extraction` 被扁平到根部 | 生成模型未稳定遵循嵌套 schema | 20 条输出 JSON 可解析但 schema invalid | 增加 schema guard/后处理并重跑 152 条；先消除结构污染再判断模型能力 |
| order-id 高风险漏检 | 小字号视觉读取弱，target 没显式表达矛盾双方 | 剩余 both_wrong 集中为无双侧 order-id evidence | R2/R3 显式区分订单截图与报销单 order_id，前置双侧 evidence 并加 verifier |
| DPO v1 loss 很漂亮但业务退化 | chosen 是已学答案，rejected 是容易的规则篡改 | preference accuracy 早到 1.0，margin 最终约 74.7 | 不再用训练 loss 当业务结论，改为 model-error-mined hard pairs |
| DPO margin 虚高 | 长 JSON 使用 token log-prob 累加，长度放大差值 | v1/v2 与业务 KPI 脱钩 | v3 改 assistant-token mean log-prob，并禁止跨版本直接比较 margin |
| DPO v3 局部提升、全量退化 | 训练/probe 偏 order-id，审核动作边界向 manual_review 漂移 | probe reward +0.167；full gate 新增 7 条 HRM，14 条 reject 变 manual_review | probe 与 full gate 分层；cp15 仅保留 research candidate |
| 正常样本过度升级 | 单独奖励 high/reject 会诱导全量拒绝 | false escalation/错误迁移可见 | 加 low/pass calibration 和 false escalation 惩罚 |
| 单卡 policy/reference OOM | 8B VLM 的两套模型、激活和优化状态超过 32GB | 五候选单卡并发失败 | DPO 改单进程五卡模型切分；SFT 仍用五进程 DDP |
| SCP 中断或只拉到部分文件 | 大文件和长连接不稳定 | 本地文件存在不等于归档完整 | 单 adapter 打包、重拉、archive hash + file hash 双重校验 |
| watcher 误判终态 | fallback 或脚本尾部异常写入 `FAILED` | 结果已生成但 watcher 拒绝关机 | `FAILED` 优先保护，人工核验 manifest 后再关机 |
| BOM/UTF-8 问题 | Windows/PowerShell 与 Linux shell 编码差异 | shebang 或中文报告异常 | UTF-8 显式读写、XML/Markdown 解析和 replacement character 检查 |

核心经验是把“看清字段、表达证据、遵守 schema、做对决策”分开评测，否则会把感知错误交给 RL，或把后处理修复误认为模型能力提升。

## 代码结构与输入输出

完整逐文件说明见 [docs/code_inventory.md](docs/code_inventory.md)。下面是复现主链所需的精简地图。

| 目录/文件 | 输入 | 作用 | 输出 |
| --- | --- | --- | --- |
| `src/mv_audit/data_gen/` | 字典、YAML、schema、seed | 生成业务 case、注入异常、风险打标、case split | raw case JSONL |
| `src/mv_audit/rendering/` | raw cases | 渲染四类凭证并记录字段框 | images、bbox annotations |
| `src/mv_audit/converters/` | cases、annotations、images | 转换 SFT/DPO/GRPO/repair 数据 | train/val/pair JSONL |
| [train_sft.py](src/mv_audit/training/train_sft.py) | base model、adapter、SFT JSONL、YAML | LoRA-SFT/Repair SFT | adapter、trainer logs |
| [train_dpo.py](src/mv_audit/training/train_dpo.py) | policy/reference、preference pairs | DPO/IPO/weighted DPO | checkpoints、history、probe stats |
| [reward_function.py](src/mv_audit/training/reward_function.py) | prompt、truth、completion | 字段/证据/风险组合 reward | scalar reward 与分项 |
| [batch_inference.py](src/mv_audit/inference/batch_inference.py) | model、adapter、case images | 多图生成、断点写入 | predictions JSONL |
| [schema_guard.py](src/mv_audit/inference/schema_guard.py) | raw model JSON | 恢复嵌套 schema、order-id guard | 合法业务输出 |
| [sample_model_mined.py](src/mv_audit/inference/sample_model_mined.py) | R3、Train-only pool | 多次采样 | raw completions |
| [evaluate_all.py](src/mv_audit/evaluation/evaluate_all.py) | predictions、ground truth、schema | 汇总结构、字段、审计、证据和 bbox | metrics.json、errors.jsonl |
| [repair_paired_analysis.py](tools/repair_paired_analysis.py) | baseline/new errors | paired diff 与错误归因 | attribution JSON/Markdown |
| [model_mined_dpo_v3.py](tools/model_mined_dpo_v3.py) | completions、truth、reward | 选 hard pairs、probe 审计 | train/holdout/probe pairs |
| [build_phase10_post_training_report.py](tools/build_phase10_post_training_report.py) | CSV、metrics、errors、logs | 汇总结果与案例 | Phase 10 表格/JSON/PNG |
| [audit_model_lineage_archive.py](tools/audit_model_lineage_archive.py) | lineage manifest、本地 adapter | 校验配置、权重、hash、父级 | audit JSON/Markdown |

主流水线入口：

```text
scripts/01_*    数据生成、异常注入和切分
scripts/02_*    图片渲染
scripts/03_*    训练集转换
scripts/04_*    SFT
scripts/05_*    DPO/IPO
scripts/07_*    推理
scripts/08_*    评测
scripts/12_*    Repair SFT v3
scripts/13-17_* Model-Mined DPO v3 与 full gate
```

## 服务器训练、推理、评测与归档

以下命令使用占位符，不记录密码、真实密钥或可复用凭据。

### 1. 开机后的只读检查

```bash
ssh -p <PORT> <USER>@<HOST>
cd /root/autodl-tmp/VLM-Post-Training
export PATH=/root/miniconda3/bin:/root/anaconda3/bin:$PATH
nvidia-smi
pgrep -af 'train_sft|train_dpo|batch_inference|sample_model_mined'
git status --short
```

输入是服务器现状；输出是 GPU、进程、仓库和已有 runroot。若已有训练，不重复启动。

### 2. 单元测试与 dry-run

```bash
PYTHONPATH=src pytest -q
python -m compileall src/mv_audit tools tests
DRY_RUN=1 MAX_SAMPLES=4 bash scripts/12_run_order_id_repair_sft_v3_server.sh
```

dry-run 检查模型/adapter 可加载、图片存在、label mask 正确、JSON/schema 可评测、reward 对 high-risk pass 强惩罚。dry-run 通过不等于模型有效。

### 3. 显式批准后训练

```bash
ALLOW_TRAINING=1 bash scripts/12_run_order_id_repair_sft_v3_server.sh
ALLOW_TRAINING=1 bash scripts/13_run_model_mined_dpo_v3_server.sh
```

真实流程只运行当次批准的阶段。日志写入 `outputs/runtime/`，checkpoint 写入 `outputs/checkpoints/`；状态文件供 watcher 判断。

### 4. 五卡推理、合并与评测

```text
train_decode_dev.jsonl
  -> shard-0..4
  -> GPU0..4 独立 predictions
  -> deterministic merge
  -> evaluate_all
  -> metrics.json + errors.jsonl
```

每个 shard 按 case id 独立，可断点续跑；merge 后必须验证总数为 152、无重复、无缺失。SFT gate 看 JSON/schema、Audit、HRM、Evidence；DPO 还必须检查 false escalation、error migration 和 alignment probe。

### 5. 归档和关机

远端只打包可加载 adapter 与 processor/tokenizer metadata；历史 optimizer/checkpoint 子目录不进入 minimal archive。下载后比较 archive SHA256，再解包校验 `adapter_config.json`、权重大小和 safetensors header。

```powershell
python tools/audit_model_lineage_archive.py --strict --output_json docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive_audit.json --output_md docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive_audit.md
```

只有 manifest 完整且无 `FAILED` 才请求 `shutdown -h now`，随后确认 SSH 关闭。归档失败时保留日志和 manifest，不删除远端文件。

## 模型谱系与本地归档

| 模型 | 父级 | 角色 | 本地状态 | adapter 权重 |
| --- | --- | --- | --- | ---: |
| M2 SFT | base Qwen3-VL | historical sample500 baseline | `VERIFIED` | 174,663,096 B |
| Repair R1 | M2 | high-risk repair stage 1 | `VERIFIED` | 174,663,096 B |
| Repair R2 | R1 | order-id repair stage 2 | `VERIFIED` | 174,663,096 B |
| Repair R3 | R2 | production candidate, not deployed | `VERIFIED` | 174,663,096 B |
| DPO weak checkpoint-40 | R3 | strong continuation initialization | `VERIFIED` | 174,663,096 B |
| DPO strong checkpoint-15 | weak-40 | alignment research candidate | `VERIFIED` | 174,663,096 B |

机器可读清单：[model_lineage_archive.json](docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive.json)；审计结果：[JSON](docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive_audit.json) / [Markdown](docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive_audit.md)。

M2/R1/R2/weak-40 的 minimal tar 在 `outputs/model_candidates/model_lineage/archives/`；R3 与 strong-15 已有独立归档。minimal 表示保留推理复现所需 adapter，不声称包含 optimizer state 或每个历史 checkpoint。

## AI 辅助开发方法

项目采用 AI-assisted development，而不是把 AI 生成的代码未经核验直接投入训练：

1. **人定义问题和边界**：业务 schema、风险规则、Train/Test 隔离、预算和停止条件。
2. **AI 辅助实现**：草拟 converter、训练脚本、分析工具、测试和文档，辅助搜索错误路径。
3. **静态审计**：逐文件 inventory，核对输入输出、接口、配置路径和数据来源。
4. **小规模验证**：compile、pytest、generation smoke、reward smoke、label-mask test 和 dry-run。
5. **实验门禁**：先 train_decode_dev，再决定是否扩大；loss、pair accuracy 不替代业务指标。
6. **人工决策**：根据 errors.jsonl、paired diff 和案例决定做 SFT repair、DPO 或停止训练。
7. **证据归档**：原始日志、metrics、predictions、错误案例、图表和 SHA256 可回查。

这种表述既诚实说明 AI 的参与，也体现需求拆解、代码审计、实验设计、成本控制和结果验证能力。

## 最终模型选择

![模型选择门禁](docs/experiments/phase10_model_error_mined_dpo_v3/figures/model_selection_gate.png)

- **Structured Repair SFT v3**：`FINAL_HOLDOUT_FAILED / NOT_DEPLOYED`。曾通过 152 条 Train-only 开发门禁，但 final_holdout_v1 与 sample500 诊断均退化。
- **DPO v3 strong checkpoint-15**：`ALIGNMENT_RESEARCH_CANDIDATE / ALIGNMENT_GATE_NOT_MET`。probe 有局部可测收益，full gate 不允许部署。
- **M2**：`HISTORICAL_SAMPLE500_BASELINE`。用于保留历史 benchmark，不与 train_decode_dev 横向伪比较。

最终正向结论是：完成可复现的 SFT + preference alignment 全链路，证明 model-mined DPO 能改变目标偏好，同时用独立业务门禁识别其泛化边界，并把最稳健的 SFT 与研究型 DPO 分层治理。

简历与面试口径见 [docs/resume_vlm_post_training.md](docs/resume_vlm_post_training.md)。

## 技术附录：历史实验与工程实录

以下章节保留早期数据、损失函数、M0-M3v2 结果、服务器故障和恢复过程。部分章节中的“当前”是相应历史阶段的状态；最终状态以本文前半部分和 [model_selection.json](docs/experiments/phase10_model_error_mined_dpo_v3/model_selection.json) 为准。

## 摘要与核心结论

本项目研究的问题不是普通 OCR，也不是单图 VQA，而是多张报销凭证之间的结构化一致性审计。模型输入是一个 case 下的 2 到 4 张凭证图片，输出必须是固定 schema 的 JSON，包含字段抽取、跨图一致性判断、风险等级、审核建议、原因、证据、bbox 和不确定性。

当前最重要结论：

| 结论 | 证据 | 判断 |
| --- | --- | --- |
| LoRA-SFT 有效 | M2 在 sample500 平均 `Audit Accuracy=0.7735`、`Evidence Support Rate=0.8035`、`JSON Validity=1.000` | SFT 成功建立结构化输出、字段抽取、证据引用和基础审计能力 |
| 原生模型不足 | M0 zero-shot 平均 `Audit Accuracy=0.000`，M1 few-shot 平均 `0.0785` | 仅靠 prompt/few-shot 不能完成该任务 |
| DPO v1 是负结果 | M3 的 DPO loss 降到 `0.000568`、preference margin 到 `74.731`，但 `Audit Accuracy` 从 M2 `0.7735` 降到 `0.6685` | pair 训练成功不等于业务成功 |
| DPO v2 部分修复 | M3v2 `Audit Accuracy=0.7645`，接近 M2；但 `High-risk Miss Rate=0.2546`，差于 M2 `0.2427` | 修复了 accuracy 崩塌，但没有解决高风险漏检 |
| two-candidate ablation 不值得扩大 | `dpo_v2_baseline` 与 `auxdpo_v2_strong` 在 Train decode dev 核心指标相同，High-risk Miss 都是 `0.2299` | 继续烧完整 DPO/IPO ablation 不划算 |
| Structured Repair SFT v3 未泛化 | 在 152 条 `train_decode_dev` 上 `Audit Accuracy=0.9671`、`High-risk Miss=0.0575`，但 final_holdout_v1 只有 `0.7160/0.3152`，sample500 诊断只有 `0.6075/0.4217` | 标记为 `FINAL_HOLDOUT_FAILED / NOT_DEPLOYED` |
| DPO v3 有局部对齐信号 | 24 条 probe reward 提升 `0.1667`、order-id 双侧证据命中率提升 `11.11pp` | 说明偏好信号可改变目标行为 |
| DPO v3 未通过全量门禁 | 152 条上 `Audit Accuracy=0.8684`、`High-risk Miss=0.1379` | 仅标记为 `ALIGNMENT_RESEARCH_CANDIDATE`，禁止替代 SFT v3 |

一句话总结：M2 仍是冻结的 sample500 历史基线；Structured Repair SFT v3 是当前 Train-only 开发门禁上的 production candidate，DPO v3 完成了可测量的局部偏好对齐，但因全量门禁未通过而保留为 research candidate。

| 阶段 | 动机 | 做法 | 效果 |
|---|---|---|---|
| DPO v1 | 先验证 DPO 能不能让模型偏向正确审计答案 | 用人工规则构造 rejected，比如把风险等级、审核结论改错 | 训练 loss 很好看，但业务效果变差，说明 rejected 太简单，模型学到了捷径 |
| DPO v2 | 防止 v1 那种“训练上赢、业务上输” | 加 hard rejected、高风险漏报 pair、protective pair、normal calibration，并加 SFT auxiliary loss 保护原能力 | 比 v1 稳一点，但仍没有超过 SFT，高风险漏报没有改善 |
| DPO v3 | 让偏好数据更贴近模型真实错误 | 让 SFT 模型自己生成多个答案，从真实错误输出里挖 hard pairs | 局部 probe 有提升，但完整 152 条门禁下降，所以不能替代 SFT |

DPO v1 证明“简单错答案不行”；DPO v2 尝试加保护和校准；DPO v3 尝试用模型真实错误做偏好对，但最后发现它只修了一部分问题，同时伤到了整体审计决策边界。


## 1. 任务定义与 Benchmark

### 1.1 业务任务

企业费用报销审核通常需要同时检查多份材料：

- 发票：金额、税额、销售方、发票号、日期。
- 支付截图：支付金额、收款方、支付人、支付流水号、支付日期。
- 报销申请单：申请人、报销金额、申请日期、费用类型。
- 订单截图：订单金额、商户、订单用户、订单号、订单日期。

模型需要判断：

- 金额是否一致，例如发票金额、支付金额、订单金额和报销金额是否矛盾。
- 商户是否一致，例如发票销售方和支付收款方是否明显不同。
- 人员是否一致，例如申请人、付款人和订单用户是否一致。
- 日期是否合理，例如支付日期、订单日期、发票日期和申请日期顺序是否异常。
- 订单号和支付流水是否存在或一致。
- 材料是否缺失、图片是否不可读、是否存在重复凭证。

### 1.2 输入与输出

输入是一个 case 的多图集合：

```text
case_id
+ invoice image
+ payment image
+ reimbursement_form image
+ optional order image
```

输出必须是 Evidence-Grounded JSON，核心字段包括：

| 字段 | 含义 |
| --- | --- |
| `case_id` | 当前报销 case 的唯一 ID |
| `field_extraction` | 抽取金额、商户、人员、日期、订单号等字段 |
| `consistency_check` | 跨图一致性布尔判断 |
| `anomaly_types` | 异常类型列表 |
| `risk_level` | `low`、`medium`、`high` |
| `audit_result` | `pass`、`manual_review`、`reject_recommendation`、`missing_info` |
| `reason` | 审核结论的自然语言理由 |
| `evidence` | 支持结论的来源图片、字段、值、bbox、证据文本 |
| `uncertainty` | 不确定字段和是否需要人工复核 |

### 1.3 Benchmark 与模型编号

项目内部 benchmark 不是通用 VLM 榜单，而是 `sample500` 业务评测：四个 split，每个 split 500 条，共 2000 条测试样本。评测产物归档在 `docs/experiments/`。

| 编号 | 模型形态 | 评测状态 |
| --- | --- | --- |
| M0 | `Qwen3-VL-8B-Instruct` zero-shot | 已完成 Phase07 sample500 |
| M1 | `Qwen3-VL-8B-Instruct` few-shot | 已完成 Phase07 sample500 |
| M2 | `Qwen3-VL-8B-Instruct + LoRA-SFT` | 已完成训练、val loss、sample500，当前业务基线 |
| M3 | `M2 + DPO v1 adapter` | 已完成 DPO sample1000 和 sample500，业务失败 |
| M3v2 | `M2 + conservative DPO v2 adapter` | 已完成 DPO v2 和 sample500，部分修复但未达目标 |
| M4 | `M2/M3 + GRPO` | 未正式完成，仅有 smoke 级别代码与验证 |
| repair_sft_r1/r2 | `M2 + High-risk Repair SFT` | 历史修复分支，不是当前候选 |
| repair_sft_r3 | `R2 + Order-ID Structured Repair SFT` | `FINAL_HOLDOUT_FAILED / NOT_DEPLOYED`；152 条开发门禁高分未泛化到 final_holdout_v1 和 sample500 诊断 |
| DPO v3 checkpoint-15 | `repair_sft_r3 + Model-Error-Mined DPO` | `ALIGNMENT_RESEARCH_CANDIDATE`；probe 有提升，全量 gate 未通过 |

## 2. 合成数据集 MultiVoucher-Audit

### 2.1 为什么使用合成数据

真实企业报销数据包含个人姓名、商户、订单、支付流水和财务信息，直接用于实验存在隐私和合规问题。本项目采用可控合成数据，以便同时获得：

- 字段真值：每个金额、商户、人员、日期都有标准答案。
- 异常真值：每个 case 的异常类型、风险等级、审核建议可程序生成。
- 证据真值：每个字段在图片上的 bbox 和证据文本可追踪。
- 切分控制：Train/Val/Test 按 case 级别切分，避免同一 case 泄漏。

### 2.2 数据生成链路

```text
词典与 schema
-> 生成正常报销 case
-> 注入业务异常
-> 风险规则打标
-> case-level split
-> 渲染四类凭证图片
-> 记录字段 bbox
-> 构造 SFT / DPO / GRPO / eval sets
```

核心代码入口：

| 阶段 | 代码/脚本 |
| --- | --- |
| case 生成 | `src/mv_audit/data_gen/generate_base_cases.py` |
| 异常注入 | `src/mv_audit/data_gen/anomaly_injector.py` |
| 风险规则 | `src/mv_audit/data_gen/risk_rule_engine.py` |
| case 切分 | `src/mv_audit/data_gen/split_builder.py` |
| 图片渲染 | `src/mv_audit/rendering/render_all.py` |
| bbox 记录 | `src/mv_audit/rendering/bbox_recorder.py` |
| SFT/DPO/GRPO 构造 | `src/mv_audit/converters/` |

### 2.3 数据规模

主实验数据规划为 41,000 个 case：

| split | cases | 用途 |
| --- | ---: | --- |
| train | 30,000 | SFT、DPO、GRPO、High-risk Repair 候选来源 |
| val_in_template | 2,000 | SFT validation 和模板内验证 |
| val_unseen_template | 1,000 | 未见模板验证 |
| test_clean | 2,000 | 标准测试 |
| test_robust | 2,000 | 图像扰动测试 |
| test_unseen_template | 2,000 | 未见模板测试 |
| test_hard_negative | 2,000 | 高风险/困难负例测试 |

一个 case 最多包含 4 张图片，因此完整 main 数据约为 16 万级图片。历史核查记录显示 `images_main` 约 162,770 张图片。

本次真实 SFT 训练使用的是服务器 existing-images 子集：

| 数据 | 路径 | 样本数 | 用途 |
| --- | --- | ---: | --- |
| SFT train | `data/mv_audit/sft_main/train_existing_images.jsonl` | 21,682 | LoRA-SFT |
| SFT val | `data/mv_audit/sft_main/val_existing_images.jsonl` | 1,138 | validation loss |
| DPO v1 pairs | `data/mv_audit/dpo_main/pairs_train.jsonl` | sample1000 使用 1000 pairs | M3 DPO |
| DPO v2 train pairs | `data/mv_audit/dpo_v2/pairs_train.jsonl` | 3,000 | M3v2 DPO |
| DPO v2 holdout pairs | `data/mv_audit/dpo_v2/pairs_holdout.jsonl` | 300 | pair holdout 监控 |
| Train decode dev | `data/mv_audit/dpo_v2/train_decode_dev.jsonl` | 152 实际可解码行 | 训练域小评测 |
| High-risk Repair Pack | `docs/experiments/phase08_high_risk_repair_pack_20260813/repair_pack_sft.jsonl` | 120 | repair_sft_r1 修复样本 |
| repair SFT mix | `docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix.jsonl` | 240 | 120 repair + 120 calibration |

### 2.4 异常类型

主要异常族：

| 异常 | 含义 | 常见风险 |
| --- | --- | --- |
| `amount_mismatch` | 发票、支付、订单、报销金额不一致 | medium/high |
| `over_reimbursement` | 报销金额高于真实支付或发票金额 | high |
| `merchant_mismatch` | 商户或收款方不一致 | high |
| `applicant_mismatch` | 申请人、付款人、订单用户不一致 | high |
| `date_mismatch` | 订单、支付、发票、申请日期顺序异常 | medium/high |
| `order_id_mismatch` | 订单号不一致或无法对应 | high |
| `missing_document` | 缺少关键凭证 | high，但常输出 `missing_info` |
| `unreadable_image` | 关键字段不可读 | high/manual review |
| `duplicate_in_batch` | 批内重复凭证 | high |
| `none` | 无异常 | low/pass |

### 2.5 数据边界

本项目特别强调防止评测泄漏：

- DPO/GRPO/High-risk Repair 候选只来自 MV-Train。
- sample500/test 只用于冻结后的业务评测和报告，不进入训练、调参或候选选择。
- DPO v2 使用 case-level train/holdout/decode-dev 切分，overlap 均为 0。
- High-risk Repair Pack 与 DPO holdout、train_decode_dev、sample500 overlap 均为 0。

## 3. 模型架构与工程链路

### 3.1 基座模型

基座模型为 `Qwen3-VL-8B-Instruct`。本地模型资产位于：

```text
models/Qwen3-VL-8B-Instruct/
```

重要模型元数据：

| 文件 | 作用 |
| --- | --- |
| `config.json` | 模型结构配置 |
| `tokenizer.json`、`vocab.json`、`merges.txt` | 文本分词 |
| `chat_template.json` | 多轮对话模板 |
| `preprocessor_config.json` | 图片预处理 |
| `generation_config.json` | 默认生成参数 |
| `model.safetensors.index.json` | 权重 shard 索引 |
| `*.safetensors` | 大模型权重，不进 Git |

### 3.2 Adapter 结构

本项目不全量微调 8B 基座模型，而是训练 adapter：

```text
Qwen3-VL-8B-Instruct
    |
    +-- LoRA-SFT adapter -> M2
             |
             +-- DPO adapter -> M3 / M3v2 / ablation variants
             |
             +-- High-risk Repair SFT adapter -> repair_sft_r1
```

LoRA 目标模块：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### 3.3 端到端工程链路

```text
case schema + dictionaries
-> raw cases
-> anomaly injection + risk rules
-> voucher rendering + bbox recording
-> SFT/DPO/GRPO data
-> LoRA-SFT / DPO / GRPO training
-> batch inference
-> evaluate_all metrics
-> error migration / archive / repair pack
```

推理统一入口是 `src/mv_audit/inference/batch_inference.py`，评测统一入口是 `src/mv_audit/evaluation/evaluate_all.py`。

## 4. 训练目标、损失函数与优化器

### 4.1 SFT 目标

SFT 训练目标是让模型在给定多图输入 `x` 后生成标准答案 `Y = {y_1, ..., y_T}`。训练时只对 assistant answer token 计算 loss，用户 prompt 和图片占位不计入监督损失。

公式：

```text
L_SFT = - (1 / |Y|) * sum_t log p_theta(y_t | x, y_<t)
```

实现位置：

- `src/mv_audit/training/train_sft.py`
- `DataCollatorForQwenVLSFT`
- Hugging Face `Trainer`

SFT 优化器由 `Trainer` 根据 `TrainingArguments` 使用 AdamW 类优化器；训练配置里显式设置 learning rate、batch、gradient accumulation、bf16、eval/save steps。

### 4.2 DPO 目标

DPO 使用同一个 prompt 下的 chosen answer 和 rejected answer。令：

```text
logit = [log pi_theta(y_c|x) - log pi_theta(y_r|x)]
        - [log pi_ref(y_c|x) - log pi_ref(y_r|x)]
```

DPO loss：

```text
L_DPO = - log sigmoid(beta * logit)
```

其中：

- `pi_theta` 是当前训练 policy。
- `pi_ref` 是冻结 reference policy。
- `y_c` 是 chosen。
- `y_r` 是 rejected。
- `beta` 控制偏好间隔强度。

实现位置：`src/mv_audit/training/train_dpo.py::_preference_loss_values`。

### 4.3 Weighted DPO

DPO v2 给不同 pair 加权：

```text
L_weighted = sum_i w_i * L_i / sum_i w_i
```

权重来自：

```text
w = severity_weight * hardness_weight * reliability_weight
```

并用 `max_weight=3.0` 限制上限。DPO v2 pair report 中权重分布为：

| min | max | mean |
| ---: | ---: | ---: |
| 0.75 | 3.0 | 2.506 |

### 4.4 IPO 与 AuxDPO

IPO loss 与实现一致：

```text
L_IPO = (logit - 1 / (2 * beta))^2
```

AuxDPO 加入 chosen answer 的 SFT/NLL 辅助项：

```text
L = L_preference + lambda_sft * L_NLL(chosen)
```

本项目用它尝试防止 DPO 破坏 SFT 已学到的结构化输出和审计边界。two-candidate 中实际完成了 `dpo_v2_baseline` 和 `auxdpo_v2_strong`，后续 IPO 候选因为省钱策略暂停。

### 4.5 DPO 优化器

DPO 训练显式使用：

```text
torch.optim.AdamW(trainable_parameters, lr=learning_rate)
```

实现位置：`src/mv_audit/training/train_dpo.py`。

## 5. 评测指标

评测器不是只看文本相似度，而是按业务能力拆分：

| 指标 | 含义 | 当前用途 |
| --- | --- | --- |
| JSON Validity | 输出是否能解析成合法 JSON | 结构化输出底线 |
| Schema Compliance | JSON 是否符合 `output_schema.json` | 防止字段缺失或输出契约错误 |
| Field EM | 字段抽取 exact match | 衡量金额、商户、人员、日期等抽取能力 |
| Risk Type Macro-F1 | 异常类型宏平均 F1 | 衡量异常识别 |
| Audit Accuracy | 审核结论是否正确 | 核心业务指标 |
| High-risk Miss Rate | 高风险样本是否被错误放行或降级 | 当前最关键问题 |
| False Manual Review Rate | 正常样本是否被过度转人工 | 控制误拦截 |
| Evidence Support Rate | 输出证据是否支持结论 | 防止无证据审计 |
| Hallucination Rate | 是否编造字段或证据 | 控制幻觉 |
| Evidence BBox Accuracy Relaxed | bbox 是否大致定位正确 | 衡量可追溯性 |
| Error Cases | 至少一个关键指标失败的 case 数 | 人工诊断入口 |

核心实现：

- `src/mv_audit/evaluation/json_parser.py`
- `src/mv_audit/evaluation/audit_metrics.py`
- `src/mv_audit/evaluation/evidence_metrics.py`
- `src/mv_audit/evaluation/bbox_evaluator.py`
- `src/mv_audit/evaluation/evaluate_all.py`

## 6. SFT 数据集与样例

### 6.1 SFT 数据格式

SFT 每一行对应一个报销 case，而不是单张图片。每行包含：

| 字段 | 含义 |
| --- | --- |
| `id` | 样本 ID |
| `case_id` | 报销 case ID |
| `task_type` | 任务类型，例如 `full_audit` 或 `consistency_check` |
| `images` | 多张凭证图片路径和 doc type |
| `messages` | user prompt 与 assistant 标准答案 |
| `answer` | 结构化标准 JSON |

### 6.2 SFT 数据集示例

以下是精简后的格式示例，来自项目已有 SFT/repair SFT 样本结构，省略了部分字段值：

```json
{
  "id": "MV_MAIN_000109_consistency_check",
  "case_id": "MV_MAIN_000109",
  "task_type": "consistency_check",
  "images": [
    {
      "image_id": "MV_MAIN_000109_reimbursement_form",
      "doc_type": "reimbursement_form",
      "image_path": "data/mv_audit/images_main/train/MV_MAIN_000109_reimbursement_form.png"
    },
    {
      "image_id": "MV_MAIN_000109_payment",
      "doc_type": "payment",
      "image_path": "data/mv_audit/images_main/train/MV_MAIN_000109_payment.png"
    },
    {
      "image_id": "MV_MAIN_000109_order",
      "doc_type": "order",
      "image_path": "data/mv_audit/images_main/train/MV_MAIN_000109_order.png"
    },
    {
      "image_id": "MV_MAIN_000109_invoice",
      "doc_type": "invoice",
      "image_path": "data/mv_audit/images_main/train/MV_MAIN_000109_invoice.png"
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "你是企业费用报销一致性审计助手。请只输出一个合法 JSON..."
    },
    {
      "role": "assistant",
      "content": "{\"case_id\":\"MV_MAIN_000109\", ... }"
    }
  ],
  "answer": {
    "case_id": "MV_MAIN_000109",
    "field_extraction": {
      "invoice_amount": "2295.26",
      "payment_amount": "2295.26",
      "reimbursement_amount": "2295.26",
      "applicant": "许磊",
      "payer": "赵明远"
    },
    "consistency_check": {
      "amount_consistent": true,
      "person_consistent": false,
      "document_complete": true
    },
    "anomaly_types": ["applicant_mismatch"],
    "risk_level": "high",
    "audit_result": "reject_recommendation",
    "evidence": [
      {
        "source_image_id": "MV_MAIN_000109_payment",
        "source_doc_type": "payment",
        "field": "payer",
        "value": "赵明远",
        "bbox": [240, 454, 306, 484],
        "evidence_text": "付款人：赵明远"
      }
    ],
    "uncertainty": {
      "has_uncertain_fields": false,
      "requires_manual_review": false
    }
  }
}
```

读这个例子时要注意两点：

- `images` 告诉模型看哪几张图片。
- `answer.evidence` 告诉模型结论必须有图像来源和 bbox 证据，不能只输出自由文本判断。

## 7. SFT 训练过程、超参数与结果

### 7.1 SFT 训练配置

SFT 配置来自 `configs/train/sft_lora_qwen3vl_8b_phase07_server.yaml`。

| 项目 | 值 |
| --- | --- |
| 基座模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| 本地模型目录 | `models/Qwen3-VL-8B-Instruct` |
| train file | `data/mv_audit/sft_main/train_existing_images.jsonl` |
| val file | `data/mv_audit/sft_main/val_existing_images.jsonl` |
| output dir | `outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1` |
| learning rate | `1.0e-4` |
| epochs | `1` |
| per-device train batch | `1` |
| gradient accumulation | `16` |
| bf16 | `true` |
| gradient checkpointing | `true` |
| eval steps | `500` |
| save steps | `500` |
| seed | `42` |

LoRA 配置：

| 参数 | 值 |
| --- | --- |
| r | `16` |
| alpha | `32` |
| dropout | `0.05` |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |

### 7.2 SFT 训练过程

SFT 使用 `Trainer` 执行 teacher-forcing 训练：

```text
读取 SFT JSONL
-> 检查图片和 answer JSON
-> 构造 Qwen-VL 多图 chat message
-> processor 编码文本和图片
-> mask prompt tokens
-> 只对 assistant answer 计算 NLL
-> 保存 LoRA adapter
```

关键代码：

- `src/mv_audit/training/train_sft.py`
- `DataCollatorForQwenVLSFT`
- `SFTDataset`
- `scripts/04_train_sft.sh`

### 7.3 SFT validation loss

| 记录 | epoch | eval loss |
| --- | ---: | ---: |
| eval 1 | 0.37 | 0.0002858767 |
| eval 2 | 0.74 | 0.0001022846 |

最终训练摘要：

| 项目 | 值 |
| --- | ---: |
| final epoch | 1.0 |
| final train loss | 0.0049826302 |

解释：validation loss 很低说明模型在 teacher-forcing 条件下已拟合答案格式，但业务能力必须通过 sample500 生成式评测确认，不能只看 loss。

## 8. 与原生模型对比：M0/M1/M2

### 8.1 Phase07 Benchmark 设置

Phase07 只比较：

- M0：原生 Qwen3-VL zero-shot。
- M1：原生 Qwen3-VL few-shot。
- M2：Qwen3-VL + LoRA-SFT。

评测规模：

```text
3 models * 4 splits * 500 cases = 6000 predictions
```

归档路径：`docs/experiments/phase07_sample500/`。

### 8.2 平均指标表

| Model | JSON | Schema | Field EM | Risk F1 | Audit Acc | High-risk Miss | Evidence | Hallucination | BBox Relaxed | Avg Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 zero-shot | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 500.0 |
| M1 few-shot | 1.000 | 0.116 | 0.114 | 0.022 | 0.079 | 0.999 | 0.025 | 0.575 | 0.010 | 500.0 |
| M2 LoRA-SFT | 1.000 | 0.877 | 0.876 | 0.743 | 0.774 | 0.243 | 0.804 | 0.001 | 0.795 | 164.5 |

### 8.3 M2 分 split 指标

| Split | JSON | Schema | Field EM | Risk F1 | Audit Acc | High-risk Miss | Evidence | BBox Relaxed | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.832 | 0.831 | 0.780 | 0.744 | 0.270 | 0.747 | 0.737 | 173 |
| test_robust | 1.000 | 0.810 | 0.809 | 0.798 | 0.742 | 0.276 | 0.727 | 0.723 | 176 |
| test_unseen_template | 1.000 | 0.864 | 0.863 | 0.754 | 0.742 | 0.274 | 0.775 | 0.764 | 176 |
| test_hard_negative | 1.000 | 1.000 | 1.000 | 0.639 | 0.866 | 0.152 | 0.965 | 0.954 | 133 |

### 8.4 SFT 实验结果图

![M0/M1/M2 平均指标](docs/experiments/phase07_sample500/figures/model_average_metrics.png)

![M2 分 split 指标](docs/experiments/phase07_sample500/figures/m2_split_metrics.png)

![Phase07 错误样本数量](docs/experiments/phase07_sample500/figures/error_cases_by_split.png)

### 8.5 SFT 实验结果分析

SFT 的主要价值很明确：

- 原生模型 M0 虽然能输出可解析 JSON，但完全不符合任务 schema，业务指标为 0。
- few-shot M1 有轻微改善，但 `Audit Accuracy=0.0785`，高风险漏检仍接近 1。
- M2 的 schema、字段、审计和证据指标同时跃升，说明 LoRA-SFT 学到了该任务的输出契约和多图审计模式。
- M2 的主要残留问题是 High-risk Miss Rate 仍为 `0.2427`，尤其在 clean、robust、unseen template 上约 `0.27`，这正是 Phase08 想用偏好优化继续修的点。

结论：M2 是当前可靠业务基线，后续所有方法都必须与 M2 对比，不能只与原生模型对比。

## 9. DPO 数据集与样例

### 9.1 DPO v1 数据

DPO v1 使用 `data/mv_audit/dpo_main/pairs_train.jsonl`，从 MV-Train 的 SFT 标准答案构造 chosen/rejected pair。

v1 的 rejected answer 包括：

- invalid JSON。
- 风险等级错误。
- 审核结论错误。
- 证据错误或不足。
- 高风险放行。
- 幻觉字段。

DPO v1 sample1000 训练时：

| 项目 | 值 |
| --- | ---: |
| requested examples | 1000 |
| skipped missing images | 324 |
| global step | 63 |
| chosen mean reward | 1.000000 |
| rejected mean reward | 0.460112 |
| mean reward gap | 0.539888 |
| positive reward gap rate | 0.844 |
| rejected JSON valid rate | 0.896 |
| rejected high-risk miss rate | 0.155 |

### 9.2 DPO v2 数据

DPO v2 针对 v1 失败做了更保守设计：

| 约束 | 设计 |
| --- | --- |
| 数据来源 | Train-only，只读 `data/mv_audit/raw_cases/main/train_cases.jsonl` |
| case split | train/holdout/decode-dev overlap 全为 0 |
| 样本质量 | chosen/rejected 都要求 JSON/schema/evidence/bbox 合法 |
| pair 类型 | hard rejected、high-risk miss、protective、normal calibration |
| 权重 | severity、hardness、reliability 加权 |
| 训练强度 | lr 降到 `1e-6`，`max_train_steps=80`，加入 holdout |
| 辅助项 | 可选 `lambda_sft * chosen NLL` |

DPO v2 pair report：

| 项目 | 值 |
| --- | ---: |
| input cases | 30,000 |
| train pairs | 3,000 |
| holdout pairs | 300 |
| decode dev rows | 152 |
| skipped missing images | 18,021 |
| train/holdout overlap | 0 |
| train/decode-dev overlap | 0 |
| holdout/decode-dev overlap | 0 |

Pair 类型分布：

| pair type | count |
| --- | ---: |
| hard rejected | 1,170 |
| high-risk miss | 959 |
| protective | 647 |
| normal calibration | 524 |

Rejected error 类型：

| rejected error type | count |
| --- | ---: |
| high_risk_downgrade | 739 |
| risk_level_wrong | 744 |
| audit_result_wrong | 334 |
| high_risk_pass | 959 |
| normal_over_reject | 524 |

### 9.3 DPO 数据集示例

以下是简化后的 pair 结构示例：

```json
{
  "case_id": "MV_MAIN_009815",
  "images": [
    {"image_id": "MV_MAIN_009815_invoice", "doc_type": "invoice"},
    {"image_id": "MV_MAIN_009815_payment", "doc_type": "payment"},
    {"image_id": "MV_MAIN_009815_order", "doc_type": "order"}
  ],
  "prompt": "请完成多凭证一致性审核，输出 Evidence-Grounded JSON。",
  "chosen": {
    "risk_level": "high",
    "audit_result": "reject_recommendation",
    "anomaly_types": ["merchant_mismatch"],
    "evidence": [
      {
        "source_doc_type": "invoice",
        "field": "invoice_merchant",
        "value": "深圳天成办公用品有限公司"
      },
      {
        "source_doc_type": "payment",
        "field": "payment_merchant",
        "value": "外部供应商8919有限公司"
      }
    ]
  },
  "rejected": {
    "risk_level": "low",
    "audit_result": "pass",
    "anomaly_types": [],
    "evidence": []
  },
  "pair_type": "high_risk_miss",
  "weight": 3.0
}
```

读者提示：DPO 的关键不是“再给模型一个正确答案”，而是告诉模型同一输入下哪个回答比另一个更好。但实验说明，如果 rejected 不够贴近真实模型错误，或者偏好目标和业务指标不一致，就可能出现 loss 成功、业务失败。

## 10. DPO 训练过程、超参数与结果

### 10.1 DPO v1 训练配置

配置：`configs/train/dpo_qwen3vl_8b.yaml`。

| 项目 | 值 |
| --- | --- |
| 初始 adapter | `outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1` |
| train file | `data/mv_audit/dpo_main/pairs_train.jsonl` |
| output dir | `outputs/checkpoints/dpo/qwen3vl_8b_dpo_from_existing_epoch1` |
| learning rate | `5.0e-6` |
| epochs | `1` |
| per-device batch | `1` |
| gradient accumulation | `16` |
| beta | `0.1` |
| bf16 | `true` |
| optimizer | `torch.optim.AdamW` |

DPO v1 训练动态：

| 指标 | first step | last step |
| --- | ---: | ---: |
| global step | 1 | 63 |
| loss | 0.687787 | 0.000568 |
| chosen logp | -0.000874 | -0.007151 |
| rejected logp | -50.337337 | -128.990936 |
| preference margin | 0.107502 | 74.731003 |

图表：

![DPO v1 loss 曲线](docs/experiments/phase08_dpo_sample1000/figures/dpo_loss_curve.png)

![DPO v1 preference margin](docs/experiments/phase08_dpo_sample1000/figures/dpo_preference_margin.png)

![DPO v1 chosen/rejected logp](docs/experiments/phase08_dpo_sample1000/figures/dpo_logp_comparison.png)

### 10.2 M3 Sample500 业务结果

M3 是 `M2 + DPO v1 adapter`。

| Split | JSON | Schema | Field EM | Risk F1 | Audit Acc | High-risk Miss | Evidence | BBox Relaxed | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.808 | 0.807 | 0.806 | 0.650 | 0.264 | 0.727 | 0.716 | 227 |
| test_robust | 1.000 | 0.804 | 0.803 | 0.815 | 0.652 | 0.264 | 0.724 | 0.722 | 203 |
| test_unseen_template | 1.000 | 0.868 | 0.867 | 0.750 | 0.634 | 0.276 | 0.778 | 0.766 | 231 |
| test_hard_negative | 1.000 | 1.000 | 1.000 | 0.641 | 0.738 | 0.145 | 0.966 | 0.957 | 183 |

M2 vs M3 平均对比：

| Model | Audit Acc | High-risk Miss | Evidence | Error Cases |
| --- | ---: | ---: | ---: | ---: |
| M2 SFT | 0.7735 | 0.2427 | 0.8035 | 164.5 |
| M3 DPO v1 | 0.6685 | 0.2373 | 0.7987 | 211.0 |
| Delta | -0.1050 | -0.0054 | -0.0048 | +46.5 |

图表：

![M2 vs M3 平均指标](docs/experiments/phase08_m3_sample500/figures/m2_vs_m3_average_metrics.png)

![M3 分 split 指标](docs/experiments/phase08_m3_sample500/figures/m3_split_metrics.png)

![M3 错误样本数量](docs/experiments/phase08_m3_sample500/figures/m3_error_cases_by_split.png)

### 10.3 DPO v2 训练配置

配置：`configs/train/dpo_v2_qwen3vl_8b.yaml`。

| 项目 | 值 |
| --- | --- |
| train file | `data/mv_audit/dpo_v2/pairs_train.jsonl` |
| holdout file | `data/mv_audit/dpo_v2/pairs_holdout.jsonl` |
| decode dev file | `data/mv_audit/dpo_v2/train_decode_dev.jsonl` |
| output dir | `outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_conservative` |
| learning rate | `1.0e-6` |
| max train steps | `80` |
| loss type | `dpo` |
| beta | `0.1` |
| lambda_sft | `0.1` |
| max weight | `3.0` |
| eval steps | `10` |
| save steps | `20` |
| max holdout examples | `128` |

DPO v2 训练摘要：

| 指标 | 值 |
| --- | ---: |
| final global step | 80 |
| final train loss | 0.003978 |
| final preference margin | 55.5998 |
| holdout pair accuracy | 1.000 |
| chosen mean reward | 1.000 |
| rejected mean reward | 0.0569 |
| positive reward gap rate | 1.000 |

### 10.4 M3v2 Sample500 业务结果

M2/M3/M3v2 平均对比：

| Model | Schema | Field EM | Risk F1 | Audit Acc | High-risk Miss | Evidence | BBox Relaxed | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 SFT | 0.877 | 0.876 | 0.743 | 0.773 | 0.243 | 0.804 | 0.795 | 164.5 |
| M3 DPO v1 | 0.870 | 0.869 | 0.753 | 0.668 | 0.237 | 0.799 | 0.790 | 211.0 |
| M3v2 DPO | 0.870 | 0.869 | 0.742 | 0.764 | 0.255 | 0.795 | 0.786 | 166.3 |

M3v2 分 split：

| Split | JSON | Schema | Field EM | Risk F1 | Audit Acc | High-risk Miss | Evidence | BBox Relaxed | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test_clean | 1.000 | 0.816 | 0.815 | 0.781 | 0.734 | 0.284 | 0.731 | 0.719 | 186 |
| test_robust | 1.000 | 0.804 | 0.803 | 0.801 | 0.740 | 0.279 | 0.722 | 0.718 | 165 |
| test_unseen_template | 1.000 | 0.860 | 0.859 | 0.747 | 0.732 | 0.288 | 0.767 | 0.756 | 183 |
| test_hard_negative | 1.000 | 1.000 | 1.000 | 0.636 | 0.852 | 0.168 | 0.961 | 0.949 | 131 |

图表：

![M2/M3/M3v2 核心指标](docs/experiments/phase08_m3v2_sample500/figures/m2_m3_m3v2_core_metrics.png)

![M3v2 分 split 指标](docs/experiments/phase08_m3v2_sample500/figures/m3v2_split_metrics.png)

![M2 到 M3v2 错误迁移](docs/experiments/phase08_m3v2_sample500/figures/m2_m3v2_error_transitions.png)

### 10.5 Two-candidate DPO v2 Ablation

为了节省服务器成本，完整 5 候选 ablation 被截停，只完成：

- `dpo_v2_baseline`
- `auxdpo_v2_strong`

后续 `auxdpo_v2_stronger`、`ipo_v1`、`ipo_aux_v1` 未继续跑。

Train decode dev 结果：

| Variant | Cases | JSON | Schema | Field EM | Risk F1 | Audit Acc | High-risk Miss | Evidence | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dpo_v2_baseline | 152 | 1.000 | 0.868 | 0.868 | 0.901 | 0.836 | 0.230 | 0.810 | 44 |
| auxdpo_v2_strong | 152 | 1.000 | 0.868 | 0.868 | 0.901 | 0.836 | 0.230 | 0.810 | 42 |

图表：

![Train decode core metrics](docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/figures/train_decode_core_metrics.png)

![Train decode validity metrics](docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/figures/train_decode_validity_metrics.png)

![Train decode error cases](docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/figures/train_decode_error_cases.png)

结论：AuxDPO 只减少 2 个 error cases，核心业务指标没有改善，不值得直接扩大到 sample500。

## 11. DPO 实验结果分析

### 11.1 为什么 DPO v1 失败

DPO v1 表面成功：

- loss 从 `0.6878` 降到 `0.000568`。
- preference margin 从 `0.1075` 增到 `74.731`。
- chosen reward 均值为 `1.0`，rejected reward 均值为 `0.4601`。

但业务结果失败：

- `Audit Accuracy` 从 M2 `0.7735` 降到 M3 `0.6685`。
- `Error Cases Avg` 从 `164.5` 升到 `211.0`。
- `High-risk Miss Rate` 只从 `0.2427` 小幅到 `0.2373`，没有达到至少下降 `0.03` 的目标。

这说明 DPO v1 学会了区分当前 pair，却没有学会更稳健的审计行为。

### 11.2 错误迁移

DPO 失败诊断归档在 `docs/experiments/phase08_dpo_diagnosis/`。

图表：

![DPO transition counts](docs/experiments/phase08_dpo_diagnosis/figures/dpo_transition_counts.png)

![Issue count shift](docs/experiments/phase08_dpo_diagnosis/figures/issue_count_shift.png)

![M3 minus M2 metric delta](docs/experiments/phase08_dpo_diagnosis/figures/m3_minus_m2_metric_delta.png)

核心现象：

| 问题 | 解释 |
| --- | --- |
| pair 太容易 | loss 接近 0、margin 极大，说明模型把训练 pair 区分开了，但 pair 可能不能代表真实生成错误 |
| 目标过窄 | DPO 只优化 chosen/rejected 偏好，不直接优化 Audit Accuracy、Evidence Support 或 High-risk Miss |
| 审计边界负迁移 | 模型在一些 M2 原本正确的 case 上变错，说明偏好训练破坏了部分 SFT 能力 |
| 训练强度过大 | sample1000 已把 margin 拉到非常大，可能过度拟合 pair 信号 |
| 缺少 holdout 监控 | v1 主要看训练动态，缺少独立 pair holdout 与 train decode dev 闭环 |

### 11.3 DPO v2 修复了什么

DPO v2 的改进包括：

- Train-only 数据来源。
- case-level holdout 和 decode-dev。
- hard rejected、high-risk miss、protective、normal calibration 四类 pair。
- weighted DPO。
- 较低 learning rate。
- `lambda_sft * chosen NLL` 辅助项。
- holdout pair accuracy 和 margin 监控。

结果上，M3v2 相比 M3：

- `Audit Accuracy` 从 `0.6685` 回升到 `0.7645`。
- `Error Cases Avg` 从 `211.0` 降到 `166.3`。
- Evidence Support 只小幅下降，在可接受边界内。

但 M3v2 没有解决关键目标：

- High-risk Miss 从 M2 `0.2427` 变为 M3v2 `0.2546`。
- 方向变差，不满足“至少下降 0.03”的成功标准。

### 11.4 最终 DPO 判断

当前 DPO 结论应写成负结果，而不是“还没调好”的模糊结论：

```text
DPO v1: loss/margin 成功，但业务显著失败。
DPO v2: 修复总体 accuracy 负迁移，但 high-risk miss 目标失败。
two-candidate ablation: AuxDPO 没有带来核心指标改善。
```

因此不建议直接扩大 DPO/GRPO。下一步应先修 high-risk miss 的错误机制和数据分布。

## 12. High-risk Repair 与后续路线

### 12.1 当前修复包

High-risk Repair Pack 归档：

```text
docs/experiments/phase08_high_risk_repair_pack_20260813/
```

主要文件：

| 文件 | 内容 |
| --- | --- |
| `high_risk_miss_diagnosis_report.md` | 高风险漏检诊断报告 |
| `metric_snapshot.csv` | M2/M3/M3v2/two-candidate 指标快照 |
| `error_source_summary.csv` | 错误来源归因 |
| `representative_high_risk_cases.jsonl` | 代表性高风险错误 |
| `candidate_cases.jsonl` | 候选修复 case |
| `repair_pack_sft.jsonl` | 120 条 repair SFT 样本 |
| `repair_sft_train_mix.jsonl` | 240 条 repair + calibration mix |
| `leakage_check.json` | 泄漏检查 |
| `repair_pack_manifest.json` | manifest |

Repair Pack 约束：

| 约束 | 值 |
| --- | --- |
| selected candidates | 120 |
| DPO v2 holdout overlap | 0 |
| train decode dev overlap | 0 |
| sample500 overlap | 0 |
| policy | Train-only high-risk non-pass repair candidates |

### 12.2 repair_sft_r1 目标

下一步不是继续堆 DPO/GRPO，而是跑小规模 repair SFT：

```text
120 high-risk repair samples
+ 120 calibration samples
-> repair_sft_r1 LoRA-SFT
-> only train_decode_dev inference
-> evaluate_all
-> gate decision
```

验收门槛：

| 指标 | 门槛 |
| --- | --- |
| JSON Validity | `1.0` |
| Audit Accuracy | 不低于 M2，或最多下降 `0.01` |
| High-risk Miss Rate | 相比 M2 至少下降 `0.03` |
| Evidence Support Rate | 最多下降 `0.01` |

停止条件：

- 如果只减少 error cases，但 High-risk Miss 不降，停止训练路线。
- 如果 schema invalid 明显存在，先修 prompt/schema 输出契约。
- 小集不达标，不进入 sample500/test。

## 13. 复现实验与代码索引

### 13.1 快速代码索引

完整程序清单： [docs/code_inventory.md](docs/code_inventory.md)

| 想看什么 | 入口 |
| --- | --- |
| 数据生成 | `src/mv_audit/data_gen/` |
| 图片渲染和 bbox | `src/mv_audit/rendering/` |
| SFT/DPO/GRPO 数据构造 | `src/mv_audit/converters/` |
| SFT/DPO/GRPO 训练 | `src/mv_audit/training/` |
| 批量推理 | `src/mv_audit/inference/batch_inference.py` |
| 评测指标 | `src/mv_audit/evaluation/evaluate_all.py` |
| DPO 失败诊断 | `src/mv_audit/analysis/dpo_error_migration.py` |
| High-risk Repair | `src/mv_audit/analysis/high_risk_repair_pack.py` |

### 13.2 主要脚本入口

| 目标 | 脚本 |
| --- | --- |
| 准备环境 | `scripts/00_prepare_env.sh` |
| 下载 Qwen3-VL | `scripts/00_download_qwen3vl.sh` |
| 生成 main case | `scripts/01_generate_main_cases.sh` |
| 渲染 main 图片 | `scripts/02_render_main_images.sh` |
| 构造 SFT/DPO/GRPO 数据 | `scripts/03_build_main_train_data.sh` |
| 训练 SFT | `scripts/04_train_sft.sh` |
| 训练 DPO v1 | `scripts/05_train_dpo.sh` |
| 构造 DPO v2 pairs | `scripts/05_build_dpo_v2_pairs.sh` |
| 训练 DPO v2 | `scripts/05_train_dpo_v2.sh` |
| 通用推理 | `scripts/07_run_inference.sh` |
| 通用评测 | `scripts/08_evaluate.sh` |
| DPO 错误迁移 | `scripts/09_analyze_dpo_error_migration.sh` |
| DPO v2 ablation | `scripts/10_run_dpo_v2_ablation_5gpu_server.sh` |
| DPO v2 本地归档 watcher | `scripts/10_watch_and_archive_dpo_v2_ablation.ps1` |
| repair_sft_r1 小闭环 | `scripts/11_run_high_risk_repair_sft_r1_server.sh` |
| 服务器只读状态检查 | `outputs/runtime/*/LATEST_RUN_ID`、`FAILED`、`READY_TO_ARCHIVE`、日志目录 |

### 13.3 关键归档路径

| 实验 | 路径 |
| --- | --- |
| Phase07 M0/M1/M2 sample500 | `docs/experiments/phase07_sample500/` |
| Phase08 DPO v1 sample1000 | `docs/experiments/phase08_dpo_sample1000/` |
| Phase08 M3 sample500 | `docs/experiments/phase08_m3_sample500/` |
| DPO 失败诊断 | `docs/experiments/phase08_dpo_diagnosis/` |
| Phase08 M3v2 sample500 | `docs/experiments/phase08_m3v2_sample500/` |
| DPO v2 baseline partial | `docs/experiments/phase08_loss_ablation_baseline_partial_20260812_5gpu_ablation_r3/` |
| two-candidate decode dev | `docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/` |
| High-risk Repair Pack | `docs/experiments/phase08_high_risk_repair_pack_20260813/` |

### 13.4 当时的待办及最终处理

| Phase 8 当时的待办 | 最终处理 |
| --- | --- |
| repair_sft_r1 服务器训练 | 后续已完成 R1，并继续演进到 R2/R3；六级 adapter 谱系已归档 |
| repair_sft_r1 train_decode_dev 推理与评测 | 后续已完成 paired diff、schema/order-id 归因和 R3 152-case gate |
| repair_sft_r1 是否进入 sample500 | 没有进入；Phase 9/10 坚持 Train-only 开发边界 |
| 正式 GRPO/M4 | 未扩大；最终选择成本更低、可复现的 Model-Mined DPO 作为偏好对齐研究 |
| Phase08 negative result | 已纳入完整实验链，并由 DPO v3 probe/full gate 补充方法诊断 |

### 13.5 Git 与资产边界

Git 中保留：

- 代码。
- 配置。
- 小词典。
- README 和文档。
- 实验 metrics、图表、error cases 摘要、manifest。

Git 中不保留：

- 模型权重。
- LoRA/DPO checkpoint 大文件。
- 全量 predictions。
- 原始训练日志全集。
- 大规模渲染图片。

当前 README 只报告已验证结果；未完成的 `repair_sft_r1` 不被写成已完成。

## 14. 服务器工程实现实录

本章回答一个很具体的问题：连接上 Linux 服务器后，这个项目到底是怎么跑起来的？

结论先说清楚：不是点击“运行”按钮，也不是在网页控制台里手动点某个训练入口。本项目的服务器实验主要通过 SSH 登录 Linux，进入远端仓库目录，显式配置 Python/Conda 路径，然后用 `bash` 脚本、`nohup` 后台任务、日志轮询、`pgrep`、`nvidia-smi` 和归档 watcher 完成训练、推理、评测和关机保护。

本章事实分两类：

- 仓库归档日志可验证：当前本地仓库里已有 `docs/experiments/.../logs/`、manifest、metrics、README_APPEND 等文件，可直接复查。
- 历史执行摘要可追溯：部分早期环境修复、数据软链接和依赖处理来自会话执行摘要；这些内容在本地归档中没有完整原始 shell transcript，因此在文中标为“历史执行摘要记录显示”。

### 14.1 初始只读检查：先判断本地有没有结果

在真正判断“服务器训练怎么样”之前，先做本地只读检查。原因是本地仓库和服务器不是实时同步的：本地没有 checkpoint、predictions 或 metrics，不等于服务器没有在跑；同样，本地有旧归档，也不等于服务器当前还在线。

本地通常先看这些位置：

```powershell
git status --short --branch
rg --files outputs/checkpoints outputs/predictions outputs/eval_reports docs/experiments
rg -n "metrics_summary|High-risk Miss|Audit Accuracy|READY_TO_ARCHIVE|FAILED" docs/experiments outputs
```

这些检查解决的是“本地已经保存了什么”。它不能替代服务器状态检查，因为服务器上的长任务、GPU 利用率、`nohup` 日志和 `READY_TO_ARCHIVE` 标记都在远端 `/root/autodl-tmp/VLM-Post-Training` 下。

### 14.2 SSH 连接与服务器目录

历史执行摘要记录显示，服务器项目目录和数据目录为：

| 项目 | 路径 |
| --- | --- |
| SSH 用户和主机 | `<USER>@<HOST>` |
| 端口 | 随当次服务器实例变化，运行时通过 `<PORT>` 传入 |
| 远端代码仓库 | `/root/autodl-tmp/VLM-Post-Training` |
| 远端上传数据 | `/root/autodl-tmp/data` |
| 可用 Python | `/root/miniconda3/bin/python` |

代表性连接方式：

```bash
ssh -p <PORT> <USER>@<HOST>
cd /root/autodl-tmp/VLM-Post-Training
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"
```

这里的 `<PORT>` 不是永久固定配置，而是云服务器实例当次分配的 SSH 端口。README 不建议把历史端口当作永远可用的连接方式。

### 14.3 环境配置：用 shell 脚本，不用 GUI

服务器环境不是通过图形界面配置的。核心入口是：

| 目标 | 文件 |
| --- | --- |
| 安装 Python 依赖 | `scripts/00_prepare_env.sh` |
| 下载或定位基座模型 | `scripts/00_download_qwen3vl.sh` |
| SFT 训练入口 | `scripts/04_train_sft.sh` |
| DPO v2 训练入口 | `scripts/05_train_dpo_v2.sh` |
| Phase08 DPO v2 服务器编排 | `scripts/10_run_dpo_v2_ablation_5gpu_server.sh` |

`scripts/00_prepare_env.sh` 安装的是项目运行依赖，例如 `transformers`、`qwen-vl-utils`、`accelerate`、`peft`、`trl`、`pillow`、`opencv-python`、`jsonschema`、`pyyaml`、`huggingface_hub`、`modelscope`。脚本明确不默认安装 `flash-attn`，因为它对 CUDA、torch 和编译环境要求更高，容易把一次可控实验变成环境编译排错。

历史执行摘要记录显示，服务器环境实际遇到过这些兼容问题：

| 问题 | 表现 | 处理 |
| --- | --- | --- |
| 默认 shell 找不到可用 `python/conda` | 直接执行 `python` 或 `conda` 不可靠 | 显式使用 `/root/miniconda3/bin/python`，并把 `/root/miniconda3/bin` 加到 `PATH` |
| NumPy `2.2.6` ABI 不兼容 | torch/matplotlib 相关导入异常 | 降到 `numpy==1.26.4` |
| torch `2.1.2` 与新 transformers/Qwen3-VL 不兼容 | 缺少 `register_pytree_node` 相关能力 | 升级到 torch `2.4.1` 方向 |
| 长 SSH 命令容易断 | 安装或训练中途连接抖动 | 长任务改为 `nohup` 后台运行，短命令轮询日志 |

环境配置的关键点是“脚本化和可追踪”：依赖安装、模型下载、训练启动都留在 shell 脚本或日志里，而不是依赖人工点按钮。

### 14.4 烟熏测试：证明能跑，不等于证明有效

服务器正式训练前，会先做 smoke/dry-run。典型检查包括：

```bash
nvidia-smi
python -m compileall src/mv_audit tests
DRY_RUN=1 MAX_SAMPLES=8 CONFIG=configs/train/dpo_v2_baseline_ablation_qwen3vl_8b.yaml bash scripts/05_train_dpo_v2.sh
pgrep -af train_dpo
pgrep -af batch_inference
tail -f outputs/runtime/dpo_v2_ablation_5gpu/<RUN_ID>/main.log
```

`docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/logs/main.log` 中可验证：

- 服务器启动时有 5 张 `NVIDIA vGPU-32GB`，每张约 `32760 MiB`。
- `python -m compileall src/mv_audit ... tests` 已运行。
- 当时 `pytest` 不存在，因此日志写入 `pytest not found; skipping unit test`。
- 五个 DPO v2 候选都先跑了 `dry_run`。

smoke/dry-run 的含义非常有限：

| 检查 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| `compileall` | Python 文件语法能被编译 | 模型指标有效 |
| DPO/SFT dry-run | 配置、数据读取、训练入口能走通少量样本 | 完整训练一定能结束 |
| `nvidia-smi` | GPU 是否可见、显存和利用率大致状态 | 模型是否在有效学习 |
| `pgrep` | 进程是否存在 | 评测结果是否变好 |
| `tail` 日志 | 当前脚本执行到了哪一步 | 业务指标是否达标 |

所以 README 中所有“dry-run 通过”都只表示工程入口可用，不等于实验结论。

### 14.5 SFT 服务器流程：先修数据可用性，再训练

历史执行摘要记录显示，SFT 阶段没有直接重渲染全量图片，而是先处理“服务器上哪些图片真的存在”的问题。

服务器上曾有这样的数据现实：

- 代码仓库内的 `data/` 体积较小。
- 上传的大数据在 `/root/autodl-tmp/data`。
- SFT 所需图片并非每一行都完整存在。
- 测试四个 split 的图片和 annotations 较完整。

因此采用的流程是：

```bash
cd /root/autodl-tmp/VLM-Post-Training
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"

# 逻辑动作：把仓库期望的数据入口指向上传数据
# images_main -> /root/autodl-tmp/data/...
# annotations_main -> /root/autodl-tmp/data/...

# 逻辑动作：先统计缺失图片，再构造只包含完整图片的训练子集
# 输出 train_existing_images.jsonl / val_existing_images.jsonl

DRY_RUN=1 MAX_SAMPLES=64 CONFIG=configs/train/sft_lora_qwen3vl_8b_server_debug.yaml bash scripts/04_train_sft.sh
```

为什么不直接运行全量重渲染？因为 `render_all.py --all_splits` 或 `scripts/02_render_main_images.sh` 会重写整批图片和 annotations。对于服务器已有大数据但存在少量缺图的情况，更稳妥的路线是先统计缺图，再过滤完整样本或只补缺失样本。历史执行摘要记录显示，最终生成了 `train_existing_images.jsonl`、`val_existing_images.jsonl` 和 64/16 debug 子集，并得到 `phase07_sft_dry_run=ok`。

### 14.6 DPO v2 服务器流程：五候选尝试、OOM、fallback

DPO v2 的服务器主控脚本是：

```bash
RUN_ID=20260812_5gpu_ablation_r3 nohup bash scripts/10_run_dpo_v2_ablation_5gpu_server.sh > outputs/runtime/dpo_v2_ablation_5gpu/20260812_5gpu_ablation_r3.nohup.log 2>&1 &
```

脚本内部做了几件事：

- 写入 `outputs/runtime/dpo_v2_ablation_5gpu/LATEST_RUN_ID`。
- 为 run 创建 `outputs/runtime/dpo_v2_ablation_5gpu/<RUN_ID>/`。
- 设置 `READY_TO_ARCHIVE` 和 `FAILED` 两类终止标记。
- 先对五个候选做 dry-run。
- 尝试五个候选各占一张 GPU 并发训练。
- 如果并发失败，则对未完成候选逐个使用 `CUDA_VISIBLE_DEVICES="0,1,2,3,4"` 顺序训练。

当次五个候选为：

| 候选 | 配置 |
| --- | --- |
| `dpo_v2_baseline` | `configs/train/dpo_v2_baseline_ablation_qwen3vl_8b.yaml` |
| `auxdpo_v2_strong` | `configs/train/dpo_v2_auxstrong_qwen3vl_8b.yaml` |
| `auxdpo_v2_stronger` | `configs/train/dpo_v2_auxstronger_qwen3vl_8b.yaml` |
| `ipo_v1` | `configs/train/dpo_v2_ipo_qwen3vl_8b.yaml` |
| `ipo_aux_v1` | `configs/train/dpo_v2_ipo_aux_qwen3vl_8b.yaml` |

`main.log` 可验证，当次流程在 `2026-08-12T14:43:21+08:00` 启动五候选并发训练，但约两分钟后全部 `train_failed`，随后进入：

```text
parallel training had failures; retrying incomplete variants sequentially with all GPUs
```

历史执行摘要记录显示，失败原因是 32GB 单卡上同时放 DPO policy/reference 显存不足，典型错误为 `torch.OutOfMemoryError`，显存接近 31GB。fallback 不是 DDP，也不是 FSDP，而是单候选使用 5 张 GPU 的 `device_map=auto` 模型分片。因此它能降低单卡显存压力，但计算利用率不一定平均分布到 5 张卡。

当次可验证的训练结果：

| 时间 | 事件 |
| --- | --- |
| `2026-08-12T14:45:15+08:00` | 开始 `dpo_v2_baseline` 五卡顺序训练 |
| `2026-08-12T18:08:36+08:00` | `dpo_v2_baseline` 训练完成 |
| `2026-08-12T18:08:36+08:00` | 开始 `auxdpo_v2_strong` 五卡顺序训练 |
| `2026-08-12T21:32:35+08:00` | `auxdpo_v2_strong` 训练完成 |

随后用户要求“第二个候选完成后就停，不继续跑剩下 3 个候选”，因此没有继续完整运行 `auxdpo_v2_stronger`、`ipo_v1`、`ipo_aux_v1`。

### 14.7 Decode 恢复流程：从 baseline 74/152 接着跑

截停脚本相关文件在远端 runroot 下：

```bash
bash outputs/runtime/dpo_v2_ablation_5gpu/20260812_5gpu_ablation_r3/curtail_after_auxstrong.sh
```

这里有一个 Linux 脚本细节：`curtail_after_auxstrong.sh` 出现过带 BOM 的 shebang 问题，归档日志中能看到：

```text
curtail_after_auxstrong.sh: line 1: ﻿#!/usr/bin/env: No such file or directory
```

解决方式是显式用 `bash` 调脚本，而不是依赖第一行 shebang 自动解释。

后续 resume 日志位于：

```text
docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/logs/resume_curtail_decode_20260813_090155.nohup.log
```

该日志可验证：

| 时间 | 事件 |
| --- | --- |
| `2026-08-13T09:01:55+08:00` | `curtail watcher started` |
| `2026-08-13T09:01:55+08:00` | 检测到 auxstrong adapter，停止原主控后续候选 |
| `2026-08-13T09:02:01+08:00` | 开始 `dpo_v2_baseline` decode |
| `2026-08-13T10:29:13+08:00` | baseline decode 完成 |
| `2026-08-13T10:29:13+08:00` | 开始 `auxdpo_v2_strong` decode |
| `2026-08-13T13:30:11+08:00` | auxstrong decode 完成 |

推理使用 `--resume` 语义，保留已写出的预测，不删除、不重跑 baseline 前 74 条。最终业务产物显示：

- `dpo_v2_baseline`：`152/152`，其中 resume 新增 `78` 条，跳过已有 `74` 条。
- `auxdpo_v2_strong`：`152/152`，新增 `152` 条。

这就是第 10.5 节 two-candidate Train decode dev 对比表的来源。

### 14.8 自动归档与关机：本地 watcher 和服务器 watcher 的区别

本项目尝试过两种 watcher。

第一种是本地 PowerShell watcher：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/10_watch_and_archive_dpo_v2_ablation.ps1 `
  -HostName <HOST> `
  -Port <PORT> `
  -User root `
  -RemoteProject /root/autodl-tmp/VLM-Post-Training `
  -LocalRepo "D:\Reserach\Projects\VLM post-training" `
  -PollSeconds 120 `
  -ShutdownOnSuccess
```

它的策略是：

1. SSH 轮询远端 `LATEST_RUN_ID`。
2. 如果远端出现 `FAILED`，本地记录失败并退出，不关机。
3. 如果远端出现 `READY_TO_ARCHIVE`，读取 `archive_tar_path`。
4. 用 `scp` 拉取 tar。
5. 解压到本地真实归档目录。
6. 校验 `artifact_manifest.json` 的 SHA256。
7. 检查 README append 和 `git diff --check`。
8. 只有本地拉取和校验成功，才发 `shutdown -h now`。

后来因为用户准备关闭本地电脑，策略改成服务器端 watcher：不再依赖本地持续在线，而是在服务器上等 tar 归档成功后自行关机。归档中的脚本为：

```text
docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/logs/server_auto_shutdown_after_archive_20260813_090155.sh
```

核心逻辑是：

```bash
if test -f "$runroot/FAILED"; then
  echo "remote_failed_no_shutdown=$(date -Is)" >> "$log"
  cat "$runroot/FAILED" >> "$log"
  exit 2
fi

if test -f "$runroot/READY_TO_ARCHIVE" && test -s "$runroot/archive_tar_path"; then
  archive=$(cat "$runroot/archive_tar_path")
  if test -s "$archive" && tar -tzf "$archive" >/dev/null 2>&1; then
    sync
    sleep 10
    shutdown -h now
  fi
fi
```

这里最重要的安全边界是：只要出现 `FAILED`，watcher 就不关机。当前归档中的 `server_auto_shutdown_after_archive_active.log` 可验证，当次 watcher 记录了：

```text
auto_shutdown_watcher_start=2026-08-13T09:01:55+08:00 run_id=20260812_5gpu_ablation_r3
remote_failed_no_shutdown=2026-08-13T13:31:55+08:00
```

原因是 resume 脚本末尾出现 `NameError: name 'sys' is not defined`，写入了 `FAILED`。虽然 baseline 和 auxstrong decode 已实际完成，并且后续本地归档中已有 two-candidate metrics，但服务器端 watcher 按安全策略选择“不关机”。这是一种保守保护：宁可人工确认，也不在异常标记存在时自动关机。

### 14.9 服务器执行中出现的问题与解决

| 问题 | 发生位置 | 影响 | 解决或当前策略 |
| --- | --- | --- | --- |
| 32GB 单卡 DPO OOM | 五候选单卡并发训练 | 五个候选并发均失败 | 改为单候选 5 卡 `device_map=auto` 顺序训练 |
| `device_map=auto` 被误解为 5 卡并行 | 状态汇报和资源判断 | 容易高估 5 卡计算利用率 | README 明确它是模型分片，不是 DDP/FSDP |
| BOM shebang | `curtail_after_auxstrong.sh` | 直接执行脚本时报 `#!/usr/bin/env` 不存在 | 显式 `bash outputs/runtime/.../curtail_after_auxstrong.sh` |
| `ERR` trap 误写 `FAILED` | DPO 主控脚本 fallback 阶段 | 预期 OOM fallback 被记录成终态失败 | fallback 阶段需要抑制 terminal `FAILED`，并传播真实子进程退出码 |
| resume 末尾 `NameError: sys` | `resume_curtail_decode_20260813_090155.nohup.log` | decode 完成后仍写入 `FAILED`，服务器 watcher 不关机 | 保留安全策略，先人工核对 metrics 和归档 |
| PowerShell 带空格路径 | 本地 watcher 启动 | `D:\Reserach\Projects\VLM post-training` 易被拆参数 | PowerShell 参数统一加引号，必要时前台验证日志 |
| SCP 多文件中断 | 服务器到本地拉取 | 可能部分文件缺失但误以为完成 | 单文件重拉，检查文件大小、manifest、SHA256 |
| 中文 UTF-8 乱码 | PowerShell 生成 README/报告 | 中文可能显示为问号或乱码 | `Get-Content -Encoding UTF8` 抽查，并用关键词搜索 |
| 显存占用不等于计算利用率 | `nvidia-smi` 状态判断 | 看到显存占用但 GPU-Util 低 | 同时看 `pgrep`、日志、checkpoint、READY/FAILED |

### 14.10 代表性命令清单

下面命令是工程记录格式，展示服务器上如何执行和检查。昂贵训练命令不要在没有重新确认的情况下直接重跑。

```bash
# 连接服务器
ssh -p <PORT> <USER>@<HOST>

# 进入项目并固定 Python/Conda 路径
cd /root/autodl-tmp/VLM-Post-Training
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"

# 查看 GPU
nvidia-smi

# 本地语法级 smoke
python -m compileall src/mv_audit tests

# DPO v2 少样本 dry-run
DRY_RUN=1 MAX_SAMPLES=8 CONFIG=configs/train/dpo_v2_baseline_ablation_qwen3vl_8b.yaml bash scripts/05_train_dpo_v2.sh

# 启动服务器端 DPO v2 ablation 主流程
RUN_ID=20260812_5gpu_ablation_r3 nohup bash scripts/10_run_dpo_v2_ablation_5gpu_server.sh > outputs/runtime/dpo_v2_ablation_5gpu/20260812_5gpu_ablation_r3.nohup.log 2>&1 &

# 只读查看进程和日志
pgrep -af train_dpo
pgrep -af batch_inference
tail -f outputs/runtime/dpo_v2_ablation_5gpu/20260812_5gpu_ablation_r3/main.log

# 手动恢复 two-candidate decode
bash outputs/runtime/dpo_v2_ablation_5gpu/20260812_5gpu_ablation_r3/curtail_after_auxstrong.sh

# 归档可读性检查
tar -tzf docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3.tar.gz

# 只有 watcher 判断归档完整且无 FAILED 时，才允许关机
shutdown -h now
```

Windows 本地拉取和校验时的代表性命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/10_watch_and_archive_dpo_v2_ablation.ps1 -ShutdownOnSuccess -PollSeconds 120
git diff --check
Get-Content -Encoding UTF8 README.md -TotalCount 60
```

### 14.11 本次服务器相关代码和脚本补充

| 文件 | 类型 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| `scripts/10_run_dpo_v2_ablation_5gpu_server.sh` | 服务器编排脚本 | 组织 Phase08 DPO v2 五候选 dry-run、训练、decode、评测和归档；维护 `LATEST_RUN_ID`、`FAILED`、`READY_TO_ARCHIVE` | 已用于 `20260812_5gpu_ablation_r3` |
| `scripts/10_watch_and_archive_dpo_v2_ablation.ps1` | 本地 watcher | 从远端读取真实 tar 路径，拉取归档，校验 manifest，必要时 append README，校验通过后可发关机命令 | 已修正为按真实 tar 目录名解压 |
| `outputs/runtime/.../curtail_after_auxstrong.sh` | 远端临时截停脚本 | 在第二个候选完成后停止剩余三候选，并恢复 baseline/auxstrong Train decode dev | 远端生成，归档日志记录了执行过程 |
| `logs/server_auto_shutdown_after_archive_20260813_090155.sh` | 服务器端 watcher | 本地电脑关闭时，服务器自行等待归档 tar 可读后关机；若 `FAILED` 存在则不关机 | 当次因 `FAILED` 触发保护，未关机 |
| `scripts/11_run_high_risk_repair_sft_r1_server.sh` | repair SFT 服务器入口 | 构造 repair mix，SFT dry-run，训练 repair_sft_r1，只跑 Train decode dev，归档结果 | 已完成历史 R1 阶段，后续继续到 R2/R3 |
| `src/mv_audit/analysis/high_risk_repair_pack.py` | 分析脚本 | 诊断 high-risk miss，生成 120 条 Train-only repair cases | 已生成 repair pack |
| `src/mv_audit/converters/build_high_risk_repair_sft_mix.py` | 数据构造脚本 | 将 120 条 repair cases 与 120 条 calibration 样本混合为 repair SFT 训练集 | 已 dry-run 验证 |
| `src/mv_audit/analysis/archive_high_risk_repair_sft.py` | 归档脚本 | 归档 repair_sft_r1 的配置、日志、metrics、摘要和 manifest，不默认归档大 checkpoint | 已用于历史 repair 流程；最终六级 adapter 由 lineage audit 统一核验 |

### 14.12 服务器复现入口和安全边界

如果后续要复现服务器流程，推荐顺序是：

1. 先读 `docs/code_inventory.md`，明确每个脚本和模块负责什么。
2. 本地只读检查 `docs/experiments/`、`outputs/`、`git status`。
3. SSH 到服务器后只读检查 `LATEST_RUN_ID`、`FAILED`、`READY_TO_ARCHIVE`、`pgrep`、`nvidia-smi`、最新日志。
4. 只在用户明确确认后启动昂贵训练或推理。
5. 长任务一律用 `nohup` 或等价后台方式，并写入 runroot 日志。
6. 归档必须先生成 manifest，再校验 tar 可读性。
7. 自动关机必须以“无 `FAILED` 且归档可读”为前提。

本章列出的服务器命令是工程实现记录，不是无条件复跑说明。尤其是 DPO v2、repair_sft_r1、sample500/test 推理都属于昂贵或有数据边界要求的操作，不能在没有新确认的情况下直接执行。

## 15. Structured Repair SFT v3 与 Model-Mined DPO v3

Phase09/10 将项目从历史 M2/DPO v1/v2 扩展到面向真实残留错误的 Structured Repair SFT 和 model-error-mined preference alignment。完整实验报告见 [Phase 10 Model-Error-Mined DPO v3](docs/experiments/phase10_model_error_mined_dpo_v3/README.md)。

### 15.1 后训练链路

![VLM 后训练链路](docs/experiments/phase10_model_error_mined_dpo_v3/figures/post_training_pipeline.png)

可编辑源文件：[post_training_pipeline.drawio](docs/experiments/phase10_model_error_mined_dpo_v3/figures/post_training_pipeline.drawio)

Structured Repair SFT v3 使用 480 条 Train-only mix，从 R2 adapter 继续训练；Model-Mined DPO v3 从 SFT v3 的真实生成错误构造困难偏好对，并使用独立 probe 和 152 条 full gate 两级选择。sample500/Test/final holdout 均未用于 Phase10 调参或候选选择。

### 15.2 分 benchmark 指标

历史 sample500 与当前 train_decode_dev 不是同一评测协议，必须分栏报告：

| Benchmark | 模型 | Audit Accuracy | High-risk Miss | Evidence Support | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| sample500 四 split 平均 | M2 SFT | 0.7735 | 0.2427 | 0.8035 | 历史 baseline |
| sample500 四 split 平均 | DPO v1 | 0.6685 | 0.2373 | 0.7987 | research ablation |
| sample500 四 split 平均 | DPO v2 | 0.7645 | 0.2546 | 0.7952 | research ablation |
| sample500 四 split 平均 | Structured Repair SFT v3 | 0.6075 | 0.4217 | 0.6801 | final 失败后诊断 |
| train_decode_dev 152 | Structured Repair SFT v3 | **0.9671** | **0.0575** | 0.9876 | production candidate |
| train_decode_dev 152 | DPO v3 checkpoint-15 | 0.8684 | 0.1379 | **0.9904** | alignment research candidate |

![分 benchmark 指标](docs/experiments/phase10_model_error_mined_dpo_v3/figures/post_training_metrics_by_benchmark.png)

机器可读总表：[post_training_metrics.csv](docs/experiments/phase10_model_error_mined_dpo_v3/post_training_metrics.csv)

### 15.3 DPO v3 的有效信号与边界

checkpoint-15 在 24 条 case-disjoint alignment probe 上将 mean task reward 提升 `0.1667`，order-id 双侧证据命中率提升 `11.11pp`，并修复 `MV_MAIN_004522`、`MV_MAIN_020454` 两个 probe case。这证明偏好训练能够改变目标行为。

但在 152 条 full gate 上，DPO v3 保留了原 5 条 High-risk Miss 并新增 7 条；14 条应拒绝样本被改成 `manual_review`。`MV_MAIN_023069`、`MV_MAIN_015818` 都属于“amount mismatch 已识别、证据正确、审核动作退化”的过拟合案例。因此该 checkpoint 不具备部署资格。

![DPO v3 probe](docs/experiments/phase10_model_error_mined_dpo_v3/figures/dpo_v3_probe_checkpoints.png)

### 15.4 最终模型选择

![模型选择门禁](docs/experiments/phase10_model_error_mined_dpo_v3/figures/model_selection_gate.png)

| 角色 | 模型 | 发布状态 |
| --- | --- | --- |
| `FINAL_HOLDOUT_FAILED` | `repair_sft_r3` | `NOT_DEPLOYED`；final_holdout_v1 已消耗且 sample500 诊断同步退化 |
| `ALIGNMENT_RESEARCH_CANDIDATE` | DPO v3 checkpoint-15 | `deployment_eligible=false`；全量 gate 未通过 |
| `HISTORICAL_SAMPLE500_BASELINE` | M2 SFT | 冻结历史 benchmark |

机器可读选择记录见 [model_selection.json](docs/experiments/phase10_model_error_mined_dpo_v3/model_selection.json)，简历和面试口径见 [docs/resume_vlm_post_training.md](docs/resume_vlm_post_training.md)。

SFT v3 adapter 已归档到 `outputs/model_candidates/repair_sft_r3/` 并完成远端/本地 SHA256 一致性校验。后续 final_holdout_v1 已消耗且失败，sample500 历史口径诊断也退化；该状态是 `FINAL_HOLDOUT_FAILED / NOT_DEPLOYED`，不能解读为已部署。
