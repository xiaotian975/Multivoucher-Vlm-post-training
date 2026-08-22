# MultiVoucher-Audit：多凭证报销审计 VLM 后训练项目

更新时间：2026-08-22

> 本项目基于 `Qwen3-VL-8B-Instruct`，让视觉语言模型同时读取同一报销 case 下的发票、支付截图、报销申请单和订单截图，输出可校验、可定位、可追责的 Evidence-Grounded JSON。它不是普通 OCR，而是一个从合成数据、图片渲染、LoRA-SFT、Repair SFT、DPO 对齐、业务评测到模型归档的完整后训练实验项目。

## 先看最终结论

当前没有可部署模型。

| 对象 | 当前状态 | 说明 |
| --- | --- | --- |
| `repair_sft_r3` | `FINAL_HOLDOUT_FAILED / NOT_DEPLOYED` | 曾在 152 条 Train-only 开发门禁上表现很好，但在独立 `final_holdout_v1` 和历史 `sample500` 诊断中退化 |
| DPO v3 checkpoint-15 | `ALIGNMENT_RESEARCH_CANDIDATE` | 在 24 条 probe 上有局部对齐收益，但 152 条 full gate 未通过，不能替代 SFT |
| M2 LoRA-SFT | `HISTORICAL_SAMPLE500_BASELINE` | 历史 sample500 上最稳的业务基线，用于对比和复盘 |
| `final_holdout_v1` | `FINAL_HOLDOUT_CONSUMED` | 已正式使用并失败，后续只能做诊断，不能再用于训练、调参或选模型 |

最重要的经验是：小规模 Train-only 开发门禁高分，不等于模型已经泛化。`repair_sft_r3` 在 `train_decode_dev` 上 Audit Accuracy 达到 `96.71%`，但在 `final_holdout_v1` 上只有 `71.60%`，因此最终没有部署。

## 项目要解决什么问题

企业费用审核不是只读一张发票。一个真实报销 case 往往包含多张材料：

| 凭证 | 模型需要读出的信息 |
| --- | --- |
| 发票 `invoice` | 发票号、开票日期、销售方、项目、金额、税额、价税合计 |
| 支付截图 `payment` | 支付金额、支付时间、收款方、付款人、支付流水号 |
| 报销申请单 `reimbursement_form` | 申请人、费用类型、报销金额、申请日期、事由、订单号 |
| 订单截图 `order` | 订单号、商品或服务、商户、订单金额、订单用户、下单时间 |

模型输入是一组图片和 `case_id`，输出必须是一个固定 schema 的 JSON：

```text
images + case_id
  -> field_extraction      字段抽取
  -> consistency_check     跨凭证一致性检查
  -> anomaly_types         异常类型
  -> risk_level            风险等级
  -> audit_result          审核建议
  -> reason                简短原因
  -> evidence[]            图片来源、字段、值、bbox、证据文本
  -> uncertainty           不确定字段和是否需要人工复核
```

模型不能只说“有问题”。它必须指出问题来自哪张图、哪个字段、读到了什么值，以及证据框在哪里。完整输出契约见 [docs/global_contracts.md](docs/global_contracts.md)。

## 数据从哪里来

项目从结构化业务真值开始，先生成报销 case，再注入异常，之后渲染成四类凭证图片，并保存字段级 bbox。

```text
业务字典和 schema
  -> 生成正常报销 case
  -> 注入异常并打 risk/audit 标签
  -> case-level split
  -> 渲染 invoice/payment/reimbursement_form/order 图片
  -> 保存 bbox annotations
  -> 转换 SFT/DPO/GRPO/Repair 训练格式
  -> 推理和评测
```

主数据配置来自 [configs/data_gen/main.yaml](configs/data_gen/main.yaml)，总规模为 `41,000` 个 case。所有划分都是 case-level，同一个 case 的多张图片不会被拆到不同 split。

## 数据集和测试集说明

### 主数据 split

| Split | 规模 | 用途 | 构成说明 |
| --- | ---: | --- | --- |
| `train` | 30,000 | SFT、DPO、Repair、model mining 的来源 | 约 9,643 条正常 `none/pass`，其余覆盖金额、超额报销、日期、商户、人员、订单号、缺材料、重复、不可读等异常 |
| `val_in_template` | 2,000 | 验证和开发诊断候选池 | 使用训练模板组，覆盖正常、低/中/高风险和四类审核动作 |
| `val_unseen_template` | 1,000 | 模板泛化验证候选池 | 使用验证模板组，主要检查模型是否依赖固定版式 |
| `test_clean` | 2,000 | 常规测试全集 | 标准测试模板、常规渲染，含正常和全部异常类型 |
| `test_robust` | 2,000 | 视觉鲁棒性测试全集 | 在常规测试上加入亮度、对比度、模糊和噪声扰动，不改变 bbox 坐标系 |
| `test_unseen_template` | 2,000 | 强模板泛化测试全集 | 使用 `strong_generalization_test` 模板组 |
| `test_hard_negative` | 2,000 | 高风险易混负例测试全集 | 只包含 `amount_mismatch`、`merchant_mismatch`、`applicant_mismatch`、`order_id_mismatch` 四类易误判异常 |

### 四个 test split 的异常构成

| Split | none | amount | over | date | merchant | applicant | order-id | missing | duplicate | unreadable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test_clean` | 622 | 236 | 157 | 164 | 165 | 154 | 156 | 152 | 90 | 104 |
| `test_robust` | 601 | 270 | 155 | 165 | 163 | 152 | 136 | 157 | 110 | 91 |
| `test_unseen_template` | 584 | 251 | 154 | 158 | 164 | 171 | 150 | 153 | 110 | 105 |
| `test_hard_negative` | 0 | 600 | 0 | 0 | 457 | 489 | 454 | 0 | 0 | 0 |

### 常见 benchmark 不要混用

| 名称 | 规模 | 来源 | 用途 | 边界 |
| --- | ---: | --- | --- | --- |
| `sample500` | 4 个 split，每个 500，总 2,000 | 从四个 test split 抽样的历史 benchmark | M0/M1/M2/DPO v1/DPO v2 历史对比；R3 失败后诊断 | 不再用于新模型选择或调参 |
| `train_decode_dev` | 152 | Train-only 开发集 | 快速解码门禁、灾难性退化检查、DPO v3 full gate | 不是独立测试集，不能和 sample500/final 直接横比 |
| `final_holdout_v1` | 1,000 | 四个 test split 各 250，排除历史已用 case | 最终冻结评测 | 已消费且失败，只能诊断，不能回流训练 |

`final_holdout_v1` 的构成是四个 split 各 `250` 条；总体异常分布包括 `amount_mismatch=138`、`applicant_mismatch=138`、`merchant_mismatch=137`、`order_id_mismatch=137`、`none=75`、`over_reimbursement=75`、`date_mismatch=75`、`missing_document=75`、`duplicate_in_batch=75`、`unreadable_image=75`。风险分布为 high `736`、medium `189`、low `75`；审核动作分布为 `reject_recommendation=631`、`manual_review=219`、`missing_info=75`、`pass=75`。

### 训练数据规模

| 数据 | 规模 | 说明 |
| --- | ---: | --- |
| SFT main 生成文件 | 28,500 train + 1,500 val | 从主数据转换出的 SFT 格式文件 |
| M2 实际 SFT 训练子集 | 21,682 train + 1,138 val | 服务器上按 existing-images 过滤后的真实训练文件，避免缺图样本污染训练 |
| DPO v2 正式归档 | 3,000 train pairs + 300 holdout pairs + 152 decode-dev rows | Train-only，包含 hard、high-risk miss、protective、normal calibration pair |
| Repair SFT R3 | 480 Train-only mix | 240 条 R1/R2 carryover，120 条 order-id structured repair，120 条 calibration |
| DPO v3 model mining | 240 case x 4 completions | 得到 960 个采样输出，其中 894 个 schema 合法，用于构造困难偏好对 |
| DPO v3 pairs | 120 train pairs + 24 probe/holdout pairs | 以模型真实错误和结构化 expert target 构造，train/holdout overlap 为 0 |

## 后训练主线

### 1. M0/M1：只靠 prompt 不够

M0 是 zero-shot，M1 是 few-shot。两者能证明原始模型具备视觉问答能力，但不能稳定输出本项目要求的 Evidence-Grounded JSON。M0 在历史 sample500 上 Audit Accuracy 为 `0.0000`，M1 也只有 `0.0785`。

### 2. M2 LoRA-SFT：建立基础审计能力

M2 用 existing-images 子集做 LoRA-SFT，使模型学会固定 JSON schema、字段抽取、跨图一致性、风险等级、审核建议和 bbox 证据。M2 在历史 sample500 上达到：

- JSON Validity：`1.000`
- Audit Accuracy：`0.7735`
- High-risk Miss Rate：`0.2427`
- Evidence Support Rate：`0.8035`

这说明 SFT 有效建立了基础能力，但高风险漏检仍偏高。

### 3. DPO v1：训练 loss 好看，但业务退化

DPO v1 使用规则式 easy rejected。训练 loss 很低，preference margin 很高，但 sample500 Audit Accuracy 从 M2 的 `0.7735` 降到 `0.6685`。结论是：偏好 pair 可分，不等于业务审计变好。

### 4. DPO v2：部分恢复，但没解决高风险漏检

DPO v2 引入 Train-only、保护型 pair、hard rejected、normal calibration、weighted DPO 和 SFT auxiliary loss。它把 Audit Accuracy 恢复到 `0.7645`，接近 M2，但 High-risk Miss Rate 为 `0.2546`，仍差于 M2。

因此当时没有继续扩大 GRPO，而是转向错误归因和 Repair SFT。

### 5. Repair SFT R1/R2/R3：针对 order-id 和高风险漏检做结构化修复

Repair SFT 的思路不是小数据从零训练，而是在 M2/R1/R2 已有能力上，用高置信错题和 calibration 做定向纠偏。

R3 特别强化 `order_id_mismatch`：

- `reason` 要写明订单截图订单号 A 与报销单订单号 B 不一致。
- `evidence` 前置订单截图和报销单两侧 order-id 证据。
- 保留 low/pass、manual_review、missing_info 等 calibration，降低遗忘风险。

R3 在 152 条 `train_decode_dev` 上表现很好，但后续 final 和 sample500 诊断证明它没有泛化。

### 6. Model-Mined DPO v3：从模型真实错误中挖 hard pairs

DPO v3 不再主要依赖人工规则篡改的 easy rejected，而是让 R3 对隔离 Train-only case 多次采样，再从模型真实错误中构造 hard preference pairs。

关键改动：

- 从 `240 x 4` 个模型输出中筛选 schema 合法 completion。
- 使用模型真实失败输出和 expert-corrected target 构造 chosen/rejected。
- 使用 assistant-token mean log-prob，避免长 JSON 的 sequence-sum margin 虚高。
- 先跑 24 条 alignment probe，再让唯一候选跑 152 条 full gate。

DPO v3 checkpoint-15 在 probe 上有局部收益，但 full gate 上新增高风险漏检，因此只保留为研究候选。

## 实验结果总表

### final_holdout_v1：最终独立评测

`final_holdout_v1` 已消费。结果不能再用于训练、reward tuning 或 checkpoint 选择。

| total_cases | JSON | Schema | Audit Accuracy | High-risk Miss | Evidence Support | Error Cases | 结论 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1000 | 1.0000 | 0.8460 | 0.7160 | 0.3152 | 0.8358 | 454 | `FINAL_HOLDOUT_FAILED / NOT_DEPLOYED` |

分 split 结果：

| Split | Cases | Schema | Audit Accuracy | High-risk Miss | Evidence Support | Error Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `test_clean` | 250 | 0.8200 | 0.7040 | 0.3392 | 0.8236 | 122 |
| `test_robust` | 250 | 0.8560 | 0.7360 | 0.2840 | 0.8598 | 111 |
| `test_unseen_template` | 250 | 0.7640 | 0.6440 | 0.3920 | 0.7586 | 133 |
| `test_hard_negative` | 250 | 0.9440 | 0.7800 | 0.2455 | 0.9013 | 88 |

详细报告见 [docs/experiments/final_holdout_v1/summary.md](docs/experiments/final_holdout_v1/summary.md)。

### 历史 sample500：M2、DPO v1/v2、R3 诊断

sample500 是历史 benchmark，四个 split 各 500 条。R3 行是 final 失败后的补充诊断，不能用于重新选模。

| 模型 | Audit Accuracy | High-risk Miss | Evidence Support | 结论 |
| --- | ---: | ---: | ---: | --- |
| M2 LoRA-SFT | 0.7735 | 0.2427 | 0.8035 | 历史业务 baseline |
| DPO v1 | 0.6685 | 0.2373 | 0.7987 | loss 收敛但 Audit 明显负迁移 |
| DPO v2 | 0.7645 | 0.2546 | 0.7952 | 恢复 Accuracy，但 HRM 未改善 |
| Structured Repair SFT v3 | 0.6075 | 0.4217 | 0.6801 | final 失败后诊断；相对 M2 明显退化 |

R3 sample500 诊断见 [docs/experiments/repair_sft_r3_sample500_diagnostic/README.md](docs/experiments/repair_sft_r3_sample500_diagnostic/README.md)。

### Train-only train_decode_dev：开发门禁

这里的 `152` 条不是独立测试集，只能说明训练域小样本门禁表现。

| 模型 | JSON / Schema | Audit Accuracy | High-risk Miss | Evidence Support | Error Cases | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Structured Repair SFT v3 | 1.000 / 1.000 | 0.9671 | 0.0575 | 0.9876 | 23 | 开发门禁曾通过 |
| DPO v3 checkpoint-15 | 1.000 / 1.000 | 0.8684 | 0.1379 | 0.9904 | 40 | full gate 未通过 |

Phase 10 完整报告见 [docs/experiments/phase10_model_error_mined_dpo_v3/README.md](docs/experiments/phase10_model_error_mined_dpo_v3/README.md)。

## 当前现存问题

1. **开发门禁和独立泛化存在明显落差。** R3 在 `train_decode_dev` 上高分，但在 `final_holdout_v1` 和 sample500 上退化，说明训练域修复过窄。
2. **高风险漏检仍是核心业务风险。** final holdout 中 `model_missed_high_risk=229`，High-risk Miss Rate 为 `0.3152`。
3. **schema 合法性仍不够稳。** final holdout JSON Validity 是 `1.0000`，但 Schema Compliance 只有 `0.8460`，仍有结构契约失败。
4. **DPO 目标和业务 KPI 容易错位。** DPO v1/v2 的 loss、margin、pair accuracy 都不能直接代表 Audit Accuracy 或 High-risk Miss 改善。
5. **DPO v3 出现决策边界漂移。** probe 能修 order-id 局部问题，但 full gate 中新增 7 条高风险漏检，部分应拒绝 case 被降为 `manual_review`。
6. **sample500 和 final holdout 不能回流。** 它们现在只能做报告和诊断，不能拿来继续训练、调 reward 或选 checkpoint。
7. **服务器数据资产有缺图现实。** 历史 M2 训练使用 existing-images 子集，README 中任何训练规模都要区分“理论生成文件”和“实际训练文件”。

## 下一步合理路线

不应该继续在已消费的 final holdout 上试错。更安全的路线是：

```text
冻结当前失败结论
  -> 从未污染的数据中建立新的 development/validation 闭环
  -> 针对 final 诊断中的问题做 error attribution
  -> 判断问题是感知/证据/schema 还是决策边界
  -> 如果是感知或证据问题，优先做数据和 SFT repair
  -> 如果确认为决策瓶颈，再考虑小规模 RL/reward smoke
  -> 完成全部开发选择后，重新锁定 future final_holdout_v2
```

原则是：先诊断，再修复；先小门禁，再扩大；所有昂贵训练和最终评测都必须有显式允许。

## 代码地图

| 目录或文件 | 作用 |
| --- | --- |
| `src/mv_audit/data_gen/` | 生成 case、注入异常、打风险标签、case-level split |
| `src/mv_audit/rendering/` | 渲染四类凭证图片并记录 bbox |
| `src/mv_audit/perturbation/` | 生成视觉扰动、不可读区域、重复凭证 |
| `src/mv_audit/converters/` | 构造 SFT、DPO、GRPO、Repair 数据 |
| `src/mv_audit/training/` | LoRA-SFT、DPO/IPO、GRPO 和 reward function |
| `src/mv_audit/inference/` | Qwen3-VL 加载、多图输入、adapter 推理、schema guard |
| `src/mv_audit/evaluation/` | JSON/schema、字段、审计、证据、bbox、幻觉等评测 |
| `src/mv_audit/analysis/` | final holdout、错误归因、数据边界和门禁 |
| `tools/` | 实验报告、DPO v3 pair 构造、shard merge、模型谱系审计 |
| `scripts/` | 本地和服务器运行入口 |
| `docs/experiments/` | 小型可进 Git 的实验报告、metrics、manifest、图表 |
| `outputs/` | checkpoint、prediction、runtime log 等大产物或可再生产物 |

更详细的逐文件解释见 [docs/code_inventory.md](docs/code_inventory.md)。

## 常用入口

以下命令是入口索引，不代表可以无条件重跑昂贵实验。

```bash
# 环境准备
bash scripts/00_prepare_env.sh

# 数据生成和图片渲染
bash scripts/01_generate_main_cases.sh
bash scripts/02_render_main_images.sh
bash scripts/03_build_main_train_data.sh

# SFT / DPO / evaluation
bash scripts/04_train_sft.sh
bash scripts/05_train_dpo_v2.sh
bash scripts/07_run_inference.sh
bash scripts/08_evaluate.sh

# Repair SFT v3 和 DPO v3 相关入口
ALLOW_TRAINING=1 bash scripts/12_run_order_id_repair_sft_v3_server.sh
bash scripts/13_run_model_mined_dpo_v3_server.sh
bash scripts/16_run_model_error_mined_dpo_v3_strong_server.sh
bash scripts/17_resume_dpo_v3_strong_full_gate_server.sh
```

正式训练、完整推理、sample500 诊断和 final holdout 都属于昂贵或有数据边界风险的操作，不能在没有新确认的情况下直接运行。

## 关键文档

| 文档 | 适合读什么 |
| --- | --- |
| [docs/project_brief.md](docs/project_brief.md) | 项目目标、任务边界、输入输出 |
| [docs/global_contracts.md](docs/global_contracts.md) | JSON schema、风险规则、审核规则、数据泄漏边界 |
| [docs/code_inventory.md](docs/code_inventory.md) | 代码目录和逐文件职责 |
| [docs/experiments/final_holdout_v1/summary.md](docs/experiments/final_holdout_v1/summary.md) | final holdout 最终失败结果 |
| [docs/experiments/phase10_model_error_mined_dpo_v3/README.md](docs/experiments/phase10_model_error_mined_dpo_v3/README.md) | R3、DPO v3、模型选择和错误迁移 |
| [docs/experiments/repair_sft_r3_sample500_diagnostic/README.md](docs/experiments/repair_sft_r3_sample500_diagnostic/README.md) | R3 在历史 sample500 上的失败后诊断 |
| [docs/resume_vlm_post_training.md](docs/resume_vlm_post_training.md) | 简历和面试口径，注意其中 production candidate 不等于已部署 |

## Git 和归档边界

Git 中保留代码、配置、小型 manifest、metrics、图表和报告。模型权重、完整图片数据、全量 predictions 和大型运行日志通常不直接进入 Git。

模型谱系和归档审计见：

- [docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive_audit.md](docs/experiments/phase10_model_error_mined_dpo_v3/model_lineage_archive_audit.md)
- [docs/experiments/phase10_model_error_mined_dpo_v3/model_selection.json](docs/experiments/phase10_model_error_mined_dpo_v3/model_selection.json)

最终状态再次强调：`repair_sft_r3` 是已归档、可审计的失败候选，不是已部署模型；DPO v3 是有研究价值但未通过业务门禁的对齐候选。
