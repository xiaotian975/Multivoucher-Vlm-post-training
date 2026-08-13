# MultiVoucher-Audit 代码程序清单

更新时间：2026-08-13

本文面向“项目几乎全程由 AI 生成、但需要理解每块代码在做什么”的读者。它不是 API 文档，而是代码地图：每个文件负责什么、属于 SFT/DPO/GRPO/推理/评测中的哪一段、输入输出是什么、和其他文件怎么连接。

## 0. 给非代码读者的总览

### 0.1 一句话理解项目代码

这个项目把“企业报销 case”从结构化数据生成出来，渲染成多张凭证图片，再把图片和真值转换成 SFT/DPO/GRPO 训练格式，训练 Qwen3-VL，最后用程序评测模型输出的 JSON、字段、审计结论、证据和 bbox 是否正确。

### 0.2 端到端流程

```text
配置与 schema
-> 生成正常报销 case
-> 注入异常并打 risk/audit 标签
-> 按 case 划分 train/val/test
-> 渲染发票、支付截图、报销单、订单图片
-> 记录每个字段的 bbox
-> 构造 SFT / DPO / GRPO 数据
-> LoRA-SFT / DPO / GRPO 训练
-> 批量推理生成 predictions
-> evaluate_all 计算业务指标和 error cases
-> analysis 脚本做错误迁移、归档、High-risk Repair
```

### 0.3 代码目录分工

| 目录 | 对非代码读者的解释 | 主要职责 |
| --- | --- | --- |
| `src/mv_audit/data_gen/` | 造“报销业务真值”的地方 | 生成 case、注入异常、分 split、校验 schema、打风险标签 |
| `src/mv_audit/rendering/` | 把真值画成图片的地方 | 渲染四类凭证图片，记录字段 bbox |
| `src/mv_audit/perturbation/` | 制造视觉/材料难度的地方 | 模糊、噪声、不可读字段、重复凭证 |
| `src/mv_audit/converters/` | 把业务数据转成训练样本的地方 | 生成 SFT JSONL、DPO pairs、GRPO prompts、repair SFT mix |
| `src/mv_audit/training/` | 真正训练模型的地方 | LoRA-SFT、DPO/IPO、GRPO、reward function |
| `src/mv_audit/inference/` | 模型加载和批量预测的地方 | Qwen3-VL 加载、多图输入、adapter 加载、prediction JSONL |
| `src/mv_audit/evaluation/` | 评分裁判 | JSON/schema、字段、审计、证据、bbox、幻觉等指标 |
| `src/mv_audit/analysis/` | 实验后处理和报告 | 错误迁移、ablation 归档、High-risk Repair Pack、repair 归档 |
| `src/mv_audit/utils/` | 通用小工具 | 读写 JSONL/YAML、日志、随机种子、phase00 检查 |
| `scripts/` | 给人运行的入口 | 把上面 Python 模块串成完整 pipeline |
| `configs/` | 实验参数 | 数据规模、模型路径、schema、训练超参数、输出目录 |
| `tests/` | 自动测试 | 当前重点测试 DPO/IPO loss 类型 |
| `data/` | 数据和训练样本产物 | 词典、raw cases、annotations、图片、SFT/DPO/GRPO JSONL、eval sets |
| `outputs/` | 实验运行产物 | checkpoint、prediction、metrics、日志、runtime watcher 状态、临时检查目录 |
| `models/` | 本地基座模型资产 | Qwen3-VL tokenizer/config/processor/权重索引和权重 shard |
| `external/` | 第三方参考代码 | Qwen3-VL 官方示例、finetune 参考实现、qwen-vl-utils、benchmark 脚本 |
| `notebooks/` | 临时探索区 | 当前只有占位文件，未来用于一次性分析或可视化草稿 |

## 1. 目录分层

### 1.1 `src/mv_audit/`

核心 Python 包。所有真正的业务逻辑、训练逻辑、推理逻辑和评测逻辑都在这里。脚本通常只是调用这里的模块。

### 1.2 `scripts/`

命令行入口。文件名开头数字代表 pipeline 阶段，例如 `01_generate_main_cases.sh` 生成 main 数据，`04_train_sft.sh` 启动 SFT，`08_evaluate.sh` 评测 predictions。

### 1.3 `configs/`

所有可调参数的集中地。模型路径、数据路径、训练超参数、LoRA target modules、sample manifest、输出目录都在这里，不应硬写进训练代码。

### 1.4 `tests/`

自动测试。目前主要覆盖 DPO/IPO loss 计算，确保后续改 DPO 训练时不破坏核心 loss 逻辑。

### 1.5 `data/`

数据与训练样本产物层。这里既有 Git 跟踪的小词典，也有 `.gitignore` 忽略的大规模 raw cases、图片、bbox annotations、SFT/DPO/GRPO JSONL 和评测集合。非代码读者可以把它理解为“教材、考卷和答案底稿”。

### 1.6 `outputs/`

实验运行结果层。训练 checkpoint、批量推理 predictions、评测 metrics、日志、归档运行状态和临时检查目录都在这里。这里大多是可再生成或体积较大的产物，通常不直接提交 Git。

### 1.7 `models/`

本地基座模型资产层。当前主要是 `Qwen3-VL-8B-Instruct` 的配置、tokenizer、processor 元数据、权重索引和权重 shard。训练代码和推理代码通过 config 指向这里。

### 1.8 `external/`

第三方参考代码层。当前是 Qwen3-VL 官方仓库代码，主要用于理解官方输入处理、finetune 脚本、demo、benchmark 和 `qwen-vl-utils`；本项目的主流程不直接在这里开发。

### 1.9 `notebooks/`

交互式探索层。当前只有 `.gitkeep` 占位，说明预留了 notebook 目录，但正式 pipeline 仍以 `src/`、`scripts/`、`configs/` 为准。

## 2. 核心 Python 包清单

### 2.1 包入口

#### `src/mv_audit/__init__.py`

- 类型：包入口。
- 负责内容：声明 `mv_audit` 是一个可导入 Python 包。
- 关键函数/类：无。
- 输入：无。
- 输出：无。
- 上游：所有 `python -m mv_audit.xxx` 命令都会依赖包结构。
- 下游：全部子模块。
- 读者提示：这是门牌号，不承载业务逻辑。

### 2.2 数据生成：`src/mv_audit/data_gen/`

#### `src/mv_audit/data_gen/generate_base_cases.py`

- 类型：数据生成。
- 负责内容：生成没有异常或尚未注入异常的基础报销 case，包括人员、商户、金额、日期、订单号、发票号、支付流水号等字段。
- 关键函数/类：`load_dictionaries`、`generate_case`、`generate_cases`、`main`。
- 输入：`configs/data_gen/*.yaml` 中的数据规模和字典路径；`data/mv_audit/dictionaries/` 下的人名、城市、商户、费用类型等字典。
- 输出：`base_cases_*.jsonl`。
- 上游：`scripts/01_generate_base_cases.sh`、`scripts/01_generate_cases.sh`、`scripts/01_generate_main_cases.sh`。
- 下游：`anomaly_injector.py` 读取基础 case 后注入异常。
- 读者提示：这是“造正常报销单”的工厂，还没有真正制造作弊、缺材料或不可读图片。

#### `src/mv_audit/data_gen/anomaly_injector.py`

- 类型：数据生成 / 异常构造。
- 负责内容：按照配置比例给 base cases 注入异常，例如金额不一致、超额报销、日期异常、商户异常、人员异常、订单号异常、缺材料、批内重复、不可读图片。
- 关键函数/类：`inject_anomalies`、`_inject_amount_mismatch`、`_inject_over_reimbursement`、`_inject_missing_document`、`_inject_unreadable_image`。
- 输入：`base_cases_*.jsonl`；异常比例配置。
- 输出：`all_cases_with_anomaly_*.jsonl` 和异常统计 JSON。
- 上游：`scripts/01_inject_anomalies.sh` 或生成 main/debug 数据的组合脚本。
- 下游：`risk_rule_engine.py` 给异常后的 case 计算风险等级和审核建议；`split_builder.py` 做 case-level split。
- 读者提示：这是“故意制造问题样本”的地方，是后续模型能学会审计异常的来源。

#### `src/mv_audit/data_gen/risk_rule_engine.py`

- 类型：业务规则 / 标签生成。
- 负责内容：根据 case 字段和异常类型计算 `risk_level` 与 `audit_result`，例如高风险应 `reject_recommendation`，缺材料应 `missing_info`。
- 关键函数/类：`risk_reasons`、`assign_risk_level`、`assign_audit_result`、`update_case_labels`。
- 输入：已经生成或注入异常的 case。
- 输出：带 `risk_level`、`audit_result`、规则原因的 case。
- 上游：`anomaly_injector.py`、数据构造流程。
- 下游：SFT/DPO/GRPO 数据构造和评测真值。
- 读者提示：这是“审计规则老师”。模型最终学到的风险/审核判断，很多来自这里生成的真值。

#### `src/mv_audit/data_gen/split_builder.py`

- 类型：数据切分。
- 负责内容：按 case 级别划分 train、val、test，不把同一个 case 的多张图片拆到不同 split。
- 关键函数/类：`build_splits`、`_split_targets`、`_prepare_split_cases`。
- 输入：`all_cases_with_anomaly_*.jsonl`；split 比例配置。
- 输出：`train_cases.jsonl`、`val_in_template_cases.jsonl`、`test_clean_cases.jsonl` 等 split 文件。
- 上游：异常注入后的 case。
- 下游：渲染、SFT 数据、sample500 manifest、评测集。
- 读者提示：这是防数据泄漏的关键文件。同一个报销 case 必须整体留在同一个 split。

#### `src/mv_audit/data_gen/case_validator.py`

- 类型：数据校验。
- 负责内容：用 JSON schema 和业务约束检查 case 是否合法，例如金额格式、日期顺序、必需字段、枚举值。
- 关键函数/类：`validate_case`、`validate_cases`、`load_schema`。
- 输入：case JSONL；`configs/schema/case_schema.json`。
- 输出：校验通过或报错。
- 上游：数据生成脚本或人工检查。
- 下游：保证渲染、训练数据构造、评测不会吃到坏 case。
- 读者提示：这是数据工厂的质检员。

#### `src/mv_audit/data_gen/__init__.py`

- 类型：包入口。
- 负责内容：声明 `data_gen` 子包。
- 关键函数/类：无。
- 输入/输出：无。
- 上游/下游：被 Python 导入系统使用。
- 读者提示：无业务逻辑。

### 2.3 图片渲染：`src/mv_audit/rendering/`

#### `src/mv_audit/rendering/layout.py`

- 类型：图片渲染基础设施。
- 负责内容：定义凭证画布、字体加载、文字绘制和位置布局等通用能力。
- 关键函数/类：`VoucherCanvas`、`resolve_font_path`、`load_font`。
- 输入：字体路径、画布大小、文本字段。
- 输出：PIL 图片对象和绘制位置。
- 上游：四类凭证渲染器调用。
- 下游：`render_invoice.py`、`render_payment.py`、`render_reimbursement_form.py`、`render_order.py`。
- 读者提示：这是所有凭证图片共同使用的“画板和画笔”。

#### `src/mv_audit/rendering/bbox_recorder.py`

- 类型：bbox 记录。
- 负责内容：把图片上字段的绝对坐标转成 0-1000 归一化 bbox，并生成字段证据记录。
- 关键函数/类：`ImageSpec`、`normalize_bbox`、`make_bbox_record`、`mark_unreadable`。
- 输入：字段名、字段值、图片大小、绝对 bbox。
- 输出：字段级 bbox record。
- 上游：各凭证渲染器。
- 下游：SFT evidence、评测 evidence/bbox、不可读字段处理。
- 读者提示：这是“证据定位”的根。模型输出的 bbox 评测真值来自这里。

#### `src/mv_audit/rendering/render_invoice.py`

- 类型：图片渲染 / 发票。
- 负责内容：把 case 中的发票字段画成发票图片，并记录发票字段 bbox。
- 关键函数/类：`render_invoice`。
- 输入：case、`ImageSpec`、字体。
- 输出：发票 PNG 和 bbox records。
- 上游：`render_all.py`。
- 下游：图片文件、`field_bboxes_*.jsonl`、SFT/DPO/GRPO 数据。
- 读者提示：只负责发票这一类凭证。

#### `src/mv_audit/rendering/render_payment.py`

- 类型：图片渲染 / 支付截图。
- 负责内容：把支付金额、付款人、收款方、支付日期、支付流水号等字段画成支付截图。
- 关键函数/类：`render_payment`。
- 输入：case、`ImageSpec`、字体。
- 输出：支付截图 PNG 和 bbox records。
- 上游：`render_all.py`。
- 下游：训练和评测数据。
- 读者提示：只负责 payment 图。

#### `src/mv_audit/rendering/render_reimbursement_form.py`

- 类型：图片渲染 / 报销申请单。
- 负责内容：把申请人、报销金额、申请日期、费用类型、订单号等字段画成报销单。
- 关键函数/类：`render_reimbursement_form`。
- 输入：case、`ImageSpec`、字体。
- 输出：报销单 PNG 和 bbox records。
- 上游：`render_all.py`。
- 下游：训练和评测数据。
- 读者提示：报销单是 audit_result 很依赖的凭证之一。

#### `src/mv_audit/rendering/render_order.py`

- 类型：图片渲染 / 订单。
- 负责内容：把订单号、订单金额、下单日期、订单用户、订单商户等字段画成订单截图。
- 关键函数/类：`render_order`。
- 输入：case、`ImageSpec`、字体。
- 输出：订单 PNG 和 bbox records。
- 上游：`render_all.py`。
- 下游：训练和评测数据。
- 读者提示：订单号不一致、高风险拒绝等问题常依赖这个凭证。

#### `src/mv_audit/rendering/render_all.py`

- 类型：图片渲染总入口。
- 负责内容：按 split 批量渲染所有 case 的所有凭证图片，处理缺材料、重复凭证、不可读图片和 robust/unseen split 的视觉增强。
- 关键函数/类：`render_case`、`render_split`、`main`。
- 输入：split case JSONL、图片输出目录、bbox 输出目录、字体和随机种子。
- 输出：`images_main/<split>/*.png`、`annotations_main/field_bboxes_<split>.jsonl`。
- 上游：`scripts/02_render_images.sh`、`scripts/02_render_main_images.sh`。
- 下游：SFT/DPO/GRPO 数据构造、batch inference、bbox 评测。
- 读者提示：这是“把结构化数据变成模型能看的图片”的总开关。

#### `src/mv_audit/rendering/__init__.py`

- 类型：包入口。
- 负责内容：声明 `rendering` 子包。
- 关键函数/类：无。
- 输入/输出：无。
- 读者提示：无业务逻辑。

### 2.4 视觉扰动：`src/mv_audit/perturbation/`

#### `src/mv_audit/perturbation/visual_augment.py`

- 类型：视觉增强。
- 负责内容：轻量调整亮度、对比度、噪声等，让图片不要太“模板完美”。
- 关键函数/类：`apply_light_augment`。
- 输入：PIL 图片、随机种子。
- 输出：增强后的图片。
- 上游：`render_all.py`。
- 下游：robust/unseen 风格的训练或评测图片。
- 读者提示：模拟真实截图的轻微视觉变化。

#### `src/mv_audit/perturbation/robust_augment.py`

- 类型：鲁棒性增强。
- 负责内容：更强的视觉扰动，用于 robust 测试 split。
- 关键函数/类：`apply_robust_augment`。
- 输入：PIL 图片、随机种子。
- 输出：增强后的图片。
- 上游：`render_all.py`。
- 下游：`test_robust` 图片。
- 读者提示：用来测试模型面对画质变化是否还稳。

#### `src/mv_audit/perturbation/unreadable_generator.py`

- 类型：不可读字段模拟。
- 负责内容：遮挡或模糊指定字段，同时把对应 bbox records 标成不可读。
- 关键函数/类：`obscure_fields`。
- 输入：图片、字段 bbox records、字段集合、随机种子。
- 输出：遮挡后的图片和更新后的 records。
- 上游：`render_all.py`。
- 下游：SFT/评测中的 `unreadable_image`、`missing_info` 或人工复核逻辑。
- 读者提示：这是训练模型“不确定时别编”的关键数据来源。

#### `src/mv_audit/perturbation/duplicate_generator.py`

- 类型：重复凭证模拟。
- 负责内容：复制图片和 bbox records，制造 batch 内重复凭证。
- 关键函数/类：`duplicate_image_and_records`。
- 输入：图片、bbox records、复制编号。
- 输出：重复图片和对应 records。
- 上游：`render_all.py`。
- 下游：`duplicate_in_batch` 异常训练和评测。
- 读者提示：让模型学会识别同一 case 内重复提交。

#### `src/mv_audit/perturbation/__init__.py`

- 类型：包入口。
- 负责内容：声明 `perturbation` 子包。
- 读者提示：无业务逻辑。

### 2.5 数据转换：`src/mv_audit/converters/`

#### `src/mv_audit/converters/common.py`

- 类型：训练数据公共构造。
- 负责内容：把 case 和 bbox records 转成模型需要的 prompt、answer、field_extraction、consistency_check、evidence、uncertainty。
- 关键函数/类：`build_prompt`、`build_audit_output`、`build_evidence`、`build_field_extraction`、`make_messages`。
- 输入：case、bbox records、output schema。
- 输出：Evidence-Grounded JSON answer 和 messages。
- 上游：SFT、DPO、GRPO converters 和 batch inference。
- 下游：`build_sft_data.py`、`build_dpo_pairs.py`、`build_dpo_v2_pairs.py`、`build_grpo_prompts.py`、`batch_inference.py`。
- 读者提示：这是训练样本“标准答案”的共同制造器，SFT/DPO/推理真值都绕不开它。

#### `src/mv_audit/converters/build_sft_data.py`

- 类型：SFT 数据构造。
- 负责内容：把 train/val case 和图片 bbox 转为 SFT JSONL，每行包含 `images`、user prompt、assistant JSON answer。
- 关键函数/类：`build_examples`、`_assign_task`、`main`。
- 输入：split case JSONL、`field_bboxes_*.jsonl`、output schema。
- 输出：`data/mv_audit/sft*/train.jsonl`、`val.jsonl` 或 existing-images 版本。
- 上游：`scripts/03_build_train_data.sh`、`scripts/03_build_main_train_data.sh`。
- 下游：`train_sft.py`、few-shot 示例、repair mix。
- 读者提示：这是 SFT 的“教材编写器”。

#### `src/mv_audit/converters/build_dpo_pairs.py`

- 类型：DPO v1 数据构造。
- 负责内容：基于 SFT 正确答案制造 chosen/rejected 偏好对，包括 invalid JSON、风险错、审计错、证据错等 rejected。
- 关键函数/类：`build_pairs`、`_mutate_output`、`_eligible_rejected_types`。
- 输入：MV-Train case、bbox records、output schema。
- 输出：`data/mv_audit/dpo*/pairs_train.jsonl`。
- 上游：DPO v1 数据脚本。
- 下游：`train_dpo.py`。
- 读者提示：这是第一版 DPO 的偏好数据生成器，后来业务失败的重要原因之一就是 pair 设计和业务指标不够对齐。

#### `src/mv_audit/converters/build_dpo_v2_pairs.py`

- 类型：DPO v2 数据构造。
- 负责内容：构造更保守的 Train-only DPO v2 pairs，加入 hard rejected、high-risk miss、protective pair、normal calibration、case-level holdout、train_decode_dev。
- 关键函数/类：`build_dpo_v2`、`_make_rejected`、`_select_pairs`、`_decode_dev_rows`、`_quality_checks`。
- 输入：MV-Train case、bbox records、output schema、existing image 路径。
- 输出：`pairs_train.jsonl`、`pairs_holdout.jsonl`、`train_decode_dev.jsonl`、pair report。
- 上游：`scripts/05_build_dpo_v2_pairs.sh`。
- 下游：DPO v2 training configs、M3v2、loss ablation、High-risk Repair 排除集合。
- 读者提示：这是 DPO v2 的核心数据文件，重点是“不要用测试集修模型”。

#### `src/mv_audit/converters/build_grpo_prompts.py`

- 类型：GRPO 数据构造。
- 负责内容：把 case 转成 GRPO prompt 数据，供 reward function 在线打分。
- 关键函数/类：`build_prompts`、`main`。
- 输入：MV-Train case、bbox records。
- 输出：GRPO prompts JSONL。
- 上游：GRPO 数据脚本。
- 下游：`train_grpo.py`。
- 读者提示：当前 GRPO 只有 smoke 级别，不是正式成功结果。

#### `src/mv_audit/converters/build_high_risk_repair_sft_mix.py`

- 类型：SFT 修复数据构造 / High-risk Repair。
- 负责内容：把 120 条 High-risk Repair Pack 和 120 条 calibration 样本混合成 repair_sft_r1 小规模 SFT 数据。
- 关键函数/类：`build_mix`、`_load_excluded_case_ids`、`_calibration_bucket`。
- 输入：`repair_pack_sft.jsonl`、`data/mv_audit/sft_main/train.jsonl`、DPO holdout、Train decode dev、sample500 case ids。
- 输出：`repair_sft_train_mix.jsonl`、`repair_sft_train_mix_manifest.json`。
- 上游：`scripts/11_run_high_risk_repair_sft_r1_server.sh` 或本地 dry-run。
- 下游：`train_sft.py` 使用 repair mix 训练 `repair_sft_r1`。
- 读者提示：这是当前下一步小闭环的“教材拼装器”，特别强调防止评测泄漏。

#### `src/mv_audit/converters/__init__.py`

- 类型：包入口。
- 负责内容：声明 `converters` 子包。
- 读者提示：无业务逻辑。

### 2.6 训练：`src/mv_audit/training/`

#### `src/mv_audit/training/train_sft.py`

- 类型：SFT 训练。
- 负责内容：读取 SFT JSONL，组装多图 chat messages，使用 Qwen3-VL processor 处理图文输入，配置 LoRA，用 Hugging Face `Trainer` 训练 adapter。
- 关键函数/类：`SFTExample`、`SFTDataset`、`DataCollatorForQwenVLSFT`、`_read_examples`、`_conversation`、`_train`、`_dry_run`。
- 输入：SFT train/val JSONL、Qwen3-VL 基座模型、LoRA 配置。
- 输出：SFT adapter checkpoint，例如 `outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/` 或 `qwen3vl_8b_high_risk_repair_r1/`。
- 上游：`scripts/04_train_sft.sh`。
- 下游：`batch_inference.py` 加载 SFT adapter 做 M2 或 `repair_sft_r1` 推理。
- 读者提示：这是 SFT 的核心训练代码。里面的 collator 负责“哪些 token 算 loss”，也就是只让模型学习 assistant answer，不惩罚用户 prompt。

#### `src/mv_audit/training/train_dpo.py`

- 类型：DPO/IPO 训练。
- 负责内容：读取 DPO pairs，加载 SFT policy adapter，计算 chosen/rejected logprob，支持 DPO loss、IPO loss、样本权重、SFT/NLL 辅助 loss、holdout pair 监控。
- 关键函数/类：`DPOExample`、`_read_examples`、`_score_preference_examples`、`_preference_loss_values`、`_evaluate_preference_logits`、`_load_sft_policy`、`_train`。
- 输入：DPO pair JSONL、SFT adapter、DPO/IPO 训练配置、reward function。
- 输出：DPO adapter、training history、holdout history、reward audit。
- 上游：`scripts/05_train_dpo.sh`、`scripts/05_train_dpo_v2.sh`、loss ablation server 脚本。
- 下游：`batch_inference.py` 加载 DPO adapter 做 M3/M3v2 推理；analysis 归档训练历史。
- 读者提示：这是 DPO 实验真正“更新模型偏好”的地方。项目失败诊断说明：这里 loss 下降不等于业务指标成功。

#### `src/mv_audit/training/train_grpo.py`

- 类型：GRPO 训练。
- 负责内容：读取 GRPO prompts，生成多个候选回答，用 reward function 评分并做强化学习式更新。
- 关键函数/类：`GRPOExample`、`_read_examples`、`_train`。
- 输入：GRPO prompt JSONL、模型/adapter、reward function。
- 输出：GRPO checkpoint 和训练日志。
- 上游：`scripts/06_train_grpo.sh`。
- 下游：理论上用于 M4 推理，但当前只有 smoke 证据。
- 读者提示：这是预留的 M4 路线，目前不应当当作正式实验结果。

#### `src/mv_audit/training/reward_function.py`

- 类型：Reward / 业务打分。
- 负责内容：把模型输出和 ground truth 对比，给 JSON/schema、字段、审计、证据、bbox、幻觉、高风险漏检等维度打 reward。
- 关键函数/类：`score_output`、`reward_for_grpo`、`normalize_group_rewards`、`summarize_reward_outputs`。
- 输入：raw output 或 parsed JSON、ground truth、output schema、image items。
- 输出：reward 分数和明细。
- 上游：`train_dpo.py` 做 reward audit；`train_grpo.py` 做在线 reward；`scripts/test_reward_function.py` 测试。
- 下游：DPO/GRPO 训练诊断。
- 读者提示：这是“模型回答好不好”的业务裁判之一，但最终仍要看 evaluate_all 的真实业务指标。

#### `src/mv_audit/training/__init__.py`

- 类型：包入口。
- 负责内容：声明 `training` 子包。
- 读者提示：无业务逻辑。

### 2.7 推理：`src/mv_audit/inference/`

#### `src/mv_audit/inference/qwen3vl_common.py`

- 类型：模型加载 / 生成公共逻辑。
- 负责内容：加载 Qwen3-VL 模型和 processor，处理图片 URI，构建图文输入，移动 tensor 到设备，生成文本，记录 smoke test 日志。
- 关键函数/类：`resolve_model_path`、`load_qwen3vl_model_and_processor`、`process_messages`、`move_inputs_to_model`、`generate_text`、`run_generation`。
- 输入：模型配置、图片路径、messages。
- 输出：模型 raw text、运行快照、日志。
- 上游：smoke test、batch inference。
- 下游：`batch_inference.py`、`qwen3vl_smoke_test.py`、`qwen3vl_multi_image_test.py`。
- 读者提示：这是和 Qwen3-VL 真正打交道的底层工具。

#### `src/mv_audit/inference/batch_inference.py`

- 类型：批量推理。
- 负责内容：读取测试 split 或 train_decode_dev，构造模型输入，按 model_id 加载对应 SFT/DPO adapter，逐 case 生成 raw_output，支持断点续跑。
- 关键函数/类：`build_eval_rows`、`_load_model_for_inference`、`run_inference`、`_dry_run`。
- 输入：训练/推理配置、raw cases 或 train_decode_dev、sample manifest、adapter。
- 输出：predictions JSONL 和 ground truth JSONL。
- 上游：`scripts/07_run_inference.sh`、`scripts/07_run_phase08_m3v2_train_decode_dev.sh`、sample500 脚本。
- 下游：`evaluate_all.py` 计算 metrics。
- 读者提示：这是 M0/M1/M2/M3/M3v2/repair_sft_r1 批量预测的统一入口。

#### `src/mv_audit/inference/qwen3vl_smoke_test.py`

- 类型：单图 smoke test。
- 负责内容：用一张图片测试 Qwen3-VL 能否加载和生成。
- 关键函数/类：`build_messages`、`main`。
- 输入：模型配置、单张图片。
- 输出：smoke test log。
- 上游：`scripts/00_prepare_env.sh` 后人工运行。
- 下游：验证环境是否能跑模型。
- 读者提示：只验证模型环境，不代表业务效果。

#### `src/mv_audit/inference/qwen3vl_multi_image_test.py`

- 类型：多图 smoke test。
- 负责内容：用多张图片测试 Qwen3-VL 多图输入是否可用。
- 关键函数/类：`build_messages`、`main`。
- 输入：模型配置、多张图片。
- 输出：smoke test log。
- 上游：环境准备。
- 下游：证明多凭证输入技术上可行。
- 读者提示：比单图 smoke 更接近本项目业务形态，但仍不是评测。

#### `src/mv_audit/inference/__init__.py`

- 类型：包入口。
- 负责内容：声明 `inference` 子包。
- 读者提示：无业务逻辑。

### 2.8 评测：`src/mv_audit/evaluation/`

#### `src/mv_audit/evaluation/evaluate_all.py`

- 类型：总评测入口。
- 负责内容：读取 ground truth 和 predictions，解析 raw_output，校验 schema，计算字段、consistency、risk/audit、evidence、bbox、hallucination 等指标，并输出 error cases。
- 关键函数/类：`evaluate`、`_extract_ground_truth`、`_schema_ok`、`main`。
- 输入：ground truth JSONL、prediction JSONL、`output_schema.json`。
- 输出：metrics JSON、errors JSONL。
- 上游：`scripts/08_evaluate.sh`。
- 下游：报告、图表、DPO error migration、High-risk Repair diagnosis。
- 读者提示：这是项目最重要的“裁判总入口”，比训练 loss 更可信。

#### `src/mv_audit/evaluation/json_parser.py`

- 类型：输出解析。
- 负责内容：从模型 raw text 中提取 JSON 候选，判断 JSON Validity。
- 关键函数/类：`ParseResult`、`parse_json_output`。
- 输入：模型 raw_output 字符串。
- 输出：解析结果、parse error、parsed JSON。
- 上游：`evaluate_all.py`、DPO error migration、reward function。
- 下游：schema 评测和业务指标。
- 读者提示：模型可能输出多余文字，这里负责把 JSON 抠出来。

#### `src/mv_audit/evaluation/audit_metrics.py`

- 类型：业务审计指标。
- 负责内容：计算 risk macro-F1、Audit Accuracy、High-risk Miss Rate、False Manual Review Rate。
- 关键函数/类：`macro_f1`、`audit_accuracy`、`high_risk_miss_rate`、`false_manual_review_rate`。
- 输入：truth outputs 和 pred outputs。
- 输出：审计类指标。
- 上游：`evaluate_all.py`。
- 下游：README、报告、是否继续实验的 gate。
- 读者提示：High-risk Miss Rate 就在这里定义，是当前最关心的指标。

#### `src/mv_audit/evaluation/field_metrics.py`

- 类型：字段抽取指标。
- 负责内容：计算 `field_extraction` 中字段值 exact match 数。
- 关键函数/类：`field_exact_counts`。
- 输入：truth/pred JSON。
- 输出：字段正确数和总数。
- 上游：`evaluate_all.py`。
- 下游：Field EM。
- 读者提示：评估模型有没有读对金额、商户、人员、日期等字段。

#### `src/mv_audit/evaluation/consistency_metrics.py`

- 类型：一致性指标。
- 负责内容：计算 `consistency_check` 布尔项是否和真值一致。
- 关键函数/类：`consistency_exact_counts`。
- 输入：truth/pred JSON。
- 输出：一致性判断正确数和总数。
- 上游：`evaluate_all.py`。
- 下游：总体 schema/business 评分。
- 读者提示：评估模型有没有判断多张凭证之间是否一致。

#### `src/mv_audit/evaluation/evidence_metrics.py`

- 类型：证据支持指标。
- 负责内容：比较 evidence 条目里的 source image、doc type、field、value 和 bbox 是否支持真值。
- 关键函数/类：`evidence_counts`。
- 输入：truth/pred JSON。
- 输出：support count、bbox count 等。
- 上游：`evaluate_all.py`、DPO error migration。
- 下游：Evidence Support Rate、BBox Accuracy。
- 读者提示：评估模型“结论有没有证据链”。

#### `src/mv_audit/evaluation/bbox_evaluator.py`

- 类型：bbox 几何评测。
- 负责内容：计算预测 bbox 和真值 bbox 是否严格或宽松匹配。
- 关键函数/类：`bbox_iou`、`center_inside`、`strict_match`、`relaxed_match`。
- 输入：预测 bbox、真值 bbox。
- 输出：匹配布尔值。
- 上游：`evidence_metrics.py`。
- 下游：Evidence BBox Accuracy。
- 读者提示：判断模型框的位置是不是合理。

#### `src/mv_audit/evaluation/hallucination_metrics.py`

- 类型：幻觉检测。
- 负责内容：检查模型 evidence 是否引用不存在图片、缺失字段或不可信来源。
- 关键函数/类：`hallucination_count`。
- 输入：truth/pred JSON、image items。
- 输出：幻觉数量和明细。
- 上游：`evaluate_all.py`、reward function。
- 下游：Hallucination Rate。
- 读者提示：防止模型编造证据。

#### `src/mv_audit/evaluation/__init__.py`

- 类型：包入口。
- 负责内容：声明 `evaluation` 子包。
- 读者提示：无业务逻辑。

### 2.9 实验分析与归档：`src/mv_audit/analysis/`

#### `src/mv_audit/analysis/dpo_error_migration.py`

- 类型：错误迁移分析。
- 负责内容：比较 baseline 和 candidate 每个 case 是否从对变错、错变对、都对、都错，同时统计 issue delta 和 high-risk miss 变化。
- 关键函数/类：`analyze`、`_case_issues`、`_transition`、`main`。
- 输入：ground truth、baseline predictions、candidate predictions、output schema。
- 输出：case transition CSV、transition summary JSON。
- 上游：`scripts/09_analyze_dpo_error_migration.sh`。
- 下游：DPO 失败诊断、README 报告。
- 读者提示：这是解释“DPO 到底修了什么、破坏了什么”的核心工具。

#### `src/mv_audit/analysis/archive_loss_ablation.py`

- 类型：实验归档。
- 负责内容：汇总 DPO v2 loss ablation 的训练历史、decode metrics、sample500 metrics、图表、manifest，并打包 archive。
- 关键函数/类：`archive`、`_training_summary`、`_decode_summary`、`_copy_artifacts`、`_write_report`。
- 输入：runtime、metrics、logs、predictions、configs。
- 输出：`docs/experiments/phase08_loss_ablation_*` 归档目录和 tar。
- 上游：`scripts/10_run_dpo_v2_ablation_5gpu_server.sh`。
- 下游：本地实验报告、README、High-risk 诊断。
- 读者提示：这是把服务器实验结果整理成可提交文档的工具。

#### `src/mv_audit/analysis/high_risk_repair_pack.py`

- 类型：High-risk Miss 诊断 / 数据修复包构造。
- 负责内容：汇总 M2/M3/M3v2/two-candidate error cases，归因 schema/业务/证据问题，从 MV-Train 挖相似 high-risk non-pass repair cases。
- 关键函数/类：`build`、`_diagnose_errors`、`_build_repair_pack`、`_load_excluded_case_ids`。
- 输入：phase07/phase08 归档、SFT train、raw cases、DPO holdout/decode/sample500 case ids。
- 输出：诊断报告、candidate cases、repair pack、leakage check。
- 上游：人工执行或后续复现实验。
- 下游：`build_high_risk_repair_sft_mix.py` 和 `repair_sft_r1`。
- 读者提示：这是当前从 DPO 失败转向小规模 SFT 修复的桥。

#### `src/mv_audit/analysis/archive_high_risk_repair_sft.py`

- 类型：repair_sft_r1 归档。
- 负责内容：收集 repair_sft_r1 的配置、日志、metrics、errors、prediction summary、manifest，并生成 README 和 tar。
- 关键函数/类：`archive`、`_comparison_rows`、`_prediction_summary`、`_write_readme`。
- 输入：repair_sft_r1 runtime、eval reports、predictions、repair pack manifest。
- 输出：`docs/experiments/phase08_high_risk_repair_sft_r1_$RUN_ID/` 和 tar。
- 上游：`scripts/11_run_high_risk_repair_sft_r1_server.sh`。
- 下游：本地归档、README、是否进入 sample500 的决策。
- 读者提示：训练跑完后，它负责把结果整理成能看的证据包。

#### `src/mv_audit/analysis/__init__.py`

- 类型：包入口。
- 负责内容：声明 `analysis` 子包。
- 读者提示：无业务逻辑。

### 2.10 通用工具：`src/mv_audit/utils/`

#### `src/mv_audit/utils/io_utils.py`

- 类型：I/O 工具。
- 负责内容：统一读写 JSONL、YAML 和创建目录。
- 关键函数/类：`ensure_dir`、`iter_jsonl`、`read_jsonl`、`write_jsonl`、`read_yaml`、`write_yaml`。
- 输入：文件路径。
- 输出：Python 对象或磁盘文件。
- 上游：几乎所有模块。
- 下游：几乎所有模块。
- 读者提示：这是项目的文件读写水管。

#### `src/mv_audit/utils/config_utils.py`

- 类型：配置和随机性工具。
- 负责内容：读取 YAML 配置，设置随机种子。
- 关键函数/类：`load_config`、`set_random_seed`。
- 输入：config YAML 路径、seed。
- 输出：配置字典、稳定随机行为。
- 上游：训练、推理、数据生成。
- 下游：所有需要配置和 seed 的模块。
- 读者提示：保证同一配置下尽量可复现。

#### `src/mv_audit/utils/logging_utils.py`

- 类型：日志工具。
- 负责内容：统一设置 console/file logging。
- 关键函数/类：`setup_logging`、`get_logger`。
- 输入：日志路径和 logger 名。
- 输出：logger。
- 上游：早期 pipeline 和工具脚本。
- 下游：运行日志。
- 读者提示：辅助调试，不决定模型效果。

#### `src/mv_audit/utils/phase00_check.py`

- 类型：环境/项目骨架检查。
- 负责内容：检查必需目录和基础工具函数是否存在。
- 关键函数/类：`check_required_dirs`、`check_utilities`、`main`。
- 输入：当前项目目录。
- 输出：检查通过或异常。
- 上游：Phase 00 验收。
- 下游：早期工程搭建确认。
- 读者提示：这是项目最早期的“体检脚本”。

#### `src/mv_audit/utils/__init__.py`

- 类型：工具包导出。
- 负责内容：把常用工具函数集中导出，方便其他模块 `from mv_audit.utils import read_jsonl`。
- 关键函数/类：导出 `ensure_dir`、`iter_jsonl`、`read_jsonl`、`write_jsonl` 等。
- 输入/输出：无直接业务 I/O。
- 上游：所有使用工具函数的模块。
- 下游：所有子系统。
- 读者提示：这是工具函数的总插座。

## 3. SFT 相关代码细分

### 3.1 SFT 数据从哪里来

| 环节 | 文件 | 做什么 |
| --- | --- | --- |
| 生成业务 case | `generate_base_cases.py`、`anomaly_injector.py`、`risk_rule_engine.py` | 得到带风险和审核标签的报销真值 |
| 渲染图片和 bbox | `render_all.py` + 四类 `render_*.py` | 把 case 画成图片并记录字段位置 |
| 组装标准答案 | `converters/common.py` | 生成 Evidence-Grounded JSON answer |
| 生成 SFT JSONL | `build_sft_data.py` | 写出 train/val SFT 样本 |
| High-risk 修复 mix | `build_high_risk_repair_sft_mix.py` | 写出 240 条 repair_sft_r1 小训练集 |

### 3.2 SFT 训练内部细节

| SFT 子步骤 | 文件/函数 | 解释 |
| --- | --- | --- |
| 读取样本 | `train_sft.py::_read_examples` | 从 SFT JSONL 读出 `case_id/images/user_prompt/answer` |
| 校验样本 | `train_sft.py::_validate_examples` | 检查图片存在、answer 是 JSON、case_id 对得上 |
| 组装多图对话 | `train_sft.py::_conversation` | 把每张凭证变成 Qwen-VL message 中的 image block |
| 构造 batch | `DataCollatorForQwenVLSFT` | 用 processor 编码图文，mask 掉 prompt token，只训练 assistant answer |
| 加载模型 | `train_sft.py::_train` + `qwen3vl_common.py` | 加载 Qwen3-VL 基座模型 |
| 加 LoRA | `train_sft.py::_train` | 使用 `LoraConfig` 包装模型，只训练 LoRA 参数 |
| 训练循环 | Hugging Face `Trainer` | 根据 config 中 batch、epoch、lr、save/eval steps 训练 |
| 保存 adapter | `trainer.save_model(output_dir)` | 输出 M2 或 repair_sft_r1 adapter |

### 3.3 SFT 和模型编号的关系

| 模型编号 | 代码关系 |
| --- | --- |
| M2 | `train_sft.py` 训练出的 LoRA adapter + `batch_inference.py` 用 model_id `m2_sft` 加载 `sft_adapter_dir` |
| repair_sft_r1 | `build_high_risk_repair_sft_mix.py` 构造小训练集，`train_sft.py` 训练，`batch_inference.py` 用 model_id `repair_sft_r1` 加载 `sft_adapter_dir` |

## 4. DPO/GRPO 相关代码细分

### 4.1 DPO v1

| 环节 | 文件 | 说明 |
| --- | --- | --- |
| pair 构造 | `build_dpo_pairs.py` | 从正确 answer 生成 rejected answer |
| DPO 训练 | `train_dpo.py` | 计算 chosen/rejected logprob 和 DPO loss |
| 评测 | `batch_inference.py` + `evaluate_all.py` | 加载 DPO adapter 做 M3 sample500 |
| 失败诊断 | `dpo_error_migration.py` | 发现 M3 新增大量 audit_mismatch |

### 4.2 DPO v2 / ablation

| 环节 | 文件 | 说明 |
| --- | --- | --- |
| v2 pair 构造 | `build_dpo_v2_pairs.py` | 增加 hard rejected、high-risk miss、protective、normal calibration |
| loss 类型 | `train_dpo.py::_preference_loss_values` | 支持 `dpo` 和 `ipo` |
| SFT 辅助项 | `train_dpo.py` | 用 `lambda_sft * chosen NLL` 保护能力 |
| holdout 监控 | `train_dpo.py::_evaluate_preference_logits` | 监控 pair accuracy 和 margin |
| loss ablation | `scripts/10_run_dpo_v2_ablation_5gpu_server.sh` | 跑多个 DPO/IPO/AuxDPO 候选 |
| 归档 | `archive_loss_ablation.py` | 汇总训练/评测/日志 |

### 4.3 GRPO

| 环节 | 文件 | 说明 |
| --- | --- | --- |
| prompt 构造 | `build_grpo_prompts.py` | 生成 GRPO 输入 |
| reward | `reward_function.py` | 用业务指标计算 reward |
| 训练 | `train_grpo.py` | 预留强化训练入口 |
| 当前状态 | README / 实验报告 | 只有 smoke 级产物，不是正式 M4 |

## 5. 推理和评测链路

### 5.1 推理链路

```text
config
-> batch_inference.build_eval_rows
-> 按 split 读取 raw cases / sample manifest / train_decode_dev
-> converters.common.build_prompt 或直接使用 train_decode_dev prompt
-> qwen3vl_common 加载模型
-> 按 model_id 加载 SFT/DPO adapter
-> generate_text
-> 写 predictions JSONL
-> 写 ground truth JSONL
```

关键点：

- `m0_zero_shot` 不加载 adapter。
- `m1_few_shot` 使用 SFT train 中的少量 few-shot 示例，但不训练。
- `m2_sft` 加载 `sft_adapter_dir`。
- `m3_dpo`、`m3v2_dpo` 加载 `dpo_adapter_dir`。
- `repair_sft_r1` 加载 `sft_adapter_dir`，输出到独立 repair 路径。

### 5.2 评测链路

```text
ground_truth JSONL + predictions JSONL
-> json_parser 解析 raw_output
-> schema 校验
-> field_metrics
-> consistency_metrics
-> audit_metrics
-> evidence_metrics / bbox_evaluator
-> hallucination_metrics
-> metrics JSON + errors JSONL
```

核心指标位置：

| 指标 | 文件 |
| --- | --- |
| JSON Validity / Schema Compliance | `evaluate_all.py` + `json_parser.py` |
| Field EM | `field_metrics.py` |
| Audit Accuracy / High-risk Miss Rate | `audit_metrics.py` |
| Evidence Support Rate | `evidence_metrics.py` |
| BBox Accuracy | `bbox_evaluator.py` |
| Hallucination Rate | `hallucination_metrics.py` |

## 6. 脚本入口清单

### 6.1 环境和模型

#### `scripts/00_prepare_env.sh`

- 类型：脚本 / 环境准备。
- 负责内容：安装或检查项目依赖。
- 关键函数/类：无。
- 输入：当前 Linux/服务器环境。
- 输出：可运行 Python 包和依赖。
- 上游：人工或服务器初始化。
- 下游：所有后续脚本。
- 读者提示：先把机器准备好。

#### `scripts/00_download_qwen3vl.sh`

- 类型：脚本 / 模型下载。
- 负责内容：下载 Qwen3-VL-8B-Instruct 到本地模型目录，可切换 ModelScope。
- 输入：网络、模型源、`USE_MODELSCOPE` 环境变量。
- 输出：`models/Qwen3-VL-8B-Instruct/`。
- 上游：人工运行。
- 下游：SFT/DPO/GRPO/推理。
- 读者提示：没有模型权重，训练和推理都跑不了。

### 6.2 数据生成和渲染

#### `scripts/01_generate_base_cases.sh`

- 类型：脚本 / 数据生成。
- 负责内容：只生成基础 case，不注入异常、不 split、不渲染。
- 上游：debug 数据准备。
- 下游：`01_inject_anomalies.sh`。
- 读者提示：用于分步调试。

#### `scripts/01_inject_anomalies.sh`

- 类型：脚本 / 异常注入。
- 负责内容：调用 `anomaly_injector.py`。
- 上游：base cases。
- 下游：`01_split_cases.sh`。
- 读者提示：制造异常标签。

#### `scripts/01_split_cases.sh`

- 类型：脚本 / split。
- 负责内容：调用 `split_builder.py`。
- 上游：注入异常后的 case。
- 下游：渲染和训练数据构造。
- 读者提示：保证 case-level 防泄漏。

#### `scripts/01_generate_cases.sh`

- 类型：脚本 / debug 数据总入口。
- 负责内容：串联 base case、异常注入、split。
- 上游：`configs/data_gen/debug.yaml`。
- 下游：`02_render_images.sh`。
- 读者提示：小规模本地调试用。

#### `scripts/01_generate_main_cases.sh`

- 类型：脚本 / main 数据总入口。
- 负责内容：生成主实验规模 case、异常和 split。
- 上游：`configs/data_gen/main.yaml`。
- 下游：`02_render_main_images.sh`。
- 读者提示：正式数据准备入口。

#### `scripts/02_render_images.sh`

- 类型：脚本 / debug 图片渲染。
- 负责内容：调用 `render_all.py` 渲染 debug split。
- 输出：debug 图片和 bbox records。
- 下游：debug SFT/DPO/评测。

#### `scripts/02_render_main_images.sh`

- 类型：脚本 / main 图片渲染。
- 负责内容：调用 `render_all.py` 渲染 main split。
- 输出：`images_main`、`annotations_main`。
- 下游：主 SFT/DPO/评测。

#### `scripts/visualize_bbox.py`

- 类型：脚本 / bbox 可视化。
- 负责内容：抽样把 bbox 画到图片上，人工检查证据位置是否正确。
- 关键函数/类：`visualize`。
- 输入：bbox records、图片目录。
- 输出：bbox 可视化 PNG。
- 上游：渲染完成后。
- 下游：人工质检。

### 6.3 训练数据和训练

#### `scripts/03_build_train_data.sh`

- 类型：脚本 / debug 训练数据。
- 负责内容：构造 debug SFT、DPO、GRPO 数据。
- 上游：debug images/bbox/cases。
- 下游：debug train/evaluate。

#### `scripts/03_build_main_train_data.sh`

- 类型：脚本 / main SFT 数据。
- 负责内容：构造 main SFT train/val JSONL。
- 上游：main images/bbox/cases。
- 下游：`04_train_sft.sh`。

#### `scripts/04_train_sft.sh`

- 类型：脚本 / SFT 训练入口。
- 负责内容：调用 `python -m mv_audit.training.train_sft`，支持 `DRY_RUN=1` 和 `MAX_SAMPLES`。
- 输入：`CONFIG` 环境变量指定 SFT 配置。
- 输出：SFT adapter。
- 下游：M2 推理、DPO 初始 adapter、repair_sft_r1。

#### `scripts/05_train_dpo.sh`

- 类型：脚本 / DPO v1 训练入口。
- 负责内容：调用 `train_dpo.py` 使用 DPO v1 配置。
- 下游：M3。

#### `scripts/05_build_dpo_v2_pairs.sh`

- 类型：脚本 / DPO v2 数据构造。
- 负责内容：调用 `build_dpo_v2_pairs.py`。
- 输出：DPO v2 train/holdout/decode-dev。
- 下游：`05_train_dpo_v2.sh`。

#### `scripts/05_train_dpo_v2.sh`

- 类型：脚本 / DPO v2 训练入口。
- 负责内容：调用 `train_dpo.py` 使用 DPO v2/AuxDPO/IPO 配置。
- 下游：M3v2 和 loss ablation。

#### `scripts/05_run_dpo_v2_loss_ablation.sh`

- 类型：脚本 / DPO v2 loss ablation 本地入口。
- 负责内容：按模式运行 DPO/IPO/AuxDPO 候选 dry-run 或小训练。
- 下游：候选比较。

#### `scripts/06_train_grpo.sh`

- 类型：脚本 / GRPO 训练入口。
- 负责内容：调用 `train_grpo.py`。
- 当前边界：只有 smoke 级别证据，不作为正式 M4。

### 6.4 推理、评测和 sample500

#### `scripts/07_run_inference.sh`

- 类型：脚本 / 通用推理入口。
- 负责内容：按 split 和 model_id 调用 `batch_inference.py`，支持 resume。
- 下游：`08_evaluate.sh`。

#### `scripts/07_run_phase07_sample500.sh`

- 类型：脚本 / Phase07 sample500。
- 负责内容：运行 M0/M1/M2 四个 split 的 sample500 推理与评测流程。
- 输出：Phase07 predictions 和 eval reports。

#### `scripts/build_phase07_sample_manifest.py`

- 类型：脚本 / sample500 manifest。
- 负责内容：按 anomaly 分布抽样每个 split 的 500 条 case_id。
- 关键函数/类：`build_split_manifest`、`_quota_for_split`。
- 输出：`data/mv_audit/eval_sets_phase07_sample500/manifests/`。
- 读者提示：保证 sample500 不是随便抽的，而是分布受控。

#### `scripts/07_run_phase08_m3v2_sample500.sh`

- 类型：脚本 / M3v2 sample500 推理。
- 负责内容：调用 `batch_inference.py` 跑 Phase08 M3v2 sample500。
- 下游：`08_evaluate.sh`。

#### `scripts/07_run_phase08_m3v2_train_decode_dev.sh`

- 类型：脚本 / Train decode dev。
- 负责内容：只跑 Train-only decode dev split，并立即评测。
- 下游：DPO v2 ablation、repair_sft_r1。

#### `scripts/08_evaluate.sh`

- 类型：脚本 / 评测入口。
- 负责内容：循环 model_id 和 split，调用 `evaluate_all.py`，汇总 `metrics_summary.csv`。
- 输入：GROUND_TRUTH_DIR、PREDICTIONS_DIR、REPORT_DIR、MODELS、SPLITS。
- 输出：metrics JSON、errors JSONL、summary CSV。
- 读者提示：这是实验结果表的生成器。

#### `scripts/make_fake_predictions.py`

- 类型：脚本 / evaluator 测试辅助。
- 负责内容：根据 ground truth 生成 perfect/broken fake predictions，验证 evaluator 行为。
- 读者提示：fake prediction 只能测评测器，不能当真实模型结果。

### 6.5 分析、归档和服务器编排

#### `scripts/09_analyze_dpo_error_migration.sh`

- 类型：脚本 / DPO 错误迁移。
- 负责内容：调用 `dpo_error_migration.py` 分析 M2/M3 或 M2/M3v2 逐 case 变化。
- 下游：DPO 失败报告。

#### `scripts/10_run_dpo_v2_ablation_5gpu_server.sh`

- 类型：脚本 / 服务器 DPO v2 ablation 编排。
- 负责内容：dry-run、训练多个 DPO/IPO/AuxDPO 候选、Train decode dev、可选 sample500、错误迁移、归档。
- 关键函数：`run_train`、`run_decode_dev`、`select_variant`、`run_sample500`、`package_archive`。
- 输入：DPO v2 configs、服务器 GPU、runtime 环境变量。
- 输出：runtime logs、metrics、archive tar、`READY_TO_ARCHIVE`。
- 读者提示：这是最复杂的服务器实验总控脚本。

#### `scripts/10_watch_and_archive_dpo_v2_ablation.ps1`

- 类型：PowerShell / 本地 watcher。
- 负责内容：轮询远端 `READY_TO_ARCHIVE`，拉取归档、校验 manifest、更新 README 后可关服务器。
- 输入：SSH 目标、本地仓库、远端 run root。
- 输出：本地归档和 watcher log。
- 读者提示：这是为了长时间服务器任务无人值守而写的自动拉取/关机工具。

#### `scripts/11_run_high_risk_repair_sft_r1_server.sh`

- 类型：脚本 / repair_sft_r1 小闭环。
- 负责内容：构建 240 条 repair SFT mix、dry-run、训练 repair SFT、只跑 Train decode dev、评测、归档、写 `READY_TO_ARCHIVE`。
- 输入：`high_risk_repair_sft_r1_qwen3vl_8b_server.yaml`、High-risk Repair Pack、Qwen3-VL 模型。
- 输出：repair_sft_r1 adapter、predictions、eval reports、archive。
- 读者提示：这是当前下一步最推荐运行的服务器脚本。

#### `scripts/run_debug_pipeline.sh`

- 类型：脚本 / Linux debug 全流程。
- 负责内容：串联 debug 数据生成、渲染、训练数据、fake prediction、评测。
- 读者提示：用于快速检查工程链路，不训练大模型。

#### `scripts/run_debug_pipeline.ps1`

- 类型：PowerShell / Windows debug 全流程。
- 负责内容：Windows 版本 debug pipeline。
- 读者提示：本地 Windows 开发验证用。

#### `scripts/test_reward_function.py`

- 类型：脚本 / reward function 测试。
- 负责内容：手写多个输出场景测试 reward，例如 perfect answer、invalid JSON、高风险 pass、缺材料 pass、错误证据来源。
- 下游：确认 GRPO/DPO reward 逻辑合理。

### 6.6 测试文件

#### `tests/test_dpo_loss_types.py`

- 类型：测试 / DPO loss。
- 负责内容：用 fake torch tensor 测试 DPO loss、IPO loss 和未知 loss type 报错。
- 关键函数/类：`FakeTensor`、`test_dpo_preference_loss_matches_logsigmoid`、`test_ipo_preference_loss_targets_finite_margin`。
- 输入：`train_dpo.py::_preference_loss_values`。
- 输出：pytest 通过或失败。
- 上游：本地 CI/手动测试。
- 下游：保护 DPO/IPO loss 不被误改。

## 7. 配置文件清单

### 7.1 数据生成配置

#### `configs/data_gen/debug.yaml`

- 类型：配置 / debug 数据。
- 负责内容：小规模数据生成参数，用于本地快速验证 pipeline。
- 上游：`scripts/01_generate_cases.sh`。
- 下游：debug raw cases、debug images、debug train data。

#### `configs/data_gen/main.yaml`

- 类型：配置 / main 数据。
- 负责内容：主实验数据规模、异常比例、split 规模、模板等参数。
- 上游：`scripts/01_generate_main_cases.sh`。
- 下游：主实验 raw cases 和后续所有训练/评测数据。

### 7.2 模型配置

#### `configs/model/qwen3vl_8b.yaml`

- 类型：配置 / 模型加载。
- 负责内容：Qwen3-VL-8B-Instruct 的模型名、本地模型目录、dtype、device_map、trust_remote_code 等。
- 上游：smoke test、训练和推理模块。
- 下游：`qwen3vl_common.py`。

### 7.3 Schema 配置

#### `configs/schema/case_schema.json`

- 类型：配置 / 输入 case schema。
- 负责内容：定义数据生成出的报销 case 应有哪些字段、类型和枚举。
- 上游：`case_validator.py`。
- 下游：数据质量保障。

#### `configs/schema/output_schema.json`

- 类型：配置 / 模型输出 schema。
- 负责内容：定义模型最终必须输出的 Evidence-Grounded JSON 顶层字段和内部结构。
- 上游：SFT/DPO/GRPO 数据构造、reward、evaluate_all。
- 下游：Schema Compliance、JSON contract。
- 读者提示：这是模型输出格式的铁律，DPO/two-candidate 中很多错误都和 schema 不合规有关。

### 7.4 训练配置

#### `configs/train/sft_lora_qwen3vl_8b.yaml`

- 类型：配置 / 通用 SFT。
- 负责内容：本地默认 SFT 配置，指向 `sft_main/train.jsonl` 和 `val.jsonl`。
- 对应模型：M2 的基础版本。

#### `configs/train/sft_lora_qwen3vl_8b_phase07_server.yaml`

- 类型：配置 / Phase07 服务器 SFT。
- 负责内容：服务器使用 existing-images 子集训练 M2。
- 对应模型：M2。

#### `configs/train/sft_lora_qwen3vl_8b_phase07_sample500_server.yaml`

- 类型：配置 / Phase07 sample500 推理。
- 负责内容：指定 M2 adapter、sample500 manifest、推理输出目录。
- 对应模型：M2 sample500。

#### `configs/train/dpo_qwen3vl_8b.yaml`

- 类型：配置 / DPO v1。
- 负责内容：DPO v1 的 pair 数据、SFT adapter、DPO 输出目录、训练超参数。
- 对应模型：M3。

#### `configs/train/phase08_m3_sample500_server.yaml`

- 类型：配置 / M3 sample500。
- 负责内容：加载 DPO v1 adapter 跑 sample500。
- 对应模型：M3。

#### `configs/train/dpo_v2_qwen3vl_8b.yaml`

- 类型：配置 / DPO v2 保守版。
- 负责内容：DPO v2 train/holdout/decode-dev、SFT loss 辅助、早停指标。
- 对应模型：M3v2。

#### `configs/train/phase08_m3v2_sample500_server.yaml`

- 类型：配置 / M3v2 sample500。
- 负责内容：加载 DPO v2 adapter 跑 sample500 和 Train decode dev。
- 对应模型：M3v2。

#### `configs/train/dpo_v2_baseline_ablation_qwen3vl_8b.yaml`

- 类型：配置 / DPO v2 loss ablation。
- 负责内容：baseline DPO v2 候选，`loss_type=dpo`，基础 `lambda_sft`。
- 对应模型：`dpo_v2_baseline`。

#### `configs/train/dpo_v2_auxstrong_qwen3vl_8b.yaml`

- 类型：配置 / AuxDPO。
- 负责内容：更强 SFT/NLL 辅助项，测试能否保护 SFT 能力。
- 对应模型：`auxdpo_v2_strong`。

#### `configs/train/dpo_v2_auxstronger_qwen3vl_8b.yaml`

- 类型：配置 / AuxDPO 更强版。
- 负责内容：比 auxstrong 更强的能力保护候选。
- 当前状态：已准备但本轮为省成本未正式跑完。

#### `configs/train/dpo_v2_ipo_qwen3vl_8b.yaml`

- 类型：配置 / IPO。
- 负责内容：把 preference loss 从 DPO 换成 IPO，尝试抑制 margin 过冲。
- 当前状态：已准备但未完整实验。

#### `configs/train/dpo_v2_ipo_aux_qwen3vl_8b.yaml`

- 类型：配置 / IPO + SFT 辅助。
- 负责内容：同时测试 IPO 和能力保护。
- 当前状态：已准备但未完整实验。

#### `configs/train/grpo_qwen3vl_8b.yaml`

- 类型：配置 / GRPO。
- 负责内容：GRPO prompt、模型、reward、训练输出参数。
- 当前状态：只有 smoke 级别，未正式形成 M4。

#### `configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml`

- 类型：配置 / High-risk Repair SFT。
- 负责内容：指定 240 条 repair SFT mix、输出 adapter、Train decode dev 152 条评测路径和 repair_sft_r1 推理输出目录。
- 对应模型：`repair_sft_r1`。
- 读者提示：这是当前下一步小闭环最重要的配置。

## 8. 文件关系索引

### 8.1 如果想看“数据是怎么造出来的”

阅读顺序：

1. `configs/data_gen/main.yaml`
2. `src/mv_audit/data_gen/generate_base_cases.py`
3. `src/mv_audit/data_gen/anomaly_injector.py`
4. `src/mv_audit/data_gen/risk_rule_engine.py`
5. `src/mv_audit/data_gen/split_builder.py`
6. `src/mv_audit/data_gen/case_validator.py`

### 8.2 如果想看“图片和 bbox 是怎么来的”

阅读顺序：

1. `src/mv_audit/rendering/render_all.py`
2. `src/mv_audit/rendering/layout.py`
3. 四个 `render_*.py`
4. `src/mv_audit/rendering/bbox_recorder.py`
5. `src/mv_audit/perturbation/*.py`

### 8.3 如果想看“SFT 中每一部分”

阅读顺序：

1. `src/mv_audit/converters/common.py`
2. `src/mv_audit/converters/build_sft_data.py`
3. `configs/train/sft_lora_qwen3vl_8b_phase07_server.yaml`
4. `scripts/04_train_sft.sh`
5. `src/mv_audit/training/train_sft.py`
6. `src/mv_audit/inference/batch_inference.py`
7. `scripts/08_evaluate.sh`

### 8.4 如果想看“DPO 为什么失败”

阅读顺序：

1. `src/mv_audit/converters/build_dpo_pairs.py`
2. `src/mv_audit/training/train_dpo.py`
3. `src/mv_audit/training/reward_function.py`
4. `docs/experiments/phase08_m3_sample500/metrics_by_model.csv`
5. `src/mv_audit/analysis/dpo_error_migration.py`
6. `docs/experiments/phase08_dpo_diagnosis/`
7. README 第 14-16 节

### 8.5 如果想看“DPO v2 做了哪些修正”

阅读顺序：

1. `src/mv_audit/converters/build_dpo_v2_pairs.py`
2. `configs/train/dpo_v2_qwen3vl_8b.yaml`
3. `src/mv_audit/training/train_dpo.py`
4. `configs/train/dpo_v2_*_qwen3vl_8b.yaml`
5. `scripts/10_run_dpo_v2_ablation_5gpu_server.sh`
6. `src/mv_audit/analysis/archive_loss_ablation.py`
7. `docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/`

### 8.6 如果想看“High-risk Repair 下一步”

阅读顺序：

1. `src/mv_audit/analysis/high_risk_repair_pack.py`
2. `docs/experiments/phase08_high_risk_repair_pack_20260813/high_risk_miss_diagnosis_report.md`
3. `src/mv_audit/converters/build_high_risk_repair_sft_mix.py`
4. `configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml`
5. `scripts/11_run_high_risk_repair_sft_r1_server.sh`
6. `src/mv_audit/analysis/archive_high_risk_repair_sft.py`

### 8.7 如果想看“最终指标怎么算”

阅读顺序：

1. `src/mv_audit/evaluation/evaluate_all.py`
2. `src/mv_audit/evaluation/json_parser.py`
3. `src/mv_audit/evaluation/audit_metrics.py`
4. `src/mv_audit/evaluation/evidence_metrics.py`
5. `src/mv_audit/evaluation/bbox_evaluator.py`
6. `src/mv_audit/evaluation/hallucination_metrics.py`

### 8.8 如果想看“服务器上该跑哪个脚本”

| 目标 | 脚本 |
| --- | --- |
| 准备环境 | `scripts/00_prepare_env.sh` |
| 下载模型 | `scripts/00_download_qwen3vl.sh` |
| 训练 M2 SFT | `scripts/04_train_sft.sh` |
| 训练 DPO v1/M3 | `scripts/05_train_dpo.sh` |
| 训练 DPO v2/M3v2 | `scripts/05_train_dpo_v2.sh` |
| 跑 DPO v2 ablation | `scripts/10_run_dpo_v2_ablation_5gpu_server.sh` |
| 跑当前推荐的 repair_sft_r1 | `scripts/11_run_high_risk_repair_sft_r1_server.sh` |
| 评测 predictions | `scripts/08_evaluate.sh` |

### 8.9 如果想看“数据从哪里来、最后放在哪里”

阅读顺序：

1. `data/mv_audit/dictionaries/*.json`
2. `configs/data_gen/main.yaml`
3. `scripts/01_generate_main_cases.sh`
4. `data/mv_audit/raw_cases/main/`
5. `scripts/02_render_main_images.sh`
6. `data/mv_audit/images_main/`
7. `data/mv_audit/annotations_main/`
8. `scripts/03_build_main_train_data.sh`
9. `data/mv_audit/sft_main/`
10. `data/mv_audit/dpo_v2/`

### 8.10 如果想看“训练和评测产物在哪里”

阅读顺序：

1. `outputs/checkpoints/sft/`
2. `outputs/checkpoints/dpo/`
3. `outputs/predictions/`
4. `outputs/eval_reports/`
5. `outputs/logs/`
6. `outputs/runtime/`
7. `docs/experiments/`

### 8.11 如果想看“Qwen3-VL 官方代码参考哪里”

阅读顺序：

1. `external/Qwen3-VL-code/README.md`
2. `external/Qwen3-VL-code/qwen-vl-utils/src/qwen_vl_utils/vision_process.py`
3. `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/data/data_processor.py`
4. `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/train/train_qwen.py`
5. `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/train/trainer.py`
6. `external/Qwen3-VL-code/cookbooks/`

### 8.12 如果想看“哪些东西不要提交到 Git”

优先看：

1. `.gitignore`
2. `models/`
3. `external/`
4. `data/mv_audit/images*/`
5. `data/mv_audit/raw_cases*/`
6. `data/mv_audit/sft*/`
7. `data/mv_audit/dpo*/`
8. `data/mv_audit/grpo*/`
9. `outputs/`

这些目录多为下载资产、生成数据、大模型权重、checkpoint、预测或日志。项目通常只提交代码、配置、少量词典、报告摘要和 manifest，不提交大文件本体。

## 9. 非代码目录与资产清单

这一节解释 `data/`、`outputs/`、`models/`、`external/`、`notebooks/`。它们不是核心业务代码，但决定训练是否能跑、实验结果在哪里、哪些资产来自外部、哪些文件不应提交 Git。

### 9.1 `data/` 总览

`data/` 是项目的数据资产层。它从小词典开始，经过 case 生成、异常注入、图片渲染、bbox 标注和训练样本转换，最后形成 SFT/DPO/GRPO/评测使用的 JSONL 和图片。

| 路径 | 类型 | 负责内容 | 上游 | 下游 | Git 边界 |
| --- | --- | --- | --- | --- | --- |
| `data/mv_audit/dictionaries/` | 数据词典 | 生成合成报销 case 时抽样城市、商户、姓名、费用类型 | 人工维护的小 JSON | `generate_base_cases.py` | 小文件可提交 |
| `data/mv_audit/raw_cases/` | 生成数据 | debug 或小规模 raw case | `01_generate_cases.sh` | 渲染、SFT/DPO/GRPO 构造 | 生成产物，通常忽略 |
| `data/mv_audit/raw_cases/main/` | 主规模生成数据 | main 规模 raw case 和 split 真值 | `01_generate_main_cases.sh` | main 渲染、main SFT/DPO/GRPO | 生成产物，通常忽略 |
| `data/mv_audit/annotations/` | bbox/字段标注 | debug 图片字段位置和 annotation | `render_all.py` | SFT answer、bbox 评测 | 生成产物，通常忽略 |
| `data/mv_audit/annotations_main/` | main bbox/字段标注 | main 规模字段 bbox、case annotation | `02_render_main_images.sh` | `sft_main`、DPO v2、评测 | 生成产物，通常忽略 |
| `data/mv_audit/images/` | 渲染图片 | debug/smoke 凭证图片 | `02_render_images.sh` | 训练、推理、可视化 | 图片大文件，忽略 |
| `data/mv_audit/images_main/` | 主规模渲染图片 | main 规模四类凭证图片 | `02_render_main_images.sh` | M2/M3/M3v2/repair 推理训练 | 图片大文件，忽略 |
| `data/mv_audit/sft/` | SFT JSONL | debug SFT train/val 样本 | `03_build_train_data.sh` | `train_sft.py` dry-run | 生成产物，通常忽略 |
| `data/mv_audit/sft_main/` | 主 SFT JSONL | Phase07 M2 SFT train/val 样本 | `03_build_main_train_data.sh` | M2 训练、few-shot、repair calibration | 生成产物，通常忽略 |
| `data/mv_audit/dpo/` | DPO v1 pairs | M3 DPO v1 preference pairs | `build_dpo_pairs.py` | `train_dpo.py` | 生成产物，通常忽略 |
| `data/mv_audit/dpo_main/` | 主 DPO pairs | main 数据上的 DPO v1 pairs | `03_build_main_train_data.sh` | DPO v1 训练 | 生成产物，通常忽略 |
| `data/mv_audit/dpo_v2/` | DPO v2 pairs/splits | conservative DPO v2 train/holdout/decode-dev | `05_build_dpo_v2_pairs.sh` | M3v2、loss ablation、repair 泄漏检查 | 生成产物，通常忽略 |
| `data/mv_audit/grpo/` | GRPO prompts | debug GRPO prompt 数据 | `build_grpo_prompts.py` | `train_grpo.py` smoke | 生成产物，通常忽略 |
| `data/mv_audit/grpo_main/` | 主 GRPO prompts | main 数据上的 GRPO prompt 数据 | `03_build_main_train_data.sh` | GRPO 预留训练 | 生成产物，通常忽略 |
| `data/mv_audit/eval_sets/` | 评测集合 | sample manifest、train_decode_dev、评测 split 指针 | sample 构造脚本、DPO v2 builder | `batch_inference.py`、`evaluate_all.py` | 小 manifest 可归档，大数据忽略 |
| `data/mv_audit/templates/` | 模板占位 | 预留凭证模板目录 | 人工模板或未来生成模板 | rendering | 当前多为占位 |

#### `data/mv_audit/dictionaries/cities.json`

- 类型：数据词典。
- 负责内容：提供城市名称，生成报销城市、出差地、商户所在地等字段。
- 关键字段：城市字符串列表。
- 输入：人工整理的小词表。
- 输出：被 raw case 生成器采样进 case JSON。
- 上游：`configs/data_gen/*.yaml` 指定词典目录。
- 下游：`generate_base_cases.py`。
- 读者提示：这是“造业务样本时可抽到哪些城市”的素材库，不是模型训练结果。

#### `data/mv_audit/dictionaries/expense_types.json`

- 类型：数据词典。
- 负责内容：提供餐饮、交通、住宿等费用类别候选。
- 关键字段：费用类型字符串列表。
- 输入：人工整理的小词表。
- 输出：case 中的 expense type、规则判断和报告文本。
- 上游：数据生成配置。
- 下游：`generate_base_cases.py`、`risk_rule_engine.py`。
- 读者提示：它影响 case 的业务语义，也影响异常规则是否合理。

#### `data/mv_audit/dictionaries/merchants.json`

- 类型：数据词典。
- 负责内容：提供商户名称、供应商或消费场景名称。
- 关键字段：商户字符串列表。
- 输入：人工整理的小词表。
- 输出：发票、支付截图、订单、报销单中的 merchant 字段。
- 上游：数据生成配置。
- 下游：渲染、字段一致性评测、商户不一致异常。
- 读者提示：跨图 merchant 是否一致是审计任务的重要部分。

#### `data/mv_audit/dictionaries/names.json`

- 类型：数据词典。
- 负责内容：提供申请人、支付人、订单人等姓名候选。
- 关键字段：姓名字符串列表。
- 输入：人工整理的小词表。
- 输出：case 中人员相关字段。
- 上游：数据生成配置。
- 下游：人员一致性异常、SFT/DPO/评测真值。
- 读者提示：它让“申请人与付款人不一致”等异常有可控真值。

#### `data/mv_audit/**/*.gitkeep`

- 类型：目录占位文件。
- 负责内容：让 Git 保留空目录结构，例如 `raw_cases/`、`images/`、`sft/`、`dpo/`、`grpo/`、`eval_sets/`。
- 关键函数/类：无。
- 输入：无。
- 输出：无业务输出。
- 上游：仓库初始化。
- 下游：脚本可以直接写入这些目录。
- 读者提示：`.gitkeep` 本身没有业务含义，只是“这个目录应该存在”的标记。

### 9.2 `data/mv_audit/raw_cases*`

- 类型：生成数据 / case 真值。
- 负责内容：保存结构化报销 case，包括凭证字段、异常标签、风险等级、审计建议、证据真值。
- 关键文件模式：`*.jsonl`、split JSON/manifest、按 run 或 split 命名的 raw case 文件。
- 输入：`configs/data_gen/debug.yaml` 或 `configs/data_gen/main.yaml`、词典 JSON、异常注入规则。
- 输出：后续渲染和训练数据构造的源头。
- 上游：`scripts/01_generate_cases.sh`、`scripts/01_generate_main_cases.sh`、`generate_base_cases.py`、`anomaly_injector.py`、`split_builder.py`。
- 下游：`render_all.py`、`build_sft_data.py`、`build_dpo_pairs.py`、`build_dpo_v2_pairs.py`、`build_grpo_prompts.py`、`evaluate_all.py`。
- 读者提示：这是项目的“标准答案数据库”。模型看不到它的结构化真值，但评测和训练样本构造依赖它。

### 9.3 `data/mv_audit/annotations*`

- 类型：渲染标注 / bbox 真值。
- 负责内容：记录每张凭证图片中字段文本所在位置，通常包含 `source_image_id`、`source_doc_type`、字段名、字段值和 bbox。
- 关键文件模式：`field_bboxes*.jsonl`、annotation JSONL、case-image 映射文件。
- 输入：raw cases、rendering layout、字体与图片尺寸设置。
- 输出：SFT answer 中的 evidence/bbox 真值，以及 bbox evaluator 的标准答案。
- 上游：`render_all.py`、`bbox_recorder.py`、四个 `render_*.py`。
- 下游：`build_sft_data.py`、`batch_inference.py`、`bbox_evaluator.py`、`evidence_metrics.py`。
- 读者提示：这是“模型说证据在图上哪里”的评分依据。

### 9.4 `data/mv_audit/images*`

- 类型：渲染图片资产。
- 负责内容：保存生成出的发票、支付截图、报销单、订单截图。
- 关键目录：`train/`、`val_in_template/`、`val_unseen_template/`、`test_clean/`、`test_hard_negative/`、`test_robust/`、`test_unseen_template/`。
- 关键文件模式：`*.png`、`*.jpg`、按 case/image id 命名的凭证图片。
- 输入：raw cases、layout、模板、扰动策略。
- 输出：训练、推理和人工检查使用的多图输入。
- 上游：`scripts/02_render_images.sh`、`scripts/02_render_main_images.sh`。
- 下游：`train_sft.py`、`train_dpo.py`、`train_grpo.py`、`batch_inference.py`、`visualize_bbox.py`。
- 读者提示：这是 VLM 真正“看到”的东西，体积很大，不应直接进入 Git。

### 9.5 `data/mv_audit/sft*`

- 类型：SFT 训练样本。
- 负责内容：保存 `images + user prompt + assistant JSON answer` 格式的监督微调样本。
- 关键文件模式：`train.jsonl`、`val.jsonl`、`*_existing_images*.jsonl`、repair mix JSONL。
- 输入：raw cases、annotations、images、输出 schema。
- 输出：LoRA-SFT 训练和 dry-run 校验输入。
- 上游：`scripts/03_build_train_data.sh`、`scripts/03_build_main_train_data.sh`、`build_sft_data.py`、`build_high_risk_repair_sft_mix.py`。
- 下游：`train_sft.py`、few-shot 示例抽取、repair_sft_r1。
- 读者提示：这是 SFT 的“教材”。每一行通常对应一个 case，而不是一张图片。

### 9.6 `data/mv_audit/dpo*`

- 类型：DPO/IPO 偏好数据。
- 负责内容：保存 chosen/rejected answer pairs，以及 DPO v2 的 train/holdout/decode-dev 拆分。
- 关键文件模式：`pairs_train.jsonl`、`pairs_holdout.jsonl`、`train_decode_dev.jsonl`、pair audit/manifest。
- 输入：SFT 标准答案、raw cases、annotations、DPO pair 构造策略。
- 输出：DPO/IPO 训练输入和 Train decode dev 评测输入。
- 上游：`build_dpo_pairs.py`、`build_dpo_v2_pairs.py`、`scripts/05_build_dpo_v2_pairs.sh`。
- 下游：`train_dpo.py`、`batch_inference.py`、`dpo_error_migration.py`、High-risk Repair 泄漏检查。
- 读者提示：DPO 失败分析主要看这里的 pair 是否真的对齐业务指标。

### 9.7 `data/mv_audit/grpo*`

- 类型：GRPO prompt 数据。
- 负责内容：保存强化训练时用于生成多个候选回答的 prompt 数据。
- 关键文件模式：`prompts*.jsonl`、GRPO manifest。
- 输入：raw cases、images、output schema、reward function 需求。
- 输出：GRPO smoke 或正式训练输入。
- 上游：`build_grpo_prompts.py`。
- 下游：`train_grpo.py`、`reward_function.py`。
- 读者提示：当前 GRPO 还不是正式成功路线，更多是预留和 smoke 验证。

### 9.8 `data/mv_audit/eval_sets/`

- 类型：评测集合与 manifest。
- 负责内容：保存 sample500、train_decode_dev 或其他评测 split 的 case 列表和路径指针。
- 关键文件模式：`*.jsonl`、`*.json`、manifest。
- 输入：split builder、sample manifest builder、DPO v2 builder。
- 输出：批量推理和评测的固定输入集合。
- 上游：`scripts/build_phase07_sample_manifest.py`、`build_dpo_v2_pairs.py`。
- 下游：`batch_inference.py`、`evaluate_all.py`。
- 读者提示：这是“考卷名单”，不能随便混入训练候选选择，否则会造成泄漏。

### 9.9 `outputs/` 总览

`outputs/` 是实验运行后产生的结果区。它不定义业务规则，也不定义模型结构；它保存“跑出来的东西”。

| 路径 | 类型 | 负责内容 | 上游 | 下游 | Git 边界 |
| --- | --- | --- | --- | --- | --- |
| `outputs/checkpoints/sft/` | 模型产物 | LoRA-SFT adapter checkpoint | `train_sft.py` | `batch_inference.py`、DPO 初始模型 | 大文件忽略 |
| `outputs/checkpoints/dpo/` | 模型产物 | DPO/IPO adapter checkpoint | `train_dpo.py` | M3/M3v2 推理 | 大文件忽略 |
| `outputs/checkpoints/grpo/` | 模型产物 | GRPO checkpoint | `train_grpo.py` | M4 预留推理 | 大文件忽略 |
| `outputs/predictions/` | 推理产物 | 各模型 raw output 和 prediction JSONL | `batch_inference.py` | `evaluate_all.py`、analysis | 大文件忽略，摘要可归档 |
| `outputs/eval_reports/` | 评测产物 | metrics、error cases、figures、summary CSV | `evaluate_all.py` | README、docs/experiments、diagnosis | 关键摘要可归档 |
| `outputs/logs/` | 运行日志 | 训练/推理/脚本日志 | `scripts/*.sh`、Python logging | Debug、归档 | 日志通常忽略 |
| `outputs/runtime/` | 运行状态 | run id、watcher 日志、READY/FAILED 标记、归档路径 | server/local watcher 脚本 | 自动归档、关机、人工审计 | 状态产物通常忽略 |
| `outputs/tmp/` | 临时文件 | smoke、dry-run、临时评测缓存 | 各类检查命令 | 通常无下游 | 可删除/忽略 |
| `outputs/tmp_phase07_sample_manifest_check*` | 临时检查目录 | Phase07 sample manifest 检查过程输出 | sample manifest 检查 | 人工确认后无长期下游 | 可删除/忽略 |
| `outputs/.gitkeep` | 占位文件 | 保留 outputs 根目录 | 仓库初始化 | 让脚本有默认落点 | 可提交 |

#### `outputs/checkpoints/sft/qwen3vl_8b_lora*`

- 类型：SFT checkpoint。
- 负责内容：M2 或早期 SFT adapter。
- 输入：`data/mv_audit/sft_main/train.jsonl`、Qwen3-VL 基座模型、SFT 配置。
- 输出：LoRA adapter 文件、trainer 状态、训练配置快照。
- 上游：`scripts/04_train_sft.sh`。
- 下游：M2 推理、DPO policy 初始化、sample500 对比。
- 读者提示：这是 SFT 学出来的增量权重，不是完整基座模型。

#### `outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r1`

- 类型：High-risk Repair SFT checkpoint。
- 负责内容：保存 repair_sft_r1 小闭环训练出的 LoRA adapter。
- 输入：240 条 repair SFT mix、基座模型、repair SFT 配置。
- 输出：repair_sft_r1 adapter 和训练状态。
- 上游：`scripts/11_run_high_risk_repair_sft_r1_server.sh`。
- 下游：Train decode dev 推理和 High-risk Miss 验收。
- 读者提示：这是下一步小闭环的模型产物，是否有效要看 `outputs/eval_reports/phase08_high_risk_repair_train_decode_dev/`。

#### `outputs/checkpoints/dpo/qwen3vl_8b_dpo*`

- 类型：DPO/IPO checkpoint。
- 负责内容：保存 M3、M3v2、DPO v2 baseline、AuxDPO、IPO 等偏好训练 adapter。
- 输入：DPO pairs、SFT adapter、DPO/IPO 配置。
- 输出：adapter、training history、holdout history、reward audit。
- 上游：`scripts/05_train_dpo.sh`、`scripts/05_train_dpo_v2.sh`、`scripts/10_run_dpo_v2_ablation_5gpu_server.sh`。
- 下游：DPO 推理、error migration、ablation archive。
- 读者提示：这些目录证明“训练跑过”，但是否成功必须看业务指标，不看 loss 单独下结论。

#### `outputs/checkpoints/grpo/qwen3vl_8b_grpo*`

- 类型：GRPO checkpoint。
- 负责内容：保存 GRPO smoke 或预留强化训练 checkpoint。
- 输入：GRPO prompts、reward function、模型配置。
- 输出：GRPO adapter 或训练状态。
- 上游：`scripts/06_train_grpo.sh`。
- 下游：理论上的 M4 推理。
- 读者提示：当前不是正式结论路线，不能当作完成的 GRPO 实验成果。

### 9.10 `models/Qwen3-VL-8B-Instruct/`

`models/` 是下载到本地或服务器的基座模型目录。本项目训练的是 LoRA/DPO/GRPO adapter，不会把 Qwen3-VL 的完整权重复制进 Git。

#### `models/Qwen3-VL-8B-Instruct/config.json`

- 类型：模型配置。
- 负责内容：定义 Qwen3-VL 架构、层数、hidden size、vision/text 模块参数等。
- 输入：模型发布方提供。
- 输出：`AutoModelForVision2Seq` 或相关加载函数据此实例化模型。
- 上游：`scripts/00_download_qwen3vl.sh` 或手动下载。
- 下游：`qwen3vl_common.py`、`train_sft.py`、`train_dpo.py`、`batch_inference.py`。
- 读者提示：这是“模型说明书”，不是训练脚本。

#### `models/Qwen3-VL-8B-Instruct/configuration.json`

- 类型：模型元数据。
- 负责内容：保存模型仓库侧的额外配置或兼容性信息。
- 输入：模型发布方提供。
- 输出：辅助 Transformers/自定义代码识别模型配置。
- 上游：模型下载。
- 下游：模型加载。
- 读者提示：和 `config.json` 一样属于模型资产元数据。

#### `models/Qwen3-VL-8B-Instruct/generation_config.json`

- 类型：生成配置。
- 负责内容：保存默认生成参数，例如 eos/pad token、采样或解码默认值。
- 输入：模型发布方提供。
- 输出：推理时默认 generation 行为。
- 上游：模型下载。
- 下游：`batch_inference.py` 的生成过程。
- 读者提示：项目仍可在推理配置里覆盖温度、max tokens 等参数。

#### `models/Qwen3-VL-8B-Instruct/tokenizer.json`

- 类型：tokenizer 文件。
- 负责内容：定义文本如何切成 token。
- 输入：模型发布方提供。
- 输出：模型可处理的 token ids。
- 上游：模型下载。
- 下游：训练、DPO logprob、推理。
- 读者提示：它决定文字如何进入模型。

#### `models/Qwen3-VL-8B-Instruct/tokenizer_config.json`

- 类型：tokenizer 配置。
- 负责内容：定义 tokenizer 的特殊 token、padding/truncation 等行为。
- 输入：模型发布方提供。
- 输出：processor/tokenizer 加载参数。
- 上游：模型下载。
- 下游：`qwen3vl_common.py`。
- 读者提示：这是 tokenizer 的说明书。

#### `models/Qwen3-VL-8B-Instruct/chat_template.json`

- 类型：对话模板。
- 负责内容：定义 user/assistant/image message 如何拼成模型理解的 chat 格式。
- 输入：模型发布方提供。
- 输出：训练和推理的 prompt 拼接格式。
- 上游：模型下载。
- 下游：`train_sft.py`、`train_dpo.py`、`batch_inference.py`。
- 读者提示：如果模板错了，模型可能不是在回答同一种对话格式。

#### `models/Qwen3-VL-8B-Instruct/preprocessor_config.json`

- 类型：图像预处理配置。
- 负责内容：定义图片 resize、patch、归一化等视觉输入处理。
- 输入：模型发布方提供。
- 输出：processor 对图片的张量化结果。
- 上游：模型下载。
- 下游：`qwen3vl_common.py`、`vision_process.py`。
- 读者提示：它影响图片如何变成模型可读的视觉 token。

#### `models/Qwen3-VL-8B-Instruct/video_preprocessor_config.json`

- 类型：视频预处理配置。
- 负责内容：定义视频输入预处理参数。
- 输入：模型发布方提供。
- 输出：视频任务 processor 行为。
- 上游：模型下载。
- 下游：当前主任务通常不用，官方 demo 可能用。
- 读者提示：本项目主要是多图凭证，不是视频理解。

#### `models/Qwen3-VL-8B-Instruct/model.safetensors.index.json`

- 类型：权重索引。
- 负责内容：说明完整模型权重被拆成哪些 shard，以及每个参数在哪个 shard 里。
- 输入：模型发布方提供。
- 输出：Transformers 按索引加载多个 safetensors shard。
- 上游：模型下载。
- 下游：模型加载。
- 读者提示：这不是权重本体，只是“权重目录表”；真正 `.safetensors` shard 很大，不提交 Git。

#### `models/Qwen3-VL-8B-Instruct/merges.txt`

- 类型：BPE tokenizer 规则。
- 负责内容：定义子词合并规则。
- 输入：模型发布方提供。
- 输出：tokenizer 分词行为。
- 上游：模型下载。
- 下游：训练和推理。
- 读者提示：它和 `vocab.json`、`tokenizer.json` 一起决定文本分词。

#### `models/Qwen3-VL-8B-Instruct/vocab.json`

- 类型：tokenizer 词表。
- 负责内容：保存 token 到 id 的映射。
- 输入：模型发布方提供。
- 输出：token ids。
- 上游：模型下载。
- 下游：训练和推理。
- 读者提示：这是模型“认识哪些 token”的表。

#### `models/Qwen3-VL-8B-Instruct/README.md`

- 类型：模型说明文档。
- 负责内容：模型发布方对用法、能力、限制、许可证或引用方式的说明。
- 输入：模型发布方提供。
- 输出：供人阅读。
- 上游：模型下载。
- 下游：环境配置和问题排查。
- 读者提示：遇到模型加载、输入格式、license 问题先看这里。

#### `models/Qwen3-VL-8B-Instruct/*.safetensors`

- 类型：基座模型权重 shard。
- 负责内容：保存 Qwen3-VL-8B 的完整参数。
- 输入：模型发布方提供。
- 输出：训练和推理加载的基座权重。
- 上游：`scripts/00_download_qwen3vl.sh` 或手动下载。
- 下游：SFT/DPO/GRPO/推理。
- 读者提示：这是超大二进制资产，必须放本地或服务器，不进 Git。

### 9.11 `external/Qwen3-VL-code/`

`external/` 是第三方参考代码。它可以帮助理解官方 Qwen3-VL 的输入处理、finetune、demo 和 benchmark，但本项目主链路仍在 `src/mv_audit/`。

#### `external/Qwen3-VL-code/README.md`

- 类型：第三方说明文档。
- 负责内容：Qwen3-VL 官方仓库总说明。
- 上游：外部仓库下载。
- 下游：理解模型能力、依赖和官方示例。
- 读者提示：这是参考资料，不是本项目 README。

#### `external/Qwen3-VL-code/LICENSE`

- 类型：许可证。
- 负责内容：说明外部 Qwen3-VL 代码的使用许可。
- 上游：外部仓库。
- 下游：引用、分发、复用时的合规判断。
- 读者提示：第三方代码不能当作本项目原创代码。

#### `external/Qwen3-VL-code/requirements_web_demo.txt`

- 类型：依赖清单。
- 负责内容：官方 web demo 所需 Python 包。
- 上游：外部仓库。
- 下游：`web_demo_mm.py`。
- 读者提示：本项目训练环境主要看根目录 `requirements.txt`，这个文件只服务官方 demo。

#### `external/Qwen3-VL-code/web_demo_mm.py`

- 类型：官方多模态 demo。
- 负责内容：提供 Qwen3-VL web demo 入口。
- 上游：官方模型和 demo 依赖。
- 下游：人工试用模型能力。
- 读者提示：它不是本项目批量推理入口；本项目用 `src/mv_audit/inference/batch_inference.py`。

### 9.12 `external/Qwen3-VL-code/cookbooks/`

官方 cookbooks 是 notebook 示例集合，用来展示 Qwen3-VL 在 OCR、文档解析、空间理解、移动端 agent、视频理解等任务上的用法。

| 文件 | 类型 | 负责内容 | 与本项目关系 |
| --- | --- | --- | --- |
| `external/Qwen3-VL-code/cookbooks/ocr.ipynb` | notebook / OCR 示例 | 演示文字识别 | 可参考凭证 OCR 能力 |
| `external/Qwen3-VL-code/cookbooks/document_parsing.ipynb` | notebook / 文档解析 | 演示复杂文档结构理解 | 可参考发票/报销单解析 |
| `external/Qwen3-VL-code/cookbooks/long_document_understanding.ipynb` | notebook / 长文档 | 演示长文档理解 | 与多凭证长上下文相关 |
| `external/Qwen3-VL-code/cookbooks/2d_grounding.ipynb` | notebook / 2D 定位 | 演示 2D grounding | 可参考 bbox/evidence 定位 |
| `external/Qwen3-VL-code/cookbooks/3d_grounding.ipynb` | notebook / 3D 定位 | 演示 3D grounding | 当前项目基本不用 |
| `external/Qwen3-VL-code/cookbooks/spatial_understanding.ipynb` | notebook / 空间理解 | 演示场景空间关系 | 只作为 VLM 能力参考 |
| `external/Qwen3-VL-code/cookbooks/mmcode.ipynb` | notebook / 多模态代码 | 图像到代码示例 | 与本项目无直接训练关系 |
| `external/Qwen3-VL-code/cookbooks/mobile_agent.ipynb` | notebook / 手机 agent | 移动界面理解与操作 | 当前项目不用 |
| `external/Qwen3-VL-code/cookbooks/computer_use.ipynb` | notebook / computer use | 屏幕理解与操作 | 当前项目不用 |
| `external/Qwen3-VL-code/cookbooks/omni_recognition.ipynb` | notebook / 通用识别 | 通用图像识别示例 | 只作模型能力参考 |
| `external/Qwen3-VL-code/cookbooks/think_with_images.ipynb` | notebook / 图像推理 | 图片辅助推理示例 | 可参考 prompt 风格 |
| `external/Qwen3-VL-code/cookbooks/video_understanding.ipynb` | notebook / 视频理解 | 视频输入示例 | 当前项目不用 |
| `external/Qwen3-VL-code/cookbooks/german_document_ocr.ipynb` | notebook / 德语 OCR | 非中文/英文文档 OCR 示例 | 当前项目不用 |

#### `external/Qwen3-VL-code/cookbooks/assets/**`

- 类型：官方示例图片/视频/JSON。
- 负责内容：给 cookbooks 提供输入素材，例如 OCR 图片、文档解析图片、空间理解图片、移动端截图、`cam_infos.json`。
- 代表文件：`external/Qwen3-VL-code/cookbooks/assets/spatial_understanding/cam_infos.json` 保存空间理解示例中的相机信息；其余 `*.jpg`、`*.jpeg`、`*.png` 是 notebook 展示图片。
- 输入：官方仓库自带。
- 输出：notebook demo 的可视化示例。
- 上游：外部仓库。
- 下游：cookbooks。
- 读者提示：这些不是 MultiVoucher-Audit 的报销凭证数据。

#### `external/Qwen3-VL-code/cookbooks/utils/agent_function_call.py`

- 类型：官方 cookbook 工具。
- 负责内容：辅助 agent/function call 示例组织工具调用。
- 上游：cookbooks。
- 下游：`mobile_agent.ipynb` 等示例。
- 读者提示：本项目审计 pipeline 不依赖它。

#### `external/Qwen3-VL-code/cookbooks/utils/multimodal_coding/take_screenshot.py`

- 类型：官方 cookbook 工具。
- 负责内容：为 multimodal coding 示例截屏。
- 上游：cookbooks。
- 下游：`mmcode.ipynb`。
- 读者提示：与本项目凭证渲染无直接关系。

#### `external/Qwen3-VL-code/cookbooks/utils/multimodal_coding/test_mmcode.py`

- 类型：官方 cookbook 测试/示例。
- 负责内容：测试 multimodal coding 示例效果。
- 上游：cookbooks。
- 下游：官方 demo。
- 读者提示：不是本项目单元测试。

### 9.13 `external/Qwen3-VL-code/qwen-vl-utils/`

#### `external/Qwen3-VL-code/qwen-vl-utils/README.md`

- 类型：第三方工具说明。
- 负责内容：说明 `qwen-vl-utils` 如何处理 image/video 输入。
- 上游：外部仓库。
- 下游：理解 Qwen-VL processor 的输入格式。
- 读者提示：本项目多图输入处理可对照这里排查。

#### `external/Qwen3-VL-code/qwen-vl-utils/pyproject.toml`

- 类型：第三方包配置。
- 负责内容：定义 `qwen-vl-utils` 包名、依赖、构建配置。
- 上游：外部仓库。
- 下游：安装官方 utils。
- 读者提示：不是本项目 `pyproject.toml`。

#### `external/Qwen3-VL-code/qwen-vl-utils/requirements.lock`

- 类型：第三方依赖锁定文件。
- 负责内容：记录运行依赖版本。
- 上游：外部仓库。
- 下游：官方 utils 环境复现。
- 读者提示：只对外部工具包有效。

#### `external/Qwen3-VL-code/qwen-vl-utils/requirements-dev.lock`

- 类型：第三方开发依赖锁定文件。
- 负责内容：记录官方 utils 开发/测试依赖。
- 上游：外部仓库。
- 下游：官方 utils 开发环境。
- 读者提示：本项目不需要照它训练。

#### `external/Qwen3-VL-code/qwen-vl-utils/src/qwen_vl_utils/__init__.py`

- 类型：第三方包入口。
- 负责内容：声明 `qwen_vl_utils` 包并导出工具函数。
- 上游：外部仓库。
- 下游：官方示例和可能的模型输入处理。
- 读者提示：门牌号文件。

#### `external/Qwen3-VL-code/qwen-vl-utils/src/qwen_vl_utils/vision_process.py`

- 类型：第三方视觉输入处理。
- 负责内容：处理 Qwen-VL image/video message，生成 processor 可吃的视觉输入。
- 关键函数/类：官方视觉处理函数。
- 输入：chat messages 中的 image/video 字段。
- 输出：图像/视频张量或路径解析结果。
- 上游：官方 Qwen-VL 示例。
- 下游：本项目的 `qwen3vl_common.py` 可参考其输入格式。
- 读者提示：如果多图输入加载报错，这是最值得对照的官方文件。

### 9.14 `external/Qwen3-VL-code/qwen-vl-finetune/`

#### `external/Qwen3-VL-code/qwen-vl-finetune/README.md`

- 类型：官方 finetune 说明。
- 负责内容：说明 Qwen-VL 官方微调流程、数据格式和脚本。
- 上游：外部仓库。
- 下游：本项目 SFT/DPO 训练实现参考。
- 读者提示：本项目没有直接照搬官方训练入口，而是在 `src/mv_audit/training/` 写了任务专用实现。

#### `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/data/__init__.py`

- 类型：第三方包入口。
- 负责内容：声明官方 finetune data 子包。
- 上游：外部仓库。
- 下游：官方 finetune 数据模块。
- 读者提示：门牌号文件。

#### `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/data/data_processor.py`

- 类型：官方 finetune 数据处理。
- 负责内容：读取官方格式训练数据，处理图文消息和 labels。
- 输入：官方 finetune 数据 JSON。
- 输出：模型训练 batch。
- 上游：官方 finetune 脚本。
- 下游：`train_qwen.py`。
- 读者提示：可用来对照本项目 `DataCollatorForQwenVLSFT` 为什么要 mask prompt token。

#### `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/data/rope2d.py`

- 类型：官方位置编码工具。
- 负责内容：处理视觉/文本混合输入中的 2D RoPE 相关逻辑。
- 输入：模型输入位置和视觉 token 信息。
- 输出：位置编码相关张量。
- 上游：官方 finetune 数据处理。
- 下游：官方训练。
- 读者提示：底层模型适配参考，一般不用读。

#### `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/train/argument.py`

- 类型：官方训练参数。
- 负责内容：定义 finetune 命令行参数和训练 dataclass。
- 上游：官方 finetune。
- 下游：`train_qwen.py`。
- 读者提示：本项目参数主要在 `configs/train/*.yaml`，不是这里。

#### `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/train/train_qwen.py`

- 类型：官方 Qwen-VL 训练入口。
- 负责内容：加载模型、数据、训练参数并启动官方 finetune。
- 输入：官方脚本参数、数据、模型。
- 输出：官方 finetune checkpoint。
- 上游：`qwen-vl-finetune/scripts/*.sh`。
- 下游：官方训练产物。
- 读者提示：这是官方训练主入口；本项目自己的训练入口是 `train_sft.py`、`train_dpo.py`、`train_grpo.py`。

#### `external/Qwen3-VL-code/qwen-vl-finetune/qwenvl/train/trainer.py`

- 类型：官方 Trainer 扩展。
- 负责内容：封装官方 finetune 中对 Hugging Face Trainer 的定制。
- 输入：模型、数据、训练参数。
- 输出：训练 step、保存 checkpoint。
- 上游：`train_qwen.py`。
- 下游：官方 checkpoint。
- 读者提示：可作为训练机制参考，但本项目有自己的 collator 和损失函数。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/sft.sh`

- 类型：官方 SFT 脚本。
- 负责内容：启动默认 Qwen-VL SFT。
- 上游：官方 finetune。
- 下游：`train_qwen.py`。
- 读者提示：不是本项目 `scripts/04_train_sft.sh`。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/sft_7b.sh`

- 类型：官方 SFT 脚本。
- 负责内容：启动 7B 规模模型 SFT 示例。
- 上游：官方 finetune。
- 下游：`train_qwen.py`。
- 读者提示：本项目用 Qwen3-VL-8B，不直接用此脚本。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/sft_30a3b.sh`

- 类型：官方 SFT 脚本。
- 负责内容：启动 30A3B 相关 SFT 示例。
- 上游：官方 finetune。
- 下游：`train_qwen.py`。
- 读者提示：只作官方参数参考。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/sft_30a3b_lora.sh`

- 类型：官方 LoRA SFT 脚本。
- 负责内容：启动 30A3B LoRA 微调示例。
- 上游：官方 finetune。
- 下游：`train_qwen.py`。
- 读者提示：可参考 LoRA 参数，但项目实际配置在 `configs/train/*.yaml`。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/sft_32b.sh`

- 类型：官方 SFT 脚本。
- 负责内容：启动 32B 规模模型 SFT 示例。
- 上游：官方 finetune。
- 下游：`train_qwen.py`。
- 读者提示：本项目不用 32B。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/sft_qwen3_4b.sh`

- 类型：官方 SFT 脚本。
- 负责内容：启动 Qwen3 4B 模型 SFT 示例。
- 上游：官方 finetune。
- 下游：`train_qwen.py`。
- 读者提示：只作小模型参数参考。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/zero2.json`

- 类型：DeepSpeed 配置。
- 负责内容：ZeRO-2 训练优化配置。
- 上游：官方 finetune。
- 下游：官方训练脚本。
- 读者提示：如果未来改 DeepSpeed，可参考但不能直接当成本项目已启用。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/zero3.json`

- 类型：DeepSpeed 配置。
- 负责内容：ZeRO-3 参数切分训练配置。
- 上游：官方 finetune。
- 下游：官方训练脚本。
- 读者提示：当前 DPO v2 实际采用的是 `device_map=auto` 分片，不是 ZeRO-3。

#### `external/Qwen3-VL-code/qwen-vl-finetune/scripts/zero3_offload.json`

- 类型：DeepSpeed 配置。
- 负责内容：ZeRO-3 + offload 训练配置。
- 上游：官方 finetune。
- 下游：官方训练脚本。
- 读者提示：适合显存紧张时参考，但 offload 会更慢。

#### `external/Qwen3-VL-code/qwen-vl-finetune/tools/check_image.py`

- 类型：官方数据检查工具。
- 负责内容：检查 finetune 数据中的图片可读性。
- 上游：官方 finetune 数据。
- 下游：官方训练前校验。
- 读者提示：本项目图片校验主要在自己的 dry-run 和数据构造里做。

#### `external/Qwen3-VL-code/qwen-vl-finetune/tools/pack_data.py`

- 类型：官方数据打包工具。
- 负责内容：将官方格式数据打包成训练使用形式。
- 上游：官方数据。
- 下游：官方 finetune。
- 读者提示：本项目不使用它构造 SFT/DPO 数据。

#### `external/Qwen3-VL-code/qwen-vl-finetune/tools/process_bbox.ipynb`

- 类型：官方 bbox notebook。
- 负责内容：处理或展示 bbox 相关数据。
- 上游：官方示例数据。
- 下游：官方 grounding/finetune 示例。
- 读者提示：可参考 bbox 格式思路，本项目 bbox 真值由 `bbox_recorder.py` 生成。

#### `external/Qwen3-VL-code/qwen-vl-finetune/demo/single_images.json`

- 类型：官方 demo 数据。
- 负责内容：单图 demo 输入列表。
- 上游：官方 demo 图片。
- 下游：官方 finetune/demo。
- 读者提示：不是本项目训练数据。

#### `external/Qwen3-VL-code/qwen-vl-finetune/demo/video.json`

- 类型：官方 demo 数据。
- 负责内容：视频 demo 输入列表。
- 上游：官方 demo 视频。
- 下游：官方 finetune/demo。
- 读者提示：本项目不做视频审计。

#### `external/Qwen3-VL-code/qwen-vl-finetune/demo/images/*` 和 `demo/videos/*`

- 类型：官方 demo 多媒体资产。
- 负责内容：给官方 finetune/demo 提供示例图片和视频。
- 上游：外部仓库。
- 下游：官方 demo JSON。
- 读者提示：不是 MultiVoucher-Audit 图片。

### 9.15 `external/Qwen3-VL-code/evaluation/`

官方 evaluation 目录包含通用 VLM benchmark 的推理与评分脚本，和本项目业务评测不是一套指标。

| 子目录 | 类型 | 负责内容 | 典型文件 | 与本项目关系 |
| --- | --- | --- | --- | --- |
| `external/Qwen3-VL-code/evaluation/MathVision/` | 官方 benchmark | 数学视觉任务评测 | `run_mathv.py`、`infer_*.sh`、`eval_*.sh` | 只作官方评测参考 |
| `external/Qwen3-VL-code/evaluation/mmmu/` | 官方 benchmark | MMMU 多模态问答评测 | `run_mmmu.py`、`dataset_utils.py` | 与报销审计指标不同 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/` | 官方 benchmark | 目标检测/定位类任务 | `run_odinw.py`、`eval_utils.py` | 可参考 grounding 评测风格 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/` | 官方 benchmark | 真实世界问答 | `run_realworldqa.py` | 只作泛化能力参考 |
| `external/Qwen3-VL-code/evaluation/VideoMME/` | 官方 benchmark | 视频多模态评测 | `run_videomme.py` | 当前项目不用 |

每个 benchmark 子目录中常见文件含义：

- `README.md`：该 benchmark 的运行说明。
- `requirements.txt`：该 benchmark 额外依赖。
- `common_utils.py`：通用工具函数。
- `dataset_utils.py`：数据集读取与格式转换。
- `eval_utils.py`：指标计算工具。
- `infer_instruct.sh` / `infer_think.sh`：不同模式的推理脚本。
- `eval_instruct.sh` / `eval_think.sh`：不同模式的评测脚本。
- `run_*.py`：该 benchmark 的主运行入口。

逐文件路径索引：

| 文件 | 类型 | 负责内容 |
| --- | --- | --- |
| `external/Qwen3-VL-code/evaluation/MathVision/README.md` | 官方 benchmark 文档 | MathVision 运行说明 |
| `external/Qwen3-VL-code/evaluation/MathVision/requirements.txt` | 依赖清单 | MathVision 额外依赖 |
| `external/Qwen3-VL-code/evaluation/MathVision/common_utils.py` | 工具代码 | MathVision 通用工具 |
| `external/Qwen3-VL-code/evaluation/MathVision/dataset_utils.py` | 数据代码 | MathVision 数据读取和格式转换 |
| `external/Qwen3-VL-code/evaluation/MathVision/eval_utils.py` | 评测代码 | MathVision 指标计算 |
| `external/Qwen3-VL-code/evaluation/MathVision/infer_instruct.sh` | 推理脚本 | instruct 模式推理 |
| `external/Qwen3-VL-code/evaluation/MathVision/infer_think.sh` | 推理脚本 | think 模式推理 |
| `external/Qwen3-VL-code/evaluation/MathVision/eval_instruct.sh` | 评测脚本 | instruct 模式评测 |
| `external/Qwen3-VL-code/evaluation/MathVision/eval_think.sh` | 评测脚本 | think 模式评测 |
| `external/Qwen3-VL-code/evaluation/MathVision/run_mathv.py` | benchmark 入口 | MathVision 主运行入口 |
| `external/Qwen3-VL-code/evaluation/mmmu/README.md` | 官方 benchmark 文档 | MMMU 运行说明 |
| `external/Qwen3-VL-code/evaluation/mmmu/requirements.txt` | 依赖清单 | MMMU 额外依赖 |
| `external/Qwen3-VL-code/evaluation/mmmu/common_utils.py` | 工具代码 | MMMU 通用工具 |
| `external/Qwen3-VL-code/evaluation/mmmu/dataset_utils.py` | 数据代码 | MMMU 数据读取和格式转换 |
| `external/Qwen3-VL-code/evaluation/mmmu/eval_utils.py` | 评测代码 | MMMU 指标计算 |
| `external/Qwen3-VL-code/evaluation/mmmu/infer_instruct.sh` | 推理脚本 | instruct 模式推理 |
| `external/Qwen3-VL-code/evaluation/mmmu/infer_think.sh` | 推理脚本 | think 模式推理 |
| `external/Qwen3-VL-code/evaluation/mmmu/eval_instruct.sh` | 评测脚本 | instruct 模式评测 |
| `external/Qwen3-VL-code/evaluation/mmmu/eval_think.sh` | 评测脚本 | think 模式评测 |
| `external/Qwen3-VL-code/evaluation/mmmu/run_mmmu.py` | benchmark 入口 | MMMU 主运行入口 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/README.md` | 官方 benchmark 文档 | ODinW-13 运行说明 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/requirements.txt` | 依赖清单 | ODinW-13 额外依赖 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/dataset_utils.py` | 数据代码 | ODinW-13 数据读取和格式转换 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/eval_utils.py` | 评测代码 | ODinW-13 指标计算 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/infer_instruct.sh` | 推理脚本 | instruct 模式推理 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/infer_think.sh` | 推理脚本 | think 模式推理 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/eval_instruct.sh` | 评测脚本 | instruct 模式评测 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/eval_think.sh` | 评测脚本 | think 模式评测 |
| `external/Qwen3-VL-code/evaluation/ODinW-13/run_odinw.py` | benchmark 入口 | ODinW-13 主运行入口 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/README.md` | 官方 benchmark 文档 | RealWorldQA 运行说明 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/requirements.txt` | 依赖清单 | RealWorldQA 额外依赖 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/common_utils.py` | 工具代码 | RealWorldQA 通用工具 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/dataset_utils.py` | 数据代码 | RealWorldQA 数据读取和格式转换 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/eval_utils.py` | 评测代码 | RealWorldQA 指标计算 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/infer_instruct.sh` | 推理脚本 | instruct 模式推理 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/infer_think.sh` | 推理脚本 | think 模式推理 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/eval_instruct.sh` | 评测脚本 | instruct 模式评测 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/eval_think.sh` | 评测脚本 | think 模式评测 |
| `external/Qwen3-VL-code/evaluation/RealWorldQA/run_realworldqa.py` | benchmark 入口 | RealWorldQA 主运行入口 |
| `external/Qwen3-VL-code/evaluation/VideoMME/README.md` | 官方 benchmark 文档 | VideoMME 运行说明 |
| `external/Qwen3-VL-code/evaluation/VideoMME/requirements.txt` | 依赖清单 | VideoMME 额外依赖 |
| `external/Qwen3-VL-code/evaluation/VideoMME/dataset_utils.py` | 数据代码 | VideoMME 数据读取和格式转换 |
| `external/Qwen3-VL-code/evaluation/VideoMME/eval_utils.py` | 评测代码 | VideoMME 指标计算 |
| `external/Qwen3-VL-code/evaluation/VideoMME/infer_instruct.sh` | 推理脚本 | instruct 模式推理 |
| `external/Qwen3-VL-code/evaluation/VideoMME/infer_think.sh` | 推理脚本 | think 模式推理 |
| `external/Qwen3-VL-code/evaluation/VideoMME/eval_instruct.sh` | 评测脚本 | instruct 模式评测 |
| `external/Qwen3-VL-code/evaluation/VideoMME/eval_think.sh` | 评测脚本 | think 模式评测 |
| `external/Qwen3-VL-code/evaluation/VideoMME/run_videomme.py` | benchmark 入口 | VideoMME 主运行入口 |

读者提示：本项目最终指标不是这些 benchmark 指标，而是 `src/mv_audit/evaluation/evaluate_all.py` 输出的审计业务指标。

### 9.16 `external/Qwen3-VL-code/docker/`

#### `external/Qwen3-VL-code/docker/Dockerfile-qwen3vl-cu128`

- 类型：官方 Dockerfile。
- 负责内容：构建 CUDA 12.8 相关 Qwen3-VL 运行镜像。
- 上游：外部仓库。
- 下游：官方 demo 或训练环境。
- 读者提示：本项目服务器环境不是靠这个 Dockerfile 证明的，实际环境仍看本项目脚本和服务器日志。

#### `external/Qwen3-VL-code/docker/docker_web_demo.sh`

- 类型：官方 Docker 启动脚本。
- 负责内容：启动 web demo 容器。
- 上游：官方 Dockerfile。
- 下游：`web_demo_mm.py`。
- 读者提示：不是本项目训练入口。

### 9.17 `notebooks/`

#### `notebooks/.gitkeep`

- 类型：目录占位文件。
- 负责内容：保留 notebooks 目录。
- 输入：无。
- 输出：无业务输出。
- 上游：仓库初始化。
- 下游：未来临时探索 notebook。
- 读者提示：当前项目没有正式 notebook pipeline；正式流程看 `scripts/` 和 `src/`。

## 10. 覆盖范围和排除项

本文覆盖：

- `src/mv_audit/**/*.py`
- `scripts/*.sh`
- `scripts/*.ps1`
- `scripts/*.py`
- `configs/**/*.yaml`
- `configs/**/*.json`
- `tests/**/*.py`
- `data/mv_audit/dictionaries/*.json`
- `data/mv_audit/*/.gitkeep`
- `data/mv_audit/raw_cases*/`、`annotations*/`、`images*/`、`sft*/`、`dpo*/`、`grpo*/`、`eval_sets/`、`templates/` 的目录职责和常见产物模式
- `outputs/` 下 checkpoint、prediction、eval report、log、runtime、tmp 的目录职责和常见产物模式
- `models/Qwen3-VL-8B-Instruct/` 下 tokenizer、config、processor、generation、index、README 等模型元数据
- `models/Qwen3-VL-8B-Instruct/*.safetensors` 的资产类型和 Git 边界
- `external/Qwen3-VL-code/` 下官方 README、demo、cookbooks、qwen-vl-utils、qwen-vl-finetune、evaluation、docker 的参考作用
- `notebooks/.gitkeep`

本文不覆盖：

- `__pycache__/`：Python 自动生成缓存。
- `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`：本地工具缓存。
- `.gitkeep` 的业务逻辑，因为它只是目录占位；但本文说明了各 `.gitkeep` 所在目录的用途。
- 图片、视频、模型权重 shard、checkpoint、全量 predictions、原始训练日志的逐个二进制内容。
- `docs/experiments/` 下每个实验产物的逐文件解释；这些属于结果归档，不是代码逻辑。

## 11. 最短理解路径

如果只想快速知道代码怎么串起来，按这个顺序读：

1. `src/mv_audit/converters/common.py`：标准 prompt/answer/evidence 是怎么定义的。
2. `src/mv_audit/training/train_sft.py`：SFT 怎么训练模型。
3. `src/mv_audit/training/train_dpo.py`：DPO/IPO 怎么进一步训练偏好。
4. `src/mv_audit/inference/batch_inference.py`：模型怎么批量生成 predictions。
5. `src/mv_audit/evaluation/evaluate_all.py`：结果怎么打分。
6. `src/mv_audit/analysis/high_risk_repair_pack.py`：为什么现在转向 High-risk Repair。
7. `scripts/11_run_high_risk_repair_sft_r1_server.sh`：下一步小闭环怎么跑。
