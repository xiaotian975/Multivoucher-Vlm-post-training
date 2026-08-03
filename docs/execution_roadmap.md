# MultiVoucher-Audit Execution Roadmap

## 执行原则

本文使用 phase 00 到 phase 08 作为唯一工程执行阶段编号。不要把“模型训练阶段 0/1/2/3”和本路线的工程阶段混用。

第一版完整路线走到 phase 08，也就是包含 LoRA-SFT、DPO 和小规模 GRPO。phase 00 到 phase 07 先形成可运行、可评测的核心闭环；phase 08 在 SFT 输出稳定后完成风险偏好和规则奖励增强。

每次只推进一个 phase。跨阶段公共文件，尤其是 `configs/schema/case_schema.json`、`configs/schema/output_schema.json` 和 `src/mv_audit/data_gen/risk_rule_engine.py`，只能在所属阶段或显式 schema migration 中修改。

## Phase Roadmap

| Phase | 目标 | 允许修改范围 | 主要产出文件 | 验收标准 | 下一阶段依赖 |
| --- | --- | --- | --- | --- | --- |
| 00 | 建立项目骨架、环境、目录、基础工具 | 项目根配置、`configs/`、`data/`、`src/mv_audit/`、`scripts/`、`outputs/`、`notebooks/` | `README.md`、`pyproject.toml`、`requirements.txt`、基础 utils、空 debug pipeline | `pip install -e .` 可用，`mv_audit` 可导入，基础 JSONL/YAML 工具可用 | 可运行 Python 包和目录规范 |
| 01 | 下载或加载 `Qwen3-VL-8B-Instruct`，跑通单图/多图 smoke test | `configs/model/`、`src/mv_audit/inference/`、模型准备脚本、README 说明 | Qwen3-VL config、下载脚本、单图和多图 smoke test | 能加载模型、读取图片、生成文本，多图 prompt 不报错 | 已验证模型接口、显存和图片输入格式 |
| 02 | 定义 case schema、字典和正常底层交易真值表 | `configs/schema/case_schema.json`、`configs/data_gen/`、`data/mv_audit/dictionaries/`、`src/mv_audit/data_gen/` 中 base case 生成和校验 | 字典、debug 配置、`generate_base_cases.py`、`case_validator.py`、正常 base cases | 数量正确，case_id 不重复，金额/日期/枚举/schema 校验通过 | 稳定的正常 case 输入 |
| 03 | 实现异常注入、risk rule engine 和 case 级数据划分 | `src/mv_audit/data_gen/anomaly_injector.py`、`risk_rule_engine.py`、`split_builder.py`、raw case 输出 | 异常 case、各 split JSONL、统计报告 | 异常分布、risk_level、audit_result 统计合理；抽样标签符合规则 | 带异常和 split 的结构化 cases |
| 04 | 渲染四类凭证图片、记录 bbox、生成视觉扰动 | `src/mv_audit/rendering/`、`src/mv_audit/perturbation/`、图片和 bbox annotations | 四类渲染器、bbox recorder、扰动模块、bbox 可视化样本 | 随机 50 张 bbox 可视化检查，字段框与文字对齐 | 图片路径、image_id、bbox records |
| 05 | 构造 SFT/DPO/GRPO 数据格式 | `src/mv_audit/converters/`、`configs/schema/output_schema.json`、训练格式数据 | SFT JSONL、DPO pairs、GRPO prompts、output schema | SFT answer 是合法 JSON；DPO chosen/rejected 同 prompt；GRPO ground truth 足够 reward 使用 | 训练和 reward 所需数据 |
| 06 | 实现 JSON parser、bbox evaluator 和基础评测 | `src/mv_audit/evaluation/`、fake prediction 测试脚本 | parser、field/consistency/audit/evidence/hallucination metrics、`evaluate_all.py` | 完美预测接近 1，故意错误预测能触发高风险漏检、幻觉和 bbox 错误 | 可信评测系统 |
| 07 | 完成 LoRA-SFT、M0/M1/M2 推理与评测 | `src/mv_audit/training/train_sft.py`、`src/mv_audit/inference/batch_inference.py`、SFT config 和推理评测脚本 | SFT checkpoint、M0/M1/M2 predictions、metrics summary、error cases | JSON Validity、Schema Compliance、Audit Accuracy、High-risk Miss Rate 等指标可报告 | 稳定 SFT 模型和错误样本 |
| 08 | 完成 DPO、小规模 GRPO、reward function 和 M2/M3/M4 对比 | `src/mv_audit/training/train_dpo.py`、`train_grpo.py`、`reward_function.py`、DPO/GRPO configs | DPO checkpoint、GRPO checkpoint、reward tests、完整对比报告 | 高风险漏检和幻觉下降，Evidence Support Rate 上升，False Manual Review Rate 未明显恶化 | 完整第一版实验结果 |

## 阶段边界

- phase 00 不写训练、数据生成、渲染或评测逻辑。
- phase 01 只验证模型推理，不做 SFT、DPO 或 GRPO。
- phase 02 只生成正常结构化 case，不注入异常、不渲染图片。
- phase 03 只处理异常、规则和 split，不渲染、不训练。
- phase 04 只处理图片、bbox 和视觉扰动，不改 risk rule。
- phase 05 只转换训练数据格式，不训练模型。
- phase 06 只评测，不训练。
- phase 07 只做 baseline、LoRA-SFT、推理和评测，不做 DPO/GRPO。
- phase 08 只在 SFT 稳定输出合法 JSON 后推进 DPO/GRPO。

## 第一阶段建议

正式实现从 phase 00 开始。先建立项目骨架、依赖、目录、基础 IO/config/logging 工具和空 debug pipeline。phase 00 通过后，再进入 phase 01 验证 `Qwen3-VL-8B-Instruct` 的单图和多图推理链路。

## 待确认问题

- phase 04 的字体路径和字体授权方式需要确认。
- phase 01 默认下载源需要结合环境确认。
- 主实验规模是否严格一次扩到 30,000/3,000/8,000 cases，需要确认。
- DPO rejected 中人工手写反例是否纳入第一版执行，需要确认。
