# Phase 00: 项目初始化与工程规范

## 阶段目标

建立 MultiVoucher-Audit 的 Python 项目骨架、目录规范、基础依赖、配置入口、日志入口和最小 debug pipeline。此阶段只铺设工程地基，不实现真实数据生成、图片渲染、评测或训练逻辑。

## 允许修改范围

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
- `configs/`
- `data/`
- `src/mv_audit/`
- `scripts/`
- `outputs/`
- `notebooks/`

## 禁止事项

- 不下载模型权重。
- 不实现训练逻辑。
- 不生成真实大规模数据。
- 不渲染凭证图片。
- 不写评测指标。
- 不修改尚未定义的 schema 或 risk rule 细节。

## 输入

- 项目总方案。
- 本文档集。
- 预设目录结构。

## 输出

应建立这些目录：

```text
configs/data_gen
configs/train
configs/eval
configs/schema
data/mv_audit/dictionaries
data/mv_audit/templates
data/mv_audit/raw_cases
data/mv_audit/images
data/mv_audit/annotations
data/mv_audit/sft
data/mv_audit/dpo
data/mv_audit/grpo
data/mv_audit/eval_sets
src/mv_audit/data_gen
src/mv_audit/rendering
src/mv_audit/perturbation
src/mv_audit/converters
src/mv_audit/training
src/mv_audit/inference
src/mv_audit/evaluation
src/mv_audit/utils
scripts
outputs/checkpoints
outputs/predictions
outputs/eval_reports
outputs/logs
notebooks
```

应创建基础文件：

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `.gitignore`
- `src/mv_audit/__init__.py`
- 各子模块 `__init__.py`
- `src/mv_audit/utils/io_utils.py`
- `src/mv_audit/utils/config_utils.py`
- `src/mv_audit/utils/logging_utils.py`
- `scripts/run_debug_pipeline.sh`

## 测试方式

- 运行 `pip install -e .`。
- 运行 `python -c "import mv_audit"`。
- 用基础 utils 读写一条 JSONL 和一个 YAML。
- 运行 `bash scripts/run_debug_pipeline.sh`，确认不会因为缺少目录而失败。

## 完成定义

- 项目可以被 editable install。
- `mv_audit` 包可以导入。
- 基础工具函数具备真实读写能力，不是空壳。
- debug pipeline 只包含占位命令和 TODO，不调用不存在的训练逻辑。

## 下一阶段依赖

phase 01 依赖 phase 00 提供的 Python 包结构、`configs/model/` 落点、`src/mv_audit/inference/` 落点、日志目录和 `.gitignore`。
