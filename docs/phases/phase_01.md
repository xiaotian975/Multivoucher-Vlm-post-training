# Phase 01: Qwen3-VL 基座模型推理验证

## 阶段目标

验证 `Qwen3-VL-8B-Instruct` 能在当前机器上加载、读取图片、完成单图和多图推理。此阶段只做 smoke test，不训练模型，不评价业务准确率。

## 允许修改范围

- `configs/model/`
- `src/mv_audit/inference/`
- `scripts/00_prepare_env.sh`
- `scripts/00_download_qwen3vl.sh`
- `README.md`
- `outputs/logs/`

## 禁止事项

- 不做 LoRA-SFT、DPO 或 GRPO。
- 不改数据生成、渲染、评测模块。
- 不把模型权重提交到仓库。
- 不强行安装依赖具体 CUDA 环境的 `flash-attn`，只能作为可选说明。

## 输入

- `Qwen/Qwen3-VL-8B-Instruct` 权重。
- 官方 Qwen3-VL 使用方式或 Hugging Face/ModelScope 下载方式。
- 一张测试图片。
- 两到四张测试凭证图片。

## 输出

- `configs/model/qwen3vl_8b.yaml`
- `scripts/00_prepare_env.sh`
- `scripts/00_download_qwen3vl.sh`
- `src/mv_audit/inference/qwen3vl_smoke_test.py`
- `src/mv_audit/inference/qwen3vl_multi_image_test.py`
- `outputs/logs/qwen3vl_smoke_test.log`

## 测试方式

推荐命令：

```bash
bash scripts/00_prepare_env.sh
bash scripts/00_download_qwen3vl.sh

python -m mv_audit.inference.qwen3vl_smoke_test \
  --config configs/model/qwen3vl_8b.yaml \
  --image examples/test_invoice.png \
  --output outputs/logs/qwen3vl_smoke_test.log

python -m mv_audit.inference.qwen3vl_multi_image_test \
  --config configs/model/qwen3vl_8b.yaml \
  --images examples/invoice.png examples/payment.png examples/reimbursement_form.png examples/order.png \
  --output outputs/logs/qwen3vl_multi_image_test.log
```

## 完成定义

- 模型目录不存在、图片不存在或 CUDA 不可用时有清晰错误信息。
- 单图推理能生成文本输出。
- 多图 prompt 能成功运行，不因图片输入格式失败。
- 日志记录模型路径、推理耗时、显存占用和最大可输入图片数。

## 下一阶段依赖

phase 02 不依赖模型权重，但后续 phase 07 和 phase 08 依赖本阶段确认的模型加载方式、processor 使用方式和多图输入格式。

## 待确认问题

- 默认下载源使用 Hugging Face 还是 ModelScope，需要根据机器网络环境确认。
