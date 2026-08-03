# MultiVoucher-Audit

MultiVoucher-Audit is a post-training research project for multi-image enterprise reimbursement audit with `Qwen3-VL-8B-Instruct`.

The project is not a single-invoice OCR demo. It targets a full audit case containing invoice, payment screenshot, reimbursement form, and order screenshot images. The model is expected to extract fields, compare information across images, identify anomalies, predict risk, produce an audit recommendation, and ground the decision in Evidence-Grounded JSON with value and bbox evidence.

## Current Status

This repository is at phase 00: project initialization and engineering conventions.

Implemented in this phase:

- Python package skeleton under `src/mv_audit`.
- Standard project directories for configs, data, scripts, notebooks, and outputs.
- Minimal utility modules for JSONL, YAML, directory creation, random seeds, and logging.
- A placeholder debug pipeline that validates the skeleton without running later-stage logic.

Not implemented in this phase:

- Model download or inference.
- Data generation.
- Image rendering or bbox annotation.
- SFT/DPO/GRPO data conversion.
- Evaluation metrics.
- Training code.

## Phase Plan

The canonical roadmap is documented in `docs/execution_roadmap.md`.

Phase 00 to phase 08 are the engineering stages:

| Phase | Goal |
| --- | --- |
| 00 | Project skeleton, environment, directories, basic utilities |
| 01 | Qwen3-VL base model download/loading and single/multi-image smoke tests |
| 02 | Case schema, dictionaries, and normal transaction truth table generation |
| 03 | Anomaly injection, risk rule engine, and case-level data split |
| 04 | Voucher image rendering, bbox recording, and visual perturbation |
| 05 | SFT/DPO/GRPO data format conversion |
| 06 | JSON parser, bbox evaluator, and base evaluation |
| 07 | LoRA-SFT, inference, and M0/M1/M2 baseline evaluation |
| 08 | DPO, small-scale GRPO, reward function, and M2/M3/M4 comparison |

## Minimal Debug Check

After installing the project in editable mode, run:

```bash
pip install -e .
bash scripts/run_debug_pipeline.sh
```

On Windows PowerShell without Bash, run:

```powershell
pip install -e .
.\scripts\run_debug_pipeline.ps1
```

The debug pipeline only validates the package import and utility functions. It intentionally does not call model, data generation, rendering, evaluation, or training code.

## Phase 01: Qwen3-VL Smoke Tests

Phase 01 verifies that `Qwen/Qwen3-VL-8B-Instruct` can be downloaded or loaded locally, can read one image, and can accept two to four images in a single prompt. It does not train or evaluate model quality.

Prepare dependencies and download the model:

```bash
bash scripts/00_prepare_env.sh
bash scripts/00_download_qwen3vl.sh
```

Use ModelScope instead of Hugging Face when needed:

```bash
USE_MODELSCOPE=1 bash scripts/00_download_qwen3vl.sh
```

Run single-image smoke test:

```bash
python -m mv_audit.inference.qwen3vl_smoke_test \
  --config configs/model/qwen3vl_8b.yaml \
  --image examples/test_invoice.png \
  --output outputs/logs/qwen3vl_smoke_test.log
```

Run multi-image smoke test:

```bash
python -m mv_audit.inference.qwen3vl_multi_image_test \
  --config configs/model/qwen3vl_8b.yaml \
  --images examples/invoice.png examples/payment.png examples/reimbursement_form.png examples/order.png \
  --output outputs/logs/qwen3vl_multi_image_test.log
```

The smoke-test output file is a JSON run record containing the model path, image list, prompt, raw output, elapsed time, CUDA availability, device count, and peak allocated CUDA memory when available.

## Important Contracts

Global schema, risk, bbox, and data leakage constraints live in `docs/global_contracts.md`.

Key phase 00 constraints:

- Do not download model weights.
- Do not implement training logic.
- Do not generate real large-scale data.
- Do not render voucher images.
- Do not implement evaluation metrics.
- Do not define future schema or risk-rule internals ahead of their phase.
