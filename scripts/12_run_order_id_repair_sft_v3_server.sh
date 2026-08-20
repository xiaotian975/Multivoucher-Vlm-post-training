#!/usr/bin/env bash
set -euo pipefail

# Order-ID Structured Repair SFT v3.
# Default path is dry-run only. Formal training requires ALLOW_TRAINING=1.

CONFIG="${CONFIG:-configs/train/high_risk_repair_sft_v3_order_id_structured_from_r2_qwen3vl_8b_server.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"

export PYTHONPATH="${PYTHONPATH:-src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"

python -m mv_audit.training.train_sft \
  --config "$CONFIG" \
  --dry_run \
  --max_samples "$MAX_SAMPLES"

if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  echo "repair_sft_v3_dry_run_only=ok"
  exit 0
fi

if [[ "${ALLOW_TRAINING:-0}" != "1" ]]; then
  echo "Repair SFT v3 training blocked: set ALLOW_TRAINING=1 after dry-run approval." >&2
  exit 22
fi

if [[ "${SFT_DDP:-1}" == "1" ]]; then
  if ! torchrun --standalone --nproc_per_node=5 \
    -m mv_audit.training.train_sft \
    --config "$CONFIG"; then
    echo "repair_sft_v3_ddp_failed=1; retrying with five-GPU model sharding" >&2
    CUDA_VISIBLE_DEVICES="0,1,2,3,4" python -m mv_audit.training.train_sft \
      --config "$CONFIG"
  fi
else
  python -m mv_audit.training.train_sft \
    --config "$CONFIG"
fi
