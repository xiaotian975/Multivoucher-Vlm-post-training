#!/usr/bin/env bash
set -euo pipefail

# Phase 07: LoRA-SFT entrypoint. Set DRY_RUN=1 to validate config/data only.

CONFIG="${CONFIG:-configs/train/sft_lora_qwen3vl_8b.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

args=(--config "$CONFIG")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run --max_samples "${MAX_SAMPLES:-2}")
elif [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max_samples "$MAX_SAMPLES")
fi

python -m mv_audit.training.train_sft "${args[@]}"
