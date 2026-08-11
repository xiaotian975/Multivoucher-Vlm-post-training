#!/usr/bin/env bash
set -euo pipefail

# DPO v2 entrypoint. Set DRY_RUN=1 for local validation without loading the VLM.

CONFIG="${CONFIG:-configs/train/dpo_v2_qwen3vl_8b.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

args=(--config "$CONFIG")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run --max_samples "${MAX_SAMPLES:-4}")
elif [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max_samples "$MAX_SAMPLES")
fi

python -m mv_audit.training.train_dpo "${args[@]}"
