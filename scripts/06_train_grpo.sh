#!/usr/bin/env bash
set -euo pipefail

# Phase 08: GRPO entrypoint. Set DRY_RUN=1 to validate config/data and reward only.

CONFIG="${CONFIG:-configs/train/grpo_qwen3vl_8b.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

args=(--config "$CONFIG")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run --max_samples "${MAX_SAMPLES:-2}")
elif [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max_samples "$MAX_SAMPLES")
fi

python -m mv_audit.training.train_grpo "${args[@]}"
