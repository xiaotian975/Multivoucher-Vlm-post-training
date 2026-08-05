#!/usr/bin/env bash
set -euo pipefail

# Phase 07: run M0/M1/M2 inference on held-out test splits.

CONFIG="${CONFIG:-configs/train/sft_lora_qwen3vl_8b.yaml}"
LIMIT="${LIMIT:-}"
MODELS="${MODELS:-m0_zero_shot m1_few_shot m2_sft}"
SPLITS="${SPLITS:-test_clean test_robust test_unseen_template test_hard_negative}"

for model_id in $MODELS; do
  for split in $SPLITS; do
    args=(--config "$CONFIG" --model_id "$model_id" --split "$split")
    if [[ -n "$LIMIT" ]]; then
      args+=(--limit "$LIMIT")
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      args+=(--dry_run)
    fi
    python -m mv_audit.inference.batch_inference "${args[@]}"
  done
done
