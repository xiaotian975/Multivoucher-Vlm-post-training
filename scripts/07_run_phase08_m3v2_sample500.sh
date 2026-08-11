#!/usr/bin/env bash
set -euo pipefail

# Phase08 M3v2 fixed-config sample500 inference/evaluation entrypoint.
# This is intentionally separate from M3 v1 outputs.

CONFIG="${CONFIG:-configs/train/phase08_m3v2_sample500_server.yaml}"
MODEL_ID="${MODEL_ID:-m3v2_dpo}"
SPLITS="${SPLITS:-test_clean test_robust test_unseen_template test_hard_negative}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"

for split in $SPLITS; do
  args=(--config "$CONFIG" --model_id "$MODEL_ID" --split "$split")
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if [[ "$RESUME" == "1" ]]; then
    args+=(--resume)
  fi
  python -m mv_audit.inference.batch_inference "${args[@]}"
done

MODELS="$MODEL_ID" \
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-data/mv_audit/eval_sets_phase07_sample500}" \
PREDICTIONS_DIR="${PREDICTIONS_DIR:-outputs/predictions/phase08_m3v2_sample500}" \
REPORT_DIR="${REPORT_DIR:-outputs/eval_reports/phase08_m3v2_sample500}" \
bash scripts/08_evaluate.sh
