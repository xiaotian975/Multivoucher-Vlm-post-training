#!/usr/bin/env bash
set -euo pipefail

# Train-only decode-dev proxy for M3v2. Use LIMIT=1 for smoke tests.

CONFIG="${CONFIG:-configs/train/phase08_m3v2_sample500_server.yaml}"
MODEL_ID="${MODEL_ID:-m3v2_dpo}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"
REPORT_DIR="${REPORT_DIR:-outputs/eval_reports/phase08_m3v2_train_decode_dev}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-outputs/predictions/phase08_m3v2_sample500}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-outputs/eval_sets/phase08_m3v2_train_decode_dev}"

args=(--config "$CONFIG" --model_id "$MODEL_ID" --split train_decode_dev)
if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
  args+=(--resume)
fi
python -m mv_audit.inference.batch_inference "${args[@]}"

MODELS="$MODEL_ID" \
SPLITS="train_decode_dev" \
GROUND_TRUTH_DIR="$GROUND_TRUTH_DIR" \
PREDICTIONS_DIR="$PREDICTIONS_DIR" \
REPORT_DIR="$REPORT_DIR" \
bash scripts/08_evaluate.sh
