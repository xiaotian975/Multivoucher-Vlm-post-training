#!/usr/bin/env bash
set -euo pipefail

# Run the locked final_holdout_v1 exactly once for repair_sft_r3.
# This is intentionally guarded because the result must not be used for further
# training, reward tuning, or model selection retries.

CONFIG="${CONFIG:-configs/train/repair_sft_r3_final_holdout.yaml}"
MODEL_ID="${MODEL_ID:-repair_sft_r3}"
SPLITS="${SPLITS:-test_clean test_robust test_unseen_template test_hard_negative}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-outputs/eval_sets/final_holdout_v1}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-outputs/predictions/final_holdout_v1}"
REPORT_DIR="${REPORT_DIR:-outputs/eval_reports/final_holdout_v1/repair_sft_r3}"
LOCK_ROOT="${LOCK_ROOT:-data/mv_audit/final_holdout_v1}"
NUM_SHARDS="${NUM_SHARDS:-}"

if [[ "${ALLOW_FINAL_HOLDOUT:-}" != "YES_I_UNDERSTAND" ]]; then
  echo "Final holdout blocked: set ALLOW_FINAL_HOLDOUT=YES_I_UNDERSTAND to run the one-time final evaluation." >&2
  exit 2
fi

if [[ "$MODEL_ID" != "repair_sft_r3" ]]; then
  echo "Final holdout v1 is locked to MODEL_ID=repair_sft_r3, got ${MODEL_ID}." >&2
  exit 2
fi

if [[ ! -s "$LOCK_ROOT/FINAL_HOLDOUT_LOCKED" ]]; then
  echo "Missing $LOCK_ROOT/FINAL_HOLDOUT_LOCKED. Build the holdout first." >&2
  exit 2
fi

if [[ ! -s "$LOCK_ROOT/FINAL_MODEL_LOCK" ]]; then
  echo "Missing $LOCK_ROOT/FINAL_MODEL_LOCK. Build the holdout/model lock first." >&2
  exit 2
fi

if [[ -e "$LOCK_ROOT/FINAL_HOLDOUT_CONSUMED" && "${RESUME:-0}" != "1" ]]; then
  echo "Final holdout is already consumed. Set RESUME=1 only for an interrupted identical run." >&2
  exit 2
fi

if [[ -n "$NUM_SHARDS" ]]; then
  for split in $SPLITS; do
    for shard in $(seq 0 "$((NUM_SHARDS - 1))"); do
      python -m mv_audit.inference.batch_inference \
        --config "$CONFIG" \
        --model_id "$MODEL_ID" \
        --split "$split" \
        --shard_index "$shard" \
        --num_shards "$NUM_SHARDS" \
        --resume
    done
    python tools/merge_inference_shards.py \
      --config "$CONFIG" \
      --model_id "$MODEL_ID" \
      --split "$split" \
      --num_shards "$NUM_SHARDS"
  done
else
  CONFIG="$CONFIG" MODELS="$MODEL_ID" SPLITS="$SPLITS" RESUME=1 bash scripts/07_run_inference.sh
fi

GROUND_TRUTH_DIR="$GROUND_TRUTH_DIR" \
PREDICTIONS_DIR="$PREDICTIONS_DIR" \
REPORT_DIR="$REPORT_DIR" \
MODELS="$MODEL_ID" \
SPLITS="$SPLITS" \
bash scripts/08_evaluate.sh

python tools/summarize_final_holdout.py \
  --manifest "$LOCK_ROOT/final_holdout_v1_manifest.json" \
  --metrics_summary "$REPORT_DIR/metrics_summary.csv" \
  --errors_dir "$REPORT_DIR" \
  --output_dir "docs/experiments/final_holdout_v1" \
  --consume_marker "$LOCK_ROOT/FINAL_HOLDOUT_CONSUMED"
