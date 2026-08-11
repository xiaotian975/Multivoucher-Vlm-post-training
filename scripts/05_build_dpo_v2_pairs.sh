#!/usr/bin/env bash
set -euo pipefail

# DPO v2 conservative Train-only pair construction.
# This reads MV-Train cases and train annotations only. It does not read Val/Test
# or Phase07/08 sample500 predictions.

CASES="${CASES:-data/mv_audit/raw_cases/main/train_cases.jsonl}"
ANNOTATIONS="${ANNOTATIONS:-data/mv_audit/annotations_main/field_bboxes_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-data/mv_audit/dpo_v2}"
MAX_PAIRS="${MAX_PAIRS:-3000}"
HOLDOUT_RATIO="${HOLDOUT_RATIO:-0.10}"
DECODE_DEV_CASES="${DECODE_DEV_CASES:-200}"
SEED="${SEED:-42}"
MAX_INPUT_CASES="${MAX_INPUT_CASES:-}"

args=(
  --cases "$CASES"
  --annotations "$ANNOTATIONS"
  --output_schema configs/schema/output_schema.json
  --output_dir "$OUTPUT_DIR"
  --max_pairs "$MAX_PAIRS"
  --holdout_ratio "$HOLDOUT_RATIO"
  --decode_dev_cases "$DECODE_DEV_CASES"
  --seed "$SEED"
)
if [[ -n "$MAX_INPUT_CASES" ]]; then
  args+=(--max_input_cases "$MAX_INPUT_CASES")
fi

python -m mv_audit.converters.build_dpo_v2_pairs "${args[@]}"
