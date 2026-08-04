#!/usr/bin/env bash
set -euo pipefail

# Main-scale phase 05 conversion from MV-Train only.
# This script does not train models, evaluate predictions, or read held-out Val/Test cases.

python -m mv_audit.converters.build_sft_data \
  --cases data/mv_audit/raw_cases/main/train_cases.jsonl \
  --annotations data/mv_audit/annotations_main/field_bboxes_train.jsonl \
  --output_schema configs/schema/output_schema.json \
  --train_output data/mv_audit/sft_main/train.jsonl \
  --val_output data/mv_audit/sft_main/val.jsonl

python -m mv_audit.converters.build_dpo_pairs \
  --cases data/mv_audit/raw_cases/main/train_cases.jsonl \
  --annotations data/mv_audit/annotations_main/field_bboxes_train.jsonl \
  --output_schema configs/schema/output_schema.json \
  --output data/mv_audit/dpo_main/pairs_train.jsonl

python -m mv_audit.converters.build_grpo_prompts \
  --cases data/mv_audit/raw_cases/main/train_cases.jsonl \
  --annotations data/mv_audit/annotations_main/field_bboxes_train.jsonl \
  --output_schema configs/schema/output_schema.json \
  --output data/mv_audit/grpo_main/prompts_train.jsonl
