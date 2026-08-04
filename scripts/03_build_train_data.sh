#!/usr/bin/env bash
set -euo pipefail

# Phase 05: build SFT, DPO, and GRPO data formats from MV-Train only.
# This script does not train models, evaluate predictions, or read Val/Test cases.

python -m mv_audit.converters.build_sft_data \
  --cases data/mv_audit/raw_cases/train_cases.jsonl \
  --annotations data/mv_audit/annotations/field_bboxes_train.jsonl \
  --output_schema configs/schema/output_schema.json \
  --train_output data/mv_audit/sft/train.jsonl \
  --val_output data/mv_audit/sft/val.jsonl

python -m mv_audit.converters.build_dpo_pairs \
  --cases data/mv_audit/raw_cases/train_cases.jsonl \
  --annotations data/mv_audit/annotations/field_bboxes_train.jsonl \
  --output_schema configs/schema/output_schema.json \
  --output data/mv_audit/dpo/pairs_train.jsonl

python -m mv_audit.converters.build_grpo_prompts \
  --cases data/mv_audit/raw_cases/train_cases.jsonl \
  --annotations data/mv_audit/annotations/field_bboxes_train.jsonl \
  --output_schema configs/schema/output_schema.json \
  --output data/mv_audit/grpo/prompts_train.jsonl
