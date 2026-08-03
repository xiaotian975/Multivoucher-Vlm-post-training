#!/usr/bin/env bash
set -euo pipefail

# Phase 03: build case-level splits only.
# Train/Val/Test are split by case_id; no image rendering or training happens here.

python -m mv_audit.data_gen.split_builder \
  --config configs/data_gen/debug.yaml \
  --schema configs/schema/case_schema.json \
  --input data/mv_audit/raw_cases/all_cases_with_anomaly_debug.jsonl \
  --output_dir data/mv_audit/raw_cases \
  --stats_output data/mv_audit/raw_cases/split_stats_debug.json
