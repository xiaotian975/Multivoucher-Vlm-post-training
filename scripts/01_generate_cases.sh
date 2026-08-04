#!/usr/bin/env bash
set -euo pipefail

# Phase 03 convenience pipeline for structured case data.
# It stops after anomaly injection and case-level splitting.

python -m mv_audit.data_gen.generate_base_cases \
  --config configs/data_gen/debug.yaml \
  --schema configs/schema/case_schema.json \
  --output data/mv_audit/raw_cases/base_cases_debug.jsonl \
  --num_cases 10000 \
  --split_name debug

python -m mv_audit.data_gen.anomaly_injector \
  --config configs/data_gen/debug.yaml \
  --schema configs/schema/case_schema.json \
  --input data/mv_audit/raw_cases/base_cases_debug.jsonl \
  --output data/mv_audit/raw_cases/all_cases_with_anomaly_debug.jsonl \
  --stats_output data/mv_audit/raw_cases/anomaly_stats_debug.json

python -m mv_audit.data_gen.split_builder \
  --config configs/data_gen/debug.yaml \
  --schema configs/schema/case_schema.json \
  --input data/mv_audit/raw_cases/all_cases_with_anomaly_debug.jsonl \
  --output_dir data/mv_audit/raw_cases \
  --stats_output data/mv_audit/raw_cases/split_stats_debug.json
