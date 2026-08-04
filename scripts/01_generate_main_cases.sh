#!/usr/bin/env bash
set -euo pipefail

# Main-scale structured case generation.
# This script does not render images, build training data, evaluate predictions, or train models.

python -m mv_audit.data_gen.generate_base_cases \
  --config configs/data_gen/main.yaml \
  --schema configs/schema/case_schema.json \
  --output data/mv_audit/raw_cases/main/base_cases_main.jsonl \
  --num_cases 41000 \
  --split_name main

python -m mv_audit.data_gen.anomaly_injector \
  --config configs/data_gen/main.yaml \
  --schema configs/schema/case_schema.json \
  --input data/mv_audit/raw_cases/main/base_cases_main.jsonl \
  --output data/mv_audit/raw_cases/main/all_cases_with_anomaly_main.jsonl \
  --stats_output data/mv_audit/raw_cases/main/anomaly_stats_main.json

python -m mv_audit.data_gen.split_builder \
  --config configs/data_gen/main.yaml \
  --schema configs/schema/case_schema.json \
  --input data/mv_audit/raw_cases/main/all_cases_with_anomaly_main.jsonl \
  --output_dir data/mv_audit/raw_cases/main \
  --stats_output data/mv_audit/raw_cases/main/split_stats_main.json
