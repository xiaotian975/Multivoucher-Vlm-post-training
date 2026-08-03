#!/usr/bin/env bash
set -euo pipefail

# Phase 03: inject structured anomalies only.
# This script does not render images, create bbox data, convert training data, or train models.

python -m mv_audit.data_gen.anomaly_injector \
  --config configs/data_gen/debug.yaml \
  --schema configs/schema/case_schema.json \
  --input data/mv_audit/raw_cases/base_cases_debug.jsonl \
  --output data/mv_audit/raw_cases/all_cases_with_anomaly_debug.jsonl \
  --stats_output data/mv_audit/raw_cases/anomaly_stats_debug.json
