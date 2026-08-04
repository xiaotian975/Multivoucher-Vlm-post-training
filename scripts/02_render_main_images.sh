#!/usr/bin/env bash
set -euo pipefail

# Main-scale phase 04 rendering. This script does not build training data or train models.

python -m mv_audit.rendering.render_all \
  --all_splits \
  --raw_cases_dir data/mv_audit/raw_cases/main \
  --images_dir data/mv_audit/images_main \
  --annotations_dir data/mv_audit/annotations_main

python scripts/visualize_bbox.py \
  --annotations data/mv_audit/annotations_main/field_bboxes_*.jsonl \
  --output_dir outputs/eval_reports/figures/bbox_samples_main \
  --sample_count 50
