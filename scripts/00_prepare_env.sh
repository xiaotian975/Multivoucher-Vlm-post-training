#!/usr/bin/env bash
set -euo pipefail

# Phase 01 environment preparation for Qwen3-VL smoke tests.
# This script intentionally does not install flash-attn because flash-attn is
# CUDA, compiler, and PyTorch-version specific. Install it manually only after
# confirming the server CUDA/PyTorch environment.

python -m pip install --upgrade \
  "transformers>=4.57.0" \
  "qwen-vl-utils==0.0.14" \
  accelerate \
  peft \
  trl \
  pillow \
  opencv-python \
  jsonschema \
  pyyaml \
  huggingface_hub \
  modelscope

echo "Phase 01 base dependencies installed."
echo "Optional: install flash-attn only if your CUDA/PyTorch environment supports it."
