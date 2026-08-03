#!/usr/bin/env bash
set -euo pipefail

# Download Qwen3-VL-8B-Instruct into a local models/ directory.
# Set USE_MODELSCOPE=1 to use ModelScope; otherwise Hugging Face is used.

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_SCOPE_ID="${MODEL_SCOPE_ID:-Qwen/Qwen3-VL-8B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-models/Qwen3-VL-8B-Instruct}"

mkdir -p "${LOCAL_DIR}"

if [[ "${USE_MODELSCOPE:-0}" == "1" ]]; then
  python - <<PY
from modelscope import snapshot_download

model_id = "${MODEL_SCOPE_ID}"
local_dir = "${LOCAL_DIR}"
print(f"Downloading {model_id} from ModelScope to {local_dir}")
snapshot_download(model_id, local_dir=local_dir)
PY
else
  python - <<PY
from huggingface_hub import snapshot_download

model_id = "${MODEL_ID}"
local_dir = "${LOCAL_DIR}"
print(f"Downloading {model_id} from Hugging Face to {local_dir}")
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
PY
fi

echo "Model files are available at ${LOCAL_DIR}"
