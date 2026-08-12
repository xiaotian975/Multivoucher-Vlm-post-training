#!/usr/bin/env bash
set -Eeuo pipefail

# Server-side orchestrator for the Phase08 DPO v2/AuxDPO/IPO ablation.
# It does not commit or push. A local watcher should pull the final archive and
# send the shutdown command after local validation succeeds.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_BASE="${RUN_BASE:-outputs/runtime/dpo_v2_ablation_5gpu}"
RUN_ROOT="$RUN_BASE/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
DECODE_DEV_LIMIT="${DECODE_DEV_LIMIT:-50}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-5}"
MODEL_ID="${MODEL_ID:-m3v2_dpo}"
ARCHIVE_DIR="docs/experiments/phase08_loss_ablation_$RUN_ID"
READY_FILE="$RUN_ROOT/READY_TO_ARCHIVE"
FAILED_FILE="$RUN_ROOT/FAILED"
ALLOW_FAILURE="${ALLOW_FAILURE:-0}"

VARIANTS=(dpo_v2_baseline auxdpo_v2_strong auxdpo_v2_stronger ipo_v1 ipo_aux_v1)
GPUS=(0 1 2 3 4)
SPLITS=(test_clean test_robust test_unseen_template test_hard_negative)

declare -A CONFIGS=(
  [dpo_v2_baseline]="configs/train/dpo_v2_baseline_ablation_qwen3vl_8b.yaml"
  [auxdpo_v2_strong]="configs/train/dpo_v2_auxstrong_qwen3vl_8b.yaml"
  [auxdpo_v2_stronger]="configs/train/dpo_v2_auxstronger_qwen3vl_8b.yaml"
  [ipo_v1]="configs/train/dpo_v2_ipo_qwen3vl_8b.yaml"
  [ipo_aux_v1]="configs/train/dpo_v2_ipo_aux_qwen3vl_8b.yaml"
)
declare -A ADAPTERS=(
  [dpo_v2_baseline]="outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_baseline_ablation"
  [auxdpo_v2_strong]="outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_auxstrong"
  [auxdpo_v2_stronger]="outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_auxstronger"
  [ipo_v1]="outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_ipo"
  [ipo_aux_v1]="outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_ipo_aux"
)
declare -A REPORT_DIRS=(
  [dpo_v2_baseline]="phase08_dpo_v2_baseline_ablation"
  [auxdpo_v2_strong]="phase08_dpo_v2_auxstrong"
  [auxdpo_v2_stronger]="phase08_dpo_v2_auxstronger"
  [ipo_v1]="phase08_dpo_v2_ipo"
  [ipo_aux_v1]="phase08_dpo_v2_ipo_aux"
)

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_BASE"
echo "$RUN_ID" > "$RUN_BASE/LATEST_RUN_ID"
exec > >(tee -a "$RUN_ROOT/main.log") 2>&1

fail() {
  local code=$?
  if [[ "${ALLOW_FAILURE:-0}" == "1" ]]; then
    return 0
  fi
  echo "failed_at=$(date -Is)" > "$FAILED_FILE"
  echo "exit_code=$code" >> "$FAILED_FILE"
  echo "run_root=$RUN_ROOT" >> "$FAILED_FILE"
  echo "[FAILED] exit_code=$code run_root=$RUN_ROOT" >&2
}
trap fail ERR

log() {
  echo "[$(date -Is)] $*"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing_required_path=$path" >&2
    exit 20
  fi
}

write_runtime_config() {
  local variant="$1"
  local output="$2"
  local predictions_dir="$3"
  local ground_truth_dir="$4"
  local manifest_dir="$5"
  python - "$variant" "$output" "$predictions_dir" "$ground_truth_dir" "$manifest_dir" <<'PY'
import sys
from pathlib import Path
import yaml

variant, output, predictions_dir, ground_truth_dir, manifest_dir = sys.argv[1:6]
base_path = Path("configs/train/phase08_m3v2_sample500_server.yaml")
config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
adapters = {
    "dpo_v2_baseline": "outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_baseline_ablation",
    "auxdpo_v2_strong": "outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_auxstrong",
    "auxdpo_v2_stronger": "outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_auxstronger",
    "ipo_v1": "outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_ipo",
    "ipo_aux_v1": "outputs/checkpoints/dpo/qwen3vl_8b_dpo_v2_ipo_aux",
}
config.setdefault("training", {})["output_dir"] = adapters[variant]
inference = config.setdefault("inference", {})
inference["dpo_adapter_dir"] = adapters[variant]
inference["predictions_dir"] = predictions_dir
inference["ground_truth_dir"] = ground_truth_dir
inference["sample_manifest_dir"] = manifest_dir
inference["train_decode_dev_ground_truth_dir"] = ground_truth_dir
inference["flush_every"] = 1
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
}

run_dry_run() {
  local variant="$1"
  local gpu="$2"
  log "dry_run variant=$variant gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" DRY_RUN=1 MAX_SAMPLES=8 CONFIG="${CONFIGS[$variant]}" \
    bash scripts/05_train_dpo_v2.sh > "$LOG_DIR/${variant}_dry_run.log" 2>&1
}

run_train() {
  local variant="$1"
  local gpu="$2"
  log "train_start variant=$variant gpu=$gpu"
  if [[ "$gpu" == "all" ]]; then
    if ! CUDA_VISIBLE_DEVICES="0,1,2,3,4" CONFIG="${CONFIGS[$variant]}" \
      bash scripts/05_train_dpo_v2.sh > "$LOG_DIR/${variant}_train_all_gpus.log" 2>&1; then
      log "train_failed variant=$variant gpu=$gpu"
      return 1
    fi
  else
    if ! CUDA_VISIBLE_DEVICES="$gpu" CONFIG="${CONFIGS[$variant]}" \
      bash scripts/05_train_dpo_v2.sh > "$LOG_DIR/${variant}_train.log" 2>&1; then
      log "train_failed variant=$variant gpu=$gpu"
      return 1
    fi
  fi
  log "train_done variant=$variant gpu=$gpu"
}

adapter_done() {
  local variant="$1"
  [[ -s "${ADAPTERS[$variant]}/adapter_config.json" && -s "${ADAPTERS[$variant]}/adapter_model.safetensors" ]]
}

train_all_variants() {
  local -A pids=()
  local -A gpu_by_variant=()
  set +e
  ALLOW_FAILURE=1
  for index in "${!VARIANTS[@]}"; do
    local variant="${VARIANTS[$index]}"
    local gpu="${GPUS[$index]}"
    gpu_by_variant[$variant]="$gpu"
    run_train "$variant" "$gpu" &
    pids[$variant]=$!
  done
  local failed=0
  for variant in "${VARIANTS[@]}"; do
    wait "${pids[$variant]}"
    local code=$?
    if [[ "$code" != "0" ]]; then
      echo "parallel_train_failed variant=$variant code=$code" | tee -a "$RUN_ROOT/train_fallback.log"
      failed=1
    fi
  done
  ALLOW_FAILURE=0
  set -e
  if [[ "$failed" == "1" ]]; then
    log "parallel training had failures; retrying incomplete variants sequentially with all GPUs"
    for variant in "${VARIANTS[@]}"; do
      if ! adapter_done "$variant"; then
        run_train "$variant" "all"
      fi
    done
  fi
}

run_decode_dev() {
  local variant="$1"
  local gpu="$2"
  local config="$RUN_ROOT/configs/${variant}_train_decode_dev.yaml"
  local pred_dir="outputs/predictions/phase08_loss_ablation_train_decode_dev/$variant"
  local gt_dir="outputs/eval_sets/phase08_loss_ablation_train_decode_dev/$variant"
  local report_dir="outputs/eval_reports/phase08_loss_ablation_train_decode_dev/$variant"
  write_runtime_config "$variant" "$config" "$pred_dir" "data/mv_audit/eval_sets_phase07_sample500" "data/mv_audit/eval_sets_phase07_sample500/manifests"
  python - "$variant" "$gt_dir" <<'PY'
import sys
from pathlib import Path
from mv_audit.utils import read_jsonl, write_jsonl

variant, gt_dir = sys.argv[1:3]
rows = read_jsonl("data/mv_audit/dpo_v2/train_decode_dev.jsonl")
Path(gt_dir).mkdir(parents=True, exist_ok=True)
write_jsonl(rows, Path(gt_dir) / "train_decode_dev.jsonl")
PY
  log "decode_dev_start variant=$variant gpu=$gpu limit=$DECODE_DEV_LIMIT"
  CUDA_VISIBLE_DEVICES="$gpu" CONFIG="$config" MODEL_ID="$MODEL_ID" LIMIT="$DECODE_DEV_LIMIT" RESUME=1 \
    PREDICTIONS_DIR="$pred_dir" REPORT_DIR="$report_dir" GROUND_TRUTH_DIR="$gt_dir" \
    bash scripts/07_run_phase08_m3v2_train_decode_dev.sh > "$LOG_DIR/${variant}_decode_dev.log" 2>&1
  log "decode_dev_done variant=$variant gpu=$gpu"
}

select_variant() {
  python - "$RUN_ROOT" "${VARIANTS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
variants = sys.argv[2:]
rows = []
for variant in variants:
    path = Path("outputs/eval_reports/phase08_loss_ablation_train_decode_dev") / variant / "metrics_summary.csv"
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle))
    if not data:
        continue
    row = data[0]
    rows.append(
        {
            "variant": variant,
            "audit_accuracy": float(row.get("audit_accuracy") or 0),
            "high_risk_miss_rate": float(row.get("high_risk_miss_rate") or 1),
            "evidence_support_rate": float(row.get("evidence_support_rate") or 0),
            "error_cases": float(row.get("error_cases") or 999999),
        }
    )
if not rows:
    raise SystemExit("No decode-dev metrics found.")
selected = sorted(rows, key=lambda r: (r["high_risk_miss_rate"], -r["audit_accuracy"], -r["evidence_support_rate"], r["error_cases"]))[0]
(run_root / "variant_selection.json").write_text(json.dumps({"selected": selected, "candidates": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(run_root / "selected_variant").write_text(selected["variant"] + "\n", encoding="utf-8")
print(selected["variant"])
PY
}

make_shards() {
  local variant="$1"
  python - "$RUN_ROOT" "$variant" "$SAMPLE_WORKERS" "${SPLITS[@]}" <<'PY'
import sys
from pathlib import Path
from mv_audit.utils import iter_jsonl, write_jsonl

run_root = Path(sys.argv[1])
variant = sys.argv[2]
workers = int(sys.argv[3])
splits = sys.argv[4:]
manifest_root = Path("data/mv_audit/eval_sets_phase07_sample500/manifests")
for split in splits:
    rows = list(iter_jsonl(manifest_root / f"{split}_case_ids.jsonl"))
    for shard in range(workers):
        shard_dir = run_root / "manifests" / variant / f"{split}_shard{shard}"
        shard_rows = [row for index, row in enumerate(rows) if index % workers == shard]
        write_jsonl(shard_rows, shard_dir / f"{split}_case_ids.jsonl")
PY
}

run_sample_shard() {
  local variant="$1"
  local split="$2"
  local shard="$3"
  local gpu="$4"
  local config="$RUN_ROOT/configs/${variant}_${split}_shard${shard}.yaml"
  local pred_dir="outputs/predictions/phase08_loss_ablation_sample500_shards/$variant/${split}_shard${shard}"
  local gt_dir="outputs/eval_sets/phase08_loss_ablation_sample500_shards/$variant/${split}_shard${shard}"
  local manifest_dir="$RUN_ROOT/manifests/$variant/${split}_shard${shard}"
  write_runtime_config "$variant" "$config" "$pred_dir" "$gt_dir" "$manifest_dir"
  log "sample_shard_start variant=$variant split=$split shard=$shard gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -m mv_audit.inference.batch_inference \
    --config "$config" --model_id "$MODEL_ID" --split "$split" --resume \
    > "$LOG_DIR/${variant}_${split}_shard${shard}.log" 2>&1
  log "sample_shard_done variant=$variant split=$split shard=$shard gpu=$gpu"
}

run_sample500() {
  local variant="$1"
  make_shards "$variant"
  local tasks=()
  for split in "${SPLITS[@]}"; do
    for shard in $(seq 0 $((SAMPLE_WORKERS - 1))); do
      tasks+=("$split:$shard")
    done
  done
  local next=0
  while [[ "$next" -lt "${#tasks[@]}" ]]; do
    local -A pids=()
    for gpu in $(seq 0 $((SAMPLE_WORKERS - 1))); do
      if [[ "$next" -ge "${#tasks[@]}" ]]; then
        break
      fi
      IFS=: read -r split shard <<< "${tasks[$next]}"
      run_sample_shard "$variant" "$split" "$shard" "$gpu" &
      pids[$gpu]=$!
      next=$((next + 1))
    done
    for gpu in "${!pids[@]}"; do
      wait "${pids[$gpu]}"
    done
  done
}

merge_sample500() {
  local variant="$1"
  python - "$variant" "$SAMPLE_WORKERS" "$RUN_ROOT" "${SPLITS[@]}" <<'PY'
import json
import sys
from pathlib import Path
from mv_audit.utils import iter_jsonl, write_jsonl

variant = sys.argv[1]
workers = int(sys.argv[2])
run_root = Path(sys.argv[3])
splits = sys.argv[4:]
model_id = "m3v2_dpo"
manifest_root = Path("data/mv_audit/eval_sets_phase07_sample500/manifests")
final_root = Path("outputs/predictions/phase08_loss_ablation_sample500") / variant / model_id
summary = {}
for split in splits:
    expected = [str(row["case_id"]) for row in iter_jsonl(manifest_root / f"{split}_case_ids.jsonl")]
    predictions = {}
    duplicates = 0
    for shard in range(workers):
        path = Path("outputs/predictions/phase08_loss_ablation_sample500_shards") / variant / f"{split}_shard{shard}" / model_id / f"{split}.jsonl"
        for row in iter_jsonl(path):
            case_id = str(row["case_id"])
            if case_id in predictions:
                duplicates += 1
            predictions[case_id] = row
    missing = [case_id for case_id in expected if case_id not in predictions]
    extra = sorted(set(predictions) - set(expected))
    if missing or extra or duplicates:
        raise SystemExit(f"merge_failed split={split} missing={len(missing)} extra={len(extra)} duplicates={duplicates}")
    ordered = [predictions[case_id] for case_id in expected]
    write_jsonl(ordered, final_root / f"{split}.jsonl")
    summary[split] = {"expected": len(expected), "written": len(ordered), "missing": 0, "extra": 0, "duplicates_seen_before_dedup": duplicates}
(run_root / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
}

evaluate_sample500() {
  local variant="$1"
  MODELS="$MODEL_ID" \
  GROUND_TRUTH_DIR="data/mv_audit/eval_sets_phase07_sample500" \
  PREDICTIONS_DIR="outputs/predictions/phase08_loss_ablation_sample500/$variant" \
  REPORT_DIR="outputs/eval_reports/phase08_loss_ablation_sample500/$variant" \
  bash scripts/08_evaluate.sh
}

analyze_migration() {
  local variant="$1"
  local out_dir="outputs/eval_reports/phase08_loss_ablation_error_migration/$variant"
  mkdir -p "$out_dir"
  for split in "${SPLITS[@]}"; do
    local baseline="outputs/predictions/phase07_sample500/m2_sft/${split}.jsonl"
    local candidate="outputs/predictions/phase08_loss_ablation_sample500/$variant/$MODEL_ID/${split}.jsonl"
    if [[ -s "$baseline" && -s "$candidate" ]]; then
      GROUND_TRUTH="data/mv_audit/eval_sets_phase07_sample500/${split}.jsonl" \
      BASELINE_PREDICTIONS="$baseline" \
      CANDIDATE_PREDICTIONS="$candidate" \
      OUTPUT_CSV="$out_dir/${split}_case_transitions.csv" \
      SUMMARY_OUTPUT="$out_dir/${split}_transition_summary.json" \
      bash scripts/09_analyze_dpo_error_migration.sh > "$LOG_DIR/${variant}_${split}_migration.log" 2>&1
    else
      echo "skip_migration split=$split missing_baseline_or_candidate" | tee -a "$RUN_ROOT/migration_skipped.log"
    fi
  done
}

package_archive() {
  local variant="$1"
  local report_dirs_arg=""
  for item in "${VARIANTS[@]}"; do
    report_dirs_arg+="$item=${REPORT_DIRS[$item]} "
  done
  python -m mv_audit.analysis.archive_loss_ablation \
    --project_root . \
    --run_id "$RUN_ID" \
    --run_root "$RUN_ROOT" \
    --archive_dir "$ARCHIVE_DIR" \
    --selected_variant "$variant" \
    --variants "${VARIANTS[*]}" \
    --report_dirs "$report_dirs_arg"
  tar -czf "$ARCHIVE_DIR.tar.gz" "$ARCHIVE_DIR"
  echo "$ARCHIVE_DIR.tar.gz" > "$RUN_ROOT/archive_tar_path"
}

main() {
  cd "$PROJECT_ROOT"
  log "run_id=$RUN_ID"
  log "project_root=$PROJECT_ROOT"
  nvidia-smi || true
  require_path "models/Qwen3-VL-8B-Instruct"
  require_path "outputs/checkpoints/sft/qwen3vl_8b_lora_existing_epoch1/adapter_config.json"
  require_path "data/mv_audit/raw_cases/main/train_cases.jsonl"
  require_path "data/mv_audit/annotations_main/field_bboxes_train.jsonl"
  require_path "data/mv_audit/eval_sets_phase07_sample500/manifests/test_clean_case_ids.jsonl"
  require_path "data/mv_audit/raw_cases/main/test_clean_cases.jsonl"
  require_path "data/mv_audit/annotations_main/field_bboxes_test_clean.jsonl"
  for variant in "${VARIANTS[@]}"; do
    require_path "${CONFIGS[$variant]}"
  done
  python -m compileall src/mv_audit tests
  if command -v pytest >/dev/null 2>&1; then
    PYTHONPATH=src pytest -q tests/test_dpo_loss_types.py
  else
    log "pytest not found; skipping unit test"
  fi
  if [[ ! -s "data/mv_audit/dpo_v2/pairs_train.jsonl" || ! -s "data/mv_audit/dpo_v2/pairs_holdout.jsonl" || ! -s "data/mv_audit/dpo_v2/train_decode_dev.jsonl" ]]; then
    log "building_train_only_dpo_v2_pairs"
    MAX_PAIRS="${MAX_PAIRS:-3000}" DECODE_DEV_CASES="${DECODE_DEV_CASES:-200}" \
      bash scripts/05_build_dpo_v2_pairs.sh > "$LOG_DIR/build_dpo_v2_pairs.log" 2>&1
  fi
  require_path "data/mv_audit/dpo_v2/pairs_train.jsonl"
  require_path "data/mv_audit/dpo_v2/pairs_holdout.jsonl"
  require_path "data/mv_audit/dpo_v2/train_decode_dev.jsonl"
  for index in "${!VARIANTS[@]}"; do
    run_dry_run "${VARIANTS[$index]}" "${GPUS[$index]}"
  done
  train_all_variants
  for index in "${!VARIANTS[@]}"; do
    run_decode_dev "${VARIANTS[$index]}" "${GPUS[$index]}" &
  done
  wait
  local selected
  selected="$(select_variant)"
  log "selected_variant=$selected"
  run_sample500 "$selected"
  merge_sample500 "$selected"
  evaluate_sample500 "$selected"
  analyze_migration "$selected"
  package_archive "$selected"
  echo "ready_at=$(date -Is)" > "$READY_FILE"
  echo "run_id=$RUN_ID" >> "$READY_FILE"
  echo "archive=$ARCHIVE_DIR.tar.gz" >> "$READY_FILE"
  log "READY_TO_ARCHIVE archive=$ARCHIVE_DIR.tar.gz"
}

main "$@"
