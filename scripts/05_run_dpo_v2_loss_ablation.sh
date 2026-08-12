#!/usr/bin/env bash
set -euo pipefail

# DPO v2 loss ablation entrypoint. Defaults to dry-run so it cannot
# accidentally start a costly training sweep.

MODE="${MODE:-dry_run}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"

declare -A CONFIGS=(
  [dpo_v2_baseline]="configs/train/dpo_v2_baseline_ablation_qwen3vl_8b.yaml"
  [auxdpo_v2_strong]="configs/train/dpo_v2_auxstrong_qwen3vl_8b.yaml"
  [auxdpo_v2_stronger]="configs/train/dpo_v2_auxstronger_qwen3vl_8b.yaml"
  [ipo_v1]="configs/train/dpo_v2_ipo_qwen3vl_8b.yaml"
  [ipo_aux_v1]="configs/train/dpo_v2_ipo_aux_qwen3vl_8b.yaml"
)

VARIANTS="${VARIANTS:-dpo_v2_baseline auxdpo_v2_strong auxdpo_v2_stronger ipo_v1 ipo_aux_v1}"

for variant in $VARIANTS; do
  config="${CONFIGS[$variant]:-}"
  if [[ -z "$config" ]]; then
    echo "unknown_variant=$variant" >&2
    exit 2
  fi
  echo "variant=$variant"
  echo "config=$config"
  case "$MODE" in
    dry_run)
      DRY_RUN=1 MAX_SAMPLES="$MAX_SAMPLES" CONFIG="$config" bash scripts/05_train_dpo_v2.sh
      ;;
    train)
      CONFIG="$config" bash scripts/05_train_dpo_v2.sh
      ;;
    *)
      echo "MODE must be dry_run or train, got: $MODE" >&2
      exit 2
      ;;
  esac
done
