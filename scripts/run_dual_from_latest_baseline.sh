#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

baseline_root="outputs/mikasa10_baseline"
latest_state="$(find "$baseline_root" -maxdepth 1 -type f -name 'trainer_state--*_checkpoint.pt' -printf '%f\n' \
  | sed -n 's/^trainer_state--\([0-9][0-9]*\)_checkpoint\.pt$/\1 &/p' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -n "$latest_state" ]]; then
  checkpoint="$baseline_root"
  step="${latest_state#trainer_state--}"
  step="${step%_checkpoint.pt}"
else
  checkpoint="$(find "$baseline_root" -maxdepth 1 -type d -name '*--*_chkpt' -printf '%p\n' \
  | sed -n 's/.*--\([0-9][0-9]*\)_chkpt$/\1 &/p' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  step="${checkpoint##*--}"
  step="${step%_chkpt}"
fi
if [[ -z "$checkpoint" ]]; then
  echo "No baseline checkpoint found under $baseline_root" >&2
  exit 1
fi

if [[ ! "$step" =~ ^[0-9]+$ ]]; then
  echo "Could not parse checkpoint step from $checkpoint" >&2
  exit 2
fi

echo "Starting memory-only stage from $checkpoint (step $step)"
BATCH_SIZE="${BATCH_SIZE:-24}" EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-24}" \
  bash scripts/run_mikasa_10h.sh configs/train/rdvla_mikasa10_dual.yaml \
    --model.config_path="$checkpoint" \
    --resume_path="$checkpoint" \
    --resume_step="$step" \
    --restore_trainer_state=false \
    "$@"
