#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/train/rdvla_mikasa10_baseline.yaml}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-8}"

if (( EFFECTIVE_BATCH_SIZE % BATCH_SIZE != 0 )); then
  echo "EFFECTIVE_BATCH_SIZE must be divisible by BATCH_SIZE" >&2
  exit 2
fi

GRAD_ACCUMULATION_STEPS=$((EFFECTIVE_BATCH_SIZE / BATCH_SIZE))

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec .venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  run.py --config "$CONFIG" --mode train \
  --batch_size="$BATCH_SIZE" \
  --grad_accumulation_steps="$GRAD_ACCUMULATION_STEPS" \
  --max_wall_time_hours=10
