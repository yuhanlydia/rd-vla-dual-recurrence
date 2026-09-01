#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for batch_size in 64 32 16 8 4 2 1; do
  echo "Probing batch_size=${batch_size}"
  if TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 \
      run.py --config configs/train/rdvla_mikasa10_baseline.yaml --mode train \
      --batch_size="$batch_size" --grad_accumulation_steps=1 \
      --max_steps=1 --save_freq=999999 --merge_lora=false \
      --output_dir="outputs/batch_probe_${batch_size}"; then
    echo "Largest passing batch size: ${batch_size}"
    exit 0
  fi
  echo "batch_size=${batch_size} failed; trying the next size" >&2
done

echo "No batch size passed" >&2
exit 1
