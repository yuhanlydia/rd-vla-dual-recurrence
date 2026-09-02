#!/usr/bin/env bash
set -euo pipefail

# Keep a dual-recurrence run alive until an absolute UTC deadline.  The
# currently running stage is never interrupted; this watcher only takes over
# after WATCH_PID exits, then resumes from the newest validated checkpoint.

cd "$(dirname "$0")/.."
WATCH_PID="${1:?usage: $0 WATCH_PID DEADLINE_EPOCH [CONFIG] [OUTPUT_DIR]}"
DEADLINE_EPOCH="${2:?usage: $0 WATCH_PID DEADLINE_EPOCH [CONFIG] [OUTPUT_DIR]}"
CONFIG="${3:-configs/train/rdvla_mikasa10_dual.yaml}"
OUTPUT_DIR="${4:-outputs/mikasa10_dual}"
FALLBACK_DIR="${FALLBACK_DIR:-}"
FALLBACK_STEP="${FALLBACK_STEP:-}"

while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 60
done

while (( $(date +%s) < DEADLINE_EPOCH )); do
  latest_state_path="$(find "$OUTPUT_DIR" -maxdepth 3 -type f \
    -name 'trainer_state--*_checkpoint.pt' -printf '%p\n' \
    | sed -n 's/.*\/trainer_state--\([0-9][0-9]*\)_checkpoint\.pt$/\1 &/p' \
    | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ -z "$latest_state_path" ]]; then
    if [[ -n "$FALLBACK_DIR" && -n "$FALLBACK_STEP" && \
      -f "$FALLBACK_DIR/trainer_state--${FALLBACK_STEP}_checkpoint.pt" ]]; then
      latest_state_path="$FALLBACK_DIR/trainer_state--${FALLBACK_STEP}_checkpoint.pt"
      echo "No checkpoint in $OUTPUT_DIR; using fallback step $FALLBACK_STEP from $FALLBACK_DIR" >&2
    else
      echo "No resumable dual checkpoint before deadline" >&2
      exit 1
    fi
  fi

  latest_state="${latest_state_path##*/}"
  step="${latest_state#trainer_state--}"
  step="${step%_checkpoint.pt}"
  .venv/bin/python - "$latest_state_path" "$step" <<'PY'
import sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
expected = int(sys.argv[2])
state = torch.load(path, map_location="cpu", weights_only=False)
required = {"step", "optimizers", "scheduler", "cuda_rng", "numpy_rng", "python_rng", "torch_rng"}
missing = required.difference(state)
if missing or int(state["step"]) != expected or not state["optimizers"]:
    raise SystemExit(f"invalid checkpoint {path}: missing={sorted(missing)} step={state.get('step')}")
print(f"Validated takeover checkpoint step={expected}")
PY

  remaining="$((DEADLINE_EPOCH - $(date +%s)))"
  hours="$(awk -v s="$remaining" 'BEGIN { printf "%.4f", s/3600.0 }')"
  echo "Starting takeover stage from step $step for ${hours}h"
  set +e
  MAX_WALL_TIME_HOURS="$hours" BATCH_SIZE="${BATCH_SIZE:-24}" \
    EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-24}" \
    bash scripts/run_mikasa_10h.sh "$CONFIG" \
      --model.config_path="${latest_state_path%/trainer_state--*_checkpoint.pt}" \
      --resume_path="${latest_state_path%/trainer_state--*_checkpoint.pt}" \
      --resume_step="$step" --restore_trainer_state=true \
      --output_dir="$OUTPUT_DIR"
  rc=$?
  set -e
  (( $(date +%s) < DEADLINE_EPOCH )) || break
  echo "Takeover stage exited rc=$rc; retrying from newest checkpoint"
  sleep 30
done

echo "Dual deadline reached"
