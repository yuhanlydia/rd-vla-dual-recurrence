#!/usr/bin/env bash
set -euo pipefail

# Wait for an already-running dual-recurrence stage to finish its wall-time
# boundary, then resume from the newest dual checkpoint for another bounded
# stage.  This keeps checkpoint hand-off automatic without interrupting the
# current trainer.

cd "$(dirname "$0")/.."

WATCH_PID="${1:?usage: $0 TRAINER_OR_WRAPPER_PID [HOURS]}"
CONTINUE_HOURS="${2:-3}"

while kill -0 "$WATCH_PID" 2>/dev/null; do
  sleep 60
done

latest_state_path="$(find outputs/mikasa10_dual -maxdepth 3 -type f \
  -name 'trainer_state--*_checkpoint.pt' -printf '%p\n' \
  | sed -n 's/.*\/trainer_state--\([0-9][0-9]*\)_checkpoint\.pt$/\1 &/p' \
  | sort -n | tail -n 1 | cut -d' ' -f2-)"

if [[ -z "$latest_state_path" ]]; then
  echo "No dual checkpoint found after watched process exited" >&2
  exit 1
fi

checkpoint="${latest_state_path%/trainer_state--*_checkpoint.pt}"
latest_state="${latest_state_path##*/}"
step="${latest_state#trainer_state--}"
step="${step%_checkpoint.pt}"

if [[ ! "$step" =~ ^[0-9]+$ ]]; then
  echo "Could not parse dual checkpoint step from $latest_state_path" >&2
  exit 2
fi

# Validate the resumable trainer state before handing it to the next process.
# A partially written file must never silently restart a stage from bad RNG or
# optimizer state.
.venv/bin/python - "$latest_state_path" "$step" <<'PY'
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
state = torch.load(path, map_location="cpu", weights_only=False)
required = {"step", "optimizers", "scheduler", "cuda_rng", "numpy_rng", "python_rng", "torch_rng"}
missing = required.difference(state)
if missing:
    raise SystemExit(f"Checkpoint {path} missing keys: {sorted(missing)}")
if int(state["step"]) != expected_step:
    raise SystemExit(f"Checkpoint step mismatch: filename={expected_step}, payload={state['step']}")
if not state["optimizers"]:
    raise SystemExit(f"Checkpoint {path} has no optimizer state")
print(f"Validated checkpoint step={expected_step}: optimizer/scheduler/RNG state present")
PY

echo "Resuming dual stage from $checkpoint (step $step) for ${CONTINUE_HOURS}h"
MAX_WALL_TIME_HOURS="$CONTINUE_HOURS" \
BATCH_SIZE="${BATCH_SIZE:-24}" \
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-24}" \
  bash scripts/run_mikasa_10h.sh configs/train/rdvla_mikasa10_dual.yaml \
    --model.config_path="$checkpoint" \
    --resume_path="$checkpoint" \
    --resume_step="$step" \
    --restore_trainer_state=false
