#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Run both trainable stages within one wall-time budget.  Each stage keeps its
# own latest-only checkpoint directory; the handoff script discovers the
# newest baseline trainer state and starts the frozen-backbone memory stage.
BASELINE_HOURS="${BASELINE_HOURS:-5}"
DUAL_HOURS="${DUAL_HOURS:-5}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/train/rdvla_mikasa10_baseline.yaml}"

echo "[campaign] baseline stage: ${BASELINE_HOURS}h"
MAX_WALL_TIME_HOURS="$BASELINE_HOURS" bash scripts/run_mikasa_10h.sh "$BASELINE_CONFIG" "$@"

echo "[campaign] handing off to dual memory stage: ${DUAL_HOURS}h"
MAX_WALL_TIME_HOURS="$DUAL_HOURS" bash scripts/run_dual_from_latest_baseline.sh
