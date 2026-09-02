#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${1:?usage: $0 CHECKPOINT OUTPUT_JSONL [evaluator args ...]}"
OUTPUT="${2:?usage: $0 CHECKPOINT OUTPUT_JSONL [evaluator args ...]}"
shift 2
MEMORY="${MEMORY:-correct}"
K_VALUES="${K_VALUES:-1 2 4 8 12 16}"

# Each invocation is append-only and the evaluator's resume key includes
# (task, seed, memory, K, stale_delta), so interrupted sweeps are safe to rerun.
for k in $K_VALUES; do
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: checkpoint=%s output=%s memory=%s k=%s args=' \
      "$CHECKPOINT" "$OUTPUT" "$MEMORY" "$k"
    printf '%q ' "$@"
    printf '\n'
    continue
  fi
  scripts/run_mikasa_eval.sh \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --memory "$MEMORY" \
    --k "$k" \
    "$@"
done
