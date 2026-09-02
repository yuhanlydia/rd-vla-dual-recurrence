#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${1:?usage: $0 CHECKPOINT OUTPUT_JSONL [evaluator args ...]}"
OUTPUT="${2:?usage: $0 CHECKPOINT OUTPUT_JSONL [evaluator args ...]}"
shift 2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for spec in reset:1 reset:12 correct:1 correct:12 shuffle:12; do
    IFS=: read -r memory k <<<"$spec"
    printf 'DRY_RUN: checkpoint=%s output=%s memory=%s k=%s args=' \
      "$CHECKPOINT" "$OUTPUT" "$memory" "$k"
    printf '%q ' "$@"
    printf '\n'
  done
  exit 0
fi

# The evaluator is append-only and skips completed (task, seed, memory, K)
# tuples, so this command is safe to resume after a simulator or driver exit.
for spec in reset:1 reset:12 correct:1 correct:12 shuffle:12; do
  IFS=: read -r memory k <<<"$spec"
  scripts/run_mikasa_eval.sh \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --memory "$memory" \
    --k "$k" \
    "$@"
done

.venv/bin/python experiments/robot/mikasa_robo/interaction_analysis.py "$OUTPUT" \
  > "${OUTPUT%.jsonl}.report.json"
echo "Wrote ${OUTPUT%.jsonl}.report.json"
