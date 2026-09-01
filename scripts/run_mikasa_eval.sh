#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SAPIEN_OIDN_DIR="$(pwd)/.venv/lib/python3.10/site-packages/sapien/oidn_library"
if [[ -d "$SAPIEN_OIDN_DIR" ]]; then
  export LD_LIBRARY_PATH="$SAPIEN_OIDN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec .venv/bin/python experiments/robot/mikasa_robo/run_mikasa_eval.py "$@"
