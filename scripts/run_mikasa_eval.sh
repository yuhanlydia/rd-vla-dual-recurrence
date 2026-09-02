#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# SAPIEN 3.0.0b1 still imports the removed ``pkg_resources`` API, while the
# project venv intentionally contains the newer setuptools without that
# compatibility module.  Reuse the system compatibility package when it is
# available; this keeps the evaluator self-contained and makes the eventual
# Vulkan fix sufficient to launch the benchmark.
if ! .venv/bin/python -c 'import pkg_resources' >/dev/null 2>&1; then
  for compat_root in /usr/local/lib/python3.10/dist-packages /usr/lib/python3/dist-packages; do
    if [[ -f "$compat_root/pkg_resources/__init__.py" ]]; then
      export PYTHONPATH="$compat_root${PYTHONPATH:+:$PYTHONPATH}"
      break
    fi
  done
fi

SAPIEN_OIDN_DIR="$(pwd)/.venv/lib/python3.10/site-packages/sapien/oidn_library"
if [[ -d "$SAPIEN_OIDN_DIR" ]]; then
  export LD_LIBRARY_PATH="$SAPIEN_OIDN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec .venv/bin/python experiments/robot/mikasa_robo/run_mikasa_eval.py "$@"
