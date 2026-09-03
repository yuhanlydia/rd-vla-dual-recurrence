#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# SAPIEN 3.0.0b1 still imports the removed ``pkg_resources`` API, while the
# project venv intentionally contains the newer setuptools without that
# compatibility module.  The Ubuntu ``/usr/lib`` compatibility package is
# preferred: ``/usr/local`` also contains a minimal NumPy shim that would
# shadow the real venv NumPy and make torch/SAPIEN fail during import.
if ! .venv/bin/python -c 'import pkg_resources' >/dev/null 2>&1; then
  for compat_root in /usr/lib/python3/dist-packages /usr/local/lib/python3.10/dist-packages; do
    if [[ -f "$compat_root/pkg_resources/__init__.py" ]]; then
      # Keep the venv site-packages first so its NumPy/pyparsing versions are
      # used for torch, ManiSkill, and SAPIEN; only pkg_resources comes from
      # the compatibility path.
      VENV_SITE="$(pwd)/.venv/lib/python3.10/site-packages"
      export PYTHONPATH="$VENV_SITE:$compat_root${PYTHONPATH:+:$PYTHONPATH}"
      break
    fi
  done
fi

# In this image the NVIDIA container runtime exposes CUDA but not the normal
# graphics ICD.  SAPIEN ships a compatible EGL-backed NVIDIA ICD descriptor;
# use it unless the caller supplied an explicit Vulkan selection.
SAPIEN_NVIDIA_ICD="$(pwd)/.venv/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f "$SAPIEN_NVIDIA_ICD" ]]; then
  export VK_ICD_FILENAMES="$SAPIEN_NVIDIA_ICD"
fi

SAPIEN_OIDN_DIR="$(pwd)/.venv/lib/python3.10/site-packages/sapien/oidn_library"
if [[ -d "$SAPIEN_OIDN_DIR" ]]; then
  export LD_LIBRARY_PATH="$SAPIEN_OIDN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec .venv/bin/python experiments/robot/mikasa_robo/run_mikasa_eval.py "$@"
