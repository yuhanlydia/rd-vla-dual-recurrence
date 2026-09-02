#!/usr/bin/env bash
set -euo pipefail

# Cheap preflight for MIKASA closed-loop evaluation. This intentionally does
# not import torch, ManiSkill, or the VLA checkpoint.

echo "[runtime] NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[runtime] ERROR: nvidia-smi is unavailable" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if [[ -n "${NVIDIA_DRIVER_CAPABILITIES:-}" ]]; then
  caps=",${NVIDIA_DRIVER_CAPABILITIES,,},"
  if [[ "$caps" != *,graphics,* && "$caps" != *,all,* ]]; then
    echo "[runtime] ERROR: graphics capability is not exposed; recreate the container with NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics (and display when needed)" >&2
    exit 3
  fi
fi

if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "[runtime] ERROR: vulkaninfo is unavailable" >&2
  exit 4
fi
report="$(vulkaninfo --summary 2>&1 || true)"
if ! printf '%s\n' "$report" | rg -q 'PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU'; then
  echo "[runtime] ERROR: no Vulkan discrete/integrated GPU was found" >&2
  printf '%s\n' "$report" | rg -m 5 'vkCreateInstance|deviceType|deviceName|llvmpipe|error' >&2 || true
  exit 5
fi
echo "[runtime] OK: CUDA and Vulkan GPU render device are visible"
