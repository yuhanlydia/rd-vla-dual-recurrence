#!/usr/bin/env bash
set -euo pipefail

# Cheap preflight for MIKASA closed-loop evaluation. This intentionally does
# not import torch, ManiSkill, or the VLA checkpoint.

echo "[runtime] NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[runtime] ERROR: nvidia-smi is unavailable" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if [[ -n "${NVIDIA_DRIVER_CAPABILITIES:-}" ]]; then
  caps=",${NVIDIA_DRIVER_CAPABILITIES,,},"
  if [[ "$caps" != *,graphics,* && "$caps" != *,all,* ]]; then
    echo "[runtime] WARN: graphics capability label is absent; probing SAPIEN's bundled NVIDIA ICD" >&2
  fi
fi

if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "[runtime] ERROR: vulkaninfo is unavailable" >&2
  exit 4
fi
report="$(vulkaninfo --summary 2>&1 || true)"
if ! printf '%s\n' "$report" | rg -q 'PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU'; then
  # Some NVIDIA container-runtime versions install the ICD under /etc while
  # the loader only scans /usr/share by default.  Probe it explicitly so a
  # broken/missing graphics mount is distinguishable from a Mesa-only host.
  # SAPIEN also bundles an EGL-backed NVIDIA descriptor that works in images
  # where the runtime capability label is compute-only.
  for icd in \
    /etc/vulkan/icd.d/nvidia_icd.json \
    /usr/share/vulkan/icd.d/nvidia_icd.json \
    "$SCRIPT_ROOT/.venv/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"; do
    [[ -f "$icd" ]] || continue
    candidate_report="$(VK_ICD_FILENAMES="$icd" vulkaninfo --summary 2>&1 || true)"
    if printf '%s\n' "$candidate_report" | rg -q 'PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU'; then
      report="$candidate_report"
      break
    fi
    report="$report\n[explicit ICD $icd]\n$candidate_report"
  done
fi
if ! printf '%b\n' "$report" | rg -q 'PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU'; then
  echo "[runtime] ERROR: no Vulkan discrete/integrated GPU was found" >&2
  printf '%b\n' "$report" | rg -m 8 'vkCreateInstance|deviceType|deviceName|llvmpipe|error|ICD' >&2 || true
  exit 5
fi
echo "[runtime] OK: CUDA and Vulkan GPU render device are visible"
