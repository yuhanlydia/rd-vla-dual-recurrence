import os
import subprocess
from pathlib import Path


def test_eval_wrapper_keeps_venv_numpy_before_pkg_resources_compat_shim():
    """The wrapper's compatibility path must not shadow the venv's NumPy."""

    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["scripts/run_mikasa_eval.sh", "--help"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "numpy.ndarray" not in result.stderr


def test_runtime_preflight_accepts_sapiens_bundled_nvidia_icd():
    """The preflight can use SAPIEN's working ICD in compute-only images."""

    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("VK_ICD_FILENAMES", None)
    env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
    result = subprocess.run(
        ["scripts/check_mikasa_runtime.sh"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NVIDIA GeForce RTX 3090" in result.stdout
