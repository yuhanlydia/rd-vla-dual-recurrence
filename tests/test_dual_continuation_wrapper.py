import os
import subprocess
from pathlib import Path

import torch


def test_continuation_dry_run_validates_custom_output_dir(tmp_path):
    checkpoint = tmp_path / "trainer_state--321_checkpoint.pt"
    torch.save(
        {
            "step": 321,
            "optimizers": [{"state": {}}],
            "scheduler": {},
            "cuda_rng": b"cuda",
            "numpy_rng": ("MT19937", [], 0, 0, 0.0),
            "python_rng": (3, (1, 2, 3), None),
            "torch_rng": torch.zeros(1, dtype=torch.uint8),
        },
        checkpoint,
    )
    result = subprocess.run(
        ["bash", "scripts/continue_dual_after_stage.sh", "99999999", "3", str(tmp_path)],
        env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Validated checkpoint step=321" in result.stdout
    assert f"output_dir={tmp_path}" in result.stdout


def test_continuation_dry_run_does_not_wait_for_live_pid(tmp_path):
    checkpoint = tmp_path / "trainer_state--654_checkpoint.pt"
    torch.save(
        {
            "step": 654,
            "optimizers": [{"state": {}}],
            "scheduler": {},
            "cuda_rng": b"cuda",
            "numpy_rng": ("MT19937", [], 0, 0, 0.0),
            "python_rng": (3, (1, 2, 3), None),
            "torch_rng": torch.zeros(1, dtype=torch.uint8),
        },
        checkpoint,
    )
    result = subprocess.run(
        ["bash", "scripts/continue_dual_after_stage.sh", str(os.getpid()), "3", str(tmp_path)],
        env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "DRY_RUN: not waiting for live pid" in result.stdout
    assert "Validated checkpoint step=654" in result.stdout
