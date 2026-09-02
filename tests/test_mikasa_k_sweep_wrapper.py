import os
import subprocess
from pathlib import Path


def test_k_sweep_dry_run_expands_full_reasoning_depths():
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    result = subprocess.run(
        [
            "bash",
            "scripts/run_mikasa_k_sweep.sh",
            "checkpoint-dir",
            "results.jsonl",
            "--task",
            "RememberColor9-VLA-v0",
        ],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 6
    assert [line.split()[4] for line in lines] == [
        "k=1", "k=2", "k=4", "k=8", "k=12", "k=16"
    ]
