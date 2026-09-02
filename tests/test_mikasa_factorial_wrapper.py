import subprocess
from pathlib import Path


def test_factorial_wrapper_dry_run_expands_all_conditions_without_evaluator():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "bash",
            "scripts/run_mikasa_factorial.sh",
            "checkpoint-dir",
            "results.jsonl",
            "--task",
            "RememberColor9-VLA-v0",
        ],
        cwd=repo,
        env={"DRY_RUN": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert len(lines) == 6
    assert ["memory=reset", "memory=reset", "memory=correct", "memory=correct", "memory=shuffle", "memory=stale"] == [
        line.split()[3] for line in lines
    ]
    assert ["k=1", "k=12", "k=1", "k=12", "k=12", "k=12"] == [line.split()[4] for line in lines]
