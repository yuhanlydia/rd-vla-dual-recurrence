#!/usr/bin/env python3
"""Validate the RLDS metadata used by the ten-task MIKASA pilot.

This check is intentionally independent of TensorFlow/SAPIEN so it can run
while a GPU training job is active. It verifies that every selected task has
the expected train trajectory count before an expensive run is launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


PILOT_TASKS = (
    "remember_color_9_vla_v0",
    "remember_color_9_long_vla_v0",
    "chain_of_colors_5_vla_v0",
    "chain_of_colors_5_long_vla_v0",
    "shell_game_shuffle_touch_vla_v0",
    "shell_game_shuffle_touch_long_vla_v0",
    "blink_count_button_press_hard_vla_v0",
    "blink_count_button_press_hard_long_vla_v0",
    "timed_transfer_hard_vla_v0",
    "timed_transfer_hard_long_vla_v0",
)


def _train_count(metadata: dict, path: Path) -> int:
    splits = metadata.get("splits")
    if not isinstance(splits, list):
        raise ValueError(f"{path}: expected a list in 'splits'")
    train = next((split for split in splits if split.get("name") == "train"), None)
    if train is None:
        raise ValueError(f"{path}: missing train split")
    lengths = train.get("shardLengths")
    if not isinstance(lengths, list) or not lengths:
        raise ValueError(f"{path}: train.shardLengths is empty or malformed")
    try:
        count = sum(int(length) for length in lengths)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: shardLengths must be integers") from exc
    if count <= 0:
        raise ValueError(f"{path}: train trajectory count is {count}")
    return count


def validate(root: Path, tasks: Iterable[str] = PILOT_TASKS, expected: int = 250) -> dict[str, int]:
    """Return task-to-count mapping or raise a descriptive validation error."""

    counts: dict[str, int] = {}
    for task in tasks:
        path = root / task / "1.0.0" / "dataset_info.json"
        if not path.is_file():
            raise FileNotFoundError(f"{task}: missing {path}")
        try:
            metadata = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON") from exc
        count = _train_count(metadata, path)
        if count != expected:
            raise ValueError(f"{task}: expected {expected} train trajectories, found {count}")
        counts[task] = count
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="MIKASA RLDS directory")
    parser.add_argument("--expected", type=int, default=250, help="expected train trajectories per task")
    parser.add_argument("--task", action="append", dest="tasks", help="task name (repeatable; defaults to pilot)")
    args = parser.parse_args()
    if args.expected <= 0:
        parser.error("--expected must be positive")
    tasks = tuple(args.tasks) if args.tasks else PILOT_TASKS
    counts = validate(args.root, tasks, args.expected)
    print(f"validated {len(counts)} tasks x {args.expected} train trajectories = {sum(counts.values())}")
    for task, count in counts.items():
        print(f"{task}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
