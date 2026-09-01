"""Factorial and causal analysis for MIKASA dual-recurrence evaluations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = {
    "reactive": ("reset", 1),
    "reasoning": ("reset", 12),
    "memory": ("correct", 1),
    "dual": ("correct", 12),
}


def _condition(row):
    key = (row["memory"], int(row["k"]))
    for name, value in CONDITIONS.items():
        if value == key:
            return name
    return None


def success_rates(rows):
    grouped = defaultdict(list)
    for row in rows:
        condition = _condition(row)
        if condition is not None:
            grouped[(row["task"], condition)].append(float(row["success"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def interactions(rows):
    rates = success_rates(rows)
    tasks = sorted({row["task"] for row in rows})
    result = {}
    for task in tasks:
        values = {name: rates.get((task, name), float("nan")) for name in CONDITIONS}
        values["interaction"] = (
            (values["dual"] - values["memory"])
            - (values["reasoning"] - values["reactive"])
        )
        result[task] = values
    return result


def paired_bootstrap_interaction(rows, samples=10_000, seed=42):
    """Resample episode seeds, preserving all four interventions per seed."""
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[(row["task"], int(row["episode_seed"]))].append(row)
    tasks = sorted({task for task, _ in by_seed})
    rng = np.random.default_rng(seed)
    output = {}
    for task in tasks:
        episodes = [values for (name, _), values in by_seed.items() if name == task]
        complete = [ep for ep in episodes if {_condition(row) for row in ep} >= set(CONDITIONS)]
        if not complete:
            continue
        estimates = []
        for _ in range(samples):
            picked = rng.integers(0, len(complete), size=len(complete))
            sampled = [row for index in picked for row in complete[index]]
            estimates.append(interactions(sampled)[task]["interaction"])
        point = interactions([row for ep in complete for row in ep])[task]["interaction"]
        output[task] = {
            "interaction": point,
            "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
            "episodes": len(complete),
        }
    return output


def go_no_go(rows):
    table = interactions(rows)
    long_tasks = [task for task in table if any(r["task"] == task and r.get("horizon") == "long" for r in rows)]
    wins = sum(table[task]["dual"] > table[task]["reasoning"] for task in long_tasks)
    gains = [table[task]["dual"] - table[task]["reasoning"] for task in long_tasks]
    interactions_long = [table[task]["interaction"] for task in long_tasks]
    return {
        "go": bool(
            len(long_tasks) >= 5
            and wins >= 4
            and float(np.mean(gains)) >= 0.10
            and float(np.mean(interactions_long)) > 0
        ),
        "long_family_wins": wins,
        "long_tasks": len(long_tasks),
        "mean_long_gain": float(np.mean(gains)) if gains else float("nan"),
        "mean_long_interaction": float(np.mean(interactions_long)) if interactions_long else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="JSONL evaluation rows")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    report = {
        "tasks": interactions(rows),
        "bootstrap": paired_bootstrap_interaction(rows, args.bootstrap_samples),
        "gate": go_no_go(rows),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
