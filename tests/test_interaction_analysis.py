from experiments.robot.mikasa_robo.interaction_analysis import (
    go_no_go,
    interactions,
    paired_bootstrap_interaction,
)


def _rows():
    rows = []
    values = {"reactive": 0, "reasoning": 0, "memory": 0, "dual": 1}
    mapping = {
        "reactive": ("reset", 1),
        "reasoning": ("reset", 12),
        "memory": ("correct", 1),
        "dual": ("correct", 12),
    }
    for family in range(5):
        for seed in range(4):
            for condition, success in values.items():
                memory, k = mapping[condition]
                rows.append({
                    "task": f"long_{family}", "horizon": "long", "episode_seed": seed,
                    "memory": memory, "k": k, "success": success,
                })
            rows.append({
                "task": f"long_{family}", "horizon": "long", "episode_seed": seed,
                "memory": "shuffle", "k": 12, "success": 0,
            })
            for condition, (memory, k) in mapping.items():
                rows.append({
                    "task": f"short_{family}", "horizon": "short", "episode_seed": seed,
                    "memory": memory, "k": k, "success": 0,
                })
    return rows


def test_positive_factorial_interaction_and_gate():
    rows = _rows()
    table = interactions(rows)
    assert table["long_0"]["interaction"] == 1.0
    gate = go_no_go(rows)
    assert gate["go"]
    assert gate["long_family_wins"] == 5
    assert gate["mean_correct_minus_shuffle"] == 1.0
    assert gate["mean_long_gain"] > gate["mean_nonlong_gain"]


def test_gate_rejects_missing_memory_destruction_evidence():
    rows = [row for row in _rows() if row["memory"] != "shuffle"]
    assert not go_no_go(rows)["go"]


def test_paired_bootstrap_preserves_episode_seed_structure():
    rows = []
    for seed in range(8):
        for memory, k, success in (
            ("reset", 1, 0),
            ("reset", 12, 0),
            ("correct", 1, 0),
            ("correct", 12, 1),
        ):
            rows.append(
                {
                    "task": "long_probe",
                    "horizon": "long",
                    "episode_seed": seed,
                    "memory": memory,
                    "k": k,
                    "success": success,
                }
            )

    report = paired_bootstrap_interaction(rows, samples=300, seed=7)["long_probe"]
    assert report["interaction"] == 1.0
    assert report["episodes"] == 8
    assert report["ci95"] == [1.0, 1.0]


def test_paired_bootstrap_excludes_incomplete_episode():
    rows = []
    for seed in range(3):
        conditions = (
            ("reset", 1, 0),
            ("reset", 12, 0),
            ("correct", 1, 0),
            ("correct", 12, 1),
        )
        if seed == 2:
            conditions = conditions[:-1]
        for memory, k, success in conditions:
            rows.append({
                "task": "long_incomplete",
                "horizon": "long",
                "episode_seed": seed,
                "memory": memory,
                "k": k,
                "success": success,
            })

    report = paired_bootstrap_interaction(rows, samples=100, seed=3)["long_incomplete"]
    assert report["episodes"] == 2
    assert report["interaction"] == 1.0
