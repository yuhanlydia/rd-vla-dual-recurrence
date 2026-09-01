from experiments.robot.mikasa_robo.interaction_analysis import go_no_go, interactions


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
