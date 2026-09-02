import json

from scripts.validate_mikasa_metadata import validate


def _write_metadata(root, task, lengths):
    path = root / task / "1.0.0"
    path.mkdir(parents=True)
    (path / "dataset_info.json").write_text(
        json.dumps({"splits": [{"name": "train", "shardLengths": lengths}]})
    )


def test_validate_sums_string_shard_lengths(tmp_path):
    _write_metadata(tmp_path, "task_a", ["2", "3"])
    assert validate(tmp_path, ["task_a"], expected=5) == {"task_a": 5}


def test_validate_rejects_wrong_trajectory_count(tmp_path):
    _write_metadata(tmp_path, "task_a", ["2"])
    try:
        validate(tmp_path, ["task_a"], expected=5)
    except ValueError as exc:
        assert "expected 5" in str(exc)
    else:
        raise AssertionError("validate() accepted the wrong trajectory count")
