import json
import sys
import tempfile
from pathlib import Path

import torch

import experiments.robot.mikasa_robo.run_mikasa_eval as eval_runner
from experiments.robot.mikasa_robo.run_mikasa_eval import (
    _check_vulkan_render_device,
    _require_canonical_tasks,
    run_episode,
)


class _Policy:
    def __init__(self):
        self.resets = 0
        self.calls = 0

    def reset_episode(self):
        self.resets += 1

    def forward(self, obs):
        self.calls += 1
        return torch.zeros(2, 7)


class _Env:
    max_episode_steps = 4

    def __init__(self):
        self.unwrapped = self
        self.device = torch.device("cpu")
        self.steps = 0

    def reset(self, seed):
        self.steps = 0
        return {"seed": seed}, {}

    def step(self, action):
        assert action.shape == (1, 7)
        self.steps += 1
        done = torch.tensor([self.steps == 3])
        info = {"success": torch.tensor([self.steps == 2])}
        return {}, torch.tensor([0.5]), done, torch.tensor([False]), info


def test_episode_resets_memory_latches_success_and_uses_chunk_fifo():
    policy = _Policy()
    success, episode_return, length = run_episode(_Env(), policy, seed=123)
    assert policy.resets == 1
    assert policy.calls == 2
    assert success is True
    assert episode_return == 1.5
    assert length == 3


def test_unknown_task_ids_fail_fast_instead_of_using_custom_fallback():
    class _Task:
        def __init__(self, env_id, split):
            self.env_id = env_id
            self.split = split

    try:
        _require_canonical_tasks([_Task("typo-task", "custom")])
    except SystemExit as exc:
        assert "Unknown MIKASA task ID" in str(exc)
    else:
        raise AssertionError("custom task fallback was not rejected")
    assert _require_canonical_tasks([_Task("RememberColor9-VLA-v0", "short")])[0].env_id == "RememberColor9-VLA-v0"


def test_vulkan_probe_reports_cpu_only_render_host():
    try:
        _check_vulkan_render_device()
    except SystemExit as exc:
        message = str(exc)
        assert "Vulkan GPU render device" in message or "graphics" in message


def test_main_writes_and_resumes_append_only_jsonl():
    """Exercise canonical selection, episode execution, and resume skipping."""

    class _FakePolicy:
        def __init__(self, checkpoint, instruction, k, memory):
            self.instruction = instruction
            self.calls = 0

        def set_instruction(self, instruction):
            self.instruction = instruction

        def reset_episode(self):
            pass

        def forward(self, obs):
            self.calls += 1
            return torch.zeros(1, 7)

    class _FakeEnv:
        max_episode_steps = 2

        def __init__(self):
            self.unwrapped = self
            self.device = torch.device("cpu")
            self.steps = 0

        def reset(self, seed):
            self.steps = 0
            return {}, {}

        def step(self, action):
            self.steps += 1
            return (
                {},
                torch.tensor([1.0]),
                torch.tensor([True]),
                torch.tensor([False]),
                {"success": torch.tensor([True])},
            )

        def close(self):
            pass

    old_policy = eval_runner.RDVLAMemoryPolicy
    old_env = eval_runner._make_eval_env
    old_check = eval_runner._check_vulkan_render_device
    old_argv = sys.argv[:]
    try:
        eval_runner.RDVLAMemoryPolicy = _FakePolicy
        eval_runner._make_eval_env = lambda task, config, sim_backend: _FakeEnv()
        eval_runner._check_vulkan_render_device = lambda: None
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "episodes.jsonl"
            sys.argv = [
                "run_mikasa_eval.py",
                "--checkpoint", tmp,
                "--output", str(output),
                "--task", "RememberColor9-VLA-v0",
                "--episodes", "1",
                "--k", "1",
                "--memory", "reset",
                "--sim-backend", "cpu",
            ]
            eval_runner.main()
            eval_runner.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            assert len(rows) == 1
            assert rows[0]["task"] == "RememberColor9-VLA-v0"
            assert rows[0]["success"] is True
            assert rows[0]["memory"] == "reset"
    finally:
        eval_runner.RDVLAMemoryPolicy = old_policy
        eval_runner._make_eval_env = old_env
        eval_runner._check_vulkan_render_device = old_check
        sys.argv = old_argv
