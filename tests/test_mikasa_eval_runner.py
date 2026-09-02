import json
import sys
import tempfile
import types
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


def test_vulkan_probe_rejects_unrecognised_empty_report(monkeypatch):
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "graphics")
    monkeypatch.setattr(eval_runner.shutil, "which", lambda _: "vulkaninfo")

    class _Result:
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        eval_runner.subprocess, "run", lambda *args, **kwargs: _Result()
    )
    try:
        _check_vulkan_render_device()
    except SystemExit as exc:
        assert "requires a Vulkan GPU render device" in str(exc)
    else:
        raise AssertionError("empty Vulkan report was incorrectly accepted")


def test_main_writes_and_resumes_append_only_jsonl():
    """Exercise canonical selection, episode execution, and resume skipping."""

    class _FakePolicy:
        def __init__(self, checkpoint, instruction, k, memory, stale_delta=100):
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
    fake_root = types.ModuleType("mikasa_robo_suite")
    fake_root.__path__ = []
    fake_vla = types.ModuleType("mikasa_robo_suite.vla")
    fake_vla.__path__ = []
    fake_benchmark = types.ModuleType("mikasa_robo_suite.vla.benchmarking")

    class _FakeBenchmarkConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_benchmark.BenchmarkConfig = _FakeBenchmarkConfig
    fake_benchmark.make_benchmark_env = lambda *args, **kwargs: _FakeEnv()
    fake_benchmark.select_benchmark_tasks = lambda **kwargs: [
        types.SimpleNamespace(
            env_id="RememberColor9-VLA-v0",
            split="short",
            memory_type="object",
            language_instruction="remember the color",
        )
    ]
    old_modules = {
        name: sys.modules.get(name)
        for name in (
            "mikasa_robo_suite",
            "mikasa_robo_suite.vla",
            "mikasa_robo_suite.vla.benchmarking",
        )
    }
    try:
        eval_runner.RDVLAMemoryPolicy = _FakePolicy
        eval_runner._make_eval_env = lambda task, config, sim_backend: _FakeEnv()
        eval_runner._check_vulkan_render_device = lambda: None
        sys.modules["mikasa_robo_suite"] = fake_root
        sys.modules["mikasa_robo_suite.vla"] = fake_vla
        sys.modules["mikasa_robo_suite.vla.benchmarking"] = fake_benchmark
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
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_cpu_simulation_uses_vulkan_gpu_renderer():
    calls = {}
    fake_gym = types.ModuleType("gymnasium")

    def make(*args, **kwargs):
        calls["env_id"] = args[0]
        calls.update(kwargs)
        return object()

    fake_gym.make = make
    fake_benchmark = types.ModuleType("mikasa_robo_suite.vla.benchmarking")
    fake_benchmark.apply_mikasa_vla_wrappers = lambda env, include_overlays: env
    old_gym = sys.modules.get("gymnasium")
    old_benchmark = sys.modules.get("mikasa_robo_suite.vla.benchmarking")
    sys.modules["gymnasium"] = fake_gym
    sys.modules["mikasa_robo_suite.vla.benchmarking"] = fake_benchmark
    try:
        task = types.SimpleNamespace(env_id="RememberColor9-VLA-v0")
        config = types.SimpleNamespace(
            save_videos=False,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pose",
            reward_mode="sparse",
            include_overlays=False,
        )
        eval_runner._make_eval_env(task, config, "cpu")
        assert calls["sim_backend"] == "cpu"
        assert calls["render_backend"] == "gpu"
    finally:
        if old_gym is None:
            sys.modules.pop("gymnasium", None)
        else:
            sys.modules["gymnasium"] = old_gym
        if old_benchmark is None:
            sys.modules.pop("mikasa_robo_suite.vla.benchmarking", None)
        else:
            sys.modules["mikasa_robo_suite.vla.benchmarking"] = old_benchmark
