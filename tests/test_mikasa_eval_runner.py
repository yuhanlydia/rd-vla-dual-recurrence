import torch

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
        assert "Vulkan GPU render device" in message or "graphics/display" in message
