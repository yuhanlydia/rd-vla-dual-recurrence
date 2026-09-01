import torch

from experiments.robot.mikasa_robo.memory_interventions import MemoryIntervention


def test_reset_shuffle_and_stale_are_causal_state_replacements():
    current = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(MemoryIntervention("reset").apply(current), torch.zeros_like(current))
    assert torch.equal(MemoryIntervention("shuffle").apply(current), current.roll(1, 0))

    stale = MemoryIntervention("stale", stale_delta=2)
    assert not stale.apply(current).any()
    stale.apply(current + 1)
    assert torch.equal(stale.apply(current + 2), current)


def test_batch_one_shuffle_uses_another_episode():
    intervention = MemoryIntervention("shuffle")
    other = torch.ones(1, 3, 4)
    intervention.reset_episode(other)
    result = intervention.apply(torch.full_like(other, 2))
    assert torch.equal(result, other)
