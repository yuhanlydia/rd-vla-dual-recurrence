"""Causal interventions on persistent state during closed-loop evaluation."""

from __future__ import annotations

from collections import deque

import torch


class MemoryIntervention:
    """Track correct/stale/cross-episode states without changing current inputs."""

    def __init__(self, mode="correct", stale_delta=100, bank_size=256, seed=42):
        if mode not in {"correct", "reset", "shuffle", "stale"}:
            raise ValueError(f"Unknown memory intervention: {mode}")
        if stale_delta < 1:
            raise ValueError("stale_delta must be positive")
        self.mode = mode
        self.stale_delta = stale_delta
        self.history = deque(maxlen=stale_delta + 1)
        self.episode_bank = deque(maxlen=bank_size)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def reset_episode(self, final_memory=None):
        if final_memory is not None:
            for state in final_memory.detach().cpu():
                self.episode_bank.append(state.clone())
        self.history.clear()

    def apply(self, correct_memory):
        if correct_memory is None:
            return None
        detached = correct_memory.detach()
        self.history.append(detached.clone())
        if self.mode == "correct":
            return correct_memory
        if self.mode == "reset":
            return torch.zeros_like(correct_memory)
        if self.mode == "stale":
            if len(self.history) <= self.stale_delta:
                return torch.zeros_like(correct_memory)
            return self.history[0].to(correct_memory.device, correct_memory.dtype)

        # Shuffle uses a different completed episode. A batch roll is valid when
        # multiple episodes are evaluated together; batch=1 draws from the bank.
        if correct_memory.shape[0] > 1:
            return correct_memory.roll(1, dims=0)
        if not self.episode_bank:
            return torch.zeros_like(correct_memory)
        index = torch.randint(len(self.episode_bank), (1,), generator=self.generator).item()
        state = self.episode_bank[index].to(correct_memory.device, correct_memory.dtype)
        return state.unsqueeze(0)
