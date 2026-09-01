"""Optimizer, scheduler, and RNG state for exact training continuation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def trainer_state_path(checkpoint_dir, step: int) -> Path:
    return Path(checkpoint_dir) / f"trainer_state--{step}_checkpoint.pt"


def save_trainer_state(checkpoint_dir, step, optimizers, scheduler) -> Path:
    path = trainer_state_path(checkpoint_dir, step)
    payload = {
        "version": 1,
        "step": int(step),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }
    torch.save(payload, path)
    return path


def restore_trainer_state(checkpoint_dir, step, optimizers, scheduler, *, required=False) -> bool:
    path = trainer_state_path(checkpoint_dir, step)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["step"]) != int(step):
        raise ValueError(f"Trainer-state step {payload['step']} does not match requested step {step}")
    if len(payload["optimizers"]) != len(optimizers):
        raise ValueError(
            f"Checkpoint has {len(payload['optimizers'])} optimizers, current run has {len(optimizers)}"
        )
    for optimizer, state in zip(optimizers, payload["optimizers"]):
        optimizer.load_state_dict(state)
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"] is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return True
