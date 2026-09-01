import random
import tempfile

import numpy as np
import torch

from prismatic.training.checkpointing import restore_trainer_state, save_trainer_state


def test_trainer_state_round_trip_restores_optimizer_scheduler_and_rng():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=0.1)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as directory:
        save_trainer_state(directory, 123, [optimizer], scheduler)
        expected = (random.random(), np.random.rand(), torch.rand(1))

        restored_optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
        restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            restored_optimizer, milestones=[9], gamma=0.5
        )
        assert restore_trainer_state(
            directory, 123, [restored_optimizer], restored_scheduler, required=True
        )
        actual = (random.random(), np.random.rand(), torch.rand(1))

    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
