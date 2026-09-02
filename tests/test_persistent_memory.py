import torch

from prismatic.models.action_heads import RecurrentConfigInternal, VLARecurrent


def _model():
    cfg = RecurrentConfigInternal(
        hidden_dim=32,
        num_heads=4,
        recurrent_vlm_layers=(0,),
        action_chunk_len=4,
        action_dim=7,
        mean_recurrence=2,
        backprop_depth=2,
        random_iterations=False,
        use_persistent_memory=True,
        memory_tokens=4,
        memory_dim=16,
        memory_layers=2,
        memory_heads=4,
        memory_ffn_dim=64,
        memory_dropout=0.0,
    )
    return VLARecurrent(cfg)


def _inputs(batch=2):
    h_a = torch.randn(batch, 1, 5, 32)
    h_t = torch.randn(batch, 1, 6, 32)
    proprio = torch.randn(batch, 1, 32)
    previous_action = torch.randn(batch, 7)
    return h_a, h_t, proprio, previous_action


def test_memory_is_returned_and_carried_across_decisions():
    torch.manual_seed(0)
    model = _model().eval()
    h_a, h_t, proprio, previous_action = _inputs()
    action_1, memory_1 = model(
        h_a, h_t, proprio, previous_action=previous_action, num_iter=2, return_memory=True
    )
    action_2, memory_2 = model(
        h_a, h_t, proprio, memory_state=memory_1,
        previous_action=action_1[:, 0], num_iter=2, return_memory=True,
    )
    assert action_1.shape == (2, 4, 7)
    assert memory_1.shape == (2, 4, 16)
    assert not torch.allclose(memory_1, memory_2)
    assert not torch.allclose(action_1, action_2)


def test_memory_destruction_interventions_change_the_prediction():
    torch.manual_seed(1)
    model = _model().eval()
    h_a, h_t, proprio, previous_action = _inputs()
    _, memory = model(
        h_a, h_t, proprio, previous_action=previous_action, num_iter=1, return_memory=True
    )
    torch.manual_seed(7)
    correct = model(
        h_a, h_t, proprio, memory_state=memory,
        previous_action=previous_action, num_iter=1,
    )
    torch.manual_seed(7)
    shuffled = model(
        h_a, h_t, proprio, memory_state=memory.flip(0),
        previous_action=previous_action, num_iter=1,
    )
    assert not torch.allclose(correct, shuffled)


def test_action_loss_trains_memory_writer_and_projection():
    torch.manual_seed(2)
    model = _model().train()
    h_a, h_t, proprio, previous_action = _inputs()
    prediction, _ = model(
        h_a, h_t, proprio, previous_action=previous_action,
        num_iter=2, memory_dropout=False, return_memory=True,
    )
    prediction.square().mean().backward()
    assert model.memory_to_scratchpad.weight.grad is not None
    assert model.memory_updater.prelude_projection.weight.grad is not None
    assert model.memory_updater.prelude_projection.weight.grad.abs().sum() > 0


def test_reasoning_depth_sweep_preserves_action_shape_and_finiteness():
    torch.manual_seed(3)
    model = _model().eval()
    h_a, h_t, proprio, previous_action = _inputs(batch=1)
    for k in (1, 2, 4, 8, 12, 16):
        torch.manual_seed(17)
        prediction = model(
            h_a,
            h_t,
            proprio,
            previous_action=previous_action,
            num_iter=k,
        )
        assert prediction.shape == (1, 4, 7)
        assert torch.isfinite(prediction).all()


def test_memory_dropout_mask_can_be_fixed_for_a_tbptt_window():
    torch.manual_seed(4)
    model = _model().train()
    model.cfg.memory_dropout = 0.3
    h_a, h_t, proprio, previous_action = _inputs()
    torch.manual_seed(9)
    dropped = model(
        h_a, h_t, proprio, previous_action=previous_action, num_iter=1,
        memory_dropout_mask=torch.zeros(2),
    )
    torch.manual_seed(9)
    kept = model(
        h_a, h_t, proprio, previous_action=previous_action, num_iter=1,
        memory_dropout_mask=torch.ones(2),
    )
    assert not torch.allclose(dropped, kept)
