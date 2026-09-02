from configs import TrainConfig, get_legacy_config


def test_train_legacy_config_carries_memory_dropout():
    cfg = TrainConfig()
    cfg.action_head.recurrent.memory_dropout = 0.17
    legacy = get_legacy_config(cfg, "train")
    assert legacy.memory_dropout == 0.17
