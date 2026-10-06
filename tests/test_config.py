from factor_model.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["sample"]["holdout_start"] > cfg["sample"]["start"]
    assert cfg["walk_forward"]["window"] in {"expanding", "rolling"}
