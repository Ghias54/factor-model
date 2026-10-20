from factor_model.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["sample"]["holdout_start"] > cfg["sample"]["start"]
    assert cfg["sample"]["first_oos"] > cfg["sample"]["start"]
    assert cfg["sample"]["holdout_start"] > cfg["sample"]["first_oos"]
    assert cfg["walk_forward"]["window"] in {"expanding", "rolling"}


def test_locked_sample_and_universe():
    cfg = load_config()
    assert cfg["sample"]["start"] == "2006-01-31"
    assert cfg["sample"]["end"] == "2026-08-31"
    assert cfg["sample"]["holdout_start"] == "2024-01-31"
    assert cfg["sample"]["first_oos"] == "2011-01-31"
    assert cfg["universe"]["top_n_by_market_cap"] == 1000
    assert cfg["universe"]["min_price"] == 5.0
    assert cfg["universe"]["price_field"] == "close_raw"
    assert cfg["universe"]["market_cap"] == "close_x_shares_diluted"
    assert "robustness_top_n" not in cfg["universe"]


def test_locked_target_features_models():
    cfg = load_config()
    assert cfg["target"]["type"] == "rank"
    assert cfg["target"]["risk_free"] == "month3"
    assert len(cfg["features"]["names"]) == 26
    assert cfg["preprocessing"]["rank_scale"] == [-1, 1]
    assert cfg["preprocessing"]["missing_to"] == 0
    assert cfg["portfolio"]["long_only"] is True
    assert cfg["portfolio"]["weighting"] == "equal"
    names = [m["name"] for m in cfg["models"]]
    assert names == [
        "classic4_ew",
        "lasso",
        "ridge",
        "random_forest",
        "xgboost",
        "mlp",
        "ensemble",
    ]


def test_evaluation_regimes():
    cfg = load_config()
    regimes = cfg["evaluation"]["regimes"]
    assert regimes["covid"]["start"] == "2020-02-29"
    assert regimes["ratehike"]["end"] == "2022-12-31"
    assert regimes["survivorship_biased"]["end"] == "2020-12-31"
    assert regimes["survivorship_clean"]["start"] == "2021-01-31"
