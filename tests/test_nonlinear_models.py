"""Smoke / leakage / holdout tests for RF, XGBoost, MLP."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_model.factors.classic import MAX_FORMATION_MONTH
from factor_model.features.compute import FEATURE_NAMES
from factor_model.models import MLPModel, RandomForestModel, XGBoostModel
from factor_model.walkforward.engine import (
    WalkForwardEngine,
    monthly_rank_ic,
    shuffle_target_within_month,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel" / "panel.parquet"
PRED = ROOT / "data" / "predictions"


def _tiny_panel(n_months: int = 90, n_names: int = 25, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2006-01-31", periods=n_months, freq="ME")
    rows = []
    for dt in months:
        for sid in range(n_names):
            feats = {f: float(rng.normal()) for f in FEATURE_NAMES}
            ret = 0.02 * feats["value_bm"] + rng.normal(scale=0.05)
            rows.append(
                {
                    "date": dt,
                    "security_id": sid,
                    "symbol": f"S{sid}",
                    "mcap": 1e9,
                    "rank": sid + 1,
                    "ret_1m": ret,
                    **feats,
                }
            )
    df = pd.DataFrame(rows)
    df["target_rank"] = df.groupby("date")["ret_1m"].rank(pct=True, method="average")
    for f in FEATURE_NAMES:
        r = df.groupby("date")[f].rank(pct=True, method="average")
        df[f] = (2.0 * r - 1.0).fillna(0.0)
    return df


def _small_model(Model):
    if Model is RandomForestModel:
        m = RandomForestModel(n_estimators=30, n_jobs=1, max_depth=3, min_samples_leaf=50)
        m.param_grid = {"max_depth": [3], "min_samples_leaf": [50], "max_features": [0.5]}
        return m
    if Model is XGBoostModel:
        m = XGBoostModel(n_estimators=40, early_stopping_rounds=5, n_jobs=1)
        m.param_grid = {
            "max_depth": [2],
            "learning_rate": [0.1],
            "min_child_weight": [1],
            "reg_lambda": [1.0],
        }
        return m
    m = MLPModel(n_seeds=2, max_epochs=8, patience=3, batch_size=512)
    m.param_grid = {"hidden": [(16, 8)], "dropout": [0.1], "weight_decay": [1e-3]}
    return m


@pytest.mark.parametrize("Model", [RandomForestModel, XGBoostModel, MLPModel])
def test_nonlinear_synthetic_respects_holdout(Model):
    panel = _tiny_panel()
    max_f = pd.Timestamp("2012-11-30")
    eng = WalkForwardEngine(
        first_oos_year=2011,
        last_oos_year=2012,
        max_formation=max_f,
        validation_months=12,
        n_jobs=1,
    )
    result = eng.run(_small_model(Model), panel)
    assert result.predictions["formation_month"].max() <= max_f
    assert len(result.year_fits) >= 1


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built")
@pytest.mark.parametrize("Model", [RandomForestModel, XGBoostModel, MLPModel])
def test_nonlinear_leakage_shuffle_near_zero(Model):
    """Within-month label shuffle → OOS rank IC ≈ 0 (mean over seeds).

    Threshold is looser than Lasso's 0.005: tree/MLP IC has higher
    finite-sample variance on short OOS windows under shuffle.
    """
    panel = pd.read_parquet(PANEL)
    eng = WalkForwardEngine(
        n_jobs=1,
        validation_months=24,
        first_oos_year=2011,
        last_oos_year=2013,
        max_formation=pd.Timestamp("2013-11-30"),
    )
    model = _small_model(Model)

    ics = []
    for seed in (0, 1, 2):
        shuffled = shuffle_target_within_month(panel, seed=seed)
        result = eng.run(model, shuffled)
        ics.append(
            monthly_rank_ic(
                result.predictions["prediction"],
                result.predictions["realized_ret"],
                result.predictions["formation_month"],
            )
        )
    mean_ic = float(np.nanmean(ics))
    assert abs(mean_ic) < 0.02, f"{Model.name} leakage IC={mean_ic} (seeds={ics})"


@pytest.mark.skipif(not PRED.exists(), reason="predictions not written")
def test_nonlinear_saved_predictions_pre_holdout():
    for name in ("random_forest", "xgboost", "mlp", "ensemble"):
        path = PRED / f"{name}_predictions.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        assert pd.to_datetime(df["formation_month"]).max() <= MAX_FORMATION_MONTH
