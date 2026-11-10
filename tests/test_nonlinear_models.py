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
    monthly_rank_ic_series,
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
    """Fast suite gate: short OOS window, many seeds, assert |t| < 2.

    Absolute |mean IC| < 0.005 is reserved for the full-window script
    ``scripts/run_leakage_check.py`` (155 months × several seeds). On a
    2–3 year window the mean alone is too noisy, so this test pools monthly
    ICs across 5 shuffle seeds and requires |t_mean| < 2 (and |mean| < 0.02
    as a soft sanity bound).
    """
    panel = pd.read_parquet(PANEL)
    eng = WalkForwardEngine(
        n_jobs=1,
        validation_months=24,
        first_oos_year=2011,
        last_oos_year=2012,
        max_formation=pd.Timestamp("2012-11-30"),
    )
    model = _small_model(Model)

    pooled: list[float] = []
    for seed in (0, 1, 2, 3, 4):
        shuffled = shuffle_target_within_month(panel, seed=seed)
        result = eng.run(model, shuffled)
        series = monthly_rank_ic_series(
            result.predictions["prediction"],
            result.predictions["realized_ret"],
            result.predictions["formation_month"],
        )
        pooled.extend(series.dropna().tolist())

    arr = np.asarray(pooled, dtype=float)
    mean_ic = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    t = mean_ic / (sd / np.sqrt(len(arr))) if len(arr) > 1 and sd > 0 else float("nan")
    assert abs(t) < 2.0, f"{Model.name} leakage t={t} mean={mean_ic} n={len(arr)}"
    assert abs(mean_ic) < 0.02, f"{Model.name} leakage mean IC={mean_ic} (sanity)"


@pytest.mark.skipif(not PRED.exists(), reason="predictions not written")
def test_nonlinear_saved_predictions_pre_holdout():
    for name in ("random_forest", "xgboost", "mlp", "ensemble"):
        path = PRED / f"{name}_predictions.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        assert pd.to_datetime(df["formation_month"]).max() <= MAX_FORMATION_MONTH
