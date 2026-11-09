"""Walk-forward engine: timing, holdout, leakage, synthetic unit tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_model.factors.classic import MAX_FORMATION_MONTH
from factor_model.features.compute import FEATURE_NAMES
from factor_model.models.linear import LassoModel, RidgeModel
from factor_model.walkforward.engine import (
    WalkForwardEngine,
    monthly_rank_ic,
    scale_target_rank,
    shuffle_target_within_month,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel" / "panel.parquet"
PRED_DIR = ROOT / "data" / "predictions"


def _synthetic_panel(
    n_months: int = 96,
    n_names: int = 40,
    start: str = "2006-01-31",
    seed: int = 0,
) -> pd.DataFrame:
    """Tiny panel with a planted linear signal in value_bm."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, periods=n_months, freq="ME")
    rows = []
    for dt in months:
        for sid in range(n_names):
            feats = {f: float(rng.normal()) for f in FEATURE_NAMES}
            # Planted signal: next-month return correlates with value_bm
            noise = rng.normal(scale=0.05)
            ret = 0.02 * feats["value_bm"] + noise
            rows.append(
                {
                    "date": dt,
                    "security_id": sid,
                    "symbol": f"S{sid}",
                    "mcap": 1e9 * (1 + abs(rng.normal())),
                    "rank": sid + 1,
                    "ret_1m": ret,
                    **feats,
                }
            )
    df = pd.DataFrame(rows)
    df["target_rank"] = df.groupby("date")["ret_1m"].rank(pct=True, method="average")
    # Rank-scale features like the real panel
    for f in FEATURE_NAMES:
        r = df.groupby("date")[f].rank(pct=True, method="average")
        df[f] = (2.0 * r - 1.0).fillna(0.0)
    return df


def test_scale_target_rank_matches_feature_scale():
    s = pd.Series([0.0, 0.5, 1.0])
    out = scale_target_rank(s)
    assert list(out) == pytest.approx([-1.0, 0.0, 1.0])


def test_engine_synthetic_runs_and_respects_holdout_cap():
    panel = _synthetic_panel(n_months=100, n_names=30)
    # Cap artificially inside the synthetic range
    max_f = pd.Timestamp("2013-11-30")
    eng = WalkForwardEngine(
        first_oos_year=2011,
        last_oos_year=2013,
        max_formation=max_f,
        validation_months=12,
        n_jobs=1,
    )
    result = eng.run(RidgeModel(alpha=1.0), panel)
    assert result.predictions["formation_month"].max() <= max_f
    assert result.predictions["formation_month"].min() >= pd.Timestamp("2011-01-31")
    assert len(result.year_fits) >= 1


def test_timing_assert_rejects_overlapping_train_pred():
    eng = WalkForwardEngine(n_jobs=1)
    with pytest.raises(AssertionError):
        eng.assert_timing(
            [pd.Timestamp("2010-12-31"), pd.Timestamp("2011-01-31")],
            pd.Timestamp("2011-01-31"),
        )


def test_timing_assert_accepts_dec_before_jan():
    eng = WalkForwardEngine(n_jobs=1)
    eng.assert_timing(
        [pd.Timestamp("2010-01-31"), pd.Timestamp("2010-12-31")],
        pd.Timestamp("2011-01-31"),
    )


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built")
def test_real_panel_timing_and_holdout_for_each_refit_year():
    panel = pd.read_parquet(PANEL)
    eng = WalkForwardEngine(n_jobs=1, validation_months=36)
    df = eng.prepare_panel(panel)
    calendar = eng._formation_calendar(df)
    for year, train_end, first_pred, pred_dates in eng._year_windows(calendar):
        train_dates = calendar[(calendar >= eng.sample_start) & (calendar <= train_end)]
        eng.assert_timing(train_dates, first_pred)
        assert pred_dates.max() <= MAX_FORMATION_MONTH
        assert train_dates.max() < first_pred
        assert train_dates.max().year == year - 1 or train_dates.max() < first_pred


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built")
def test_leakage_shuffle_oos_ic_near_zero():
    """Within-month label shuffle → OOS rank IC ≈ 0 (mean over seeds)."""
    panel = pd.read_parquet(PANEL)
    eng = WalkForwardEngine(
        n_jobs=-1,
        validation_months=24,
        first_oos_year=2011,
        last_oos_year=2013,
        max_formation=pd.Timestamp("2013-11-30"),
    )
    model = LassoModel(alpha=0.01)
    model.param_grid = {"alpha": [0.001, 0.01, 0.1]}
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
    assert abs(mean_ic) < 0.005, f"leakage mean IC too large: {mean_ic} (seeds={ics})"


@pytest.mark.skipif(not PRED_DIR.exists(), reason="predictions not written yet")
def test_saved_predictions_pre_holdout():
    paths = [p for p in PRED_DIR.glob("*_predictions.parquet")]
    assert paths, "expected prediction parquet files"
    for p in paths:
        df = pd.read_parquet(p)
        if "formation_month" in df.columns:
            col = "formation_month"
        elif "date" in df.columns:
            col = "date"
        else:
            # portfolio returns file indexed by return month
            idx = pd.to_datetime(df.index)
            assert idx.max() <= pd.Timestamp("2023-12-31")
            continue
        assert pd.to_datetime(df[col]).max() <= MAX_FORMATION_MONTH
