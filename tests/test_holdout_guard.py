"""Normal walk-forward / classic / evaluation paths still refuse holdout months.

The only entry point allowed to read formations ≥ 2023-12 is
``scripts/run_holdout.py --i-am-running-the-holdout``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_model.factors.classic import (
    MAX_FORMATION_MONTH,
    MAX_RETURN_MONTH,
    assert_pre_holdout,
    filter_panel,
)
from factor_model.walkforward.engine import LAST_OOS_YEAR, WalkForwardEngine

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel" / "panel.parquet"
PRED = ROOT / "data" / "predictions"
HOLDOUT_FIRST = pd.Timestamp("2023-12-01")


def test_default_engine_caps_at_pre_holdout():
    eng = WalkForwardEngine()
    assert eng.max_formation == MAX_FORMATION_MONTH
    assert eng.last_oos_year == LAST_OOS_YEAR == 2023


def test_assert_pre_holdout_rejects_dec_2023_formation():
    with pytest.raises(AssertionError):
        assert_pre_holdout(pd.Series([pd.Timestamp("2023-12-29")]))


def test_assert_pre_holdout_rejects_jan_2024_return_month():
    with pytest.raises(AssertionError):
        assert_pre_holdout(
            pd.Series([pd.Timestamp("2023-11-30")]),
            pd.Series([pd.Timestamp("2024-01-31")]),
        )


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built")
def test_default_prepare_panel_excludes_holdout_formations():
    panel = pd.read_parquet(PANEL)
    eng = WalkForwardEngine(n_jobs=1)
    df = eng.prepare_panel(panel)
    assert df["date"].max() <= MAX_FORMATION_MONTH
    assert (df["date"] >= HOLDOUT_FIRST).sum() == 0
    # Panel itself still contains holdout months
    assert (pd.to_datetime(panel["date"]) >= HOLDOUT_FIRST).any()


@pytest.mark.skipif(not PANEL.exists(), reason="panel not built")
def test_classic_filter_panel_excludes_holdout():
    raw = pd.read_parquet(
        PANEL,
        columns=[
            "date",
            "security_id",
            "mcap",
            "value_bm",
            "size_ln_mcap",
            "momentum_12_1",
            "quality_roe",
            "ret_1m",
        ],
    )
    out = filter_panel(raw)
    assert out["date"].max() <= MAX_FORMATION_MONTH
    assert out["return_month"].max() <= MAX_RETURN_MONTH
    assert (out["date"] >= HOLDOUT_FIRST).sum() == 0


@pytest.mark.skipif(not PRED.exists(), reason="predictions not written")
def test_oos_prediction_artifacts_still_pre_holdout():
    """Top-level OOS prediction files must remain capped (holdout lives in subdir)."""
    paths = list(PRED.glob("*_predictions.parquet"))
    assert paths, "expected OOS prediction parquet files"
    for p in paths:
        df = pd.read_parquet(p)
        col = "formation_month" if "formation_month" in df.columns else "date"
        if col not in df.columns:
            idx = pd.to_datetime(df.index)
            assert idx.max() <= MAX_RETURN_MONTH
            continue
        assert pd.to_datetime(df[col]).max() <= MAX_FORMATION_MONTH
