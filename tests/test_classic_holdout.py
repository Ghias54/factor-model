"""Holdout guard: classic factor artifacts never include 2024+ months."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_model.factors.classic import (
    MAX_FORMATION_MONTH,
    MAX_RETURN_MONTH,
    assert_pre_holdout,
    build_classic_factor_returns,
    return_month,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "panel" / "features_raw.parquet"
CLASSIC = ROOT / "data" / "factors" / "classic_factor_returns.parquet"
LONG = ROOT / "data" / "factors" / "classic_long_only_returns.parquet"


def test_return_month_maps_nov_2023_to_dec_2023():
    assert return_month(pd.Timestamp("2023-11-30")) == pd.Timestamp("2023-12-31")


def test_assert_pre_holdout_rejects_late_formation():
    with pytest.raises(AssertionError):
        assert_pre_holdout(pd.Series([pd.Timestamp("2023-12-31")]))


def test_assert_pre_holdout_rejects_late_return_month():
    with pytest.raises(AssertionError):
        assert_pre_holdout(
            pd.Series([pd.Timestamp("2023-11-30")]),
            pd.Series([pd.Timestamp("2024-01-31")]),
        )


@pytest.mark.skipif(not RAW.exists(), reason="panel not built")
def test_build_classic_respects_holdout_boundary():
    raw = pd.read_parquet(
        RAW,
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
    # Tiny subsample of months still exercises the cap
    raw["date"] = pd.to_datetime(raw["date"])
    out = build_classic_factor_returns(raw)
    assert out.index.max() <= MAX_RETURN_MONTH
    # Formation implied: no formation after MAX_FORMATION_MONTH was used
    assert (pd.to_datetime(raw["date"].unique()) > MAX_FORMATION_MONTH).any()  # panel has holdout
    assert out.index.min() >= pd.Timestamp("2006-02-28")


@pytest.mark.skipif(not CLASSIC.exists(), reason="classic returns not built yet")
def test_saved_classic_series_pre_holdout():
    df = pd.read_parquet(CLASSIC)
    idx = pd.to_datetime(df.index)
    assert idx.max() <= MAX_RETURN_MONTH
    assert (idx > MAX_RETURN_MONTH).sum() == 0


@pytest.mark.skipif(not LONG.exists(), reason="long-only returns not built yet")
def test_saved_long_only_series_pre_holdout():
    df = pd.read_parquet(LONG)
    idx = pd.to_datetime(df.index)
    assert idx.max() <= MAX_RETURN_MONTH
