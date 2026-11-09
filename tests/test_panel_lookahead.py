"""Lookahead guards for the monthly panel.

Offline unit tests always run. Dev / parquet tests skip when data is absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

from factor_model.data import loaders
from factor_model.features.compute import FEATURE_NAMES, preprocess_panel

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "panel" / "features_raw.parquet"


def test_preprocess_ranks_to_unit_interval_and_fills_missing():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 5 + ["2020-02-29"] * 5),
            "security_id": list(range(5)) * 2,
            "symbol": ["A"] * 10,
            "mcap": 1.0,
            "rank": 1,
            "ret_1m": 0.01,
            "excess_ret_1m": 0.0,
            "target_rank": 0.5,
            "value_bm": [1, 2, 3, 4, np.nan] * 2,
        }
    )
    for c in FEATURE_NAMES:
        if c not in raw:
            raw[c] = np.nan
    out = preprocess_panel(raw)
    assert out["value_bm"].between(-1, 1).all()
    # the missing raw value becomes 0 after fill
    assert (out.loc[raw["value_bm"].isna(), "value_bm"] == 0.0).all()


def test_asof_cutoff_is_1600_et():
    dates = pd.Series(pd.to_datetime(["2020-01-31", "2020-06-30"]))
    cut = loaders.asof_cutoff(dates)
    local = cut.dt.tz_convert("America/New_York")
    assert list(local.dt.strftime("%H:%M")) == ["16:00", "16:00"]


@pytest.mark.skipif(not RAW.exists(), reason="features_raw.parquet not built yet")
def test_parquet_fundamentals_and_earnings_not_lookahead():
    raw = pd.read_parquet(RAW, columns=["date", "security_id", "fund_date_available", "report_date"])
    sample = raw.sample(n=min(200, len(raw)), random_state=42)
    has_fund = sample["fund_date_available"].notna()
    if has_fund.any():
        cut = loaders.asof_cutoff(sample.loc[has_fund, "date"])
        avail = pd.to_datetime(sample.loc[has_fund, "fund_date_available"], utc=True)
        assert (avail <= cut).all()
    has_earn = sample["report_date"].notna()
    if has_earn.any():
        assert (
            pd.to_datetime(sample.loc[has_earn, "report_date"])
            < pd.to_datetime(sample.loc[has_earn, "date"])
        ).all()


@pytest.mark.skipif(not os.environ.get("STOCKDB_DEV_DSN"), reason="STOCKDB_DEV_DSN not set")
def test_dev_asof_fundamentals_respects_cutoff():
    me = loaders.month_ends("2018-01-01", "2019-12-31", dev=True)
    if len(me) == 0:
        pytest.skip("no month ends")
    fund = loaders.fundamentals(dev=True)
    # AAPL if present
    sec = loaders.securities(dev=True)
    aapl = sec.loc[sec["last_symbol"] == "AAPL", "security_id"]
    if aapl.empty:
        pytest.skip("no AAPL in dev")
    keys = pd.DataFrame({"security_id": int(aapl.iloc[0]), "date": me})
    out = loaders.asof_fundamentals(keys, fund)
    cut = loaders.asof_cutoff(out["date"])
    ok = out["date_available"].notna()
    assert (out.loc[ok, "date_available"] <= cut[ok]).all()
