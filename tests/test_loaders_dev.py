"""Smoke tests for data loaders against the 5-ticker dev database.

Skipped unless STOCKDB_DEV_DSN is set.
"""
import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from factor_model.data import loaders

load_dotenv()
pytestmark = pytest.mark.skipif(
    not os.environ.get("STOCKDB_DEV_DSN"), reason="STOCKDB_DEV_DSN not set"
)


@pytest.fixture(scope="module")
def aapl_id():
    sec = loaders.securities(dev=True)
    ids = sec.loc[sec["last_symbol"] == "AAPL", "security_id"]
    assert len(ids) == 1
    return int(ids.iloc[0])


def test_month_ends_are_ordered_sessions():
    me = loaders.month_ends("2020-01-01", "2020-12-31", dev=True)
    assert len(me) == 12
    assert me.is_monotonic_increasing
    assert me.iloc[0] == pd.Timestamp("2020-01-31")


def test_prices_on_month_ends(aapl_id):
    me = loaders.month_ends("2020-01-01", "2020-12-31", dev=True)
    px = loaders.prices_on_dates(me, dev=True)
    aapl = px[px["security_id"] == aapl_id].set_index("date")
    assert len(aapl) == 12
    # 4:1 split on 2020-08-31: raw price drops, split-adjusted close does not.
    assert aapl.loc["2020-07-31", "close_raw"] > 3 * aapl.loc["2020-07-31", "close"]


def test_fundamentals_pit_join_never_uses_future_rows(aapl_id):
    fund = loaders.fundamentals(dev=True)
    assert {"date_available", "fiscal_period_end", "total_equity"} <= set(fund.columns)
    keys = pd.DataFrame({"security_id": aapl_id, "date": loaders.month_ends("2015-01-01", "2020-12-31", dev=True)})
    out = loaders.asof_fundamentals(keys, fund)
    cutoff = loaders.asof_cutoff(out["date"])
    assert (out["date_available"].dropna() <= cutoff[out["date_available"].notna()]).all()
    assert out["total_equity"].notna().mean() > 0.9


def test_earnings_asof_is_strictly_before(aapl_id):
    ev = loaders.earnings_events(dev=True)
    keys = pd.DataFrame({"security_id": aapl_id, "date": loaders.month_ends("2018-01-01", "2020-12-31", dev=True)})
    out = loaders.asof_earnings(keys, ev)
    has = out["report_date"].notna()
    assert has.mean() > 0.9
    assert (out.loc[has, "report_date"] < out.loc[has, "date"]).all()


def test_listed_on_keeps_dead_names():
    sec = loaders.securities(dev=True)
    dead = sec[sec["last_trade_date"].notna()]
    if dead.empty:
        pytest.skip("no delisted security in dev")
    row = dead.iloc[0]
    day = row["last_trade_date"] - pd.Timedelta(days=1)
    pairs = loaders.listed_on(sec, pd.Series([day]))
    assert row["security_id"] in set(pairs["security_id"])
