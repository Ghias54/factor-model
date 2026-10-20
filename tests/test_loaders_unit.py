"""Offline tests for point-in-time join helpers (no database)."""
import pandas as pd

from factor_model.data import loaders


def _fund(rows):
    df = pd.DataFrame(rows, columns=["security_id", "fiscal_period_end", "fiscal_period", "date_available", "total_equity"])
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    df["date_available"] = pd.to_datetime(df["date_available"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return df


def test_guard_reimputes_period_end_availability():
    fund = _fund(
        [
            (1, "1990-03-31", "Q1", "1990-03-31 00:00", 10.0),
            (1, "1990-12-31", "FY", "1990-12-31 00:00", 11.0),
            (1, "2020-03-31", "Q1", "2020-05-01 16:05", 12.0),
        ]
    )
    out = loaders.guard_zero_lag(fund)
    local = out["date_available"].dt.tz_convert("America/New_York")
    assert local.iloc[0] == pd.Timestamp("1990-05-15 16:00", tz="America/New_York")
    assert local.iloc[1] == pd.Timestamp("1991-03-31 16:00", tz="America/New_York")
    assert local.iloc[2] == pd.Timestamp("2020-05-01 16:05", tz="America/New_York")
    assert out["zero_lag_reimputed"].tolist() == [True, True, False]


def test_asof_respects_16et_cutoff_and_ignores_future_rows():
    fund = _fund(
        [
            (1, "2020-03-31", "Q1", "2020-04-30 15:59", 1.0),
            (1, "2020-06-30", "Q2", "2020-07-31 16:01", 2.0),
            (1, "2020-09-30", "Q3", "2020-10-30 09:00", 3.0),
        ]
    )
    keys = pd.DataFrame({"security_id": 1, "date": pd.to_datetime(["2020-04-30", "2020-07-31", "2020-08-31"])})
    out = loaders.asof_fundamentals(keys, fund).set_index("date")["total_equity"]
    assert out.tolist() == [1.0, 1.0, 2.0]

    # Lookahead check: rows made public later can never change earlier values.
    shifted = fund.copy()
    shifted.loc[2, "date_available"] += pd.Timedelta(days=400)
    out2 = loaders.asof_fundamentals(keys, shifted).set_index("date")["total_equity"]
    assert out2.tolist() == out.tolist()


def test_asof_earnings_excludes_same_day_report():
    ev = pd.DataFrame(
        {
            "security_id": [1, 1],
            "report_date": pd.to_datetime(["2020-07-30", "2020-07-31"]),
            "eps_actual": [1.0, 2.0],
            "eps_estimated": [0.9, 1.8],
        }
    )
    keys = pd.DataFrame({"security_id": [1], "date": pd.to_datetime(["2020-07-31"])})
    out = loaders.asof_earnings(keys, ev)
    assert out["eps_actual"].iloc[0] == 1.0
