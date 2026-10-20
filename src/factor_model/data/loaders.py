"""Batched, read-only loaders for the stock-database tables this project uses.

One query per table per call; no per-symbol loops. Point-in-time joins happen in
pandas (``asof_fundamentals``, ``asof_earnings``) so the availability clock is
explicit and testable. See docs/data-dictionary.md for table semantics.

Price basis (verified on AAPL, see docs/data-audit.md):
- ``close``: split-adjusted to today's share basis. Pair with statement share
  counts, which FMP also restates to today's split basis.
- ``close_raw``: price as actually traded on that day. Use for price filters.
- ``close_adj``: split- and dividend-adjusted. Use for returns only.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import time

import numpy as np
import pandas as pd

from factor_model.data.db import read_sql

NY_TZ = "America/New_York"

FUNDAMENTAL_COLUMNS = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "eps_diluted",
    "weighted_average_shares",
    "weighted_average_shares_diluted",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_debt",
    "cash_from_operations",
)

_ALLOWED_FUNDAMENTAL_COLUMNS = frozenset(
    FUNDAMENTAL_COLUMNS
    + (
        "cost_of_revenue",
        "research_and_development",
        "selling_general_and_admin",
        "interest_expense",
        "ebitda",
        "income_before_tax",
        "income_tax_expense",
        "eps",
        "cash_and_equivalents",
        "short_term_investments",
        "accounts_receivable",
        "inventory",
        "total_current_assets",
        "ppe_net",
        "goodwill",
        "intangible_assets",
        "accounts_payable",
        "short_term_debt",
        "total_current_liabilities",
        "long_term_debt",
        "depreciation_and_amortization",
        "stock_based_compensation",
        "change_in_working_capital",
        "capex",
        "cash_from_investing",
        "dividends_paid",
        "cash_from_financing",
        "free_cash_flow",
    )
)


def month_ends(
    start: str | None = None,
    end: str | None = None,
    complete_only: bool = True,
    dev: bool = False,
) -> pd.Series:
    """Last XNYS session of each calendar month, ascending.

    ``trading_day`` stops at the last loaded session and flags it ``is_month_end``
    even mid-month (2026-09-17). ``complete_only`` drops such a partial month.
    """
    df = read_sql(
        """
        SELECT date, date = (SELECT max(date) FROM trading_day) AS is_last_loaded
        FROM trading_day
        WHERE is_month_end
          AND (CAST(:start AS date) IS NULL OR date >= CAST(:start AS date))
          AND (CAST(:end AS date) IS NULL OR date <= CAST(:end AS date))
        ORDER BY date
        """,
        {"start": start, "end": end},
        dev=dev,
    )
    dates = pd.to_datetime(df["date"])
    if complete_only:
        last_bday = dates + pd.offsets.BMonthEnd(0)
        partial = df["is_last_loaded"] & ((last_bday - dates).dt.days > 3)
        dates = dates[~partial]
    return dates.rename("date").reset_index(drop=True)


def securities(dev: bool = False) -> pd.DataFrame:
    """Every security ever in the universe, including delisted names.

    ``first_symbol``/``last_symbol`` are for display only; join on ``security_id``.
    """
    df = read_sql(
        """
        SELECT s.security_id, s.company_id, s.share_class, s.is_active,
               s.first_trade_date, s.last_trade_date,
               c.cik, c.name AS company_name,
               (SELECT sh.symbol FROM symbol_history sh WHERE sh.security_id = s.security_id
                ORDER BY sh.valid_from LIMIT 1) AS first_symbol,
               (SELECT sh.symbol FROM symbol_history sh WHERE sh.security_id = s.security_id
                ORDER BY sh.valid_from DESC LIMIT 1) AS last_symbol
        FROM security s
        JOIN company c USING (company_id)
        ORDER BY s.security_id
        """,
        dev=dev,
    )
    for col in ("first_trade_date", "last_trade_date"):
        df[col] = pd.to_datetime(df[col])
    return df


def symbol_history(dev: bool = False) -> pd.DataFrame:
    """Ticker ranges, half-open ``[valid_from, valid_to)``."""
    df = read_sql("SELECT security_id, symbol, valid_from, valid_to FROM symbol_history", dev=dev)
    for col in ("valid_from", "valid_to"):
        df[col] = pd.to_datetime(df[col])
    return df


def prices_on_dates(dates: Sequence, dev: bool = False) -> pd.DataFrame:
    """``price_daily`` rows for the given sessions (e.g. month-ends), all securities."""
    dates = [pd.Timestamp(d).date() for d in dates]
    df = read_sql(
        """
        SELECT security_id, date, close, close_raw, close_adj, volume, price_quality
        FROM price_daily
        WHERE date = ANY(:dates)
        """,
        {"dates": dates},
        dev=dev,
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ("close", "close_raw", "close_adj", "volume"):
        df[col] = df[col].astype("float64")
    return df


def price_span(dev: bool = False) -> pd.DataFrame:
    """Per security: first/last session with a close, and the final close values."""
    df = read_sql(
        """
        WITH span AS (
            SELECT security_id, min(date) AS first_price_date, max(date) AS last_price_date,
                   count(*) AS n_price_days
            FROM price_daily
            WHERE close IS NOT NULL
            GROUP BY security_id
        )
        SELECT span.*, p.close AS last_close, p.close_raw AS last_close_raw,
               p.close_adj AS last_close_adj, p.volume AS last_volume
        FROM span
        JOIN price_daily p
          ON p.security_id = span.security_id AND p.date = span.last_price_date
        """,
        dev=dev,
    )
    for col in ("first_price_date", "last_price_date"):
        df[col] = pd.to_datetime(df[col])
    for col in ("last_close", "last_close_raw", "last_close_adj", "last_volume"):
        df[col] = df[col].astype("float64")
    return df


def guard_zero_lag(
    fund: pd.DataFrame,
    min_lag_days: int = 1,
    quarter_lag_days: int = 45,
    annual_lag_days: int = 90,
) -> pd.DataFrame:
    """Re-impute availability for rows public less than ``min_lag_days`` after period end.

    ~45k rows carry date_available = 00:00 ET on fiscal_period_end (vendor
    filingDate = period end, labelled fmp_accepted). No filer publishes on the
    period-end day, so these get the stock-database imputed_lag convention:
    period end + 45d (Q1–Q3) or + 90d (Q4/FY) at 16:00 ET.
    """
    out = fund.copy()
    avail_et = out["date_available"].dt.tz_convert(NY_TZ).dt.tz_localize(None).dt.normalize()
    lag = (avail_et - out["fiscal_period_end"]).dt.days
    bad = lag < min_lag_days
    annual = out["fiscal_period"].isin(["Q4", "FY"])
    days = pd.Series(quarter_lag_days, index=out.index).where(~annual, annual_lag_days)
    imputed = (out["fiscal_period_end"] + pd.to_timedelta(days, unit="D") + pd.Timedelta(hours=16))
    imputed = imputed.dt.tz_localize(NY_TZ).dt.tz_convert("UTC")
    out.loc[bad, "date_available"] = imputed[bad]
    out["zero_lag_reimputed"] = bad.astype("boolean")
    if "date_source" in out:
        out.loc[bad, "date_source"] = "imputed_lag_guard"
    return out


def fundamentals(
    columns: Sequence[str] = FUNDAMENTAL_COLUMNS,
    quarterly_only: bool = True,
    guard: bool = True,
    dev: bool = False,
) -> pd.DataFrame:
    """Statement rows with their availability clock. Excludes vendor placeholder dates.

    ``quarterly_only`` keeps Q1–Q4 rows; FY rows share period ends with Q4 and
    carry annual flows, so mixing them breaks flow-variable features.
    ``guard`` applies ``guard_zero_lag``.
    """
    bad = set(columns) - _ALLOWED_FUNDAMENTAL_COLUMNS
    if bad:
        raise ValueError(f"unknown fundamental columns: {sorted(bad)}")
    cols = ", ".join(columns)
    period_filter = "AND fiscal_period LIKE 'Q%'" if quarterly_only else ""
    df = read_sql(
        f"""
        SELECT security_id, fiscal_period_end, fiscal_period, date_available,
               date_source, filer_type, {cols}
        FROM fundamental_quarterly
        WHERE NOT accepted_date_placeholder {period_filter}
        """,
        dev=dev,
    )
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    df["date_available"] = pd.to_datetime(df["date_available"], utc=True)
    for col in columns:
        df[col] = df[col].astype("float64")
    return guard_zero_lag(df) if guard else df


def asof_cutoff(dates: pd.Series, cutoff: time = time(16, 0)) -> pd.Series:
    """Month-end session date -> UTC timestamp of the information cutoff (default 16:00 ET)."""
    local = pd.to_datetime(dates).dt.normalize() + pd.Timedelta(
        hours=cutoff.hour, minutes=cutoff.minute
    )
    return local.dt.tz_localize(NY_TZ).dt.tz_convert("UTC")


def asof_fundamentals(
    keys: pd.DataFrame,
    fund: pd.DataFrame,
    max_staleness_days: int | None = 548,
    cutoff: time = time(16, 0),
) -> pd.DataFrame:
    """Latest statement row public by ``cutoff`` ET on each ``keys.date``.

    ``keys`` has ``security_id`` and ``date``. Ties on ``date_available`` (e.g. a
    10-K that also restates a quarter) resolve to the latest fiscal period.
    Rows older than ``max_staleness_days`` (by availability) are dropped.
    """
    left = keys[["security_id", "date"]].copy()
    left["security_id"] = left["security_id"].astype("int64")
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    left["_cutoff"] = asof_cutoff(left["date"], cutoff).astype("datetime64[ns, UTC]")
    # merge_asof requires the on-key to be globally sorted (by-groups stay ordered).
    left = left.sort_values("_cutoff", kind="mergesort")
    right = fund.copy()
    right["security_id"] = right["security_id"].astype("int64")
    right["date_available"] = pd.to_datetime(right["date_available"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    right = right.sort_values(
        ["date_available", "fiscal_period_end"], kind="mergesort"
    ).drop_duplicates(["security_id", "date_available"], keep="last")
    out = pd.merge_asof(
        left,
        right,
        left_on="_cutoff",
        right_on="date_available",
        by="security_id",
        direction="backward",
    )
    if max_staleness_days is not None:
        stale = out["date_available"] < out["_cutoff"] - pd.Timedelta(days=max_staleness_days)
        value_cols = [c for c in right.columns if c not in ("security_id",)]
        out.loc[stale, value_cols] = pd.NA
    return out.drop(columns="_cutoff")


def earnings_events(dev: bool = False) -> pd.DataFrame:
    """Reported EPS/revenue vs the vendor consensus, one row per report.

    ``report_date`` is the announcement day; timing (BMO/AMC) is in
    ``earnings_event_timing_best`` and is often unknown, so treat the surprise
    as public on the session *after* ``report_date`` unless timing says BMO.
    """
    df = read_sql(
        """
        SELECT e.event_id, e.security_id, e.report_date, e.eps_actual, e.eps_estimated,
               e.revenue_actual, e.revenue_estimated, e.last_updated, t.timing
        FROM earnings_event e
        LEFT JOIN earnings_event_timing_best t USING (event_id)
        """,
        dev=dev,
    )
    for col in ("report_date", "last_updated"):
        df[col] = pd.to_datetime(df[col])
    for col in ("eps_actual", "eps_estimated", "revenue_actual", "revenue_estimated"):
        df[col] = df[col].astype("float64")
    return df


def asof_earnings(
    keys: pd.DataFrame, events: pd.DataFrame, max_age_days: int | None = 120
) -> pd.DataFrame:
    """Latest earnings event strictly before each ``keys.date`` (report_date < date)."""
    left = keys[["security_id", "date"]].sort_values("date")
    right = events.dropna(subset=["report_date"]).sort_values("report_date")
    out = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="report_date",
        by="security_id",
        direction="backward",
        allow_exact_matches=False,
    )
    if max_age_days is not None:
        stale = out["report_date"] < out["date"] - pd.Timedelta(days=max_age_days)
        value_cols = [c for c in right.columns if c != "security_id"]
        out.loc[stale, value_cols] = pd.NA
    return out


def analyst_estimates(dev: bool = False) -> pd.DataFrame:
    """Vendor consensus by target period. NOT point-in-time: one current snapshot
    per (security, target_period), ``date_available`` is NULL. Audit use only."""
    df = read_sql(
        """
        SELECT security_id, target_period, period_type, date_available, analyst_count,
               eps_avg, eps_low, eps_high, revenue_avg
        FROM analyst_estimate
        """,
        dev=dev,
    )
    df["target_period"] = pd.to_datetime(df["target_period"])
    return df


def vendor_market_cap_on_dates(dates: Sequence, dev: bool = False) -> pd.DataFrame:
    """Vendor ``historical-market-capitalization`` on the given sessions (reconciliation only)."""
    dates = [pd.Timestamp(d).date() for d in dates]
    df = read_sql(
        """
        SELECT security_id, date, market_cap
        FROM market.market_cap_daily
        WHERE date = ANY(:dates)
        """,
        {"dates": dates},
        dev=dev,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["market_cap"] = df["market_cap"].astype("float64")
    return df


def listed_on(sec: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """(date, security_id) pairs where first_trade_date <= date <= last_trade_date.

    Uses only listing dates, never today's status, so delisted names stay in.
    """
    sec = sec[["security_id", "first_trade_date", "last_trade_date"]]
    d = pd.DataFrame({"date": pd.to_datetime(dates)})
    pairs = d.merge(sec, how="cross")
    ok = (pairs["first_trade_date"].isna() | (pairs["first_trade_date"] <= pairs["date"])) & (
        pairs["last_trade_date"].isna() | (pairs["last_trade_date"] >= pairs["date"])
    )
    return pairs.loc[ok, ["date", "security_id"]].reset_index(drop=True)


def prices_daily(
    start: str,
    end: str,
    security_ids: Sequence[int] | None = None,
    columns: Sequence[str] = ("close", "close_raw", "close_adj", "volume", "price_quality"),
    dev: bool = False,
) -> pd.DataFrame:
    """Daily bars in ``[start, end]``. Optionally restrict to ``security_ids``."""
    cols = ", ".join(["security_id", "date", *columns])
    if security_ids is None:
        df = read_sql(
            f"""
            SELECT {cols} FROM price_daily
            WHERE date BETWEEN CAST(:start AS date) AND CAST(:end AS date)
            """,
            {"start": start, "end": end},
            dev=dev,
        )
    else:
        ids = [int(x) for x in security_ids]
        df = read_sql(
            f"""
            SELECT {cols} FROM price_daily
            WHERE date BETWEEN CAST(:start AS date) AND CAST(:end AS date)
              AND security_id = ANY(:ids)
            """,
            {"start": start, "end": end, "ids": ids},
            dev=dev,
        )
    df["date"] = pd.to_datetime(df["date"])
    for col in columns:
        if col != "price_quality":
            df[col] = df[col].astype("float64")
    return df


def index_prices(
    symbol: str = "^GSPC",
    start: str | None = None,
    end: str | None = None,
    dev: bool = False,
) -> pd.DataFrame:
    """Index EOD from ``market.index_price_daily`` (not joined to ``security``)."""
    df = read_sql(
        """
        SELECT date, close, close_adj, price_quality
        FROM market.index_price_daily
        WHERE symbol = :symbol
          AND (CAST(:start AS date) IS NULL OR date >= CAST(:start AS date))
          AND (CAST(:end AS date) IS NULL OR date <= CAST(:end AS date))
        ORDER BY date
        """,
        {"symbol": symbol, "start": start, "end": end},
        dev=dev,
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ("close", "close_adj"):
        df[col] = df[col].astype("float64")
    return df


def treasury_rates(start: str | None = None, end: str | None = None, dev: bool = False) -> pd.DataFrame:
    """Daily Treasury curve; use ``month3`` as the risk-free rate."""
    df = read_sql(
        """
        SELECT date, month1, month3, year1, year10
        FROM treasury_rate
        WHERE (CAST(:start AS date) IS NULL OR date >= CAST(:start AS date))
          AND (CAST(:end AS date) IS NULL OR date <= CAST(:end AS date))
        ORDER BY date
        """,
        {"start": start, "end": end},
        dev=dev,
    )
    df["date"] = pd.to_datetime(df["date"])
    for col in ("month1", "month3", "year1", "year10"):
        df[col] = df[col].astype("float64")
    return df


def symbol_asof(keys: pd.DataFrame, history: pd.DataFrame | None = None, dev: bool = False) -> pd.Series:
    """Ticker valid on each ``keys.date`` (half-open ``[valid_from, valid_to)``)."""
    hist = history if history is not None else symbol_history(dev=dev)
    left = keys[["security_id", "date"]].copy().reset_index(drop=True)
    left["date"] = pd.to_datetime(left["date"]).astype("datetime64[ns]")
    left["security_id"] = left["security_id"].astype("int64")
    left["_i"] = np.arange(len(left))
    right = hist.copy()
    right["valid_from"] = pd.to_datetime(right["valid_from"]).astype("datetime64[ns]")
    right["valid_to"] = pd.to_datetime(right["valid_to"]).astype("datetime64[ns]")
    right["security_id"] = right["security_id"].astype("int64")
    right["valid_to"] = right["valid_to"].fillna(pd.Timestamp("2262-04-11"))
    # Interval join: one history row per security covers most names; still cheap.
    m = left.merge(right, on="security_id", how="left")
    ok = (m["date"] >= m["valid_from"]) & (m["date"] < m["valid_to"])
    m = m.loc[ok, ["_i", "symbol"]].drop_duplicates("_i", keep="last")
    out = left[["_i"]].merge(m, on="_i", how="left")
    return out.sort_values("_i")["symbol"].reset_index(drop=True)


def asof_last_n_quarters(
    keys: pd.DataFrame,
    fund: pd.DataFrame,
    n: int = 4,
    max_staleness_days: int | None = 548,
    cutoff: time = time(16, 0),
    max_gap_days: int = 120,
) -> pd.DataFrame:
    """Last ``n`` quarterly rows public by cutoff for each key, wide-formatted.

    Returns columns ``q0_<col>`` (latest) … ``q{n-1}_<col>``, plus ``ttm_ok``
    (True when all ``n`` rows exist and consecutive fiscal ends are ≤ ``max_gap_days`` apart).
    """
    left = keys[["security_id", "date"]].copy().reset_index(drop=True)
    left["_cutoff"] = asof_cutoff(left["date"], cutoff)
    left["_row"] = left.index
    value_cols = [
        c
        for c in fund.columns
        if c
        not in (
            "security_id",
            "date_available",
            "date_source",
            "filer_type",
            "zero_lag_reimputed",
        )
    ]
    out_frames: list[pd.DataFrame] = []
    fund_sorted = fund.sort_values(["security_id", "date_available", "fiscal_period_end"])
    fund_sorted["date_available"] = pd.to_datetime(fund_sorted["date_available"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    fund_sorted["fiscal_period_end"] = pd.to_datetime(fund_sorted["fiscal_period_end"]).astype(
        "datetime64[ns]"
    )
    fund_by = {sid: g for sid, g in fund_sorted.groupby("security_id", sort=False)}
    for sid, gkeys in left.groupby("security_id", sort=False):
        gf = fund_by.get(sid)
        if gf is None or gf.empty:
            continue
        avail = (
            pd.to_datetime(gf["date_available"], utc=True)
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )
        fpe = gf["fiscal_period_end"].to_numpy(dtype="datetime64[ns]")
        vals = {c: gf[c].to_numpy() for c in value_cols}
        rows = []
        for ridx, cut in zip(gkeys["_row"].to_numpy(), gkeys["_cutoff"].to_numpy(), strict=True):
            cut_ns = np.datetime64(pd.Timestamp(cut).tz_convert("UTC").tz_localize(None), "ns")
            # rightmost index with date_available <= cut
            hi = int(np.searchsorted(avail, cut_ns, side="right"))
            if hi == 0:
                continue
            end = hi - 1
            if max_staleness_days is not None and avail[end] < cut_ns - np.timedelta64(
                max_staleness_days, "D"
            ):
                continue
            # walk back by fiscal_period_end among rows with avail <= cut
            idxs = []
            last_fpe = None
            for j in range(end, -1, -1):
                if last_fpe is not None and fpe[j] >= last_fpe:
                    continue  # same or later period (duplicate avail)
                if last_fpe is not None and (last_fpe - fpe[j]) > np.timedelta64(max_gap_days, "D"):
                    break
                if last_fpe is not None and (last_fpe - fpe[j]) < np.timedelta64(1, "D"):
                    continue
                idxs.append(j)
                last_fpe = fpe[j]
                if len(idxs) == n:
                    break
            rec: dict = {"_row": int(ridx), "ttm_ok": len(idxs) == n}
            for k, j in enumerate(idxs):
                for c in value_cols:
                    rec[f"q{k}_{c}"] = vals[c][j]
                rec[f"q{k}_date_available"] = avail[j]
                rec[f"q{k}_fiscal_period_end"] = fpe[j]
            rows.append(rec)
        if rows:
            out_frames.append(pd.DataFrame(rows))
    if not out_frames:
        empty = left[["_row"]].copy()
        empty["ttm_ok"] = False
        return empty
    wide = pd.concat(out_frames, ignore_index=True)
    return left[["_row", "security_id", "date"]].merge(wide, on="_row", how="left").drop(
        columns="_row"
    )
