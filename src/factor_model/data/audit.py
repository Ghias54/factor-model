"""Data audit of the stock-database: table stats, monthly universe counts, coverage.

Read-only. Every table is read with one batched query; independent queries run
in parallel threads. Results feed docs/data-audit.md.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from factor_model.data import loaders
from factor_model.data.db import read_sql

# One row per table: rows, date range on the column you would join on,
# distinct securities, plus a few table-specific null/flag counts.
TABLE_STATS_SQL: dict[str, str] = {
    "public.security": """
        SELECT count(*) AS n_rows, min(first_trade_date)::text AS date_min,
               max(last_trade_date)::text AS date_max, count(*) AS n_securities,
               count(*) FILTER (WHERE last_trade_date IS NOT NULL) AS n_delisted,
               count(*) FILTER (WHERE first_trade_date IS NULL) AS n_no_first_trade_date
        FROM security""",
    "public.symbol_history": """
        SELECT count(*) AS n_rows, min(valid_from)::text AS date_min, max(valid_from)::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM symbol_history""",
    "public.company_attribute_history": """
        SELECT count(*) AS n_rows, min(valid_from)::text AS date_min, max(valid_from)::text AS date_max,
               count(DISTINCT company_id) AS n_companies,
               count(*) FILTER (WHERE valid_to IS NOT NULL) AS n_closed_versions
        FROM company_attribute_history""",
    "public.price_daily": """
        SELECT count(*) AS n_rows, min(date)::text AS date_min, max(date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities,
               count(*) FILTER (WHERE close IS NULL) AS n_close_null,
               count(*) FILTER (WHERE close_raw IS NULL) AS n_close_raw_null,
               count(*) FILTER (WHERE close_adj IS NULL) AS n_close_adj_null,
               count(*) FILTER (WHERE price_quality = 'suspect') AS n_suspect,
               count(*) FILTER (WHERE split_repair_factor IS NOT NULL) AS n_split_repaired,
               count(*) FILTER (WHERE volume IS NULL OR volume = 0) AS n_zero_or_null_volume
        FROM price_daily""",
    "public.corporate_action": """
        SELECT count(*) AS n_rows, min(ex_date)::text AS date_min, max(ex_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities,
               count(*) FILTER (WHERE action_type = 'split') AS n_splits
        FROM corporate_action""",
    "market.market_cap_daily": """
        SELECT count(*) AS n_rows, min(date)::text AS date_min, max(date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities,
               count(*) FILTER (WHERE market_cap IS NULL) AS n_null
        FROM market.market_cap_daily""",
    "public.fundamental_quarterly": """
        SELECT count(*) AS n_rows, min(date_available)::date::text AS date_min,
               max(date_available)::date::text AS date_max,
               count(DISTINCT security_id) AS n_securities,
               min(fiscal_period_end)::text AS fpe_min, max(fiscal_period_end)::text AS fpe_max,
               count(*) FILTER (WHERE fiscal_period LIKE 'Q%') AS n_quarterly,
               count(*) FILTER (WHERE fiscal_period = 'FY') AS n_annual,
               count(*) FILTER (WHERE accepted_date_placeholder) AS n_placeholder,
               count(*) FILTER (WHERE date_source = 'imputed_lag') AS n_imputed_lag,
               count(*) FILTER (WHERE date_source IS NULL) AS n_date_source_null
        FROM fundamental_quarterly""",
    "public.analyst_estimate": """
        SELECT count(*) AS n_rows, min(target_period)::text AS date_min,
               max(target_period)::text AS date_max, count(DISTINCT security_id) AS n_securities,
               count(*) FILTER (WHERE date_available IS NULL) AS n_date_available_null,
               count(*) FILTER (WHERE period_type = 'annual') AS n_annual,
               count(*) FILTER (WHERE period_type = 'quarter') AS n_quarter
        FROM analyst_estimate""",
    "public.earnings_event": """
        SELECT count(*) AS n_rows, min(report_date)::text AS date_min, max(report_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities,
               count(*) FILTER (WHERE eps_actual IS NOT NULL AND eps_estimated IS NOT NULL) AS n_with_surprise,
               count(*) FILTER (WHERE last_updated > report_date + 30) AS n_updated_30d_after,
               count(*) FILTER (WHERE report_date > CURRENT_DATE) AS n_future
        FROM earnings_event""",
    "public.earnings_event_timing": """
        SELECT count(*) AS n_rows, count(DISTINCT event_id) AS n_events,
               count(*) FILTER (WHERE conflict) AS n_conflict
        FROM earnings_event_timing""",
    "public.earnings_calendar_event": """
        SELECT count(*) AS n_rows, min(report_date)::text AS date_min, max(report_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities
        FROM earnings_calendar_event""",
    "public.analyst_action": """
        SELECT count(*) AS n_rows, min(action_date)::text AS date_min, max(action_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM analyst_action""",
    "public.analyst_consensus_daily": """
        SELECT count(*) AS n_rows, min(as_of_date)::text AS date_min, max(as_of_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM analyst_consensus_daily""",
    "public.analyst_price_target": """
        SELECT count(*) AS n_rows, min(date_available)::date::text AS date_min,
               max(date_available)::date::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM analyst_price_target""",
    "public.ownership_snapshot": """
        SELECT count(*) AS n_rows, min(period_end)::text AS date_min, max(period_end)::text AS date_max,
               count(DISTINCT security_id) AS n_securities,
               count(*) FILTER (WHERE date_available::date = period_end) AS n_avail_eq_period_end
        FROM ownership_snapshot""",
    "altdata.institutional_holder_position": """
        SELECT count(*) AS n_rows, min(period_end)::text AS date_min, max(period_end)::text AS date_max,
               count(DISTINCT security_id) AS n_securities
        FROM altdata.institutional_holder_position""",
    "public.insider_transaction": """
        SELECT count(*) AS n_rows, min(filing_date)::text AS date_min, max(filing_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM insider_transaction""",
    "public.index_membership": """
        SELECT count(*) AS n_rows, min(start_date)::text AS date_min, max(start_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM index_membership""",
    "public.owner_earnings": """
        SELECT count(*) AS n_rows, min(date_available)::date::text AS date_min,
               max(date_available)::date::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM owner_earnings""",
    "public.revenue_segment": """
        SELECT count(*) AS n_rows, min(date_available)::date::text AS date_min,
               max(date_available)::date::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM revenue_segment""",
    "public.employee_count_history": """
        SELECT count(*) AS n_rows, min(date_available)::date::text AS date_min,
               max(date_available)::date::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM employee_count_history""",
    "public.transcript_date": """
        SELECT count(*) AS n_rows, min(event_date)::text AS date_min, max(event_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM transcript_date""",
    "public.monthly_panel": """
        SELECT count(*) AS n_rows, min(as_of_date)::text AS date_min, max(as_of_date)::text AS date_max,
               count(DISTINCT security_id) AS n_securities FROM monthly_panel""",
    "public.trading_day": """
        SELECT count(*) AS n_rows, min(date)::text AS date_min, max(date)::text AS date_max,
               count(*) FILTER (WHERE is_month_end) AS n_month_ends FROM trading_day""",
    "public.treasury_rate": """
        SELECT count(*) AS n_rows, min(date)::text AS date_min, max(date)::text AS date_max,
               count(*) FILTER (WHERE month3 IS NULL) AS n_month3_null FROM treasury_rate""",
    "market.index_price_daily": """
        SELECT count(*) AS n_rows, min(date)::text AS date_min, max(date)::text AS date_max,
               count(DISTINCT symbol) AS n_symbols,
               min(date) FILTER (WHERE symbol = '^GSPC')::text AS gspc_min
        FROM market.index_price_daily""",
}

LAG_SQL = """
    WITH f AS (
        SELECT CASE WHEN fiscal_period LIKE 'Q%' THEN 'quarter' ELSE 'annual' END AS period_kind,
               coalesce(date_source, 'null') AS date_source,
               (floor(extract(year FROM fiscal_period_end) / 5) * 5)::int AS fpe_5y,
               ((date_available AT TIME ZONE 'America/New_York')::date - fiscal_period_end) AS lag_days
        FROM fundamental_quarterly
        WHERE NOT accepted_date_placeholder
    )
    SELECT period_kind, date_source, fpe_5y, count(*) AS n,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY lag_days) AS median_lag,
           percentile_cont(0.1) WITHIN GROUP (ORDER BY lag_days) AS p10_lag,
           percentile_cont(0.9) WITHIN GROUP (ORDER BY lag_days) AS p90_lag,
           count(*) FILTER (WHERE lag_days < 1) AS n_lag_lt_1,
           count(*) FILTER (WHERE lag_days > 365) AS n_lag_gt_365
    FROM f
    GROUP BY GROUPING SETS ((period_kind), (period_kind, date_source), (period_kind, fpe_5y))
"""

COVERAGE_ITEMS = (
    "price",
    "market_cap",
    "book_equity",
    "net_income",
    "revenue",
    "gross_profit",
    "total_assets",
    "total_debt",
    "eps_estimate_nonpit",
    "eps_consensus_at_last_report",
    "earnings_surprise",
    "vendor_market_cap",
)


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_parallel(jobs: dict[str, Callable[[], Any]], max_workers: int = 8) -> tuple[dict, dict]:
    """Run independent read jobs in threads. Returns (results, wall seconds per job)."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {k: ex.submit(_timed, fn) for k, fn in jobs.items()}
        res = {k: f.result() for k, f in futs.items()}
    return {k: v[0] for k, v in res.items()}, {k: round(v[1], 2) for k, v in res.items()}


def table_stats(dev: bool = False) -> tuple[pd.DataFrame, dict]:
    jobs = {t: (lambda q=q: read_sql(q, dev=dev)) for t, q in TABLE_STATS_SQL.items()}
    res, walls = run_parallel(jobs)
    rows = []
    for t, df in res.items():
        rec = df.iloc[0].to_dict()
        rec["table"] = t
        rows.append(rec)
    return pd.DataFrame(rows).set_index("table"), walls


def lag_stats(dev: bool = False) -> pd.DataFrame:
    return read_sql(LAG_SQL, dev=dev)


def _has_forward_estimate(keys: pd.DataFrame, est: pd.DataFrame, horizon_days: int) -> pd.Series:
    """True if any estimate targets a period in (date, date + horizon]. Not PIT."""
    left = keys[["security_id", "date"]].reset_index().sort_values("date")
    right = (
        est.dropna(subset=["eps_avg"])[["security_id", "target_period"]]
        .drop_duplicates()
        .sort_values("target_period")
    )
    m = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="target_period",
        by="security_id",
        direction="forward",
        allow_exact_matches=False,
        tolerance=pd.Timedelta(days=horizon_days),
    )
    return m.set_index("index")["target_period"].notna().reindex(keys.index)


def monthly_audit(
    start: str = "1990-01-01",
    end: str | None = None,
    min_price: float = 5.0,
    top_n: int = 1000,
    max_fund_staleness_days: int = 548,
    max_earnings_age_days: int = 120,
    dev: bool = False,
) -> dict[str, Any]:
    """Universe counts and per-item coverage for every month-end from ``start``."""
    me = loaders.month_ends(start, end, dev=dev)
    jobs = {
        "securities": lambda: loaders.securities(dev=dev),
        "prices": lambda: loaders.prices_on_dates(me, dev=dev),
        "fundamentals": lambda: loaders.fundamentals(dev=dev),
        "fundamentals_fy": lambda: loaders.fundamentals(
            ("total_equity", "net_income"), quarterly_only=False, dev=dev
        ),
        "earnings": lambda: loaders.earnings_events(dev=dev),
        "estimates": lambda: loaders.analyst_estimates(dev=dev),
        "vendor_mcap": lambda: loaders.vendor_market_cap_on_dates(me, dev=dev),
        "price_span": lambda: loaders.price_span(dev=dev),
    }
    data, walls = run_parallel(jobs)
    sec, px = data["securities"], data["prices"]

    # Drop trailing month-ends with no prices yet (calendar runs past the data).
    last_px = px["date"].max()
    me = me[me <= last_px]

    t0 = time.perf_counter()
    listed = loaders.listed_on(sec, me)
    u = listed.merge(px, on=["security_id", "date"], how="left")
    u["has_price"] = u["close"].notna() & (u["close"] > 0)

    # Prices that fall outside a security's listing dates (identity/date anomalies).
    off_listing = px[px["date"].isin(me)].merge(
        listed, on=["security_id", "date"], how="left", indicator=True
    )
    off_listing = off_listing[off_listing["_merge"] == "left_only"].merge(
        sec[["security_id", "first_trade_date", "last_trade_date"]], on="security_id"
    )
    price_outside_listing = {
        "n_month_end_rows": len(off_listing),
        "n_securities": int(off_listing["security_id"].nunique()),
        "n_before_first_trade_date": int((off_listing["date"] < off_listing["first_trade_date"]).sum()),
        "n_after_last_trade_date": int((off_listing["date"] > off_listing["last_trade_date"]).sum()),
    }

    priced = u[u["has_price"]].copy()
    f = loaders.asof_fundamentals(
        priced[["security_id", "date"]], data["fundamentals"], max_fund_staleness_days
    )
    priced = priced.merge(
        f.drop_duplicates(["security_id", "date"]), on=["security_id", "date"], how="left"
    )
    shares = priced["weighted_average_shares_diluted"].where(
        priced["weighted_average_shares_diluted"] > 0, priced["weighted_average_shares"]
    )
    priced["market_cap"] = priced["close"] * shares.where(shares > 0)

    ev = loaders.asof_earnings(priced[["security_id", "date"]], data["earnings"], max_earnings_age_days)
    ev = ev.drop_duplicates(["security_id", "date"])
    priced = priced.merge(
        ev[["security_id", "date", "report_date", "eps_actual", "eps_estimated"]],
        on=["security_id", "date"],
        how="left",
    )
    priced = priced.merge(
        data["vendor_mcap"].rename(columns={"market_cap": "vendor_market_cap"}),
        on=["security_id", "date"],
        how="left",
    )
    priced["eps_estimate_nonpit"] = _has_forward_estimate(priced, data["estimates"], 400)

    # FY-only fallback: how much coverage is lost by using quarterly rows only.
    fy = loaders.asof_fundamentals(
        priced[["security_id", "date"]], data["fundamentals_fy"], max_fund_staleness_days
    ).drop_duplicates(["security_id", "date"])
    priced = priced.merge(
        fy[["security_id", "date", "total_equity"]].rename(columns={"total_equity": "be_any_period"}),
        on=["security_id", "date"],
        how="left",
    )

    px_filter = priced["close_raw"].fillna(priced["close"])
    priced["passes_price"] = px_filter >= min_price
    priced["close_raw_missing"] = priced["close_raw"].isna()
    ranked = priced[priced["passes_price"] & priced["market_cap"].notna()].copy()
    ranked["mcap_rank"] = ranked.groupby("date")["market_cap"].rank(ascending=False, method="first")
    priced = priced.merge(
        ranked[["security_id", "date", "mcap_rank"]], on=["security_id", "date"], how="left"
    )
    priced["top_n"] = priced["mcap_rank"] <= top_n

    delisted_ids = set(sec.loc[sec["last_trade_date"].notna(), "security_id"])
    priced["later_delisted"] = priced["security_id"].isin(delisted_ids)
    ltd = priced["security_id"].map(sec.set_index("security_id")["last_trade_date"])
    priced["delists_within_12m"] = (ltd > priced["date"]) & (
        ltd <= priced["date"] + pd.Timedelta(days=365)
    )
    delist_rate = pd.DataFrame(
        {
            "all_priced": priced.groupby("date")["delists_within_12m"].mean(),
            "price_ge_min": priced[priced["passes_price"]].groupby("date")["delists_within_12m"].mean(),
            "top_n": priced[priced["top_n"]].groupby("date")["delists_within_12m"].mean(),
        }
    )
    fund = data["fundamentals"]
    zero_lag = {
        "n_quarterly_rows": len(fund),
        "n_reimputed": int(fund["zero_lag_reimputed"].sum()),
        "reimputed_by_fpe_5y": fund.loc[fund["zero_lag_reimputed"], "fiscal_period_end"]
        .dt.year.floordiv(5)
        .mul(5)
        .value_counts()
        .sort_index()
        .to_dict(),
    }

    counts = pd.DataFrame(
        {
            "n_listed": u.groupby("date").size(),
            "n_priced": priced.groupby("date").size(),
            "n_price_ge_min": priced[priced["passes_price"]].groupby("date").size(),
            "n_price_ge_min_with_mcap": priced[priced["passes_price"] & priced["market_cap"].notna()]
            .groupby("date")
            .size(),
            "n_top_n": priced[priced["top_n"]].groupby("date").size(),
            "n_priced_later_delisted": priced[priced["later_delisted"]].groupby("date").size(),
            "n_close_raw_missing": priced[priced["close_raw_missing"]].groupby("date").size(),
        }
    ).fillna(0).astype(int)
    counts["pct_priced_later_delisted"] = counts["n_priced_later_delisted"] / counts["n_priced"]

    def cov(df: pd.DataFrame) -> pd.DataFrame:
        items = {
            "market_cap": df["market_cap"].notna(),
            "book_equity": df["total_equity"].notna(),
            "net_income": df["net_income"].notna(),
            "revenue": df["revenue"].notna(),
            "gross_profit": df["gross_profit"].notna(),
            "total_assets": df["total_assets"].notna(),
            "total_debt": df["total_debt"].notna(),
            "eps_estimate_nonpit": df["eps_estimate_nonpit"].astype(bool),
            "eps_consensus_at_last_report": df["eps_estimated"].notna(),
            "earnings_surprise": df["eps_estimated"].notna() & df["eps_actual"].notna(),
            "vendor_market_cap": df["vendor_market_cap"].notna(),
            "book_equity_any_period": df["be_any_period"].notna(),
        }
        return pd.DataFrame(items).groupby(df["date"]).mean()

    coverage_all = cov(priced)
    coverage_all.insert(0, "price", counts["n_priced"] / counts["n_listed"])
    coverage_5 = cov(priced[priced["passes_price"]])
    coverage_5.insert(0, "price", 1.0)
    coverage_top = cov(priced[priced["top_n"]])
    coverage_top.insert(0, "price", 1.0)

    # Vendor vs computed market cap on matched rows (share-basis sanity check).
    both = priced.dropna(subset=["market_cap", "vendor_market_cap"])
    both = both[both["vendor_market_cap"] > 0]
    mcap_ratio = (
        (both["market_cap"] / both["vendor_market_cap"])
        .groupby(both["date"].dt.year)
        .agg(["median", lambda s: ((s - 1).abs() < 0.2).mean()])
        .set_axis(["median_ratio", "share_within_20pct"], axis=1)
    )

    compute_s = time.perf_counter() - t0
    return {
        "month_ends": me,
        "counts": counts,
        "coverage_all": coverage_all,
        "coverage_price5": coverage_5,
        "coverage_top_n": coverage_top,
        "mcap_ratio_by_year": mcap_ratio,
        "delist_rate_12m": delist_rate,
        "zero_lag": zero_lag,
        "price_outside_listing": price_outside_listing,
        "priced": priced,
        "securities": sec,
        "price_span": data["price_span"],
        "earnings": data["earnings"],
        "walls": walls,
        "compute_s": round(compute_s, 2),
    }


def earliest_good_month(
    coverage: pd.DataFrame, items: tuple[str, ...], threshold: float = 0.8
) -> dict[str, Any]:
    """First month all ``items`` exceed ``threshold``, and first month after which they always do."""
    ok = (coverage[list(items)] > threshold).all(axis=1)
    first = ok.idxmax() if ok.any() else None
    sustained = None
    if ok.iloc[-1]:
        bad = ok[~ok]
        sustained = ok.index[0] if bad.empty else ok.index[ok.index > bad.index[-1]][0]
    return {"first": first, "sustained": sustained}


def delisting_audit(sec: pd.DataFrame, span: pd.DataFrame, me: pd.Series) -> dict[str, Any]:
    """Can a final price / delisting-month return be derived for delisted names?"""
    d = sec[sec["last_trade_date"].notna()].merge(span, on="security_id", how="left")
    gap = (d["last_trade_date"] - d["last_price_date"]).dt.days
    me = pd.Series(pd.to_datetime(me)).sort_values()
    prev_me = pd.merge_asof(
        d[["security_id", "last_trade_date"]].sort_values("last_trade_date"),
        pd.DataFrame({"prev_me": me}),
        left_on="last_trade_date",
        right_on="prev_me",
        direction="backward",
        allow_exact_matches=False,
    )
    # Crawl-floor signature: history starts on the first session of the first
    # 2005-09-17..2026-09-17 window although the name listed earlier.
    allsec = sec.merge(span, on="security_id", how="left")
    trunc = allsec["first_price_date"].between("2005-09-19", "2005-09-23") & (
        allsec["first_trade_date"] < pd.Timestamp("2005-09-01")
    )
    return {
        "n_price_history_truncated_2005": int(trunc.sum()),
        "n_price_history_truncated_2005_delisted": int((trunc & allsec["last_trade_date"].notna()).sum()),
        "n_delisted": len(d),
        "n_with_any_price": int(d["last_price_date"].notna().sum()),
        "n_last_price_eq_last_trade": int((gap == 0).sum()),
        "n_last_price_within_5d": int(gap.between(0, 5).sum()),
        "n_last_price_after_last_trade": int((gap < 0).sum()),
        "gap_days_quantiles": gap.quantile([0.5, 0.9, 0.99]).to_dict(),
        "n_last_close_adj_present": int(d["last_close_adj"].notna().sum()),
        "n_last_close_raw_lt_1": int((d["last_close_raw"] < 1).sum()),
        "delist_year_counts": d["last_trade_date"].dt.year.value_counts().sort_index().to_dict(),
        "n_prev_month_end_found": int(prev_me["prev_me"].notna().sum()),
    }


def sivb_check(priced: pd.DataFrame, sec: pd.DataFrame) -> dict[str, Any]:
    row = sec[(sec["last_symbol"] == "SIVB") | (sec["first_symbol"] == "SIVB")]
    if row.empty:
        return {"found": False}
    sid = int(row["security_id"].iloc[0])
    s = priced[priced["security_id"] == sid]
    return {
        "found": True,
        "security_id": sid,
        "first_trade_date": str(row["first_trade_date"].iloc[0].date()),
        "last_trade_date": str(row["last_trade_date"].iloc[0].date()),
        "first_month_in_universe": str(s["date"].min().date()) if len(s) else None,
        "last_month_in_universe": str(s["date"].max().date()) if len(s) else None,
        "n_months_in_universe": len(s),
        "n_months_passing_price": int(s["passes_price"].sum()),
        "n_months_top_n": int(s["top_n"].sum()),
        "in_universe_2023_02": bool((s["date"] == pd.Timestamp("2023-02-28")).any()),
        "in_universe_2023_03": bool((s["date"] == pd.Timestamp("2023-03-31")).any()),
    }


def yearly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Collapse a month-indexed frame to calendar years (December or last month)."""
    g = df.groupby(pd.to_datetime(df.index).year)
    return g.last() if how == "last" else g.mean()


def to_markdown(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Minimal markdown table (avoids a tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)
