"""Monthly top-N universe by point-in-time market cap."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from factor_model.config import load_config
from factor_model.data import loaders

FEATURE_FUND_COLS = (
    "revenue",
    "gross_profit",
    "net_income",
    "eps_diluted",
    "weighted_average_shares",
    "weighted_average_shares_diluted",
    "total_assets",
    "total_equity",
    "total_debt",
    "cash_from_operations",
    "total_current_assets",
    "total_current_liabilities",
)


def build_universe(
    start: str | None = None,
    end: str | None = None,
    cfg: dict | None = None,
    dev: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Top-N names each month-end by PIT mcap with ``close_raw`` ≥ min_price.

    Membership = non-suspect month-end price. Market cap = close × diluted shares
    (zero-lag-guarded as-of). Vendor ``market_cap_daily`` fills ranking gaps only.
    """
    cfg = cfg or load_config()
    start = start or cfg["sample"]["start"]
    end = end or cfg["sample"]["end"]
    top_n = int(cfg["universe"]["top_n_by_market_cap"])
    min_price = float(cfg["universe"]["min_price"])
    stale = int(cfg["features"]["staleness_days_fundamentals"])
    t0 = time.perf_counter()

    # Need prior month-ends for momentum lookbacks when computing features later;
    # universe itself only needs [start, end].
    me = loaders.month_ends(start, end, complete_only=True, dev=dev)
    sec = loaders.securities(dev=dev)
    px = loaders.prices_on_dates(me, dev=dev)
    fund = loaders.fundamentals(columns=FEATURE_FUND_COLS, quarterly_only=True, guard=True, dev=dev)
    vendor = loaders.vendor_market_cap_on_dates(me, dev=dev)
    hist = loaders.symbol_history(dev=dev)

    # Eligible: non-suspect price present with close_raw >= min_price
    elig = px[
        px["close"].notna()
        & (px["close"] > 0)
        & px["close_raw"].notna()
        & (px["close_raw"] >= min_price)
        & (px["price_quality"].isna() | (px["price_quality"] != "suspect"))
    ][["security_id", "date", "close", "close_raw", "close_adj", "volume"]].copy()

    shares = loaders.asof_fundamentals(elig[["security_id", "date"]], fund, stale)
    shares = shares[
        [
            "security_id",
            "date",
            "weighted_average_shares",
            "weighted_average_shares_diluted",
            "date_available",
            "fiscal_period_end",
        ]
    ]
    elig = elig.merge(shares, on=["security_id", "date"], how="left")
    dil = elig["weighted_average_shares_diluted"].where(
        elig["weighted_average_shares_diluted"] > 0, elig["weighted_average_shares"]
    )
    elig["mcap_computed"] = elig["close"] * dil.where(dil > 0)
    elig = elig.merge(
        vendor.rename(columns={"market_cap": "mcap_vendor"}),
        on=["security_id", "date"],
        how="left",
    )
    elig["mcap_from_vendor"] = elig["mcap_computed"].isna() & elig["mcap_vendor"].notna()
    elig["mcap"] = elig["mcap_computed"].fillna(elig["mcap_vendor"])
    elig = elig[elig["mcap"].notna() & (elig["mcap"] > 0)].copy()

    elig["rank"] = elig.groupby("date")["mcap"].rank(ascending=False, method="first").astype(int)
    univ = elig[elig["rank"] <= top_n].copy().reset_index(drop=True)
    univ["symbol"] = loaders.symbol_asof(univ, hist).to_numpy()
    # Attach listing dates for delisting handling downstream
    univ = univ.merge(
        sec[["security_id", "first_trade_date", "last_trade_date"]],
        on="security_id",
        how="left",
    )

    meta = {
        "n_months": int(univ["date"].nunique()),
        "n_rows": len(univ),
        "n_securities": int(univ["security_id"].nunique()),
        "months_below_top_n": univ.groupby("date").size().loc[lambda s: s < top_n].to_dict(),
        "n_vendor_mcap_fallback": int(univ["mcap_from_vendor"].sum()),
        "pct_vendor_mcap_fallback": float(univ["mcap_from_vendor"].mean()),
        "wall_s": round(time.perf_counter() - t0, 2),
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "top_n": top_n,
        "min_price": min_price,
    }
    # Convert Timestamp keys in months_below for JSON
    meta["months_below_top_n"] = {
        str(pd.Timestamp(k).date()): int(v) for k, v in meta["months_below_top_n"].items()
    }
    cols = [
        "date",
        "security_id",
        "symbol",
        "mcap",
        "rank",
        "close",
        "close_raw",
        "close_adj",
        "volume",
        "mcap_computed",
        "mcap_vendor",
        "mcap_from_vendor",
        "weighted_average_shares",
        "weighted_average_shares_diluted",
        "date_available",
        "fiscal_period_end",
        "first_trade_date",
        "last_trade_date",
    ]
    return univ[cols].sort_values(["date", "rank"]).reset_index(drop=True), meta
