"""Point-in-time computation of the 26 locked features and the return target."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from factor_model.config import load_config
from factor_model.data import loaders
from factor_model.universe.build import FEATURE_FUND_COLS

FEATURE_NAMES = [
    "value_bm",
    "size_ln_mcap",
    "momentum_12_1",
    "quality_roe",
    "earnings_yield",
    "sales_to_price",
    "cashflow_to_price",
    "reversal_1m",
    "momentum_6m",
    "reversal_36_13",
    "volatility_12m",
    "beta_60m",
    "max_daily_ret_1m",
    "turnover",
    "dollar_volume_ln",
    "amihud",
    "gross_margin",
    "roa",
    "gross_profit_to_assets",
    "delta_roe_4q",
    "revenue_growth_yoy",
    "asset_growth_yoy",
    "accruals",
    "debt_to_assets",
    "current_ratio",
    "sue",
]


def _month_end_lags(me: pd.Series) -> pd.DataFrame:
    me = pd.to_datetime(me).sort_values().reset_index(drop=True)
    df = pd.DataFrame({"date": me})
    for lag in (1, 6, 12, 13, 36):
        df[f"me_m{lag}"] = me.shift(lag)
    df["me_next"] = me.shift(-1)
    return df


def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


def _attach_month_px(univ: pd.DataFrame, px: pd.DataFrame, me_map: pd.DataFrame) -> pd.DataFrame:
    out = univ.merge(me_map, on="date", how="left")
    base = px[["security_id", "date", "close_adj"]].rename(
        columns={"date": "_d", "close_adj": "_px"}
    )

    def join_at(frame: pd.DataFrame, col_date: str) -> pd.Series:
        m = frame[["security_id", col_date]].merge(
            base, left_on=["security_id", col_date], right_on=["security_id", "_d"], how="left"
        )
        return m["_px"]

    out["px_t"] = out["close_adj"]
    out["px_m1"] = join_at(out, "me_m1")
    out["px_m6"] = join_at(out, "me_m6")
    out["px_m12"] = join_at(out, "me_m12")
    out["px_m13"] = join_at(out, "me_m13")
    out["px_m36"] = join_at(out, "me_m36")
    out["px_next"] = join_at(out, "me_next")
    return out


def _delisting_final_px(univ: pd.DataFrame, span: pd.DataFrame) -> pd.Series:
    u = univ[["security_id", "me_next", "last_trade_date"]].copy()
    u = u.merge(
        span[["security_id", "last_price_date", "last_close_adj"]], on="security_id", how="left"
    )
    u["effective_delist"] = u[["last_trade_date", "last_price_date"]].max(axis=1)
    needs = u["me_next"].notna() & u["effective_delist"].notna() & (
        u["effective_delist"] < u["me_next"]
    )
    final = pd.Series(np.nan, index=u.index, dtype="float64")
    final.loc[needs] = u.loc[needs, "last_close_adj"].to_numpy()
    return final


def _daily_features_for_sid(sid: int, months: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date")
    g = g[g["close_adj"].notna() & (g["close_adj"] > 0)].copy()
    if g.empty or months.empty:
        return pd.DataFrame()
    g["ret"] = g["close_adj"].pct_change()
    dollar = g["close"] * g["volume"]
    g["amihud_d"] = g["ret"].abs() / dollar.replace(0, np.nan)
    dates = g["date"].to_numpy()
    rets = g["ret"].to_numpy()
    amihuds = g["amihud_d"].to_numpy()
    rows = []
    for date, start_12m, start_1m in zip(
        months["date"].to_numpy(),
        months["_start_12m"].to_numpy(),
        months["_start_1m"].to_numpy(),
        strict=True,
    ):
        hi = dates.searchsorted(np.datetime64(date), side="right")
        lo12 = dates.searchsorted(np.datetime64(start_12m), side="right")
        lo1 = dates.searchsorted(np.datetime64(start_1m), side="right")
        r12 = rets[lo12:hi]
        r12 = r12[~np.isnan(r12)]
        r1 = rets[lo1:hi]
        r1 = r1[~np.isnan(r1)]
        a1 = amihuds[lo1:hi]
        a1 = a1[~np.isnan(a1)]
        rows.append(
            {
                "security_id": sid,
                "date": date,
                "volatility_12m": float(np.std(r12, ddof=1)) if len(r12) >= 60 else np.nan,
                "max_daily_ret_1m": float(np.max(r1)) if len(r1) >= 5 else np.nan,
                "amihud": float(np.mean(a1)) if len(a1) >= 5 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _betas(
    monthly_ret: pd.DataFrame, mkt_ret: pd.Series, min_obs: int = 24, window: int = 60
) -> pd.DataFrame:
    wide = monthly_ret.pivot(index="date", columns="security_id", values="ret").sort_index()
    mkt = mkt_ret.reindex(wide.index).astype(float)
    betas = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for i in range(len(wide)):
        lo = max(0, i - window + 1)
        ywin = wide.iloc[lo : i + 1]
        x = mkt.iloc[lo : i + 1].to_numpy()
        valid_x = ~np.isnan(x)
        if valid_x.sum() < min_obs:
            continue
        x = x[valid_x]
        x_dm = x - x.mean()
        var_x = float(x_dm @ x_dm)
        if var_x <= 0:
            continue
        ymat = ywin.to_numpy()[valid_x]
        n_obs = np.sum(~np.isnan(ymat), axis=0)
        y_dm = ymat - np.nanmean(ymat, axis=0)
        cov = np.nansum(y_dm * x_dm[:, None], axis=0)
        b = cov / var_x
        b = np.where(n_obs >= min_obs, b, np.nan)
        betas.iloc[i] = b
    long = betas.stack(future_stack=True).rename("beta_60m")
    long.index = long.index.set_names(["date", "security_id"])
    return long.reset_index()


def compute_targets_and_features(
    univ: pd.DataFrame,
    cfg: dict | None = None,
    n_jobs: int = -1,
    dev: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute 26 raw features + targets for the universe panel."""
    cfg = cfg or load_config()
    t0 = time.perf_counter()
    walls: dict[str, float] = {}
    end = cfg["sample"]["end"]
    stale = int(cfg["features"]["staleness_days_fundamentals"])
    earn_age = int(cfg["features"]["earnings_max_age_days"])

    me_ext = loaders.month_ends("2002-01-01", "2026-09-30", complete_only=True, dev=dev)
    me_map = _month_end_lags(me_ext)
    me_map = me_map[me_map["date"].isin(pd.to_datetime(univ["date"].unique()))]

    t1 = time.perf_counter()
    px_me = loaders.prices_on_dates(me_ext, dev=dev)
    px_me = px_me[px_me["price_quality"].isna() | (px_me["price_quality"] != "suspect")]
    walls["prices_month_end_s"] = round(time.perf_counter() - t1, 2)

    t1 = time.perf_counter()
    fund = loaders.fundamentals(columns=FEATURE_FUND_COLS, quarterly_only=True, guard=True, dev=dev)
    walls["fundamentals_s"] = round(time.perf_counter() - t1, 2)

    t1 = time.perf_counter()
    events = loaders.earnings_events(dev=dev)
    events = events[
        (events["report_date"] > "1980-01-01")
        & (events["report_date"] <= pd.Timestamp(end) + pd.Timedelta(days=40))
        & events["eps_actual"].notna()
        & events["eps_estimated"].notna()
    ].copy()
    walls["earnings_s"] = round(time.perf_counter() - t1, 2)

    t1 = time.perf_counter()
    tre = loaders.treasury_rates("2001-01-01", "2026-10-15", dev=dev)
    idx = loaders.index_prices("^GSPC", "2001-01-01", "2026-10-15", dev=dev)
    idx = idx[idx["price_quality"].isna() | (idx["price_quality"] != "suspect")]
    span = loaders.price_span(dev=dev)
    walls["macro_index_s"] = round(time.perf_counter() - t1, 2)

    keys = univ[
        [
            "date",
            "security_id",
            "symbol",
            "mcap",
            "rank",
            "close",
            "close_raw",
            "close_adj",
            "volume",
            "weighted_average_shares",
            "weighted_average_shares_diluted",
            "last_trade_date",
        ]
    ].copy()

    t1 = time.perf_counter()
    keys = _attach_month_px(keys, px_me, me_map)
    keys["final_px"] = _delisting_final_px(keys, span)
    next_px = keys["px_next"].where(keys["px_next"].notna(), keys["final_px"])
    keys["ret_1m"] = _safe_div(next_px, keys["px_t"]) - 1.0
    tre_me = pd.merge_asof(
        keys[["date"]].drop_duplicates().sort_values("date"),
        tre[["date", "month3"]].sort_values("date"),
        on="date",
        direction="backward",
    ).rename(columns={"month3": "rf_month3_pct"})
    keys = keys.merge(tre_me, on="date", how="left")
    keys["rf_1m"] = keys["rf_month3_pct"] / 100.0 / 12.0
    keys["excess_ret_1m"] = keys["ret_1m"] - keys["rf_1m"]
    keys["target_rank"] = keys.groupby("date")["ret_1m"].rank(pct=True, method="average")
    keys["reversal_1m"] = _safe_div(keys["px_t"], keys["px_m1"]) - 1.0
    keys["momentum_6m"] = _safe_div(keys["px_t"], keys["px_m6"]) - 1.0
    keys["momentum_12_1"] = _safe_div(keys["px_m1"], keys["px_m12"]) - 1.0
    keys["reversal_36_13"] = _safe_div(keys["px_m13"], keys["px_m36"]) - 1.0
    walls["price_features_s"] = round(time.perf_counter() - t1, 2)

    t1 = time.perf_counter()
    ttm = loaders.asof_last_n_quarters(
        keys[["security_id", "date"]], fund, n=5, max_staleness_days=stale
    )
    keys = keys.merge(ttm, on=["security_id", "date"], how="left")
    walls["ttm_asof_s"] = round(time.perf_counter() - t1, 2)

    keys["ttm4_ok"] = (
        keys["q0_net_income"].notna()
        & keys["q1_net_income"].notna()
        & keys["q2_net_income"].notna()
        & keys["q3_net_income"].notna()
    )
    ttm_ni = keys[["q0_net_income", "q1_net_income", "q2_net_income", "q3_net_income"]].sum(
        axis=1, min_count=4
    )
    ttm_rev = keys[["q0_revenue", "q1_revenue", "q2_revenue", "q3_revenue"]].sum(axis=1, min_count=4)
    ttm_gp = keys[["q0_gross_profit", "q1_gross_profit", "q2_gross_profit", "q3_gross_profit"]].sum(
        axis=1, min_count=4
    )
    ttm_cfo = keys[
        [
            "q0_cash_from_operations",
            "q1_cash_from_operations",
            "q2_cash_from_operations",
            "q3_cash_from_operations",
        ]
    ].sum(axis=1, min_count=4)
    ttm_eps = keys[["q0_eps_diluted", "q1_eps_diluted", "q2_eps_diluted", "q3_eps_diluted"]].sum(
        axis=1, min_count=4
    )

    equity = keys["q0_total_equity"]
    assets = keys["q0_total_assets"]
    keys["value_bm"] = _safe_div(equity, keys["mcap"])
    keys["size_ln_mcap"] = np.log(keys["mcap"].where(keys["mcap"] > 0))
    keys["quality_roe"] = _safe_div(ttm_ni, equity.where(equity > 0))
    keys["earnings_yield"] = _safe_div(ttm_eps, keys["close"])
    keys["sales_to_price"] = _safe_div(ttm_rev, keys["mcap"])
    keys["cashflow_to_price"] = _safe_div(ttm_cfo, keys["mcap"])
    keys["gross_margin"] = _safe_div(keys["q0_gross_profit"], keys["q0_revenue"])
    keys["roa"] = _safe_div(ttm_ni, assets)
    keys["gross_profit_to_assets"] = _safe_div(ttm_gp, assets)
    keys["accruals"] = _safe_div(ttm_ni - ttm_cfo, assets)
    keys["debt_to_assets"] = _safe_div(keys["q0_total_debt"], assets)
    keys["current_ratio"] = _safe_div(
        keys["q0_total_current_assets"], keys["q0_total_current_liabilities"]
    )
    keys["revenue_growth_yoy"] = _safe_div(keys["q0_revenue"], keys["q4_revenue"]) - 1.0
    keys["asset_growth_yoy"] = _safe_div(keys["q0_total_assets"], keys["q4_total_assets"]) - 1.0
    prior_ttm_ni = keys[
        ["q1_net_income", "q2_net_income", "q3_net_income", "q4_net_income"]
    ].sum(axis=1, min_count=4)
    prior_roe = _safe_div(prior_ttm_ni, keys["q4_total_equity"].where(keys["q4_total_equity"] > 0))
    keys["delta_roe_4q"] = keys["quality_roe"] - prior_roe

    shares = keys["weighted_average_shares_diluted"].where(
        keys["weighted_average_shares_diluted"] > 0, keys["weighted_average_shares"]
    )
    shares = shares.fillna(keys.get("q0_weighted_average_shares_diluted")).fillna(
        keys.get("q0_weighted_average_shares")
    )
    keys["turnover"] = _safe_div(keys["volume"], shares)
    keys["dollar_volume_ln"] = np.log((keys["close"] * keys["volume"]).where(lambda s: s > 0))

    t1 = time.perf_counter()
    sue = loaders.asof_earnings(keys[["security_id", "date"]], events, earn_age)
    keys = keys.merge(
        sue[["security_id", "date", "report_date", "eps_actual", "eps_estimated"]],
        on=["security_id", "date"],
        how="left",
    )
    keys["sue"] = _safe_div(keys["eps_actual"] - keys["eps_estimated"], keys["close"])
    keys["fund_date_available"] = keys["q0_date_available"]
    keys["fund_fiscal_period_end"] = keys["q0_fiscal_period_end"]
    walls["sue_s"] = round(time.perf_counter() - t1, 2)

    t1 = time.perf_counter()
    sids = [int(x) for x in keys["security_id"].unique()]
    daily = loaders.prices_daily(
        "2005-01-01",
        "2026-10-15",
        security_ids=sids,
        columns=("close", "close_adj", "volume", "price_quality"),
        dev=dev,
    )
    daily = daily[daily["price_quality"].isna() | (daily["price_quality"] != "suspect")]
    walls["daily_prices_load_s"] = round(time.perf_counter() - t1, 2)

    t1 = time.perf_counter()
    month_meta = keys[["security_id", "date"]].drop_duplicates().copy()
    month_meta["_start_12m"] = month_meta["date"] - pd.DateOffset(months=12)
    month_meta["_start_1m"] = month_meta["date"] - pd.DateOffset(months=1)
    daily_by = {sid: g for sid, g in daily.groupby("security_id", sort=False)}
    meta_by = {sid: g for sid, g in month_meta.groupby("security_id", sort=False)}

    def _one(sid: int) -> pd.DataFrame:
        return _daily_features_for_sid(sid, meta_by.get(sid, pd.DataFrame()), daily_by.get(sid, pd.DataFrame()))

    parts = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(sid) for sid in sids)
    daily_feat = (
        pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
        if parts
        else pd.DataFrame()
    )
    if len(daily_feat):
        keys = keys.merge(daily_feat, on=["security_id", "date"], how="left")
    else:
        keys["volatility_12m"] = np.nan
        keys["max_daily_ret_1m"] = np.nan
        keys["amihud"] = np.nan
    walls["daily_features_s"] = round(time.perf_counter() - t1, 2)

    t1 = time.perf_counter()
    px_beta = px_me[px_me["security_id"].isin(sids)][["security_id", "date", "close_adj"]].copy()
    px_beta = px_beta.sort_values(["security_id", "date"])
    px_beta["ret"] = px_beta.groupby("security_id")["close_adj"].pct_change()
    idx_me = pd.merge_asof(
        pd.DataFrame({"date": pd.to_datetime(me_ext)}),
        idx[["date", "close_adj"]].sort_values("date"),
        on="date",
        direction="backward",
    ).dropna(subset=["close_adj"])
    idx_me["mkt_ret"] = idx_me["close_adj"].pct_change()
    mkt_ret = idx_me.set_index("date")["mkt_ret"]
    beta_df = _betas(px_beta.dropna(subset=["ret"]), mkt_ret)
    keys = keys.merge(beta_df, on=["security_id", "date"], how="left")
    walls["beta_s"] = round(time.perf_counter() - t1, 2)

    raw = keys[
        [
            "date",
            "security_id",
            "symbol",
            "mcap",
            "rank",
            *FEATURE_NAMES,
            "ret_1m",
            "excess_ret_1m",
            "target_rank",
            "rf_1m",
            "fund_date_available",
            "fund_fiscal_period_end",
            "report_date",
            "ttm4_ok",
            "close",
            "close_adj",
        ]
    ].copy()

    meta = {
        "n_rows": len(raw),
        "n_months": int(raw["date"].nunique()),
        "feature_names": FEATURE_NAMES,
        "walls": walls,
        "wall_s": round(time.perf_counter() - t0, 2),
        "n_jobs": n_jobs,
    }
    return raw, meta


def preprocess_panel(raw: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Cross-sectional rank each feature to [-1, 1]; missing → 0."""
    feature_names = feature_names or FEATURE_NAMES
    out = raw[
        ["date", "security_id", "symbol", "mcap", "rank", "ret_1m", "excess_ret_1m", "target_rank"]
    ].copy()
    for col in feature_names:
        r = raw.groupby("date")[col].rank(pct=True, method="average")
        # pct rank in (0,1]; map to [-1,1]: 2*r - 1; NaN stays NaN then fill 0
        scaled = 2.0 * r - 1.0
        out[col] = scaled.fillna(0.0)
    return out
