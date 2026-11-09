"""Portfolio performance metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ann_return(rets: pd.Series, periods: int = 12) -> float:
    r = rets.dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() ** (periods / len(r)) - 1.0)


def ann_vol(rets: pd.Series, periods: int = 12) -> float:
    r = rets.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods))


def sharpe(rets: pd.Series, rf: pd.Series | float = 0.0, periods: int = 12) -> float:
    r = rets.dropna()
    if isinstance(rf, pd.Series):
        excess = (r - rf.reindex(r.index)).dropna()
    else:
        excess = r - float(rf)
    if excess.std(ddof=1) == 0 or excess.empty:
        return float("nan")
    return float(np.sqrt(periods) * excess.mean() / excess.std(ddof=1))


def max_drawdown(rets: pd.Series) -> float:
    r = rets.dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def information_ratio(active: pd.Series, periods: int = 12) -> float:
    a = active.dropna()
    if len(a) < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(periods) * a.mean() / a.std(ddof=1))


def summarize(
    rets: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    rf: pd.Series | float = 0.0,
) -> dict[str, float]:
    out = {
        "ann_return": ann_return(rets),
        "ann_vol": ann_vol(rets),
        "sharpe": sharpe(rets, rf=rf),
        "max_drawdown": max_drawdown(rets),
        "n_months": int(rets.dropna().shape[0]),
    }
    if turnover is not None:
        out["turnover"] = float(turnover.dropna().mean())
    if benchmark is not None:
        aligned = pd.DataFrame({"r": rets, "b": benchmark}).dropna()
        active = aligned["r"] - aligned["b"]
        out["excess_vs_benchmark"] = ann_return(active)
        out["tracking_error"] = ann_vol(active)
        out["information_ratio"] = information_ratio(active)
    return out
