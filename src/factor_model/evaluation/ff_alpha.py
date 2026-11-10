"""Fama–French 5 + Momentum alpha with Newey–West standard errors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


FF6_COLS = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"]


def ff6_alpha(
    port_excess: pd.Series,
    ff: pd.DataFrame,
    *,
    lags: int = 6,
) -> dict[str, float]:
    """Regress portfolio excess (over RF) on FF5+UMD; Newey–West HAC SE.

    ``port_excess`` and ``ff`` are indexed by return month. Returns annualized
    alpha (×12), its t-stat, and factor betas.
    """
    df = pd.DataFrame({"y": port_excess}).join(ff[FF6_COLS], how="inner").dropna()
    if len(df) < 24:
        return {"alpha_ann": float("nan"), "alpha_t": float("nan"), "n": len(df)}
    y = df["y"].to_numpy()
    X = sm.add_constant(df[FF6_COLS].to_numpy())
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {
        "alpha_ann": float(res.params[0] * 12),
        "alpha_t": float(res.tvalues[0]),
        "n": int(res.nobs),
        "r2": float(res.rsquared),
    }
    for i, col in enumerate(FF6_COLS, start=1):
        out[f"beta_{col}"] = float(res.params[i])
        out[f"t_{col}"] = float(res.tvalues[i])
    return out
