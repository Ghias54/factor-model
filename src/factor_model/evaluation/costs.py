"""Transaction-cost helpers for long-only monthly portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_costs(rets: pd.Series, turnover: pd.Series, bps_per_side: float) -> pd.Series:
    """Deduct ``turnover × bps × 2 / 1e4`` from monthly gross returns.

    Turnover is the one-way fraction replaced; buy+sell ⇒ ×2 sides.
    """
    cost = turnover.reindex(rets.index).fillna(0.0) * (bps_per_side / 1e4) * 2.0
    return rets - cost


def breakeven_bps_vs_benchmark(
    model_rets: pd.Series,
    model_turn: pd.Series,
    bench_rets: pd.Series,
    bench_turn: pd.Series | None = None,
) -> float:
    """BPS per side at which model mean excess vs benchmark (gross) is wiped out.

    Solves mean(r_m - c_m) = mean(r_b - c_b) for c = turn * bps * 2 / 1e4.
    If bench_turn is None, benchmark costs are zero.
    """
    df = pd.DataFrame(
        {
            "rm": model_rets,
            "tm": model_turn.reindex(model_rets.index).fillna(0.0),
            "rb": bench_rets.reindex(model_rets.index),
            "tb": (
                bench_turn.reindex(model_rets.index).fillna(0.0)
                if bench_turn is not None
                else 0.0
            ),
        }
    ).dropna()
    # mean(rm) - mean(tm)*2*bps/1e4 = mean(rb) - mean(tb)*2*bps/1e4
    # mean(rm)-mean(rb) = 2*bps/1e4 * (mean(tm)-mean(tb))
    excess = float(df["rm"].mean() - df["rb"].mean())
    dturn = float(df["tm"].mean() - df["tb"].mean())
    if abs(dturn) < 1e-12:
        return float("inf") if excess > 0 else 0.0
    # bps such that excess - 2*bps/1e4*dturn = 0
    bps = excess / (2.0 * dturn / 1e4)
    return float(bps)
