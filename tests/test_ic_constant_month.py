"""Constant-prediction months must contribute IC = 0, not be dropped."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model.factors.classic import rank_ic
from factor_model.walkforward.engine import monthly_rank_ic, monthly_rank_ic_series


def test_monthly_rank_ic_constant_pred_is_zero_not_dropped():
    dates = pd.to_datetime(["2011-01-31"] * 30 + ["2011-02-28"] * 30)
    # Month 1: constant predictions → IC 0; month 2: perfect rank match
    pred = pd.Series([1.0] * 30 + list(range(30)), dtype=float)
    ret = pd.Series(list(range(30)) + list(range(30)), dtype=float)
    series = monthly_rank_ic_series(pred, ret, pd.Series(dates))
    assert len(series) == 2
    assert series.iloc[0] == 0.0
    assert series.iloc[1] == 1.0
    assert series.attrs["n_constant_pred"] == 1
    assert abs(monthly_rank_ic(pred, ret, pd.Series(dates)) - 0.5) < 1e-9


def test_rank_ic_constant_score_is_zero():
    n = 40
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2011-01-31"] * n),
            "security_id": np.arange(n),
            "ret_1m": np.linspace(-0.1, 0.1, n),
        }
    )
    score = pd.Series(np.zeros(n))
    ic = rank_ic(panel, score)
    assert len(ic) == 1
    assert float(ic.iloc[0]) == 0.0
    assert ic.attrs["n_constant_pred"] == 1
