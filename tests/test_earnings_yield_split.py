"""earnings_yield = TTM net income / mcap is invariant to a pure share split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model.features.compute import _safe_div


def test_earnings_yield_unchanged_by_2_for_1_split():
    """Same net income, price halves, shares double, mcap unchanged → EY unchanged.

    Old per-share formula (TTM EPS / close) would double if EPS were left on the
    pre-split basis while close moved to the post-split basis.
    """
    ttm_ni = pd.Series([100.0, -50.0])
    # Pre 2:1
    close_pre = pd.Series([10.0, 20.0])
    shares_pre = pd.Series([100.0, 50.0])
    mcap_pre = close_pre * shares_pre  # 1000, 1000
    ttm_eps_pre = ttm_ni / shares_pre  # 1.0, -1.0

    # Post 2:1: price halves, shares double, mcap and NI unchanged
    close_post = close_pre / 2.0
    shares_post = shares_pre * 2.0
    mcap_post = close_post * shares_post
    ttm_eps_unrestated = ttm_eps_pre  # EPS left on pre-split basis (the bug)

    ey_pre = _safe_div(ttm_ni, mcap_pre)
    ey_post = _safe_div(ttm_ni, mcap_post)
    assert np.allclose(ey_pre, ey_post)
    assert list(ey_pre) == [0.1, -0.05]
    assert np.allclose(mcap_pre, mcap_post)

    old_pre = _safe_div(ttm_eps_pre, close_pre)
    old_post = _safe_div(ttm_eps_unrestated, close_post)
    # Old formula moves under mixed bases; new formula does not.
    assert not np.allclose(old_pre, old_post)
    assert np.allclose(old_pre, ey_pre)
