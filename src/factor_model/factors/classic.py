"""Classic four factor portfolio returns from the monthly panel.

Formation months are capped at ``MAX_FORMATION_MONTH`` (2023-11-30); the
corresponding last return month is 2023-12. Holdout 2024-01+ is never used.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Last allowed formation month-end (portfolio formed then earns Dec-2023 return).
MAX_FORMATION_MONTH = pd.Timestamp("2023-11-30")
# Last allowed return-month label (month-end of the return period).
MAX_RETURN_MONTH = pd.Timestamp("2023-12-31")

CLASSIC_FEATURES = {
    "value": "value_bm",  # high B/M − low  → HML
    "size": "size_ln_mcap",  # small − big   → SMB
    "momentum": "momentum_12_1",  # winners − losers → UMD
    "roe": "quality_roe",  # robust − weak → RMW
}

# Sign of the long leg on the raw characteristic (True = high values are long).
LONG_HIGH = {
    "value": True,
    "size": False,  # small = low size_ln_mcap
    "momentum": True,
    "roe": True,
}


def return_month(formation: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
    """Map formation month-end to the return month-end (t → t+1)."""
    if isinstance(formation, pd.Timestamp):
        return pd.Timestamp(formation + pd.offsets.MonthEnd(1))
    out = pd.to_datetime(formation) + pd.offsets.MonthEnd(1)
    return pd.to_datetime(out).astype("datetime64[ns]")


def assert_pre_holdout(formation: pd.Series, ret_month: pd.Series | None = None) -> None:
    """Raise if any formation/return month touches the sacred holdout."""
    formation = pd.to_datetime(formation)
    if (formation > MAX_FORMATION_MONTH).any():
        bad = formation[formation > MAX_FORMATION_MONTH].min()
        raise AssertionError(f"formation month {bad.date()} after {MAX_FORMATION_MONTH.date()}")
    if ret_month is not None:
        ret_month = pd.to_datetime(ret_month)
        if (ret_month > MAX_RETURN_MONTH).any():
            bad = ret_month[ret_month > MAX_RETURN_MONTH].min()
            raise AssertionError(f"return month {bad.date()} after {MAX_RETURN_MONTH.date()}")


def filter_panel(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df = df[df["date"] <= MAX_FORMATION_MONTH]
    df = df[df["ret_1m"].notna()]
    df["return_month"] = return_month(df["date"])
    assert_pre_holdout(df["date"], df["return_month"])
    return df


def _decile_labels(x: pd.Series, n: int = 10) -> pd.Series:
    """Cross-sectional deciles 1..n (1=lowest). NaN stays NaN."""
    # rank method first avoids qcut duplicate-edge failures on ties
    r = x.rank(method="first")
    if r.notna().sum() < n:
        return pd.Series(np.nan, index=x.index)
    try:
        return pd.qcut(r, n, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=x.index)


def _portfolio_return(rets: pd.Series, weights: pd.Series | None) -> float:
    mask = rets.notna()
    if weights is None:
        sub = rets[mask]
        return float(sub.mean()) if len(sub) else np.nan
    w = weights[mask].astype(float)
    r = rets[mask].astype(float)
    w = w.where(w > 0)
    both = w.notna() & r.notna()
    if not both.any():
        return np.nan
    w = w[both]
    r = r[both]
    return float((w / w.sum() * r).sum())


def decile_long_short(
    panel: pd.DataFrame,
    feature: str,
    *,
    long_high: bool,
    value_weighted: bool,
) -> pd.Series:
    """Top-minus-bottom decile spread, indexed by return month."""
    rows = []
    for dt, g in panel.groupby("date", sort=True):
        x = g[feature]
        valid = g.loc[x.notna()].copy()
        if len(valid) < 20:
            continue
        valid["_dec"] = _decile_labels(valid[feature])
        top = valid[valid["_dec"] == valid["_dec"].max()]
        bot = valid[valid["_dec"] == valid["_dec"].min()]
        wcol = valid["mcap"] if value_weighted else None
        r_top = _portfolio_return(top["ret_1m"], top["mcap"] if value_weighted else None)
        r_bot = _portfolio_return(bot["ret_1m"], bot["mcap"] if value_weighted else None)
        spread = (r_top - r_bot) if long_high else (r_bot - r_top)
        rows.append({"return_month": return_month(pd.Timestamp(dt)), "ret": spread})
    if not rows:
        return pd.Series(dtype=float, name=feature)
    out = pd.DataFrame(rows).set_index("return_month")["ret"].sort_index()
    out.index = out.index.astype("datetime64[ns]")
    return out


def _breakpoint_masks(x: pd.Series, low_pct: float, high_pct: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    lo = x.quantile(low_pct)
    hi = x.quantile(high_pct)
    return x <= lo, (x > lo) & (x < hi), x >= hi


def ff_2x3_spread(
    panel: pd.DataFrame,
    char_col: str,
    *,
    long_high: bool,
    size_col: str = "size_ln_mcap",
) -> tuple[pd.Series, pd.Series]:
    """FF-style 2×3 value-weighted spreads using *our universe* breakpoints.

    Returns (char_spread, smb_from_this_sort) indexed by return month.
    Char spread mirrors HML/UMD/RMW: 0.5*(S_H+B_H) - 0.5*(S_L+B_L) with
    high/low flipped when ``long_high`` is False (size factor itself uses a
    dedicated SMB construction from the B/M sort).
    """
    rows_char, rows_smb = [], []
    for dt, g in panel.groupby("date", sort=True):
        sub = g.dropna(subset=[char_col, size_col, "ret_1m", "mcap"]).copy()
        if len(sub) < 50:
            continue
        # Size: below median = small (low size_ln_mcap)
        size_med = sub[size_col].median()
        is_small = sub[size_col] <= size_med
        is_big = sub[size_col] > size_med
        low_m, _mid_m, high_m = _breakpoint_masks(sub[char_col], 0.3, 0.7)

        def vw(mask: pd.Series) -> float:
            return _portfolio_return(sub.loc[mask, "ret_1m"], sub.loc[mask, "mcap"])

        # Six portfolios
        sh, sm, sl = vw(is_small & high_m), vw(is_small & _mid_m), vw(is_small & low_m)
        bh, bm, bl = vw(is_big & high_m), vw(is_big & _mid_m), vw(is_big & low_m)

        if long_high:
            char = 0.5 * (sh + bh) - 0.5 * (sl + bl)
        else:
            # For a low-is-long characteristic: swap high/low legs
            char = 0.5 * (sl + bl) - 0.5 * (sh + bh)

        smb = ((sh + sm + sl) / 3.0) - ((bh + bm + bl) / 3.0)
        rm = return_month(pd.Timestamp(dt))
        rows_char.append({"return_month": rm, "ret": char})
        rows_smb.append({"return_month": rm, "ret": smb})

    def _to_s(rows: list[dict], name: str) -> pd.Series:
        if not rows:
            return pd.Series(dtype=float, name=name)
        s = pd.DataFrame(rows).set_index("return_month")["ret"].sort_index()
        s.index = s.index.astype("datetime64[ns]")
        s.name = name
        return s

    return _to_s(rows_char, char_col), _to_s(rows_smb, "smb")


def build_classic_factor_returns(raw: pd.DataFrame) -> pd.DataFrame:
    """Monthly classic factor returns (decile EW/VW + 2×3 VW), pre-holdout only.

    Breakpoints are within our top-1000 universe (no NYSE flag available).
    Index = return month.
    """
    panel = filter_panel(raw)
    series: dict[str, pd.Series] = {}

    for name, col in CLASSIC_FEATURES.items():
        long_high = LONG_HIGH[name]
        series[f"{name}_decile_ew"] = decile_long_short(
            panel, col, long_high=long_high, value_weighted=False
        )
        series[f"{name}_decile_vw"] = decile_long_short(
            panel, col, long_high=long_high, value_weighted=True
        )

    # 2×3: value (B/M), momentum, roe; SMB from the B/M sort
    hml, smb = ff_2x3_spread(panel, "value_bm", long_high=True)
    series["value_2x3"] = hml
    series["size_2x3"] = smb  # SMB from B/M 2×3, FF way
    umd, _ = ff_2x3_spread(panel, "momentum_12_1", long_high=True)
    series["momentum_2x3"] = umd
    rmw, _ = ff_2x3_spread(panel, "quality_roe", long_high=True)
    series["roe_2x3"] = rmw

    out = pd.DataFrame(series).sort_index()
    out.index.name = "return_month"
    if (out.index > MAX_RETURN_MONTH).any():
        raise AssertionError("return months past holdout boundary")
    return out


def long_only_top_decile(
    panel: pd.DataFrame,
    score: pd.Series,
    *,
    long_high: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Equal-weight top decile returns + monthly turnover, indexed by return month.

    ``score`` aligned to ``panel`` index. Turnover = 0.5 * L1 weight change
    (one-way), using security_id membership of the top decile (equal weight).
    """
    df = panel.copy()
    df["_score"] = score.to_numpy()
    rets = []
    turns = []
    prev_ids: set[int] | None = None
    for dt, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["_score", "ret_1m"])
        if len(g) < 20:
            continue
        g = g.copy()
        g["_dec"] = _decile_labels(g["_score"])
        target_dec = g["_dec"].max() if long_high else g["_dec"].min()
        port = g[g["_dec"] == target_dec]
        ids = set(port["security_id"].astype(int))
        r = float(port["ret_1m"].mean())
        if prev_ids is None:
            turn = 1.0  # full invest at start
        else:
            # equal-weight: turnover = fraction replaced
            if not ids:
                turn = np.nan
            else:
                stayed = len(ids & prev_ids) / len(ids)
                turn = 1.0 - stayed
        prev_ids = ids
        rm = return_month(pd.Timestamp(dt))
        rets.append({"return_month": rm, "ret": r})
        turns.append({"return_month": rm, "turnover": turn})
    ret_s = pd.DataFrame(rets).set_index("return_month")["ret"].sort_index()
    turn_s = pd.DataFrame(turns).set_index("return_month")["turnover"].sort_index()
    ret_s.index = ret_s.index.astype("datetime64[ns]")
    turn_s.index = turn_s.index.astype("datetime64[ns]")
    return ret_s, turn_s


def equal_weight_universe(panel: pd.DataFrame) -> pd.Series:
    rows = []
    for dt, g in panel.groupby("date", sort=True):
        r = g["ret_1m"].mean()
        rows.append({"return_month": return_month(pd.Timestamp(dt)), "ret": float(r)})
    s = pd.DataFrame(rows).set_index("return_month")["ret"].sort_index()
    s.index = s.index.astype("datetime64[ns]")
    s.name = "ew_universe"
    return s


def classic4_composite_score(panel_ranked: pd.DataFrame) -> pd.Series:
    """Average of rank-scaled classic four (panel already in [-1,1], missing→0).

    Size is flipped so that small (low raw size) gets a high composite score.
    """
    v = panel_ranked["value_bm"]
    # panel size_ln_mcap rank: high mcap → high rank; flip for "small is attractive"
    s = -panel_ranked["size_ln_mcap"]
    m = panel_ranked["momentum_12_1"]
    q = panel_ranked["quality_roe"]
    return (v + s + m + q) / 4.0


def rank_ic(panel: pd.DataFrame, score: pd.Series) -> pd.Series:
    """Monthly Spearman IC of score vs next-month return."""
    df = panel.copy()
    df["_score"] = score.to_numpy()
    rows = []
    for dt, g in df.groupby("date", sort=True):
        g = g.dropna(subset=["_score", "ret_1m"])
        if len(g) < 20:
            continue
        ic = g["_score"].corr(g["ret_1m"], method="spearman")
        rows.append({"return_month": return_month(pd.Timestamp(dt)), "ic": float(ic)})
    s = pd.DataFrame(rows).set_index("return_month")["ic"].sort_index()
    s.index = s.index.astype("datetime64[ns]")
    return s
