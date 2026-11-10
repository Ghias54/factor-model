"""Cross-sectional rank-average ensemble of member prediction columns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _month_is_constant(pred: pd.Series) -> bool:
    p = pred.dropna()
    if len(p) < 2:
        return True
    return int(p.nunique(dropna=True)) < 2 or float(p.std(ddof=0) or 0.0) == 0.0


def rank_average_predictions(
    frames: dict[str, pd.DataFrame],
    *,
    members: list[str],
    key_cols: list[str] | None = None,
    skip_constant_members: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight average of within-month ranks of member predictions.

    ``frames[name]`` must contain ``formation_month``, ``security_id``, ``prediction``
    plus metadata columns from the first member.

    When ``skip_constant_members`` is True, a member whose predictions are
    constant in a month is dropped from that month's average (no signal).
    Returns ``(ensemble_frame, month_member_log)`` where the log lists which
    members contributed each month.
    """
    key_cols = key_cols or ["formation_month", "security_id"]
    if not members:
        raise ValueError("members required")
    base = frames[members[0]]
    meta_cols = [c for c in base.columns if c != "prediction"]

    # Wide table of raw predictions
    wide = base[key_cols].drop_duplicates()
    for name in members:
        piece = frames[name][key_cols + ["prediction"]].rename(columns={"prediction": name})
        wide = wide.merge(piece, on=key_cols, how="inner")

    log_rows: list[dict] = []
    out_parts: list[pd.DataFrame] = []
    for dt, g in wide.groupby("formation_month", sort=True):
        active = []
        for name in members:
            if skip_constant_members and _month_is_constant(g[name]):
                continue
            active.append(name)
        if not active:
            # all constant → flat prediction (portfolio layer maps to EW)
            pred = pd.Series(0.0, index=g.index)
            used = []
        else:
            ranks = {name: g[name].rank(pct=True, method="average") for name in active}
            pred = pd.concat(ranks, axis=1).mean(axis=1)
            used = list(active)
        part = g[key_cols].copy()
        part["prediction"] = pred.to_numpy()
        out_parts.append(part)
        log_rows.append(
            {
                "formation_month": pd.Timestamp(dt),
                "n_members": len(used),
                "members": ",".join(used) if used else "",
                "dropped": ",".join(m for m in members if m not in used),
            }
        )

    ranks_out = pd.concat(out_parts, ignore_index=True)
    out = base[meta_cols].merge(ranks_out, on=key_cols, how="inner")
    log = pd.DataFrame(log_rows)
    return out, log
