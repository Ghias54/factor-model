"""Cross-sectional rank-average ensemble of member prediction columns."""
from __future__ import annotations

import pandas as pd


def rank_average_predictions(
    frames: dict[str, pd.DataFrame],
    *,
    members: list[str],
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Equal-weight average of within-month ranks of member predictions.

    ``frames[name]`` must contain ``formation_month``, ``security_id``, ``prediction``
    plus metadata columns from the first member.
    """
    key_cols = key_cols or ["formation_month", "security_id"]
    if not members:
        raise ValueError("members required")
    base = frames[members[0]]
    meta_cols = [c for c in base.columns if c != "prediction"]

    ranks = None
    for name in members:
        df = frames[name][key_cols + ["prediction"]].copy()
        df[name] = df.groupby("formation_month")["prediction"].rank(pct=True, method="average")
        piece = df[key_cols + [name]]
        ranks = piece if ranks is None else ranks.merge(piece, on=key_cols, how="inner")
    assert ranks is not None
    ranks["prediction"] = ranks[members].mean(axis=1)
    out = base[meta_cols].merge(ranks[key_cols + ["prediction"]], on=key_cols, how="inner")
    return out
