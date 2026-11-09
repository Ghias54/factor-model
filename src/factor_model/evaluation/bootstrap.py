"""Block-bootstrap confidence intervals for portfolio metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model.evaluation.metrics import sharpe


def block_bootstrap_ci(
    series: pd.Series,
    *,
    stat_fn,
    block: int = 12,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile CI for ``stat_fn(resampled_series)`` using non-overlapping-ish blocks.

    Blocks are sampled with replacement from contiguous ``block``-month chunks
    of the aligned series (NaNs dropped first).
    """
    x = series.dropna().astype(float)
    n = len(x)
    if n < block * 2:
        point = float(stat_fn(x))
        return {"point": point, "lo": point, "hi": point, "n": n, "n_boot": 0}
    values = x.to_numpy()
    n_blocks = int(np.ceil(n / block))
    # overlapping starts for more variety
    max_start = n - block
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        chunks = [values[s : s + block] for s in starts]
        sample = np.concatenate(chunks)[:n]
        boots[i] = float(stat_fn(pd.Series(sample)))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(stat_fn(x)),
        "lo": float(lo),
        "hi": float(hi),
        "n": n,
        "n_boot": n_boot,
        "block": block,
    }


def sharpe_diff_ci(
    a: pd.Series,
    b: pd.Series,
    *,
    rf: float = 0.0,
    block: int = 12,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Block-bootstrap CI for Sharpe(a) - Sharpe(b) on the common index."""
    df = pd.DataFrame({"a": a, "b": b}).dropna()

    def _diff(sample_idx_values: pd.Series) -> float:
        # sample_idx_values unused pattern — we bootstrap rows of df
        raise NotImplementedError

    n = len(df)
    if n < block * 2:
        sa = sharpe(df["a"], rf=rf)
        sb = sharpe(df["b"], rf=rf)
        d = sa - sb
        return {"point": d, "lo": d, "hi": d, "n": n, "n_boot": 0}

    values = df.to_numpy()
    max_start = n - block
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        chunks = [values[s : s + block] for s in starts]
        sample = np.concatenate(chunks, axis=0)[:n]
        sa = sharpe(pd.Series(sample[:, 0]), rf=rf)
        sb = sharpe(pd.Series(sample[:, 1]), rf=rf)
        boots[i] = sa - sb
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    point = sharpe(df["a"], rf=rf) - sharpe(df["b"], rf=rf)
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n": n, "n_boot": n_boot, "block": block}
