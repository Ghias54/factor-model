"""Compare new data/panel vs data/panel_tradinghost_backup.

Prints row/month/feature counts, per-month universe size diffs, and
per-feature mean/std/null-rate with >1% absolute flags.

    python scripts/compare_panels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "panel"
OLD = ROOT / "data" / "panel_tradinghost_backup"

FEATURE_CANDIDATES = [
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


def _load_raw(path: Path) -> pd.DataFrame:
    p = path / "features_raw.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return pd.read_parquet(p)


def main() -> None:
    old = _load_raw(OLD)
    new = _load_raw(NEW)
    old_meta = json.loads((OLD / "build_meta.json").read_text()) if (OLD / "build_meta.json").exists() else {}
    new_meta = json.loads((NEW / "build_meta.json").read_text()) if (NEW / "build_meta.json").exists() else {}

    feats = [c for c in FEATURE_CANDIDATES if c in old.columns and c in new.columns]
    print("=== counts ===")
    print(f"old rows={len(old):,} months={old['date'].nunique()} features={len(feats)}")
    print(f"new rows={len(new):,} months={new['date'].nunique()} features={len(feats)}")
    print(f"feature list equal: {feats == [c for c in FEATURE_CANDIDATES if c in old.columns]}")
    print(f"old feature_names meta: {old_meta.get('feature_names')}")
    print(f"new feature_names meta: {new_meta.get('feature_names')}")

    old_sz = old.groupby("date").size()
    new_sz = new.groupby("date").size()
    both = old_sz.index.intersection(new_sz.index)
    size_diff = (new_sz.reindex(both) - old_sz.reindex(both)).abs()
    print("\n=== per-month universe size ===")
    print(f"months in both: {len(both)}")
    print(f"months only old: {old_sz.index.difference(new_sz.index).tolist()[:10]}")
    print(f"months only new: {new_sz.index.difference(old_sz.index).tolist()[:10]}")
    print(f"max |Δn|: {int(size_diff.max()) if len(size_diff) else 'n/a'}")
    print(f"months with |Δn|>0: {int((size_diff > 0).sum())}")
    if (size_diff > 0).any():
        bad = size_diff[size_diff > 0].sort_values(ascending=False).head(10)
        for d, v in bad.items():
            print(f"  {pd.Timestamp(d).date()}: old={int(old_sz[d])} new={int(new_sz[d])} Δ={int(v)}")

    rows = []
    for c in feats:
        o, n = old[c], new[c]
        o_null = float(o.isna().mean())
        n_null = float(n.isna().mean())
        o_mean = float(o.mean(skipna=True))
        n_mean = float(n.mean(skipna=True))
        o_std = float(o.std(skipna=True))
        n_std = float(n.std(skipna=True))
        # relative mean/std diffs; absolute for null rate
        mean_rel = abs(n_mean - o_mean) / (abs(o_mean) + 1e-12)
        std_rel = abs(n_std - o_std) / (abs(o_std) + 1e-12)
        null_abs = abs(n_null - o_null)
        flag = null_abs > 0.01 or mean_rel > 0.01
        rows.append(
            {
                "feature": c,
                "old_null": o_null,
                "new_null": n_null,
                "null_abs_diff": null_abs,
                "old_mean": o_mean,
                "new_mean": n_mean,
                "mean_rel_diff": mean_rel,
                "old_std": o_std,
                "new_std": n_std,
                "std_rel_diff": std_rel,
                "flag_>1pct": flag,
            }
        )
    stats = pd.DataFrame(rows)
    print("\n=== per-feature mean/std/null ===")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 50)
    print(
        stats.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )
    flagged = stats[stats["flag_>1pct"]]
    print(f"\n=== flagged (>1% null abs or mean rel): {len(flagged)} ===")
    if len(flagged):
        print(flagged[["feature", "null_abs_diff", "mean_rel_diff", "std_rel_diff"]].to_string(index=False))
    else:
        print("none")

    # exact equality on aligned (date, security_id) if shapes match
    print("\n=== exact equality check (inner join on date,security_id) ===")
    m = old.merge(
        new,
        on=["date", "security_id"],
        how="inner",
        suffixes=("_old", "_new"),
        indicator=False,
    )
    print(f"inner rows: {len(m):,} (old {len(old):,} new {len(new):,})")
    for c in feats:
        a, b = m[f"{c}_old"], m[f"{c}_new"]
        both_na = a.isna() & b.isna()
        eq = (a == b) | both_na
        # float tolerance
        close = np.isclose(a.fillna(0), b.fillna(0), rtol=0, atol=0, equal_nan=True)
        n_diff = int((~eq).sum())
        if n_diff:
            # numerical near-eq
            num_diff = int((~np.isclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), rtol=1e-9, atol=1e-12, equal_nan=True)).sum())
            print(f"  {c}: exact_diff={n_diff:,} float_tol_diff={num_diff:,}")
    exact_all = True
    for c in feats:
        a, b = m[f"{c}_old"], m[f"{c}_new"]
        if not np.allclose(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            rtol=0,
            atol=0,
            equal_nan=True,
        ):
            exact_all = False
            break
    print(f"all features bitwise/NaN-equal on inner join: {exact_all}")

    out = ROOT / "data" / "panel" / "compare_vs_tradinghost.json"
    out.write_text(
        json.dumps(
            {
                "old_rows": len(old),
                "new_rows": len(new),
                "old_months": int(old["date"].nunique()),
                "new_months": int(new["date"].nunique()),
                "features": feats,
                "flagged": flagged.to_dict(orient="records"),
                "stats": stats.to_dict(orient="records"),
                "max_month_size_abs_diff": int(size_diff.max()) if len(size_diff) else None,
                "months_size_diff": int((size_diff > 0).sum()) if len(size_diff) else None,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
