"""Build the monthly universe, raw features, and processed panel.

Writes (gitignored) under data/panel/:
  universe.parquet, features_raw.parquet, panel.parquet, build_meta.json

Also writes docs/panel-report.md (committed).

    python scripts/build_panel.py [--dev]
"""
from __future__ import annotations

import argparse
import json
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from factor_model.config import load_config
from factor_model.data import loaders
from factor_model.features.compute import (
    FEATURE_NAMES,
    compute_targets_and_features,
    preprocess_panel,
)
from factor_model.universe.build import build_universe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "panel"
REPORT = ROOT / "docs" / "panel-report.md"


def _peak_mb() -> float:
    # Linux: max RSS in KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _coverage_by_year(raw: pd.DataFrame) -> pd.DataFrame:
    yrs = raw["date"].dt.year
    rows = []
    for y, g in raw.groupby(yrs):
        rec = {"year": int(y), "n": len(g)}
        for c in FEATURE_NAMES:
            rec[c] = float(g[c].notna().mean())
        rows.append(rec)
    return pd.DataFrame(rows).set_index("year")


def _to_md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(float(v), floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def write_report(
    univ: pd.DataFrame,
    raw: pd.DataFrame,
    panel: pd.DataFrame,
    univ_meta: dict,
    feat_meta: dict,
    peak_mb: float,
    total_s: float,
) -> None:
    counts = univ.groupby("date").size()
    cov = _coverage_by_year(raw)
    weak = []
    for y, row in cov.iterrows():
        for c in FEATURE_NAMES:
            if row[c] < 0.70:
                weak.append((int(y), c, round(float(row[c]), 3)))

    stats = raw[FEATURE_NAMES].agg(["median", lambda s: s.quantile(0.01), lambda s: s.quantile(0.99)]).T
    stats.columns = ["median", "p1", "p99"]

    # mcap cross-check vs vendor on universe dates
    vendor = loaders.vendor_market_cap_on_dates(univ["date"].unique())
    m = univ.merge(vendor, on=["security_id", "date"], how="left", suffixes=("", "_v"))
    both = m.dropna(subset=["mcap", "market_cap"])
    both = both[both["market_cap"] > 0]
    ratio = both["mcap"] / both["market_cap"]
    mcap_check = {
        "median_ratio": float(ratio.median()),
        "share_within_20pct": float(((ratio - 1).abs() < 0.2).mean()),
        "n": len(ratio),
    }

    # AAPL / SIVB
    aapl = raw[raw["symbol"] == "AAPL"].sort_values("date")
    sivb = raw[raw["symbol"] == "SIVB"].sort_values("date")
    spot = {
        "aapl_n_months": len(aapl),
        "aapl_first": str(aapl["date"].min().date()) if len(aapl) else None,
        "aapl_last": str(aapl["date"].max().date()) if len(aapl) else None,
        "sivb_n_months": len(sivb),
        "sivb_first": str(sivb["date"].min().date()) if len(sivb) else None,
        "sivb_last": str(sivb["date"].max().date()) if len(sivb) else None,
        "sivb_in_2023_02": bool((sivb["date"] == "2023-02-28").any()) if len(sivb) else False,
        "sivb_in_2023_03": bool((sivb["date"] == "2023-03-31").any()) if len(sivb) else False,
    }

    # Target sanity vs ^GSPC over the same t → t+1 window as ret_1m.
    idx = loaders.index_prices("^GSPC", "2005-01-01", "2026-10-15")
    me = raw[["date"]].drop_duplicates().sort_values("date").copy()
    me["date"] = pd.to_datetime(me["date"]).astype("datetime64[ns]")
    idx = idx.copy()
    idx["date"] = pd.to_datetime(idx["date"]).astype("datetime64[ns]")
    idx_me = pd.merge_asof(
        me, idx[["date", "close_adj"]].sort_values("date"), on="date", direction="backward"
    )
    idx_me["mkt_fwd"] = idx_me["close_adj"].shift(-1) / idx_me["close_adj"] - 1.0
    u_ret = raw.groupby("date")["ret_1m"].mean()
    aligned = pd.DataFrame({"u": u_ret, "mkt": idx_me.set_index("date")["mkt_fwd"]}).dropna()
    target_sanity = {
        "mean_universe_monthly_ret": float(u_ret.mean()),
        "corr_vs_gspc": float(aligned["u"].corr(aligned["mkt"])) if len(aligned) > 12 else None,
        "n_months": int(len(aligned)),
    }

    sizes = {p.name: p.stat().st_size for p in OUT.glob("*.parquet")}

    cov_pct = (cov[FEATURE_NAMES] * 100).round(0)
    lines = [
        "# Panel build report",
        "",
        f"Built {time.strftime('%Y-%m-%d %H:%M')} · sample {univ_meta['start']} → {univ_meta['end']}.",
        "",
        "## Universe",
        "",
        f"- Months: {univ_meta['n_months']} · rows: {univ_meta['n_rows']:,} · securities: {univ_meta['n_securities']:,}",
        f"- Top N: {univ_meta['top_n']} · min `close_raw`: ${univ_meta['min_price']}",
        (
            f"- Vendor mcap fallback for ranking: {univ_meta['n_vendor_mcap_fallback']:,} rows "
            f"({100 * univ_meta['pct_vendor_mcap_fallback']:.2f}%)"
        ),
        (
            f"- Months with fewer than {univ_meta['top_n']} names: "
            + (
                ", ".join(f"{k} ({v})" for k, v in univ_meta["months_below_top_n"].items())
                if univ_meta["months_below_top_n"]
                else "none"
            )
        ),
        "",
        "### Stocks per December",
        "",
        _to_md(
            counts[counts.index.month == 12]
            .rename("n")
            .to_frame()
            .assign(year=lambda d: d.index.year)
            .set_index("year"),
            ".0f",
        ),
        "",
        "## Feature coverage (% non-missing raw, by year)",
        "",
        _to_md(cov_pct, ".0f"),
        "",
        "### Features below 70% in any year",
        "",
    ]
    if weak:
        lines.append("| year | feature | coverage |")
        lines.append("|---|---|---|")
        for y, c, v in weak:
            lines.append(f"| {y} | {c} | {100 * v:.1f}% |")
    else:
        lines.append("None.")
    lines += [
        "",
        "## Raw feature summary stats",
        "",
        _to_md(stats, ".4g"),
        "",
        "## Market-cap cross-check (computed vs vendor)",
        "",
        f"- Median ratio: {mcap_check['median_ratio']:.4f}",
        f"- Share within ±20%: {100 * mcap_check['share_within_20pct']:.1f}% (n={mcap_check['n']:,})",
        "",
        "## Spot checks",
        "",
        f"- AAPL: {spot['aapl_n_months']} months, {spot['aapl_first']} → {spot['aapl_last']}",
        (
            f"- SIVB: {spot['sivb_n_months']} months, {spot['sivb_first']} → {spot['sivb_last']}; "
            f"in 2023-02: {spot['sivb_in_2023_02']}; in 2023-03: {spot['sivb_in_2023_03']}"
        ),
        "",
        "## Target sanity",
        "",
        f"- Mean equal-weight universe monthly return: {100 * target_sanity['mean_universe_monthly_ret']:.3f}%",
        f"- Corr vs ^GSPC monthly: {target_sanity['corr_vs_gspc']:.3f} (n={target_sanity['n_months']})",
        "",
        "## Performance",
        "",
        f"- Total wall: {total_s:.1f}s · universe {univ_meta['wall_s']}s · features {feat_meta['wall_s']}s",
        f"- Feature stage walls: `{json.dumps(feat_meta['walls'])}`",
        f"- Peak RSS: {peak_mb:.0f} MB",
        f"- Parquet sizes: { {k: f'{v/1e6:.1f} MB' for k, v in sizes.items()} }",
        "",
        "## Lookahead",
        "",
        (
            "See `tests/test_panel_lookahead.py` (sampled 200 rows: fundamentals "
            "`date_available` ≤ 16:00 ET on t; earnings `report_date` < t)."
        ),
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    (OUT / "report_extras.json").write_text(
        json.dumps(
            {
                "weak_features": weak,
                "mcap_check": mcap_check,
                "spot": spot,
                "target_sanity": target_sanity,
                "peak_mb": peak_mb,
                "total_s": total_s,
            },
            indent=2,
            default=str,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    args = p.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    t_all = time.perf_counter()

    univ, univ_meta = build_universe(args.start, args.end, cfg=cfg, dev=args.dev)
    univ.to_parquet(OUT / "universe.parquet", compression="zstd")
    (OUT / "universe_meta.json").write_text(json.dumps(univ_meta, indent=2, default=str))
    print("universe", univ_meta)

    raw, feat_meta = compute_targets_and_features(
        univ, cfg=cfg, n_jobs=cfg["compute"]["n_jobs"], dev=args.dev
    )
    raw.to_parquet(OUT / "features_raw.parquet", compression="zstd")
    panel = preprocess_panel(raw)
    panel.to_parquet(OUT / "panel.parquet", compression="zstd")
    (OUT / "build_meta.json").write_text(json.dumps(feat_meta, indent=2, default=str))
    print("features", feat_meta)

    total_s = time.perf_counter() - t_all
    peak_mb = _peak_mb()
    write_report(univ, raw, panel, univ_meta, feat_meta, peak_mb, total_s)
    print(f"done wall={total_s:.1f}s peak_rss={peak_mb:.0f}MB wrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
