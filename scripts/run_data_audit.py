"""Run the stock-database data audit. Read-only.

Writes data/audit/ (gitignored): monthly counts and coverage as parquet, a JSON
summary, and report.md with the tables quoted in docs/data-audit.md.

    python scripts/run_data_audit.py [--start 1990-01-01] [--dev]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from factor_model.config import load_config
from factor_model.data import audit

OUT = Path(__file__).resolve().parents[1] / "data" / "audit"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="1990-01-01")
    p.add_argument("--top-n", type=int, default=1000)
    p.add_argument("--dev", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    min_price = float(cfg["universe"]["min_price"])

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    (stats, stat_walls), t_stats = audit._timed(lambda: audit.table_stats(dev=args.dev))
    lags, t_lags = audit._timed(lambda: audit.lag_stats(dev=args.dev))
    res, t_monthly = audit._timed(
        lambda: audit.monthly_audit(
            args.start, min_price=min_price, top_n=args.top_n, dev=args.dev
        )
    )

    counts = res["counts"]
    covs = {k: res[k] for k in ("coverage_all", "coverage_price5", "coverage_top_n")}
    delist = audit.delisting_audit(res["securities"], res["price_span"], res["month_ends"])
    sivb = audit.sivb_check(res["priced"], res["securities"])
    core = ("price", "market_cap", "book_equity", "net_income")
    earliest = {k: audit.earliest_good_month(v, core) for k, v in covs.items()}

    ev = res["earnings"]
    upd_lag = (ev["last_updated"] - ev["report_date"]).dt.days
    earnings_meta = {
        "n_events": len(ev),
        "timing_known_share": float(ev["timing"].notna().mean()),
        "timing_counts": ev["timing"].value_counts(dropna=False).to_dict(),
        "last_updated_minus_report_days_quantiles": upd_lag.quantile([0.5, 0.9, 0.99]).to_dict(),
        "share_updated_more_than_30d_after": float((upd_lag > 30).mean()),
        "report_year_counts": ev["report_date"].dt.year.value_counts().sort_index().to_dict(),
    }

    counts.to_parquet(OUT / "monthly_counts.parquet")
    for k, v in covs.items():
        v.to_parquet(OUT / f"{k}.parquet")
    stats.to_csv(OUT / "table_stats.csv")
    lags.to_csv(OUT / "lag_stats.csv", index=False)

    walls = {
        "table_stats_total_s": round(t_stats, 2),
        "table_stats_per_table_s": stat_walls,
        "lag_stats_s": round(t_lags, 2),
        "monthly_audit_total_s": round(t_monthly, 2),
        "monthly_audit_loads_s": res["walls"],
        "monthly_audit_pandas_compute_s": res["compute_s"],
        "total_s": round(time.perf_counter() - t_all, 2),
    }
    summary = {
        "start": args.start,
        "min_price": min_price,
        "top_n": args.top_n,
        "n_month_ends": len(res["month_ends"]),
        "last_month_end": str(res["month_ends"].max().date()),
        "price_outside_listing": res["price_outside_listing"],
        "zero_lag_fundamentals": res["zero_lag"],
        "earliest_good_month": {
            k: {kk: (str(vv.date()) if vv is not None else None) for kk, vv in v.items()}
            for k, v in earliest.items()
        },
        "sivb": sivb,
        "delisting": delist,
        "earnings": earnings_meta,
        "walls": walls,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    md = []
    md.append("## Table stats\n\n" + audit.to_markdown(stats.astype(object).fillna("")))
    md.append("## Monthly counts (December of each year)\n\n" + audit.to_markdown(audit.yearly(counts, "last")))
    for k, v in covs.items():
        md.append(f"## {k} (yearly mean of monthly coverage)\n\n" + audit.to_markdown(audit.yearly(v)))
    md.append("## Vendor vs computed market cap\n\n" + audit.to_markdown(res["mcap_ratio_by_year"]))
    res["delist_rate_12m"].to_parquet(OUT / "delist_rate_12m.parquet")
    md.append(
        "## Share of universe delisting within 12 months (December of each year)\n\n"
        + audit.to_markdown(audit.yearly(res["delist_rate_12m"], "last"), ".3f")
    )
    lag_q = lags[lags["date_source"].isna() & lags["fpe_5y"].isna()]
    md.append("## Lag by period kind\n\n" + audit.to_markdown(lag_q.set_index("period_kind")))
    lag_src = lags[lags["date_source"].notna()]
    md.append("## Lag by date_source\n\n" + audit.to_markdown(lag_src.set_index(["period_kind", "date_source"]).drop(columns="fpe_5y")))
    lag_5y = lags[lags["fpe_5y"].notna()]
    md.append("## Lag by fiscal-period-end 5y bucket\n\n" + audit.to_markdown(lag_5y.set_index(["period_kind", "fpe_5y"]).drop(columns="date_source")))
    (OUT / "report.md").write_text("\n\n".join(md) + "\n")

    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
