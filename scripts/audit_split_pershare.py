"""Audit per-share feature exposure to the split-adjustment bug.

Read-only. Does not change configs or feature code.
Writes JSON under logs/split_audit/.

    python scripts/audit_split_pershare.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factor_model.config import load_config
from factor_model.data import loaders
from factor_model.data.db import read_sql
from factor_model.universe.build import build_universe

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "logs" / "split_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Step-1 inventory (mirrors compute.py; kept here so the report is self-contained)
FEATURE_BASIS = [
    {
        "feature": "earnings_yield",
        "formula": "sum(q0..q3 eps_diluted) / close",
        "numerator": "TTM diluted EPS from fundamental_quarterly (not split-restated)",
        "denominator": "close (split-adjusted to today)",
        "same_basis": False,
        "per_share": True,
    },
    {
        "feature": "sue",
        "formula": "(eps_actual - eps_estimated) / close",
        "numerator": "earnings_event surprise (vendor may rewrite; not guaranteed PIT)",
        "denominator": "close (split-adjusted to today)",
        "same_basis": "uncertain",  # vendor last_updated rewrites may already be on today basis
        "per_share": True,
    },
    {
        "feature": "turnover",
        "formula": "volume / WAS_diluted",
        "numerator": "volume (split-adjusted)",
        "denominator": "WAS from asof fundamentals (not restated) unless mcap_vendor_sanity fired",
        "same_basis": False,
        "per_share": False,
        "shares_based": True,
    },
    {
        "feature": "value_bm",
        "formula": "total_equity / mcap",
        "numerator": "book equity ($)",
        "denominator": "mcap = close×WAS or vendor after sanity",
        "same_basis": "if_mcap_ok",
        "per_share": False,
    },
    {
        "feature": "size_ln_mcap",
        "formula": "log(mcap)",
        "numerator": "mcap",
        "denominator": "—",
        "same_basis": "if_mcap_ok",
        "per_share": False,
    },
    {
        "feature": "sales_to_price",
        "formula": "TTM revenue / mcap",
        "numerator": "revenue ($)",
        "denominator": "mcap",
        "same_basis": "if_mcap_ok",
        "per_share": False,
    },
    {
        "feature": "cashflow_to_price",
        "formula": "TTM CFO / mcap",
        "numerator": "CFO ($)",
        "denominator": "mcap",
        "same_basis": "if_mcap_ok",
        "per_share": False,
    },
    {
        "feature": "dollar_volume_ln",
        "formula": "log(close × volume)",
        "numerator": "close and volume (both split-adj)",
        "denominator": "—",
        "same_basis": True,
        "per_share": False,
    },
]


def _load_splits() -> pd.DataFrame:
    sp = read_sql(
        """
        SELECT security_id, ex_date, ratio_or_amount
        FROM corporate_action
        WHERE action_type = 'split'
          AND ratio_or_amount IS NOT NULL
          AND ratio_or_amount > 0
        ORDER BY security_id, ex_date
        """
    )
    sp["ex_date"] = pd.to_datetime(sp["ex_date"])
    sp["ratio_or_amount"] = sp["ratio_or_amount"].astype("float64")
    return sp


def _cum_split_after(splits_by: dict[int, pd.DataFrame], sid: int, after: pd.Timestamp) -> float:
    """Product of split ratios with ex_date > after. Price is divided by this product vs pre-split basis."""
    g = splits_by.get(int(sid))
    if g is None or g.empty or pd.isna(after):
        return 1.0
    after = pd.Timestamp(after)
    mask = g["ex_date"] > after
    if not mask.any():
        return 1.0
    return float(g.loc[mask, "ratio_or_amount"].prod())


def step2_pershare(raw: pd.DataFrame, splits: pd.DataFrame) -> dict:
    """Measure earnings_yield and sue exposure to post-period splits."""
    t0 = time.perf_counter()
    splits_by = {sid: g for sid, g in splits.groupby("security_id", sort=False)}
    df = raw[
        [
            "date",
            "security_id",
            "symbol",
            "close",
            "earnings_yield",
            "sue",
            "fund_fiscal_period_end",
            "report_date",
        ]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["fund_fiscal_period_end"] = pd.to_datetime(df["fund_fiscal_period_end"])
    df["report_date"] = pd.to_datetime(df["report_date"])

    # Need TTM quarter EPSs — reload from features_raw if present, else recompute factors only
    # features_raw may not store q*_eps; recompute factor from fund FPE = q0 FPE (conservative:
    # any split after latest quarter end mismatches TTM EPS vs today's close).
    # Also load q3 FPE from asof would be better; approximate lookback start = FPE - 365d.
    df["ey_lookback_start"] = df["fund_fiscal_period_end"] - pd.DateOffset(days=400)
    df["sue_lookback_start"] = df["report_date"]  # surprise vs price at month-end

    # Cumulative split factor AFTER the EPS reference date (all later splits baked into close)
    ey_factor = []
    sue_factor = []
    ey_in_window = []
    sue_in_window = []
    for sid, fpe, rep, d, lb_ey, lb_sue in zip(
        df["security_id"].to_numpy(),
        df["fund_fiscal_period_end"].to_numpy(),
        df["report_date"].to_numpy(),
        df["date"].to_numpy(),
        df["ey_lookback_start"].to_numpy(),
        df["sue_lookback_start"].to_numpy(),
        strict=True,
    ):
        g = splits_by.get(int(sid))
        if g is None or g.empty:
            ey_factor.append(1.0)
            sue_factor.append(1.0)
            ey_in_window.append(False)
            sue_in_window.append(False)
            continue
        # factor for restating EPS onto price basis: eps_adj = eps / prod(ratios after FPE)
        f_ey = _cum_split_after(splits_by, int(sid), fpe if pd.notna(fpe) else pd.NaT)
        f_sue = _cum_split_after(splits_by, int(sid), rep if pd.notna(rep) else pd.NaT)
        ey_factor.append(f_ey)
        sue_factor.append(f_sue)
        if pd.isna(lb_ey) or pd.isna(d):
            ey_in_window.append(False)
        else:
            ey_in_window.append(bool(((g["ex_date"] > pd.Timestamp(lb_ey)) & (g["ex_date"] <= pd.Timestamp(d))).any()))
        if pd.isna(lb_sue) or pd.isna(d):
            sue_in_window.append(False)
        else:
            # split between report and month-end, OR any later split that already restated close
            sue_in_window.append(
                bool(((g["ex_date"] > pd.Timestamp(lb_sue)) & (g["ex_date"] <= pd.Timestamp(d))).any())
                or (f_sue != 1.0)
            )

    df["ey_split_factor"] = ey_factor
    df["sue_split_factor"] = sue_factor
    df["ey_split_in_lookback"] = ey_in_window
    df["sue_split_in_lookback"] = sue_in_window
    # Broader: any subsequent split after EPS period (close already on post-split basis)
    df["ey_has_post_split"] = df["ey_split_factor"].ne(1.0)
    df["sue_has_post_split"] = df["sue_split_factor"].ne(1.0)

    # Restate: earnings_yield_adj = (ttm_eps / factor) / close = earnings_yield / factor
    # because current ey = ttm_eps / close, adj = (ttm_eps/factor)/close = ey/factor
    df["earnings_yield_adj"] = df["earnings_yield"] / df["ey_split_factor"].replace(0, np.nan)
    # sue similarly
    df["sue_adj"] = df["sue"] / df["sue_split_factor"].replace(0, np.nan)

    def _impact(col: str, adj: str, exposed: str) -> dict:
        sub = df[df[exposed] & df[col].notna() & df[adj].notna()].copy()
        if sub.empty:
            return {
                "n_exposed_nonnull": 0,
                "n_change_gt_10pct": 0,
                "n_decile_move": 0,
                "by_year": [],
                "worst30": [],
            }
        # relative change
        rel = (sub[adj] - sub[col]).abs() / (sub[col].abs() + 1e-12)
        sub["rel_change"] = rel
        # cross-sectional decile on original vs adj within month
        # compute ranks on full panel dates for exposed rows' dates
        ranks_before = df.groupby("date")[col].rank(pct=True)
        # temporary adj column on full df for ranking
        tmp = df.copy()
        tmp["_adj"] = tmp[col]
        tmp.loc[sub.index, "_adj"] = sub[adj]
        ranks_after = tmp.groupby("date")["_adj"].rank(pct=True)
        sub["decile_before"] = np.floor(ranks_before.loc[sub.index] * 10).clip(0, 9)
        sub["decile_after"] = np.floor(ranks_after.loc[sub.index] * 10).clip(0, 9)
        sub["decile_move"] = (sub["decile_before"] - sub["decile_after"]).abs() >= 1

        by_year = (
            sub.assign(year=sub["date"].dt.year)
            .groupby("year")
            .agg(
                n=("security_id", "size"),
                n_gt10=("rel_change", lambda s: int((s > 0.10).sum())),
                n_decile=("decile_move", "sum"),
            )
            .reset_index()
        )
        # panel share by year
        panel_year = df.assign(year=df["date"].dt.year).groupby("year").size().rename("n_panel")
        by_year = by_year.merge(panel_year, on="year", how="left")
        by_year["share_of_panel"] = by_year["n"] / by_year["n_panel"]

        worst = sub.nlargest(30, "rel_change")[
            ["symbol", "date", "security_id", col, adj, "rel_change", "decile_before", "decile_after"]
        ].copy()
        worst["date"] = worst["date"].dt.strftime("%Y-%m-%d")
        return {
            "n_exposed_nonnull": int(len(sub)),
            "n_change_gt_10pct": int((rel > 0.10).sum()),
            "n_decile_move": int(sub["decile_move"].sum()),
            "share_of_panel": float(len(sub) / len(df)),
            "by_year": json.loads(by_year.to_json(orient="records")),
            "worst30": json.loads(worst.to_json(orient="records")),
        }

    # Also count lookback-window exposure (split between start and month-end)
    def _counts(exposed_col: str) -> dict:
        exp = df[exposed_col]
        by_year = (
            df.assign(year=df["date"].dt.year, exp=exp)
            .groupby("year")
            .agg(n_exposed=("exp", "sum"), n_panel=("exp", "size"))
            .reset_index()
        )
        by_year["share"] = by_year["n_exposed"] / by_year["n_panel"]
        return {
            "n_exposed": int(exp.sum()),
            "share": float(exp.mean()),
            "by_year": json.loads(by_year.to_json(orient="records")),
        }

    out = {
        "wall_s": round(time.perf_counter() - t0, 2),
        "n_panel": len(df),
        "earnings_yield": {
            "note": (
                "Restated ey = ey / cum_split_factor_after_q0_FPE. "
                "Exposed = any split with ex_date > fund_fiscal_period_end (close already on later basis)."
            ),
            "exposure_post_split": _counts("ey_has_post_split"),
            "exposure_in_lookback_window": _counts("ey_split_in_lookback"),
            "impact_post_split": _impact("earnings_yield", "earnings_yield_adj", "ey_has_post_split"),
        },
        "sue": {
            "note": (
                "sue = (actual-estimate)/close; restated sue = sue / cum_split_after_report_date. "
                "If earnings_event EPS is already vendor-restated, this overstates the problem."
            ),
            "exposure_post_split": _counts("sue_has_post_split"),
            "exposure_in_lookback_window": _counts("sue_split_in_lookback"),
            "impact_post_split": _impact("sue", "sue_adj", "sue_has_post_split"),
        },
    }
    # save detail sample
    detail = df.loc[
        df["ey_has_post_split"] | df["sue_has_post_split"],
        [
            "date",
            "security_id",
            "symbol",
            "earnings_yield",
            "earnings_yield_adj",
            "ey_split_factor",
            "sue",
            "sue_adj",
            "sue_split_factor",
        ],
    ]
    detail.to_parquet(OUT / "pershare_exposed_rows.parquet", compression="zstd")
    return out


def step3_mcap_ratio(univ: pd.DataFrame, splits: pd.DataFrame) -> dict:
    u = univ.copy()
    u["date"] = pd.to_datetime(u["date"])
    both = u[u["mcap_computed"].notna() & u["mcap_vendor"].notna() & (u["mcap_vendor"] > 0)].copy()
    both["ratio"] = both["mcap_computed"] / both["mcap_vendor"]
    # histogram edges
    edges = [0, 0.1, 0.2, 1 / 3, 1 / 1.5, 1 / 1.2, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0, np.inf]
    labels = [
        "<0.1",
        "0.1-0.2",
        "0.2-1/3",
        "1/3-1/1.5",
        "1/1.5-1/1.2",
        "1/1.2-1",
        "1-1.2",
        "1.2-1.5",
        "1.5-2",
        "2-3",
        "3-4",
        "4-5",
        "5-10",
        ">10",
    ]
    both["bin"] = pd.cut(both["ratio"], bins=edges, labels=labels, right=False)
    hist = both["bin"].value_counts().reindex(labels).fillna(0).astype(int)
    in_gate_high = both[(both["ratio"] >= 1.5) & (both["ratio"] <= 3.0)]
    in_gate_low = both[(both["ratio"] >= 1.0 / 3.0) & (both["ratio"] <= 1.0 / 1.5)]
    in_gate = pd.concat([in_gate_high, in_gate_low])

    # attach latest fund date from universe
    splits_by = {sid: g for sid, g in splits.groupby("security_id", sort=False)}
    explained = 0
    no_split = 0
    for sid, fpe, d in zip(
        in_gate["security_id"].to_numpy(),
        pd.to_datetime(in_gate["fiscal_period_end"]).to_numpy(),
        in_gate["date"].to_numpy(),
        strict=True,
    ):
        g = splits_by.get(int(sid))
        if g is None or g.empty or pd.isna(fpe):
            no_split += 1
            continue
        hit = ((g["ex_date"] > pd.Timestamp(fpe)) & (g["ex_date"] <= pd.Timestamp(d))).any()
        # also count any post-FPE split (already in close)
        hit = hit or ((g["ex_date"] > pd.Timestamp(fpe)).any())
        if hit:
            explained += 1
        else:
            no_split += 1

    return {
        "n_both": int(len(both)),
        "histogram": {str(k): int(v) for k, v in hist.items()},
        "in_gate_1.5_to_3": int(len(in_gate_high)),
        "in_gate_1/3_to_1/1.5": int(len(in_gate_low)),
        "in_gate_total": int(len(in_gate)),
        "in_gate_split_explained": explained,
        "in_gate_no_split": no_split,
        "in_gate_split_explained_share": float(explained / len(in_gate)) if len(in_gate) else None,
        "median_ratio": float(both["ratio"].median()),
        "share_within_1.2": float(((both["ratio"] - 1).abs() < 0.2).mean()),
    }


def step3_ratio_15_test() -> dict:
    """Rebuild universe with max_ratio=1.5 without changing default config file."""
    t0 = time.perf_counter()
    cfg = load_config()
    cfg = json.loads(json.dumps(cfg))  # deep copy via JSON
    cfg.setdefault("universe", {}).setdefault("mcap_vendor_sanity", {})
    cfg["universe"]["mcap_vendor_sanity"]["enabled"] = True
    cfg["universe"]["mcap_vendor_sanity"]["max_ratio"] = 1.5

    univ15, meta15 = build_universe(cfg=cfg)
    univ3 = pd.read_parquet(PANEL / "universe.parquet")
    univ3["date"] = pd.to_datetime(univ3["date"])
    univ15["date"] = pd.to_datetime(univ15["date"])

    k3 = set(zip(univ3["date"].astype(str), univ3["security_id"], strict=False))
    k15 = set(zip(univ15["date"].astype(str), univ15["security_id"], strict=False))
    dropped = k3 - k15
    added = k15 - k3

    # top securities by months affected
    drop_df = univ3.merge(
        univ15[["date", "security_id"]], on=["date", "security_id"], how="left", indicator=True
    )
    drop_df = drop_df[drop_df["_merge"] == "left_only"]
    top_drop = (
        drop_df.groupby("security_id")
        .agg(symbol=("symbol", "first"), n_months=("date", "size"), med_rank=("rank", "median"))
        .sort_values("n_months", ascending=False)
        .head(20)
        .reset_index()
    )
    add_df = univ15.merge(
        univ3[["date", "security_id"]], on=["date", "security_id"], how="left", indicator=True
    )
    add_df = add_df[add_df["_merge"] == "left_only"]
    top_add = (
        add_df.groupby("security_id")
        .agg(symbol=("symbol", "first"), n_months=("date", "size"), med_rank=("rank", "median"))
        .sort_values("n_months", ascending=False)
        .head(20)
        .reset_index()
    )

    # save test universe (gitignored via parquet)
    univ15.to_parquet(OUT / "universe_max_ratio_1_5.parquet", compression="zstd")
    (OUT / "universe_max_ratio_1_5_meta.json").write_text(json.dumps(meta15, indent=2, default=str))

    return {
        "wall_s": round(time.perf_counter() - t0, 2),
        "meta_max_ratio_1_5": meta15,
        "meta_current_max_ratio_3": json.loads((PANEL / "universe_meta.json").read_text()),
        "n_sanity_replaced_1_5": int(meta15.get("n_mcap_sanity_replaced", 0)),
        "n_sanity_replaced_3": int(
            json.loads((PANEL / "universe_meta.json").read_text()).get("n_mcap_sanity_replaced", 0)
        ),
        "membership_dropped": len(dropped),
        "membership_added": len(added),
        "top20_dropped": json.loads(top_drop.to_json(orient="records")),
        "top20_added": json.loads(top_add.to_json(orient="records")),
    }


def main() -> None:
    t_all = time.perf_counter()
    print("loading panel + splits…")
    raw = pd.read_parquet(PANEL / "features_raw.parquet")
    univ = pd.read_parquet(PANEL / "universe.parquet")
    splits = _load_splits()
    print(f"panel={len(raw):,} splits={len(splits):,}")

    step1 = {"features": FEATURE_BASIS}
    (OUT / "step1_feature_basis.json").write_text(json.dumps(step1, indent=2))

    print("step2 per-share impact…")
    step2 = step2_pershare(raw, splits)
    (OUT / "step2_pershare_impact.json").write_text(json.dumps(step2, indent=2, default=str))
    print(
        "  ey exposed",
        step2["earnings_yield"]["exposure_post_split"]["n_exposed"],
        "sue exposed",
        step2["sue"]["exposure_post_split"]["n_exposed"],
    )

    print("step3 mcap ratio histogram…")
    step3_hist = step3_mcap_ratio(univ, splits)
    (OUT / "step3_mcap_ratio_hist.json").write_text(json.dumps(step3_hist, indent=2))
    print("  in-gate [1.5,3]∪[1/3,1/1.5]", step3_hist["in_gate_total"])

    print("step3 max_ratio=1.5 universe test (no config write)…")
    step3_test = step3_ratio_15_test()
    (OUT / "step3_max_ratio_1_5_test.json").write_text(json.dumps(step3_test, indent=2, default=str))
    print(
        "  replaced 3→1.5:",
        step3_test["n_sanity_replaced_3"],
        "→",
        step3_test["n_sanity_replaced_1_5"],
        "swaps",
        step3_test["membership_dropped"],
        "/",
        step3_test["membership_added"],
    )

    summary = {
        "wall_s": round(time.perf_counter() - t_all, 2),
        "step1": step1,
        "step2": step2,
        "step3_hist": step3_hist,
        "step3_test": {
            k: step3_test[k]
            for k in (
                "wall_s",
                "n_sanity_replaced_1_5",
                "n_sanity_replaced_3",
                "membership_dropped",
                "membership_added",
                "top20_dropped",
                "top20_added",
            )
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"done wall={summary['wall_s']}s → {OUT}")


if __name__ == "__main__":
    main()
