"""Build classic factor returns, FF correlations, and long-only baselines.

Writes (gitignored) under data/factors/ and figures/tables under
reports/classic_factors/. Formation months <= 2023-11 only.

    python scripts/run_classic_factors.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_model.config import ROOT, load_config
from factor_model.evaluation.metrics import summarize
from factor_model.external.ff import download_ff, load_ff_factors
from factor_model.factors.classic import (
    CLASSIC_FEATURES,
    MAX_FORMATION_MONTH,
    MAX_RETURN_MONTH,
    assert_pre_holdout,
    build_classic_factor_returns,
    classic4_composite_score,
    decile_long_short,
    equal_weight_universe,
    filter_panel,
    long_only_top_decile,
    rank_ic,
)

REPORT = ROOT / "reports" / "classic_factors"
DATA_OUT = ROOT / "data" / "factors"
RAW = ROOT / "data" / "panel" / "features_raw.parquet"
PANEL = ROOT / "data" / "panel" / "panel.parquet"

FF_MAP = {
    "value": "HML",
    "size": "SMB",
    "momentum": "UMD",
    "roe": "RMW",
}

SUBPERIODS = {
    "2006-2010": ("2006-01-31", "2010-12-31"),
    "2011-2020": ("2011-01-31", "2020-12-31"),
    "2021-2023": ("2021-01-31", "2023-12-31"),
}

REGIMES = {
    "covid": ("2020-02-29", "2020-12-31"),
    "ratehike": ("2022-01-31", "2022-12-31"),
}


def _corr(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 12:
        return None, len(df)
    return float(df["a"].corr(df["b"])), len(df)


def _rolling_corr_plot(ours: pd.Series, ff: pd.Series, title: str, path: Path) -> None:
    df = pd.DataFrame({"ours": ours, "ff": ff}).dropna()
    roll = df["ours"].rolling(12, min_periods=6).corr(df["ff"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(roll.index, roll.values, color="#1a1a1a", lw=1.2)
    ax.axhline(0.6, color="#888888", ls="--", lw=0.8, label="0.6")
    ax.axhline(0.0, color="#cccccc", lw=0.6)
    ax.set_title(title)
    ax.set_ylabel("12m corr")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _perf_block(rets: pd.Series, ff: pd.Series | None = None) -> dict:
    cfg_rf = 0.0
    out = {
        "ann_return": float("nan"),
        "ann_vol": float("nan"),
        "sharpe": float("nan"),
        "n": int(rets.dropna().shape[0]),
    }
    s = summarize(rets, rf=cfg_rf)
    out.update({k: s[k] for k in ("ann_return", "ann_vol", "sharpe")})
    if ff is not None:
        sf = summarize(ff.reindex(rets.index).dropna(), rf=cfg_rf)
        out["ff_ann_return"] = sf["ann_return"]
        out["ff_ann_vol"] = sf["ann_vol"]
        out["ff_sharpe"] = sf["sharpe"]
    return out


def main() -> None:
    t0 = time.perf_counter()
    REPORT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    print("downloading / loading Ken French…")
    download_ff(force=False)
    ff = load_ff_factors()

    print("loading panel…")
    raw = pd.read_parquet(RAW)
    panel_ranked = pd.read_parquet(PANEL)
    # Restrict ranked panel to same formation window for long-only
    panel_ranked["date"] = pd.to_datetime(panel_ranked["date"])
    panel_ranked = panel_ranked[panel_ranked["date"] <= MAX_FORMATION_MONTH]

    print("building classic factor returns…")
    classic = build_classic_factor_returns(raw)
    classic_path = DATA_OUT / "classic_factor_returns.parquet"
    classic.to_parquet(classic_path, compression="zstd")
    assert_pre_holdout(
        formation=pd.Series([MAX_FORMATION_MONTH]),  # boundary ok
        ret_month=classic.index.to_series(),
    )
    print(f"  wrote {classic_path} shape={classic.shape}")

    # --- Step 3: FF correlations ---
    corr_rows = []
    for name, ff_col in FF_MAP.items():
        for version in ("decile_ew", "decile_vw", "2x3"):
            col = f"{name}_{version}"
            if col not in classic.columns:
                continue
            ours = classic[col]
            theirs = ff[ff_col]
            c, n = _corr(ours, theirs)
            row = {
                "factor": name,
                "version": version,
                "ff": ff_col,
                "corr": c,
                "n": n,
                "pass_0.6": bool(c is not None and c >= 0.6),
            }
            for label, (a, b) in SUBPERIODS.items():
                sub_o = ours.loc[a:b]
                sub_f = theirs.loc[a:b]
                cs, ns = _corr(sub_o, sub_f)
                row[f"corr_{label}"] = cs
                row[f"n_{label}"] = ns
            # perf on overlapping months
            both = pd.DataFrame({"o": ours, "f": theirs}).dropna()
            row.update({f"ours_{k}": v for k, v in _perf_block(both["o"]).items()})
            row.update({f"ff_{k}": v for k, v in _perf_block(both["f"]).items() if k != "n"})
            corr_rows.append(row)
            _rolling_corr_plot(
                ours,
                theirs,
                f"{name} {version} vs {ff_col}",
                REPORT / f"rolling_corr_{name}_{version}.png",
            )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(REPORT / "ff_correlations.csv", index=False)

    # Correlation matrices
    our_cols = [f"{n}_2x3" for n in FF_MAP]
    our_mat = classic[our_cols].corr()
    ff_mat = ff[["HML", "SMB", "UMD", "RMW"]].loc[classic.index.min() : classic.index.max()].corr()
    our_mat.to_csv(REPORT / "corr_matrix_ours_2x3.csv")
    ff_mat.to_csv(REPORT / "corr_matrix_ff.csv")

    # --- Step 4: long-only baselines ---
    filt = filter_panel(raw)
    # Align ranked panel scores onto filt via merge keys
    ranked = panel_ranked[
        ["date", "security_id", "value_bm", "size_ln_mcap", "momentum_12_1", "quality_roe"]
    ].copy()
    ranked = ranked.rename(
        columns={
            "value_bm": "r_value",
            "size_ln_mcap": "r_size",
            "momentum_12_1": "r_mom",
            "quality_roe": "r_roe",
        }
    )
    merged = filt.merge(ranked, on=["date", "security_id"], how="left")
    for c in ("r_value", "r_size", "r_mom", "r_roe"):
        merged[c] = merged[c].fillna(0.0)

    bench = equal_weight_universe(filt)
    # RF from FF for Sharpe
    rf = ff["RF"].reindex(bench.index).fillna(0.0)

    long_series = {"ew_universe": bench}
    turn_series = {}
    long_rows = []
    ic_rows = []

    for name, col in CLASSIC_FEATURES.items():
        score = filt[col]
        # For long-only: attractive direction — high B/M, small, high mom, high ROE
        # long_only_top_decile with long_high=True takes top of score; for size flip score
        if name == "size":
            score_use = -score
            long_high = True
        else:
            score_use = score
            long_high = True
        rets, turns = long_only_top_decile(filt, score_use, long_high=long_high)
        long_series[f"long_{name}"] = rets
        turn_series[f"long_{name}"] = turns
        # diagnostics from raw LS
        ls_ew = classic[f"{name}_decile_ew"]
        ic = rank_ic(filt, score_use)
        stats = summarize(rets, benchmark=bench, turnover=turns, rf=rf)
        stats.update(
            {
                "name": f"long_{name}",
                "top_minus_bottom_ew": float(ls_ew.mean() * 12),  # ann mean of monthly spread
                "rank_ic_mean": float(ic.mean()),
            }
        )
        long_rows.append(stats)
        ic_rows.append(ic.rename(name))

        for reg, (a, b) in REGIMES.items():
            sub = rets.loc[a:b]
            bsub = bench.loc[a:b]
            rs = summarize(sub, benchmark=bsub, rf=rf.reindex(sub.index))
            long_rows.append({**rs, "name": f"long_{name}", "regime": reg})

    # Composite from rank-scaled panel features (size flipped inside helper)
    ranked_for_comp = merged.rename(
        columns={
            "r_value": "value_bm",
            "r_size": "size_ln_mcap",
            "r_mom": "momentum_12_1",
            "r_roe": "quality_roe",
        }
    )
    comp_score = classic4_composite_score(ranked_for_comp)
    rets_c, turns_c = long_only_top_decile(filt, comp_score, long_high=True)
    long_series["long_classic4"] = rets_c
    turn_series["long_classic4"] = turns_c
    ic_c = rank_ic(filt, comp_score)
    filt_comp = filt.copy()
    filt_comp["_comp"] = comp_score.to_numpy()
    ls_comp = decile_long_short(filt_comp, "_comp", long_high=True, value_weighted=False)
    stats = summarize(rets_c, benchmark=bench, turnover=turns_c, rf=rf)
    stats.update(
        {
            "name": "long_classic4",
            "top_minus_bottom_ew": float(ls_comp.mean() * 12),
            "rank_ic_mean": float(ic_c.mean()),
        }
    )
    long_rows.append(stats)
    for reg, (a, b) in REGIMES.items():
        sub = rets_c.loc[a:b]
        bsub = bench.loc[a:b]
        rs = summarize(sub, benchmark=bsub, rf=rf.reindex(sub.index))
        long_rows.append({**rs, "name": "long_classic4", "regime": reg})

    long_rows.append({**summarize(bench, rf=rf), "name": "ew_universe"})

    long_df = pd.DataFrame(long_series)
    long_df.index.name = "return_month"
    if (long_df.index > MAX_RETURN_MONTH).any():
        raise AssertionError("long-only series past holdout")
    long_df.to_parquet(DATA_OUT / "classic_long_only_returns.parquet", compression="zstd")
    pd.DataFrame(turn_series).to_parquet(
        DATA_OUT / "classic_long_only_turnover.parquet", compression="zstd"
    )

    metrics_df = pd.DataFrame(long_rows)
    metrics_df.to_csv(REPORT / "long_only_metrics.csv", index=False)
    if "regime" in metrics_df.columns:
        full = metrics_df[metrics_df["regime"].isna()].copy()
        regimes = metrics_df[metrics_df["regime"].notna()].copy()
    else:
        full = metrics_df.copy()
        regimes = pd.DataFrame()
    full.to_csv(REPORT / "long_only_full_sample.csv", index=False)
    regimes.to_csv(REPORT / "long_only_regimes.csv", index=False)

    summary = {
        "wall_s": round(time.perf_counter() - t0, 2),
        "max_formation": str(MAX_FORMATION_MONTH.date()),
        "max_return_month": str(MAX_RETURN_MONTH.date()),
        "n_return_months": int(len(classic)),
        "classic_columns": list(classic.columns),
        "ff_corr": corr_df.to_dict(orient="records"),
        "breakpoints": "universe-own (top-1000); no NYSE flag",
        "holdout_start_config": cfg["sample"]["holdout_start"],
    }
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("wall_s", "n_return_months")}, indent=2))
    print(corr_df[["factor", "version", "ff", "corr", "pass_0.6"]].to_string(index=False))
    print("done →", REPORT, DATA_OUT)


if __name__ == "__main__":
    main()
