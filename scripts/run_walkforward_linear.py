"""Walk-forward Lasso/Ridge: train, evaluate, feature-selection report.

    python scripts/run_walkforward_linear.py
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
from factor_model.evaluation.bootstrap import block_bootstrap_ci, sharpe_diff_ci
from factor_model.evaluation.metrics import summarize
from factor_model.external.ff import load_ff_factors
from factor_model.factors.classic import (
    MAX_FORMATION_MONTH,
    classic4_composite_score,
    decile_long_short,
    equal_weight_universe,
    long_only_top_decile,
    rank_ic,
)
from factor_model.models.linear import LassoModel, RidgeModel
from factor_model.walkforward.engine import WalkForwardEngine, monthly_rank_ic

PANEL = ROOT / "data" / "panel" / "panel.parquet"
PRED = ROOT / "data" / "predictions"
FITS = ROOT / "data" / "predictions" / "fits"
REPORT = ROOT / "reports" / "linear"
OOS_START = pd.Timestamp("2011-01-31")
OOS_END = MAX_FORMATION_MONTH  # 2023-11


def _portfolio_from_predictions(preds: pd.DataFrame) -> tuple[pd.Series, pd.Series, float, float]:
    """Long-only top decile + diagnostics from prediction frame."""
    panel = preds.rename(columns={"formation_month": "date", "realized_ret": "ret_1m"})
    rets, turns = long_only_top_decile(panel, panel["prediction"], long_high=True)
    ic = rank_ic(panel, panel["prediction"])
    ls = decile_long_short(panel, "prediction", long_high=True, value_weighted=False)
    return rets, turns, float(ic.mean()), float(ls.mean() * 12)


def _excess_by_year(active: pd.Series) -> pd.DataFrame:
    rows = []
    for y, g in active.groupby(active.index.year):
        rows.append({"year": int(y), "excess": float((1 + g).prod() - 1), "ann_excess": float(g.mean() * 12)})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.perf_counter()
    PRED.mkdir(parents=True, exist_ok=True)
    FITS.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    seed = int(cfg["compute"]["random_seed"])

    print("loading panel…")
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])

    eng = WalkForwardEngine(n_jobs=cfg["compute"]["n_jobs"], random_seed=seed)
    # Timing dry-run assertions for every year
    prepared = eng.prepare_panel(panel)
    calendar = eng._formation_calendar(prepared)
    timing_rows = []
    for year, train_end, first_pred, pred_dates in eng._year_windows(calendar):
        train_dates = calendar[(calendar >= eng.sample_start) & (calendar <= train_end)]
        eng.assert_timing(train_dates, first_pred)
        timing_rows.append(
            {
                "year": year,
                "train_end": str(train_end.date()),
                "first_pred": str(first_pred.date()),
                "last_pred": str(pred_dates.max().date()),
                "n_train_months": int(len(train_dates)),
                "n_pred_months": int(len(pred_dates)),
                "ok": True,
            }
        )
    pd.DataFrame(timing_rows).to_csv(REPORT / "timing_checks.csv", index=False)
    print("timing checks OK:", len(timing_rows), "years")

    results = {}
    for Model in (LassoModel, RidgeModel):
        name = Model.name
        print(f"walk-forward {name}…")
        t1 = time.perf_counter()
        res = eng.run(Model(), panel)
        out_path = PRED / f"{name}_predictions.parquet"
        res.to_parquet(out_path)
        res.fits_to_json(FITS / f"{name}_year_fits.json")
        results[name] = res
        print(f"  {name}: {len(res.predictions):,} rows, wall={time.perf_counter()-t1:.1f}s")

    # --- OOS panel slice for baselines ---
    oos_panel = panel[(panel["date"] >= OOS_START) & (panel["date"] <= OOS_END)].copy()
    oos_panel = oos_panel[oos_panel["ret_1m"].notna()]

    bench = equal_weight_universe(oos_panel)
    comp_score = classic4_composite_score(oos_panel)
    classic_rets, classic_turns = long_only_top_decile(oos_panel, comp_score, long_high=True)
    classic_ic = rank_ic(oos_panel, comp_score)
    classic_ls = decile_long_short(
        oos_panel.assign(_c=comp_score.to_numpy()), "_c", long_high=True, value_weighted=False
    )

    try:
        ff = load_ff_factors()
        rf = ff["RF"].reindex(bench.index).fillna(0.0)
    except Exception:
        rf = 0.0

    series = {"ew_universe": bench, "classic4": classic_rets}
    turns_map = {"classic4": classic_turns}
    diag = {
        "classic4": {
            "rank_ic_mean": float(classic_ic.mean()),
            "rank_ic_pos_share": float((classic_ic > 0).mean()),
            "top_minus_bottom_ew_ann": float(classic_ls.mean() * 12),
        }
    }

    for name, res in results.items():
        rets, turns, ic_mean, ls_ann = _portfolio_from_predictions(res.predictions)
        # IC positive share
        panel_p = res.predictions.rename(columns={"formation_month": "date", "realized_ret": "ret_1m"})
        ic_s = rank_ic(panel_p, panel_p["prediction"])
        series[name] = rets
        turns_map[name] = turns
        diag[name] = {
            "rank_ic_mean": ic_mean,
            "rank_ic_pos_share": float((ic_s > 0).mean()),
            "top_minus_bottom_ew_ann": ls_ann,
            "mean_monthly_ic_engine": monthly_rank_ic(
                res.predictions["prediction"],
                res.predictions["realized_ret"],
                res.predictions["formation_month"],
            ),
        }

    # Align all series on OOS return months
    ret_df = pd.DataFrame(series).sort_index()
    ret_df = ret_df.loc[
        (ret_df.index >= pd.Timestamp("2011-02-28"))
        & (ret_df.index <= pd.Timestamp("2023-12-31"))
    ]
    ret_df.to_parquet(PRED / "linear_portfolio_returns.parquet", compression="zstd")

    # Metrics table
    metric_rows = []
    for name in ret_df.columns:
        rets = ret_df[name]
        turn = turns_map.get(name)
        if turn is not None:
            turn = turn.reindex(rets.index)
        stats = summarize(
            rets,
            benchmark=ret_df["ew_universe"] if name != "ew_universe" else None,
            turnover=turn,
            rf=rf if not isinstance(rf, float) else rf,
        )
        stats["name"] = name
        stats.update(diag.get(name, {}))
        # Sharpe diff vs classic4
        if name not in ("ew_universe", "classic4"):
            sd = sharpe_diff_ci(rets, ret_df["classic4"], rf=0.0 if isinstance(rf, float) else 0.0)
            stats["sharpe_minus_classic4"] = sd["point"]
            stats["sharpe_minus_classic4_lo"] = sd["lo"]
            stats["sharpe_minus_classic4_hi"] = sd["hi"]
            excess = rets - ret_df["ew_universe"]
            ex_ci = block_bootstrap_ci(excess, stat_fn=lambda s: float(s.mean() * 12))
            stats["excess_ann_vs_ew"] = ex_ci["point"]
            stats["excess_ann_vs_ew_lo"] = ex_ci["lo"]
            stats["excess_ann_vs_ew_hi"] = ex_ci["hi"]
        metric_rows.append(stats)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORT / "oos_metrics.csv", index=False)

    # Excess by year
    year_rows = []
    for name in ("lasso", "ridge", "classic4"):
        if name not in ret_df:
            continue
        active = ret_df[name] - ret_df["ew_universe"]
        ydf = _excess_by_year(active)
        ydf["name"] = name
        year_rows.append(ydf)
    years = pd.concat(year_rows, ignore_index=True)
    years.to_csv(REPORT / "excess_by_year.csv", index=False)

    # Regimes
    regimes = {
        "covid": ("2020-02-29", "2020-12-31"),
        "ratehike": ("2022-01-31", "2022-12-31"),
    }
    reg_rows = []
    for reg, (a, b) in regimes.items():
        for name in ret_df.columns:
            sub = ret_df[name].loc[a:b]
            bsub = ret_df["ew_universe"].loc[a:b]
            s = summarize(sub, benchmark=bsub if name != "ew_universe" else None, rf=0.0)
            s.update({"name": name, "regime": reg})
            reg_rows.append(s)
    pd.DataFrame(reg_rows).to_csv(REPORT / "regimes.csv", index=False)

    # Cumulative returns chart
    wealth = (1.0 + ret_df.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, color in zip(
        ["ew_universe", "classic4", "lasso", "ridge"],
        ["#888888", "#1a1a1a", "#b85c38", "#2f5d8a"],
        strict=True,
    ):
        if col in wealth:
            ax.plot(wealth.index, wealth[col], label=col, color=color, lw=1.3)
    ax.set_title("OOS cumulative return (formation 2011-01 → 2023-11)")
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORT / "cumulative_returns.png", dpi=120)
    plt.close(fig)

    # --- Step 5: Lasso / Ridge feature selection ---
    lasso_fits = results["lasso"].year_fits
    ridge_fits = results["ridge"].year_fits
    features = list(eng.feature_names)

    # Sign matrix for Lasso
    sign_rows = []
    for fit in lasso_fits:
        coefs = fit.coefficients or {}
        for f in features:
            c = coefs.get(f, 0.0)
            if abs(c) < 1e-12:
                s = 0
            else:
                s = 1 if c > 0 else -1
            sign_rows.append({"year": fit.year, "feature": f, "sign": s, "coef": c})
    sign_df = pd.DataFrame(sign_rows)
    sign_wide = sign_df.pivot(index="feature", columns="year", values="sign").fillna(0).astype(int)
    year_cols = list(sign_wide.columns)
    sign_wide["share_years_nonzero"] = (sign_wide[year_cols] != 0).mean(axis=1)
    sign_wide.to_csv(REPORT / "lasso_selection_signs.csv")

    # Flip flags
    flips = []
    for f, row in sign_wide[year_cols].iterrows():
        nz = row[row != 0]
        if len(nz) >= 2 and nz.min() < 0 and nz.max() > 0:
            flips.append(f)
    pd.Series(flips, name="feature").to_csv(REPORT / "lasso_sign_flips.csv", index=False)

    # Average |coef| ranks
    def avg_abs(fits):
        mat = []
        for fit in fits:
            coefs = fit.coefficients or {}
            mat.append({f: abs(coefs.get(f, 0.0)) for f in features})
        return pd.DataFrame(mat).mean()

    lasso_imp = avg_abs(lasso_fits).sort_values(ascending=False)
    ridge_imp = avg_abs(ridge_fits).sort_values(ascending=False)
    imp = pd.DataFrame(
        {
            "lasso_avg_abs_coef": lasso_imp,
            "lasso_rank": lasso_imp.rank(ascending=False),
            "ridge_avg_abs_coef": ridge_imp,
            "ridge_rank": ridge_imp.rank(ascending=False),
        }
    ).sort_values("lasso_rank")
    imp["rank_corr_spearman"] = float(
        lasso_imp.rank(ascending=False).corr(ridge_imp.rank(ascending=False), method="spearman")
    )
    imp.to_csv(REPORT / "feature_importance_lasso_ridge.csv")

    summary = {
        "wall_s": round(time.perf_counter() - t0, 2),
        "target_scaling": results["lasso"].target_scaling,
        "oos_formation": f"{OOS_START.date()} → {OOS_END.date()}",
        "n_timing_years": len(timing_rows),
        "metrics": metrics.to_dict(orient="records"),
        "lasso_sign_flips": flips,
        "lasso_ridge_rank_corr": float(
            lasso_imp.rank(ascending=False).corr(ridge_imp.rank(ascending=False), method="spearman")
        ),
        "model_interface": "BaseModel: fit/predict/param_grid/name/clone; coefficients_ optional",
    }
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(metrics[["name", "ann_return", "ann_vol", "sharpe", "max_drawdown"]].to_string(index=False))
    print("done wall=", summary["wall_s"], "→", REPORT)


if __name__ == "__main__":
    main()
