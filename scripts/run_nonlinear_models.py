"""Walk-forward RF / XGBoost / MLP / ensemble + costs, FF alpha, interpretability.

    python scripts/run_nonlinear_models.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from factor_model.config import ROOT, load_config
from factor_model.evaluation.bootstrap import block_bootstrap_ci, sharpe_diff_ci
from factor_model.evaluation.costs import apply_costs, breakeven_bps_vs_benchmark
from factor_model.evaluation.ff_alpha import ff6_alpha
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
from factor_model.features.compute import FEATURE_NAMES
from factor_model.models import (
    LassoModel,
    MLPModel,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
    rank_average_predictions,
)
from factor_model.walkforward.engine import WalkForwardEngine, monthly_rank_ic

PANEL = ROOT / "data" / "panel" / "panel.parquet"
PRED = ROOT / "data" / "predictions"
FITS = PRED / "fits"
REPORT = ROOT / "reports" / "models"
INTERP = REPORT / "interpretability"
OOS_START = pd.Timestamp("2011-01-31")
OOS_END = MAX_FORMATION_MONTH
COST_BPS = (10.0, 25.0)


def _universe_turnover(panel: pd.DataFrame) -> pd.Series:
    from factor_model.factors.classic import return_month

    turns = []
    prev: set[int] | None = None
    for dt, g in panel.groupby("date", sort=True):
        ids = set(g["security_id"].astype(int))
        if prev is None:
            t = 1.0
        else:
            t = 1.0 - len(ids & prev) / max(len(ids), 1)
        prev = ids
        turns.append({"return_month": return_month(pd.Timestamp(dt)), "turnover": t})
    s = pd.DataFrame(turns).set_index("return_month")["turnover"].sort_index()
    s.index = s.index.astype("datetime64[ns]")
    return s


def _port_from_preds(preds: pd.DataFrame):
    panel = preds.rename(columns={"formation_month": "date", "realized_ret": "ret_1m"})
    rets, turns = long_only_top_decile(panel, panel["prediction"], long_high=True)
    ic = rank_ic(panel, panel["prediction"])
    ls = decile_long_short(panel, "prediction", long_high=True, value_weighted=False)
    return rets, turns, ic, float(ls.mean() * 12)


def _run_model(eng: WalkForwardEngine, model, panel: pd.DataFrame) -> tuple:
    t0 = time.perf_counter()
    res = eng.run(model, panel)
    wall = time.perf_counter() - t0
    path = PRED / f"{model.name}_predictions.parquet"
    res.to_parquet(path)
    res.fits_to_json(FITS / f"{model.name}_year_fits.json")
    return res, wall


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    t_all = time.perf_counter()
    for p in (PRED, FITS, REPORT, INTERP):
        p.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    seed = int(cfg["compute"]["random_seed"])

    print("loading panel + FF…")
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    ff = load_ff_factors()
    eng = WalkForwardEngine(n_jobs=cfg["compute"]["n_jobs"], random_seed=seed)

    walls: dict[str, float] = {}
    frames: dict[str, pd.DataFrame] = {}

    # Reuse existing linear predictions if present and holdout-clean
    for name in ("lasso", "ridge"):
        path = PRED / f"{name}_predictions.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["formation_month"] = pd.to_datetime(df["formation_month"])
            assert df["formation_month"].max() <= MAX_FORMATION_MONTH
            frames[name] = df
            walls[name] = 0.0
            print(f"reused {name} predictions")
        else:
            Model = LassoModel if name == "lasso" else RidgeModel
            res, w = _run_model(eng, Model(), panel)
            frames[name] = res.predictions
            walls[name] = w
            print(f"{name} wall={w:.1f}s")

    for Model in (RandomForestModel, XGBoostModel, MLPModel):
        print(f"walk-forward {Model.name}…")
        res, w = _run_model(eng, Model(), panel)
        frames[Model.name] = res.predictions
        walls[Model.name] = w
        print(f"  {Model.name} wall={w:.1f}s rows={len(res.predictions):,}")

    # Ensemble: rank-average of Lasso, XGBoost, MLP
    print("ensemble…")
    ens = rank_average_predictions(
        frames, members=["lasso", "xgboost", "mlp"]
    )
    ens.to_parquet(PRED / "ensemble_predictions.parquet", compression="zstd")
    frames["ensemble"] = ens
    walls["ensemble"] = 0.0

    # --- Baselines on OOS window ---
    oos = panel[(panel["date"] >= OOS_START) & (panel["date"] <= OOS_END)].copy()
    oos = oos[oos["ret_1m"].notna()]
    bench = equal_weight_universe(oos)
    bench_turn = _universe_turnover(oos)
    comp = classic4_composite_score(oos)
    c_rets, c_turns = long_only_top_decile(oos, comp, long_high=True)
    c_ic = rank_ic(oos, comp)
    c_ls = decile_long_short(oos.assign(_c=comp.to_numpy()), "_c", long_high=True, value_weighted=False)

    series = {"ew_universe": bench, "classic4": c_rets}
    turns_map = {"ew_universe": bench_turn, "classic4": c_turns}
    diag = {
        "classic4": {
            "rank_ic_mean": float(c_ic.mean()),
            "rank_ic_pos_share": float((c_ic > 0).mean()),
            "top_minus_bottom_ew_ann": float(c_ls.mean() * 12),
        },
        "ew_universe": {
            "rank_ic_mean": float("nan"),
            "rank_ic_pos_share": float("nan"),
            "top_minus_bottom_ew_ann": float("nan"),
        },
    }

    for name, preds in frames.items():
        rets, turns, ic, ls_ann = _port_from_preds(preds)
        series[name] = rets
        turns_map[name] = turns
        diag[name] = {
            "rank_ic_mean": float(ic.mean()),
            "rank_ic_pos_share": float((ic > 0).mean()),
            "top_minus_bottom_ew_ann": ls_ann,
        }

    ret_df = pd.DataFrame(series).sort_index()
    ret_df = ret_df.loc[
        (ret_df.index >= pd.Timestamp("2011-02-28"))
        & (ret_df.index <= pd.Timestamp("2023-12-31"))
    ]
    turn_df = pd.DataFrame({k: v.reindex(ret_df.index) for k, v in turns_map.items()})
    ret_df.to_parquet(PRED / "all_portfolio_returns.parquet", compression="zstd")
    turn_df.to_parquet(PRED / "all_portfolio_turnover.parquet", compression="zstd")

    rf = ff["RF"].reindex(ret_df.index).fillna(0.0)

    # --- Grand metrics ---
    rows = []
    for name in ret_df.columns:
        gross = ret_df[name]
        turn = turn_df[name]
        base = summarize(
            gross,
            benchmark=ret_df["ew_universe"] if name != "ew_universe" else None,
            turnover=turn,
            rf=rf,
        )
        base["name"] = name
        base.update(diag.get(name, {}))
        base["wall_s"] = walls.get(name, float("nan"))

        for bps in COST_BPS:
            net = apply_costs(gross, turn, bps)
            ns = summarize(net, benchmark=apply_costs(ret_df["ew_universe"], turn_df["ew_universe"], bps) if name != "ew_universe" else None, rf=rf)
            base[f"net_sharpe_{int(bps)}bps"] = ns["sharpe"]
            base[f"net_ann_return_{int(bps)}bps"] = ns["ann_return"]
            if name != "ew_universe":
                base[f"net_excess_{int(bps)}bps"] = ns.get("excess_vs_benchmark", float("nan"))

        # FF6 alpha gross and net 10bps
        port_ex = gross - rf
        a_g = ff6_alpha(port_ex, ff)
        base["ff6_alpha_ann"] = a_g["alpha_ann"]
        base["ff6_alpha_t"] = a_g["alpha_t"]
        base["ff6_beta_Mkt_RF"] = a_g.get("beta_Mkt_RF", float("nan"))
        for col in ("SMB", "HML", "RMW", "CMA", "UMD"):
            base[f"ff6_beta_{col}"] = a_g.get(f"beta_{col}", float("nan"))
        net10 = apply_costs(gross, turn, 10.0) - rf
        a_n = ff6_alpha(net10, ff)
        base["ff6_alpha_ann_net10"] = a_n["alpha_ann"]
        base["ff6_alpha_t_net10"] = a_n["alpha_t"]

        # Active vs EW
        if name != "ew_universe":
            active = gross - ret_df["ew_universe"]
            a_act = ff6_alpha(active, ff)  # active already excess-like vs EW; still regress on FF
            base["ff6_active_alpha_ann"] = a_act["alpha_ann"]
            base["ff6_active_alpha_t"] = a_act["alpha_t"]
            base["breakeven_bps_vs_classic4"] = breakeven_bps_vs_benchmark(
                gross, turn, ret_df["classic4"], turn_df["classic4"]
            )

            for label, other in (("classic4", ret_df["classic4"]), ("lasso", ret_df["lasso"])):
                sd = sharpe_diff_ci(gross, other, rf=0.0)
                base[f"sharpe_minus_{label}"] = sd["point"]
                base[f"sharpe_minus_{label}_lo"] = sd["lo"]
                base[f"sharpe_minus_{label}_hi"] = sd["hi"]
                net_m = apply_costs(gross, turn, 10.0)
                net_o = apply_costs(other, turn_df[label], 10.0)
                sdn = sharpe_diff_ci(net_m, net_o, rf=0.0)
                base[f"net10_sharpe_minus_{label}"] = sdn["point"]
                base[f"net10_sharpe_minus_{label}_lo"] = sdn["lo"]
                base[f"net10_sharpe_minus_{label}_hi"] = sdn["hi"]

        rows.append(base)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT / "grand_metrics.csv", index=False)

    # Excess by year / regimes / chart
    year_rows = []
    for name in ret_df.columns:
        if name == "ew_universe":
            continue
        active = ret_df[name] - ret_df["ew_universe"]
        for y, g in active.groupby(active.index.year):
            year_rows.append({"year": int(y), "name": name, "ann_excess": float(g.mean() * 12)})
    pd.DataFrame(year_rows).to_csv(REPORT / "excess_by_year.csv", index=False)

    reg_rows = []
    for reg, (a, b) in {"covid": ("2020-02-29", "2020-12-31"), "ratehike": ("2022-01-31", "2022-12-31")}.items():
        for name in ret_df.columns:
            sub = ret_df[name].loc[a:b]
            net = apply_costs(sub, turn_df[name].loc[a:b], 10.0)
            s = summarize(net, benchmark=apply_costs(ret_df["ew_universe"].loc[a:b], turn_df["ew_universe"].loc[a:b], 10.0) if name != "ew_universe" else None, rf=0.0)
            s.update({"name": name, "regime": reg})
            reg_rows.append(s)
    pd.DataFrame(reg_rows).to_csv(REPORT / "regimes_net10.csv", index=False)

    # Cumulative net 10bps
    wealth = {}
    for name in ret_df.columns:
        wealth[name] = (1.0 + apply_costs(ret_df[name], turn_df[name], 10.0).fillna(0)).cumprod()
    wdf = pd.DataFrame(wealth)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "ew_universe": "#999999",
        "classic4": "#222222",
        "lasso": "#b85c38",
        "ridge": "#8b4513",
        "random_forest": "#2f5d8a",
        "xgboost": "#1b7f4e",
        "mlp": "#6b3fa0",
        "ensemble": "#c41e3a",
    }
    for col in wdf.columns:
        ax.plot(wdf.index, wdf[col], label=col, color=colors.get(col, None), lw=1.2)
    ax.set_title("OOS cumulative return, net 10 bps/side")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT / "cumulative_net10.png", dpi=120)
    plt.close(fig)

    # --- Interpretability ---
    print("interpretability…")
    _run_interpretability(panel, frames, eng, INTERP)

    summary = {
        "wall_s": round(time.perf_counter() - t_all, 2),
        "per_model_wall_s": walls,
        "metrics_file": str(REPORT / "grand_metrics.csv"),
        "oos": f"{OOS_START.date()} → {OOS_END.date()}",
    }
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    cols = [
        "name", "ann_return", "sharpe", "max_drawdown", "turnover",
        "excess_vs_benchmark", "net_sharpe_10bps", "net_sharpe_25bps",
        "ff6_alpha_ann", "ff6_alpha_t", "ff6_beta_Mkt_RF", "rank_ic_mean",
    ]
    print(metrics[[c for c in cols if c in metrics.columns]].to_string(index=False))
    print("done wall=", summary["wall_s"])


def _run_interpretability(panel, frames, eng, out_dir: Path) -> None:
    """SHAP for trees, permutation IC for MLP, coef ranks for linear; agreement matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = list(FEATURE_NAMES)

    # Linear |coef| from year fits if available
    imp = {}
    for name in ("lasso", "ridge"):
        path = FITS / f"{name}_year_fits.json"
        if not path.exists():
            continue
        fits = json.loads(path.read_text())
        mat = []
        for f in fits:
            coefs = f.get("coefficients") or {}
            mat.append({c: abs(coefs.get(c, 0.0)) for c in feats})
        imp[name] = pd.DataFrame(mat).mean()

    # Fit one RF and one XGB on pre-2011 for SHAP baseline sample from OOS preds
    # Use stored year fits' last year model by re-fitting quickly on train through 2022
    oos_panel = eng.prepare_panel(panel)
    oos_panel = oos_panel[(oos_panel["date"] >= OOS_START) & (oos_panel["date"] <= OOS_END)]

    # RF / XGB SHAP: refit on 2006–2022, explain sample of 2023 OOS rows
    train = oos_panel[oos_panel["date"] < pd.Timestamp("2023-01-01")]
    # Actually train should be through Dec 2022 from full panel
    full = eng.prepare_panel(panel)
    train = full[full["date"] <= pd.Timestamp("2022-12-31")]
    test = full[(full["date"] >= pd.Timestamp("2023-01-31")) & (full["date"] <= OOS_END)]
    X_tr = train[feats].to_numpy(dtype=float)
    y_tr = train["y"].to_numpy(dtype=float)
    # sample up to 15k OOS rows across years for SHAP
    rng = np.random.default_rng(42)
    sample_n = min(15_000, len(test))
    samp_idx = rng.choice(len(test), size=sample_n, replace=False)
    test_s = test.iloc[samp_idx]
    X_te = test_s[feats].to_numpy(dtype=float)

    rf = RandomForestModel(max_depth=5, min_samples_leaf=500, max_features=0.33, n_jobs=-1)
    rf.fit(X_tr, y_tr, feature_names=feats)
    explainer = shap.TreeExplainer(rf._model)
    sv = explainer.shap_values(X_te)
    rf_imp = pd.Series(np.abs(sv).mean(axis=0), index=feats, name="random_forest")
    imp["random_forest"] = rf_imp
    shap.summary_plot(sv, X_te, feature_names=feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_rf.png", dpi=120, bbox_inches="tight")
    plt.close()

    xgb = XGBoostModel(max_depth=3, learning_rate=0.05, min_child_weight=10, n_estimators=200, n_jobs=-1)
    # no early stopping for interpretability refit
    xgb.params["best_n_estimators"] = 200
    xgb.fit(X_tr, y_tr, feature_names=feats)
    explainer = shap.TreeExplainer(xgb._model)
    sv = explainer.shap_values(X_te)
    xgb_imp = pd.Series(np.abs(sv).mean(axis=0), index=feats, name="xgboost")
    imp["xgboost"] = xgb_imp
    shap.summary_plot(sv, X_te, feature_names=feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_xgb.png", dpi=120, bbox_inches="tight")
    plt.close()

    # XGB interaction pairs (smaller sample)
    inter_n = min(3000, len(X_te))
    Xi = X_te[:inter_n]
    try:
        inter = xgb._model.predict(Xi, pred_interactions=True)
        # inter shape (n, F+1, F+1) including bias — take feature block
        mean_abs = np.abs(inter[:, :-1, :-1]).mean(axis=0)
        pairs = []
        f = len(feats)
        for i in range(f):
            for j in range(i + 1, f):
                pairs.append((feats[i], feats[j], float(mean_abs[i, j])))
        pairs = sorted(pairs, key=lambda x: -x[2])[:5]
        pd.DataFrame(pairs, columns=["feature_a", "feature_b", "mean_abs_interaction"]).to_csv(
            out_dir / "xgb_top_interactions.csv", index=False
        )
    except Exception as e:
        (out_dir / "xgb_top_interactions.csv").write_text(f"error,{e}\n")

    # MLP permutation importance on 2023 OOS (rank IC drop)
    mlp = MLPModel(hidden=(32, 16), dropout=0.1, weight_decay=1e-3, n_seeds=3, max_epochs=30)
    val_months = sorted(train["date"].unique())[-36:]
    tr_m = train[~train["date"].isin(val_months)]
    va_m = train[train["date"].isin(val_months)]
    mlp.fit(
        tr_m[feats].to_numpy(dtype=float),
        tr_m["y"].to_numpy(dtype=float),
        feature_names=feats,
        X_val=va_m[feats].to_numpy(dtype=float),
        y_val=va_m["y"].to_numpy(dtype=float),
    )
    base_pred = mlp.predict(X_te)
    base_ic = monthly_rank_ic(
        pd.Series(base_pred),
        pd.Series(test_s["ret_1m"].to_numpy()),
        pd.Series(test_s["date"].to_numpy()),
    )
    drops = {}
    rng = np.random.default_rng(0)
    for j, f in enumerate(feats):
        ics = []
        for _ in range(5):
            Xp = X_te.copy()
            # shuffle within month
            tmp = test_s.copy()
            tmp["_x"] = Xp[:, j]
            parts = []
            for _, g in tmp.groupby("date", sort=False):
                g = g.copy()
                g["_x"] = rng.permutation(g["_x"].to_numpy())
                parts.append(g)
            shuf = pd.concat(parts)
            Xp[:, j] = shuf["_x"].to_numpy()
            pred = mlp.predict(Xp)
            ics.append(
                monthly_rank_ic(
                    pd.Series(pred),
                    pd.Series(test_s["ret_1m"].to_numpy()),
                    pd.Series(test_s["date"].to_numpy()),
                )
            )
        drops[f] = float(base_ic - np.nanmean(ics))
    imp["mlp"] = pd.Series(drops, name="mlp")

    # Importance table (rank-normalized)
    imp_df = pd.DataFrame(imp)
    rank_df = imp_df.rank(ascending=False)
    rank_df.to_csv(out_dir / "importance_ranks.csv")
    imp_df.to_csv(out_dir / "importance_raw.csv")
    corr = rank_df.corr(method="spearman")
    corr.to_csv(out_dir / "importance_rank_corr.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    # normalize columns to [0,1] for display
    norm = imp_df / imp_df.max(axis=0).replace(0, np.nan)
    im = ax.imshow(norm.fillna(0).to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(norm.index)))
    ax.set_yticklabels(norm.index, fontsize=7)
    ax.set_xticks(range(len(norm.columns)))
    ax.set_xticklabels(norm.columns, rotation=45, ha="right")
    ax.set_title("Feature importance (col-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_dir / "importance_heatmap.png", dpi=120)
    plt.close(fig)

    # Beta / vol lean
    lean = {}
    for model_name, s in imp.items():
        total = float(s.sum()) + 1e-12
        lean[model_name] = {
            "beta_60m_share": float(s.get("beta_60m", 0.0) / total),
            "volatility_12m_share": float(s.get("volatility_12m", 0.0) / total),
            "max_daily_ret_1m_share": float(s.get("max_daily_ret_1m", 0.0) / total),
        }
    pd.DataFrame(lean).T.to_csv(out_dir / "beta_vol_lean.csv")


if __name__ == "__main__":
    main()
