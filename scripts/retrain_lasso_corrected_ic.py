"""Verify other models' tuning, retrain Lasso under corrected IC, regenerate reports.

No formation after 2023-11. Ensemble skips constant members per month; portfolios
hold EW when predictions are constant.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_model.evaluation.bootstrap import sharpe_diff_ci
from factor_model.evaluation.costs import apply_costs, breakeven_bps_vs_benchmark
from factor_model.evaluation.ff_alpha import FF6_COLS, ff6_alpha
from factor_model.evaluation.metrics import summarize
from factor_model.external.ff import load_ff_factors
from factor_model.factors.classic import (
    MAX_FORMATION_MONTH,
    classic4_composite_score,
    decile_long_short,
    equal_weight_universe,
    long_only_top_decile,
    rank_ic,
    return_month,
)
from factor_model.models import (
    LassoModel,
    MLPModel,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
)
from factor_model.models.ensemble import rank_average_predictions
from factor_model.models.linear import LASSO_ALPHAS, RIDGE_ALPHAS
from factor_model.walkforward.engine import WalkForwardEngine, _param_product, monthly_rank_ic_series

PANEL = ROOT / "data" / "panel" / "panel.parquet"
PRED = ROOT / "data" / "predictions"
FITS = PRED / "fits"
REPORT = ROOT / "reports" / "models"
OOS_START = pd.Timestamp("2011-01-31")
OOS_END = MAX_FORMATION_MONTH
COST_BPS = (10.0, 25.0)


def _is_constant(pred: np.ndarray) -> bool:
    p = pred[np.isfinite(pred)]
    if len(p) < 2:
        return True
    return int(pd.Series(p).nunique(dropna=True)) < 2 or float(np.std(p)) == 0.0


def _val_constant_months(pred: np.ndarray, dates: np.ndarray) -> int:
    n = 0
    for d in pd.unique(dates):
        m = dates == d
        if m.sum() < 20:
            continue
        if _is_constant(pred[m]):
            n += 1
    return n


def _make_model(name: str):
    if name == "ridge":
        return RidgeModel()
    if name == "random_forest":
        return RandomForestModel(n_jobs=2)
    if name == "xgboost":
        return XGBoostModel(n_jobs=2)
    if name == "mlp":
        # 1 seed is enough to detect flat outputs; production still uses 5
        return MLPModel(n_seeds=1, max_epochs=40)
    raise KeyError(name)


def verify_model_tuning(eng: WalkForwardEngine, panel: pd.DataFrame, name: str) -> dict:
    """Scan every year × grid point for constant validation predictions."""
    print(f"Step 1 verify {name}…", flush=True)
    df = eng.prepare_panel(panel)
    calendar = eng._formation_calendar(df)
    windows = eng._year_windows(calendar)
    model0 = _make_model(name)
    grid = _param_product(model0.param_grid)
    rows = []
    any_const = False
    for year, train_end, first_pred, _preds in windows:
        train = df[(df["date"] >= eng.sample_start) & (df["date"] <= train_end)]
        months = pd.DatetimeIndex(sorted(train["date"].unique()))
        val_months = months[-eng.validation_months :]
        fit_months = months[: -eng.validation_months]
        tr = train[train["date"].isin(fit_months)]
        va = train[train["date"].isin(val_months)]
        X_tr = tr[eng.feature_names].to_numpy(dtype=float)
        y_tr = tr["y"].to_numpy(dtype=float)
        X_va = va[eng.feature_names].to_numpy(dtype=float)
        dates_va = va["date"].to_numpy()
        ret_va = va["ret_1m"].to_numpy(dtype=float)

        def _one(params: dict) -> tuple[dict, int, float, float]:
            m = model0.clone(**params)
            if getattr(m, "needs_validation", False) and name == "xgboost":
                m.fit(X_tr, y_tr, feature_names=eng.feature_names, X_val=X_va, y_val=va["y"].to_numpy())
            elif getattr(m, "needs_validation", False):
                m.fit(X_tr, y_tr, feature_names=eng.feature_names, X_val=X_va, y_val=va["y"].to_numpy())
            else:
                m.fit(X_tr, y_tr, feature_names=eng.feature_names)
            pred = m.predict(X_va)
            n_c = _val_constant_months(pred, dates_va)
            # new vs old IC for this grid point
            s = monthly_rank_ic_series(pd.Series(pred), pd.Series(ret_va), pd.Series(dates_va))
            ic_new = float(s.mean()) if len(s) else float("nan")
            # old: drop constants
            ics = []
            for d in pd.unique(dates_va):
                mask = dates_va == d
                if mask.sum() < 20:
                    continue
                if _is_constant(pred[mask]):
                    continue
                sub_p, sub_r = pred[mask], ret_va[mask]
                if len(np.unique(sub_r[np.isfinite(sub_r)])) < 2:
                    continue
                ics.append(float(pd.Series(sub_p).corr(pd.Series(sub_r), method="spearman")))
            ic_old = float(np.nanmean(ics)) if ics else float("nan")
            return params, n_c, ic_new, ic_old

        outer = 4 if name in {"random_forest", "xgboost", "mlp"} else -1
        results = Parallel(n_jobs=outer)(delayed(_one)(p) for p in grid)

        def _pick(items, use_new: bool):
            def key(it):
                params, n_c, ic_new, ic_old = it
                ic = ic_new if use_new else ic_old
                ic_v = -np.inf if (ic is None or (isinstance(ic, float) and np.isnan(ic))) else ic
                alpha = params.get("alpha", 0.0)
                # prefer higher alpha / deeper etc. as secondary — match engine for linear
                return (ic_v, float(alpha) if "alpha" in params else 0.0)

            return max(items, key=key)

        best_old = _pick(results, use_new=False)
        best_new = _pick(results, use_new=True)
        year_const = sum(r[1] for r in results)
        if year_const > 0:
            any_const = True
        changed = best_old[0] != best_new[0]
        rows.append(
            {
                "model": name,
                "year": year,
                "n_grid": len(grid),
                "n_constant_val_months_sum_over_grid": int(year_const),
                "any_grid_point_had_constant_month": year_const > 0,
                "params_old": json.dumps(best_old[0], default=str),
                "params_new": json.dumps(best_new[0], default=str),
                "params_changed": changed,
            }
        )
        print(
            f"  {name} {year}: const_sum={year_const} changed={changed}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    return {
        "model": name,
        "any_constant_val": any_const,
        "any_params_changed": bool(out["params_changed"].any()),
        "table": out,
    }


def retrain_lasso(eng: WalkForwardEngine, panel: pd.DataFrame) -> tuple:
    print("Step 2: retrain Lasso…", flush=True)
    # backup old
    old_pred = PRED / "lasso_predictions.parquet"
    old_fits = FITS / "lasso_year_fits.json"
    if old_pred.exists():
        shutil.copy2(old_pred, PRED / "lasso_predictions_before_icfix.parquet")
    if old_fits.exists():
        shutil.copy2(old_fits, FITS / "lasso_year_fits_before_icfix.json")
        old = json.loads(old_fits.read_text())
    else:
        old = []

    t0 = time.perf_counter()
    res = eng.run(LassoModel(), panel)
    wall = time.perf_counter() - t0
    assert res.predictions["formation_month"].max() <= MAX_FORMATION_MONTH
    res.to_parquet(PRED / "lasso_predictions.parquet")
    res.fits_to_json(FITS / "lasso_year_fits.json")

    new = [{**r.__dict__} for r in res.year_fits]
    # YearFitRecord has coefficients etc.
    new_fits = json.loads((FITS / "lasso_year_fits.json").read_text())
    old_by_y = {r["year"]: r for r in old}
    cmp_rows = []
    for r in new_fits:
        y = r["year"]
        o = old_by_y.get(y, {})
        oa = o.get("chosen_params", {}).get("alpha")
        na = r["chosen_params"].get("alpha")
        coef_match = True
        if y != 2011 and o.get("coefficients") and r.get("coefficients"):
            for k, v in r["coefficients"].items():
                if abs(float(v) - float(o["coefficients"].get(k, 0))) > 1e-8:
                    coef_match = False
                    break
        cmp_rows.append(
            {
                "year": y,
                "alpha_old": oa,
                "alpha_new": na,
                "alpha_match": oa == na if y != 2011 else (na == 10.0),
                "coef_match_expected": coef_match if y != 2011 else True,
                "validation_ic_new": r.get("validation_ic"),
            }
        )
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(REPORT / "lasso_retrain_year_compare.csv", index=False)
    ok_2011 = abs(float(new_fits[0]["chosen_params"]["alpha"]) - 10.0) < 1e-12
    later_ok = all(
        abs(float(r["alpha_old"]) - float(r["alpha_new"])) < 1e-15
        for _, r in cmp[cmp["year"] > 2011].iterrows()
    )
    # also coef match for later years
    later_coef = all(bool(r["coef_match_expected"]) for _, r in cmp[cmp["year"] > 2011].iterrows())
    print(cmp.to_string(index=False), flush=True)
    print(f"lasso wall={wall:.1f}s 2011_alpha10={ok_2011} later_alpha_match={later_ok} later_coef={later_coef}", flush=True)

    # constant OOS months
    preds = res.predictions
    n_const = 0
    for _, g in preds.groupby("formation_month"):
        if _is_constant(g["prediction"].to_numpy()):
            n_const += 1
    print(f"lasso OOS constant months={n_const}", flush=True)
    return res, {
        "wall_s": wall,
        "ok_2011": ok_2011,
        "later_alpha_match": later_ok,
        "later_coef_match": later_coef,
        "n_oos_constant_months": n_const,
    }


def _universe_turnover(panel: pd.DataFrame) -> pd.Series:
    turns = []
    prev: set[int] | None = None
    for dt, g in panel.groupby("date", sort=True):
        ids = set(g["security_id"].astype(int))
        t = 1.0 if prev is None else 1.0 - len(ids & prev) / max(len(ids), 1)
        prev = ids
        turns.append({"return_month": return_month(pd.Timestamp(dt)), "turnover": t})
    s = pd.DataFrame(turns).set_index("return_month")["turnover"].sort_index()
    s.index = s.index.astype("datetime64[ns]")
    return s


def _port_from_preds(preds: pd.DataFrame):
    panel = preds.rename(columns={"formation_month": "date", "realized_ret": "ret_1m"})
    rets, turns = long_only_top_decile(
        panel, panel["prediction"], long_high=True, ew_when_constant=True
    )
    ic = rank_ic(panel, panel["prediction"])
    ls = decile_long_short(panel, "prediction", long_high=True, value_weighted=False)
    return rets, turns, ic, float(ls.mean() * 12)


def regenerate_reports(panel: pd.DataFrame) -> pd.DataFrame:
    print("Step 3: regenerate portfolios / metrics…", flush=True)
    ff = load_ff_factors()
    before_gm = pd.read_csv(REPORT / "grand_metrics.csv")
    before_aa = pd.read_csv(REPORT / "active_alpha.csv")

    frames = {}
    for name in ("lasso", "ridge", "random_forest", "xgboost", "mlp"):
        df = pd.read_parquet(PRED / f"{name}_predictions.parquet")
        df["formation_month"] = pd.to_datetime(df["formation_month"])
        assert df["formation_month"].max() <= MAX_FORMATION_MONTH
        frames[name] = df

    ens, ens_log = rank_average_predictions(frames, members=["lasso", "xgboost", "mlp"])
    ens.to_parquet(PRED / "ensemble_predictions.parquet", compression="zstd")
    ens_log.to_csv(REPORT / "ensemble_member_months.csv", index=False)
    frames["ensemble"] = ens
    n_drop_lasso = int((ens_log["dropped"].fillna("").str.contains("lasso")).sum())
    print(f"ensemble months with lasso dropped: {n_drop_lasso}", flush=True)

    oos = panel[(panel["date"] >= OOS_START) & (panel["date"] <= OOS_END)].copy()
    oos = oos[oos["ret_1m"].notna()]
    bench = equal_weight_universe(oos)
    bench_turn = _universe_turnover(oos)
    comp = classic4_composite_score(oos)
    c_rets, c_turns = long_only_top_decile(oos, comp, long_high=True)
    c_ic = rank_ic(oos, comp)

    series = {"ew_universe": bench, "classic4": c_rets}
    turns_map = {"ew_universe": bench_turn, "classic4": c_turns}
    diag = {
        "classic4": {
            "rank_ic_mean": float(c_ic.mean()),
            "rank_ic_pos_share": float((c_ic > 0).mean()),
            "top_minus_bottom_ew_ann": float(
                decile_long_short(oos.assign(_c=comp.to_numpy()), "_c", long_high=True, value_weighted=False).mean()
                * 12
            ),
        },
        "ew_universe": {"rank_ic_mean": float("nan"), "rank_ic_pos_share": float("nan"), "top_minus_bottom_ew_ann": float("nan")},
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
        (ret_df.index >= pd.Timestamp("2011-02-28")) & (ret_df.index <= pd.Timestamp("2023-12-31"))
    ]
    turn_df = pd.DataFrame({k: v.reindex(ret_df.index) for k, v in turns_map.items()})
    ret_df.to_parquet(PRED / "all_portfolio_returns.parquet", compression="zstd")
    turn_df.to_parquet(PRED / "all_portfolio_turnover.parquet", compression="zstd")
    rf = ff["RF"].reindex(ret_df.index).fillna(0.0)

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
        for bps in COST_BPS:
            net = apply_costs(gross, turn, bps)
            ns = summarize(
                net,
                benchmark=apply_costs(ret_df["ew_universe"], turn_df["ew_universe"], bps)
                if name != "ew_universe"
                else None,
                rf=rf,
            )
            base[f"net_sharpe_{int(bps)}bps"] = ns["sharpe"]
            base[f"net_ann_return_{int(bps)}bps"] = ns["ann_return"]
            if name != "ew_universe":
                base[f"net_excess_{int(bps)}bps"] = ns.get("excess_vs_benchmark", float("nan"))
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
        if name != "ew_universe":
            active = gross - ret_df["ew_universe"]
            a_act = ff6_alpha(active, ff)
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
                sdn = sharpe_diff_ci(
                    apply_costs(gross, turn, 10.0),
                    apply_costs(other, turn_df[label], 10.0),
                    rf=0.0,
                )
                base[f"net10_sharpe_minus_{label}"] = sdn["point"]
                base[f"net10_sharpe_minus_{label}_lo"] = sdn["lo"]
                base[f"net10_sharpe_minus_{label}_hi"] = sdn["hi"]
        rows.append(base)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT / "grand_metrics.csv", index=False)

    # active alpha table
    aa_rows = []
    for name in ["classic4", "lasso", "ridge", "random_forest", "xgboost", "mlp", "ensemble"]:
        for label, series in (
            ("gross", ret_df[name] - ret_df["ew_universe"]),
            (
                "net10",
                apply_costs(ret_df[name], turn_df[name], 10.0)
                - apply_costs(ret_df["ew_universe"], turn_df["ew_universe"], 10.0),
            ),
        ):
            a = ff6_alpha(series, ff)
            row = {
                "name": name,
                "cost": label,
                "alpha_ann": a["alpha_ann"],
                "alpha_t": a["alpha_t"],
                "n": a.get("n"),
                "r2": a.get("r2"),
            }
            for col in FF6_COLS:
                row[f"beta_{col}"] = a.get(f"beta_{col}")
                row[f"t_{col}"] = a.get(f"t_{col}")
            aa_rows.append(row)
    pd.DataFrame(aa_rows).to_csv(REPORT / "active_alpha.csv", index=False)

    # excess by year / regimes / chart (reuse nonlinear script patterns lightly)
    year_rows = []
    for name in ret_df.columns:
        if name == "ew_universe":
            continue
        active = ret_df[name] - ret_df["ew_universe"]
        for y, g in active.groupby(active.index.year):
            year_rows.append({"year": int(y), "name": name, "ann_excess": float(g.mean() * 12)})
    pd.DataFrame(year_rows).to_csv(REPORT / "excess_by_year.csv", index=False)

    # before/after for lasso + ensemble
    old_lasso = pd.read_parquet(PRED / "lasso_predictions_before_icfix.parquet")
    old_lasso["formation_month"] = pd.to_datetime(old_lasso["formation_month"])
    old_frames = dict(frames)
    old_frames["lasso"] = old_lasso
    old_ens, _ = rank_average_predictions(old_frames, members=["lasso", "xgboost", "mlp"])

    def _excess_2011(preds: pd.DataFrame) -> float:
        rets, _, _, _ = _port_from_preds(preds)
        active = rets - ret_df["ew_universe"].reindex(rets.index)
        m = active.index.year == 2011
        return float(active[m].mean() * 12) if m.any() else float("nan")

    ba = []
    for name, old_preds, new_preds in (
        ("lasso", old_lasso, frames["lasso"]),
        ("ensemble", old_ens, frames["ensemble"]),
    ):
        b = before_gm[before_gm["name"] == name].iloc[0]
        a = metrics[metrics["name"] == name].iloc[0]
        ba.append(
            {
                "name": name,
                "sharpe_before": float(b["sharpe"]),
                "sharpe_after": float(a["sharpe"]),
                "net10_before": float(b["net_sharpe_10bps"]),
                "net10_after": float(a["net_sharpe_10bps"]),
                "excess_before": float(b.get("excess_vs_benchmark", np.nan)),
                "excess_after": float(a.get("excess_vs_benchmark", np.nan)),
                "active_alpha_before": float(b.get("ff6_active_alpha_ann", np.nan)),
                "active_alpha_after": float(a.get("ff6_active_alpha_ann", np.nan)),
                "active_t_before": float(b.get("ff6_active_alpha_t", np.nan)),
                "active_t_after": float(a.get("ff6_active_alpha_t", np.nan)),
                "excess_2011_before": _excess_2011(old_preds),
                "excess_2011_after": _excess_2011(new_preds),
            }
        )
    pd.DataFrame(ba).to_csv(REPORT / "lasso_ensemble_before_after.csv", index=False)

    # cumulative net 10bps chart
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ret_df.columns:
        net = apply_costs(ret_df[name], turn_df[name], 10.0)
        ax.plot(net.index, (1 + net).cumprod(), label=name, lw=1.2)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Cumulative wealth, net 10 bps")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT / "cumulative_net10.png", dpi=120)
    plt.close()

    # update interpretability lasso coefficients (mean |coef| from new fits)
    fits = json.loads((FITS / "lasso_year_fits.json").read_text())
    coefs = []
    for r in fits:
        if r.get("coefficients"):
            coefs.append(pd.Series(r["coefficients"]))
    if coefs:
        mean_abs = pd.concat(coefs, axis=1).abs().mean(axis=1)
        mean_abs.to_csv(REPORT / "interpretability" / "lasso_mean_abs_coef.csv", header=["mean_abs_coef"])

    return metrics


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "interpretability").mkdir(parents=True, exist_ok=True)
    print("loading panel…", flush=True)
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    eng = WalkForwardEngine(n_jobs=-1, random_seed=42)

    # Step 1
    verify_tables = []
    verify_meta = {}
    for name in ("ridge", "xgboost", "random_forest", "mlp"):
        meta = verify_model_tuning(eng, panel, name)
        verify_tables.append(meta["table"])
        verify_meta[name] = {
            "any_constant_val": meta["any_constant_val"],
            "any_params_changed": meta["any_params_changed"],
        }
        if meta["any_params_changed"]:
            print(f"UNEXPECTED param change for {name} — will need retrain", flush=True)
    vdf = pd.concat(verify_tables, ignore_index=True)
    vdf.to_csv(REPORT / "tuning_constant_val_verify.csv", index=False)
    (REPORT / "tuning_constant_val_verify.json").write_text(json.dumps(verify_meta, indent=2), encoding="utf-8")
    unexpected = [k for k, v in verify_meta.items() if v["any_params_changed"]]
    print("verify_meta", verify_meta, flush=True)

    # Step 2
    _res, lasso_meta = retrain_lasso(eng, panel)

    # Step 3
    metrics = regenerate_reports(panel)
    print(metrics[["name", "sharpe", "net_sharpe_10bps", "excess_vs_benchmark", "ff6_active_alpha_ann", "ff6_active_alpha_t"]].to_string(index=False))

    summary = {
        "verify": verify_meta,
        "unexpected_param_changes": unexpected,
        "lasso": lasso_meta,
        "merge_ok_inputs": {
            "no_unexpected": len(unexpected) == 0,
            "lasso_2011_ok": lasso_meta["ok_2011"],
            "lasso_later_ok": lasso_meta["later_alpha_match"] and lasso_meta["later_coef_match"],
        },
    }
    (REPORT / "retrain_lasso_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
