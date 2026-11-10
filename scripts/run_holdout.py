"""ONE-TIME holdout evaluation. Requires an explicit acknowledgement flag.

    python scripts/run_holdout.py --i-am-running-the-holdout

Development walk-forward / classic / evaluation paths remain capped at
formation 2023-11. This script alone may read formations ≥ 2023-12.
"""
from __future__ import annotations

import argparse
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
from factor_model.evaluation.costs import apply_costs
from factor_model.evaluation.ff_alpha import ff6_alpha
from factor_model.evaluation.metrics import summarize
from factor_model.external.ff import download_ff, load_ff_factors
from factor_model.factors.classic import (
    MAX_FORMATION_MONTH,
    classic4_composite_score,
    equal_weight_universe,
    long_only_top_decile,
    rank_ic,
    return_month,
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
RAW = ROOT / "data" / "panel" / "features_raw.parquet"
OOS_PRED = ROOT / "data" / "predictions"
OOS_FITS = OOS_PRED / "fits"
HOLDOUT_PRED = ROOT / "data" / "predictions" / "holdout"
HOLDOUT_FITS = HOLDOUT_PRED / "fits"
REPORT = ROOT / "reports" / "holdout"
OOS_METRICS = ROOT / "reports" / "models" / "grand_metrics.csv"
OOS_IMP_RANKS = ROOT / "reports" / "models" / "interpretability" / "importance_ranks.csv"

HOLDOUT_MAX_FORMATION = pd.Timestamp("2026-08-31")
HOLDOUT_FIRST_FORMATION = pd.Timestamp("2023-12-01")
OOS_START = pd.Timestamp("2011-01-31")
COST_BPS = (10.0, 25.0)
MODEL_ORDER = [
    "ew_universe",
    "classic4",
    "lasso",
    "ridge",
    "random_forest",
    "xgboost",
    "mlp",
    "ensemble",
]
ML_MODELS = {
    "lasso": LassoModel,
    "ridge": RidgeModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "mlp": MLPModel,
}


def _require_flag() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-am-running-the-holdout",
        action="store_true",
        required=True,
        help="Required acknowledgement that this is the one-time holdout run.",
    )
    parser.parse_args()


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
    return rets, turns, ic


def data_checks(panel: pd.DataFrame) -> dict:
    """Step 2. Raise if panel/returns/coverage are broken."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    hold = panel[panel["date"] >= HOLDOUT_FIRST_FORMATION]
    months = sorted(hold["date"].unique())
    if len(months) != 33:
        raise AssertionError(f"expected 33 holdout formations, got {len(months)}")
    if pd.Timestamp(months[0]).to_period("M") != pd.Period("2023-12", "M"):
        raise AssertionError(f"first holdout formation not 2023-12: {months[0]}")
    if pd.Timestamp(months[-1]) != HOLDOUT_MAX_FORMATION and pd.Timestamp(months[-1]).to_period(
        "M"
    ) != pd.Period("2026-08", "M"):
        raise AssertionError(f"last holdout formation not 2026-08: {months[-1]}")

    sizes = hold.groupby("date").size()
    if (sizes != 1000).any():
        bad = sizes[sizes != 1000]
        raise AssertionError(f"holdout months without 1000 names:\n{bad}")

    ret_null = hold.groupby("date")["ret_1m"].apply(lambda s: float(s.isna().mean()))
    if (ret_null > 0.05).any():
        raise AssertionError(f"ret_1m null share > 5% in holdout:\n{ret_null[ret_null > 0.05]}")

    # Feature coverage vs 2011–2023 average (raw, pre-fill)
    raw = pd.read_parquet(RAW, columns=["date", *FEATURE_NAMES])
    raw["date"] = pd.to_datetime(raw["date"])
    oos = raw[(raw["date"] >= OOS_START) & (raw["date"] <= MAX_FORMATION_MONTH)]
    hraw = raw[raw["date"] >= HOLDOUT_FIRST_FORMATION]
    cov_rows = []
    flags = []
    for f in FEATURE_NAMES:
        oos_nn = float(oos[f].notna().mean())
        hold_nn = float(hraw[f].notna().mean())
        drop = oos_nn - hold_nn
        row = {
            "feature": f,
            "oos_nonnull_share": oos_nn,
            "holdout_nonnull_share": hold_nn,
            "drop": drop,
        }
        cov_rows.append(row)
        # Flag: absolute drop ≥ 10pp, or relative drop ≥ 20% when OOS coverage > 0.5
        if drop >= 0.10 or (oos_nn > 0.5 and hold_nn < 0.8 * oos_nn):
            flags.append(f)

    cov = pd.DataFrame(cov_rows).sort_values("drop", ascending=False)
    cov.to_csv(REPORT / "feature_coverage.csv", index=False)
    if flags:
        raise AssertionError(f"feature coverage big drop(s): {flags}")

    # Fresh Ken French
    download_ff(force=True)
    ff = load_ff_factors()
    ff_end = ff.dropna(subset=["Mkt_RF", "SMB", "HML", "RMW", "CMA", "UMD"]).index.max()
    meta = {
        "n_holdout_formations": 33,
        "formation_start": str(pd.Timestamp(months[0]).date()),
        "formation_end": str(pd.Timestamp(months[-1]).date()),
        "names_per_month_min": int(sizes.min()),
        "names_per_month_max": int(sizes.max()),
        "ret_1m_null_share_max": float(ret_null.max()),
        "ff_last_month": str(ff_end.date()),
        "feature_coverage_flags": flags,
    }
    (REPORT / "data_checks.json").write_text(json.dumps(meta, indent=2))
    print("data checks OK:", json.dumps(meta, indent=2))
    return {"ff": ff, "ff_last_month": ff_end, "coverage": cov}


def _predict_month_with_fit(
    model_cls,
    df: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    pred_dates: pd.DatetimeIndex,
    params: dict,
    feature_names: list[str],
) -> tuple[pd.DataFrame, object]:
    """Fit on formations ≤ train_end; predict pred_dates. Returns (pred frame, fitted)."""
    train = df[df["date"] <= train_end]
    pred_slice = df[df["date"].isin(pred_dates)].copy()
    if pred_slice.empty:
        raise RuntimeError(f"no rows for pred_dates {pred_dates}")
    X_full = train[feature_names].to_numpy(dtype=float)
    y_full = train["y"].to_numpy(dtype=float)
    fitted = model_cls().clone(**params)
    # MLP / XGB early-stopping: use last 36 train months as val monitor (same as engine)
    if getattr(fitted, "needs_validation", False):
        train_months = pd.DatetimeIndex(sorted(train["date"].unique()))
        val_months = train_months[-36:]
        va = train[train["date"].isin(val_months)]
        X_va = va[feature_names].to_numpy(dtype=float)
        y_va = va["y"].to_numpy(dtype=float)
        if "best_n_estimators" in params:
            fitted.fit(X_full, y_full, feature_names=feature_names)
        else:
            fitted.fit(
                X_full,
                y_full,
                feature_names=feature_names,
                X_val=X_va,
                y_val=y_va,
            )
    else:
        fitted.fit(X_full, y_full, feature_names=feature_names)
    pred_slice["prediction"] = fitted.predict(pred_slice[feature_names].to_numpy(dtype=float))
    out = pred_slice[
        [
            "security_id",
            "date",
            "prediction",
            "ret_1m",
            "y",
            "return_month",
            "symbol",
            "mcap",
        ]
    ].rename(columns={"date": "formation_month", "ret_1m": "realized_ret"})
    return out, fitted


def run_models(panel: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Step 3: refit 2024–2026; attach 2023-12 from the OOS 2023 fit params."""
    seed = int(cfg["compute"]["random_seed"])
    eng = WalkForwardEngine(
        first_oos_year=2024,
        last_oos_year=2026,
        max_formation=HOLDOUT_MAX_FORMATION,
        n_jobs=cfg["compute"]["n_jobs"],
        random_seed=seed,
    )
    df = eng.prepare_panel(panel)
    # Dec 2023 formation (trading month-end in panel)
    dec2023 = pd.DatetimeIndex(
        sorted(d for d in df["date"].unique() if pd.Timestamp(d).to_period("M") == pd.Period("2023-12", "M"))
    )
    if len(dec2023) != 1:
        raise AssertionError(f"expected one Dec-2023 formation, got {dec2023}")
    train_end_2022 = pd.DatetimeIndex(
        sorted(d for d in df["date"].unique() if pd.Timestamp(d).year == 2022)
    ).max()

    frames: dict[str, pd.DataFrame] = {}
    walls: dict[str, float] = {}
    all_fits: dict[str, list] = {}

    for name, Model in ML_MODELS.items():
        t0 = time.perf_counter()
        print(f"holdout walk-forward {name}…")
        # 2024–2026 via engine (retune each January)
        res = eng.run(Model(), panel)
        # 2023-12 from locked OOS 2023 hyperparameters
        oos_fits = json.loads((OOS_FITS / f"{name}_year_fits.json").read_text())
        y2023 = next(f for f in oos_fits if f["year"] == 2023)
        params = dict(y2023["chosen_params"])
        # XGB stores best_n_estimators; keep it for the frozen 2023 refit
        dec_pred, fitted_2023 = _predict_month_with_fit(
            Model,
            df,
            train_end=train_end_2022,
            pred_dates=dec2023,
            params=params,
            feature_names=eng.feature_names,
        )
        coefs = None
        if fitted_2023.coefficients_ is not None:
            coefs = {k: float(v) for k, v in fitted_2023.coefficients_.items()}
        rec_2023 = {
            "year": 2023,
            "train_start": y2023["train_start"],
            "train_end": str(train_end_2022.date()),
            "val_start": y2023["val_start"],
            "val_end": y2023["val_end"],
            "first_pred": str(dec2023.min().date()),
            "last_pred": str(dec2023.max().date()),
            "chosen_params": params,
            "validation_ic": y2023["validation_ic"],
            "n_train": y2023["n_train"],
            "n_val": y2023["n_val"],
            "coefficients": coefs,
            "note": "2023-12 only; hyperparameters from OOS 2023 fit (not retuned)",
        }
        preds = pd.concat([dec_pred, res.predictions], ignore_index=True)
        preds = preds[preds["formation_month"] >= HOLDOUT_FIRST_FORMATION].copy()
        preds["formation_month"] = pd.to_datetime(preds["formation_month"]).astype("datetime64[ns]")
        path = HOLDOUT_PRED / f"{name}_predictions.parquet"
        preds.to_parquet(path, compression="zstd")
        fits_payload = [rec_2023] + [
            {
                "year": r.year,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "val_start": r.val_start,
                "val_end": r.val_end,
                "first_pred": r.first_pred,
                "last_pred": r.last_pred,
                "chosen_params": r.chosen_params,
                "validation_ic": r.validation_ic,
                "n_train": r.n_train,
                "n_val": r.n_val,
                "coefficients": r.coefficients,
            }
            for r in res.year_fits
        ]
        (HOLDOUT_FITS / f"{name}_year_fits.json").write_text(
            json.dumps(fits_payload, indent=2, default=str), encoding="utf-8"
        )
        frames[name] = preds
        walls[name] = time.perf_counter() - t0
        all_fits[name] = fits_payload
        print(
            f"  {name} wall={walls[name]:.1f}s rows={len(preds):,} "
            f"formations={preds['formation_month'].nunique()}"
        )
        for rec in fits_payload:
            print(
                f"    year={rec['year']} params={rec['chosen_params']} "
                f"val_ic={rec['validation_ic']}"
            )

    print("ensemble…")
    ens, ens_log = rank_average_predictions(frames, members=["lasso", "xgboost", "mlp"])
    ens.to_parquet(HOLDOUT_PRED / "ensemble_predictions.parquet", compression="zstd")
    ens_log.to_csv(REPORT / "ensemble_member_months.csv", index=False)
    frames["ensemble"] = ens
    walls["ensemble"] = 0.0

    # Reports keep a lean tuning log (params + val IC); full fits with coefs stay in data/.
    lean = {
        name: [
            {
                "year": r["year"],
                "train_end": r["train_end"],
                "first_pred": r["first_pred"],
                "last_pred": r["last_pred"],
                "chosen_params": r["chosen_params"],
                "validation_ic": r["validation_ic"],
                **({"note": r["note"]} if "note" in r else {}),
            }
            for r in recs
        ]
        for name, recs in all_fits.items()
    }
    (REPORT / "holdout_tuning.json").write_text(
        json.dumps({"walls_s": walls, "fits": lean}, indent=2, default=str),
        encoding="utf-8",
    )
    return frames


def _metrics_table(
    ret_df: pd.DataFrame,
    turn_df: pd.DataFrame,
    ic_map: dict[str, pd.Series],
    ff: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    rf = ff["RF"].reindex(ret_df.index)
    # dropna months only where RF missing — leave NaN RF as 0 only if truly missing early
    rf = rf.fillna(0.0)
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
        base["window"] = label
        ic = ic_map.get(name)
        if ic is not None and len(ic):
            base["rank_ic_mean"] = float(ic.mean())
            base["rank_ic_pos_share"] = float((ic > 0).mean())
        else:
            base["rank_ic_mean"] = float("nan")
            base["rank_ic_pos_share"] = float("nan")

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
        base["ff6_n"] = a_g["n"]
        if name != "ew_universe":
            active = gross - ret_df["ew_universe"]
            a_act = ff6_alpha(active, ff)
            base["ff6_active_alpha_ann"] = a_act["alpha_ann"]
            base["ff6_active_alpha_t"] = a_act["alpha_t"]
            base["ff6_active_n"] = a_act["n"]
        rows.append(base)
    return pd.DataFrame(rows)


def _build_series_from_preds(
    panel_slice: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    bench = equal_weight_universe(panel_slice)
    bench_turn = _universe_turnover(panel_slice)
    comp = classic4_composite_score(panel_slice)
    c_rets, c_turns = long_only_top_decile(panel_slice, comp, long_high=True)
    c_ic = rank_ic(panel_slice, comp)

    series = {"ew_universe": bench, "classic4": c_rets}
    turns_map = {"ew_universe": bench_turn, "classic4": c_turns}
    ic_map: dict[str, pd.Series] = {"classic4": c_ic}

    for name, preds in frames.items():
        rets, turns, ic = _port_from_preds(preds)
        series[name] = rets
        turns_map[name] = turns
        ic_map[name] = ic

    ret_df = pd.DataFrame(series).sort_index()
    turn_df = pd.DataFrame({k: v.reindex(ret_df.index) for k, v in turns_map.items()})
    return ret_df, turn_df, ic_map


def evaluate_holdout(panel: pd.DataFrame, frames: dict[str, pd.DataFrame], ff: pd.DataFrame) -> dict:
    """Step 4: head-to-head, yearly, charts, OOS comparison."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    hold = panel[(panel["date"] >= HOLDOUT_FIRST_FORMATION) & (panel["date"] <= HOLDOUT_MAX_FORMATION)]
    hold = hold[hold["ret_1m"].notna()].copy()

    ret_df, turn_df, ic_map = _build_series_from_preds(hold, frames)
    # Return months: formation 2023-12 → return 2024-01, … formation 2026-08 → 2026-09
    ret_df.to_parquet(HOLDOUT_PRED / "holdout_portfolio_returns.parquet", compression="zstd")
    turn_df.to_parquet(HOLDOUT_PRED / "holdout_portfolio_turnover.parquet", compression="zstd")

    # 33-formation window (all)
    m33 = _metrics_table(ret_df, turn_df, ic_map, ff, "holdout_33")
    # 32-formation: drop first return month (from 2023-12 formation)
    first_rm = return_month(pd.Timestamp(sorted(hold["date"].unique())[0]))
    ret32 = ret_df.loc[ret_df.index > first_rm]
    turn32 = turn_df.reindex(ret32.index)
    # rank_ic is indexed by return_month — drop the Dec-2023 → Jan-2024 month
    ic32: dict[str, pd.Series] = {}
    for k, v in ic_map.items():
        if v is None or len(v) == 0:
            ic32[k] = v
        else:
            ic32[k] = v.loc[v.index > first_rm]
    m32 = _metrics_table(ret32, turn32, ic32, ff, "holdout_32")

    metrics = pd.concat([m33, m32], ignore_index=True)
    metrics.to_csv(REPORT / "holdout_metrics.csv", index=False)

    # Year-by-year on 33-window (by return-month year)
    year_rows = []
    for name in ret_df.columns:
        for y, g in ret_df[name].groupby(ret_df.index.year):
            sub_turn = turn_df[name].reindex(g.index)
            rf = ff["RF"].reindex(g.index).fillna(0.0)
            s = summarize(
                g,
                benchmark=ret_df["ew_universe"].reindex(g.index) if name != "ew_universe" else None,
                turnover=sub_turn,
                rf=rf,
            )
            net = apply_costs(g, sub_turn, 10.0)
            ns = summarize(net, rf=rf)
            ic = ic_map.get(name)
            # IC by formation year ≈ return year - 1 for Dec→Jan boundary; use return year on IC index
            ic_y = float("nan")
            ic_pos = float("nan")
            if ic is not None and len(ic):
                ic_sub = ic[ic.index.year == y]
                if len(ic_sub):
                    ic_y = float(ic_sub.mean())
                    ic_pos = float((ic_sub > 0).mean())
            year_rows.append(
                {
                    "year": int(y),
                    "name": name,
                    "ann_return": s["ann_return"],
                    "sharpe": s["sharpe"],
                    "net_sharpe_10bps": ns["sharpe"],
                    "excess_vs_benchmark": s.get("excess_vs_benchmark", float("nan")),
                    "rank_ic_mean": ic_y,
                    "rank_ic_pos_share": ic_pos,
                    "n_months": s["n_months"],
                }
            )
    yearly = pd.DataFrame(year_rows)
    yearly.to_csv(REPORT / "holdout_by_year.csv", index=False)

    # Cumulative net 10 bps
    wealth = {
        name: (1.0 + apply_costs(ret_df[name], turn_df[name], 10.0).fillna(0)).cumprod()
        for name in ret_df.columns
    }
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
        ax.plot(wdf.index, wdf[col], label=col, color=colors.get(col), lw=1.2)
    ax.set_title("Holdout cumulative return, net 10 bps/side (33 formations)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT / "holdout_cumulative_net10.png", dpi=120)
    plt.close(fig)

    # OOS vs holdout comparison
    oos = pd.read_csv(OOS_METRICS)
    hold_m = m33.set_index("name")
    oos_m = oos.set_index("name")
    scored = [n for n in MODEL_ORDER if n in hold_m.index and n in oos_m.index and n != "ew_universe"]
    rows = []
    for name in MODEL_ORDER:
        if name not in hold_m.index or name not in oos_m.index:
            continue
        rows.append(
            {
                "name": name,
                "oos_sharpe": float(oos_m.loc[name, "sharpe"]),
                "holdout_sharpe": float(hold_m.loc[name, "sharpe"]),
                "oos_net_sharpe_10bps": float(oos_m.loc[name, "net_sharpe_10bps"]),
                "holdout_net_sharpe_10bps": float(hold_m.loc[name, "net_sharpe_10bps"]),
                "oos_excess": float(oos_m.loc[name, "excess_vs_benchmark"])
                if name != "ew_universe" and pd.notna(oos_m.loc[name].get("excess_vs_benchmark", np.nan))
                else float("nan"),
                "holdout_excess": float(hold_m.loc[name, "excess_vs_benchmark"])
                if name != "ew_universe"
                else float("nan"),
                "oos_rank_ic": float(oos_m.loc[name, "rank_ic_mean"])
                if pd.notna(oos_m.loc[name].get("rank_ic_mean", np.nan))
                else float("nan"),
                "holdout_rank_ic": float(hold_m.loc[name, "rank_ic_mean"])
                if pd.notna(hold_m.loc[name, "rank_ic_mean"])
                else float("nan"),
            }
        )
    cmp = pd.DataFrame(rows)
    # Spearman of Sharpe ranks among models (exclude ew)
    sub = cmp[cmp["name"].isin(scored)]
    spearman = float(sub["oos_sharpe"].corr(sub["holdout_sharpe"], method="spearman"))
    cmp_meta = {"spearman_sharpe_oos_vs_holdout": spearman, "scored_names": scored}
    cmp.to_csv(REPORT / "oos_vs_holdout.csv", index=False)
    (REPORT / "oos_vs_holdout_rankcorr.json").write_text(json.dumps(cmp_meta, indent=2))
    print("OOS vs holdout Sharpe Spearman:", spearman)

    return {
        "ret_df": ret_df,
        "turn_df": turn_df,
        "ic_map": ic_map,
        "metrics": metrics,
        "m33": m33,
        "spearman": spearman,
    }


def evaluate_full_period(
    panel: pd.DataFrame, holdout_frames: dict[str, pd.DataFrame], ff: pd.DataFrame
) -> None:
    """Step 5: combined 2011-01 → 2026-08."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    full = panel[(panel["date"] >= OOS_START) & (panel["date"] <= HOLDOUT_MAX_FORMATION)]
    full = full[full["ret_1m"].notna()].copy()

    # Merge OOS + holdout predictions per model
    frames = {}
    for name in ML_MODELS:
        oos = pd.read_parquet(OOS_PRED / f"{name}_predictions.parquet")
        oos["formation_month"] = pd.to_datetime(oos["formation_month"])
        assert oos["formation_month"].max() <= MAX_FORMATION_MONTH
        h = holdout_frames[name].copy()
        h["formation_month"] = pd.to_datetime(h["formation_month"])
        merged = pd.concat([oos, h], ignore_index=True)
        merged = merged.drop_duplicates(subset=["formation_month", "security_id"], keep="last")
        frames[name] = merged
    ens, _ = rank_average_predictions(frames, members=["lasso", "xgboost", "mlp"])
    frames["ensemble"] = ens

    ret_df, turn_df, ic_map = _build_series_from_preds(full, frames)
    # Keep return months from first OOS return through last holdout return
    ret_df = ret_df.loc[
        (ret_df.index >= pd.Timestamp("2011-02-28"))
        & (ret_df.index <= return_month(HOLDOUT_MAX_FORMATION))
    ]
    turn_df = turn_df.reindex(ret_df.index)
    metrics = _metrics_table(ret_df, turn_df, ic_map, ff, "full_2011_2026")
    metrics.to_csv(REPORT / "full_period_metrics.csv", index=False)

    wealth = {
        name: (1.0 + apply_costs(ret_df[name], turn_df[name], 10.0).fillna(0)).cumprod()
        for name in ret_df.columns
    }
    wdf = pd.DataFrame(wealth)
    fig, ax = plt.subplots(figsize=(11, 5))
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
        ax.plot(wdf.index, wdf[col], label=col, color=colors.get(col), lw=1.1)
    holdout_start_rm = return_month(pd.Timestamp("2023-12-29"))
    ax.axvline(holdout_start_rm, color="black", ls="--", lw=1.0, label="holdout start")
    ax.set_title("Full sample cumulative return, net 10 bps/side (2011 → 2026)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT / "full_period_cumulative_net10.png", dpi=120)
    plt.close(fig)


def interpretability(panel: pd.DataFrame) -> None:
    """Step 6: Lasso signs vs 2023; XGB SHAP holdout vs OOS ranks."""
    feats = list(FEATURE_NAMES)
    # Lasso non-zero features / signs
    fits = json.loads((HOLDOUT_FITS / "lasso_year_fits.json").read_text())
    rows = []
    by_year = {f["year"]: f for f in fits}
    base = by_year[2023].get("coefficients") or {}
    for year in (2023, 2024, 2025, 2026):
        coefs = by_year[year].get("coefficients") or {}
        for f in feats:
            v = float(coefs.get(f, 0.0))
            rows.append(
                {
                    "year": year,
                    "feature": f,
                    "coef": v,
                    "sign": int(np.sign(v)) if v != 0 else 0,
                    "nonzero": abs(v) > 1e-12,
                }
            )
    lasso_df = pd.DataFrame(rows)
    lasso_df.to_csv(REPORT / "lasso_coefs_by_year.csv", index=False)

    # Compare non-zero sets / sign flips vs 2023
    base_nz = {f for f, v in base.items() if abs(float(v)) > 1e-12}
    cmp_rows = []
    for year in (2024, 2025, 2026):
        coefs = by_year[year].get("coefficients") or {}
        nz = {f for f, v in coefs.items() if abs(float(v)) > 1e-12}
        flips = []
        for f in sorted(base_nz & nz):
            if np.sign(float(base[f])) != np.sign(float(coefs[f])) and float(coefs[f]) != 0:
                flips.append(f)
        cmp_rows.append(
            {
                "year": year,
                "n_nonzero": len(nz),
                "n_nonzero_2023": len(base_nz),
                "added": ",".join(sorted(nz - base_nz)),
                "dropped": ",".join(sorted(base_nz - nz)),
                "sign_flips_vs_2023": ",".join(flips),
            }
        )
    pd.DataFrame(cmp_rows).to_csv(REPORT / "lasso_vs_2023.csv", index=False)

    # XGB SHAP on holdout rows: refit each holdout year with chosen params, explain that year's rows
    eng = WalkForwardEngine(
        first_oos_year=2024,
        last_oos_year=2026,
        max_formation=HOLDOUT_MAX_FORMATION,
        n_jobs=1,
        random_seed=42,
    )
    df = eng.prepare_panel(panel)
    xgb_fits = json.loads((HOLDOUT_FITS / "xgboost_year_fits.json").read_text())
    shap_acc = np.zeros(len(feats), dtype=float)
    n_rows = 0
    for rec in xgb_fits:
        year = int(rec["year"])
        train_end = pd.Timestamp(rec["train_end"])
        if year == 2023:
            pred_mask = df["date"].dt.to_period("M") == pd.Period("2023-12", "M")
        else:
            pred_mask = (df["date"].dt.year == year) & (df["date"] <= HOLDOUT_MAX_FORMATION)
        train = df[df["date"] <= train_end]
        test = df.loc[pred_mask]
        if test.empty:
            continue
        params = dict(rec["chosen_params"])
        model = XGBoostModel().clone(**params)
        # freeze n_estimators if present
        model.fit(
            train[feats].to_numpy(dtype=float),
            train["y"].to_numpy(dtype=float),
            feature_names=feats,
        )
        # subsample for SHAP if large
        rng = np.random.default_rng(42)
        take = min(8_000, len(test))
        idx = rng.choice(len(test), size=take, replace=False)
        X_te = test.iloc[idx][feats].to_numpy(dtype=float)
        explainer = shap.TreeExplainer(model._model)
        sv = explainer.shap_values(X_te)
        shap_acc += np.abs(sv).sum(axis=0)
        n_rows += take
    hold_imp = pd.Series(shap_acc / max(n_rows, 1), index=feats, name="holdout_mean_abs_shap")
    hold_rank = hold_imp.rank(ascending=False)
    hold_imp.to_csv(REPORT / "xgb_holdout_shap_mean_abs.csv")
    hold_rank.to_csv(REPORT / "xgb_holdout_shap_ranks.csv")

    oos_ranks = pd.read_csv(OOS_IMP_RANKS, index_col=0)
    if "xgboost" not in oos_ranks.columns:
        spearman = float("nan")
    else:
        aligned = pd.DataFrame(
            {"oos": oos_ranks["xgboost"], "holdout": hold_rank}
        ).dropna()
        spearman = float(aligned["oos"].corr(aligned["holdout"], method="spearman"))
    (REPORT / "xgb_shap_oos_vs_holdout.json").write_text(
        json.dumps({"spearman_rank_corr": spearman, "n_features": int(len(feats))}, indent=2)
    )
    print("XGB SHAP OOS vs holdout rank Spearman:", spearman)


def main() -> None:
    _require_flag()
    warnings.filterwarnings("ignore", category=UserWarning)
    t_all = time.perf_counter()
    for p in (HOLDOUT_PRED, HOLDOUT_FITS, REPORT):
        p.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    print("loading panel…")
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])

    print("Step 2: data checks…")
    checks = data_checks(panel)
    ff = checks["ff"]

    print("Step 3: holdout refits…")
    frames = run_models(panel, cfg)

    print("Step 4: evaluate holdout…")
    ev = evaluate_holdout(panel, frames, ff)

    print("Step 5: full period…")
    evaluate_full_period(panel, frames, ff)

    print("Step 6: interpretability…")
    interpretability(panel)

    summary = {
        "wall_s": round(time.perf_counter() - t_all, 2),
        "ff_last_month": str(checks["ff_last_month"].date()),
        "spearman_sharpe_oos_vs_holdout": ev["spearman"],
        "n_holdout_formations": 33,
        "code_commit_preregistered": "6fff7a54e0b1fe761caaa5dc4caa5ac7c1413a4d",
    }
    (REPORT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done wall=", summary["wall_s"])
    print(ev["m33"][["name", "sharpe", "net_sharpe_10bps", "rank_ic_mean", "ff6_active_alpha_ann"]].to_string(index=False))


if __name__ == "__main__":
    main()
