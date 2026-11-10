"""Lasso full-window shuffle with 40 seeds + real-run validation alpha check.

Seeds 0-4 are reused from data/predictions/leakage/; seeds 5-39 are new.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_model.factors.classic import MAX_FORMATION_MONTH
from factor_model.models import LassoModel
from factor_model.models.linear import LASSO_ALPHAS
from factor_model.walkforward.engine import (
    WalkForwardEngine,
    monthly_rank_ic_series,
    shuffle_target_within_month,
)

PANEL = ROOT / "data" / "panel" / "panel.parquet"
LEAK = ROOT / "data" / "predictions" / "leakage"
FITS = ROOT / "data" / "predictions" / "fits"
REPORT = ROOT / "reports" / "models"
ALL_SEEDS = list(range(40))  # 0-4 existing, 5-39 new
PASS_ABS = 0.005
PASS_T = 2.0


def _ic_old_drop(pred, ret, dates) -> float:
    df = pd.DataFrame({"p": pred, "r": ret, "d": dates}).dropna()
    ics = []
    n_drop = 0
    for _, g in df.groupby("d", sort=True):
        if len(g) < 20:
            continue
        if g["p"].nunique(dropna=True) < 2 or float(g["p"].std(ddof=0) or 0.0) == 0.0:
            n_drop += 1
            continue
        if g["r"].nunique(dropna=True) < 2:
            n_drop += 1
            continue
        ics.append(float(g["p"].corr(g["r"], method="spearman")))
    return (float(np.nanmean(ics)) if ics else float("nan")), n_drop


def _seed_stats(preds: pd.DataFrame) -> dict:
    s = monthly_rank_ic_series(
        preds["prediction"], preds["realized_ret"], preds["formation_month"]
    )
    arr = s.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    sd = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    t = float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    return {
        "mean_ic": mean,
        "t_mean_ic": t,
        "pos_share": float(np.mean(arr > 0)) if n else float("nan"),
        "n_months": n,
        "n_constant_pred": int(s.attrs.get("n_constant_pred", 0)),
        "series": s,
    }


def run_new_seeds(eng: WalkForwardEngine, panel: pd.DataFrame) -> None:
    LEAK.mkdir(parents=True, exist_ok=True)
    for seed in ALL_SEEDS:
        path = LEAK / f"lasso_seed{seed}_predictions.parquet"
        if path.exists():
            print(f"  reuse seed={seed}", flush=True)
            continue
        print(f"  walk-forward lasso seed={seed}…", flush=True)
        t0 = time.perf_counter()
        shuffled = shuffle_target_within_month(panel, seed=seed)
        res = eng.run(LassoModel(), shuffled)
        assert res.predictions["formation_month"].max() <= MAX_FORMATION_MONTH
        res.predictions.to_parquet(path, compression="zstd")
        st = _seed_stats(res.predictions)
        print(
            f"  seed={seed} wall={time.perf_counter()-t0:.1f}s "
            f"mean_ic={st['mean_ic']:.5f} t={st['t_mean_ic']:.2f} "
            f"const={st['n_constant_pred']}",
            flush=True,
        )


def summarize_40() -> tuple[pd.DataFrame, dict]:
    seed_rows = []
    series_list = []
    for seed in ALL_SEEDS:
        preds = pd.read_parquet(LEAK / f"lasso_seed{seed}_predictions.parquet")
        st = _seed_stats(preds)
        series_list.append(st["series"])
        seed_rows.append(
            {
                "seed": seed,
                "mean_ic": st["mean_ic"],
                "t_mean_ic": st["t_mean_ic"],
                "pos_share": st["pos_share"],
                "n_months": st["n_months"],
                "n_constant_pred": st["n_constant_pred"],
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_means = seed_df["mean_ic"].to_numpy(dtype=float)
    n_seeds = len(seed_means)
    pooled_mean = float(np.mean(seed_means))
    sd_seeds = float(np.std(seed_means, ddof=1))
    se = sd_seeds / np.sqrt(n_seeds)
    ci_lo = pooled_mean - 1.96 * se
    ci_hi = pooled_mean + 1.96 * se

    pooled_months = []
    n_const = 0
    for s in series_list:
        a = s.to_numpy(dtype=float)
        pooled_months.extend(a[np.isfinite(a)].tolist())
        n_const += int(s.attrs.get("n_constant_pred", 0))
    arr = np.asarray(pooled_months, dtype=float)
    n_m = len(arr)
    sd_m = float(np.std(arr, ddof=1)) if n_m > 1 else float("nan")
    mean_m = float(np.mean(arr)) if n_m else float("nan")
    t_pooled = float(mean_m / (sd_m / np.sqrt(n_m))) if n_m > 1 and sd_m > 0 else float("nan")

    n_t_gt2 = int((seed_df["t_mean_ic"].abs() > 2).sum())  # |t|>2
    n_t_pos_gt2 = int((seed_df["t_mean_ic"] > 2).sum())

    summary = {
        "n_seeds": n_seeds,
        "pooled_mean_ic": pooled_mean,
        "sd_across_seed_means": sd_seeds,
        "se_mean": float(se),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
        "pooled_monthly_mean_ic": mean_m,
        "pooled_monthly_t": t_pooled,
        "n_pooled_months": n_m,
        "n_constant_pred_total": n_const,
        "seed_mean_min": float(seed_means.min()),
        "seed_mean_max": float(seed_means.max()),
        "n_seeds_t_gt_2": n_t_pos_gt2,
        "n_seeds_abs_t_gt_2": n_t_gt2,
        "pass_abs": bool(abs(pooled_mean) < PASS_ABS),
        "pass_t": bool(abs(t_pooled) < PASS_T),
    }
    summary["pass"] = bool(summary["pass_abs"] and summary["pass_t"])

    # Save: summary row + per-seed
    out_rows = [{**summary, "row_type": "summary", "seed": -1}]
    for r in seed_rows:
        out_rows.append({**r, "row_type": "seed"})
    out = pd.DataFrame(out_rows)
    out.to_csv(REPORT / "leakage_lasso_40seed.csv", index=False)
    (REPORT / "leakage_lasso_40seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return seed_df, summary


def validation_alpha_check(panel: pd.DataFrame) -> dict:
    """Refit Lasso year windows; compare chosen alpha under old vs new val IC."""
    eng = WalkForwardEngine(n_jobs=-1, random_seed=42)
    df = eng.prepare_panel(panel)
    calendar = eng._formation_calendar(df)
    windows = eng._year_windows(calendar)
    rows = []
    n_changed = 0
    for year, train_end, first_pred, pred_dates in windows:
        train = df[(df["date"] >= eng.sample_start) & (df["date"] <= train_end)]
        train_months = pd.DatetimeIndex(sorted(train["date"].unique()))
        val_months = train_months[-eng.validation_months :]
        fit_months = train_months[: -eng.validation_months]
        tr = train[train["date"].isin(fit_months)]
        va = train[train["date"].isin(val_months)]
        X_tr = tr[eng.feature_names].to_numpy(dtype=float)
        y_tr = tr["y"].to_numpy(dtype=float)
        X_va = va[eng.feature_names].to_numpy(dtype=float)
        ret_va = va["ret_1m"].to_numpy(dtype=float)
        dates_va = va["date"].to_numpy()

        scored_new = []
        scored_old = []
        n_drop_by_alpha = {}
        for alpha in LASSO_ALPHAS:
            m = LassoModel(alpha=alpha)
            m.fit(X_tr, y_tr, feature_names=eng.feature_names)
            pred = m.predict(X_va)
            s_new = monthly_rank_ic_series(
                pd.Series(pred), pd.Series(ret_va), pd.Series(dates_va)
            )
            ic_new = float(s_new.mean()) if len(s_new) else float("nan")
            ic_old, n_drop = _ic_old_drop(pd.Series(pred), pd.Series(ret_va), pd.Series(dates_va))
            n_drop_by_alpha[alpha] = n_drop
            scored_new.append(({"alpha": alpha}, ic_new))
            scored_old.append(({"alpha": alpha}, ic_old))

        def _pick(items):
            def key(item):
                params, ic = item
                ic_v = -np.inf if (ic is None or (isinstance(ic, float) and np.isnan(ic))) else ic
                return (ic_v, params.get("alpha", 0.0))

            return max(items, key=key)

        best_new, ic_n = _pick(scored_new)
        best_old, ic_o = _pick(scored_old)
        changed = float(best_new["alpha"]) != float(best_old["alpha"])
        if changed:
            n_changed += 1
        rows.append(
            {
                "year": year,
                "alpha_old_drop": float(best_old["alpha"]),
                "ic_old": float(ic_o) if ic_o == ic_o else float("nan"),
                "alpha_new_zero": float(best_new["alpha"]),
                "ic_new": float(ic_n) if ic_n == ic_n else float("nan"),
                "changed": changed,
                "n_constant_val_months_at_chosen_new": int(
                    monthly_rank_ic_series(
                        pd.Series(
                            LassoModel(alpha=best_new["alpha"])
                            .fit(X_tr, y_tr, feature_names=eng.feature_names)
                            .predict(X_va)
                        ),
                        pd.Series(ret_va),
                        pd.Series(dates_va),
                    ).attrs.get("n_constant_pred", 0)
                ),
                "max_dropped_months_any_alpha": int(max(n_drop_by_alpha.values()) if n_drop_by_alpha else 0),
            }
        )
        print(
            f"  year={year} old_alpha={best_old['alpha']:.6g} new_alpha={best_new['alpha']:.6g} "
            f"changed={changed} max_drop={rows[-1]['max_dropped_months_any_alpha']}",
            flush=True,
        )

    val_df = pd.DataFrame(rows)
    val_df.to_csv(REPORT / "lasso_validation_ic_metric_compare.csv", index=False)
    meta = {
        "n_years": len(val_df),
        "n_years_alpha_changed": int(n_changed),
        "years_changed": val_df.loc[val_df["changed"], "year"].tolist(),
        "alpha_stable": n_changed == 0,
    }
    (REPORT / "lasso_validation_ic_metric_compare.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    REPORT.mkdir(parents=True, exist_ok=True)
    print("loading panel…", flush=True)
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    eng = WalkForwardEngine(n_jobs=-1, random_seed=42)

    t0 = time.perf_counter()
    print("Lasso shuffle seeds 0-39…", flush=True)
    run_new_seeds(eng, panel)
    seed_df, summary = summarize_40()
    print(json.dumps(summary, indent=2), flush=True)
    print(seed_df[["seed", "mean_ic", "t_mean_ic", "n_constant_pred"]].to_string(index=False), flush=True)
    print(f"shuffle wall={time.perf_counter()-t0:.1f}s", flush=True)

    print("Validation alpha old vs new metric (real panel)…", flush=True)
    t1 = time.perf_counter()
    val_meta = validation_alpha_check(panel)
    print(json.dumps(val_meta, indent=2), flush=True)
    print(f"validation check wall={time.perf_counter()-t1:.1f}s", flush=True)

    ok = bool(summary["pass"] and val_meta["alpha_stable"])
    (REPORT / "lasso_40seed_gate.json").write_text(
        json.dumps({"shuffle_pass": summary["pass"], "alpha_stable": val_meta["alpha_stable"], "merge_ok": ok}, indent=2),
        encoding="utf-8",
    )
    print("MERGE_OK" if ok else "MERGE_BLOCKED", flush=True)


if __name__ == "__main__":
    main()
