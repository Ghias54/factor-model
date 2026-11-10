"""Full-window within-month label-shuffle leakage gate (formation ≤ 2023-11).

For each model, shuffle ``target_rank`` within month, re-run the same walk-forward
grids as production, and summarize OOS rank IC under the null. Ensemble is the
CS rank-average of shuffled Lasso + XGBoost + MLP for the same seed.

MLP uses ``n_seeds=2`` (of 5) per fit to save wall time — logged below.
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
from factor_model.models import (
    LassoModel,
    MLPModel,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
)
from factor_model.models.ensemble import rank_average_predictions
from factor_model.walkforward.engine import (
    WalkForwardEngine,
    monthly_rank_ic,
    monthly_rank_ic_series,
    shuffle_target_within_month,
)

PANEL = ROOT / "data" / "panel" / "panel.parquet"
PRED = ROOT / "data" / "predictions"
OUT = ROOT / "reports" / "models"
LEAK_DIR = PRED / "leakage"
PASS_ABS = 0.005
PASS_T = 2.0

# Seeds: linear/XGB get 5; RF/MLP get 3 (expensive). Ensemble uses shared seeds.
SEEDS_FAST = (0, 1, 2, 3, 4)
SEEDS_HEAVY = (0, 1, 2)
ENSEMBLE_MEMBERS = ["lasso", "xgboost", "mlp"]
MLP_INTERNAL_SEEDS = 2  # of production 5


def _ic_stats(preds: pd.DataFrame) -> dict[str, float]:
    series = monthly_rank_ic_series(
        preds["prediction"], preds["realized_ret"], preds["formation_month"]
    )
    arr = series.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    sd = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    t = float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    pos = float(np.mean(arr > 0)) if n else float("nan")
    return {
        "mean_ic": mean,
        "sd_monthly_ic": sd,
        "t_mean_ic": t,
        "pos_share": pos,
        "n_months": n,
        "monthly_ics": series,
    }


def _pooled_t(monthly_lists: list[pd.Series]) -> tuple[float, float, float, float, int]:
    """Pool monthly ICs across seeds → mean, sd_across_seed_means, t_pooled, pos, n."""
    seed_means = []
    pooled = []
    for s in monthly_lists:
        a = s.to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        if len(a):
            seed_means.append(float(np.mean(a)))
            pooled.extend(a.tolist())
    arr = np.asarray(pooled, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    sd_seeds = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan")
    sd = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    t = float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    pos = float(np.mean(arr > 0)) if n else float("nan")
    return mean, sd_seeds, t, pos, n


def _make_model(name: str):
    if name == "lasso":
        return LassoModel()
    if name == "ridge":
        return RidgeModel()
    if name == "random_forest":
        return RandomForestModel(n_jobs=-1)
    if name == "xgboost":
        return XGBoostModel(n_jobs=-1)
    if name == "mlp":
        return MLPModel(n_seeds=MLP_INTERNAL_SEEDS)
    raise KeyError(name)


def _real_ic(name: str) -> dict[str, float]:
    path = PRED / f"{name}_predictions.parquet"
    if not path.exists():
        return {"real_mean_ic": float("nan"), "real_t": float("nan")}
    df = pd.read_parquet(path)
    st = _ic_stats(df)
    return {"real_mean_ic": st["mean_ic"], "real_t": st["t_mean_ic"]}


def _shift_target_forward(panel: pd.DataFrame) -> pd.DataFrame:
    """Features at t; train/eval target = return over t+1→t+2 (ret_1m at t+1)."""
    out = panel.sort_values(["security_id", "date"]).copy()
    out["ret_1m"] = out.groupby("security_id", sort=False)["ret_1m"].shift(-1)
    out = out[out["ret_1m"].notna()].copy()
    out["target_rank"] = out.groupby("date")["ret_1m"].rank(pct=True, method="average")
    return out.reset_index(drop=True)


def run_one(eng: WalkForwardEngine, name: str, panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    LEAK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LEAK_DIR / f"{name}_seed{seed}_predictions.parquet"
    if out_path.exists():
        print(f"  reuse {out_path.name}")
        return pd.read_parquet(out_path)
    shuffled = shuffle_target_within_month(panel, seed=seed)
    t0 = time.perf_counter()
    print(f"  walk-forward {name} seed={seed}…", flush=True)
    res = eng.run(_make_model(name), shuffled)
    assert res.predictions["formation_month"].max() <= MAX_FORMATION_MONTH
    res.predictions.to_parquet(out_path, compression="zstd")
    print(f"  {name} seed={seed} wall={time.perf_counter()-t0:.1f}s rows={len(res.predictions):,}", flush=True)
    return res.predictions


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    LEAK_DIR.mkdir(parents=True, exist_ok=True)
    print("loading panel…", flush=True)
    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    eng = WalkForwardEngine(n_jobs=-1, random_seed=42)

    plan = [
        ("lasso", SEEDS_FAST),
        ("ridge", SEEDS_FAST),
        ("xgboost", SEEDS_FAST),
        ("random_forest", SEEDS_HEAVY),
        ("mlp", SEEDS_HEAVY),
    ]

    frames_by_seed: dict[int, dict[str, pd.DataFrame]] = {}
    seed_stats: list[dict] = []
    monthly_by_model: dict[str, list[pd.Series]] = {n: [] for n, _ in plan}
    monthly_by_model["ensemble"] = []

    t_all = time.perf_counter()
    for name, seeds in plan:
        for seed in seeds:
            preds = run_one(eng, name, panel, seed)
            st = _ic_stats(preds)
            monthly_by_model[name].append(st["monthly_ics"])
            seed_stats.append(
                {
                    "model": name,
                    "seed": seed,
                    "mean_ic": st["mean_ic"],
                    "t_mean_ic": st["t_mean_ic"],
                    "pos_share": st["pos_share"],
                    "n_months": st["n_months"],
                }
            )
            frames_by_seed.setdefault(seed, {})[name] = preds
            print(
                f"  {name} seed={seed} mean_ic={st['mean_ic']:.5f} t={st['t_mean_ic']:.2f}",
                flush=True,
            )

    # Ensemble on shared seeds (lasso + xgb + mlp)
    for seed in SEEDS_HEAVY:
        members = frames_by_seed[seed]
        missing = [m for m in ENSEMBLE_MEMBERS if m not in members]
        if missing:
            raise RuntimeError(f"ensemble seed {seed} missing {missing}")
        ens, _log = rank_average_predictions(members, members=ENSEMBLE_MEMBERS)
        path = LEAK_DIR / f"ensemble_seed{seed}_predictions.parquet"
        ens.to_parquet(path, compression="zstd")
        st = _ic_stats(ens)
        monthly_by_model["ensemble"].append(st["monthly_ics"])
        seed_stats.append(
            {
                "model": "ensemble",
                "seed": seed,
                "mean_ic": st["mean_ic"],
                "t_mean_ic": st["t_mean_ic"],
                "pos_share": st["pos_share"],
                "n_months": st["n_months"],
            }
        )
        print(f"  ensemble seed={seed} mean_ic={st['mean_ic']:.5f} t={st['t_mean_ic']:.2f}", flush=True)

    # Forward-shift MLP (unshuffled features; target = t+2 return)
    print("MLP forward-shift (target = t+2 return)…", flush=True)
    fwd_path = LEAK_DIR / "mlp_forward_shift_predictions.parquet"
    if fwd_path.exists():
        fwd_preds = pd.read_parquet(fwd_path)
        print("  reuse forward-shift preds", flush=True)
    else:
        shifted = _shift_target_forward(panel)
        t0 = time.perf_counter()
        # Production internal seeds for the one-shot diagnostic
        res = eng.run(MLPModel(n_seeds=MLP_INTERNAL_SEEDS), shifted)
        assert res.predictions["formation_month"].max() <= MAX_FORMATION_MONTH
        fwd_preds = res.predictions
        fwd_preds.to_parquet(fwd_path, compression="zstd")
        print(f"  forward-shift wall={time.perf_counter()-t0:.1f}s", flush=True)
    fwd_st = _ic_stats(fwd_preds)

    # Summary table
    rows = []
    failures = []
    for name in [n for n, _ in plan] + ["ensemble"]:
        mean, sd_seeds, t, pos, n = _pooled_t(monthly_by_model[name])
        real = _real_ic(name)
        row = {
            "model": name,
            "n_seeds": len(monthly_by_model[name]),
            "shuffle_mean_ic": mean,
            "shuffle_sd_across_seeds": sd_seeds,
            "shuffle_t": t,
            "shuffle_pos_share": pos,
            "n_pooled_months": n,
            **real,
            "pass_abs": bool(abs(mean) < PASS_ABS) if np.isfinite(mean) else False,
            "pass_t": bool(abs(t) < PASS_T) if np.isfinite(t) else False,
        }
        row["pass"] = bool(row["pass_abs"] and row["pass_t"])
        if not row["pass"]:
            failures.append(name)
        rows.append(row)

    summary = pd.DataFrame(rows)
    seed_df = pd.DataFrame(seed_stats)
    summary.to_csv(OUT / "leakage_full_window.csv", index=False)
    seed_df.to_csv(OUT / "leakage_full_window_seeds.csv", index=False)
    meta = {
        "wall_s": round(time.perf_counter() - t_all, 1),
        "pass_abs": PASS_ABS,
        "pass_t": PASS_T,
        "mlp_internal_seeds": MLP_INTERNAL_SEEDS,
        "failures": failures,
        "passed": len(failures) == 0,
        "mlp_forward_shift_mean_ic": fwd_st["mean_ic"],
        "mlp_forward_shift_t": fwd_st["t_mean_ic"],
        "mlp_forward_shift_pos_share": fwd_st["pos_share"],
        "max_formation": str(MAX_FORMATION_MONTH.date()),
    }
    (OUT / "leakage_full_window_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print("forward-shift MLP:", meta["mlp_forward_shift_mean_ic"], "t=", meta["mlp_forward_shift_t"], flush=True)
    print("PASSED" if meta["passed"] else f"FAILED: {failures}", flush=True)
    print("wall=", meta["wall_s"], flush=True)


if __name__ == "__main__":
    main()
