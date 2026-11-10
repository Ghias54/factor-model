"""Re-score leakage gate + real constant-month audit after IC metric fix.

No refitting. Reads saved predictions under data/predictions/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_model.evaluation.costs import apply_costs
from factor_model.evaluation.ff_alpha import ff6_alpha
from factor_model.evaluation.metrics import summarize
from factor_model.external.ff import load_ff_factors
from factor_model.factors.classic import (
    MAX_FORMATION_MONTH,
    _decile_labels,
    equal_weight_universe,
    long_only_top_decile,
    rank_ic,
    return_month,
)
from factor_model.walkforward.engine import monthly_rank_ic_series

PRED = ROOT / "data" / "predictions"
LEAK = PRED / "leakage"
REPORT = ROOT / "reports" / "models"
PANEL = ROOT / "data" / "panel" / "panel.parquet"
PASS_ABS = 0.005
PASS_T = 2.0

MODELS = ["lasso", "ridge", "xgboost", "random_forest", "mlp", "ensemble"]
SEEDS = {
    "lasso": range(5),
    "ridge": range(5),
    "xgboost": range(5),
    "random_forest": range(3),
    "mlp": range(3),
    "ensemble": range(3),
}


def _ic_bundle(preds: pd.DataFrame) -> dict:
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


def _old_ic_dropped(preds: pd.DataFrame) -> float:
    """Replicate pre-fix skip logic for *_old columns."""
    df = preds[["prediction", "realized_ret", "formation_month"]].dropna()
    ics = []
    for _, g in df.groupby("formation_month", sort=True):
        if len(g) < 20:
            continue
        if g["prediction"].nunique(dropna=True) < 2 or g["realized_ret"].nunique(dropna=True) < 2:
            continue
        ics.append(float(g["prediction"].corr(g["realized_ret"], method="spearman")))
    return float(np.nanmean(ics)) if ics else float("nan")


def rescore_leakage() -> tuple[pd.DataFrame, bool, list[str]]:
    old_summary = pd.read_csv(REPORT / "leakage_full_window.csv")
    old_by_model = old_summary.set_index("model")

    seed_rows = []
    monthly_by_model: dict[str, list[pd.Series]] = {m: [] for m in MODELS}
    for name in MODELS:
        for seed in SEEDS[name]:
            path = LEAK / f"{name}_seed{seed}_predictions.parquet"
            preds = pd.read_parquet(path)
            assert pd.to_datetime(preds["formation_month"]).max() <= MAX_FORMATION_MONTH
            b = _ic_bundle(preds)
            monthly_by_model[name].append(b["series"])
            seed_rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "mean_ic": b["mean_ic"],
                    "mean_ic_old": _old_ic_dropped(preds),
                    "t_mean_ic": b["t_mean_ic"],
                    "pos_share": b["pos_share"],
                    "n_months": b["n_months"],
                    "n_constant_pred": b["n_constant_pred"],
                }
            )

    rows = []
    failures = []
    for name in MODELS:
        pooled = []
        seed_means = []
        n_const = 0
        for s in monthly_by_model[name]:
            a = s.to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            if len(a):
                seed_means.append(float(np.mean(a)))
                pooled.extend(a.tolist())
            n_const += int(s.attrs.get("n_constant_pred", 0))
        arr = np.asarray(pooled, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr)) if n else float("nan")
        sd_seeds = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan")
        sd = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
        t = float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan")
        pos = float(np.mean(arr > 0)) if n else float("nan")
        old = old_by_model.loc[name] if name in old_by_model.index else None
        row = {
            "model": name,
            "n_seeds": len(monthly_by_model[name]),
            "shuffle_mean_ic": mean,
            "shuffle_mean_ic_old": float(old["shuffle_mean_ic"]) if old is not None else float("nan"),
            "shuffle_sd_across_seeds": sd_seeds,
            "shuffle_t": t,
            "shuffle_t_old": float(old["shuffle_t"]) if old is not None else float("nan"),
            "shuffle_pos_share": pos,
            "n_pooled_months": n,
            "n_constant_pred_total": n_const,
            "real_mean_ic": float(old["real_mean_ic"]) if old is not None else float("nan"),
            "real_t": float(old["real_t"]) if old is not None else float("nan"),
            "pass_abs": bool(abs(mean) < PASS_ABS) if np.isfinite(mean) else False,
            "pass_t": bool(abs(t) < PASS_T) if np.isfinite(t) else False,
        }
        row["pass"] = bool(row["pass_abs"] and row["pass_t"])
        if not row["pass"]:
            failures.append(name)
        rows.append(row)

    summary = pd.DataFrame(rows)
    seeds = pd.DataFrame(seed_rows)
    summary.to_csv(REPORT / "leakage_full_window.csv", index=False)
    seeds.to_csv(REPORT / "leakage_full_window_seeds.csv", index=False)
    meta = {
        "passed": len(failures) == 0,
        "failures": failures,
        "pass_abs": PASS_ABS,
        "pass_t": PASS_T,
        "metric": "constant_pred_month_ic_equals_0",
    }
    (REPORT / "leakage_full_window_summary.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return summary, len(failures) == 0, failures


def _top_decile_tie_frac(scores: pd.Series) -> tuple[float, str]:
    """Fraction of names tied at the score used as top-decile cutoff; tie-break note."""
    x = scores.dropna()
    if len(x) < 20:
        return float("nan"), "too_few"
    # Mirror _decile_labels: rank method='first' then qcut
    r = x.rank(method="first")
    try:
        dec = pd.qcut(r, 10, labels=False, duplicates="drop") + 1
    except ValueError:
        return 1.0, "qcut_failed_all_tied"
    top = int(dec.max())
    # cutoff score = minimum raw score among names labeled top decile
    top_mask = dec == top
    cutoff = float(x.loc[top_mask].min())
    n_at_cutoff = int((x == cutoff).sum())
    frac = n_at_cutoff / len(x)
    note = (
        "rank(method='first') then qcut — ties broken by row order within the month group"
    )
    return frac, note


def audit_real_constant_months() -> pd.DataFrame:
    rows = []
    detail = []
    for name in MODELS:
        path = PRED / f"{name}_predictions.parquet"
        preds = pd.read_parquet(path)
        preds["formation_month"] = pd.to_datetime(preds["formation_month"])
        preds = preds[
            (preds["formation_month"] >= pd.Timestamp("2011-01-01"))
            & (preds["formation_month"] <= MAX_FORMATION_MONTH)
        ]
        n_const = 0
        n_tie_heavy = 0
        for dt, g in preds.groupby("formation_month", sort=True):
            g = g.dropna(subset=["prediction", "realized_ret"])
            if len(g) < 20:
                continue
            const = (
                int(g["prediction"].nunique(dropna=True)) < 2
                or float(g["prediction"].std(ddof=0) or 0.0) == 0.0
            )
            frac, note = _top_decile_tie_frac(g["prediction"])
            tie_heavy = bool(frac == frac and frac > 0.10)
            if const:
                n_const += 1
            if tie_heavy:
                n_tie_heavy += 1
            if const or tie_heavy:
                # how top decile chosen
                if const:
                    how = (
                        "all predictions identical — _decile_labels uses rank(method='first') "
                        "so top decile is the first ~10% of rows in groupby order (arbitrary)"
                    )
                else:
                    how = note + f"; {frac:.1%} of names share the cutoff score"
                detail.append(
                    {
                        "model": name,
                        "formation_month": str(pd.Timestamp(dt).date()),
                        "constant_pred": const,
                        "tie_frac_at_cutoff": frac,
                        "tie_break": how,
                    }
                )
        rows.append(
            {
                "model": name,
                "n_constant_pred_months": n_const,
                "n_tie_heavy_months": n_tie_heavy,
                "n_flagged": n_const + n_tie_heavy,
            }
        )
    counts = pd.DataFrame(rows)
    counts.to_csv(REPORT / "real_constant_months.csv", index=False)
    if detail:
        pd.DataFrame(detail).to_csv(REPORT / "real_constant_months_detail.csv", index=False)
    else:
        # empty marker
        pd.DataFrame(
            columns=["model", "formation_month", "constant_pred", "tie_frac_at_cutoff", "tie_break"]
        ).to_csv(REPORT / "real_constant_months_detail.csv", index=False)
    return counts


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    print("Re-scoring leakage gate…")
    summary, passed, failures = rescore_leakage()
    print(summary.to_string(index=False))
    print("GATE", "PASSED" if passed else f"FAILED: {failures}")

    print("\nAuditing real-run constant / heavy-tie months…")
    counts = audit_real_constant_months()
    print(counts.to_string(index=False))
    if int(counts["n_flagged"].sum()) == 0:
        print("ZERO flagged months for every model — no portfolio rerun needed.")
    else:
        print("Flagged months present — see real_constant_months_detail.csv")

    (REPORT / "ic_metric_fix_note.json").write_text(
        json.dumps(
            {
                "tuning_used_same_skip": True,
                "tuning_note": (
                    "WalkForwardEngine._tune calls monthly_rank_ic, which previously "
                    "dropped constant-pred validation months. Not retuned this task; "
                    "validation predictions were not saved, so per-year affected-month "
                    "counts cannot be recovered without refitting."
                ),
                "gate_passed": passed,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
