# cursor_output — IC metric fix + leakage re-score

**Verdict:** gate passed **no** (Lasso still fails abs-IC) · merged **no**

**Branch:** `feat/nonlinear-models` (unmerged)

## 1. Corrected shuffle table

Constant-prediction months now contribute **IC = 0** (not dropped). Gate unchanged: |mean IC| < 0.005 and |t| < 2.

| Model | Shuffle mean IC (new) | Old | t (new) | Constant-pred months (sum over seeds) | Pass? |
|---|---:|---:|---:|---:|---|
| **Lasso** | **0.00580** | 0.00619 | 1.73 | **48** | **FAIL** (|IC| by +0.00080) |
| Ridge | 0.00077 | 0.00077 | 0.25 | 0 | pass |
| XGBoost | 0.00245 | 0.00245 | 1.03 | 0 | pass |
| Random forest | −0.00346 | −0.00346 | −1.04 | 0 | pass |
| MLP | −0.00122 | −0.00122 | −0.25 | 0 | pass |
| Ensemble | 0.00075 | 0.00075 | 0.18 | 0 | pass |

Only Lasso had constant-pred months under shuffle (48 across 5 seeds). Fixing the metric moved its mean from 0.00619 → 0.00580 — still above 0.005. Per task: **stop, do not merge, do not tweak the gate.**

**Tuning note:** `WalkForwardEngine._tune` uses the same `monthly_rank_ic`, so validation previously skipped constant months too. Validation predictions were not saved; per-year affected counts would need a refit — **not done** this task.

## 2. Real-run constant / heavy-tie months

| Model | Constant-pred months | Heavy-tie months (>10% at top-decile cutoff) |
|---|---:|---:|
| Lasso | 0 | 0 |
| Ridge | 0 | 0 |
| XGBoost | 0 | **10** (all in 2011) |
| RF / MLP / ensemble | 0 | 0 |

**Tie-break:** `_decile_labels` does `rank(method="first")` then `qcut` — ties broken by **row order within the month group**.

**XGBoost rerun** (those 10 months → EW universe return / turnover; IC unchanged):

| | Ann ret | Sharpe | Excess vs EW | Net Sharpe 10bps | Active α ann | Active α t |
|---|---:|---:|---:|---:|---:|---:|
| Before | 10.94% | 0.601 | +1.92% | 0.539 | +0.03% | 0.02 |
| After | 11.01% | 0.621 | +1.89% | 0.560 | +0.19% | 0.14 |

Updated `grand_metrics.csv` and `active_alpha.csv` for xgboost only. Side-by-side: `xgboost_tie_month_rerun.csv`.

## 3. Tests

**45 passed** (`pytest -q`). New unit test: `tests/test_ic_constant_month.py`.

## 4. Not done / skipped

- **Merge** blocked by Lasso gate failure (0.00580 ≥ 0.005).
- No retune / no retrain of walk-forward models.
- Validation constant-month counts not recoverable without refit.
