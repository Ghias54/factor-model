# Holdout evaluation — pre-registration

**Date:** 2026-11-10  
**Code commit (main, pre-holdout):** `6fff7a54e0b1fe761caaa5dc4caa5ac7c1413a4d`  
**Branch:** `feat/holdout`  
**Authority:** Rehan decision 2026-11-10 — one-time holdout; same yearly January refit-and-retune as the OOS backtest. Results reported once, as they come out.

Nothing about the models changes in this evaluation: same grids, seeds, features, preprocessing, and portfolio rules as the locked OOS run. If a bug that affects results is found, stop and report; do not fix and rerun.

33 formation months is too short for statistical significance. Results are directional.

---

## Window

| | 33-formation window | 32-formation window |
|---|---|---|
| Formation months | 2023-12-29 → 2026-08-31 | 2024-01-31 → 2026-08-31 |
| Return months | 2024-01 → 2026-09 | 2024-02 → 2026-09 |
| Count | 33 | 32 |

2023-12 is included in the 33-month window because it was excluded from the OOS window by the formation cap at 2023-11 and must be evaluated somewhere. Both windows are reported.

Panel end / last complete t→t+1 month: formation 2026-08-31.

---

## Refit schedule

| Refit (January of year Y) | Train formations through | Validation | Predicts |
|---|---|---|---|
| 2023 fit (already chosen in OOS) | Dec 2022 | last 36 train months | **2023-12 only** (holdout addition) |
| Jan 2024 | Dec 2023 | last 36 train months | all 2024 formations |
| Jan 2025 | Dec 2024 | last 36 train months | all 2025 formations |
| Jan 2026 | Dec 2025 | last 36 train months | 2026-01 → 2026-08 |

Expanding window. Hyperparameters selected by mean monthly rank IC on the validation tail; chosen params then refit on the full training window. MLP uses 5 seeds (unchanged). Ensemble = within-month rank-average of Lasso / XGBoost / MLP, dropping constant members for that month (same fallback as OOS).

---

## Portfolios evaluated (8)

1. **EW universe** — equal-weight top-1000 universe  
2. **classic-4** — equal-weight composite of value, size (flipped), momentum 12-1, quality ROE  
3. **Lasso**  
4. **Ridge**  
5. **Random Forest**  
6. **XGBoost**  
7. **MLP** (5 seeds)  
8. **Ensemble** — rank-average of Lasso, XGBoost, MLP  

Portfolio rule (unchanged): long-only top decile (100 names), equal-weighted, hold 1 month. Constant-prediction months → equal-weight universe for that month.

---

## Metrics (same family as `reports/models/grand_metrics.csv`)

- Annualized return, volatility, Sharpe (excess of 3-month T-bill / FF RF)  
- Max drawdown  
- Turnover (mean one-way)  
- Excess vs EW universe; tracking error; information ratio  
- Net Sharpe and net ann. return at **10 bps** and **25 bps** per side  
- Mean monthly rank IC; share of months with IC > 0  
- Active FF6 alpha (portfolio − EW, regressed on FF5 + UMD) with Newey–West t; alpha uses Ken French months available only (state the count)

---

## Pre-registered questions

- **Q1:** Does the OOS ranking hold? Spearman rank correlation of model Sharpe (OOS 2011–2023 vs holdout), across the scored portfolios.  
- **Q2:** Does any model beat classic-4 and the EW universe **net of 10 bps**?  
- **Q3:** Does the MLP remain the best (by holdout Sharpe / net Sharpe)?  
- **Q4:** Is holdout mean rank IC positive for each model (Lasso, Ridge, RF, XGBoost, MLP, ensemble; classic-4 also reported)?

Answers will be written once in `cursor_output.md` after the single evaluation pass.
