# Holdout evaluation — cursor_output

Branch: `feat/holdout`  
Pre-registered against main `6fff7a54e0b1fe761caaa5dc4caa5ac7c1413a4d`  
Holdout wall time: **1468.8 s** (~24.5 min). RF 754 s, MLP 685 s, XGB 14 s, linear < 10 s.  
Tests: **51 passed** (on branch and again on main after merge). Merge SHA: `d16df4817eea4514e1df4498205603a8afc2c70c`.

Ken French FF5+UMD refreshed; last month **2026-09**. Active FF6 alpha uses **all 33** holdout return months.

---

## 1. Verdict (pre-registered questions)

**Q1 — Does the OOS ranking hold?** No. Spearman rank correlation of model Sharpe (OOS 2011–2023 vs holdout 33) across classic-4 / Lasso / Ridge / RF / XGB / MLP / ensemble is **−0.36**. The OOS order inverted: classic-4 jumped from mid-pack to first; MLP fell from first to near the bottom of the ML pack.

**Q2 — Does any model beat classic-4 and the EW universe net of 10 bps?** No. On the 33-formation window, net Sharpe 10 bps: classic-4 **0.82**, EW **0.71**, ensemble **0.67**, MLP **0.52**, Lasso **0.50**. No ML portfolio beats both baselines net of cost. Classic-4 itself does beat EW.

**Q3 — Does the MLP remain the best?** No. Classic-4 has the highest holdout Sharpe (0.87) and net-10 Sharpe (0.82). Among ML models the ensemble is best (Sharpe 0.76); MLP is fifth of six ML names.

**Q4 — Is holdout rank IC positive for each model?** On the **33-formation** window (incl. 2023-12): yes for classic-4 and every ML model (means from 0.0015 Ridge to 0.020 classic-4). On the **32-formation** window (from 2024-01): Lasso, Ridge, and RF flip slightly negative; XGB / MLP / ensemble / classic-4 stay positive. Directional only — 33 months is too short for significance.

---

## 2. Holdout head-to-head (33 formations: 2023-12 → 2026-08)

| Portfolio | Ann. ret | Sharpe | Net 10 bps | Net 25 bps | Excess vs EW | Rank IC | IC>0 share | Turnover | Max DD | Active FF6 α | α t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EW universe | 12.8% | 0.72 | 0.71 | 0.69 | — | — | — | 0.057 | −9.5% | — | — |
| classic-4 | 17.9% | 0.87 | 0.82 | 0.75 | +4.9% | 0.020 | 55% | 0.303 | −12.2% | +2.3% | 1.17 |
| Lasso | 11.8% | 0.59 | 0.50 | 0.36 | −1.0% | 0.004 | 42% | 0.491 | −10.8% | −3.9% | −0.94 |
| Ridge | 10.5% | 0.51 | 0.41 | 0.28 | −2.1% | 0.001 | 42% | 0.492 | −11.3% | −4.4% | −0.96 |
| Random Forest | 13.3% | 0.64 | 0.57 | 0.47 | +0.6% | 0.007 | 42% | 0.398 | −11.6% | −1.1% | −0.28 |
| XGBoost | 12.5% | 0.61 | 0.52 | 0.38 | −0.2% | 0.015 | 45% | 0.512 | −11.2% | −3.4% | −1.02 |
| MLP | 12.0% | 0.60 | 0.52 | 0.41 | −0.7% | 0.013 | 48% | 0.430 | −11.7% | −3.8% | −0.86 |
| Ensemble | 13.9% | 0.76 | 0.67 | 0.54 | +0.9% | 0.013 | 39% | 0.465 | −11.2% | −0.8% | −0.16 |

32-formation window (drop 2023-12): same qualitative story; classic-4 still leads; several ML ICs go ≤ 0. Full tables in `reports/holdout/holdout_metrics.csv`. Chart: `reports/holdout/holdout_cumulative_net10.png`.

---

## 3. Yearly holdout Sharpe (by return-month year)

| Portfolio | 2024 | 2025 | 2026 YTD |
|---|---:|---:|---:|
| EW universe | 0.64 | 0.76 | 0.75 |
| classic-4 | 0.88 | 0.74 | 0.95 |
| Lasso | 0.74 | 0.33 | 0.64 |
| Ridge | 0.72 | 0.29 | 0.39 |
| Random Forest | 0.76 | −0.08 | 1.14 |
| XGBoost | 0.82 | −0.05 | 0.89 |
| MLP | 0.63 | −0.04 | 1.25 |
| Ensemble | 0.82 | 0.00 | 1.44 |

2025 was the weak year for trees/MLP; 2026 YTD recovered. Detail: `reports/holdout/holdout_by_year.csv`.

---

## 4. OOS (2011–2023) vs holdout (33)

| Portfolio | OOS Sharpe | Holdout Sharpe | OOS net 10 | Holdout net 10 | OOS excess | Holdout excess | OOS IC | Holdout IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EW universe | 0.56 | 0.72 | 0.56 | 0.71 | — | — | — | — |
| classic-4 | 0.62 | 0.87 | 0.58 | 0.82 | +1.7% | +4.9% | 0.019 | 0.020 |
| Lasso | 0.70 | 0.59 | 0.64 | 0.50 | +3.0% | −1.0% | 0.026 | 0.004 |
| Ridge | 0.64 | 0.51 | 0.57 | 0.41 | +2.5% | −2.1% | 0.028 | 0.001 |
| Random Forest | 0.59 | 0.64 | 0.54 | 0.57 | +1.7% | +0.6% | 0.033 | 0.007 |
| XGBoost | 0.62 | 0.61 | 0.56 | 0.52 | +1.9% | −0.2% | 0.034 | 0.015 |
| MLP | 0.77 | 0.60 | 0.71 | 0.52 | +4.6% | −0.7% | 0.029 | 0.013 |
| Ensemble | 0.75 | 0.76 | 0.69 | 0.67 | +4.4% | +0.9% | 0.033 | 0.013 |

Spearman (Sharpe ranks, 7 scored portfolios): **−0.36**.

---

## 5. Full period 2011-01 → 2026-08 (188 return months)

| Portfolio | Ann. ret | Sharpe | Net 10 | Excess vs EW | Rank IC | Active FF6 α | α t |
|---|---:|---:|---:|---:|---:|---:|---:|
| EW universe | 9.7% | 0.58 | 0.58 | — | — | — | — |
| classic-4 | 12.0% | 0.66 | 0.62 | +2.3% | 0.019 | +0.5% | 0.55 |
| Lasso | 12.2% | 0.68 | 0.62 | +2.3% | 0.022 | +0.7% | 0.52 |
| Ridge | 11.3% | 0.62 | 0.55 | +1.6% | 0.023 | −0.3% | −0.19 |
| Random Forest | 11.2% | 0.60 | 0.54 | +1.5% | 0.029 | −0.6% | −0.36 |
| XGBoost | 11.2% | 0.60 | 0.53 | +1.5% | 0.031 | −0.4% | −0.27 |
| MLP | 13.6% | 0.74 | 0.68 | +3.6% | 0.026 | +1.7% | 1.07 |
| Ensemble | 13.7% | 0.75 | 0.68 | +3.8% | 0.029 | +1.7% | 1.13 |

Over the full sample, ensemble and MLP still lead; the holdout alone does not. Chart with holdout marker: `reports/holdout/full_period_cumulative_net10.png`.

---

## 6. Holdout feature / driver drift

**Coverage:** No feature failed the pre-run drop screen. Holdout non-null shares are at or above the 2011–2023 averages for every feature (lowest holdout coverage: delta_roe_4q ≈ 87%). See `reports/holdout/feature_coverage.csv`.

**Lasso vs 2023:** 2023 had 26 non-zero coefs. 2024 dropped accruals + turnover (one sign flip: debt_to_assets). 2025 kept 26 non-zeros (sign flips: debt_to_assets, turnover). 2026 sparsified to 17 non-zeros (dropped value_bm, earnings_yield, dollar_volume_ln, turnover, accruals, asset_growth_yoy, current_ratio, debt_to_assets, gross_margin). Details: `reports/holdout/lasso_vs_2023.csv`.

**XGBoost SHAP:** Holdout mean |SHAP| feature ranks vs OOS ranks Spearman **+0.87** — drivers did **not** shift much; importance ranking is stable even though portfolio Sharpe ranking is not.

---

## 7. Pre-run issues

None. Panel: 33 holdout formations, 1,000 names each, max ret_1m null share 0.3%. FF through 2026-09. Feature coverage clean. Proceeded to Step 3.

---

## 8. Tests, merge, commits, not done

**Tests:** `pytest` → **51 passed** (was 45; +6 holdout-guard tests).

**Commits on `feat/holdout`:**
1. `3280151` Pre-register holdout evaluation window, refits, and questions.
2. `cdf891b` Add holdout entry point and keep development paths capped.
3. (this commit) Holdout results, reports, cursor_output.

**Merge:** Rehan authorized in the task (Step 7) if tests green. Merged with `--no-ff` into main; merge SHA **`d16df4817eea4514e1df4498205603a8afc2c70c`**. Main retested (51 passed) and pushed.

**Not done / skipped:**
- No model/grid/feature/threshold changes (by design).
- No bug fixes to modeling code.
- No statistical significance tests / bootstrap CIs on the holdout (33 months; directional only).
- Did not re-run OOS walk-forward; reused existing OOS predictions and the locked 2023 hyperparameters for the 2023-12 formation.
- Logs under `logs/` not committed.

**Judgment calls:**
- 2023-12 predictions use OOS 2023 chosen hyperparameters (not retuned), matching the task’s “2023’s fit predicts 2023-12.”
- Primary answers use the 33-formation window; 32-formation reported alongside.
- “Beat classic-4 and EW” interpreted as net-10 Sharpe strictly greater than both.
- Active FF6 alpha = (portfolio − EW) on FF5+UMD, same as OOS grand_metrics.
