# cursor_output — Lasso corrected-IC retrain + merge

**Verdict:** merged **yes** · `main` @ **7ca3f49**

## 1. Step 1 — other models' tuning

| Model | Any constant val month (any grid point) | Params changed? |
|---|---|---|
| Ridge | no | no |
| XGBoost | no | no |
| Random forest | no | no |
| MLP | no | no |

No unexpected retrain of other models.

## 2. Lasso before / after

2011 α → **10** (confirmed). 2012–2023 alphas and coefficients match prior fits. 12 OOS months with constant predictions (all 2011) → portfolio = EW those months; IC = 0.

| | Sharpe | Net 10bps | Excess vs EW | Active α | Active t | 2011 excess |
|---|---:|---:|---:|---:|---:|---:|
| Lasso before | 0.668 | 0.608 | +2.94% | +0.87% | 0.57 | −4.79% |
| Lasso after | **0.698** | **0.640** | +2.99% | +1.10% | 0.82 | **0.0%** |
| Ensemble before | 0.749 | 0.688 | +4.35% | +1.94% | 1.33 | −3.52% |
| Ensemble after | 0.750 | 0.689 | +4.38% | +1.93% | 1.31 | −3.17% |

Ensemble dropped Lasso in 12 months (rank-average of XGB+MLP only).

## 3. Head-to-head (updated)

| Portfolio | Sharpe | Net 10bps | Excess vs EW | Active α | Active t |
|---|---:|---:|---:|---:|---:|
| EW universe | 0.564 | 0.560 | — | — | — |
| Classic-4 | 0.617 | 0.579 | +1.70% | +0.44% | 0.37 |
| Lasso | 0.698 | 0.640 | +2.99% | +1.10% | 0.82 |
| Ridge | 0.636 | 0.573 | +2.46% | +0.37% | 0.26 |
| RF | 0.593 | 0.537 | +1.73% | −0.44% | −0.27 |
| XGBoost | 0.621 | 0.560 | +1.89% | +0.19% | 0.14 |
| MLP | 0.771 | 0.710 | +4.57% | +2.30% | 1.41 |
| Ensemble | 0.750 | 0.689 | +4.38% | +1.93% | 1.31 |

## 4. Tests / merge

**45 passed** on the branch and again on `main` after merge (`7ca3f49`). Push to `origin/main` succeeded.

## 5. Not done / skipped

- Did not retrain Ridge/RF/XGB/MLP (verification found no constant validation months).
- MLP verification used `n_seeds=1` only to detect flat outputs (production still 5).
