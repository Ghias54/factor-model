# Cursor Output — walk-forward Lasso / Ridge

Date: 2026-11-08  
Branch: `feat/walkforward-linear` → merged to main as **`f78c5cf`**

## Summary

Walk-forward engine is in place (yearly January refits, 36-month validation by mean rank IC, holdout-capped at formation 2023-11). Target = `2 * target_rank − 1` (matches feature [−1, 1] scale). Over OOS formation **2011-01 → 2023-11**, **Lasso and Ridge both beat the classic-4 composite and the EW universe** on Sharpe (0.67 / 0.64 vs 0.62 / 0.56). Sharpe gaps vs classic-4 are positive but the 12-month block-bootstrap 95% CIs include zero. **36 tests passed** (timing, holdout, leakage shuffle mean |IC| < 0.005). Wall ~21 s, peak RSS ~0.9 GB.

## Head-to-head (OOS 2011–2023)

| Portfolio | Ann ret | Ann vol | Sharpe | Max DD | Turnover | Excess vs EW | IR | Mean rank IC | Sharpe − classic4 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EW universe | 9.1% | 16.2% | 0.56 | −26% | — | — | — | — | — |
| classic-4 composite | 10.8% | 17.7% | 0.62 | −31% | 0.28 | +1.7% | 0.34 | 0.019 | — |
| Lasso | 12.1% | 18.2% | **0.67** | −29% | 0.46 | +2.9% | 0.54 | 0.026 | +0.05 [−0.22, +0.36] |
| Ridge | 11.5% | 18.4% | **0.64** | −29% | 0.48 | +2.5% | 0.49 | 0.028 | +0.02 [−0.22, +0.29] |

Excess vs EW (ann. mean of monthly active), 95% block-bootstrap CI: Lasso +2.9% [in summary]; Ridge +2.5%.

## Excess vs EW by year (ann. mean × 12)

| Year | classic4 | Lasso | Ridge |
|---:|---:|---:|---:|
| 2011 | +2.9% | −4.8% | −5.0% |
| 2012 | +4.4% | +9.2% | +8.6% |
| 2013 | +5.6% | +1.0% | +3.0% |
| 2014 | −5.1% | +0.8% | +3.6% |
| 2015 | +6.0% | +1.6% | +2.6% |
| 2016 | −0.9% | −4.0% | −6.0% |
| 2017 | −1.9% | +4.7% | +4.8% |
| 2018 | −4.8% | +4.4% | −4.1% |
| 2019 | −1.8% | +5.5% | +7.4% |
| 2020 | −3.4% | +11.6% | +8.6% |
| 2021 | +8.3% | +7.3% | +6.1% |
| 2022 | +11.0% | −1.3% | +0.1% |
| 2023 | +3.5% | +3.0% | +3.1% |

### Regimes

| | COVID 2020-02→12 Sharpe | Rate-hike 2022 Sharpe |
|---|---:|---:|
| EW universe | 0.69 | −0.86 |
| classic-4 | 0.52 | −0.33 |
| Lasso | **0.98** | −0.71 |
| Ridge | **0.91** | −0.68 |

## Lasso selection

Stable positive every year: cashflow_to_price, beta_60m, reversal_36_13, gross_profit_to_assets.  
Stable negative every year: max_daily_ret_1m.  
High survival (≥75% of years): accruals (−), momentum_12_1 (+ after early years), amihud (−), sue (+), sales_to_price.

**Sign flips across years:** accruals, asset_growth_yoy, gross_margin, momentum_12_1, momentum_6m, sales_to_price, size_ln_mcap, turnover, volatility_12m.

Top |coef| (Lasso avg): amihud, gross_profit_to_assets, beta_60m, dollar_volume_ln, cashflow_to_price.  
Lasso vs Ridge importance rank correlation (Spearman): **0.95**.

## Timing checks

All 13 refit years (2011–2023): max train formation = Dec of Y−1 < first prediction of Y; last 2023 prediction = 2023-11-30. See `reports/linear/timing_checks.csv`.

## Engine interface (for RF / XGB / MLP next)

```text
BaseModel (src/factor_model/models/base.py)
  name: str
  param_grid: dict[str, list]
  fit(X, y, feature_names=None) -> self
  predict(X) -> ndarray
  clone(**params) -> BaseModel
  coefficients_: optional pd.Series after fit

WalkForwardEngine.run(model, panel) -> WalkForwardResult
  predictions parquet: security_id, formation_month, prediction, realized_ret, …
  year_fits JSON: chosen params, validation IC, coefficients
```

## Saved where

| Path | Contents |
|---|---|
| `data/predictions/lasso_predictions.parquet` | OOS predictions |
| `data/predictions/ridge_predictions.parquet` | OOS predictions |
| `data/predictions/fits/*_year_fits.json` | Per-year params / coefs / val IC |
| `reports/linear/` | metrics, regimes, selection tables, cumulative chart |
| `src/factor_model/walkforward/` | engine |
| `src/factor_model/models/` | BaseModel, Lasso, Ridge |

## Not done / skipped

- RF, XGBoost, MLP, ensemble (next task).  
- No holdout 2024+ evaluation.  
- Did not retune grids to chase Sharpe (alphas chosen only by validation IC).  
- Prediction parquets under gitignored `data/`; reports CSVs/PNG committed.
