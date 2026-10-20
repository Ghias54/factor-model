# Design decisions — 2026-10-20

Locked by Rehan in a decision-by-decision review. Source of truth for configs/default.yaml and docs/decisions.md. Based on docs/data-audit.md (2026-10-06) and the stock-database gap pull through 2026-10-15.

## 1. Data
- No new FMP pulls. Use stockdb as it stands (prices through 2026-10-15).
- Skip the pre-2005 price refetch for the 602 delisted names: irrelevant with a 2006 start.
- Skip the stock-database zero-lag fundamentals repair for now; the factor-model loaders guard it (`fundamentals(guard=True)`).

## 2. Sample
- Start 2006-01-31. End 2026-08-31 (last month-end with a complete t+1 month).
- Holdout 2024-01-31 → 2026-08-31 (32 months). Never used in fitting or tuning; evaluated once at the end.
- Survivorship: FMP lacks most pre-2016 deaths. Not "survivorship-free". Report results separately for 2011–2020 and 2021–2023.

## 3. Universe (kept simple)
- Each month-end t: the 1,000 largest US common stocks by market cap with `close_raw` ≥ $5, re-ranked monthly.
- Market cap = `close × weighted_average_shares_diluted` (never `close_adj × shares`).
- Membership = has a non-suspect price at t.
- Delisted mid-month: keep the final partial-month return (later of vendor `last_trade_date` and last price).
- No robustness universes (top 500 / all ≥$5) and no sector neutrality for now.

## 4. Target
- Cross-sectional rank of next-month return (t → t+1), scaled to [0,1] (or [-1,1]).
- Returns from `close_adj`, excluding `price_quality='suspect'`.
- Portfolio results reported in excess of the 3-month T-bill.

## 5. Features (26)
All computed at month-end t with data public by 16:00 ET on t. Fundamentals: quarterly rows only, as-of `date_available`, staleness ≤ 548 days; TTM flows summed from 4 quarters.

Classic four:
| Factor | Definition |
|---|---|
| Value | Book-to-market = total_equity / mcap |
| Size | log(mcap) |
| Momentum | Return t-12 → t-1 (skip most recent month) |
| Quality | ROE = TTM net income / latest total_equity; missing if equity ≤ 0 |

Extras (22):
- Valuation: earnings yield (TTM EPS / price), sales-to-price, cash-flow-to-price (TTM CFO / mcap)
- Price/risk: 1-month reversal, 6-month momentum, 36-month long-term reversal (t-36 → t-13), 12-month volatility, 60-month beta vs ^GSPC (monthly), max daily return last month
- Liquidity: turnover (volume / weighted_average_shares), dollar volume (log), Amihud illiquidity
- Profitability: gross margin, ROA, gross profit / assets, 4-quarter change in ROE
- Growth/investment: YoY revenue growth, YoY asset growth, accruals ((TTM NI − TTM CFO) / assets)
- Balance sheet: debt-to-assets, current ratio
- Earnings: SUE from `earnings_event` (eps_actual − eps_estimated, scaled by price), event `report_date < t`, ≤ 120 days old

Excluded: analyst estimates / dispersion / forward earnings (`analyst_estimate` not PIT), revision trends (`analyst_consensus_daily` starts 2016), sector (today's values).

Preprocessing: each month, rank every feature cross-sectionally and scale to [-1, 1]; missing → 0 (median). Gu-Kelly-Xiu style.

## 6. Models and validation
| Model | Tuned |
|---|---|
| Classic 4-factor equal-weight composite (baseline) | — |
| Lasso | alpha |
| Ridge | alpha |
| Random Forest | max_depth, n_estimators, min_samples_leaf |
| XGBoost | max_depth, learning_rate, n_estimators |
| MLP (2–3 layers, dropout, early stopping) | width, dropout, L2 |
| Ensemble: mean of Lasso, XGBoost, MLP predictions | — |

- Expanding window, refit + retune every January; predict that year's 12 months.
- Tuning: last 3 years before each refit are the validation set.
- First out-of-sample prediction 2011-01. OOS results 2011–2023; holdout touched once.
- Interpretability: SHAP (RF, XGBoost), permutation importance (MLP), coefficients (Lasso/Ridge). Compare agreement across models.
- Not included: autoencoders, rolling windows, monthly retuning.

## 7. Portfolios and evaluation
- **Long-only. No shorting.** Each month buy the top decile (100 stocks) by predicted rank, equal-weighted; hold 1 month.
- Benchmark: equal-weighted top-1000 universe. Secondary: value-weighted universe, S&P 500.
- Metrics: annualized return, volatility, Sharpe, max drawdown, turnover, excess return vs benchmark, tracking error, information ratio; Sharpe after 10 bps/side as sensitivity.
- Alpha: regress long portfolio on FF5 + momentum.
- Diagnostics only (not traded): top-minus-bottom decile spread; monthly rank IC between prediction and realized return.
- Fama-French validation: correlate own classic single-factor long-short replications with HML, SMB, UMD, RMW (expect ≳ 0.6).
- Regimes: COVID 2020-02 → 2020-12; rate-hike 2022; survivorship split 2011–2020 vs 2021–2023; holdout separately.
- Not included: short costs/borrow, capacity, optimization beyond equal-weight deciles.

## 8. Cleanup (before building)
- Merge `docs/data-audit` into main (approved by Rehan 2026-10-20).
- Discard the uncommitted working-tree edits to configs/default.yaml and docs/decisions.md (superseded; contain the incorrect "survivorship-free" claim).
- Merge stock-database `load/gap-pull-1016` into main (approved by Rehan 2026-10-20; already applied to prod).
- Write these decisions into configs/default.yaml and docs/decisions.md.

## Build order after cleanup
1. Monthly universe + 26-feature panel, 2006-01 → 2026-08.
2. Classic four factors + Fama-French correlation check.
3. Models and walk-forward.
4. Portfolios, evaluation, interpretability, regimes.
5. Holdout, once.
