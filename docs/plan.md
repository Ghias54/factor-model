# Project plan and status

Status key: [ ] not started · [~] in progress · [x] done

## Phase 0 — Setup
- [x] Data source: FMP Ultimate (one month), no WRDS
- [x] stock-database: point-in-time Postgres on tradinghost
- [x] Repo created, structure, Cursor rules, README
- [ ] Decide open questions below

## Phase 1 — Universe and panel
- [ ] Monthly universe: US common stock, survivorship-bias free, liquidity/price filters
- [ ] Monthly panel: month-end price, market cap, 1-month forward return (target), delisting returns
- [ ] Point-in-time fundamentals joined on `date_available`
- [ ] Panel sanity report: counts per month, missingness per column, return distribution

## Phase 2 — Classic factors
- [ ] Value (book-to-market, earnings yield)
- [ ] Size (log market cap)
- [ ] Momentum (12-1 month return)
- [ ] Quality (ROE / gross profitability)
- [ ] Single-factor decile sorts and long-short returns
- [ ] Validate against Ken French factors (HML, SMB, UMD/MOM, RMW) — correlation check

## Phase 3 — Additional features (15–25)
- [ ] Earnings surprise and estimate revisions
- [ ] Analyst estimate dispersion
- [ ] Volatility, beta, idiosyncratic volatility
- [ ] Short-term reversal (1-month)
- [ ] Multi-horizon momentum (3, 6, 12 month)
- [ ] Accruals, asset growth
- [ ] Profitability trends (changes, not just levels)
- [ ] Gross margin, revenue growth, leverage/debt ratios
- [ ] Liquidity / turnover / Amihud illiquidity
- [ ] Cross-sectional preprocessing: rank-normalize, missing indicators

## Phase 4 — Models
- [ ] Baseline: equal-weight composite of the four factors
- [ ] Lasso and Ridge (feature selection from Lasso)
- [ ] Random Forest (+ XGBoost/LightGBM as an extension)
- [ ] MLP with dropout and weight decay
- [ ] Ensemble (average of Lasso, tree model, MLP) — optional

## Phase 5 — Walk-forward and backtests
- [ ] Expanding-window walk-forward engine, time-ordered CV for tuning
- [ ] Monthly decile long-short portfolios per model
- [ ] Metrics: annualized return, vol, Sharpe, max drawdown, turnover, net of costs

## Phase 6 — Analysis
- [ ] Feature importance: Lasso coefficients, SHAP for trees, permutation importance for MLP
- [ ] Do model classes agree on which features matter?
- [ ] Regime analysis: 2020 COVID crash/recovery vs. 2022 rate-hike bear market
- [ ] Final holdout evaluation (once)
- [ ] Write-up / presentation

## Open questions
- Universe: all US common stock with filters, or S&P 500 / Russell 1000 style (top N by market cap)?
- Sample period start date, and the final holdout period
- Target: raw return, excess return over market, or cross-sectional rank
- Portfolio weighting: equal-weight vs. value-weight
- Transaction cost assumption (bps per side)
- Work split across teammates

## Not done / out of scope for now
- No live trading or paper trading
- No intraday data
- Autoencoder models (Gu, Kelly & Xiu 2021) — stretch goal only
