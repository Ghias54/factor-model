# Project plan and status

Status key: [ ] not started · [~] in progress · [x] done

Locked decisions: `docs/design-decisions-2026-10-20.md` → `configs/default.yaml` / `docs/decisions.md`.

## Phase 0 — Setup
- [x] Data source: FMP Ultimate (one month), no WRDS
- [x] stock-database: point-in-time Postgres on tradinghost
- [x] Repo created, structure, Cursor rules, README
- [x] Data audit (`docs/data-audit.md`, `docs/data-dictionary.md`); loaders in `src/factor_model/data/`
- [x] Design decisions locked 2026-10-20
- [x] Merge `docs/data-audit` into main; merge stock-database `load/gap-pull-1016` into main

## Build order (after cleanup)

### 1. Monthly universe + 26-feature panel (2006-01 → 2026-08)
- [ ] Monthly universe: top 1000 by `close × diluted shares` with `close_raw` ≥ $5, re-ranked monthly; membership = non-suspect price at t
- [ ] Panel: month-end prices, mcap, 1-month forward return (target rank), partial delisting returns
- [ ] Point-in-time fundamentals on `date_available` (guarded zero-lag; staleness ≤ 548 days)
- [ ] All 26 features (classic 4 + 22 extras); SUE from `earnings_event` with `report_date < t`, ≤ 120 days
- [ ] Preprocessing: cross-sectional rank to [-1, 1]; missing → 0
- [ ] Panel sanity report: counts per month, missingness, return distribution
- **Note:** universe is **not** survivorship-free before ~2021 (FMP lacks pre-2016 delistings)

### 2. Classic four factors + Fama–French check
- [ ] Value (book-to-market), Size (log mcap), Momentum (12-1), Quality (ROE)
- [ ] Single-factor decile sorts and long-short returns (diagnostics only; traded portfolio is long-only)
- [ ] Correlate with Ken French HML, SMB, UMD/MOM, RMW (expect ≳ 0.6)

### 3. Models and walk-forward
- [ ] Baseline: equal-weight composite of the classic four
- [ ] Lasso, Ridge, Random Forest, XGBoost, MLP (dropout + L2 + early stopping)
- [ ] Ensemble: mean of Lasso, XGBoost, MLP
- [ ] Expanding window; refit + retune every January; validation = last 3 years before each refit
- [ ] First OOS prediction 2011-01; OOS through 2023-12; holdout untouched

### 4. Portfolios, evaluation, interpretability, regimes
- [ ] Long-only top decile (100 names), equal-weight, hold 1 month
- [ ] Benchmark: equal-weight top-1000; secondary VW universe and S&P 500
- [ ] Metrics: ann. return/vol/Sharpe, max DD, turnover, excess vs benchmark, TE, IR; Sharpe after 10 bps/side
- [ ] Alpha: regress long portfolio on FF5 + momentum
- [ ] Diagnostics: top-minus-bottom spread; monthly rank IC
- [ ] Interpretability: SHAP (RF, XGBoost), permutation importance (MLP), coefficients (Lasso/Ridge)
- [ ] Regimes: COVID 2020-02→12, rate-hike 2022, survivorship split 2011–2020 vs 2021–2023

### 5. Holdout, once
- [ ] Evaluate 2024-01 → 2026-08 once, after all fitting and selection are frozen
- [ ] Write-up / presentation

## Not done / out of scope
- No live or paper trading; no intraday
- No shorting, borrow costs, capacity, or portfolio optimization beyond equal-weight deciles
- No autoencoders; no rolling windows; no monthly retuning
- No analyst-estimate / dispersion / revision features (`analyst_estimate` is not PIT)
- No sector neutrality (sector/industry are today's values)
- No robustness universes (top 500 / all ≥$5)
- No new FMP pulls; no pre-2005 refetch; no stock-database zero-lag repair (loaders guard it)
