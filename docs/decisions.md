# Decisions log

Locked decisions from the 2026-10-20 review. Newest first within each day.
Source of truth for the review itself: `docs/design-decisions-2026-10-20.md`.
Config mirror: `configs/default.yaml`.

| Date | Decision | Reason | Source |
|------|----------|--------|--------|
| 2026-10-20 | No new FMP pulls; use stockdb as of 2026-10-15 | Subscription is done; gap pull already loaded | design-decisions §1 |
| 2026-10-20 | Skip pre-2005 price refetch for the 602 truncated delisted names | Irrelevant with a 2006 sample start | design-decisions §1 |
| 2026-10-20 | Skip stock-database zero-lag fundamentals repair for now | Factor-model loaders already guard with `fundamentals(guard=True)` | design-decisions §1 |
| 2026-10-20 | Sample start 2006-01-31, end 2026-08-31 | First full year after the 2005-09 price crawl floor; end is last month-end with a complete t+1 return | design-decisions §2; data-audit |
| 2026-10-20 | Holdout 2024-01-31 → 2026-08-31 (32 months); never fit or tune on it | Cleanest era for a one-shot final evaluation | design-decisions §2 |
| 2026-10-20 | First out-of-sample prediction 2011-01-31 | Expanding window needs ~5 years of training + 3-year validation before first OOS | design-decisions §6 |
| 2026-10-20 | Universe = top 1000 by mcap with `close_raw` ≥ $5, re-ranked monthly | Russell-1000-like; avoids microcaps; largest clean cross-section for ML. **Not survivorship-free before ~2021** — FMP lacks most pre-2016 delistings; 12-month forward delisting rate is ~0% through 2014 and only looks realistic from ~2021 (data-audit). Report 2011–2020 and 2021–2023 separately | design-decisions §3; data-audit |
| 2026-10-20 | Market cap = `close × weighted_average_shares_diluted` | Both on today's split basis; never `close_adj × shares` (that understates historical mcap) | design-decisions §3; data-audit |
| 2026-10-20 | Membership = non-suspect price at t; delisting return = partial month to last price | Price presence at t is the listing signal; take the later of vendor `last_trade_date` and last price | design-decisions §3 |
| 2026-10-20 | No robustness universes (top 500 / all ≥$5) and no sector neutrality | Keep the first pass simple | design-decisions §3 |
| 2026-10-20 | Target = cross-sectional rank of next-month `close_adj` return (exclude suspect), scaled to [-1, 1] | Rank is robust to 2008/2020 scale shifts; portfolios reported in excess of the 3-month T-bill | design-decisions §4 |
| 2026-10-20 | 26 features (classic 4 + 22 extras); fundamentals ≤548 days stale; earnings ≤120 days; info cutoff 16:00 ET on t | Covers value/size/mom/quality plus valuation, risk, liquidity, profitability, growth, balance sheet, SUE. Excludes non-PIT analyst estimates, revision trends (start 2016), and today's sector | design-decisions §5 |
| 2026-10-20 | Preprocessing: monthly cross-sectional rank to [-1, 1]; missing → 0 | Gu–Kelly–Xiu style; ranks replace winsorizing | design-decisions §5 |
| 2026-10-20 | Models: classic-4 EW composite, Lasso, Ridge, RF, XGBoost, MLP, mean ensemble of Lasso/XGB/MLP | Baseline + linear + trees + neural + simple ensemble | design-decisions §6 |
| 2026-10-20 | Expanding window; refit + retune every January; validation = last 3 years before each refit | Yearly cadence matches typical factor research; 3-year val avoids shuffled k-fold | design-decisions §6 |
| 2026-10-20 | Long-only top decile (100 names), equal-weight, hold 1 month; no shorting | Club project is long-only; equal weight is transparent and matches the benchmark | design-decisions §7 |
| 2026-10-20 | Benchmark = equal-weight top-1000 universe; secondary VW universe and S&P 500 | Matches the traded portfolio construction | design-decisions §7 |
| 2026-10-20 | Cost sensitivity 10 bps/side (gross and net reported) | Config assumption, not hard-coded | design-decisions §7 |
| 2026-10-20 | Regimes: COVID 2020-02→12, rate-hike 2022, survivorship split 2011–2020 vs 2021–2023; holdout separate | Isolates known market regimes and the vendor survivorship break | design-decisions §7 |
| 2026-10-20 | Merge `docs/data-audit` into FE-Factor-Model main; merge stock-database `load/gap-pull-1016` into main | Approved by Rehan 2026-10-20; unlocks db.py/loaders and records the already-applied prod load | design-decisions §8 |
| 2026-10-06 | Separate repo from `stock-database` | Keep the data pipeline and the research code independent; this repo reads the DB only | repo setup |
| 2026-10-06 | Monthly frequency, walk-forward validation | Required by the FE Club project spec | repo setup |
