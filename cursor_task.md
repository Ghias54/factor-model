# Cursor Task - FE-Factor-Model
Date: 2026-10-20 13:41:54
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/panel
Priority: HIGH

## Context
All design decisions are locked in docs/design-decisions-2026-10-20.md and configs/default.yaml (on chore/lock-decisions). Read both, plus docs/data-dictionary.md and docs/data-audit.md, before writing code. Use only the existing read-only loaders in src/factor_model/data/ (extend them if needed); never use stock-database's monthly_panel builder (wrong mcap, no zero-lag guard). Rehan approved merging chore/lock-decisions on 2026-10-20.

## Task
0. Merge `chore/lock-decisions` into main (Approved by Rehan 2026-10-20): `git checkout main && git pull && git merge --no-ff chore/lock-decisions`, run pytest, push. Then `git checkout -b feat/panel`.

1. Universe (src/factor_model/universe/): for each month-end t from 2006-01-31 to 2026-08-31 (use loaders.month_ends(complete_only=True)), take securities with a non-suspect price at t, `close_raw` ≥ 5, mcap = `close × weighted_average_shares_diluted` (as-of date_available with the zero-lag guard; fall back to vendor market_cap_daily only for ranking if shares missing, and count how often), rank by mcap, keep top 1000. Output data/panel/universe.parquet (date, security_id, symbol at t, mcap, rank).

2. Target: next-month return t→t+1 from close_adj (exclude suspect rows). If a name delists during t+1, use the return to its final price (later of vendor last_trade_date and last price). Also store raw return, excess return over month3 T-bill, and cross-sectional rank of next-month return scaled to [0,1].

3. Features (src/factor_model/features/): compute the 26 features exactly as defined in docs/design-decisions-2026-10-20.md §5, all point-in-time at 16:00 ET on t. Fundamentals: quarterly rows only, asof date_available, staleness ≤ 548 days, TTM = sum of last 4 quarters (require 4 consecutive quarters, else NaN). ROE NaN if equity ≤ 0. SUE from earnings_event with report_date < t, ≤ 120 days old, drop 1970-01-01 junk and future rows. Price features from daily close_adj; beta = 60-month monthly regression on ^GSPC (min 24 months). Store raw feature values in data/panel/features_raw.parquet.

4. Preprocessing: each month, rank each feature cross-sectionally, scale to [-1,1], missing → 0. Save data/panel/panel.parquet (date, security_id, 26 processed features, target columns). Keep raw and processed separate.

5. Checks, written to docs/panel-report.md:
   - Stocks per month (should be 1000; list any month below).
   - Per-feature coverage (% non-missing raw) by year, and flag any feature < 70% in any year.
   - Lookahead tests: for a random sample of 200 (security, month) rows assert every fundamental used has date_available ≤ 16:00 ET on t and every earnings event report_date < t. Add these as unit/dev tests in tests/.
   - Sanity: summary stats per raw feature (median, p1, p99); cross-check mcap vs vendor market_cap_daily (median ratio); spot-check AAPL and SIVB rows (SIVB present until 2023-02, absent after).
   - Target sanity: mean monthly return of the universe vs ^GSPC monthly return correlation.

6. Performance: maximize tradinghost compute — batched DB reads (one query per table), vectorized pandas/polars, parallelize per-month or per-feature work across all cores. Report wall time and peak memory. Mind disk (~6.9 GB free): parquet with compression only, nothing written to the database.

7. Do NOT build the classic factor portfolios, FF check, or any models in this task.

Rules: point-in-time only; no survivorship shortcuts (never use today's lists); commit as you go; push feat/panel; do NOT merge feat/panel (human merges). data/ stays gitignored.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b feat/panel from main (or continue it if it exists)
- Commit as you go. Push the branch. Do NOT merge to main - the human merges one by one
- Point-in-time only: every feature at month-end t must use data available as of t
  (fundamentals on filing/acceptance date, not period end; no restated values)
- No survivorship bias: build the universe from historical constituents, never today's list
- Walk-forward only: train through month t, predict t+1; never fit or tune on future months
- Never touch the final holdout period in any fitting or tuning
- Heavy compute (loads, feature builds, training, CV): maximize tradinghost resources
  (parallelize across all cores, n_jobs=-1, batch DB reads)
- After completing, write results to cursor_output.md
- Before finishing: git add -A && git commit -m "<msg>" && git push -u origin feat/panel

## Expected Output in cursor_output.md
In cursor_output.md: merge commit for chore/lock-decisions and test results; feat/panel branch and commit list; universe counts summary; feature coverage table by year with flagged weak features; lookahead test results; mcap cross-check and AAPL/SIVB spot checks; target sanity numbers; file sizes; wall time and peak memory; judgment calls; explicit list of what was NOT done or skipped.

## Status
[ ] Not started
