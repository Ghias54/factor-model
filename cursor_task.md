# Cursor Task - FE-Factor-Model
Date: 2026-11-08 21:58:03
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/classic-factors
Priority: high

## Context
The panel is merged to main (d97d3be): 248 months x 1,000 names, 2006-01-31 to 2026-08-31, 26 rank-scaled features, post split-backfill, earnings_yield = TTM NI / mcap. Locked design (docs/design-decisions-2026-10-20.md): long-only top decile, equal weight, 1-month hold, benchmark = equal-weight universe; top-minus-bottom spread is a diagnostic only; holdout 2024-01 to 2026-08 is touched once at the very end. The classic four are value_bm (book-to-market), size_ln_mcap, momentum_12_1 and ROE (use the panel's ROE column name). The spec asks us to replicate the classic factors and check they correlate sensibly with the Ken French library (expect corr roughly >= 0.6 with HML/SMB/UMD/RMW). Use the raw feature values (features_raw.parquet) for sorting, and the panel's next-month return for returns.

## Task
Branch feat/classic-factors from main. Commit as you go, push. Rehan has pre-authorized merging THIS branch into main at the end, but ONLY if every test passes and the holdout check in Step 5 passes; otherwise leave it unmerged and report.

HOLDOUT RULE: everything in this task uses months up to and including 2023-12 only (return month <= 2024-01 is the last allowed: a portfolio formed 2023-12-31 earns its January 2024 return; exclude that too, so the last formation month is 2023-11-30 and the last return month is 2023-12). Assert this in code.

Step 1: Ken French data. Download from the Ken French Data Library (monthly):
- Fama/French 5 Factors (2x3): Mkt-RF, SMB, HML, RMW, CMA, RF
- Momentum Factor (UMD)
Parse into tidy parquet under data/external/ff/ with the source URL and download date in a small meta json. Convert percent to decimal. Write a loader in src/factor_model/ (for example external/ff.py) with a unit test that parses a small saved sample.

Step 2: Our classic factor returns, from the panel. For each month t (formation 2006-01 to 2023-11), sort on the raw value of each classic feature (value_bm, size_ln_mcap, momentum_12_1, ROE); skip null rows for that feature.
- (a) Decile long-short spreads, equal-weighted and value-weighted (weights = mcap at t): top minus bottom decile, signed so the sign matches FF (value: high B/M minus low; size: SMALL minus big, matching SMB; momentum: winners minus losers; ROE: robust minus weak).
- (b) An FF-style 2x3 version: size median split x 30/70 breakpoints on the characteristic, value-weighted, averaged the FF way (for example HML = 1/2(SH+BH) - 1/2(SL+BL); SMB = average of the three small minus the three big from the B/M sort). Use our universe's own breakpoints (we have no NYSE flag) and say so in the output.
- Save the monthly return series to data/factors/classic_factor_returns.parquet.

Step 3: Fama-French check. Align on return month. For each of our factors vs its FF counterpart (value -> HML, size -> SMB, momentum -> UMD, ROE -> RMW), report:
- the Pearson correlation of monthly returns, with n months;
- the same correlation by subperiod: 2006-2010, 2011-2020, 2021-2023;
- annualized mean, vol and Sharpe for ours and FF over the same months;
- a 12-month rolling correlation plot per factor (PNG under reports/classic_factors/);
- also a correlation matrix among our four factors vs among HML/SMB/UMD/RMW.
Flag any factor below 0.6 and give the likely reason (top-1000 large-cap universe vs FF's all-NYSE/AMEX/NASDAQ, our breakpoints, equal vs value weight, ROE definition vs RMW's operating profitability). Do not tune anything to raise the correlation.

Step 4: Long-only baseline (this is the bar the ML models must beat later). Each month, top decile (100 stocks), equal weight, 1-month hold, for:
- each classic factor individually (in the attractive direction: high B/M, small size, high momentum, high ROE);
- the classic 4-factor equal-weight composite: the average of the four rank-scaled features from the panel (the same [-1,1] scaling the models will see; missing -> 0, as in the panel).
Benchmark: equal-weight universe. Report annualized return, vol, Sharpe, max drawdown, monthly turnover, excess return vs benchmark, and information ratio. Also the top-minus-bottom decile spread and the monthly rank IC (Spearman of score vs next-month return) as diagnostics. Plus regime cuts: 2020-02 to 2020-12 (COVID crash/recovery) and 2022-01 to 2022-12 (rate hikes). Save to reports/classic_factors/ and the series to data/factors/.

Step 5: Holdout check. Add a test asserting that no saved series contains a formation month after 2023-11 or a return month after 2023-12. Run the full suite.

Step 6: If all tests pass and Step 5 passes: merge feat/classic-factors into main with --no-ff, run the tests on main, push. Otherwise stop and report.

Write everything to cursor_output.md. Use all cores, RAM < 25 GB.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b feat/classic-factors from main (or continue it if it exists)
- Commit as you go. Push the branch. Do NOT merge to main - the human merges one by one
- Point-in-time only: every feature at month-end t must use data available as of t
  (fundamentals on filing/acceptance date, not period end; no restated values)
- No survivorship bias: build the universe from historical constituents, never today's list
- Walk-forward only: train through month t, predict t+1; never fit or tune on future months
- Never touch the final holdout period in any fitting or tuning
- Heavy compute (loads, feature builds, training, CV): maximize tradinghost resources
  (parallelize across all cores, n_jobs=-1, batch DB reads)
- After completing, write results to cursor_output.md
- Before finishing: git add -A && git commit -m "<msg>" && git push -u origin feat/classic-factors

## Expected Output in cursor_output.md
cursor_output.md with: (1) summary paragraph; (2) correlation table (ours vs FF, overall + subperiods, decile and 2x3 versions) with pass/fail vs 0.6 and a reason for any miss; (3) long-only table for the four factors + composite vs the equal-weight benchmark; (4) regime numbers; (5) what's saved where; (6) not done / skipped. Plain names, no unexplained codes.

## Status
[ ] Not started
