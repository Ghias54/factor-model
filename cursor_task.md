# Cursor Task - FE-Factor-Model
Date: 2026-11-08 21:37:42
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/panel-ms02-rebuild
Priority: high

## Context
The current panel (data/panel/) was built 2026-11-04, BEFORE the stockdb split-restatement backfill was applied to prod on 2026-11-05 ~15:59 CT. The backfill restated weighted_average_shares, weighted_average_shares_diluted, eps, eps_diluted (raw copies kept in *_raw columns, factor in split_factor) and volume (volume_raw) for names classified needs_adjust. Stockdb reports ~8,856 panel cells (~3.6%) across 86 securities now have split_factor != 1. Stockdb validate_prod passes; prices run through 2026-11-04. KNDI/GP computed/vendor mcap now 0.73-1.00.

The 11/04 split audit (logs/split_audit/) said earnings_yield (TTM eps_diluted / close) was on mixed bases for 19.5% of rows. But stockdb's 11/08 spot checks show ordinary forward splits (NVDA 2021/2024, AAPL 2020, TSLA 2020/2022, GOOGL 2022) were already vendor-adjusted, with EPS on the same basis as close. The audit assumed EPS was never restated, so 19.5% is likely an overstatement. Decision stands anyway: earnings_yield becomes TTM net income / mcap (split-proof, same basis as sales_to_price and cashflow_to_price).

Decisions already made (do not change): keep universe.mcap_vendor_sanity max_ratio at 3.0. Stockdb is read-only. Sample, universe, target, and feature list stay as locked 2026-10-20.

## Task
Branch: continue feat/panel-ms02-rebuild. Commit as you go, push, do NOT merge.

Step 0: Backup. Copy the current data/panel/ to data/panel_ms02_pre_ey_fix/ (this is the pre-backfill, old-formula panel and is the baseline for every comparison below). Run the full test suite and record the result.

Step 1: Re-measure the earnings_yield problem on today's data (read-only, before changing code). For the old formula (TTM eps_diluted / month-end close), compare eps_diluted vs eps_diluted_raw and split_factor on the point-in-time fundamental rows the panel uses. Report how many panel rows are actually on mixed bases now, vs the 19.5% the audit estimated. One paragraph on why they differ.

Step 2: Change the feature. In src/factor_model/features/compute.py, earnings_yield = TTM net income / mcap. TTM net income = sum of the latest four quarterly net_income values available as of month-end (same point-in-time rules and staleness limits as the current EPS version). mcap = the panel's existing mcap (after the vendor sanity gate). Keep the name earnings_yield, same null handling (null if fewer than 4 quarters or mcap missing/<=0), negatives stay negative. Update docs that describe the formula with one line on why.

Step 3: Tests. Add a unit test showing earnings_yield is unchanged by a 2:1 split (same net income, price halves, mcap unchanged). Run the full suite.

Step 4: Rebuild the panel on current prod (post-backfill). Then compare against the Step 0 backup and separate the two causes of change:
- Backfill effect: every feature except earnings_yield. For each feature: rows changed, correlation old vs new, share of rows moving >= 1 cross-sectional decile. Expect changes concentrated in turnover, mcap/size and anything using shares or volume, and on the ~86 restated securities. List the top 10 securities by rows changed.
- Formula effect: earnings_yield old vs new: correlation overall and by year, share moving >= 1 decile, coverage (non-null count) before vs after.
- Universe: membership changes (rows added/dropped by month), and how many vendor-sanity replacements fire now vs before (was 2,977).
- Confirm months, row count (248,000) and that every month still has 1,000 names.

Step 5: Re-check the two features the 11/04 audit left open, on the new panel:
- sue = (eps_actual - eps_estimated) / close from earnings_event: check whether earnings_event EPS is on the same basis as close for split names (sample the 6 forward splits above plus KNDI/GP and 10 random needs_adjust names). Verdict: fine as is, or needs a fix (propose one, do not implement).
- turnover = volume / weighted_average_shares_diluted: now that both are restated, measure the residual mismatch vs volume / (vendor mcap / close). Verdict: fine as is, or needs a fix (propose, do not implement).
- Re-run the computed/vendor mcap ratio histogram (same bins as logs/split_audit/step3) and say whether the 3x gate still looks right.

Rules: point-in-time only, no survivorship shortcuts, never touch the 2024-01+ holdout in any fitting (none here). Use all cores, batch DB reads, RAM < 25 GB. Write everything to cursor_output.md, commit, push.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b feat/panel-ms02-rebuild from main (or continue it if it exists)
- Commit as you go. Push the branch. Do NOT merge to main - the human merges one by one
- Point-in-time only: every feature at month-end t must use data available as of t
  (fundamentals on filing/acceptance date, not period end; no restated values)
- No survivorship bias: build the universe from historical constituents, never today's list
- Walk-forward only: train through month t, predict t+1; never fit or tune on future months
- Never touch the final holdout period in any fitting or tuning
- Heavy compute (loads, feature builds, training, CV): maximize tradinghost resources
  (parallelize across all cores, n_jobs=-1, batch DB reads)
- After completing, write results to cursor_output.md
- Before finishing: git add -A && git commit -m "<msg>" && git push -u origin feat/panel-ms02-rebuild

## Expected Output in cursor_output.md
cursor_output.md with:
1. One-paragraph summary: what changed, tests, how much the panel moved and why.
2. Step 1: the old earnings_yield's true exposure on the old panel (mixed-basis share, vs the audit's 19.5%).
3. Step 4 table: old vs new panel, every feature, split into backfill effect vs formula effect; universe membership changes.
4. Step 5: sue, turnover, vendor-gate results and the recommendation for each.
5. Any surprises, not done / skipped.
Plain names, no unexplained codes.

## Status
[ ] Not started
