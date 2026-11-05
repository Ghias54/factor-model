# Cursor Task - FE-Factor-Model
Date: 2026-11-04 21:35:39
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/panel-ms02-rebuild
Priority: High

## Context
The MS-02 panel rebuild (see previous cursor_output) swapped 42 universe rows vs the tradinghost build. KNDI (security_id=10012, 2013-12 to 2016-03) and GP (security_id=9909, 2020-08 to 2021-12) now appear at mid-universe ranks in the top-1000-by-mcap universe. Both are small-caps, so this likely indicates a bad shares-outstanding figure (split/consolidation not applied, ADR ratio, units error) inflating market cap. A bad mcap contaminates size, value_bm, earnings_yield, sales_to_price, cashflow_to_price, and the universe itself.

## Task
Read-only investigation first, then a minimal fix only if the data is clearly wrong.

1. For security_id 10012 (KNDI) and 9909 (GP), pull monthly: close price, shares outstanding (every source the panel uses, including the vendor mcap fallback), computed mcap, vendor mcap, and universe rank. Cover the flagged months plus 12 months either side.
2. Compare shares outstanding against SEC filings already in stockdb (10-K/10-Q/20-F/40-F cover-page or dei shares). Check for split/reverse-split events and whether prices and shares are adjusted consistently.
3. Check whether the tradinghost backup panel (data/panel_tradinghost_backup/) had these names, and at what ranks. Identify which stockdb rows changed.
4. Generalize: scan the whole panel for other names whose mcap jumps more than 5x or drops below 0.2x month-over-month without a matching price move of similar size, or whose shares_out disagrees with SEC filings by more than 3x. List the top offenders.
5. Verdict per name: data error or real. If KNDI/GP are clearly data errors, implement the narrowest fix in the factor-model code (e.g. a shares-outstanding sanity filter against SEC-reported shares, or exclude rows failing the jump check) behind a config flag, rebuild the panel, and rerun tests and the compare script. Do NOT edit rows in stockdb itself; describe the DB-level fix needed for the stock-database repo instead.

Do NOT: change feature definitions, the universe rule beyond the sanity filter, the sample window, or the holdout. Do NOT build classic factors, Fama-French validation, or models. Do NOT merge any branch to main. Do NOT delete the backup panel.

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
cursor_output.md with: per-name tables (price, shares, mcap, vendor mcap, rank) for KNDI and GP; SEC comparison; verdict with evidence; the panel-wide offender list from step 4; if a fix was made, the config flag, rows affected, new universe/feature compare vs the pre-fix MS-02 panel, and test results; the recommended stock-database fix; and a "Not done / skipped" section.

## Status
[ ] Not started
