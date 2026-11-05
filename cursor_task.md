# Cursor Task - FE-Factor-Model
Date: 2026-11-04 22:26:37
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/panel-ms02-rebuild
Priority: high

## Context
Stockdb bug (vault: Knowledge/Stockdb Split-Adjustment Bug 2026-11): when a split lands, price_daily.close is restated but share counts (fundamental_quarterly.weighted_average_shares / _diluted) and EPS are not. The panel already has a vendor-mcap sanity gate (configs/default.yaml universe.mcap_vendor_sanity, max_ratio 3.0) that replaced 2,977 rows and removed KNDI/GP. Two things are still unchecked:
1. Per-share features (sue, earnings_yield, anything else in src/factor_model/features/compute.py) may divide unadjusted EPS by split-adjusted price, or compare EPS across a split.
2. 2:1 and 3:1 splits fall under the 3x gate, so they could still be distorting mcap, turnover and the universe.

The real fix is running in the stock-database repo right now (branch fix/split-restatement, dry run only). Stockdb is not being changed tonight, so read it freely but do not write to it.

Stay on branch feat/panel-ms02-rebuild (commit on top of it). Backups from earlier: data/panel_tradinghost_backup/, data/panel_ms02_pre_mcap_fix/. Keep a backup of the current panel before any rebuild.

## Task
Step 1: Find per-share inputs.
- Read src/factor_model/features/compute.py and list every feature that uses EPS, shares, dividends per share, book value per share, or mcap computed from shares. For each, say what it divides by, and whether numerator and denominator are on the same split basis.
- In particular: how sue is built (EPS change vs a year ago? scaled by price or by std dev?) and how earnings_yield is built (EPS / price, or net income / mcap?).

Step 2: Measure exposure.
- For each per-share feature, find security-months where a split (corporate_action, action_type = 'split') falls inside the feature's lookback (e.g. the 4 or 8 quarters sue looks at, or between the EPS period and the price date). Report counts and share of panel rows, by year.
- Compare those rows' feature values against the same feature with EPS restated by the cumulative split factor. Report how many change by more than 10%, and how many move by more than one decile in the cross-sectional rank. List the worst 30 (symbol, month, feature, before, after).

Step 3: 2:1 / 3:1 splits under the gate.
- Over the whole panel, histogram the computed / vendor mcap ratio (log scale; bins around 1, 1.5, 2, 3, 4, 5, 10, and the reciprocals). Report counts per bin, and the count inside [1.5, 3] and [1/3, 1/1.5] that the current gate lets through.
- For those rows: how many have a split in corporate_action between their latest fundamentals date and the month-end (split explains it) vs no split (something else)?
- Re-run the universe build with max_ratio 1.5 as a test (do not change the default yet). Report rows replaced, universe membership swaps, and the top 20 securities affected.

Step 4: Recommend, don't apply.
- Say whether per-share features need a fix now (e.g. restate EPS by split factor inside the panel build, or switch to net income / vendor mcap) or can wait for the stockdb fix.
- Say whether to tighten max_ratio to 1.5.
- Do not change configs or feature code in this task; only add the audit script and its outputs.

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
Write cursor_output.md with:
1. One-paragraph verdict: are per-share features or 2:1/3:1 splits a real problem for the panel?
2. Step 1 table: feature, formula, split-basis consistent yes/no.
3. Step 2 counts by year, decile-move counts, worst 30.
4. Step 3 ratio histogram, the in-gate counts, split-explained share, and the max_ratio 1.5 test results.
5. Recommendations (per-share fix now vs wait; gate 1.5 yes/no).
Save raw numbers under logs/split_audit/. Commit to feat/panel-ms02-rebuild; do not merge. Plain names, no unexplained codes.

## Status
[ ] Not started
