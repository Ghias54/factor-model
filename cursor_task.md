# Cursor Task - FE-Factor-Model
Date: 2026-10-06 16:42:17
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: docs/data-audit
Priority: HIGH

## Context
First real task in this repo. Before building the monthly universe and panel, we need to know exactly what the stock-database Postgres (~/stock-database, read its .cursor/rules/stock-database.mdc and docs/ first) contains and how to query it point-in-time. Several config values in configs/default.yaml are TBD (sample start, holdout, universe size, target); this audit's numbers will be used to decide them. This task is READ-ONLY against the database and writes no pipeline code beyond helpers in src/factor_model/data/.

## Task
1. Read ~/stock-database docs (phase3-schema.md, runbook.md, endpoint-audit.md, fundamentals-filing-date-design.md) and the migrations to learn the schema. Connect via src/factor_model/data/db.py (set up .env with STOCKDB_DSN if missing; never commit it).

2. Write docs/data-dictionary.md covering every table this project could use: security/identity and listing history, daily prices (which columns are split- and dividend-adjusted), delistings and delisting returns if any, income statement / balance sheet / cash flow (key columns, date_available coverage), analyst estimates and earnings surprises (are they point-in-time? is there an as-of date?), shares outstanding, and anything usable for beta/volatility/turnover. For each: primary key, date column to join on, row count, date range, and whether it is point-in-time safe. Mark clearly any table that is a "now" snapshot and must NOT be used as a feature.

3. Write docs/data-audit.md with numbers, using one batched query per table (no per-symbol loops; parallelize where useful):
   - Number of qualifying common stocks per month-end, from 1990 to latest, including delisted names (confirm SIVB appears while trading).
   - Same counts after a $5 price filter and after taking top 1000 by market cap.
   - Monthly coverage (% of universe with non-null value) for: month-end price, shares outstanding/market cap, book equity, net income, revenue, gross profit, total assets, total debt, analyst EPS estimates, earnings surprises.
   - Median lag between fiscal period end and date_available.
   - The earliest month-end where coverage is good enough (>80% for price, market cap, book equity, and net income) — this informs sample.start.
   - Whether delisting returns or a final price on delisting can be derived.

4. Add read-only helper functions in src/factor_model/data/ (e.g. loaders.py) for the queries you needed, with a smoke test in tests/ that runs against the dev database (STOCKDB_DEV_DSN) and is skipped if it is not set.

5. Do NOT build the universe, panel, or any factors yet. Do NOT change configs/default.yaml values; propose values in data-audit.md instead.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b docs/data-audit from main (or continue it if it exists)
- Commit as you go. Push the branch. Do NOT merge to main - the human merges one by one
- Point-in-time only: every feature at month-end t must use data available as of t
  (fundamentals on filing/acceptance date, not period end; no restated values)
- No survivorship bias: build the universe from historical constituents, never today's list
- Walk-forward only: train through month t, predict t+1; never fit or tune on future months
- Never touch the final holdout period in any fitting or tuning
- Heavy compute (loads, feature builds, training, CV): maximize tradinghost resources
  (parallelize across all cores, n_jobs=-1, batch DB reads)
- After completing, write results to cursor_output.md
- Before finishing: git add -A && git commit -m "<msg>" && git push -u origin docs/data-audit

## Expected Output in cursor_output.md
Branch and commit list; summary of the schema; the key coverage numbers (stocks per month by filter, earliest usable month, coverage table); proposed values for sample.start, holdout_start, and universe size with reasoning; list of point-in-time hazards found; wall time for heavy queries; and an explicit list of what was NOT done or skipped.

## Status
[ ] Not started
