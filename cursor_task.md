# Cursor Task - FE-Factor-Model
Date: 2026-11-04 21:20:09
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/panel-ms02-rebuild
Priority: High

## Context
The repo moved from tradinghost to MS-02 on Oct 26 2026. data/panel/* was rsynced over but was built from tradinghost's database. The stock database now lives on MS-02, and the panel must be rebuilt from it so everything downstream (classic factors, Fama-French validation, models) is sourced from the MS-02 DB. Design decisions are locked in docs/design-decisions-2026-10-20.md (top 1000 by mcap at >= $5 re-ranked monthly, 2006-01-31 to 2026-08-31, 26 features, holdout from 2024-01-31). Previous build: 248,000 rows, 248 months, 26 features, ~746s wall with n_jobs=-1.

## Task
1. Confirm .env / config points at MS-02's local Postgres (not tradinghost's Tailscale address). Print the resolved host/db name (no password) in the output. If anything still points at tradinghost, fix it in config only, not .env.example secrets.
2. Back up the current outputs: move data/panel/ to data/panel_tradinghost_backup/ (keep it untracked, same as data/panel).
3. Run the existing pipeline (scripts/build_panel.py, plus the universe build if it's a separate step) unchanged against MS-02's DB. Maximize compute on MS-02: n_jobs=-1, and raise any worker/chunk/thread settings so all cores and available RAM are used; report core count and peak memory.
4. Compare new vs backup: row count, month count, feature list, per-month universe size, and per-feature mean/std/null-rate. Flag any feature where null rate or mean differs by more than 1% and explain why if possible. Exact equality is expected if the DB is a faithful copy.
5. Run the full test suite, including tests/test_panel_lookahead.py, against the new panel.
6. Regenerate docs/panel-report.md from the new build.

Do NOT: change feature definitions, the universe rule, the sample window, or the holdout date. Do NOT start building the classic factors, Fama-French validation, or any models. Do NOT delete the backup folder. Do NOT merge to main; leave the branch for Rehan to review.

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
In cursor_output.md: resolved DB host, wall time per stage + total, cores/peak RAM used, row/month/feature counts, the new-vs-backup comparison table with any >1% diffs explained, test results, list of files changed, and a section titled "Not done / skipped" listing anything left out.

## Status
[ ] Not started
