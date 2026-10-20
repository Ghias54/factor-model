# Cursor Task - FE-Factor-Model
Date: 2026-10-20 13:30:51
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: chore/lock-decisions
Priority: HIGH

## Context
Rehan locked all design decisions on 2026-10-20; they are in docs/design-decisions-2026-10-20.md (read it fully first). Before building the panel, the repos need cleanup: the docs/data-audit branch (holds db.py, loaders, .gitignore fix, audit docs) is unmerged, and main cannot connect to the database without it. Uncommitted working-tree edits to configs/default.yaml and docs/decisions.md are superseded and contain an incorrect "survivorship-free" claim. In ~/stock-database, load/gap-pull-1016 is applied to prod but unmerged. Rehan explicitly approved both merges on 2026-10-20.

## Task
1. In ~/FE-Factor-Model: run `git status` and record the uncommitted diffs of configs/default.yaml and docs/decisions.md in cursor_output.md (for reference), then discard them (`git restore` / `git checkout --`). Do not discard anything else; if other uncommitted changes exist, stop and list them.
2. Merge `docs/data-audit` into main (Approved by Rehan 2026-10-20): `git checkout main && git pull && git merge --no-ff docs/data-audit`. Resolve conflicts conservatively, push main. Confirm src/factor_model/data/db.py and loaders.py are on main and `pytest` passes.
3. In ~/stock-database: merge `load/gap-pull-1016` into main (Approved by Rehan 2026-10-20) with --no-ff, run its pytest suite, push. Do NOT run any loads, crawls, repairs, or migrations, and do not touch the database.
4. Back in ~/FE-Factor-Model: `git checkout -b chore/lock-decisions` from the updated main.
5. Update configs/default.yaml to match docs/design-decisions-2026-10-20.md: sample (start 2006-01-31, end 2026-08-31, holdout_start 2024-01-31, first_oos 2011-01-31), universe (top_n_by_market_cap 1000, min_price 5.0 on close_raw, mcap = close × weighted_average_shares_diluted, rerank monthly; remove robustness_top_n), target (rank, horizon 1 month, risk-free month3), features (list of the 26 names with short definitions; fundamentals staleness 548 days, earnings ≤120 days; preprocessing rank to [-1,1], missing → 0), models (the 7 models and their tuning grids as placeholders), validation (expanding window, yearly refit in January, 3-year validation window), portfolio (long_only: true, top decile, equal weight, benchmark equal-weight universe, cost sensitivity 10 bps/side), evaluation (regimes: covid 2020-02..2020-12, ratehike 2022-01..2022-12, survivorship split 2011-2020 / 2021-2023). Keep existing keys that are still valid; make tests/test_config.py pass (update tests only where they asserted old TBD values).
6. Rewrite docs/decisions.md as a decisions log table (date, decision, reason, source) covering every section of design-decisions-2026-10-20.md. State explicitly that the universe is NOT survivorship-free before ~2021 (FMP lacks pre-2016 delistings).
7. Update docs/plan.md build order to match the "Build order after cleanup" section; keep any useful existing content.
8. Do NOT build the universe, panel, features, or models in this task.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b chore/lock-decisions from main (or continue it if it exists)
- Commit as you go. Push the branch. Do NOT merge to main - the human merges one by one
- Point-in-time only: every feature at month-end t must use data available as of t
  (fundamentals on filing/acceptance date, not period end; no restated values)
- No survivorship bias: build the universe from historical constituents, never today's list
- Walk-forward only: train through month t, predict t+1; never fit or tune on future months
- Never touch the final holdout period in any fitting or tuning
- Heavy compute (loads, feature builds, training, CV): maximize tradinghost resources
  (parallelize across all cores, n_jobs=-1, batch DB reads)
- After completing, write results to cursor_output.md
- Before finishing: git add -A && git commit -m "<msg>" && git push -u origin chore/lock-decisions

## Expected Output in cursor_output.md
For both repos: merge commits and pushed state of main, test results. The recorded (discarded) diffs. Branch chore/lock-decisions commit list. Summary of config keys changed. Any conflicts and how they were resolved. Explicit list of what was NOT done or skipped.

## Status
[ ] Not started
