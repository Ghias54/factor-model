# Cursor Task - FE-Factor-Model
Date: 2026-11-10 09:00:18
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/nonlinear-models
Priority: high

## Context
With the corrected IC (constant months = 0), Lasso's shuffled gate gives mean IC 0.00580 (t 1.73) over 5 seeds, failing |IC| < 0.005 by 0.0008. All other models pass. The seed-to-seed sd for Lasso is 0.0132, so the standard error of a 5-seed mean is about 0.006, which is larger than the 0.005 gate itself. Five seeds can't resolve a 0.005 gate for a model this noisy, and one seed (seed 0, IC 0.024) drives the excess. The principled fix is more evidence, not a looser gate: Lasso walk-forward fits take seconds, so run many more seeds. The gate is unchanged.

Rehan authorizes merging feat/nonlinear-models into main in this task if the conditions in Step 3 are met.

## Task
Stay on feat/nonlinear-models. Commit, push. The holdout rule is unchanged.

Step 1: Lasso shuffle with more seeds. Rerun the full-window (2011-01..2023-11) shuffled walk-forward for Lasso ONLY, with the same grid, tuning and corrected IC (constant months = 0), for 35 NEW seeds (40 total with the existing 5; use different seed values from the first 5). Report:
- the pooled mean IC across all 40 seeds, its standard error (sd across seed means / sqrt(40)), a 95% CI, and the pooled monthly t;
- the per-seed distribution (mean, sd, min, max, how many seeds have t > 2; under the null expect about 1 of 40);
- the total number of constant-prediction months.
Save to reports/models/leakage_lasso_40seed.csv.

Step 2: Validation constant months (report only, no retune). For the REAL Lasso run, refit year by year just enough to recompute the validation IC per alpha in the grid with the old (drop) and new (=0) metric. Report whether the chosen alpha would change in any year. If it changes in any year, list the years and do NOT merge; stop and report.

Step 3: Merge if ALL of these hold:
- the Lasso 40-seed pooled |mean IC| < 0.005 AND |t| < 2;
- Step 2 shows no change in the chosen alpha;
- the full test suite is green.
Then merge feat/nonlinear-models into main with --no-ff, run the tests on main, push. Otherwise leave it unmerged and report.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b feat/nonlinear-models from main (or continue it if it exists)
- Commit as you go. Push the branch. Merge to main only when this task explicitly
  authorizes it and every attached condition is met; otherwise the human merges.
- Point-in-time only: every feature at month-end t must use data available as of t
  (fundamentals on filing/acceptance date, not period end; no restated values)
- No survivorship bias: build the universe from historical constituents, never today's list
- Walk-forward only: train through month t, predict t+1; never fit or tune on future months
- Never touch the final holdout period in any fitting or tuning
- Heavy compute (loads, feature builds, training, CV): maximize tradinghost resources
  (parallelize across all cores, n_jobs=-1, batch DB reads)
- After completing, write results to cursor_output.md
- Before finishing: git add -A && git commit -m "<msg>" && git push -u origin feat/nonlinear-models

## Expected Output in cursor_output.md
Short cursor_output.md: a verdict line (passed / merged + SHA), the Lasso 40-seed table (mean, sd, SE, CI, t, constant months, seed histogram), the validation constant-month counts, test results, not done / skipped. Plain names.

## Status
[ ] Not started
