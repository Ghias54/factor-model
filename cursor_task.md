# Cursor Task - FE-Factor-Model
Date: 2026-11-09 22:07:11
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/nonlinear-models
Priority: high

## Context
The full-window leakage gate (reports/models/leakage_full_window*.csv) passed for Ridge, RF, XGBoost, the MLP and the ensemble. Lasso failed narrowly: shuffled mean IC 0.00619 vs the 0.005 limit, t 1.73. The cause found: under shuffled labels Lasso often zeroes all coefficients, so predictions are constant in a month, and the IC code skips months with nunique < 2 (119-143 of 155 months kept). Dropping those months biases the pooled mean toward the lucky months. The correct treatment is to count a constant-prediction month as IC = 0 (a model with no ranking has zero skill), not to drop it. This is a metric correction, NOT a threshold change: the gate stays |mean IC| < 0.005 AND |t| < 2.

Rehan authorizes merging feat/nonlinear-models into main in this task, conditional on the gate passing for ALL models under the corrected metric and all tests green (the rule file now allows explicit per-task authorization).

## Task
Stay on feat/nonlinear-models. Commit, push. No retraining unless Step 3 requires it. The holdout rule is unchanged (no formation month after 2023-11).

Step 1: Fix the IC metric everywhere. In the shared IC function (used by tuning, evaluation and the leakage script), when a month's predictions have fewer than 2 unique values (or zero variance), record IC = 0 for that month instead of dropping it. Keep a count of such months. Add a unit test. Check whether the same skip logic was used in hyperparameter tuning (validation mean IC). If it was, note it, but do NOT retune in this task; just report how many validation months it affected per model and year.

Step 2: Re-score the leakage gate from the SAVED shuffled predictions (no refitting). For every model and seed: recompute the mean IC, t, positive share, and the number of constant months, using the corrected metric. Apply the same gate (|mean IC| < 0.005 AND |t| < 2, across pooled seeds). Update reports/models/leakage_full_window*.csv (keep the old numbers as *_old columns). If any model still fails, STOP: do not merge, report.

Step 3: Constant-prediction months in the REAL runs. For each real model (Lasso, Ridge, RF, XGB, MLP, ensemble) and each formation month 2011-01..2023-11:
- count months where predictions are constant or nearly so (nunique < 2, or more than 10% of names tied at the top-decile cutoff);
- for any such month, say how the top decile was chosen (tie-breaking order).
If there are ZERO such months for every model, say so and go on. If there are some: re-score that model's portfolio and IC with the corrected IC (constant month = 0) and, for portfolio formation, hold the EW universe that month (no signal = no tilt). Report before/after Sharpe, excess and active alpha, and update reports/models/grand_metrics.csv and active_alpha.csv only for the affected models.

Step 4: Run the full test suite.

Step 5: Merge only if Step 2 passes for ALL models and Step 4 is all green: merge feat/nonlinear-models into main with --no-ff, run the tests on main, push. Otherwise leave it unmerged.

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
Short cursor_output.md:
1. A verdict line: gate passed yes/no, merged yes/no (with SHA).
2. The corrected shuffle table for all models.
3. The constant-month counts for the real runs, and any rerun results with before/after.
4. Test results.
5. Not done / skipped.
Plain names, no unexplained codes.

## Status
[ ] Not started
