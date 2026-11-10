# Cursor Task - FE-Factor-Model
Date: 2026-11-09 20:04:20
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/nonlinear-models
Priority: high

## Context
feat/nonlinear-models (b88bee0 + wrap-up) adds RF, XGBoost, the MLP and the ensemble. 43 tests pass, but only after the nonlinear leakage-shuffle threshold was loosened from 0.005 to 0.02 (3 seeds, 2011-2013 window). The real models' mean rank IC is 0.029-0.034, so a 0.02 threshold is too loose to rule out leakage. Under the null with ~1,000 names, the monthly IC sd is about 0.03, so the mean over ~155 months x several seeds should be about +/-0.002. The MLP is the best model AND uncorrelated with the others' importances, which is why it needs a proper check before anyone trusts it.

Also, the FF6 alpha of the EW universe itself is -2.3% (t -2.7), so absolute alpha penalizes every equal-weight top-1000 book. The fairer number is alpha on active return (portfolio minus EW), which is already in reports/models/grand_metrics.csv (ff6_active_alpha_*) but wasn't summarized.

Rehan explicitly authorizes merging feat/nonlinear-models into main in this task, conditional on Step 1 passing. Update the workspace rule so this kind of explicit per-task authorization is allowed going forward (Step 4).

## Task
Stay on feat/nonlinear-models. Commit as you go, push. The holdout rule is unchanged: no formation month after 2023-11 anywhere.

Step 1: Full-window leakage check (the gate for merging). For each of Lasso, Ridge, RF, XGBoost and the MLP (and the ensemble, built from the shuffled Lasso/XGB/MLP predictions):
- Shuffle the target within each formation month (independently per seed) and run the FULL walk-forward 2011-01 to 2023-11 with the SAME grids and tuning as the real run. Use 5 seeds for Lasso/Ridge/XGB; 3 seeds for RF and the MLP (their runs are expensive; the MLP may use 2 of its 5 internal seeds per fit to save time, but say so).
- Report per model: mean OOS rank IC per seed, the mean across seeds, the sd across seeds, the t-stat of the mean monthly IC (pooled across seeds), and the share of months with IC > 0. Also report the real model's mean IC and t for comparison.
- Also run, ONCE, the real (unshuffled) MLP with the label shifted forward one extra month (target = return of month t+2 instead of t+1, same everything else). A big IC there would mean the features leak future information. Report its IC.
- PASS if every model's shuffled mean |IC| < 0.005 AND |t| < 2. If anything fails, STOP. Do not merge, do not tweak thresholds; report which model, by how much, and the most likely leak path (look at the MLP's early-stopping/validation split, feature scaling or normalization fitted on data beyond the training window, and seed/ensemble averaging that touches the predicted year).
- Replace the 0.02 nonlinear threshold in the test suite with a fast version of this check: a short window is fine for speed, but use more seeds so the threshold can go back to 0.005, or else assert |t| < 2 rather than a loose absolute level. Document the choice in the test.

Step 2: Active alpha summary (no new modeling). From the existing predictions, for every portfolio (classic-4, Lasso, Ridge, RF, XGB, MLP, ensemble): FF6 (FF5+UMD) Newey-West (lag 6) regression of ACTIVE return (portfolio minus EW universe), gross and net 10 bps. Report annualized alpha, t, and the market and other betas. 2011-01 to 2023-12 return months. Save to reports/models/active_alpha.csv.

Step 3: Run the full test suite.

Step 4: Workspace rule. Edit .cursor/rules/factor-model.mdc: keep "never self-merge to main" as the default, but add an exception: "unless the current cursor_task.md explicitly states Rehan authorizes merging that specific branch, and every condition the task attaches to the merge is met." Also fix any task-template text that says "Do NOT merge to main" so it doesn't contradict the task's own merge step.

Step 5: Merge only if Step 1 PASSED and Step 3 is all green: merge feat/nonlinear-models into main with --no-ff, run the tests on main, push. Otherwise leave it unmerged.

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
1. A verdict line: leakage clean yes/no, merged yes/no (with SHA).
2. The full-window shuffle table per model, plus the explanation of why the short window failed.
3. The active-alpha table and a 2-3 sentence interpretation.
4. The rule-file change.
5. Test results.
Plain names, no unexplained codes.

## Status
[ ] Not started
