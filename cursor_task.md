# Cursor Task - FE-Factor-Model
Date: 2026-11-08 22:18:57
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/walkforward-linear
Priority: high

## Context
Main is at dd6721b: panel (248 months x 1,000 names, 26 rank-scaled features in [-1,1], missing -> 0) plus the classic-factor baseline (src/factor_model/factors/classic.py, scripts/run_classic_factors.py, data/factors/). Baseline to beat (formation 2011-01 to 2023-11 must be recomputed for a like-for-like window): the EW universe had Sharpe 0.49 and the classic-4 composite 0.47 over 2006-2023.

Locked design (docs/design-decisions-2026-10-20.md):
- target = cross-sectional rank of next-month return;
- expanding window, refit and retune every January, last 3 years of training as validation;
- first out-of-sample prediction 2011-01;
- portfolio = long-only top decile (100 stocks), equal weight, 1-month hold, vs the EW universe;
- the top-minus-bottom spread and rank IC are diagnostics only;
- holdout 2024-01 onward is touched once at the very end, never here.

This task builds the walk-forward engine plus Lasso and Ridge. Random Forest, XGBoost, the MLP and the ensemble come in the next task and will plug into the same engine, so design the model interface for that.

## Task
Branch feat/walkforward-linear from main. Commit as you go, push. Rehan pre-authorizes merging THIS branch into main at the end, but ONLY if all tests pass, including the timing and holdout tests in Step 2; otherwise leave it unmerged and report.

HOLDOUT RULE: the last formation month predicted is 2023-11 (return month 2023-12). Nothing from formation 2023-12 onward may be loaded into training, validation or evaluation. Assert this in code, reusing the caps in classic.py.

Step 1: Walk-forward engine (src/factor_model/walkforward/).
- Target: each month, the cross-sectional rank of next-month return, scaled to [-1,1] (or the same as the features); state the choice.
- Timing: for a prediction at formation month t, training may only use formation months whose labels are fully realized by t (formation month <= t-1). Refit once a year, at the first formation month of each year Y (Y = 2011..2023), on all formation months from 2006-01 through December of Y-1. Predict every formation month of year Y with that fit (Y = 2023 stops at 2023-11).
- Tuning: within each yearly refit, hold out the last 36 formation months of the training window as validation. Pick hyperparameters on validation by mean monthly rank IC (Spearman of prediction vs realized return, per month, averaged), then refit on the full training window with the chosen values.
- Model interface: a small class or protocol (fit / predict / param_grid / name), so Random Forest, XGBoost and the MLP can be added later without touching the engine. Save per-year fits: chosen params, validation IC, and coefficients where they exist.
- Outputs: one predictions parquet per model (security, formation month, prediction, realized next-month return), saved under data/predictions/.
- Use all cores; RAM < 25 GB.

Step 2: Tests.
- A timing test: for every refit year, assert that the max training formation month is < the first prediction month, and that no training label uses a return month after the prediction month.
- A holdout test: assert that no prediction or training row has a formation month after 2023-11.
- A leakage smoke test: shuffle the target within each month and confirm out-of-sample rank IC is about 0 (|IC| < 0.005 averaged).
- Unit tests for the engine on a tiny synthetic panel.

Step 3: Models.
- Lasso: grid over alpha (log spaced, wide enough that the strongest setting zeroes most features).
- Ridge: grid over alpha (log spaced).
- Both on the 26 panel features as they are (already rank-scaled; no extra scaling).
- Save each year's coefficients.

Step 4: Portfolios and evaluation. Reuse the classic-factor code for metrics. Over the out-of-sample window 2011-01 to 2023-11 (formation), compare: Lasso, Ridge, the classic-4 composite (recomputed on this same window) and the EW universe. For each, report:
- annualized return, vol, Sharpe, max drawdown, monthly turnover, excess vs the EW universe, information ratio;
- mean rank IC and the share of months with positive IC;
- the top-minus-bottom decile spread (diagnostic);
- excess return by year (2011-2023);
- regimes 2020-02 to 2020-12 and 2022;
- a block-bootstrap 95% CI on the Sharpe difference (model minus classic-4) and on mean excess vs the EW universe (12-month blocks).
Save tables and a cumulative-return chart (all four lines) under reports/linear/.

Step 5: Lasso feature selection (one of the spec's main outputs).
- For each refit year: which features have non-zero coefficients at the chosen alpha, and their signs.
- A table of features x years (non-zero/sign), plus the share of years each feature survives.
- Rank features by average |coefficient| for Lasso and for Ridge, and give the rank correlation between the two models' orderings.
- Flag any feature whose sign flips across years.
- Save under reports/linear/.

Step 6: If all tests pass, merge feat/walkforward-linear into main with --no-ff, run the tests on main, push. Otherwise stop and report.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b feat/walkforward-linear from main (or continue it if it exists)
- Commit as you go. Push the branch. Do NOT merge to main - the human merges one by one
- Point-in-time only: every feature at month-end t must use data available as of t
  (fundamentals on filing/acceptance date, not period end; no restated values)
- No survivorship bias: build the universe from historical constituents, never today's list
- Walk-forward only: train through month t, predict t+1; never fit or tune on future months
- Never touch the final holdout period in any fitting or tuning
- Heavy compute (loads, feature builds, training, CV): maximize tradinghost resources
  (parallelize across all cores, n_jobs=-1, batch DB reads)
- After completing, write results to cursor_output.md
- Before finishing: git add -A && git commit -m "<msg>" && git push -u origin feat/walkforward-linear

## Expected Output in cursor_output.md
cursor_output.md with:
1. A summary paragraph: does Lasso/Ridge beat the classic composite and the EW universe out of sample?
2. A head-to-head table with Sharpe and drawdown, including the Sharpe difference CI.
3. Year-by-year excess returns and regimes.
4. The Lasso selection table (which features carry signal, how stable), and the Ridge vs Lasso agreement.
5. Timing-check results.
6. The engine interface for the next task's models.
7. Not done / skipped.
Plain names, no unexplained codes.

## Status
[ ] Not started
