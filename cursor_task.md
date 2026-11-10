# Cursor Task - FE-Factor-Model
Date: 2026-11-10 09:25:49
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/nonlinear-models
Priority: high

## Context
The 40-seed Lasso shuffle gate PASSED: mean 0.00073, 95% CI [-0.0022, 0.0037], t 0.66. Leakage is clean for every model. The merge was blocked only because the corrected IC metric (constant-prediction months = 0) changes Lasso's tuned alpha in 2011, from 1e-5 to 10 (all coefficients zero). That's a legitimate outcome: in the 2008-2010 validation window every non-zero alpha had negative mean IC, so "no signal" (IC 0) wins. The production predictions were made with the old, buggy metric, so they must be regenerated with the corrected one. Only models that can produce constant predictions are affected: per the last report, RF/XGB/MLP/Ridge had zero constant months in validation-relevant scoring, but VERIFY this rather than assume it. The ensemble uses Lasso, so it changes too.

Portfolio rule (already decided): in a month where a model's predictions are constant, the portfolio holds the EW universe (no signal = no tilt), and IC = 0.

Rehan authorizes merging feat/nonlinear-models into main in this task if Step 4's conditions hold.

## Task
Stay on feat/nonlinear-models. Commit, push. The holdout rule is unchanged (no formation month after 2023-11).

Step 1: Verify which models' tuning changes. For Ridge, RF, XGBoost and the MLP: from each year's tuning, check whether ANY grid point produced constant predictions in any validation month (under the old metric those months were dropped). If none did for a model, its chosen params are provably unchanged; say so and don't retrain it. If any did, recompute that model's validation IC per grid point with the new metric (refit only where needed) and report whether the chosen params change. If any do, retrain that model's affected years.

Step 2: Retrain Lasso with the corrected tuning, a full walk-forward 2011-01..2023-11 (seconds). Confirm 2011 picks alpha=10, and that 2012-2023 match the earlier choices and coefficients. For constant-prediction months: IC = 0, and the portfolio = the EW universe that month (return and turnover; turnover into and out of EW counted normally).

Step 3: Regenerate everything downstream for affected models only: the Lasso predictions, the ensemble (rank-average of Lasso/XGB/MLP; if Lasso is constant in a month, rank-average the remaining two that month and note it), the portfolios, costs, FF6 absolute + active alpha, bootstrap CIs, year-by-year, regimes, the cumulative chart, and the interpretability rows that use Lasso coefficients. Update reports/models/grand_metrics.csv, active_alpha.csv and related. Report before/after for Lasso and the ensemble (Sharpe, net 10bps, excess, active alpha + t, 2011 excess).

Step 4: Run the full test suite. Merge into main with --no-ff, run the tests on main and push, ONLY if:
- the tests are green;
- Step 1 found no unexpected parameter changes in other models (or they were retrained and reported);
- Lasso 2012-2023 matches the earlier fits.
Otherwise stop and report.

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
1. A verdict line (merged yes/no + SHA).
2. The Step 1 verification table (constant validation months per model/year, any alpha changes).
3. The Lasso before/after (2011 and full window) and the ensemble before/after.
4. The updated head-to-head table for all models (Sharpe, net 10bps, excess, active alpha + t).
5. Tests, not done / skipped.
Plain names.

## Status
[ ] Not started
