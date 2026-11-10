# Cursor Task - FE-Factor-Model
Date: 2026-11-08 22:43:54
From: Claude (claude.ai)
Repo: /home/rehan-ghias/FE-Factor-Model
Branch: feat/nonlinear-models
Priority: high

## Context
Main is at f78c5cf. The walk-forward engine (src/factor_model/walkforward/), the BaseModel interface (src/factor_model/models/base.py: name, param_grid, fit, predict, clone, coefficients_), and Lasso/Ridge are done.

Results so far (out-of-sample formation 2011-01 to 2023-11, long-only top decile, EW, 1-month hold, no costs):

| Portfolio | Sharpe | Excess vs EW | Turnover |
|---|---|---|---|
| EW universe | 0.56 | — | — |
| Classic-4 | 0.62 | +1.7% | 0.28 |
| Lasso | 0.67 | +2.9% | 0.46 |
| Ridge | 0.64 | +2.5% | 0.48 |

The bootstrap CIs on the Sharpe gap vs classic-4 include zero. Lasso consistently picks beta_60m (+), so part of the edge may be market beta.

Locked design: target = monthly cross-sectional rank of next-month return in [-1,1]; yearly January refits on an expanding window from 2006; the last 36 training months are validation; hyperparameters are chosen by mean monthly rank IC; first OOS prediction 2011-01; holdout from formation 2023-12 onward is untouched. Ken French data is in data/external/ff/ (loader src/factor_model/external/ff.py).

## Task
Branch feat/nonlinear-models from main. Commit as you go, push. Rehan pre-authorizes merging THIS branch into main at the end, but ONLY if all tests pass (including the existing timing, holdout and leakage tests, extended to every new model); otherwise leave it unmerged and report.

HOLDOUT RULE unchanged: the last formation month is 2023-11; nothing from 2023-12 onward is loaded anywhere. Reuse the existing caps and asserts.

Step 1: New models through the existing engine (no engine changes beyond what's strictly needed).
- Random Forest (scikit-learn RandomForestRegressor): grid over max_depth (e.g. 3, 5, 8), min_samples_leaf (e.g. 100, 500, 2000), max_features (e.g. 0.33, 0.5), n_estimators fixed ~300; n_jobs=-1.
- XGBoost (XGBRegressor, hist): grid over max_depth (2, 3, 4, 6), learning_rate (0.01, 0.05), n_estimators with early stopping on the validation window, subsample/colsample ~0.7, min_child_weight, reg_lambda. Early stopping uses only the validation window, never the predicted year.
- MLP (PyTorch or sklearn): 2-3 hidden layers (e.g. 32-16 or 64-32-16), ReLU, dropout (0.1-0.5) and weight decay, Adam, early stopping on validation rank IC or MSE. Average 5 seeds per fit to reduce noise; report seed spread.
- Ensemble: each month, the cross-sectional rank-average of the Lasso, XGBoost and MLP predictions (per the locked design). No fitting needed.
- Keep grids modest so total wall time stays reasonable on 24 cores; report the per-model time. Tuning is by validation mean rank IC only, as for Lasso/Ridge.
- Extend the timing / holdout / leakage-shuffle tests to RF, XGBoost and the MLP (the shuffle test can use a small grid).

Step 2: Costs. Add transaction costs to the shared evaluation for EVERY portfolio (classic-4, Lasso, Ridge, RF, XGBoost, MLP, ensemble; the EW universe too using its own small rebalancing turnover). Cost = turnover x bps per side x 2 (buy and sell), deducted monthly. Report net results at 10 and 25 bps per side alongside gross. Also the break-even cost per side at which each model's excess vs classic-4 hits zero.

Step 3: Fama-French alpha. For every portfolio, regress monthly excess return (portfolio - RF) on FF5 + UMD (Mkt-RF, SMB, HML, RMW, CMA, UMD) over 2011-01 to 2023-12 return months, Newey-West standard errors (lag 6). Report annualized alpha, t-stat and all betas, gross and net at 10 bps. Also the same regression on the long-only portfolio minus the EW universe (active return). The key question: is the model edge alpha, or market beta / known factors?

Step 4: Grand head-to-head table (the spec's main comparison), all portfolios: annualized return, vol, Sharpe, max drawdown, turnover, excess vs EW, IR, mean rank IC and the share of positive-IC months, top-minus-bottom spread (diagnostic), net Sharpe at 10/25 bps, FF6 alpha + t. Plus:
- block-bootstrap 95% CIs (12-month blocks) on the Sharpe difference vs classic-4 and vs Lasso, gross and net at 10 bps;
- excess by year 2011-2023 for every model;
- regimes 2020-02 to 2020-12 and 2022;
- a cumulative-return chart (net 10 bps) with all lines.
Save to reports/models/.

Step 5: Interpretability (one of the spec's main outputs).
- RF and XGBoost: SHAP (TreeExplainer) on a sample of each year's out-of-sample month rows (e.g. up to 20k rows/year); mean |SHAP| per feature per year and overall.
- MLP: permutation importance on each year's out-of-sample predictions, measured as the drop in rank IC when a feature is shuffled within month (5 repeats).
- Lasso/Ridge: the existing average |coefficient|.
- Build a features x models importance table (normalized ranks) and the Spearman rank-correlation matrix between models. Answer: do the model classes agree on what matters? Which features are important only for the nonlinear models (a sign of interactions or nonlinearity)?
- XGBoost: the top 5 SHAP interaction pairs (on a smaller sample), e.g. does momentum matter more when quality is high (the spec's example)?
- Check specifically how much each model leans on beta_60m and volatility features, tying back to Step 3.
- Save tables and plots (SHAP summary per tree model, importance heatmap) to reports/models/interpretability/.

Step 6: If all tests pass, merge feat/nonlinear-models into main with --no-ff, run the tests on main, push. Otherwise stop and report.

Use all cores, RAM < 25 GB; GPU not needed.

## Rules
- Read existing files before editing; keep changes minimal and localized
- ONE feature per branch: git checkout -b feat/nonlinear-models from main (or continue it if it exists)
- Commit as you go. Push the branch. Do NOT merge to main - the human merges one by one
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
cursor_output.md with:
1. A summary paragraph: does nonlinear ML beat linear, does any model beat classic-4 net of costs and after FF alpha?
2. The grand table (gross / net / alpha), with CIs.
3. Year-by-year excess and regimes.
4. The interpretability comparison: top features per model, the agreement matrix, interactions, the beta check.
5. Tuning choices per year per model, compute used.
6. Not done / skipped.
Plain names, no unexplained codes.

## Status
[ ] Not started
