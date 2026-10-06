# factor-model

Machine learning for factor investing — UIUC Financial Engineering Club project.

Classic factor investing asks whether characteristics like value, size, momentum, and quality earn a return premium. This project treats those four factors as the baseline, then tests whether ML models trained on a much larger feature set (25–40 features) can predict which stocks outperform next month better than simple linear factor sorts. Everything is evaluated with walk-forward validation and long-short portfolio backtests.

## Pipeline

```
stock-database (Postgres, point-in-time)
        │
        ▼
1. Universe        monthly investable universe, survivorship-bias free
2. Panel           monthly (date, security) panel: prices, returns, fundamentals as of t
3. Factors         value, size, momentum, quality
4. Features        15–25 additional features (revisions, vol/beta, reversal, accruals, ...)
5. Models          Lasso / Ridge → Random Forest (+ XGBoost) → MLP → ensemble
6. Walk-forward    train through t, predict t+1, roll forward
7. Backtest        monthly decile long-short portfolios per model
8. Evaluation      Sharpe, drawdown, turnover, Fama-French validation, SHAP, regimes
```

## Repo layout

```
configs/                 YAML configs (dates, universe, costs, holdout)
docs/                    plan, decisions log, data dictionary
notebooks/               exploration only — outputs cleared before commit
scripts/                 thin CLIs that run pipeline stages
src/factor_model/
    data/                database access (read-only) and loaders
    universe/            monthly universe construction
    factors/             classic four factors
    features/            additional engineered features
    models/              linear, tree, neural net, ensemble
    backtest/            walk-forward engine and portfolio construction
    evaluation/          performance metrics, Fama-French, SHAP, regimes
tests/                   pytest, including lookahead tests
data/                    (gitignored) cached panels, French factors
outputs/                 (gitignored) predictions, backtest results, figures
```

## Data

The source is the `stock-database` Postgres database on tradinghost (built from Financial Modeling Prep). It is point-in-time: fundamentals are keyed to their SEC acceptance date, and delisted companies stay in the universe. This repo only reads from it.

Benchmark factor returns come from the [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## Setup

```bash
git clone https://github.com/Ghias54/factor-model.git
cd factor-model
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env    # fill in STOCKDB_DSN
pytest
```

Teammates connect to the database over Tailscale; ask Rehan for access and credentials.

## Ground rules

- **Point-in-time:** every input at month-end t must have been public at t.
- **No survivorship bias:** the universe at t is what traded at t.
- **Walk-forward only:** no shuffled cross-validation, no fitting on future months.
- **Holdout is untouched** until the final evaluation.
- **Baselines first:** every model is compared against classical factor sorts.

Full rules are in `.cursor/rules/factor-model.mdc`. Project plan and status are in `docs/plan.md`.

## References

- Fama & French (1993), "Common Risk Factors in the Returns on Stocks and Bonds"
- Fama & French (2015), "A Five-Factor Asset Pricing Model"
- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"
- Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
