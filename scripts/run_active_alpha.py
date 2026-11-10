"""FF6 Newey–West alpha on active returns (portfolio − EW), gross and net 10 bps."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_model.evaluation.costs import apply_costs
from factor_model.evaluation.ff_alpha import FF6_COLS, ff6_alpha
from factor_model.external.ff import load_ff_factors

PRED = ROOT / "data" / "predictions"
OUT = ROOT / "reports" / "models"

PORTFOLIOS = ["classic4", "lasso", "ridge", "random_forest", "xgboost", "mlp", "ensemble"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rets = pd.read_parquet(PRED / "all_portfolio_returns.parquet")
    turns = pd.read_parquet(PRED / "all_portfolio_turnover.parquet")
    rets.index = pd.to_datetime(rets.index)
    turns.index = pd.to_datetime(turns.index)
    rets = rets.loc[
        (rets.index >= pd.Timestamp("2011-02-28")) & (rets.index <= pd.Timestamp("2023-12-31"))
    ]
    turns = turns.reindex(rets.index)
    ff = load_ff_factors()
    ew = rets["ew_universe"]

    rows = []
    for name in PORTFOLIOS:
        if name not in rets.columns:
            continue
        for label, series in (
            ("gross", rets[name] - ew),
            ("net10", apply_costs(rets[name], turns[name], 10.0) - apply_costs(ew, turns["ew_universe"], 10.0)),
        ):
            a = ff6_alpha(series, ff)
            row = {
                "name": name,
                "cost": label,
                "alpha_ann": a["alpha_ann"],
                "alpha_t": a["alpha_t"],
                "n": a.get("n"),
                "r2": a.get("r2"),
            }
            for col in FF6_COLS:
                row[f"beta_{col}"] = a.get(f"beta_{col}")
                row[f"t_{col}"] = a.get(f"t_{col}")
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "active_alpha.csv", index=False)
    print(df[["name", "cost", "alpha_ann", "alpha_t", "beta_Mkt_RF"]].to_string(index=False))


if __name__ == "__main__":
    main()
