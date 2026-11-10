"""Expanding-window walk-forward engine.

Timing (locked design):
- Target ``y`` = cross-sectional pct-rank of ``ret_1m``, scaled to [-1, 1]
  (``2 * rank_pct - 1``), matching the panel feature scale. Panel column
  ``target_rank`` is the raw pct-rank in (0, 1]; we rescale here.
- Refit every January of year Y on formation months from sample start through
  December of Y-1. Predict every formation month of year Y (capped at
  ``MAX_FORMATION_MONTH`` = 2023-11).
- Within each refit, the last ``validation_months`` (default 36) of the training
  window are held out for hyperparameter selection by mean monthly rank IC.
  The chosen params are then refit on the full training window.
- Training labels for a prediction month t only use formations ``<= t-1``
  (label fully realized by t). Yearly schedule enforces this via train-through
  Dec Y-1 before predicting year Y.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def os_cpu_count() -> int:
    return os.cpu_count() or 1

from factor_model.factors.classic import MAX_FORMATION_MONTH, return_month
from factor_model.features.compute import FEATURE_NAMES
from factor_model.models.base import BaseModel

FIRST_OOS_YEAR = 2011
LAST_OOS_YEAR = 2023  # predictions stop at MAX_FORMATION_MONTH within 2023


def scale_target_rank(pct_rank: pd.Series) -> pd.Series:
    """Map panel ``target_rank`` in (0, 1] to [-1, 1]."""
    return 2.0 * pct_rank.astype(float) - 1.0


def monthly_rank_ic(pred: pd.Series, realized_ret: pd.Series, dates: pd.Series) -> float:
    """Mean cross-sectional Spearman IC of prediction vs realized return."""
    df = pd.DataFrame({"p": pred, "r": realized_ret, "d": dates}).dropna()
    if df.empty:
        return float("nan")
    ics = []
    for _, g in df.groupby("d", sort=False):
        if len(g) < 20:
            continue
        if g["p"].nunique(dropna=True) < 2 or g["r"].nunique(dropna=True) < 2:
            continue
        ics.append(g["p"].corr(g["r"], method="spearman"))
    return float(np.nanmean(ics)) if ics else float("nan")


def _param_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*vals)]


@dataclass
class YearFitRecord:
    year: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    first_pred: str
    last_pred: str
    chosen_params: dict[str, Any]
    validation_ic: float
    n_train: int
    n_val: int
    coefficients: dict[str, float] | None = None


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    year_fits: list[YearFitRecord] = field(default_factory=list)
    model_name: str = ""
    target_scaling: str = "2 * target_rank - 1  # matches feature [-1,1] scale"

    def to_parquet(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions.to_parquet(path, compression="zstd")

    def fits_to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "year": r.year,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "val_start": r.val_start,
                "val_end": r.val_end,
                "first_pred": r.first_pred,
                "last_pred": r.last_pred,
                "chosen_params": r.chosen_params,
                "validation_ic": r.validation_ic,
                "n_train": r.n_train,
                "n_val": r.n_val,
                "coefficients": r.coefficients,
            }
            for r in self.year_fits
        ]
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class WalkForwardEngine:
    """Yearly expanding-window walk-forward with IC-based hyperparameter search."""

    def __init__(
        self,
        *,
        feature_names: list[str] | None = None,
        first_oos_year: int = FIRST_OOS_YEAR,
        last_oos_year: int = LAST_OOS_YEAR,
        max_formation: pd.Timestamp = MAX_FORMATION_MONTH,
        validation_months: int = 36,
        sample_start: pd.Timestamp = pd.Timestamp("2006-01-31"),
        n_jobs: int = -1,
        random_seed: int = 42,
    ) -> None:
        self.feature_names = list(feature_names or FEATURE_NAMES)
        self.first_oos_year = int(first_oos_year)
        self.last_oos_year = int(last_oos_year)
        self.max_formation = pd.Timestamp(max_formation)
        self.validation_months = int(validation_months)
        self.sample_start = pd.Timestamp(sample_start)
        self.n_jobs = n_jobs
        self.random_seed = int(random_seed)

    def prepare_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Filter to pre-holdout rows with labels; attach scaled target."""
        df = panel.copy()
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
        df = df[df["date"] >= self.sample_start]
        df = df[df["date"] <= self.max_formation]
        df = df[df["ret_1m"].notna() & df["target_rank"].notna()].copy()
        df["y"] = scale_target_rank(df["target_rank"])
        df["return_month"] = return_month(df["date"])
        # Features: panel already fills missing → 0 after ranking
        for c in self.feature_names:
            if c not in df.columns:
                raise KeyError(f"missing feature column {c}")
            df[c] = df[c].astype(float).fillna(0.0)
        if (df["date"] > self.max_formation).any():
            raise AssertionError("prepare_panel leaked post-holdout formation months")
        return df.reset_index(drop=True)

    def _formation_calendar(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(df["date"].unique()))

    def _year_windows(
        self, calendar: pd.DatetimeIndex
    ) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]]:
        """Return (year, train_end, list of prediction formation months)."""
        out = []
        for year in range(self.first_oos_year, self.last_oos_year + 1):
            train_end_candidates = calendar[(calendar.year < year)]
            if len(train_end_candidates) == 0:
                continue
            train_end = train_end_candidates.max()
            # Prefer December of Y-1 when present
            dec = calendar[(calendar.year == year - 1) & (calendar.month == 12)]
            if len(dec):
                train_end = dec.max()
            preds = calendar[(calendar.year == year) & (calendar <= self.max_formation)]
            if len(preds) == 0:
                continue
            out.append((year, train_end, preds.min(), preds))
        return out

    def assert_timing(
        self,
        train_dates: Iterable[pd.Timestamp],
        first_pred: pd.Timestamp,
    ) -> None:
        """Hard timing guards used by tests and the runner.

        Formation dates are compared as timestamps. Return-month labels are
        compared at calendar-month granularity so a Dec label that maps to
        calendar Jan (e.g. 2015-01-31) is allowed when the first prediction
        formation is the January trading month-end (e.g. 2015-01-30).
        """
        train_dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(train_dates))))
        first_pred = pd.Timestamp(first_pred)
        if len(train_dates) == 0:
            raise AssertionError("empty training window")
        if train_dates.max() >= first_pred:
            raise AssertionError(
                f"max train formation {train_dates.max().date()} >= first pred {first_pred.date()}"
            )
        ret_months = return_month(pd.Series(train_dates))
        if ret_months.dt.to_period("M").max() > first_pred.to_period("M"):
            raise AssertionError(
                f"max train return_month {ret_months.max().date()} "
                f"after prediction month {first_pred.date()}"
            )

    def _tune(
        self,
        model: BaseModel,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        dates_tr: np.ndarray,
        X_va: np.ndarray,
        y_va: np.ndarray,
        ret_va: np.ndarray,
        dates_va: np.ndarray,
    ) -> tuple[dict[str, Any], float]:
        del dates_tr  # reserved for diagnostics
        grid = _param_product(model.param_grid)
        if len(grid) == 1 and grid[0] == {}:
            return {}, float("nan")

        pass_val = bool(getattr(model, "needs_validation", False))

        def _eval(params: dict[str, Any]) -> tuple[dict[str, Any], float]:
            m = model.clone(**params)
            if pass_val:
                m.fit(
                    X_tr,
                    y_tr,
                    feature_names=self.feature_names,
                    X_val=X_va,
                    y_val=y_va,
                )
            else:
                m.fit(X_tr, y_tr, feature_names=self.feature_names)
            pred = m.predict(X_va)
            ic = monthly_rank_ic(
                pd.Series(pred),
                pd.Series(ret_va),
                pd.Series(dates_va),
            )
            # Persist early-stopping length into chosen params for final refit
            out_params = dict(params)
            best_n = getattr(m, "best_n_estimators_", None)
            if best_n is not None:
                out_params["best_n_estimators"] = int(best_n)
            return out_params, ic

        # Tree/MLP fits are heavy; limit outer parallelism to avoid oversubscription.
        outer = self.n_jobs
        if model.name in {"random_forest", "xgboost", "mlp"}:
            outer = min(4, os_cpu_count())
        results = Parallel(n_jobs=outer)(delayed(_eval)(p) for p in grid)

        def _key(item: tuple[dict[str, Any], float]) -> tuple:
            params, ic = item
            ic_v = -np.inf if (ic is None or np.isnan(ic)) else ic
            alpha = params.get("alpha", 0.0)
            return (ic_v, alpha)

        best_params, best_ic = max(results, key=_key)
        return best_params, float(best_ic)

    def run(self, model: BaseModel, panel: pd.DataFrame) -> WalkForwardResult:
        df = self.prepare_panel(panel)
        calendar = self._formation_calendar(df)
        windows = self._year_windows(calendar)
        if not windows:
            raise RuntimeError("no walk-forward windows produced")

        pred_parts: list[pd.DataFrame] = []
        year_fits: list[YearFitRecord] = []

        for year, train_end, first_pred, pred_dates in windows:
            train_mask = (df["date"] >= self.sample_start) & (df["date"] <= train_end)
            train = df.loc[train_mask]
            train_months = pd.DatetimeIndex(sorted(train["date"].unique()))
            self.assert_timing(train_months, first_pred)

            if len(train_months) <= self.validation_months:
                raise RuntimeError(f"year {year}: not enough train months for validation")

            val_months = train_months[-self.validation_months :]
            fit_months = train_months[: -self.validation_months]
            tr = train[train["date"].isin(fit_months)]
            va = train[train["date"].isin(val_months)]

            X_tr = tr[self.feature_names].to_numpy(dtype=float)
            y_tr = tr["y"].to_numpy(dtype=float)
            X_va = va[self.feature_names].to_numpy(dtype=float)
            y_va = va["y"].to_numpy(dtype=float)
            ret_va = va["ret_1m"].to_numpy(dtype=float)

            chosen, val_ic = self._tune(
                model,
                X_tr,
                y_tr,
                tr["date"].to_numpy(),
                X_va,
                y_va,
                ret_va,
                va["date"].to_numpy(),
            )

            # Refit on full training window (still pre-OOS). Early-stopping monitors
            # the last validation_months of that window — never the predicted year.
            X_full = train[self.feature_names].to_numpy(dtype=float)
            y_full = train["y"].to_numpy(dtype=float)
            fitted = model.clone(**chosen)
            if getattr(model, "needs_validation", False) and "best_n_estimators" in chosen:
                # XGB: frozen n_estimators from tuning; fit on all train rows
                fitted.fit(X_full, y_full, feature_names=self.feature_names)
            elif getattr(model, "needs_validation", False):
                # MLP: train on full window, early-stop on the val tail
                fitted.fit(
                    X_full,
                    y_full,
                    feature_names=self.feature_names,
                    X_val=X_va,
                    y_val=y_va,
                )
            else:
                fitted.fit(X_full, y_full, feature_names=self.feature_names)

            pred_slice = df[df["date"].isin(pred_dates)].copy()
            if pred_slice["date"].max() > self.max_formation:
                raise AssertionError("prediction leaked past holdout")
            X_te = pred_slice[self.feature_names].to_numpy(dtype=float)
            pred_slice["prediction"] = fitted.predict(X_te)
            pred_parts.append(
                pred_slice[
                    [
                        "security_id",
                        "date",
                        "prediction",
                        "ret_1m",
                        "y",
                        "return_month",
                        "symbol",
                        "mcap",
                    ]
                ].rename(columns={"date": "formation_month", "ret_1m": "realized_ret"})
            )

            coefs = None
            if fitted.coefficients_ is not None:
                coefs = {k: float(v) for k, v in fitted.coefficients_.items()}

            year_fits.append(
                YearFitRecord(
                    year=year,
                    train_start=str(train_months.min().date()),
                    train_end=str(train_months.max().date()),
                    val_start=str(val_months.min().date()),
                    val_end=str(val_months.max().date()),
                    first_pred=str(pred_dates.min().date()),
                    last_pred=str(pred_dates.max().date()),
                    chosen_params=chosen,
                    validation_ic=val_ic,
                    n_train=int(len(tr)),
                    n_val=int(len(va)),
                    coefficients=coefs,
                )
            )

        preds = pd.concat(pred_parts, ignore_index=True)
        preds["formation_month"] = pd.to_datetime(preds["formation_month"]).astype("datetime64[ns]")
        if (preds["formation_month"] > self.max_formation).any():
            raise AssertionError("predictions contain formation months after holdout cap")
        return WalkForwardResult(predictions=preds, year_fits=year_fits, model_name=model.name)


def shuffle_target_within_month(
    panel: pd.DataFrame, *, seed: int = 0
) -> pd.DataFrame:
    """Permutation null: shuffle ``target_rank`` / ``ret_1m`` pairs within month.

    Actually shuffle the label ``target_rank`` (and keep ret for IC vs realized —
    for leakage test we shuffle y used for training while evaluating IC against
    true realized returns). Here we shuffle ``target_rank`` within each date so
    the training target is noise; ``ret_1m`` stays as the economic realization.
    """
    out = panel.copy()
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in out.groupby("date", sort=False):
        g = g.copy()
        idx = rng.permutation(len(g))
        g["target_rank"] = g["target_rank"].to_numpy()[idx]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
