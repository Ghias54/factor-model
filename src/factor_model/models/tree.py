"""Random Forest and XGBoost regressors for the walk-forward engine."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from factor_model.models.base import BaseModel

# Modest grids — wall time on 24 cores.
RF_GRID = {
    "max_depth": [3, 5, 8],
    "min_samples_leaf": [100, 500, 2000],
    "max_features": [0.33, 0.5],
}
# 4 × 2 × 2 = 16 configs; early stopping caps trees.
XGB_GRID = {
    "max_depth": [2, 3, 4, 6],
    "learning_rate": [0.01, 0.05],
    "min_child_weight": [10, 50],
    "reg_lambda": [1.0],
}


class RandomForestModel(BaseModel):
    name = "random_forest"
    param_grid = RF_GRID

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 500,
        max_features: float = 0.33,
        n_estimators: int = 300,
        n_jobs: int = 4,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            max_depth=None if max_depth is None else int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            max_features=float(max_features),
            n_estimators=int(n_estimators),
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            **kwargs,
        )
        self._model: RandomForestRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> RandomForestModel:
        del X_val, y_val
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        self._model = RandomForestRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            max_features=self.params["max_features"],
            n_jobs=self.params["n_jobs"],
            random_state=self.params["random_state"],
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RandomForestModel.fit must be called first")
        return self._model.predict(X)


class XGBoostModel(BaseModel):
    name = "xgboost"
    param_grid = XGB_GRID
    needs_validation = True

    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_child_weight: float = 10.0,
        reg_lambda: float = 1.0,
        n_estimators: int = 500,
        subsample: float = 0.7,
        colsample_bytree: float = 0.7,
        early_stopping_rounds: int = 25,
        n_jobs: int = 4,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            min_child_weight=float(min_child_weight),
            reg_lambda=float(reg_lambda),
            n_estimators=int(n_estimators),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            early_stopping_rounds=int(early_stopping_rounds),
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            **kwargs,
        )
        self._model: XGBRegressor | None = None
        self.best_n_estimators_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> XGBoostModel:
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        n_est = int(self.params["n_estimators"])
        # If a prior tune stored best_n_estimators in params, use it (final refit).
        if "best_n_estimators" in self.params and self.params["best_n_estimators"] is not None:
            n_est = int(self.params["best_n_estimators"])
            use_es = False
        else:
            use_es = X_val is not None and y_val is not None

        self._model = XGBRegressor(
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            min_child_weight=self.params["min_child_weight"],
            reg_lambda=self.params["reg_lambda"],
            n_estimators=n_est,
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=self.params["n_jobs"],
            random_state=self.params["random_state"],
            early_stopping_rounds=self.params["early_stopping_rounds"] if use_es else None,
        )
        if use_es:
            self._model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            best = getattr(self._model, "best_iteration", None)
            if best is not None:
                # best_iteration is 0-indexed; n_estimators for refit is best+1
                self.best_n_estimators_ = int(best) + 1
                self.params["best_n_estimators"] = self.best_n_estimators_
        else:
            self._model.fit(X, y, verbose=False)
            self.best_n_estimators_ = n_est
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoostModel.fit must be called first")
        return self._model.predict(X)

    def clone(self, **params):
        # Drop best_n_estimators unless explicitly kept — fresh tune each year.
        merged = {**self.params, **params}
        if "best_n_estimators" not in params:
            merged.pop("best_n_estimators", None)
        return self.__class__(**merged)
