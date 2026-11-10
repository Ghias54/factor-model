"""Lasso and Ridge regressors for the walk-forward engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge

from factor_model.models.base import BaseModel

# Wide log grids: strongest Lasso alpha should zero most of 26 features.
LASSO_ALPHAS = list(np.logspace(-5, 1, 13))
RIDGE_ALPHAS = list(np.logspace(-3, 3, 13))


class LassoModel(BaseModel):
    name = "lasso"
    param_grid = {"alpha": LASSO_ALPHAS}

    def __init__(self, alpha: float = 0.01, max_iter: int = 10_000, **kwargs) -> None:
        super().__init__(alpha=float(alpha), max_iter=int(max_iter), **kwargs)
        self._model: Lasso | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> LassoModel:
        del X_val, y_val
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        self._model = Lasso(
            alpha=self.params["alpha"],
            max_iter=self.params["max_iter"],
            fit_intercept=True,
            random_state=42,
        )
        self._model.fit(X, y)
        names = self.feature_names_ or [f"f{i}" for i in range(X.shape[1])]
        self.coefficients_ = pd.Series(self._model.coef_, index=names, name="coef")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LassoModel.fit must be called first")
        return self._model.predict(X)


class RidgeModel(BaseModel):
    name = "ridge"
    param_grid = {"alpha": RIDGE_ALPHAS}

    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(alpha=float(alpha), **kwargs)
        self._model: Ridge | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> RidgeModel:
        del X_val, y_val
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        self._model = Ridge(alpha=self.params["alpha"], fit_intercept=True, random_state=42)
        self._model.fit(X, y)
        names = self.feature_names_ or [f"f{i}" for i in range(X.shape[1])]
        self.coefficients_ = pd.Series(self._model.coef_, index=names, name="coef")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RidgeModel.fit must be called first")
        return self._model.predict(X)
