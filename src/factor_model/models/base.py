"""Model protocol for walk-forward learners (linear, trees, MLP, …)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Minimal fit/predict interface used by the walk-forward engine.

    Subclasses expose ``param_grid`` (dict of lists) for yearly tuning and a
    stable ``name``. After ``fit``, optional ``coefficients_`` (pd.Series indexed
    by feature) is read by the engine for linear models.
    """

    name: str = "base"
    param_grid: dict[str, list[Any]] = {}

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.coefficients_: pd.Series | None = None
        self.feature_names_: list[str] | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> BaseModel:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def set_params(self, **params: Any) -> BaseModel:
        self.params.update(params)
        return self

    def clone(self, **params: Any) -> BaseModel:
        """New instance of the same class with overridden params."""
        merged = {**self.params, **params}
        return self.__class__(**merged)
