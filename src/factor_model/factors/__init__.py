"""Classic factors: value, size, momentum, quality."""

from factor_model.factors.classic import (
    CLASSIC_FEATURES,
    MAX_FORMATION_MONTH,
    MAX_RETURN_MONTH,
    build_classic_factor_returns,
)

__all__ = [
    "CLASSIC_FEATURES",
    "MAX_FORMATION_MONTH",
    "MAX_RETURN_MONTH",
    "build_classic_factor_returns",
]
