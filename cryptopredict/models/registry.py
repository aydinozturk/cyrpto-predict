"""Model lookup by short name."""

from __future__ import annotations

from cryptopredict.core.types import Forecaster
from cryptopredict.models.baseline import LastReturn, MeanReturn, MovingAverageReturn, ZeroReturn
from cryptopredict.models.sklearn_models import GBMForecaster, RidgeForecaster

MODELS: dict[str, type] = {
    cls.name: cls
    for cls in (ZeroReturn, MeanReturn, LastReturn, MovingAverageReturn, RidgeForecaster, GBMForecaster)
}


def available_models() -> list[str]:
    """Registered model names: zero, mean, last, ma, ridge, gbm."""
    return list(MODELS)


def get_model(name: str, **kwargs) -> Forecaster:
    """Instantiate an unfitted model, e.g. ``get_model("ridge", alpha=10.0)``."""
    try:
        cls = MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(MODELS)}") from None
    return cls(**kwargs)
