"""Forecasting models: baselines, Ridge, gradient boosting, registry and persistence."""

from cryptopredict.models.baseline import LastReturn, MeanReturn, MovingAverageReturn, ZeroReturn
from cryptopredict.models.persistence import SavedModel, load_model, save_model
from cryptopredict.models.registry import available_models, get_model
from cryptopredict.models.sklearn_models import GBMForecaster, RidgeForecaster

__all__ = [
    "GBMForecaster",
    "LastReturn",
    "MeanReturn",
    "MovingAverageReturn",
    "RidgeForecaster",
    "SavedModel",
    "ZeroReturn",
    "available_models",
    "get_model",
    "load_model",
    "save_model",
]
