"""Model lookup by short name.

Optional models are imported only when their availability is queried or they
are instantiated.  This keeps the core package usable without heavyweight ML
dependencies such as LightGBM and PyTorch.
"""

from __future__ import annotations

from importlib import import_module

from cryptopredict.core.types import Forecaster
from cryptopredict.models.baseline import LastReturn, MeanReturn, MovingAverageReturn, ZeroReturn
from cryptopredict.models.sklearn_models import GBMForecaster, RidgeForecaster

MODELS: dict[str, type] = {
    cls.name: cls
    for cls in (ZeroReturn, MeanReturn, LastReturn, MovingAverageReturn, RidgeForecaster, GBMForecaster)
}

LAZY_MODELS: dict[str, str] = {
    "lgbm": "cryptopredict.models.boosting:LGBMForecaster",
    "lgbm_cls": "cryptopredict.models.boosting:LGBMDirectionForecaster",
    "lstm": "cryptopredict.models.deep:LSTMForecaster",
    "gru": "cryptopredict.models.deep:GRUForecaster",
    "ensemble": "cryptopredict.models.ensemble:EnsembleForecaster",
    "stack": "cryptopredict.models.ensemble:StackingForecaster",
}

_MODEL_EXTRAS = {
    "lgbm": "ml",
    "lgbm_cls": "ml",
    "lstm": "deep",
    "gru": "deep",
    "ensemble": "ml",
    "stack": "ml",
}


def _lazy_class(name: str) -> type:
    module_name, separator, class_name = LAZY_MODELS[name].partition(":")
    if not separator or not module_name or not class_name:
        raise ImportError(f"invalid lazy model target {LAZY_MODELS[name]!r}")
    module = import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} does not define {class_name!r}") from exc


def _model_class(name: str) -> type:
    if name in MODELS:
        return MODELS[name]
    if name not in LAZY_MODELS:
        raise KeyError(name)
    return _lazy_class(name)


def available_models() -> list[str]:
    """Return registered models whose modules and dependencies are installed."""
    available = list(MODELS)
    for name in LAZY_MODELS:
        try:
            _lazy_class(name)
        except Exception:
            # A broken optional import must never make the core registry unusable.
            continue
        available.append(name)
    return available


def default_compare_models() -> list[str]:
    """Installed models suitable for the default, quick comparison suite."""
    return [
        name
        for name in available_models()
        if not bool(getattr(_model_class(name), "heavy", False))
    ]


def get_model(name: str, **kwargs) -> Forecaster:
    """Instantiate an unfitted model, e.g. ``get_model("ridge", alpha=10.0)``."""
    try:
        cls = _model_class(name)
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(available_models())}") from None
    except Exception as exc:
        extra = _MODEL_EXTRAS.get(name, "ml")
        raise ValueError(
            f"model {name!r} is unavailable ({exc}); install its optional dependencies with "
            f"`pip install cryptopredict[{extra}]`"
        ) from None
    return cls(**kwargs)
