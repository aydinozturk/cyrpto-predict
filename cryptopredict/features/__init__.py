"""Feature engineering: technical indicators and target/dataset construction."""

from .dataset import forward_log_return, latest_features, make_dataset, make_target
from .indicators import FeatureConfig, build_features, resolve_config

__all__ = [
    "FeatureConfig",
    "build_features",
    "forward_log_return",
    "latest_features",
    "make_dataset",
    "make_target",
    "resolve_config",
]
