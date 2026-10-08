"""Feature engineering: technical indicators and target/dataset construction."""

from .dataset import forward_log_return, latest_features, make_dataset, make_target
from .indicators import (
    FeatureConfig,
    active_htf_intervals,
    build_features,
    infer_bar,
    required_history,
    resolve_config,
    segment_ids,
)

__all__ = [
    "FeatureConfig",
    "active_htf_intervals",
    "build_features",
    "forward_log_return",
    "infer_bar",
    "latest_features",
    "make_dataset",
    "make_target",
    "required_history",
    "resolve_config",
    "segment_ids",
]
