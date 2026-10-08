"""Save/load fitted models with the metadata needed to use them later."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import joblib

from cryptopredict import __version__
from cryptopredict.core.types import DEFAULT_HORIZON, Forecaster

FORMAT_VERSION = 1


class SavedModel(NamedTuple):
    model: Forecaster
    metadata: dict[str, Any]


def save_model(
    model: Forecaster,
    path: str | Path,
    *,
    feature_columns: list[str] | None = None,
    horizon: int = DEFAULT_HORIZON,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dump ``model`` and its metadata to ``path`` (joblib); returns the metadata.

    ``feature_columns`` defaults to the columns the model was fit on.
    ``extra`` is stored as-is (e.g. symbol, interval, training period).
    """
    if feature_columns is None:
        feature_columns = getattr(model, "feature_names_", None)
    if feature_columns is None:
        raise ValueError("feature_columns unknown: fit the model first or pass them explicitly")
    metadata = {
        "format_version": FORMAT_VERSION,
        "name": model.name,
        "feature_columns": list(feature_columns),
        "horizon": int(horizon),
        "cryptopredict_version": __version__,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, path)
    return metadata


def load_model(path: str | Path) -> SavedModel:
    """Load a model saved by :func:`save_model`: ``model, metadata = load_model(p)``.

    Only load files you trust: joblib unpickling can execute arbitrary code.
    """
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or "model" not in payload or "metadata" not in payload:
        raise ValueError(f"{path} is not a cryptopredict model file")
    version = payload["metadata"].get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported model format version {version!r} (expected {FORMAT_VERSION})")
    return SavedModel(payload["model"], payload["metadata"])
