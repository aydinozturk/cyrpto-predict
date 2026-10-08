"""CSV persistence for contract-compliant OHLCV frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cryptopredict.core.types import INDEX_NAME, OHLCV_COLUMNS, validate_ohlcv


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate an OHLCV CSV whose ``open_time`` values are UTC."""
    source = Path(path)
    frame = pd.read_csv(source)
    if INDEX_NAME not in frame.columns:
        raise ValueError(f"CSV is missing required {INDEX_NAME!r} column")
    frame[INDEX_NAME] = pd.to_datetime(frame[INDEX_NAME], utc=True, errors="raise")
    for column in OHLCV_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return validate_ohlcv(frame.set_index(INDEX_NAME))


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Atomically save a validated OHLCV frame and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = validate_ohlcv(df)
    temporary = destination.with_name(f".{destination.name}.tmp")
    normalized.to_csv(
        temporary,
        index=True,
        index_label=INDEX_NAME,
        date_format="%Y-%m-%dT%H:%M:%S.%fZ",
    )
    temporary.replace(destination)
    return destination
