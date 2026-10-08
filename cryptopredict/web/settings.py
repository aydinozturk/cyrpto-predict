"""Environment-backed settings for the dashboard service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cryptopredict.data.binance import DEFAULT_BASE_URL


SUPPORTED_INTERVALS = [
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
]


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration loaded from ``CRYPTOPREDICT_*`` variables."""

    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    binance_url: str = DEFAULT_BASE_URL
    dashboard_token: str | None = None
    default_symbol: str = "BTCUSDT"
    default_interval: str = "1h"
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        token = os.getenv("CRYPTOPREDICT_DASHBOARD_TOKEN") or None
        try:
            port = int(os.getenv("CRYPTOPREDICT_PORT", "8000"))
        except ValueError as exc:
            raise ValueError("CRYPTOPREDICT_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("CRYPTOPREDICT_PORT must be between 1 and 65535")
        return cls(
            data_dir=Path(os.getenv("CRYPTOPREDICT_DATA_DIR", "data")),
            model_dir=Path(os.getenv("CRYPTOPREDICT_MODEL_DIR", "models")),
            binance_url=os.getenv("CRYPTOPREDICT_BINANCE_URL", DEFAULT_BASE_URL),
            dashboard_token=token,
            default_symbol=os.getenv("CRYPTOPREDICT_DEFAULT_SYMBOL", "BTCUSDT").strip().upper(),
            default_interval=os.getenv("CRYPTOPREDICT_DEFAULT_INTERVAL", "1h").strip(),
            host=os.getenv("CRYPTOPREDICT_HOST", "0.0.0.0"),
            port=port,
        )
