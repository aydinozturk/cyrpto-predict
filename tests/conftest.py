from pathlib import Path

import pandas as pd
import pytest

from cryptopredict.core.types import INDEX_NAME, validate_ohlcv

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "btcusdt_1h_sample.csv"


def load_sample_ohlcv() -> pd.DataFrame:
    """Load the offline BTCUSDT 1h sample (600 closed bars from Binance)."""
    df = pd.read_csv(SAMPLE_CSV)
    df[INDEX_NAME] = pd.to_datetime(df[INDEX_NAME], utc=True)
    return validate_ohlcv(df.set_index(INDEX_NAME))


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """Validated BTCUSDT 1h OHLCV frame (fresh copy per test)."""
    return load_sample_ohlcv()
