"""
tests/frontend/conftest.py — fixtures for Streamlit UI tests.

Uses the REAL streamlit package (not the stub from tests/unit/conftest.py).
AppTest drives app.py through the actual Streamlit runtime.
"""
import numpy as np
import pandas as pd
import pytest


APP_PATH = "app.py"


@pytest.fixture(scope="session")
def mock_prices():
    """Synthetic 2-year price DataFrame for mocking fetch_market_data."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=504)
    return pd.DataFrame({
        "AAPL": 150 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 504))),
        "MSFT": 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 504))),
    }, index=idx)


@pytest.fixture(scope="session")
def mock_volumes(mock_prices):
    """Synthetic volume data aligned to mock_prices."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        c: rng.integers(1_000_000, 50_000_000, len(mock_prices)).astype(float)
        for c in mock_prices.columns
    }, index=mock_prices.index)
