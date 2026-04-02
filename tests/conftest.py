"""
conftest.py — shared fixtures for all QuantLab tests.

Provides deterministic synthetic market data so tests run offline
without hitting Yahoo Finance.
"""
import sys
import os
import types

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub streamlit so that app.py can be imported without a live Streamlit
# server.  All st.* calls that run at module level are no-ops.
# ---------------------------------------------------------------------------

def _make_st_stub():
    """Return a minimal mock of the streamlit module."""
    st = types.ModuleType("streamlit")

    # session_state as a simple namespace so get/set works
    class _SS(dict):
        def __getattr__(self, k):
            return self.get(k)
        def __setattr__(self, k, v):
            self[k] = v
    st.session_state = _SS(theme="light", debug_mode=False)

    # No-op decorators / functions used at module level
    def _noop(*a, **kw): return None
    def _noop_decorator(fn=None, **kw):
        if fn is None:
            return lambda f: f
        return fn

    for attr in [
        "set_page_config", "error", "warning", "success", "info",
        "markdown", "write", "caption", "code", "spinner",
        "progress", "empty", "stop", "rerun", "divider",
        "sidebar", "columns", "tabs", "expander", "container",
        "button", "checkbox", "radio", "selectbox", "multiselect",
        "slider", "number_input", "text_area", "text_input", "toggle",
        "image", "plotly_chart", "dataframe", "metric",
    ]:
        setattr(st, attr, _noop)

    # cache_data decorator — just return the function unchanged
    st.cache_data = _noop_decorator
    st.cache_resource = _noop_decorator

    # sidebar should itself respond to attribute access with no-ops
    class _Sidebar:
        def __getattr__(self, k): return _noop
    st.sidebar = _Sidebar()

    return st


# Inject before importing app
sys.modules.setdefault("streamlit", _make_st_stub())

# Now safe to import the app module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app  # noqa: E402 — must come after stub injection


# ---------------------------------------------------------------------------
# Reusable synthetic data fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 504  # ~2 years of daily data


def _price_series(mu=0.0003, sigma=0.015, n=N, start="2022-01-03", name="ASSET"):
    """Create a realistic price series via geometric random walk."""
    log_returns = RNG.normal(mu, sigma, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    idx = pd.bdate_range(start=start, periods=n)
    return pd.Series(prices, index=idx, name=name)


@pytest.fixture(scope="session")
def single_price_series():
    return _price_series(name="AAPL")


@pytest.fixture(scope="session")
def multi_price_df():
    """DataFrame with 4 correlated synthetic assets."""
    n = N
    idx = pd.bdate_range(start="2022-01-03", periods=n)
    common = RNG.normal(0, 0.008, n)          # shared market factor
    data = {}
    params = [
        ("AAPL", 0.0004, 0.012, 0.6),
        ("MSFT", 0.0003, 0.011, 0.55),
        ("BTC",  0.0005, 0.030, 0.25),
        ("GC",   0.0002, 0.008, 0.10),
    ]
    for name, mu, idio, beta in params:
        r = beta * common + RNG.normal(mu, idio, n)
        data[name] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="session")
def volume_series(single_price_series):
    """Synthetic volume aligned to single_price_series."""
    n = len(single_price_series)
    vols = RNG.integers(1_000_000, 50_000_000, n).astype(float)
    return pd.Series(vols, index=single_price_series.index, name="Volume")


@pytest.fixture(scope="session")
def benchmark_series():
    """Synthetic S&P 500 benchmark."""
    return _price_series(mu=0.0003, sigma=0.010, name="^GSPC")


@pytest.fixture(scope="session")
def returns_series(single_price_series):
    return single_price_series.pct_change().dropna()


@pytest.fixture(scope="session")
def multi_returns_df(multi_price_df):
    return multi_price_df.pct_change().dropna()
