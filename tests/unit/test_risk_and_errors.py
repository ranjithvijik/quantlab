"""
test_risk_and_errors.py — Tests for:
  - Risk score calculation
  - Custom exception hierarchy
  - show_error / handle_error (logic, not UI)
  - fetch_market_data error handling
  - Ticker parser
"""
import re
import pytest
import pandas as pd
import numpy as np
from app import (
    calculate_composite_risk_score,
    QuantLabError, DataFetchError, ValidationError, CalculationError, ExportError,
    handle_error,
    show_error,
)


# ---------------------------------------------------------------------------
# Composite Risk Score
# ---------------------------------------------------------------------------

class TestCompositeRiskScore:
    def _make_risk_data(self, vix_val=20, tnx_val=4.0, irx_val=4.5,
                        gold_ret_30d=0, vix_history_len=300):
        """Helper: build a minimal risk_data dict with controllable values."""
        # VIX series
        vix = pd.Series([vix_val] * vix_history_len)

        # Yield curve
        tnx = pd.Series([tnx_val] * 10)
        irx = pd.Series([irx_val] * 10)

        # Gold: 30-day return controlled via 2 data points
        gold_end = 100 * (1 + gold_ret_30d / 100)
        gold = pd.Series(
            [100.0] * 270 + [100.0] * 29 + [gold_end],
            dtype=float,
        )

        return {"VIX": vix, "TNX": tnx, "IRX": irx, "Gold": gold}

    def test_score_in_range(self):
        rd = self._make_risk_data()
        score = calculate_composite_risk_score(rd)
        assert 0 <= score <= 100

    def test_low_vix_low_score(self):
        rd = self._make_risk_data(vix_val=12)
        score = calculate_composite_risk_score(rd)
        assert score < 50

    def test_high_vix_raises_score(self):
        low_vix  = self._make_risk_data(vix_val=15)
        high_vix = self._make_risk_data(vix_val=40)
        assert calculate_composite_risk_score(high_vix) > \
               calculate_composite_risk_score(low_vix)

    def test_inverted_yield_curve_raises_score(self):
        normal   = self._make_risk_data(tnx_val=4.5, irx_val=3.0)  # spread > 0
        inverted = self._make_risk_data(tnx_val=3.0, irx_val=4.5)  # spread < 0
        assert calculate_composite_risk_score(inverted) > \
               calculate_composite_risk_score(normal)

    def test_gold_rally_raises_score(self):
        no_rally = self._make_risk_data(gold_ret_30d=0)
        rally    = self._make_risk_data(gold_ret_30d=15)
        assert calculate_composite_risk_score(rally) > \
               calculate_composite_risk_score(no_rally)

    def test_empty_risk_data_returns_zero(self):
        score = calculate_composite_risk_score({})
        assert score == 0.0

    def test_score_never_exceeds_100(self):
        """Extreme inputs should be capped at 100."""
        rd = self._make_risk_data(vix_val=80, tnx_val=1.0, irx_val=6.0,
                                  gold_ret_30d=30)
        assert calculate_composite_risk_score(rd) <= 100


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_data_fetch_is_quantlab_error(self):
        with pytest.raises(QuantLabError):
            raise DataFetchError("network fail")

    def test_validation_is_quantlab_error(self):
        with pytest.raises(QuantLabError):
            raise ValidationError("bad ticker")

    def test_calculation_is_quantlab_error(self):
        with pytest.raises(QuantLabError):
            raise CalculationError("div by zero")

    def test_export_is_quantlab_error(self):
        with pytest.raises(QuantLabError):
            raise ExportError("render failed")

    def test_fields_preserved(self):
        e = DataFetchError("internal", user_message="friendly", recovery_hint="retry")
        assert e.message == "internal"
        assert e.user_message == "friendly"
        assert e.recovery_hint == "retry"

    def test_user_message_defaults_to_message(self):
        e = CalculationError("something broke")
        assert e.user_message == "something broke"

    def test_str_representation(self):
        e = DataFetchError("network timeout")
        assert "network timeout" in str(e)


# ---------------------------------------------------------------------------
# handle_error decorator
# ---------------------------------------------------------------------------

class TestHandleErrorDecorator:
    def test_returns_value_on_success(self):
        @handle_error
        def good():
            return 42
        assert good() == 42

    def test_returns_none_on_data_fetch_error(self, capsys):
        @handle_error
        def fail():
            raise DataFetchError("no data")
        result = fail()
        assert result is None

    def test_returns_none_on_calculation_error(self, capsys):
        @handle_error
        def fail():
            raise CalculationError("bad math")
        result = fail()
        assert result is None

    def test_returns_none_on_generic_exception(self, capsys):
        @handle_error
        def fail():
            raise ValueError("unexpected")
        result = fail()
        assert result is None

    def test_preserves_function_name(self):
        @handle_error
        def my_special_function():
            return 1
        assert my_special_function.__name__ == "my_special_function"

    def test_passes_args_through(self):
        @handle_error
        def add(a, b):
            return a + b
        assert add(3, 4) == 7


# ---------------------------------------------------------------------------
# Ticker parser (re.split logic used in main())
# ---------------------------------------------------------------------------

class TestTickerParser:
    def _parse(self, text):
        tickers = [t for t in re.split(r'[\s,;]+', text.strip()) if t]
        return [re.sub(r'[^A-Za-z0-9.=^-]', '', t).upper() for t in tickers if t.strip()]

    def test_space_separated(self):
        assert self._parse("AAPL MSFT GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_comma_separated(self):
        assert self._parse("AAPL,MSFT,GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_comma_space_mixed(self):
        assert self._parse("AAPL, MSFT , GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_semicolon_separated(self):
        assert self._parse("AAPL;MSFT;GOOGL") == ["AAPL", "MSFT", "GOOGL"]

    def test_forex_ticker(self):
        result = self._parse("EURUSD=X")
        assert result == ["EURUSD=X"]

    def test_crypto_ticker(self):
        assert self._parse("BTC-USD") == ["BTC-USD"]

    def test_commodity_ticker(self):
        assert self._parse("GC=F") == ["GC=F"]

    def test_index_ticker(self):
        assert self._parse("^GSPC") == ["^GSPC"]

    def test_strips_special_chars(self):
        """Extra punctuation should be stripped."""
        result = self._parse("AAPL! MSFT@")
        assert "AAPL" in result
        assert "MSFT" in result

    def test_empty_input(self):
        assert self._parse("") == []

    def test_uppercase_normalisation(self):
        assert self._parse("aapl msft") == ["AAPL", "MSFT"]

    def test_duplicate_tickers_preserved(self):
        """Duplicates are the caller's responsibility to deduplicate."""
        result = self._parse("AAPL AAPL MSFT")
        assert result.count("AAPL") == 2


# ---------------------------------------------------------------------------
# fetch_market_data error paths (mock yfinance)
# ---------------------------------------------------------------------------

class TestFetchMarketDataErrors:
    """
    Because our conftest stubs st.cache_data as a pass-through decorator,
    fetch_market_data IS the raw function (no cache wrapper), so we call it directly.
    """

    def test_empty_tickers_raises(self):
        from app import fetch_market_data, DataFetchError
        with pytest.raises(DataFetchError):
            fetch_market_data([], "2023-01-01", "2024-01-01")

    def test_network_error_raises_data_fetch_error(self, monkeypatch):
        import app
        def _raise(*a, **kw):
            raise ConnectionError("network failure")
        monkeypatch.setattr(app.yf, "download", _raise)
        with pytest.raises(app.DataFetchError):
            app.fetch_market_data(["AAPL"], "2023-01-01", "2024-01-01")

    def test_rate_limit_raises_data_fetch_error(self, monkeypatch):
        import app
        def _raise(*a, **kw):
            raise Exception("rate limit exceeded 429")
        monkeypatch.setattr(app.yf, "download", _raise)
        with pytest.raises(app.DataFetchError):
            app.fetch_market_data(["AAPL"], "2023-01-01", "2024-01-01")

    def test_empty_response_raises_data_fetch_error(self, monkeypatch):
        import app
        monkeypatch.setattr(app.yf, "download", lambda *a, **kw: pd.DataFrame())
        with pytest.raises(app.DataFetchError):
            app.fetch_market_data(["FAKEXXXX"], "2023-01-01", "2024-01-01")
