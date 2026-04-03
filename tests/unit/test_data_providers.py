"""
test_data_providers.py — Unit tests for the multi-source data layer.

Tests cover:
  - DataProviderBase interface
  - YFinanceProvider data normalization (mocked yfinance)
  - AlphaVantageProvider data normalization (mocked HTTP)
  - FREDProvider data normalization (mocked HTTP)
  - FinnhubProvider data normalization (mocked HTTP)
  - DataSourceOrchestrator cascading fallback
  - Orchestrator cache behavior
  - _api_get helper
  - API key validation / guard logic
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Import from the app module (via unit conftest stub)
import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n=10, start='2024-01-02'):
    """Create a simple OHLCV DataFrame matching provider output format."""
    idx = pd.bdate_range(start=start, periods=n)
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        'Open': close - rng.uniform(0, 1, n),
        'High': close + rng.uniform(0, 2, n),
        'Low': close - rng.uniform(0, 2, n),
        'Close': close,
        'Volume': rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


def _make_av_response(n=10):
    """Create a fake Alpha Vantage TIME_SERIES_DAILY response."""
    idx = pd.bdate_range(start='2024-01-02', periods=n)
    ts = {}
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    for i, dt in enumerate(idx):
        ts[dt.strftime('%Y-%m-%d')] = {
            '1. open': str(close[i] - 0.5),
            '2. high': str(close[i] + 1.0),
            '3. low': str(close[i] - 1.0),
            '4. close': str(close[i]),
            '5. volume': str(int(rng.integers(1_000_000, 10_000_000))),
        }
    return {'Time Series (Daily)': ts}


def _make_fred_response(n=10):
    """Create a fake FRED observations response."""
    idx = pd.bdate_range(start='2024-01-02', periods=n)
    rng = np.random.default_rng(42)
    vals = 3.0 + np.cumsum(rng.normal(0, 0.05, n))
    observations = [
        {'date': dt.strftime('%Y-%m-%d'), 'value': str(round(v, 2))}
        for dt, v in zip(idx, vals)
    ]
    return {'observations': observations}


def _make_finnhub_candle_response(n=10):
    """Create a fake Finnhub /stock/candle response."""
    start = int(datetime(2024, 1, 2).timestamp())
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return {
        's': 'ok',
        't': [start + i * 86400 for i in range(n)],
        'o': (close - 0.5).tolist(),
        'h': (close + 1.0).tolist(),
        'l': (close - 1.0).tolist(),
        'c': close.tolist(),
        'v': rng.integers(1_000_000, 10_000_000, n).tolist(),
    }


# ---------------------------------------------------------------------------
# 1. DataProviderBase
# ---------------------------------------------------------------------------

class TestDataProviderBase:

    def test_base_class_attributes(self):
        p = app.DataProviderBase()
        assert p.name == "base"
        assert p.requires_key is False
        assert p.api_key is None
        assert p.timeout == 15

    def test_fetch_prices_not_implemented(self):
        p = app.DataProviderBase()
        with pytest.raises(NotImplementedError):
            p.fetch_prices(['AAPL'])

    def test_fetch_fundamentals_not_implemented(self):
        p = app.DataProviderBase()
        with pytest.raises(NotImplementedError):
            p.fetch_fundamentals('AAPL')

    def test_fetch_quote_not_implemented(self):
        p = app.DataProviderBase()
        with pytest.raises(NotImplementedError):
            p.fetch_quote('AAPL')

    def test_health_check_not_implemented(self):
        p = app.DataProviderBase()
        with pytest.raises(NotImplementedError):
            p.health_check()

    def test_custom_api_key_and_timeout(self):
        p = app.DataProviderBase(api_key='test123', timeout=30)
        assert p.api_key == 'test123'
        assert p.timeout == 30


# ---------------------------------------------------------------------------
# 2. YFinanceProvider
# ---------------------------------------------------------------------------

class TestYFinanceProvider:

    def test_name_and_key_requirement(self):
        p = app.YFinanceProvider()
        assert p.name == "Yahoo Finance"
        assert p.requires_key is False

    @patch('app.yf.download')
    def test_fetch_prices_single_ticker(self, mock_dl):
        df = _make_ohlcv_df(5)
        mock_dl.return_value = df
        p = app.YFinanceProvider()
        result = p.fetch_prices(['AAPL'], period='1mo')
        assert 'AAPL' in result
        assert list(result['AAPL'].columns) == ['Open', 'High', 'Low', 'Close', 'Volume']

    @patch('app.yf.download')
    def test_fetch_prices_returns_empty_on_none(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        p = app.YFinanceProvider()
        result = p.fetch_prices(['AAPL'])
        assert result == {}

    @patch('app.yf.download')
    def test_fetch_prices_handles_exception(self, mock_dl):
        mock_dl.side_effect = Exception("network error")
        p = app.YFinanceProvider()
        with pytest.raises(app.DataFetchError):
            p.fetch_prices(['AAPL'])

    @patch('app.yf.download')
    def test_fetch_prices_string_ticker(self, mock_dl):
        df = _make_ohlcv_df(3)
        mock_dl.return_value = df
        p = app.YFinanceProvider()
        result = p.fetch_prices('AAPL', period='5d')
        assert 'AAPL' in result

    def test_fetch_fundamentals_returns_dict(self):
        p = app.YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            'trailingPE': 25.0, 'marketCap': 3e12,
            'dividendYield': 0.005, 'beta': 1.2,
            'trailingEps': 6.5, 'totalRevenue': 400e9,
            'profitMargins': 0.25, 'sector': 'Technology',
            'industry': 'Consumer Electronics',
        }
        with patch('app.yf.Ticker', return_value=mock_ticker):
            result = p.fetch_fundamentals('AAPL')
        assert result['pe_ratio'] == 25.0
        assert result['sector'] == 'Technology'

    def test_fetch_quote_returns_dict(self):
        p = app.YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            'currentPrice': 175.5, 'regularMarketChange': 2.3,
            'regularMarketChangePercent': 1.33,
            'regularMarketVolume': 50000000, 'marketCap': 3e12,
        }
        with patch('app.yf.Ticker', return_value=mock_ticker):
            result = p.fetch_quote('AAPL')
        assert result['price'] == 175.5
        assert result['source'] == 'Yahoo Finance'


# ---------------------------------------------------------------------------
# 3. AlphaVantageProvider
# ---------------------------------------------------------------------------

class TestAlphaVantageProvider:

    def test_name_and_key_requirement(self):
        p = app.AlphaVantageProvider(api_key='test')
        assert p.name == "Alpha Vantage"
        assert p.requires_key is True

    def test_fetch_prices_no_key_raises(self):
        p = app.AlphaVantageProvider()
        with pytest.raises(app.DataFetchError, match="API key not configured"):
            p.fetch_prices(['AAPL'])

    @patch('app._api_get')
    def test_fetch_prices_normalizes_data(self, mock_get):
        mock_get.return_value = _make_av_response(10)
        p = app.AlphaVantageProvider(api_key='testkey')
        result = p.fetch_prices(['AAPL'], period='1mo')
        assert 'AAPL' in result
        df = result['AAPL']
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing

    @patch('app._api_get')
    def test_fetch_prices_rate_limit_detection(self, mock_get):
        mock_get.return_value = {'Note': 'Thank you for using Alpha Vantage!'}
        p = app.AlphaVantageProvider(api_key='testkey')
        with pytest.raises(app.DataFetchError, match="rate limit"):
            p.fetch_prices(['AAPL'])

    @patch('app._api_get')
    def test_fetch_quote(self, mock_get):
        mock_get.return_value = {
            'Global Quote': {
                '05. price': '175.50',
                '09. change': '2.30',
                '10. change percent': '1.33%',
                '06. volume': '50000000',
            }
        }
        p = app.AlphaVantageProvider(api_key='testkey')
        result = p.fetch_quote('AAPL')
        assert result['price'] == 175.5
        assert result['source'] == 'Alpha Vantage'

    @patch('app._api_get')
    def test_fetch_fundamentals(self, mock_get):
        mock_get.return_value = {
            'Symbol': 'AAPL',
            'PERatio': '25.0',
            'MarketCapitalization': '3000000000000',
            'DividendYield': '0.005',
            'Beta': '1.2',
            'EPS': '6.5',
            'ProfitMargin': '0.25',
            'Sector': 'Technology',
            'Industry': 'Consumer Electronics',
        }
        p = app.AlphaVantageProvider(api_key='testkey')
        result = p.fetch_fundamentals('AAPL')
        assert result['pe_ratio'] == 25.0
        assert result['sector'] == 'Technology'

    def test_period_to_outputsize(self):
        p = app.AlphaVantageProvider(api_key='testkey')
        assert p._period_to_outputsize('1mo') == 'compact'
        assert p._period_to_outputsize('1y') == 'full'
        assert p._period_to_outputsize('5d') == 'compact'

    def test_period_to_days(self):
        assert app.AlphaVantageProvider._period_to_days('1y') == 365
        assert app.AlphaVantageProvider._period_to_days('max') is None
        assert app.AlphaVantageProvider._period_to_days('unknown') is None

    @patch('app._api_get')
    def test_health_check_success(self, mock_get):
        mock_get.return_value = {'Global Quote': {'05. price': '420.00'}}
        p = app.AlphaVantageProvider(api_key='testkey')
        assert p.health_check() is True

    @patch('app._api_get')
    def test_health_check_failure(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        p = app.AlphaVantageProvider(api_key='testkey')
        assert p.health_check() is False


# ---------------------------------------------------------------------------
# 4. FREDProvider
# ---------------------------------------------------------------------------

class TestFREDProvider:

    def test_name_and_key_requirement(self):
        p = app.FREDProvider(api_key='test')
        assert p.name == "FRED"
        assert p.requires_key is True

    def test_series_map_has_expected_keys(self):
        expected = {'treasury_10y', 'treasury_2y', 'treasury_3m', 'fed_funds',
                    'cpi', 'unemployment', 'gdp', 'vix', 'sp500'}
        assert set(app.FREDProvider.SERIES_MAP.keys()) == expected

    def test_fetch_macro_no_key_raises(self):
        p = app.FREDProvider()
        with pytest.raises(app.DataFetchError, match="API key not configured"):
            p.fetch_macro_series('DGS10')

    @patch('app._api_get')
    def test_fetch_macro_series_normalizes(self, mock_get):
        mock_get.return_value = _make_fred_response(10)
        p = app.FREDProvider(api_key='testkey')
        result = p.fetch_macro_series('DGS10', period='1y')
        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 10

    @patch('app._api_get')
    def test_fetch_macro_series_empty_obs(self, mock_get):
        mock_get.return_value = {'observations': []}
        p = app.FREDProvider(api_key='testkey')
        result = p.fetch_macro_series('DGS10')
        assert len(result) == 0

    @patch('app._api_get')
    def test_fetch_macro_series_skips_dots(self, mock_get):
        mock_get.return_value = {
            'observations': [
                {'date': '2024-01-02', 'value': '3.5'},
                {'date': '2024-01-03', 'value': '.'},
                {'date': '2024-01-04', 'value': '3.6'},
            ]
        }
        p = app.FREDProvider(api_key='testkey')
        result = p.fetch_macro_series('DGS10')
        assert len(result) == 2

    @patch('app._api_get')
    def test_health_check_success(self, mock_get):
        mock_get.return_value = _make_fred_response(5)
        p = app.FREDProvider(api_key='testkey')
        assert p.health_check() is True

    @patch('app._api_get')
    def test_health_check_failure(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        p = app.FREDProvider(api_key='testkey')
        assert p.health_check() is False


# ---------------------------------------------------------------------------
# 5. FinnhubProvider
# ---------------------------------------------------------------------------

class TestFinnhubProvider:

    def test_name_and_key_requirement(self):
        p = app.FinnhubProvider(api_key='test')
        assert p.name == "Finnhub"
        assert p.requires_key is True

    def test_fetch_prices_no_key_raises(self):
        p = app.FinnhubProvider()
        with pytest.raises(app.DataFetchError, match="API key not configured"):
            p.fetch_prices(['AAPL'])

    @patch('app._api_get')
    def test_fetch_prices_normalizes_data(self, mock_get):
        mock_get.return_value = _make_finnhub_candle_response(10)
        p = app.FinnhubProvider(api_key='testkey')
        result = p.fetch_prices(['AAPL'], period='1mo')
        assert 'AAPL' in result
        df = result['AAPL']
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert len(df) == 10

    @patch('app._api_get')
    def test_fetch_prices_no_data(self, mock_get):
        mock_get.return_value = {'s': 'no_data'}
        p = app.FinnhubProvider(api_key='testkey')
        result = p.fetch_prices(['AAPL'])
        assert result == {}

    @patch('app._api_get')
    def test_fetch_quote(self, mock_get):
        mock_get.return_value = {
            'c': 175.5, 'd': 2.3, 'dp': 1.33,
            'v': 50000000, 'h': 177.0, 'l': 173.0,
            'o': 174.0, 'pc': 173.2,
        }
        p = app.FinnhubProvider(api_key='testkey')
        result = p.fetch_quote('AAPL')
        assert result['price'] == 175.5
        assert result['source'] == 'Finnhub'

    @patch('app._api_get')
    def test_fetch_company_news(self, mock_get):
        news = [{'headline': 'Apple earnings beat', 'source': 'Reuters'}]
        mock_get.return_value = news
        p = app.FinnhubProvider(api_key='testkey')
        result = p.fetch_company_news('AAPL', days=7)
        assert len(result) == 1
        assert result[0]['headline'] == 'Apple earnings beat'

    def test_fetch_company_news_no_key(self):
        p = app.FinnhubProvider()
        result = p.fetch_company_news('AAPL')
        assert result == []

    @patch('app._api_get')
    def test_health_check_success(self, mock_get):
        mock_get.return_value = {'c': 175.5}
        p = app.FinnhubProvider(api_key='testkey')
        assert p.health_check() is True

    @patch('app._api_get')
    def test_health_check_failure(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        p = app.FinnhubProvider(api_key='testkey')
        assert p.health_check() is False

    def test_period_to_timestamps(self):
        p = app.FinnhubProvider(api_key='testkey')
        fr, to = p._period_to_timestamps('1y')
        assert to > fr
        assert (to - fr) >= 364 * 86400


# ---------------------------------------------------------------------------
# 6. DataSourceOrchestrator
# ---------------------------------------------------------------------------

class TestOrchestrator:

    def test_initial_state(self):
        orch = app.DataSourceOrchestrator()
        assert orch.providers == []
        assert orch.active_source is None
        assert orch.last_good_data == {}
        assert orch.fetch_log == []

    def test_configure_adds_yfinance_default(self):
        orch = app.DataSourceOrchestrator()
        orch.configure([])
        assert len(orch.providers) == 1
        assert isinstance(orch.providers[0], app.YFinanceProvider)

    def test_configure_adds_optional_providers(self):
        orch = app.DataSourceOrchestrator()
        orch.configure([
            {'name': 'alpha_vantage', 'enabled': True, 'api_key': 'ak'},
            {'name': 'finnhub', 'enabled': True, 'api_key': 'fk'},
        ])
        assert len(orch.providers) == 3
        assert isinstance(orch.providers[1], app.AlphaVantageProvider)
        assert isinstance(orch.providers[2], app.FinnhubProvider)

    def test_configure_skips_disabled(self):
        orch = app.DataSourceOrchestrator()
        orch.configure([
            {'name': 'alpha_vantage', 'enabled': False, 'api_key': 'ak'},
        ])
        assert len(orch.providers) == 1

    def test_configure_skips_missing_key(self):
        orch = app.DataSourceOrchestrator()
        orch.configure([
            {'name': 'alpha_vantage', 'enabled': True, 'api_key': ''},
        ])
        assert len(orch.providers) == 1

    def test_cascade_first_provider_succeeds(self):
        orch = app.DataSourceOrchestrator()
        mock_prov = MagicMock(spec=app.YFinanceProvider)
        mock_prov.name = "Mock Provider"
        mock_prov.fetch_prices.return_value = {'AAPL': _make_ohlcv_df(5)}
        orch.providers = [mock_prov]
        result = orch.fetch_prices(['AAPL'])
        assert 'AAPL' in result
        assert orch.active_source == "Mock Provider"

    def test_cascade_fallback_on_failure(self):
        orch = app.DataSourceOrchestrator()
        failing = MagicMock(spec=app.YFinanceProvider)
        failing.name = "Failing Provider"
        failing.fetch_prices.side_effect = Exception("fail")
        succeeding = MagicMock(spec=app.AlphaVantageProvider)
        succeeding.name = "Backup Provider"
        succeeding.fetch_prices.return_value = {'AAPL': _make_ohlcv_df(5)}
        orch.providers = [failing, succeeding]
        result = orch.fetch_prices(['AAPL'])
        assert 'AAPL' in result
        assert orch.active_source == "Backup Provider"
        assert len(orch.fetch_log) == 2
        assert 'failed' in orch.fetch_log[0][1]
        assert orch.fetch_log[1][1] == 'success'

    def test_cascade_returns_cache_when_all_fail(self):
        orch = app.DataSourceOrchestrator()
        cached_data = {'AAPL': _make_ohlcv_df(5)}
        orch.last_good_data['prices'] = cached_data
        orch.last_fetch_time = datetime.now() - timedelta(minutes=5)

        failing = MagicMock(spec=app.YFinanceProvider)
        failing.name = "Failing"
        failing.fetch_prices.side_effect = Exception("fail")
        orch.providers = [failing]

        result = orch.fetch_prices(['AAPL'])
        assert 'AAPL' in result
        assert 'Cached' in orch.active_source

    def test_cascade_raises_when_all_fail_no_cache(self):
        orch = app.DataSourceOrchestrator()
        failing = MagicMock(spec=app.YFinanceProvider)
        failing.name = "Failing"
        failing.fetch_prices.side_effect = Exception("fail")
        orch.providers = [failing]
        with pytest.raises(app.DataFetchError, match="All data sources failed"):
            orch.fetch_prices(['AAPL'])

    def test_cascade_skips_empty_results(self):
        orch = app.DataSourceOrchestrator()
        empty_prov = MagicMock(spec=app.YFinanceProvider)
        empty_prov.name = "Empty"
        empty_prov.fetch_prices.return_value = {}
        good_prov = MagicMock(spec=app.AlphaVantageProvider)
        good_prov.name = "Good"
        good_prov.fetch_prices.return_value = {'AAPL': _make_ohlcv_df(5)}
        orch.providers = [empty_prov, good_prov]
        result = orch.fetch_prices(['AAPL'])
        assert orch.active_source == "Good"

    def test_fetch_fundamentals_cascade(self):
        orch = app.DataSourceOrchestrator()
        failing = MagicMock(spec=app.YFinanceProvider)
        failing.name = "Failing"
        failing.fetch_fundamentals.side_effect = Exception("fail")
        succeeding = MagicMock(spec=app.AlphaVantageProvider)
        succeeding.name = "Good"
        succeeding.fetch_fundamentals.return_value = {'pe_ratio': 25.0}
        orch.providers = [failing, succeeding]
        result = orch.fetch_fundamentals('AAPL')
        assert result['pe_ratio'] == 25.0

    def test_fetch_fundamentals_all_fail(self):
        orch = app.DataSourceOrchestrator()
        failing = MagicMock(spec=app.YFinanceProvider)
        failing.name = "Failing"
        failing.fetch_fundamentals.side_effect = Exception("fail")
        orch.providers = [failing]
        result = orch.fetch_fundamentals('AAPL')
        assert result == {}

    def test_get_status_default(self):
        orch = app.DataSourceOrchestrator()
        status = orch.get_status()
        assert status['source'] == 'Yahoo Finance'
        assert status['connected'] is False

    def test_get_status_after_fetch(self):
        orch = app.DataSourceOrchestrator()
        mock_prov = MagicMock(spec=app.YFinanceProvider)
        mock_prov.name = "Yahoo Finance"
        mock_prov.fetch_prices.return_value = {'AAPL': _make_ohlcv_df(5)}
        orch.providers = [mock_prov]
        orch.fetch_prices(['AAPL'])
        status = orch.get_status()
        assert status['source'] == 'Yahoo Finance'
        assert status['connected'] is True

    def test_cache_age_no_cache(self):
        orch = app.DataSourceOrchestrator()
        assert orch._cache_age() == 'no cache'

    def test_cache_age_recent(self):
        orch = app.DataSourceOrchestrator()
        orch.last_fetch_time = datetime.now() - timedelta(seconds=30)
        age = orch._cache_age()
        assert 's ago' in age

    def test_cache_age_minutes(self):
        orch = app.DataSourceOrchestrator()
        orch.last_fetch_time = datetime.now() - timedelta(minutes=5)
        age = orch._cache_age()
        assert 'm ago' in age

    def test_cache_age_hours(self):
        orch = app.DataSourceOrchestrator()
        orch.last_fetch_time = datetime.now() - timedelta(hours=2)
        age = orch._cache_age()
        assert 'h ago' in age

    def test_fetch_log_records_events(self):
        orch = app.DataSourceOrchestrator()
        mock_prov = MagicMock(spec=app.YFinanceProvider)
        mock_prov.name = "Test"
        mock_prov.fetch_prices.return_value = {'AAPL': _make_ohlcv_df(5)}
        orch.providers = [mock_prov]
        orch.fetch_prices(['AAPL'])
        assert len(orch.fetch_log) == 1
        assert orch.fetch_log[0][0] == "Test"
        assert orch.fetch_log[0][1] == "success"


# ---------------------------------------------------------------------------
# 7. _api_get helper
# ---------------------------------------------------------------------------

class TestApiGet:

    @patch('app.urllib.request.urlopen')
    def test_successful_get(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"key": "value"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = app._api_get('https://example.com/api')
        assert result == {'key': 'value'}

    @patch('app.urllib.request.urlopen')
    def test_url_error_raises(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("DNS fail")
        with pytest.raises(app.DataFetchError, match="API request failed"):
            app._api_get('https://example.com/api')

    @patch('app.urllib.request.urlopen')
    def test_json_decode_error_raises(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'not json'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        with pytest.raises(app.DataFetchError, match="API request failed"):
            app._api_get('https://example.com/api')


# ---------------------------------------------------------------------------
# 8. fetch_macro_data
# ---------------------------------------------------------------------------

class TestFetchMacroData:

    @patch('app.st')
    def test_returns_empty_when_fred_disabled(self, mock_st):
        mock_st.session_state = {'fred_enabled': False, 'fred_api_key': ''}
        result = app.fetch_macro_data('treasury_10y')
        assert len(result) == 0

    @patch('app._api_get')
    @patch('app.st')
    def test_uses_fred_when_enabled(self, mock_st, mock_get):
        mock_st.session_state = {
            'fred_enabled': True,
            'fred_api_key': 'testkey',
            'av_enabled': False,
            'av_api_key': '',
            'finnhub_enabled': False,
            'finnhub_api_key': '',
        }
        mock_get.return_value = _make_fred_response(10)
        result = app.fetch_macro_data('treasury_10y', period='1y')
        assert len(result) == 10


# ---------------------------------------------------------------------------
# 9. API Key Validation
# ---------------------------------------------------------------------------

class TestApiKeyValidation:

    def test_av_no_key_fetch_fundamentals_returns_empty(self):
        p = app.AlphaVantageProvider()
        assert p.fetch_fundamentals('AAPL') == {}

    def test_av_no_key_fetch_quote_returns_empty(self):
        p = app.AlphaVantageProvider()
        assert p.fetch_quote('AAPL') == {}

    def test_finnhub_no_key_fetch_quote_returns_empty(self):
        p = app.FinnhubProvider()
        assert p.fetch_quote('AAPL') == {}

    def test_fred_no_key_yield_curve_returns_empty(self):
        p = app.FREDProvider()
        assert p.fetch_yield_curve() == {}
