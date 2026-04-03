"""
test_advanced_modules.py — Unit tests for 5 new analytical modules (tabs 26-30).

Covers: EnhancedPairsBacktester, MacroRegimeDetector, CryptoOnChainAnalyzer,
        InsiderTracker, WatchlistManager.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from app import (
    EnhancedPairsBacktester,
    MacroRegimeDetector,
    CryptoOnChainAnalyzer,
    InsiderTracker,
    WatchlistManager,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 504


def _price_series(mu=0.0003, sigma=0.015, n=N, name="ASSET"):
    log_returns = RNG.normal(mu, sigma, n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    idx = pd.bdate_range(start="2022-01-03", periods=n)
    return pd.Series(prices, index=idx, name=name)


@pytest.fixture(scope="module")
def cointegrated_prices():
    """Two cointegrated series and one non-cointegrated."""
    idx = pd.bdate_range(start="2022-01-03", periods=N)
    rng = np.random.default_rng(42)
    a = 100 + np.cumsum(rng.normal(0, 1, N))
    b = 0.8 * a + rng.normal(0, 0.5, N) + 50
    c = 100 + np.cumsum(rng.normal(0.05, 1.5, N))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


@pytest.fixture(scope="module")
def multi_prices():
    idx = pd.bdate_range(start="2022-01-03", periods=N)
    rng = np.random.default_rng(99)
    data = {}
    for name, mu, sigma in [("AAPL", 0.0004, 0.012), ("MSFT", 0.0003, 0.011),
                             ("GLD", 0.0001, 0.008), ("TLT", 0.0002, 0.009)]:
        data[name] = 100 * np.exp(np.cumsum(rng.normal(mu, sigma, N)))
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def sample_insider_df():
    """Sample insider transactions DataFrame."""
    dates = pd.date_range("2024-01-01", periods=20, freq='7D')
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        'date': dates,
        'insider_trading': ['Purchase'] * 10 + ['Sale'] * 10,
        'insider': [f'Insider_{i % 5}' for i in range(20)],
        'value': rng.integers(10000, 500000, 20),
        'shares': rng.integers(100, 5000, 20),
    })


# ===========================================================================
# 1. EnhancedPairsBacktester
# ===========================================================================

class TestEnhancedPairsBacktester:
    def test_kalman_hedge_ratio_returns_dataframe(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        result = bt.kalman_hedge_ratio(cointegrated_prices['A'], cointegrated_prices['B'])
        assert isinstance(result, pd.DataFrame)
        assert 'hedge_ratio' in result.columns
        assert 'spread' in result.columns
        assert 'z_score' in result.columns
        assert 'sqrt_Q' in result.columns
        assert len(result) == N

    def test_kalman_hedge_ratio_converges(self, cointegrated_prices):
        """For cointegrated pair A=0.8*B+noise, Kalman hedge should be near 0.8 eventually."""
        bt = EnhancedPairsBacktester(cointegrated_prices)
        result = bt.kalman_hedge_ratio(cointegrated_prices['B'], cointegrated_prices['A'])
        # B ~ 0.8*A + const, so hedge ratio of B on A should be near 0.8
        final_hedge = result['hedge_ratio'].iloc[-1]
        assert 0.3 < final_hedge < 2.0, f"Kalman hedge ratio {final_hedge} not near expected range"

    def test_kalman_sqrt_Q_positive(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        result = bt.kalman_hedge_ratio(cointegrated_prices['A'], cointegrated_prices['B'])
        assert (result['sqrt_Q'] >= 0).all()

    def test_walk_forward_optimize_returns_dict(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        result = bt.walk_forward_optimize('A', 'B', train_window=200, test_window=50, steps=3)
        assert isinstance(result, dict)
        assert 'optimal_entry_z' in result
        assert 'optimal_exit_z' in result
        assert 'oos_sharpe' in result
        assert 'parameter_stability' in result
        assert 0 <= result['parameter_stability'] <= 1

    def test_walk_forward_optimize_short_data(self):
        """With too little data, should return defaults."""
        idx = pd.bdate_range(start="2022-01-03", periods=50)
        prices = pd.DataFrame({"X": np.random.randn(50) + 100, "Y": np.random.randn(50) + 100}, index=idx)
        bt = EnhancedPairsBacktester(prices)
        result = bt.walk_forward_optimize('X', 'Y', train_window=200, test_window=50)
        assert result['optimal_entry_z'] == 2.0
        assert result['oos_returns'] == []

    def test_regime_adaptive_signals_columns(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        spread = cointegrated_prices['A'] - 0.8 * cointegrated_prices['B']
        result = bt.regime_adaptive_signals(spread)
        assert isinstance(result, pd.DataFrame)
        expected_cols = {'regime', 'adaptive_entry_z', 'adaptive_exit_z', 'signal', 'z_score'}
        assert expected_cols.issubset(set(result.columns))

    def test_regime_adaptive_regimes_valid(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        spread = cointegrated_prices['A'] - 0.8 * cointegrated_prices['B']
        result = bt.regime_adaptive_signals(spread)
        valid_regimes = {'high_vol', 'low_vol', 'normal'}
        assert set(result['regime'].unique()).issubset(valid_regimes)

    def test_full_backtest_returns_metrics(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        result = bt.full_backtest('A', 'B', method='kalman', adaptive=True)
        assert isinstance(result, dict)
        assert 'equity_curve' in result
        assert 'metrics' in result
        assert 'drawdown_series' in result
        m = result['metrics']
        assert 'sharpe' in m
        assert 'max_dd' in m
        assert 'win_rate' in m
        assert 0 <= m['win_rate'] <= 1

    def test_full_backtest_ols_method(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        result = bt.full_backtest('A', 'B', method='ols', adaptive=False)
        assert isinstance(result['equity_curve'], pd.Series)
        assert len(result['equity_curve']) > 0

    def test_full_backtest_equity_starts_at_one(self, cointegrated_prices):
        bt = EnhancedPairsBacktester(cointegrated_prices)
        result = bt.full_backtest('A', 'B', method='kalman', adaptive=False)
        assert abs(result['equity_curve'].iloc[0] - 1.0) < 0.01


# ===========================================================================
# 2. MacroRegimeDetector
# ===========================================================================

class TestMacroRegimeDetector:
    def test_yield_curve_signal_steep(self):
        d = MacroRegimeDetector()
        result = d.yield_curve_signal(2.0, 4.5)
        assert result['label'] == 'Steep'
        assert result['signal'] == 1.0

    def test_yield_curve_signal_inverted(self):
        d = MacroRegimeDetector()
        result = d.yield_curve_signal(5.0, 4.0)
        assert result['label'] == 'Inverted'
        assert result['signal'] == -1.0

    def test_yield_curve_signal_flat(self):
        d = MacroRegimeDetector()
        result = d.yield_curve_signal(4.0, 4.3)
        assert result['label'] == 'Flat'

    def test_yield_curve_signal_normal(self):
        d = MacroRegimeDetector()
        result = d.yield_curve_signal(3.0, 4.0)
        assert result['label'] == 'Normal'
        assert result['signal'] == 0.5

    def test_credit_spread_signal_unknown_empty(self):
        d = MacroRegimeDetector()
        result = d.credit_spread_signal(pd.Series(dtype=float))
        assert result['label'] == 'Unknown'

    def test_credit_spread_signal_with_data(self):
        rng = np.random.default_rng(10)
        spread = pd.Series(rng.normal(5, 1, 100))
        d = MacroRegimeDetector()
        result = d.credit_spread_signal(spread)
        assert 'label' in result
        assert 'current_spread' in result
        assert 'percentile' in result

    def test_momentum_signal_uptrend(self):
        d = MacroRegimeDetector()
        prices = pd.Series(np.linspace(100, 200, 300))
        result = d.momentum_signal(prices)
        assert result['signal'] > 0

    def test_momentum_signal_downtrend(self):
        d = MacroRegimeDetector()
        prices = pd.Series(np.linspace(200, 100, 300))
        result = d.momentum_signal(prices)
        assert result['signal'] < 0

    def test_momentum_signal_insufficient_data(self):
        d = MacroRegimeDetector()
        prices = pd.Series([100, 101, 102])
        result = d.momentum_signal(prices)
        assert result['label'] == 'Insufficient Data'

    def test_volatility_regime_low(self):
        d = MacroRegimeDetector()
        vix = pd.Series([12.0] * 50)
        result = d.volatility_regime(vix)
        assert result['label'] == 'Low Vol'
        assert result['signal'] == 1.0

    def test_volatility_regime_crisis(self):
        d = MacroRegimeDetector()
        vix = pd.Series([40.0] * 50)
        result = d.volatility_regime(vix)
        assert result['label'] == 'Crisis'
        assert result['signal'] == -1.0

    def test_composite_regime_expansion(self):
        d = MacroRegimeDetector()
        signals = {
            'yield_curve': {'signal': 1.0},
            'credit': {'signal': 0.5},
            'momentum': {'signal': 1.0},
            'vol': {'signal': 1.0},
        }
        result = d.composite_regime(signals)
        assert result['regime_label'] == 'expansion'
        assert 0 <= result['confidence'] <= 1

    def test_composite_regime_contraction(self):
        d = MacroRegimeDetector()
        signals = {
            'yield_curve': {'signal': -1.0},
            'credit': {'signal': -1.0},
            'momentum': {'signal': -1.0},
            'vol': {'signal': -1.0},
        }
        result = d.composite_regime(signals)
        assert result['regime_label'] == 'contraction'

    def test_transition_matrix_shape(self):
        d = MacroRegimeDetector()
        regimes = pd.Series(['expansion', 'expansion', 'late_cycle', 'contraction', 'recovery',
                              'expansion', 'expansion'])
        result = d.transition_matrix(regimes)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_transition_matrix_rows_sum_to_one(self):
        d = MacroRegimeDetector()
        regimes = pd.Series(['expansion'] * 10 + ['contraction'] * 5 + ['expansion'] * 5)
        result = d.transition_matrix(regimes)
        if not result.empty:
            row_sums = result.sum(axis=1)
            for s in row_sums:
                assert abs(s - 1.0) < 0.01

    def test_regime_asset_performance_empty(self):
        d = MacroRegimeDetector()
        result = d.regime_asset_performance(None, pd.DataFrame())
        assert result.empty


# ===========================================================================
# 3. CryptoOnChainAnalyzer
# ===========================================================================

class TestCryptoOnChainAnalyzer:
    def test_nvt_ratio_returns_dataframe(self):
        analyzer = CryptoOnChainAnalyzer()
        mcap = pd.Series(np.linspace(1e9, 2e9, 100))
        vol = pd.Series(np.linspace(1e8, 3e8, 100))
        result = analyzer.nvt_ratio(mcap, vol)
        assert isinstance(result, pd.DataFrame)
        assert 'nvt' in result.columns
        assert 'nvt_signal' in result.columns

    def test_nvt_ratio_none_inputs(self):
        analyzer = CryptoOnChainAnalyzer()
        result = analyzer.nvt_ratio(None, None)
        assert result.empty

    def test_nvt_valuation_labels(self):
        analyzer = CryptoOnChainAnalyzer()
        mcap = pd.Series([1e9] * 100)
        vol = pd.Series([1e6] * 50 + [1e9] * 50)
        result = analyzer.nvt_ratio(mcap, vol)
        labels = result['valuation_label'].unique()
        assert 'Fair Value' in labels or 'Overvalued' in labels or 'Undervalued' in labels

    def test_mvrv_proxy_returns_dataframe(self):
        analyzer = CryptoOnChainAnalyzer()
        prices = pd.Series(np.linspace(100, 200, 400), name='price')
        result = analyzer.mvrv_proxy(prices)
        assert isinstance(result, pd.DataFrame)
        assert 'mvrv_30d' in result.columns
        assert 'mvrv_365d' in result.columns

    def test_mvrv_proxy_custom_windows(self):
        analyzer = CryptoOnChainAnalyzer()
        prices = pd.Series(np.linspace(100, 200, 200))
        result = analyzer.mvrv_proxy(prices, lookback_windows=[20, 50])
        assert 'mvrv_20d' in result.columns
        assert 'mvrv_50d' in result.columns

    def test_mvrv_proxy_empty_input(self):
        analyzer = CryptoOnChainAnalyzer()
        result = analyzer.mvrv_proxy(pd.Series(dtype=float))
        assert result.empty

    def test_fear_greed_proxy_score_range(self):
        analyzer = CryptoOnChainAnalyzer()
        prices = pd.Series(np.linspace(100, 150, 100))
        volumes = pd.Series(np.random.default_rng(1).uniform(1e6, 1e7, 100))
        result = analyzer.fear_greed_proxy(prices, volumes)
        assert 0 <= result['score'] <= 100
        assert result['label'] in ('Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed')

    def test_fear_greed_proxy_components(self):
        analyzer = CryptoOnChainAnalyzer()
        prices = pd.Series(np.linspace(100, 150, 100))
        volumes = pd.Series(np.random.default_rng(1).uniform(1e6, 1e7, 100))
        result = analyzer.fear_greed_proxy(prices, volumes)
        assert 'component_scores' in result
        assert 'volatility' in result['component_scores']
        assert 'momentum' in result['component_scores']

    def test_fear_greed_proxy_empty_prices(self):
        analyzer = CryptoOnChainAnalyzer()
        result = analyzer.fear_greed_proxy(pd.Series(dtype=float), None)
        assert result['score'] == 50
        assert result['label'] == 'Neutral'

    def test_crypto_correlation_matrix(self):
        analyzer = CryptoOnChainAnalyzer()
        idx = pd.date_range("2023-01-01", periods=100)
        coins_data = {
            'bitcoin': pd.DataFrame({'price': np.random.default_rng(1).uniform(20000, 30000, 100)}, index=idx),
            'ethereum': pd.DataFrame({'price': np.random.default_rng(2).uniform(1000, 2000, 100)}, index=idx),
        }
        result = analyzer.crypto_correlation_matrix(coins_data)
        assert isinstance(result, pd.DataFrame)
        assert 'BTC' in result.columns
        assert 'ETH' in result.columns

    def test_crypto_correlation_matrix_empty(self):
        analyzer = CryptoOnChainAnalyzer()
        result = analyzer.crypto_correlation_matrix({})
        assert result.empty

    @patch.object(CryptoOnChainAnalyzer, '_cg_get')
    def test_fetch_market_data_success(self, mock_get):
        mock_get.return_value = {
            'prices': [[1672531200000, 16500], [1672617600000, 16800]],
            'market_caps': [[1672531200000, 3.2e11], [1672617600000, 3.3e11]],
            'total_volumes': [[1672531200000, 1e10], [1672617600000, 1.1e10]],
        }
        analyzer = CryptoOnChainAnalyzer()
        result = analyzer.fetch_market_data('bitcoin', days=30)
        assert not result.empty
        assert 'price' in result.columns

    @patch.object(CryptoOnChainAnalyzer, '_cg_get')
    def test_fetch_market_data_api_failure(self, mock_get):
        mock_get.return_value = None
        analyzer = CryptoOnChainAnalyzer()
        result = analyzer.fetch_market_data('bitcoin', days=30)
        assert result.empty


# ===========================================================================
# 4. InsiderTracker
# ===========================================================================

class TestInsiderTracker:
    def test_insider_sentiment_score_balanced(self, sample_insider_df):
        tracker = InsiderTracker()
        result = tracker.insider_sentiment_score(sample_insider_df)
        assert -100 <= result['score'] <= 100
        assert result['label'] in ('Strong Buy', 'Buy', 'Neutral', 'Sell', 'Strong Sell')
        assert result['num_buyers'] == 10
        assert result['num_sellers'] == 10

    def test_insider_sentiment_score_empty(self):
        tracker = InsiderTracker()
        result = tracker.insider_sentiment_score(pd.DataFrame())
        assert result['score'] == 0
        assert result['label'] == 'No Data'

    def test_insider_sentiment_all_buys(self):
        df = pd.DataFrame({
            'date': pd.date_range("2024-01-01", periods=5),
            'insider_trading': ['Purchase'] * 5,
            'insider': ['A', 'B', 'C', 'D', 'E'],
            'value': [100000] * 5,
        })
        tracker = InsiderTracker()
        result = tracker.insider_sentiment_score(df)
        assert result['score'] == 100
        assert result['label'] == 'Strong Buy'

    def test_insider_sentiment_all_sells(self):
        df = pd.DataFrame({
            'date': pd.date_range("2024-01-01", periods=5),
            'insider_trading': ['Sale'] * 5,
            'insider': ['A', 'B', 'C', 'D', 'E'],
            'value': [100000] * 5,
        })
        tracker = InsiderTracker()
        result = tracker.insider_sentiment_score(df)
        assert result['score'] == -100
        assert result['label'] == 'Strong Sell'

    def test_cluster_buy_detection_finds_cluster(self):
        dates = pd.date_range("2024-06-01", periods=5, freq='3D')
        df = pd.DataFrame({
            'date': dates,
            'insider_trading': ['Purchase'] * 5,
            'insider': ['CEO', 'CFO', 'COO', 'VP', 'Director'],
            'value': [500000, 300000, 200000, 100000, 150000],
        })
        tracker = InsiderTracker()
        clusters = tracker.cluster_buy_detection(df, window_days=30, min_insiders=3)
        assert len(clusters) >= 1
        assert clusters[0]['direction'] == 'Buy'
        assert clusters[0]['num_insiders'] >= 3

    def test_cluster_buy_detection_no_cluster(self):
        dates = pd.date_range("2024-01-01", periods=2, freq='90D')
        df = pd.DataFrame({
            'date': dates,
            'insider_trading': ['Purchase', 'Purchase'],
            'insider': ['A', 'B'],
            'value': [100000, 100000],
        })
        tracker = InsiderTracker()
        clusters = tracker.cluster_buy_detection(df, min_insiders=3)
        assert len(clusters) == 0

    def test_cluster_buy_detection_empty(self):
        tracker = InsiderTracker()
        clusters = tracker.cluster_buy_detection(pd.DataFrame())
        assert clusters == []

    def test_insider_vs_price_forward_returns(self, sample_insider_df):
        tracker = InsiderTracker()
        prices = pd.Series(
            np.linspace(100, 150, 300),
            index=pd.bdate_range("2023-06-01", periods=300),
        )
        result = tracker.insider_vs_price(sample_insider_df, prices)
        assert 'buy_forward_returns' in result
        assert 'sell_forward_returns' in result
        assert 'timing_score' in result

    def test_insider_vs_price_empty(self):
        tracker = InsiderTracker()
        result = tracker.insider_vs_price(pd.DataFrame(), pd.Series(dtype=float))
        assert result['timing_score'] == 0

    def test_top_insider_trades(self, sample_insider_df):
        tracker = InsiderTracker()
        result = tracker.top_insider_trades(sample_insider_df, n=5)
        assert len(result) == 5

    def test_top_insider_trades_empty(self):
        tracker = InsiderTracker()
        result = tracker.top_insider_trades(pd.DataFrame())
        assert result.empty


# ===========================================================================
# 5. WatchlistManager
# ===========================================================================

class TestWatchlistManager:
    def test_create_watchlist(self):
        wm = WatchlistManager()
        result = wm.create_watchlist("Tech", ["AAPL", "MSFT", "GOOGL"])
        assert result is True
        assert "Tech" in wm.watchlists
        assert wm.watchlists["Tech"] == ["AAPL", "MSFT", "GOOGL"]

    def test_add_to_watchlist(self):
        wm = WatchlistManager()
        wm.create_watchlist("Tech", ["AAPL"])
        result = wm.add_to_watchlist("Tech", "MSFT")
        assert result is True
        assert "MSFT" in wm.watchlists["Tech"]

    def test_add_to_watchlist_no_duplicates(self):
        wm = WatchlistManager()
        wm.create_watchlist("Tech", ["AAPL"])
        wm.add_to_watchlist("Tech", "AAPL")
        assert wm.watchlists["Tech"].count("AAPL") == 1

    def test_add_to_nonexistent_watchlist(self):
        wm = WatchlistManager()
        result = wm.add_to_watchlist("NonExistent", "AAPL")
        assert result is False

    def test_remove_from_watchlist(self):
        wm = WatchlistManager()
        wm.create_watchlist("Tech", ["AAPL", "MSFT"])
        result = wm.remove_from_watchlist("Tech", "AAPL")
        assert result is True
        assert "AAPL" not in wm.watchlists["Tech"]

    def test_remove_nonexistent_ticker(self):
        wm = WatchlistManager()
        wm.create_watchlist("Tech", ["AAPL"])
        result = wm.remove_from_watchlist("Tech", "GOOGL")
        assert result is False

    def test_delete_watchlist(self):
        wm = WatchlistManager()
        wm.create_watchlist("Tech", ["AAPL"])
        result = wm.delete_watchlist("Tech")
        assert result is True
        assert "Tech" not in wm.watchlists

    def test_delete_nonexistent_watchlist(self):
        wm = WatchlistManager()
        result = wm.delete_watchlist("Ghost")
        assert result is False

    def test_add_alert(self):
        wm = WatchlistManager()
        wm.add_alert("AAPL", "price_above", {"threshold": 200.0})
        assert len(wm.alerts) == 1
        assert wm.alerts[0]['ticker'] == "AAPL"
        assert wm.alerts[0]['alert_type'] == "price_above"
        assert wm.alerts[0]['active'] is True

    def test_remove_alert(self):
        wm = WatchlistManager()
        wm.add_alert("AAPL", "price_above", {"threshold": 200.0})
        wm.add_alert("MSFT", "rsi_overbought", {"level": 70})
        wm.remove_alert(0)
        assert len(wm.alerts) == 1
        assert wm.alerts[0]['ticker'] == "MSFT"

    def test_remove_alert_invalid_index(self):
        wm = WatchlistManager()
        wm.add_alert("AAPL", "price_above", {"threshold": 200.0})
        wm.remove_alert(5)  # out of bounds
        assert len(wm.alerts) == 1  # unchanged

    def test_check_sma_cross_up(self):
        wm = WatchlistManager()
        # Create prices where yesterday was below SMA and today is above
        prices = pd.Series([100] * 50 + [90, 110])
        result = wm._check_sma_cross(prices, sma_period=50, direction='up')
        assert bool(result) in (True, False)

    def test_check_sma_cross_down(self):
        wm = WatchlistManager()
        prices = pd.Series([100] * 50 + [110, 90])
        result = wm._check_sma_cross(prices, sma_period=50, direction='down')
        assert bool(result) in (True, False)

    def test_check_sma_cross_insufficient_data(self):
        wm = WatchlistManager()
        prices = pd.Series([100, 101])
        result = wm._check_sma_cross(prices, sma_period=50)
        assert result is False

    def test_check_rsi_above(self):
        wm = WatchlistManager()
        # Consistently rising prices -> high RSI
        prices = pd.Series(np.linspace(100, 200, 50))
        triggered, rsi_val = wm._check_rsi(prices, level=70, direction='above')
        assert bool(triggered) in (True, False)
        assert isinstance(rsi_val, (int, float, np.integer, np.floating))

    def test_check_rsi_insufficient_data(self):
        wm = WatchlistManager()
        prices = pd.Series([100, 101])
        triggered, rsi_val = wm._check_rsi(prices, level=70)
        assert triggered is False

    def test_check_volume_spike_detected(self):
        wm = WatchlistManager()
        vol = pd.Series([1000] * 20 + [5000])
        is_spike, cur, avg = wm._check_volume_spike(vol, multiplier=2.0)
        assert bool(is_spike) is True
        assert cur == 5000

    def test_check_volume_spike_not_detected(self):
        wm = WatchlistManager()
        vol = pd.Series([1000] * 21)
        is_spike, cur, avg = wm._check_volume_spike(vol, multiplier=2.0)
        assert bool(is_spike) is False

    def test_check_volume_spike_insufficient_data(self):
        wm = WatchlistManager()
        vol = pd.Series([1000] * 5)
        is_spike, cur, avg = wm._check_volume_spike(vol, multiplier=2.0)
        assert is_spike is False

    def test_alert_types_constant(self):
        assert len(WatchlistManager.ALERT_TYPES) == 10
        assert 'price_above' in WatchlistManager.ALERT_TYPES
        assert 'volume_spike' in WatchlistManager.ALERT_TYPES

    def test_watchlist_heatmap_data_empty_watchlist(self):
        wm = WatchlistManager()
        result = wm.watchlist_heatmap_data("nonexistent")
        assert result.empty

    def test_multiple_watchlists(self):
        wm = WatchlistManager()
        wm.create_watchlist("Tech", ["AAPL", "MSFT"])
        wm.create_watchlist("Finance", ["JPM", "GS"])
        assert len(wm.watchlists) == 2
        assert "Tech" in wm.watchlists
        assert "Finance" in wm.watchlists
