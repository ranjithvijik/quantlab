"""
test_new_modules.py — Unit tests for the 6 new analytical modules (tabs 20-25).

Covers: PairsTradingAnalyzer, SectorRotationAnalyzer, EventAnalyzer,
        TailRiskAnalyzer, CrossAssetAnalyzer, ESGAnalyzer.
"""
import numpy as np
import pandas as pd
import pytest
from app import (
    PairsTradingAnalyzer,
    SectorRotationAnalyzer,
    EventAnalyzer,
    TailRiskAnalyzer,
    CrossAssetAnalyzer,
    ESGAnalyzer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(123)
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
    # Random walk for asset A
    a = 100 + np.cumsum(rng.normal(0, 1, N))
    # B = 0.8 * A + small noise (cointegrated with A)
    b = 0.8 * a + rng.normal(0, 0.5, N) + 50
    # C = independent random walk
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
def returns_series():
    s = _price_series(name="TEST")
    return s.pct_change().dropna()


# ===========================================================================
# 1. PairsTradingAnalyzer
# ===========================================================================

class TestPairsTradingAnalyzer:
    def test_find_cointegrated_pairs(self, cointegrated_prices):
        analyzer = PairsTradingAnalyzer(cointegrated_prices)
        pairs = analyzer.find_cointegrated_pairs(significance=0.10)
        assert isinstance(pairs, list)
        # A and B should be cointegrated
        pair_tickers = [(p[0], p[1]) for p in pairs]
        assert any((t1 in ("A", "B") and t2 in ("A", "B")) for t1, t2 in pair_tickers), \
            f"Expected A/B cointegration, got {pair_tickers}"

    def test_calculate_spread(self, cointegrated_prices):
        analyzer = PairsTradingAnalyzer(cointegrated_prices)
        spread = analyzer.calculate_spread("A", "B", lookback=60)
        assert isinstance(spread, pd.DataFrame)
        assert "spread" in spread.columns
        assert "z_score" in spread.columns
        assert len(spread) > 0

    def test_half_life_positive(self, cointegrated_prices):
        analyzer = PairsTradingAnalyzer(cointegrated_prices)
        spread_data = analyzer.calculate_spread("A", "B", lookback=60)
        hl = analyzer.half_life(spread_data["spread"].dropna())
        # For cointegrated pair, half-life should be finite and positive
        if not np.isnan(hl):
            assert hl > 0, f"Half-life should be positive, got {hl}"

    def test_generate_signals(self, cointegrated_prices):
        analyzer = PairsTradingAnalyzer(cointegrated_prices)
        spread_data = analyzer.calculate_spread("A", "B", lookback=60)
        z = spread_data["z_score"].dropna()
        signals = analyzer.generate_signals(z, entry_z=2.0, exit_z=0.5)
        assert isinstance(signals, pd.Series)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_backtest_pair(self, cointegrated_prices):
        analyzer = PairsTradingAnalyzer(cointegrated_prices)
        result = analyzer.backtest_pair("A", "B", entry_z=2.0, exit_z=0.5, lookback=60)
        assert isinstance(result, dict)
        assert "total_return" in result
        assert "sharpe" in result
        assert "max_dd" in result
        assert "num_trades" in result
        assert "win_rate" in result

    def test_backtest_insufficient_data(self):
        idx = pd.bdate_range("2023-01-01", periods=10)
        prices = pd.DataFrame({"X": range(10), "Y": range(10)}, index=idx, dtype=float)
        analyzer = PairsTradingAnalyzer(prices)
        result = analyzer.backtest_pair("X", "Y")
        assert result["total_return"] == 0


# ===========================================================================
# 2. SectorRotationAnalyzer
# ===========================================================================

class TestSectorRotationAnalyzer:
    def test_sector_etfs_dict(self):
        analyzer = SectorRotationAnalyzer()
        assert len(analyzer.SECTOR_ETFS) == 11
        assert "XLK" in analyzer.SECTOR_ETFS

    def test_relative_strength(self, multi_prices):
        analyzer = SectorRotationAnalyzer()
        sector = multi_prices["AAPL"]
        bench = multi_prices["MSFT"]
        rs = analyzer.relative_strength(sector, bench, window=63)
        assert isinstance(rs, pd.Series)
        assert len(rs) == len(sector)

    def test_momentum_rankings(self, multi_prices):
        analyzer = SectorRotationAnalyzer()
        rankings = analyzer.momentum_rankings(multi_prices, windows=[21, 63])
        assert isinstance(rankings, pd.DataFrame)
        assert "composite" in rankings.columns
        assert len(rankings) == len(multi_prices.columns)

    def test_rotation_model(self, multi_prices):
        analyzer = SectorRotationAnalyzer()
        # Use sector ETF-like tickers for testing
        idx = multi_prices.index
        rng = np.random.default_rng(7)
        sector_data = pd.DataFrame({
            "XLK": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(idx)))),
            "XLP": 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.008, len(idx)))),
            "XLF": 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, len(idx)))),
            "XLU": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.007, len(idx)))),
        }, index=idx)
        regime = analyzer.rotation_model(sector_data)
        assert isinstance(regime, dict)
        assert "regime" in regime
        assert regime["regime"] in ("Risk-On", "Risk-Off", "Neutral", "Unknown")

    def test_sector_correlation_matrix(self, multi_prices):
        analyzer = SectorRotationAnalyzer()
        returns = multi_prices.pct_change().dropna()
        corr = analyzer.sector_correlation_matrix(returns)
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)


# ===========================================================================
# 3. EventAnalyzer
# ===========================================================================

class TestEventAnalyzer:
    def test_init(self):
        ea = EventAnalyzer("AAPL")
        assert ea.ticker == "AAPL"

    def test_seasonality_analysis(self):
        idx = pd.bdate_range("2020-01-01", periods=756)
        prices = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0.0003, 0.01, 756))), index=idx)
        ea = EventAnalyzer("TEST")
        season = ea.seasonality_analysis(prices)
        assert isinstance(season, pd.DataFrame)
        assert "Avg Return" in season.columns
        assert "Win Rate" in season.columns
        assert len(season) <= 12

    def test_earnings_drift_no_data(self):
        idx = pd.bdate_range("2023-01-01", periods=100)
        prices = pd.Series(range(100), index=idx, dtype=float)
        ea = EventAnalyzer("TEST")
        result = ea.earnings_drift(prices, [], window_pre=5, window_post=20)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_event_volatility(self):
        idx = pd.bdate_range("2022-01-01", periods=500)
        prices = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0.0003, 0.01, 500))), index=idx)
        ea = EventAnalyzer("TEST")
        # Use a date that's in the middle
        event_dates = [idx[100], idx[200], idx[300]]
        result = ea.event_volatility(prices, event_dates, window=10)
        assert isinstance(result, dict)
        assert "normal_vol" in result
        assert "event_vol" in result
        assert "vol_ratio" in result
        assert result["normal_vol"] > 0

    def test_earnings_surprise_impact_empty(self):
        ea = EventAnalyzer("TEST")
        idx = pd.bdate_range("2023-01-01", periods=100)
        prices = pd.Series(range(100), index=idx, dtype=float)
        result = ea.earnings_surprise_impact(prices, pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ===========================================================================
# 4. TailRiskAnalyzer
# ===========================================================================

class TestTailRiskAnalyzer:
    def test_fit_evt(self, returns_series):
        tra = TailRiskAnalyzer(returns_series)
        evt = tra.fit_evt(threshold_percentile=5)
        assert isinstance(evt, dict)
        assert "shape" in evt
        assert "scale" in evt
        assert "threshold" in evt

    def test_fit_evt_insufficient_data(self):
        tra = TailRiskAnalyzer(pd.Series([0.01, -0.01, 0.02]))
        evt = tra.fit_evt()
        assert np.isnan(evt["shape"])

    def test_drawdown_analysis(self, returns_series):
        tra = TailRiskAnalyzer(returns_series)
        episodes = tra.drawdown_analysis()
        assert isinstance(episodes, pd.DataFrame)
        if not episodes.empty:
            assert "Depth" in episodes.columns
            assert "Start" in episodes.columns
            # All depths should be negative
            assert (episodes["Depth"] < 0).all()

    def test_drawdown_distribution(self, returns_series):
        tra = TailRiskAnalyzer(returns_series)
        stats_result = tra.drawdown_distribution()
        assert isinstance(stats_result, dict)
        if stats_result:
            assert "avg_depth" in stats_result
            assert "max_depth" in stats_result
            assert stats_result["max_depth"] <= stats_result["avg_depth"]  # max is most negative

    def test_tail_dependence(self, multi_prices):
        returns = multi_prices.pct_change().dropna()
        tra = TailRiskAnalyzer(returns.iloc[:, 0])
        td = tra.tail_dependence(returns)
        assert isinstance(td, pd.DataFrame)
        assert td.shape[0] == td.shape[1]
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(td.values), 1.0, atol=1e-10)

    def test_return_distribution_analysis(self, returns_series):
        tra = TailRiskAnalyzer(returns_series)
        dist = tra.return_distribution_analysis()
        assert isinstance(dist, dict)
        assert "skewness" in dist
        assert "kurtosis" in dist
        assert "mean" in dist
        assert "std" in dist
        assert "qq_empirical" in dist
        assert "qq_theoretical" in dist

    def test_worst_periods(self, returns_series):
        tra = TailRiskAnalyzer(returns_series)
        worst = tra.worst_periods(5)
        assert isinstance(worst, pd.DataFrame)
        assert len(worst) == 5
        # Should be sorted ascending (worst first)
        assert worst["Return"].iloc[0] <= worst["Return"].iloc[-1]


# ===========================================================================
# 5. CrossAssetAnalyzer
# ===========================================================================

class TestCrossAssetAnalyzer:
    def test_class_constants(self):
        assert len(CrossAssetAnalyzer.MAJOR_PAIRS) == 6
        assert len(CrossAssetAnalyzer.CROSS_ASSETS) == 6

    def test_rolling_correlation(self, multi_prices):
        analyzer = CrossAssetAnalyzer(multi_prices)
        returns = multi_prices.pct_change().dropna()
        rc = analyzer.rolling_correlation(returns["AAPL"], returns["MSFT"], windows=[21, 63])
        assert isinstance(rc, pd.DataFrame)
        assert "21d" in rc.columns
        assert "63d" in rc.columns

    def test_dynamic_correlation_matrix(self, multi_prices):
        analyzer = CrossAssetAnalyzer(multi_prices)
        returns = multi_prices.pct_change().dropna()
        result = analyzer.dynamic_correlation_matrix(returns, window=63)
        assert "current" in result
        assert "long_term" in result
        assert result["current"].shape == result["long_term"].shape

    def test_correlation_regime_detection(self, multi_prices):
        analyzer = CrossAssetAnalyzer(multi_prices)
        returns = multi_prices.pct_change().dropna()
        rc = returns["AAPL"].rolling(63).corr(returns["MSFT"]).dropna()
        regimes = analyzer.correlation_regime_detection(rc)
        assert isinstance(regimes, pd.Series)
        assert set(regimes.unique()).issubset({"normal", "crisis", "divergent"})

    def test_cross_asset_momentum(self, multi_prices):
        analyzer = CrossAssetAnalyzer(multi_prices)
        mom = analyzer.cross_asset_momentum(multi_prices, window=63)
        assert isinstance(mom, pd.DataFrame)
        assert "Momentum" in mom.columns
        assert "Rank" in mom.columns
        assert len(mom) == len(multi_prices.columns)


# ===========================================================================
# 6. ESGAnalyzer
# ===========================================================================

class TestESGAnalyzer:
    def test_init_single_ticker(self):
        analyzer = ESGAnalyzer("AAPL")
        assert analyzer.tickers == ["AAPL"]

    def test_init_list_tickers(self):
        analyzer = ESGAnalyzer(["AAPL", "MSFT"])
        assert analyzer.tickers == ["AAPL", "MSFT"]

    def test_esg_peer_comparison_empty(self):
        analyzer = ESGAnalyzer(["AAPL"])
        empty_df = pd.DataFrame({"Ticker": ["AAPL"], "totalEsg": [np.nan]})
        result = analyzer.esg_peer_comparison(empty_df)
        assert isinstance(result, pd.DataFrame)

    def test_esg_peer_comparison_valid(self):
        analyzer = ESGAnalyzer(["A", "B", "C"])
        scores = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "totalEsg": [25.0, 15.0, 35.0],
        })
        result = analyzer.esg_peer_comparison(scores)
        assert "ESG Z-Score" in result.columns
        # Should be sorted by totalEsg descending
        assert result.iloc[0]["totalEsg"] >= result.iloc[-1]["totalEsg"]

    def test_esg_return_analysis(self, multi_prices):
        analyzer = ESGAnalyzer(list(multi_prices.columns))
        returns = multi_prices.pct_change().dropna()
        scores = pd.DataFrame({
            "Ticker": list(multi_prices.columns),
            "totalEsg": [30.0, 20.0, 10.0, 40.0],
        })
        result = analyzer.esg_return_analysis(returns, scores)
        assert isinstance(result, dict)
        if result:
            # Should have high and low ESG portfolios
            assert "high_esg_annual_return" in result or "low_esg_annual_return" in result

    def test_esg_return_analysis_insufficient_data(self):
        analyzer = ESGAnalyzer(["A"])
        returns = pd.DataFrame({"A": [0.01, -0.01, 0.02]})
        scores = pd.DataFrame({"Ticker": ["A"], "totalEsg": [25.0]})
        result = analyzer.esg_return_analysis(returns, scores)
        assert isinstance(result, dict)
        # Only one ticker, so not enough for split
        assert result == {}
