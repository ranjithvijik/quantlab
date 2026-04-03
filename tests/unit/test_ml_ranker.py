"""
test_ml_ranker.py — Unit tests for MLAssetRanker (Tab 31).

All tests use synthetic price data — no yfinance calls.
"""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app import MLAssetRanker, ASSET_UNIVERSE


# ---------------------------------------------------------------------------
# Helpers — synthetic data generators
# ---------------------------------------------------------------------------

def _make_price_df(n_days=150, start_price=100.0, drift=0.0003, vol=0.015, seed=42, include_volume=True):
    """Create a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    log_returns = rng.normal(drift, vol, n_days)
    close = start_price * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame({
        'Open': close * (1 + rng.normal(0, 0.002, n_days)),
        'High': close * (1 + abs(rng.normal(0, 0.005, n_days))),
        'Low': close * (1 - abs(rng.normal(0, 0.005, n_days))),
        'Close': close,
    }, index=idx)
    if include_volume:
        df['Volume'] = rng.integers(500_000, 10_000_000, n_days).astype(float)
    return df


def _make_universe_prices(n_tickers=10, n_days=150, seed=42):
    """Create a dict of ticker -> OHLCV DataFrames."""
    rng = np.random.default_rng(seed)
    tickers = [f'TEST{i}' for i in range(n_tickers)]
    prices = {}
    for i, t in enumerate(tickers):
        drift = rng.normal(0.0003, 0.001)
        prices[t] = _make_price_df(n_days=n_days, drift=drift, seed=seed + i)
    return prices


def _small_universe():
    """Return a small test universe dict."""
    return {'stocks': ['TEST0', 'TEST1', 'TEST2'], 'etfs': ['TEST3', 'TEST4']}


# ---------------------------------------------------------------------------
# 1. Feature Computation Tests
# ---------------------------------------------------------------------------

class TestFeatureComputation:
    def test_momentum_features_correct(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert 'ret_5d' in features_df.columns
        assert 'ret_21d' in features_df.columns
        # ret_5d should be a finite number
        for val in features_df['ret_5d']:
            assert np.isfinite(val)

    def test_rsi_bounded_0_100(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert features_df['rsi_14'].between(0, 100).all()

    def test_z_score_computation(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert 'z_score_20d' in features_df.columns
        assert 'z_score_50d' in features_df.columns
        for val in features_df['z_score_20d']:
            assert np.isfinite(val)

    def test_volume_ratio_positive(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert (features_df['volume_ratio'] >= 0).all()

    def test_sharpe_calculation(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert 'sharpe_63d' in features_df.columns
        for val in features_df['sharpe_63d']:
            assert np.isfinite(val)

    def test_bb_position_bounded_0_1(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert features_df['bb_position'].between(0, 1).all()

    def test_features_no_nan(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert not features_df.isna().any().any()

    def test_feature_count_is_20(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert features_df.shape[1] == 20

    def test_handles_missing_volume_forex(self):
        """Forex often has no volume; features should still compute."""
        ranker = MLAssetRanker(universe={'forex': ['FX0']})
        df = _make_price_df(n_days=150, include_volume=False, seed=99)
        prices = {'FX0': df}
        features_df = ranker.compute_features(prices)
        assert len(features_df) == 1
        assert not features_df.isna().any().any()

    def test_handles_short_history(self):
        """Assets with <126 days should still get features (with fallbacks)."""
        ranker = MLAssetRanker(universe={'stocks': ['SHORT0']})
        df = _make_price_df(n_days=65, seed=10)
        prices = {'SHORT0': df}
        features_df = ranker.compute_features(prices)
        assert len(features_df) == 1
        assert not features_df.isna().any().any()

    def test_returns_are_reasonable(self):
        """Returns should be in a reasonable range for synthetic data."""
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert features_df['ret_5d'].abs().max() < 10.0  # < 1000%

    def test_max_drawdown_non_positive(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert (features_df['max_drawdown_63d'] <= 0).all()

    def test_adx_in_features(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert 'adx_14' in features_df.columns

    def test_momentum_quality_bounded(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert features_df['momentum_quality'].between(0, 1).all()

    def test_dist_from_52w_high_non_positive(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        assert (features_df['dist_from_52w_high'] <= 0.001).all()


# ---------------------------------------------------------------------------
# 2. Model Scoring Tests
# ---------------------------------------------------------------------------

class TestModelScoring:
    def test_composite_score_returns_series(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        result = ranker.composite_score(features_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(features_df)

    def test_composite_score_bounded(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        features_df = ranker.compute_features(prices)
        result = ranker.composite_score(features_df)
        assert result.between(0, 100).all()

    def test_rf_score_returns_probabilities(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5, n_days=150)
        features_df = ranker.compute_features(prices)
        hist_df = ranker.compute_historical_features(prices)
        result = ranker.rf_score(features_df, hist_df)
        assert isinstance(result, pd.Series)
        assert result.between(0, 100).all()

    def test_rf_handles_small_dataset(self):
        """RF should gracefully handle very small training sets."""
        ranker = MLAssetRanker(universe={'stocks': ['T0']})
        prices = {'T0': _make_price_df(n_days=65, seed=7)}
        features_df = ranker.compute_features(prices)
        # Empty historical features — should return default scores
        empty_hist = pd.DataFrame()
        result = ranker.rf_score(features_df, empty_hist)
        assert isinstance(result, pd.Series)
        assert len(result) == len(features_df)

    def test_cluster_score_returns_series(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5, n_days=150)
        features_df = ranker.compute_features(prices)
        hist_df = ranker.compute_historical_features(prices)
        result = ranker.cluster_score(features_df, hist_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(features_df)

    def test_ensemble_weighted_sum(self):
        """Ensemble should be 0.4*composite + 0.3*rf + 0.3*cluster."""
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5, n_days=150)
        scores_df = ranker.rank_assets(prices)
        # Verify ensemble ≈ weighted sum (within rounding)
        for _, row in scores_df.iterrows():
            expected = 0.4 * row['composite_score'] + 0.3 * row['rf_score'] + 0.3 * row['cluster_score']
            assert abs(row['ensemble_score'] - expected) < 0.01

    def test_ensemble_rank_order(self):
        """Top-ranked asset should have highest ensemble score."""
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        scores_df = ranker.rank_assets(prices)
        if len(scores_df) > 1:
            assert scores_df.iloc[0]['ensemble_score'] >= scores_df.iloc[1]['ensemble_score']

    def test_composite_score_empty_df(self):
        ranker = MLAssetRanker(universe=_small_universe())
        result = ranker.composite_score(pd.DataFrame())
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_cluster_score_bounded(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5, n_days=150)
        features_df = ranker.compute_features(prices)
        hist_df = ranker.compute_historical_features(prices)
        result = ranker.cluster_score(features_df, hist_df)
        assert result.between(0, 100).all()


# ---------------------------------------------------------------------------
# 3. Ranking Tests
# ---------------------------------------------------------------------------

class TestRanking:
    def test_rank_returns_dataframe(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        result = ranker.rank_assets(prices)
        assert isinstance(result, pd.DataFrame)

    def test_rank_has_required_columns(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        result = ranker.rank_assets(prices)
        required = ['ticker', 'category', 'ensemble_score', 'composite_score',
                     'rf_score', 'cluster_score', 'ret_5d', 'ret_21d', 'ret_63d',
                     'sharpe_63d', 'rsi_14', 'volume_ratio', 'rank', 'current_price']
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_top_10_returns_10_or_fewer(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        ranker.rank_assets(prices)
        top = ranker.get_top_n(10)
        assert len(top) <= 10

    def test_ranks_are_unique(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        result = ranker.rank_assets(prices)
        assert result['rank'].is_unique

    def test_category_breakdown_sums_to_n(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        ranker.rank_assets(prices)
        top = ranker.get_top_n(5)
        breakdown = ranker.category_breakdown()
        assert sum(breakdown.values()) == len(top)

    def test_explain_pick_returns_dict(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        ranker.rank_assets(prices)
        top = ranker.get_top_n(3)
        ticker = top.iloc[0]['ticker']
        explanation = ranker.explain_pick(ticker)
        assert isinstance(explanation, dict)

    def test_explain_pick_has_drivers(self):
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        ranker.rank_assets(prices)
        top = ranker.get_top_n(3)
        ticker = top.iloc[0]['ticker']
        explanation = ranker.explain_pick(ticker)
        assert 'primary_drivers' in explanation
        assert 'conviction' in explanation
        assert 'momentum_summary' in explanation
        assert 'risk_summary' in explanation
        assert 'technical_summary' in explanation
        assert len(explanation['primary_drivers']) == 3

    def test_handles_empty_universe(self):
        ranker = MLAssetRanker(universe={'stocks': []})
        result = ranker.rank_assets({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 4. Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_asset_universe(self):
        ranker = MLAssetRanker(universe={'stocks': ['SOLO']})
        prices = {'SOLO': _make_price_df(n_days=150, seed=1)}
        result = ranker.rank_assets(prices)
        assert len(result) == 1
        assert result.iloc[0]['ticker'] == 'SOLO'

    def test_all_same_returns(self):
        """All assets with identical price paths should get similar scores."""
        rng = np.random.default_rng(42)
        base_df = _make_price_df(n_days=150, seed=42)
        prices = {}
        tickers = ['A', 'B', 'C']
        for t in tickers:
            prices[t] = base_df.copy()
        ranker = MLAssetRanker(universe={'stocks': tickers})
        result = ranker.rank_assets(prices)
        scores = result['ensemble_score'].values
        # All scores should be very close
        assert (scores.max() - scores.min()) < 5.0

    def test_negative_returns_universe(self):
        """Universe where all assets have negative drift."""
        prices = {}
        tickers = ['NEG0', 'NEG1', 'NEG2']
        for i, t in enumerate(tickers):
            prices[t] = _make_price_df(n_days=150, drift=-0.005, seed=100 + i)
        ranker = MLAssetRanker(universe={'stocks': tickers})
        result = ranker.rank_assets(prices)
        assert len(result) == 3
        # All should have valid scores
        assert not result['ensemble_score'].isna().any()

    def test_high_nan_asset_dropped(self):
        """An asset that is too short should be dropped from features."""
        prices = {
            'GOOD': _make_price_df(n_days=150, seed=1),
            'SHORT': _make_price_df(n_days=5, seed=2),  # Too short
        }
        ranker = MLAssetRanker(universe={'stocks': ['GOOD', 'SHORT']})
        features_df = ranker.compute_features(prices)
        # SHORT should be dropped (< 10 days)
        assert 'SHORT' not in features_df.index
        assert 'GOOD' in features_df.index

    def test_forex_no_volume_handled(self):
        """Forex without volume should compute all features."""
        df = _make_price_df(n_days=150, include_volume=False, seed=50)
        prices = {'FX': df}
        ranker = MLAssetRanker(universe={'forex': ['FX']})
        result = ranker.rank_assets(prices)
        assert len(result) == 1
        assert result.iloc[0]['volume_ratio'] == 1.0

    def test_get_top_n_on_unranked(self):
        """get_top_n before rank_assets returns empty DataFrame."""
        ranker = MLAssetRanker(universe=_small_universe())
        result = ranker.get_top_n(10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_explain_missing_ticker(self):
        """explain_pick for a non-existent ticker returns empty dict."""
        ranker = MLAssetRanker(universe=_small_universe())
        prices = _make_universe_prices(5)
        ranker.rank_assets(prices)
        result = ranker.explain_pick('NONEXISTENT')
        assert result == {}

    def test_asset_universe_constant(self):
        """ASSET_UNIVERSE should have 5 categories."""
        assert len(ASSET_UNIVERSE) == 5
        assert 'stocks' in ASSET_UNIVERSE
        assert 'etfs' in ASSET_UNIVERSE
        assert 'crypto' in ASSET_UNIVERSE
        assert 'forex' in ASSET_UNIVERSE
        assert 'commodities' in ASSET_UNIVERSE

    def test_asset_universe_ticker_counts(self):
        """Verify approximate ticker counts per category."""
        assert len(ASSET_UNIVERSE['stocks']) == 30
        assert len(ASSET_UNIVERSE['etfs']) == 18
        assert len(ASSET_UNIVERSE['crypto']) == 10
        assert len(ASSET_UNIVERSE['forex']) == 8
        assert len(ASSET_UNIVERSE['commodities']) == 10
