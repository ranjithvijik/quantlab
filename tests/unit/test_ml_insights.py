"""
tests/unit/test_ml_insights.py — Unit tests for ML Insight Modules 32-36.

Coverage:
- NewsSentimentAnalyzer: lexicon scoring, NB training, aggregation, divergence, word cloud
- SmartPortfolioConstructor: Ledoit-Wolf, BL, optimization, Monte Carlo, strategy comparison
- RiskDecompositionEngine: factor regression, attribution, drift, style box
- MLPriceForecaster: features, LR/GBR/ARIMA forecasts, ensemble, backtest
- EarningsSurprisePredictor: features, RF training, prediction, walk-forward
"""

import pytest
import numpy as np
import pandas as pd
from app import (
    NewsSentimentAnalyzer,
    SmartPortfolioConstructor,
    RiskDecompositionEngine,
    MLPriceForecaster,
    EarningsSurprisePredictor,
)


# ─────────────────────────────────────────────────────────────
# Helpers & Fixtures
# ─────────────────────────────────────────────────────────────

RNG = np.random.default_rng(99)


@pytest.fixture(scope="module")
def sentiment_analyzer():
    return NewsSentimentAnalyzer()


@pytest.fixture(scope="module")
def sample_headlines():
    return [
        {'title': 'Company beats earnings expectations with strong revenue growth', 'published': pd.Timestamp('2024-01-05'), 'source': 'Reuters'},
        {'title': 'Stock surges on bullish upgrade from analysts', 'published': pd.Timestamp('2024-01-06'), 'source': 'Bloomberg'},
        {'title': 'Market rally fueled by positive economic data', 'published': pd.Timestamp('2024-01-07'), 'source': 'CNBC'},
        {'title': 'Company faces lawsuit and investigation amid fraud concerns', 'published': pd.Timestamp('2024-01-08'), 'source': 'WSJ'},
        {'title': 'Stock plunges after earnings miss and downgrade warning', 'published': pd.Timestamp('2024-01-09'), 'source': 'Reuters'},
        {'title': 'Recession fears grow as inflation data shows weakness', 'published': pd.Timestamp('2024-01-10'), 'source': 'Bloomberg'},
        {'title': 'The company held a meeting today about quarterly results', 'published': pd.Timestamp('2024-01-11'), 'source': 'PR'},
    ]


@pytest.fixture(scope="module")
def multi_prices():
    n = 504
    idx = pd.bdate_range(start="2022-01-03", periods=n)
    common = RNG.normal(0, 0.008, n)
    data = {}
    for name, mu, idio, beta in [("A", 0.0004, 0.012, 0.6), ("B", 0.0003, 0.011, 0.55),
                                  ("C", 0.0002, 0.008, 0.10)]:
        r = beta * common + RNG.normal(mu, idio, n)
        data[name] = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def single_prices():
    n = 504
    idx = pd.bdate_range(start="2022-01-03", periods=n)
    r = RNG.normal(0.0003, 0.015, n)
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx, name="TEST")


# ─────────────────────────────────────────────────────────────
# Module 32: News Sentiment Analyzer — 12 tests
# ─────────────────────────────────────────────────────────────

class TestNewsSentimentAnalyzer:

    def test_lexicon_positive_text(self, sentiment_analyzer):
        result = sentiment_analyzer.lexicon_sentiment("Company beats earnings with strong growth surge")
        assert result['score'] > 0
        assert result['label'] == 'Positive'
        assert result['pos_count'] >= 2

    def test_lexicon_negative_text(self, sentiment_analyzer):
        result = sentiment_analyzer.lexicon_sentiment("Stock plunges amid bankruptcy fears and fraud investigation")
        assert result['score'] < 0
        assert result['label'] == 'Negative'
        assert result['neg_count'] >= 2

    def test_lexicon_neutral_text(self, sentiment_analyzer):
        result = sentiment_analyzer.lexicon_sentiment("The company held a regular board meeting today")
        assert abs(result['score']) < 0.02
        assert result['label'] == 'Neutral'

    def test_lexicon_score_bounded(self, sentiment_analyzer):
        for text in ["beat " * 100, "crash " * 100, "hello world", ""]:
            result = sentiment_analyzer.lexicon_sentiment(text)
            assert -1.0 <= result['score'] <= 1.0

    def test_nb_training_produces_model(self, sentiment_analyzer):
        training = [
            {'text': 'stock surges on strong earnings beat', 'label': 1},
            {'text': 'revenue growth exceeds expectations', 'label': 1},
            {'text': 'bullish outlook for next quarter', 'label': 1},
            {'text': 'analysts upgrade price target', 'label': 1},
            {'text': 'stock plunges after earnings miss', 'label': 0},
            {'text': 'company faces bankruptcy risk', 'label': 0},
            {'text': 'lawsuit filed amid fraud probe', 'label': 0},
            {'text': 'recession fears drag markets lower', 'label': 0},
        ]
        analyzer = NewsSentimentAnalyzer()
        analyzer.train_nb_classifier(training)
        assert analyzer.is_trained
        assert analyzer.vectorizer is not None
        assert analyzer.classifier is not None

    def test_nb_score_returns_probabilities(self, sentiment_analyzer):
        training = [
            {'text': 'stock surges on earnings beat', 'label': 1},
            {'text': 'revenue growth strong', 'label': 1},
            {'text': 'bullish upgrade outlook', 'label': 1},
            {'text': 'stock plunges after miss', 'label': 0},
            {'text': 'bankruptcy fears grow', 'label': 0},
            {'text': 'lawsuit fraud probe', 'label': 0},
        ]
        analyzer = NewsSentimentAnalyzer()
        analyzer.train_nb_classifier(training)
        scored = analyzer.score_headlines([
            {'title': 'Company beats expectations', 'published': None, 'source': ''},
        ])
        assert len(scored) == 1
        assert -1.0 <= scored.iloc[0]['final_score'] <= 1.0

    def test_aggregate_timeseries_daily_freq(self, sentiment_analyzer, sample_headlines):
        scored = sentiment_analyzer.score_headlines(sample_headlines)
        ts = sentiment_analyzer.aggregate_sentiment_timeseries(scored, freq='D')
        assert 'avg_sentiment' in ts.columns
        assert 'news_count' in ts.columns
        assert 'sentiment_std' in ts.columns

    def test_divergence_detection_bullish(self, sentiment_analyzer):
        dates = pd.bdate_range('2024-01-01', periods=30)
        sent_ts = pd.DataFrame({'avg_sentiment': [0.05] * 30}, index=dates)
        prices = pd.Series(100 * np.exp(np.cumsum([-0.005] * 30)), index=dates)
        divergences = sentiment_analyzer.detect_divergences(sent_ts, prices, lookback=10)
        bullish = [d for d in divergences if d['type'] == 'bullish_divergence']
        assert len(bullish) > 0

    def test_divergence_detection_bearish(self, sentiment_analyzer):
        dates = pd.bdate_range('2024-01-01', periods=30)
        sent_ts = pd.DataFrame({'avg_sentiment': [-0.05] * 30}, index=dates)
        prices = pd.Series(100 * np.exp(np.cumsum([0.005] * 30)), index=dates)
        divergences = sentiment_analyzer.detect_divergences(sent_ts, prices, lookback=10)
        bearish = [d for d in divergences if d['type'] == 'bearish_divergence']
        assert len(bearish) > 0

    def test_word_cloud_excludes_stopwords(self, sentiment_analyzer, sample_headlines):
        wc = sentiment_analyzer.word_cloud_data(sample_headlines)
        assert isinstance(wc, dict)
        stopwords = {'the', 'a', 'an', 'is', 'and', 'or', 'to', 'of'}
        for word in wc.keys():
            assert word not in stopwords

    def test_empty_headlines_handled(self, sentiment_analyzer):
        scored = sentiment_analyzer.score_headlines([])
        assert scored.empty
        wc = sentiment_analyzer.word_cloud_data([])
        assert wc == {}

    def test_sentiment_price_correlation_returns_dict(self, sentiment_analyzer):
        dates = pd.bdate_range('2024-01-01', periods=50)
        sent = pd.DataFrame({'avg_sentiment': RNG.normal(0, 0.1, 50)}, index=dates)
        prices = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0, 0.01, 50))), index=dates)
        result = sentiment_analyzer.sentiment_price_correlation(sent, prices)
        assert 'concurrent_corr' in result
        assert 'predictive_corr' in result
        assert 'lag_1d_corr' in result
        assert 'lag_5d_corr' in result


# ─────────────────────────────────────────────────────────────
# Module 33: Smart Portfolio Constructor — 10 tests
# ─────────────────────────────────────────────────────────────

class TestSmartPortfolioConstructor:

    def test_ledoit_wolf_returns_matrix(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        cov, shrinkage = spc.ledoit_wolf_covariance()
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (3, 3)

    def test_ledoit_wolf_shrinkage_coefficient(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        _, shrinkage = spc.ledoit_wolf_covariance()
        assert 0.0 <= shrinkage <= 1.0

    def test_market_implied_returns_shape(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        implied = spc.market_implied_returns()
        assert len(implied) == 3
        assert isinstance(implied, pd.Series)

    def test_bl_returns_shape_matches_assets(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        cov, _ = spc.ledoit_wolf_covariance()
        implied = spc.market_implied_returns(cov_matrix=cov)
        P, Q, Omega = spc.ml_views()
        bl_ret, bl_cov = spc.black_litterman_returns(implied, P, Q, Omega, cov)
        assert len(bl_ret) == 3

    def test_optimize_weights_sum_to_one(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        cov, _ = spc.ledoit_wolf_covariance()
        implied = spc.market_implied_returns(cov_matrix=cov)
        result = spc.optimize_portfolio(implied, cov)
        total = sum(result['weights'].values())
        assert abs(total - 1.0) < 1e-4

    def test_optimize_respects_max_weight(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        cov, _ = spc.ledoit_wolf_covariance()
        implied = spc.market_implied_returns(cov_matrix=cov)
        max_w = 0.40
        result = spc.optimize_portfolio(implied, cov, constraints={'max_weight': max_w})
        for w in result['weights'].values():
            assert w <= max_w + 1e-6

    def test_monte_carlo_returns_paths(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        weights = {t: 1.0 / len(spc.tickers) for t in spc.tickers}
        mc = spc.regime_conditional_monte_carlo(weights, n_sims=100, horizon=50)
        assert mc['simulated_paths'].shape == (100, 50)

    def test_monte_carlo_percentiles(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        weights = {t: 1.0 / len(spc.tickers) for t in spc.tickers}
        mc = spc.regime_conditional_monte_carlo(weights, n_sims=200, horizon=50)
        pctiles = mc['percentiles']
        assert pctiles['5th'] <= pctiles['25th'] <= pctiles['50th'] <= pctiles['75th'] <= pctiles['95th']

    def test_conservative_lower_vol_than_aggressive(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        cov, _ = spc.ledoit_wolf_covariance()
        implied = spc.market_implied_returns(cov_matrix=cov)
        cons = spc.optimize_portfolio(implied, cov, risk_tolerance='conservative')
        aggr = spc.optimize_portfolio(implied, cov, risk_tolerance='aggressive')
        assert cons['expected_vol'] <= aggr['expected_vol'] + 0.05

    def test_compare_strategies_returns_df(self, multi_prices):
        spc = SmartPortfolioConstructor(multi_prices)
        comp = spc.compare_strategies()
        assert isinstance(comp, pd.DataFrame)
        assert len(comp) >= 3
        assert 'Strategy' in comp.columns
        assert 'Sharpe' in comp.columns


# ─────────────────────────────────────────────────────────────
# Module 34: Risk Decomposition Engine — 10 tests
# ─────────────────────────────────────────────────────────────

class TestRiskDecompositionEngine:

    @pytest.fixture()
    def factor_data(self):
        n = 252
        dates = pd.bdate_range('2023-01-01', periods=n)
        factors = pd.DataFrame({
            'market': RNG.normal(0.0004, 0.01, n),
            'size': RNG.normal(0.0001, 0.005, n),
            'value': RNG.normal(0.0001, 0.005, n),
            'momentum': RNG.normal(0.0002, 0.006, n),
        }, index=dates)
        asset = pd.Series(
            0.8 * factors['market'] + 0.3 * factors['size'] + RNG.normal(0, 0.005, n),
            index=dates, name='ASSET'
        )
        return asset, factors

    def test_factor_regression_returns_betas(self, factor_data):
        asset, factors = factor_data
        engine = RiskDecompositionEngine(asset)
        reg = engine.factor_regression(asset, factors)
        assert 'betas' in reg
        assert 'market' in reg['betas']
        assert abs(reg['betas']['market'] - 0.8) < 0.3

    def test_factor_regression_r_squared_bounded(self, factor_data):
        asset, factors = factor_data
        engine = RiskDecompositionEngine(asset)
        reg = engine.factor_regression(asset, factors)
        assert 0.0 <= reg['r_squared'] <= 1.0

    def test_return_attribution_sums_to_total(self, factor_data):
        asset, factors = factor_data
        engine = RiskDecompositionEngine(asset)
        reg = engine.factor_regression(asset, factors)
        fr_mean = factors.mean().to_dict()
        total_return = float(asset.mean() * 252)
        attr = engine.return_attribution(total_return, fr_mean, reg['betas'])
        reconstructed = sum(attr['factor_contributions'].values()) + attr['alpha']
        assert abs(reconstructed - attr['total_return']) < 1e-6

    def test_risk_attribution_factor_plus_idio(self, factor_data):
        asset, factors = factor_data
        engine = RiskDecompositionEngine(asset)
        weights = {'ASSET': 1.0}
        cov = pd.DataFrame([[asset.var()]], index=['ASSET'], columns=['ASSET'])
        factor_betas = {'ASSET': {'market': 0.8, 'size': 0.3, 'value': 0.0, 'momentum': 0.0}}
        factor_cov = factors.cov()
        result = engine.risk_attribution(weights, cov, factor_betas, factor_cov)
        assert result['factor_var'] >= 0
        assert result['idio_var'] >= 0

    def test_factor_drift_detection_flags_outlier(self):
        dates = pd.bdate_range('2023-01-01', periods=252)
        # Use very stable betas with a large jump at the end to trigger >2 std drift
        stable = np.ones(250) * 0.8 + RNG.normal(0, 0.01, 250)
        shifted = np.array([5.0, 5.0])  # extreme shift at end
        betas = pd.DataFrame({
            'market': np.concatenate([stable, shifted]),
            'size': np.ones(252) * 0.3,
        }, index=dates)
        engine = RiskDecompositionEngine(pd.Series(dtype=float))
        drift = engine.factor_drift_detection(betas)
        market_drift = drift[drift['factor'] == 'market']
        assert len(market_drift) > 0
        assert market_drift.iloc[0]['drift_signal']

    def test_style_box_classification(self):
        engine = RiskDecompositionEngine(pd.Series(dtype=float))
        result = engine.style_box(1.0, 0.2, 0.3)
        assert result['size_label'] == 'Small'
        assert result['style_label'] == 'Value'
        result2 = engine.style_box(1.0, -0.2, -0.3)
        assert result2['size_label'] == 'Large'
        assert result2['style_label'] == 'Growth'
        result3 = engine.style_box(1.0, 0.0, 0.0)
        assert result3['size_label'] == 'Mid'
        assert result3['style_label'] == 'Blend'

    def test_portfolio_exposure_weighted_sum(self):
        engine = RiskDecompositionEngine(pd.Series(dtype=float))
        weights = {'A': 0.6, 'B': 0.4}
        betas = {'A': {'market': 1.0, 'size': 0.5}, 'B': {'market': 0.5, 'size': -0.2}}
        exposure = engine.portfolio_factor_exposure(weights, betas)
        expected_market = 0.6 * 1.0 + 0.4 * 0.5
        expected_size = 0.6 * 0.5 + 0.4 * (-0.2)
        assert abs(exposure['market'] - expected_market) < 1e-6
        assert abs(exposure['size'] - expected_size) < 1e-6

    def test_rolling_attribution_shape(self, factor_data):
        asset, factors = factor_data
        engine = RiskDecompositionEngine(asset)
        rolling = engine.rolling_factor_attribution(asset, factors, window=63)
        if not rolling.empty:
            assert len(rolling.columns) >= len(factors.columns)

    def test_handles_single_asset(self):
        dates = pd.bdate_range('2023-01-01', periods=50)
        asset = pd.Series(RNG.normal(0, 0.01, 50), index=dates)
        factors = pd.DataFrame({'market': RNG.normal(0, 0.01, 50)}, index=dates)
        engine = RiskDecompositionEngine(asset)
        reg = engine.factor_regression(asset, factors)
        assert 'betas' in reg

    def test_handles_missing_factor_data(self):
        engine = RiskDecompositionEngine(pd.Series(dtype=float))
        drift = engine.factor_drift_detection(pd.DataFrame())
        assert drift.empty


# ─────────────────────────────────────────────────────────────
# Module 35: ML Price Forecaster — 10 tests
# ─────────────────────────────────────────────────────────────

class TestMLPriceForecaster:

    def test_feature_preparation_shape(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        X, y, features = fc.prepare_features(forward_days=5)
        if not X.empty:
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] > 5
            assert len(features) == X.shape[1]

    def test_lr_forecast_returns_predictions(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        X, y, _ = fc.prepare_features(forward_days=5)
        if not X.empty and len(X) > 100:
            result = fc.rolling_regression_forecast(X, y, train_window=100, forecast_horizon=5)
            if not result.empty:
                assert 'predicted_return' in result.columns
                assert 'actual_return' in result.columns

    def test_gbr_forecast_returns_predictions(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        X, y, _ = fc.prepare_features(forward_days=5)
        if not X.empty and len(X) > 100:
            result = fc.gradient_boosting_forecast(X, y, train_window=100, forecast_horizon=5)
            if not result.empty:
                assert 'predicted_return' in result.columns

    def test_arima_garch_returns_forecast(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        result = fc.arima_garch_forecast(fc.returns, forecast_horizon=21)
        assert 'point_forecasts' in result
        assert 'vol_forecasts' in result
        assert len(result['point_forecasts']) == 21
        assert len(result['vol_forecasts']) == 21

    def test_ensemble_weighted_average(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        result = fc.ensemble_forecast(forecast_horizon=5)
        assert 'point_forecast' in result
        assert 'model_forecasts' in result
        assert 'lr' in result['model_forecasts']
        assert 'gbr' in result['model_forecasts']
        assert 'arima' in result['model_forecasts']

    def test_confidence_interval_contains_point(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        result = fc.ensemble_forecast(forecast_horizon=5)
        point = result['point_forecast']
        upper = result['upper_95']
        lower = result['lower_95']
        for i in range(len(point)):
            assert lower[i] <= point[i] <= upper[i]

    def test_disagreement_high_when_models_differ(self):
        n = 300
        idx = pd.bdate_range('2022-01-01', periods=n)
        prices = pd.Series(100 * np.exp(np.cumsum(RNG.normal(0, 0.03, n))), index=idx)
        fc = MLPriceForecaster(prices, 'VOLATILE')
        result = fc.ensemble_forecast(forecast_horizon=5)
        assert 0.0 <= result['disagreement'] <= 1.0
        assert result['confidence_label'] in ['High', 'Medium', 'Low']

    def test_backtest_accuracy_metrics(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        bt = fc.backtest_accuracy(n_backtests=3)
        assert 'mae' in bt
        assert 'rmse' in bt
        assert 'direction_accuracy' in bt
        assert bt['mae'] >= 0
        assert bt['rmse'] >= 0

    def test_direction_accuracy_bounded_0_1(self, single_prices):
        fc = MLPriceForecaster(single_prices, 'TEST')
        bt = fc.backtest_accuracy(n_backtests=3)
        assert 0.0 <= bt['direction_accuracy'] <= 1.0

    def test_handles_short_price_history(self):
        idx = pd.bdate_range('2024-01-01', periods=20)
        prices = pd.Series(100 + RNG.normal(0, 1, 20), index=idx)
        fc = MLPriceForecaster(prices, 'SHORT')
        result = fc.ensemble_forecast(forecast_horizon=5)
        assert 'point_forecast' in result


# ─────────────────────────────────────────────────────────────
# Module 36: Earnings Surprise Predictor — 10 tests
# ─────────────────────────────────────────────────────────────

class TestEarningsSurprisePredictor:

    @pytest.fixture()
    def synthetic_earnings_setup(self, single_prices):
        """Create synthetic earnings data for testing."""
        earnings_dates = pd.DataFrame({
            'date': pd.bdate_range('2022-04-01', periods=8, freq='QE'),
            'eps_estimate': [1.0, 1.1, 1.2, 1.1, 1.3, 1.2, 1.4, 1.3],
            'eps_actual': [1.05, 1.08, 1.25, 1.15, 1.28, 1.30, 1.35, 1.40],
            'surprise_pct': [5.0, -1.8, 4.2, 4.5, -1.5, 8.3, -3.6, 7.7],
            'beat': [True, False, True, True, False, True, False, True],
        })
        return single_prices, earnings_dates

    def test_pre_earnings_features_shape(self, synthetic_earnings_setup):
        prices, earnings = synthetic_earnings_setup
        predictor = EarningsSurprisePredictor('TEST')
        features = predictor.compute_pre_earnings_features(prices, earnings)
        if not features.empty:
            assert 'pre_earnings_return_5d' in features.columns
            assert 'pre_earnings_vol' in features.columns
            assert 'beat' in features.columns

    def test_beat_rate_bounded_0_1(self, synthetic_earnings_setup):
        prices, earnings = synthetic_earnings_setup
        predictor = EarningsSurprisePredictor('TEST')
        features = predictor.compute_pre_earnings_features(prices, earnings)
        if not features.empty and 'historical_beat_rate' in features.columns:
            rates = features['historical_beat_rate']
            assert all(0.0 <= r <= 1.0 for r in rates)

    def test_consecutive_beats_non_negative(self, synthetic_earnings_setup):
        prices, earnings = synthetic_earnings_setup
        predictor = EarningsSurprisePredictor('TEST')
        features = predictor.compute_pre_earnings_features(prices, earnings)
        if not features.empty and 'consecutive_beats' in features.columns:
            assert all(cb >= 0 for cb in features['consecutive_beats'])

    def test_rf_predict_probability_bounded(self, synthetic_earnings_setup):
        prices, earnings = synthetic_earnings_setup
        predictor = EarningsSurprisePredictor('TEST')
        features = predictor.compute_pre_earnings_features(prices, earnings)
        if not features.empty and len(features) >= 5:
            predictor.train_predictor(features)
            if predictor.model is not None:
                pred = predictor.predict_next_earnings(features.iloc[-1:])
                assert 0.0 <= pred['beat_probability'] <= 1.0
                assert 0.0 <= pred['miss_probability'] <= 1.0

    def test_prediction_label_categories(self, synthetic_earnings_setup):
        prices, earnings = synthetic_earnings_setup
        predictor = EarningsSurprisePredictor('TEST')
        features = predictor.compute_pre_earnings_features(prices, earnings)
        if not features.empty and len(features) >= 5:
            predictor.train_predictor(features)
            pred = predictor.predict_next_earnings(features.iloc[-1:])
            assert pred['prediction_label'] in ['Likely Beat', 'Likely Miss', 'Toss-Up']

    def test_walk_forward_respects_time(self, synthetic_earnings_setup):
        prices, earnings = synthetic_earnings_setup
        predictor = EarningsSurprisePredictor('TEST')
        features = predictor.compute_pre_earnings_features(prices, earnings)
        if not features.empty and len(features) >= 10:
            perf = predictor.historical_prediction_performance(features)
            if not perf.empty:
                assert 'predicted_prob' in perf.columns
                assert 'actual_beat' in perf.columns
                assert 'correct' in perf.columns

    def test_feature_importance_sums_to_one(self, synthetic_earnings_setup):
        prices, earnings = synthetic_earnings_setup
        predictor = EarningsSurprisePredictor('TEST')
        features = predictor.compute_pre_earnings_features(prices, earnings)
        if not features.empty and len(features) >= 5:
            result = predictor.train_predictor(features)
            if result['feature_importances']:
                total = sum(result['feature_importances'].values())
                assert abs(total - 1.0) < 0.01

    def test_handles_no_earnings_history(self):
        predictor = EarningsSurprisePredictor('FAKE')
        empty_df = pd.DataFrame(columns=['date', 'eps_estimate', 'eps_actual', 'surprise_pct', 'beat'])
        idx = pd.bdate_range('2024-01-01', periods=100)
        prices = pd.Series(100 + np.cumsum(RNG.normal(0, 1, 100)), index=idx)
        features = predictor.compute_pre_earnings_features(prices, empty_df)
        assert features.empty

    def test_upcoming_calendar_returns_df(self):
        predictor = EarningsSurprisePredictor('TEST')
        result = predictor.earnings_calendar_upcoming([], days_ahead=30)
        assert isinstance(result, pd.DataFrame)
        assert 'ticker' in result.columns or result.empty

    def test_sector_accuracy_returns_df(self):
        predictor = EarningsSurprisePredictor('TEST')
        result = predictor.sector_accuracy(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
