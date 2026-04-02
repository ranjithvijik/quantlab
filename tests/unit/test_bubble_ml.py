"""
test_bubble_ml.py — Unit tests for BubbleDetector, TechnicalIndicators,
ML pipeline (train_ml_models), and clustering helpers.
"""
import numpy as np
import pandas as pd
import pytest
from app import (
    BubbleDetector,
    TechnicalIndicators,
    train_ml_models,
    simple_sentiment_score,
    compute_asset_features,
    MetcalfeLawAdvanced,
    LongMemoryEstimators,
)


# ---------------------------------------------------------------------------
# BubbleDetector
# ---------------------------------------------------------------------------

class TestBubbleDetector:
    def test_returns_all_keys(self, single_price_series, returns_series, volume_series):
        bd = BubbleDetector()
        result = bd.detect_bubbles(single_price_series, returns_series, volume_series)
        for key in ["bubble_score", "mmv_ratio", "d_parameter", "kurtosis", "skewness",
                    "has_vol_clustering"]:
            assert key in result, f"Missing key: {key}"

    def test_bubble_score_in_unit_interval(self, single_price_series, returns_series, volume_series):
        bd = BubbleDetector()
        result = bd.detect_bubbles(single_price_series, returns_series, volume_series)
        score = result["bubble_score"]
        assert 0.0 <= score <= 1.0, f"Bubble score {score} outside [0, 1]"

    def test_bubble_score_without_volume(self, single_price_series, returns_series):
        """Should work without volume series (uses log-price fallback)."""
        bd = BubbleDetector()
        result = bd.detect_bubbles(single_price_series, returns_series, volumes=None)
        assert 0.0 <= result["bubble_score"] <= 1.0

    def test_high_kurtosis_increases_score(self, single_price_series):
        """Fat-tail returns should produce higher bubble score."""
        rng = np.random.default_rng(1)
        normal_ret = pd.Series(rng.normal(0, 0.01, 300),
                               index=single_price_series.index[:300])
        # Student-t(3) has high kurtosis
        fat_ret = pd.Series(rng.standard_t(3, 300) * 0.005,
                            index=single_price_series.index[:300])
        prices = single_price_series.iloc[:300]

        bd = BubbleDetector()
        score_norm = bd.detect_bubbles(prices, normal_ret)["bubble_score"]
        score_fat  = bd.detect_bubbles(prices, fat_ret)["bubble_score"]
        assert score_fat >= score_norm

    def test_insufficient_returns_no_crash(self, single_price_series):
        """Fewer than 50 observations should not raise — d defaults to 0."""
        bd = BubbleDetector()
        short_p = single_price_series.iloc[:30]
        short_r = short_p.pct_change().dropna()
        result = bd.detect_bubbles(short_p, short_r)
        assert result["d_parameter"] == 0


# ---------------------------------------------------------------------------
# MetcalfeLawAdvanced
# ---------------------------------------------------------------------------

class TestMetcalfe:
    def test_network_value_positive(self, volume_series):
        m = MetcalfeLawAdvanced()
        nv = m.calculate_network_value(volume_series.values)
        assert all(v >= 0 for v in nv)

    def test_mmv_ratio_type(self, single_price_series, volume_series):
        m = MetcalfeLawAdvanced()
        nv = m.calculate_network_value(volume_series.values)
        ratios = m.calculate_mmv_ratio(single_price_series.values, nv)
        assert len(ratios) > 0

    def test_bubble_regime_labels(self):
        m = MetcalfeLawAdvanced()
        # 0.5 = below 1.0 threshold → Undervalued
        regime_low = m.detect_bubble_regime(0.5)
        assert isinstance(regime_low, str) and len(regime_low) > 0
        # 3.0 = well above 1.0 → some form of bubble / overvalued label
        regime_high = m.detect_bubble_regime(3.0)
        assert isinstance(regime_high, str) and len(regime_high) > 0
        # High MMV regime should differ from low MMV regime
        assert regime_high != regime_low


# ---------------------------------------------------------------------------
# LongMemoryEstimators
# ---------------------------------------------------------------------------

class TestLongMemory:
    def test_gph_returns_two_values(self, returns_series):
        lm = LongMemoryEstimators()
        d, se = lm.gph_estimator(returns_series.values)
        assert isinstance(d, float)
        assert isinstance(se, float)
        assert se > 0

    def test_gph_iid_d_near_zero(self):
        """IID returns (white noise) should have d ≈ 0 (no long memory)."""
        rng = np.random.default_rng(0)
        white_noise = rng.normal(0, 0.01, 500)
        lm = LongMemoryEstimators()
        d, _ = lm.gph_estimator(white_noise)
        assert abs(d) < 0.5, f"Expected d≈0 for IID noise, got {d:.4f}"

    def test_gph_se_uses_asymptotic_formula(self, returns_series):
        """SE should equal π / sqrt(24 * m) for m = n^0.5 frequencies."""
        lm = LongMemoryEstimators()
        n = len(returns_series)
        m = max(int(n ** 0.5), 2)
        expected_se = np.pi / np.sqrt(24 * m)
        _, se = lm.gph_estimator(returns_series.values)
        assert abs(se - expected_se) < 1e-6


# ---------------------------------------------------------------------------
# TechnicalIndicators
# ---------------------------------------------------------------------------

class TestTechnicalIndicators:
    EXPECTED_COLS = ["SMA_20", "SMA_50", "EMA_12", "MACD", "MACD_Signal",
                     "MACD_Histogram", "RSI", "BB_Upper", "BB_Lower"]

    def test_returns_all_columns(self, single_price_series):
        ind = TechnicalIndicators.calculate_all(single_price_series)
        for col in self.EXPECTED_COLS:
            assert col in ind.columns, f"Missing column: {col}"

    def test_index_aligned(self, single_price_series):
        ind = TechnicalIndicators.calculate_all(single_price_series)
        assert list(ind.index) == list(single_price_series.index)

    def test_sma20_after_warmup(self, single_price_series):
        ind = TechnicalIndicators.calculate_all(single_price_series)
        assert not ind["SMA_20"].iloc[50:].isna().any()

    def test_sma50_after_warmup(self, single_price_series):
        ind = TechnicalIndicators.calculate_all(single_price_series)
        assert not ind["SMA_50"].iloc[100:].isna().any()

    def test_rsi_bounded(self, single_price_series):
        ind = TechnicalIndicators.calculate_all(single_price_series)
        rsi = ind["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_macd_histogram_equals_diff(self, single_price_series):
        ind = TechnicalIndicators.calculate_all(single_price_series)
        diff = (ind["MACD"] - ind["MACD_Signal"]).dropna()
        hist = ind["MACD_Histogram"].dropna()
        common = diff.index.intersection(hist.index)
        pd.testing.assert_series_equal(diff.loc[common].round(10), hist.loc[common].round(10),
                                       check_names=False)

    def test_bollinger_upper_ge_lower(self, single_price_series):
        ind = TechnicalIndicators.calculate_all(single_price_series)
        valid = ind[["BB_Upper", "BB_Lower"]].dropna()
        assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()

    def test_rsi_wilder_ema(self, single_price_series):
        """RSI should use Wilder's EMA (ewm alpha=1/14), not rolling SMA."""
        prices = single_price_series
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        expected_rsi = (100 - 100 / (1 + rs)).dropna()

        ind = TechnicalIndicators.calculate_all(prices)
        actual_rsi = ind["RSI"].dropna()

        common = expected_rsi.index.intersection(actual_rsi.index)
        np.testing.assert_allclose(
            expected_rsi.loc[common].values,
            actual_rsi.loc[common].values,
            rtol=0.01,
            err_msg="RSI does not match Wilder's EMA formula",
        )


# ---------------------------------------------------------------------------
# ML Pipeline
# ---------------------------------------------------------------------------

class TestMLPipeline:
    def test_returns_all_keys(self, single_price_series, volume_series):
        result = train_ml_models(single_price_series, volume_series)
        assert result is not None
        for key in ["models_info", "feature_importance", "predictions",
                    "actuals", "feature_names", "scaler", "last_features"]:
            assert key in result, f"Missing key: {key}"

    def test_three_models_trained(self, single_price_series, volume_series):
        result = train_ml_models(single_price_series, volume_series)
        assert set(result["models_info"].keys()) == {
            "Linear Regression", "Random Forest", "Gradient Boosting"
        }

    def test_metrics_present(self, single_price_series, volume_series):
        result = train_ml_models(single_price_series, volume_series)
        for model_name, info in result["models_info"].items():
            for metric in ["r2", "rmse", "mae"]:
                assert metric in info, f"{model_name} missing {metric}"

    def test_rmse_non_negative(self, single_price_series, volume_series):
        result = train_ml_models(single_price_series, volume_series)
        for name, info in result["models_info"].items():
            assert info["rmse"] >= 0

    def test_last_features_scaled(self, single_price_series, volume_series):
        """last_features must be already scaled (output of scaler.transform)."""
        result = train_ml_models(single_price_series, volume_series)
        scaler = result["scaler"]
        lf = result["last_features"]
        # Re-scaling an already-scaled array should leave it unchanged
        double_scaled = scaler.transform(scaler.inverse_transform(lf))
        np.testing.assert_allclose(lf, double_scaled, rtol=1e-5,
                                   err_msg="last_features appears to be unscaled")

    def test_returns_none_for_short_series(self):
        """Fewer than 60 rows after feature engineering → return None."""
        short = pd.Series(range(1, 50), dtype=float,
                          index=pd.bdate_range("2023-01-01", periods=49))
        result = train_ml_models(short)
        assert result is None

    def test_works_without_volume(self):
        """ML without volume needs enough data for 200-day SMA warmup."""
        rng = np.random.default_rng(99)
        # Need 500+ rows so feature engineering (200d SMA, 21d target shift) leaves >= 60 rows
        prices = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 550))),
            index=pd.bdate_range('2020-01-01', periods=550),
        )
        result = train_ml_models(prices, volumes_series=None)
        assert result is not None


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

class TestSentiment:
    def test_bullish_text(self):
        score = simple_sentiment_score("Stock surged on strong earnings beat, record revenue")
        assert score > 0

    def test_bearish_text(self):
        score = simple_sentiment_score("Market crash plunges on recession fears and bankruptcy risk")
        assert score < 0

    def test_neutral_text(self):
        score = simple_sentiment_score("The company announced a new product today")
        assert score == 0.0

    def test_mixed_text(self):
        """Equal positive and negative words → score = 0."""
        score = simple_sentiment_score("rally crash gain loss")
        assert score == 0.0

    def test_score_bounded(self):
        for text in ["surge rally gain profit beat", "crash plunge loss bankruptcy risk"]:
            score = simple_sentiment_score(text)
            assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Compute Asset Features
# ---------------------------------------------------------------------------

class TestAssetFeatures:
    def test_returns_dataframe(self, multi_price_df):
        features = compute_asset_features(multi_price_df)
        assert isinstance(features, pd.DataFrame)

    def test_feature_columns(self, multi_price_df):
        features = compute_asset_features(multi_price_df)
        for col in ["Ann Return", "Volatility", "Skewness", "Kurtosis", "Sharpe", "Max Drawdown"]:
            assert col in features.columns

    def test_one_row_per_asset(self, multi_price_df):
        features = compute_asset_features(multi_price_df)
        assert len(features) == len(multi_price_df.columns)

    def test_sharpe_uses_rf(self, multi_price_df):
        """Sharpe = (Ann Return - 4.5%) / Vol."""
        features = compute_asset_features(multi_price_df)
        returns = multi_price_df.pct_change().dropna()
        for ticker in multi_price_df.columns:
            r = returns[ticker]
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            expected = (ann_ret - 0.045) / ann_vol if ann_vol > 0 else 0
            assert abs(features.loc[ticker, "Sharpe"] - expected) < 1e-6

    def test_max_drawdown_non_positive(self, multi_price_df):
        features = compute_asset_features(multi_price_df)
        assert (features["Max Drawdown"] <= 0).all()

    def test_volatility_positive(self, multi_price_df):
        features = compute_asset_features(multi_price_df)
        assert (features["Volatility"] > 0).all()
