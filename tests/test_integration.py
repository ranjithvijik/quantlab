"""
test_integration.py — End-to-end pipeline integration tests.

All tests run fully offline using synthetic fixtures — no network calls.
Covers the analysis pipeline from raw prices through valuation,
optimization, technical analysis, ML, and bubble detection.
"""
import numpy as np
import pandas as pd
import pytest
from app import (
    EnhancedValuationMetrics as EVM,
    EnhancedPortfolioOptimizer,
    BubbleDetector,
    TechnicalIndicators,
    train_ml_models,
    compute_asset_features,
    calculate_composite_risk_score,
    simple_sentiment_score,
    black_scholes_price,
    bs_greeks,
    options_payoff,
)


# ---------------------------------------------------------------------------
# Pipeline: prices → returns → metrics
# ---------------------------------------------------------------------------

class TestReturnsPipeline:
    def test_returns_computed_correctly(self, multi_price_df):
        returns = multi_price_df.pct_change().bfill().dropna(how='all')
        assert not returns.empty
        assert returns.shape[1] == multi_price_df.shape[1]

    def test_metrics_all_tickers(self, multi_price_df):
        """Performance metrics should be computable for every ticker."""
        returns = multi_price_df.pct_change().dropna()
        rf = 0.045
        metrics = {}
        for ticker in multi_price_df.columns:
            r = returns[ticker].dropna()
            ann_vol = r.std() * np.sqrt(252)
            metrics[ticker] = {
                'Annual Return': r.mean() * 252,
                'Volatility': ann_vol,
                'Sharpe': (r.mean() * 252 - rf) / ann_vol if ann_vol > 0 else 0,
                'Max Drawdown': ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),
            }
        df = pd.DataFrame(metrics).T
        assert df.shape == (len(multi_price_df.columns), 4)
        assert not df.isna().any().any()

    def test_sharpe_formula(self, single_price_series):
        r = single_price_series.pct_change().dropna()
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.045) / ann_vol
        assert isinstance(sharpe, float)
        assert -5 < sharpe < 5  # realistic range


# ---------------------------------------------------------------------------
# Pipeline: prices → portfolio → metrics
# ---------------------------------------------------------------------------

class TestPortfolioPipeline:
    def test_full_optimization_roundtrip(self, multi_price_df):
        """Run all 9 strategies and verify invariants for each."""
        opt = EnhancedPortfolioOptimizer(multi_price_df, rf_rate=0.045)
        strategies = {
            'Max Sharpe':    opt.maximum_sharpe(),
            'Min Variance':  opt.minimum_variance(),
            'Risk Parity':   opt.risk_parity(),
            'Min CVaR':      opt.minimum_cvar(),
            'Max Divers':    opt.maximum_diversification(),
            'Kelly':         opt.kelly_criterion(),
            'Black-Litterman': opt.black_litterman(),
            'HRP':           opt.hierarchical_risk_parity(),
            'Equal Weight':  opt.equal_weight(),
        }
        for name, weights in strategies.items():
            assert abs(sum(weights) - 1.0) < 1e-4, f"{name}: weights sum ≠ 1"
            assert all(w >= -1e-8 for w in weights), f"{name}: negative weight"

    def test_portfolio_metrics_from_optimized_weights(self, multi_price_df):
        opt = EnhancedPortfolioOptimizer(multi_price_df, rf_rate=0.045)
        weights = opt.maximum_sharpe()
        m = opt.calculate_portfolio_metrics(weights)
        assert m["Volatility"] > 0
        assert m["Max Drawdown"] <= 0
        assert "Sharpe Ratio" in m

    def test_bubble_aware_reduces_overvalued_weight(self, multi_price_df):
        tickers = list(multi_price_df.columns)
        scores = {tickers[0]: 1.0, tickers[1]: 0.0, tickers[2]: 0.0, tickers[3]: 0.0}
        opt_plain  = EnhancedPortfolioOptimizer(multi_price_df, rf_rate=0.045)
        opt_bubble = EnhancedPortfolioOptimizer(multi_price_df, bubble_scores=scores, rf_rate=0.045)
        w_plain  = opt_plain.maximum_sharpe(bubble_aware=False)
        w_bubble = opt_bubble.maximum_sharpe(bubble_aware=True, penalty_factor=2.0)
        assert w_bubble[0] <= w_plain[0] + 0.05


# ---------------------------------------------------------------------------
# Pipeline: prices → bubble detection
# ---------------------------------------------------------------------------

class TestBubblePipeline:
    def test_pipeline_for_all_tickers(self, multi_price_df, multi_returns_df):
        bd = BubbleDetector()
        scores = {}
        for ticker in multi_price_df.columns:
            result = bd.detect_bubbles(
                multi_price_df[ticker],
                multi_returns_df[ticker],
            )
            scores[ticker] = result["bubble_score"]
        assert len(scores) == multi_price_df.shape[1]
        assert all(0 <= s <= 1 for s in scores.values())


# ---------------------------------------------------------------------------
# Pipeline: prices → technical analysis → feature engineering
# ---------------------------------------------------------------------------

class TestTechAndFeaturesPipeline:
    def test_indicators_for_all_tickers(self, multi_price_df):
        for ticker in multi_price_df.columns:
            ind = TechnicalIndicators.calculate_all(multi_price_df[ticker])
            assert "RSI" in ind.columns
            assert "MACD_Histogram" in ind.columns
            assert not ind.empty

    def test_compute_asset_features_pipeline(self, multi_price_df):
        features = compute_asset_features(multi_price_df)
        assert features.shape[0] == multi_price_df.shape[1]
        assert not features.isna().all().all()


# ---------------------------------------------------------------------------
# Pipeline: prices → ML → predictions
# ---------------------------------------------------------------------------

class TestMLPipeline:
    def test_ml_all_models_predict(self, single_price_series, volume_series):
        result = train_ml_models(single_price_series, volume_series)
        assert result is not None
        for model_name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
            assert model_name in result["models_info"]
            preds = result["predictions"][model_name]
            assert len(preds) > 0

    def test_last_features_can_predict(self, single_price_series, volume_series):
        """last_features should be usable for forward prediction with all models."""
        result = train_ml_models(single_price_series, volume_series)
        lf = result["last_features"]
        for name, info in result["models_info"].items():
            pred = info["model"].predict(lf)
            assert len(pred) == 1
            assert np.isfinite(pred[0]), f"{name} predicted NaN/inf"

    def test_predictions_actuals_length_match(self, single_price_series, volume_series):
        result = train_ml_models(single_price_series, volume_series)
        actuals = result["actuals"]
        for name, preds in result["predictions"].items():
            assert len(preds) == len(actuals), f"{name}: length mismatch"


# ---------------------------------------------------------------------------
# Pipeline: Options pricing → Greeks → payoff
# ---------------------------------------------------------------------------

class TestOptionsPipeline:
    def test_bs_to_greeks_to_payoff(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.25

        # Price
        call_price = black_scholes_price(S, K, T, r, sigma, 'call')
        assert call_price > 0

        # Greeks
        greeks = bs_greeks(S, K, T, r, sigma, 'call')
        assert 0 < greeks["Delta"] < 1
        assert greeks["Gamma"] > 0

        # Payoff at various S values
        S_range = np.linspace(60, 140, 100)
        pf = options_payoff('Long Call', S_range, S, K1=K, premium1=call_price)
        assert max(pf) > 0  # there is some range where call is profitable

    def test_put_call_parity_pipeline(self):
        S, K, T, r, sigma = 100, 95, 1.0, 0.04, 0.18
        call = black_scholes_price(S, K, T, r, sigma, 'call')
        put  = black_scholes_price(S, K, T, r, sigma, 'put')
        import math
        parity_rhs = S - K * math.exp(-r * T)
        assert abs((call - put) - parity_rhs) < 1e-6


# ---------------------------------------------------------------------------
# Pipeline: Sentiment → Bubble → Risk (combined narrative)
# ---------------------------------------------------------------------------

class TestNarrativePipeline:
    def test_bearish_sentiment_with_high_risk(self):
        """Simulate a high-stress scenario: bearish news + high VIX + inverted curve."""
        news_texts = [
            "market crash recession fears bankruptcy layoff",
            "plunge loss decline weak bearish",
        ]
        sentiment_scores = [simple_sentiment_score(t) for t in news_texts]
        assert all(s < 0 for s in sentiment_scores), "All texts should be bearish"

        # High risk environment
        vix = pd.Series([45.0] * 300)
        tnx = pd.Series([3.0] * 10)
        irx = pd.Series([5.0] * 10)
        gold_up = pd.Series([100.0] * 299 + [120.0])
        risk_data = {"VIX": vix, "TNX": tnx, "IRX": irx, "Gold": gold_up}
        risk_score = calculate_composite_risk_score(risk_data)
        assert risk_score > 50, f"Expected high risk score, got {risk_score}"

    def test_bullish_sentiment_low_risk_scenario(self):
        texts = ["surge rally gain record earnings beat", "growth profit strong bullish"]
        scores = [simple_sentiment_score(t) for t in texts]
        assert all(s > 0 for s in scores)

        vix = pd.Series([13.0] * 300)
        tnx = pd.Series([4.5] * 10)
        irx = pd.Series([3.5] * 10)
        gold_flat = pd.Series([100.0] * 300)
        risk_score = calculate_composite_risk_score(
            {"VIX": vix, "TNX": tnx, "IRX": irx, "Gold": gold_flat}
        )
        assert risk_score < 50
