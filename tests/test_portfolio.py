"""
test_portfolio.py — Unit tests for EnhancedPortfolioOptimizer.

Covers all 9 strategies, efficient frontier, portfolio metrics,
semi-covariance, CVaR matrix, and bubble-aware penalty.
"""
import numpy as np
import pandas as pd
import pytest
from app import EnhancedPortfolioOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOLS = dict(weights_sum=1e-6, non_negative=-1e-9)


def _opt(multi_price_df, bubble_scores=None):
    return EnhancedPortfolioOptimizer(multi_price_df, bubble_scores=bubble_scores, rf_rate=0.045)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestOptimizerConstruction:
    def test_attributes(self, multi_price_df):
        opt = _opt(multi_price_df)
        n = len(multi_price_df.columns)
        assert opt.n_assets == n
        assert opt.cov_matrix.shape == (n, n)
        assert len(opt.mean_returns) == n

    def test_cov_matrix_positive_semidefinite(self, multi_price_df):
        opt = _opt(multi_price_df)
        eigvals = np.linalg.eigvalsh(opt.cov_matrix.values)
        assert all(ev >= -1e-10 for ev in eigvals), "Covariance matrix is not PSD"

    def test_semi_cov_shape(self, multi_price_df):
        opt = _opt(multi_price_df)
        assert opt.semi_cov.shape == opt.cov_matrix.shape

    def test_cvar_matrix_shape(self, multi_price_df):
        opt = _opt(multi_price_df)
        assert opt.cvar_matrix.shape == opt.cov_matrix.shape


# ---------------------------------------------------------------------------
# Weight invariants — shared across all strategies
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    "maximum_sharpe",
    "minimum_variance",
    "risk_parity",
    "minimum_cvar",
    "maximum_diversification",
    "kelly_criterion",
    "black_litterman",
    "hierarchical_risk_parity",
    "equal_weight",
]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
class TestWeightInvariants:
    """Every strategy must return weights that sum to 1 and are non-negative."""

    def test_weights_sum_to_one(self, strategy, multi_price_df):
        opt = _opt(multi_price_df)
        weights = getattr(opt, strategy)()
        assert abs(sum(weights) - 1.0) < 1e-4, \
            f"{strategy}: weights sum = {sum(weights):.8f}"

    def test_weights_non_negative(self, strategy, multi_price_df):
        opt = _opt(multi_price_df)
        weights = getattr(opt, strategy)()
        assert all(w >= TOLS["non_negative"] for w in weights), \
            f"{strategy}: has negative weight: {min(weights):.6f}"

    def test_weights_length(self, strategy, multi_price_df):
        opt = _opt(multi_price_df)
        weights = getattr(opt, strategy)()
        assert len(weights) == multi_price_df.shape[1]


# ---------------------------------------------------------------------------
# Strategy-specific properties
# ---------------------------------------------------------------------------

class TestMaxSharpe:
    def test_sharpe_exceeds_min_var(self, multi_price_df):
        """Max Sharpe portfolio should have >= Sharpe than Min Variance."""
        opt = _opt(multi_price_df)
        w_ms = opt.maximum_sharpe()
        w_mv = opt.minimum_variance()
        m = opt.calculate_portfolio_metrics

        sharpe_ms = m(w_ms)["Sharpe Ratio"]
        sharpe_mv = m(w_mv)["Sharpe Ratio"]
        # Allow small numerical tolerance
        assert sharpe_ms >= sharpe_mv - 1e-3


class TestMinVariance:
    def test_vol_le_equal_weight(self, multi_price_df):
        """Min Variance portfolio must have vol ≤ equal-weight portfolio."""
        opt = _opt(multi_price_df)
        w_mv = opt.minimum_variance()
        w_ew = opt.equal_weight()
        m = opt.calculate_portfolio_metrics

        vol_mv = m(w_mv)["Volatility"]
        vol_ew = m(w_ew)["Volatility"]
        assert vol_mv <= vol_ew + 1e-4


class TestEqualWeight:
    def test_equal_weight_is_uniform(self, multi_price_df):
        opt = _opt(multi_price_df)
        w = opt.equal_weight()
        n = multi_price_df.shape[1]
        assert all(abs(wi - 1 / n) < 1e-9 for wi in w)


class TestRiskParity:
    def test_risk_contributions_equal(self, multi_price_df):
        """Each asset's fractional variance contribution should equal 1/N."""
        opt = _opt(multi_price_df)
        w = opt.risk_parity()
        n = opt.n_assets
        cov = opt.cov_matrix.values

        port_var = float(w @ cov @ w)
        if port_var == 0:
            pytest.skip("Portfolio variance is zero")
        marginal = cov @ w
        contrib = w * marginal / port_var          # fractional contributions
        target = 1.0 / n
        assert all(abs(c - target) < 0.05 for c in contrib), \
            f"Risk contributions not equal: {contrib}"


class TestHRP:
    def test_hrp_non_trivial(self, multi_price_df):
        """HRP should not just return equal weights for correlated assets."""
        opt = _opt(multi_price_df)
        w_hrp = opt.hierarchical_risk_parity()
        w_ew = opt.equal_weight()
        # At least one weight should differ meaningfully
        assert max(abs(w_hrp - w_ew)) > 1e-4


# ---------------------------------------------------------------------------
# Bubble-aware penalty
# ---------------------------------------------------------------------------

class TestBubbleAwareness:
    def test_bubble_reduces_weight(self, multi_price_df):
        """
        When asset 0 has bubble score=1.0 and others=0, its weight under
        bubble-aware Max Sharpe should be <= its weight without the penalty.
        """
        tickers = list(multi_price_df.columns)
        scores = {tickers[0]: 1.0, tickers[1]: 0.0, tickers[2]: 0.0, tickers[3]: 0.0}

        opt_plain  = _opt(multi_price_df)
        opt_bubble = _opt(multi_price_df, bubble_scores=scores)

        w_plain  = opt_plain.maximum_sharpe(bubble_aware=False)
        w_bubble = opt_bubble.maximum_sharpe(bubble_aware=True, penalty_factor=1.0)

        assert w_bubble[0] <= w_plain[0] + 0.05


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

class TestPortfolioMetrics:
    def test_metrics_keys(self, multi_price_df):
        opt = _opt(multi_price_df)
        w = opt.equal_weight()
        metrics = opt.calculate_portfolio_metrics(w)
        for key in ["Expected Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]:
            assert key in metrics

    def test_volatility_positive(self, multi_price_df):
        opt = _opt(multi_price_df)
        w = opt.equal_weight()
        assert opt.calculate_portfolio_metrics(w)["Volatility"] > 0

    def test_sharpe_includes_rf(self, multi_price_df):
        """Sharpe = (Ann Return - rf) / Vol — verify it is != raw return/vol."""
        opt = _opt(multi_price_df)
        w = opt.equal_weight()
        m = opt.calculate_portfolio_metrics(w)
        # Sharpe Ratio should be <= Expected Return / Volatility (rf reduces numerator)
        raw_ratio = m["Expected Return"] / m["Volatility"] if m["Volatility"] > 0 else 0
        assert m["Sharpe Ratio"] <= raw_ratio + 1e-6

    def test_max_drawdown_non_positive(self, multi_price_df):
        opt = _opt(multi_price_df)
        w = opt.equal_weight()
        assert opt.calculate_portfolio_metrics(w)["Max Drawdown"] <= 0


# ---------------------------------------------------------------------------
# Semi-covariance and CVaR matrix
# ---------------------------------------------------------------------------

class TestRiskMatrices:
    def test_semi_cov_uses_downside_rows(self, multi_price_df):
        """Semi-cov should be computed on rows where any asset is negative."""
        opt = _opt(multi_price_df)
        downside = opt.returns[(opt.returns < 0).any(axis=1)]
        expected = downside.cov() * 252
        pd.testing.assert_frame_equal(opt.semi_cov.round(10), expected.round(10))

    def test_cvar_matrix_portfolio_level_tail(self, multi_price_df):
        """CVaR matrix uses equal-weight tail dates, not per-column."""
        opt = _opt(multi_price_df)
        ew = opt.returns.mean(axis=1)
        tail = ew <= ew.quantile(0.05)
        expected = opt.returns.loc[tail].cov() * 252
        pd.testing.assert_frame_equal(opt.cvar_matrix.round(10), expected.round(10))
