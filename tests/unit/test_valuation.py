"""
test_valuation.py — Unit tests for EnhancedValuationMetrics.

Covers: CAPM, Beta, WACC, DCF guard, Fama-French, APT.
"""
import numpy as np
import pandas as pd
import pytest
from app import EnhancedValuationMetrics as EVM

# ---------------------------------------------------------------------------
# CAPM
# ---------------------------------------------------------------------------

class TestCAPM:
    """E(R_i) = R_f + beta * (E(R_m) - R_f)  with  E(R_m) - R_f = 5.7%"""

    ERP = 0.057  # equity risk premium used in app

    def test_capm_zero_beta(self):
        """Zero-beta asset should earn exactly the risk-free rate."""
        rf = 0.045
        result = EVM.calculate_capm_return(rf, beta=0.0)
        assert abs(result - rf) < 1e-9

    def test_capm_unit_beta(self):
        """Beta=1 asset earns rf + ERP."""
        rf = 0.045
        expected = rf + self.ERP
        result = EVM.calculate_capm_return(rf, beta=1.0)
        assert abs(result - expected) < 1e-9

    def test_capm_high_beta(self):
        """Beta=1.5 — verify linearity."""
        rf = 0.03
        expected = rf + 1.5 * self.ERP
        result = EVM.calculate_capm_return(rf, beta=1.5)
        assert abs(result - expected) < 1e-9

    def test_capm_negative_beta(self):
        """Negative beta (e.g. gold) should return < rf."""
        rf = 0.045
        result = EVM.calculate_capm_return(rf, beta=-0.3)
        assert result < rf

    def test_capm_return_type(self):
        result = EVM.calculate_capm_return(0.04, 1.2)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class TestBeta:
    def test_beta_known_value(self, single_price_series, benchmark_series):
        """
        Construct a synthetic asset with known beta=1.5 and verify recovery.
        """
        bench_ret = benchmark_series.pct_change().dropna()
        idx = bench_ret.index
        # asset = 1.5 * market + small noise
        rng = np.random.default_rng(0)
        asset_ret = 1.5 * bench_ret.values + rng.normal(0, 0.002, len(bench_ret))
        asset_ret_s = pd.Series(asset_ret, index=idx)
        asset_prices = 100 * (1 + asset_ret_s).cumprod()

        beta = EVM.calculate_beta(asset_prices, benchmark_series)
        assert 1.3 < beta < 1.7, f"Expected beta ≈ 1.5, got {beta:.4f}"

    def test_beta_market_itself_is_one(self, benchmark_series):
        """Regressing market on itself should give beta ≈ 1.0."""
        beta = EVM.calculate_beta(benchmark_series, benchmark_series)
        assert abs(beta - 1.0) < 0.01

    def test_beta_insufficient_data(self):
        """< 30 observations → fallback beta = 1.0."""
        s = pd.Series(range(1, 20), dtype=float)
        b = pd.Series(range(1, 20), dtype=float)
        beta = EVM.calculate_beta(s, b)
        assert beta == 1.0

    def test_beta_uses_sample_variance(self, benchmark_series):
        """
        Verify ddof=1 is used: compute manually and compare.
        """
        bench = benchmark_series
        asset = bench * 1.2  # perfect beta = 1.2

        bench_ret = bench.pct_change().dropna()
        asset_ret = asset.pct_change().dropna()
        common = bench_ret.index.intersection(asset_ret.index)
        a = asset_ret.loc[common].values
        b = bench_ret.loc[common].values

        expected_beta = np.cov(a, b, ddof=1)[0][1] / np.var(b, ddof=1)
        result = EVM.calculate_beta(asset, bench)
        assert abs(result - expected_beta) < 1e-6

    def test_beta_non_overlapping_dates(self):
        """No common dates → should not raise, returns fallback 1.0."""
        s1 = pd.Series([100, 101, 102], index=pd.bdate_range("2020-01-01", periods=3))
        s2 = pd.Series([100, 101, 102], index=pd.bdate_range("2021-01-01", periods=3))
        beta = EVM.calculate_beta(s1, s2)
        assert beta == 1.0


# ---------------------------------------------------------------------------
# WACC (offline — uses defaults when yfinance returns nothing)
# ---------------------------------------------------------------------------

class TestWACC:
    def test_wacc_returns_float(self):
        wacc = EVM.calculate_wacc("AAPL", rf_rate=0.045, beta=1.0)
        assert isinstance(wacc, float)

    def test_wacc_reasonable_range(self):
        """WACC should be between 3 % and 25 % for any plausible input."""
        for beta in [0.5, 1.0, 1.5, 2.0]:
            wacc = EVM.calculate_wacc("FAKEXYZ", rf_rate=0.045, beta=beta)
            assert 0.03 <= wacc <= 0.25, f"WACC={wacc:.4f} out of range for beta={beta}"

    def test_wacc_increases_with_beta(self):
        """Higher beta → higher WACC (cost of equity rises)."""
        w1 = EVM.calculate_wacc("FAKEXYZ", rf_rate=0.045, beta=0.8)
        w2 = EVM.calculate_wacc("FAKEXYZ", rf_rate=0.045, beta=1.5)
        assert w2 >= w1

    def test_wacc_higher_rf_raises_wacc(self):
        """Higher risk-free rate → higher WACC."""
        w1 = EVM.calculate_wacc("FAKEXYZ", rf_rate=0.02, beta=1.0)
        w2 = EVM.calculate_wacc("FAKEXYZ", rf_rate=0.06, beta=1.0)
        assert w2 >= w1


# ---------------------------------------------------------------------------
# DCF — guard condition
# ---------------------------------------------------------------------------

class TestDCF:
    def test_dcf_returns_value_or_none(self):
        result = EVM.calculate_dcf_value("FAKEXYZ", rf_rate=0.045, beta=1.0)
        assert result is None or isinstance(result, (int, float))

    def test_dcf_wacc_le_terminal_growth_returns_none(self, monkeypatch):
        """When WACC ≤ terminal_growth the function must return None (div-by-zero guard)."""
        # Force calculate_wacc to return a value below terminal_growth (0.02)
        monkeypatch.setattr(EVM, "calculate_wacc", staticmethod(lambda *a, **kw: 0.01))
        result = EVM.calculate_dcf_value("FAKEXYZ", rf_rate=0.01, beta=0.0)
        assert result is None


# ---------------------------------------------------------------------------
# Fama-French & APT — smoke tests (live data not available)
# ---------------------------------------------------------------------------

class TestFactorModels:
    def test_fama_french_returns_float(self, single_price_series):
        prices_df = single_price_series.to_frame("ASSET")
        result = EVM.calculate_fama_french_return("ASSET", prices_df, rf_rate=0.045, beta=1.0)
        assert isinstance(result, float)

    def test_fama_french_positive_expected_return(self, single_price_series):
        """For a typical market beta the expected return should be positive."""
        prices_df = single_price_series.to_frame("ASSET")
        result = EVM.calculate_fama_french_return("ASSET", prices_df, rf_rate=0.045, beta=1.0)
        assert result > 0

    def test_apt_returns_float(self, single_price_series):
        prices_df = single_price_series.to_frame("ASSET")
        result = EVM.calculate_apt_return("ASSET", prices_df, rf_rate=0.045)
        assert isinstance(result, float)
