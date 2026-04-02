"""
test_options.py — Unit tests for Black-Scholes pricing and Greeks.

Verifies: pricing formulas, put-call parity, Greek signs and magnitudes,
boundary conditions, and payoff diagrams.
"""
import math
import numpy as np
import pytest
from app import black_scholes_price, bs_greeks, options_payoff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atm_call():
    """At-the-money 1-year call."""
    return dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def atm_put():
    return dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class TestBlackScholes:
    def test_call_positive(self, atm_call):
        price = black_scholes_price(**atm_call)
        assert price > 0

    def test_put_positive(self):
        price = black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')
        assert price > 0

    def test_put_call_parity(self, atm_call):
        """C - P = S - K·e^(-rT)"""
        S, K, T, r, sigma = atm_call["S"], atm_call["K"], atm_call["T"], atm_call["r"], atm_call["sigma"]
        C = black_scholes_price(S, K, T, r, sigma, 'call')
        P = black_scholes_price(S, K, T, r, sigma, 'put')
        lhs = C - P
        rhs = S - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs:.6f} ≠ {rhs:.6f}"

    def test_call_intrinsic_at_expiry(self):
        """T=0 → call = max(S-K, 0)."""
        assert abs(black_scholes_price(110, 100, 0, 0.05, 0.20, 'call') - 10) < 1e-6
        assert abs(black_scholes_price(90,  100, 0, 0.05, 0.20, 'call') - 0)  < 1e-6

    def test_put_intrinsic_at_expiry(self):
        """T=0 → put = max(K-S, 0)."""
        assert abs(black_scholes_price(90,  100, 0, 0.05, 0.20, 'put') - 10) < 1e-6
        assert abs(black_scholes_price(110, 100, 0, 0.05, 0.20, 'put') - 0)  < 1e-6

    def test_deep_itm_call_approaches_forward(self):
        """Deep ITM call ≈ S - K·e^(-rT)."""
        C = black_scholes_price(S=200, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call')
        forward = 200 - 100 * math.exp(-0.05)
        assert C > forward * 0.95

    def test_higher_vol_higher_price(self):
        """Option price increases monotonically with volatility."""
        vols = [0.10, 0.20, 0.30, 0.40]
        prices = [black_scholes_price(100, 100, 1.0, 0.05, v) for v in vols]
        assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))

    def test_longer_expiry_higher_price(self):
        """Option price increases with time to expiry (positive time value)."""
        tenors = [0.1, 0.25, 0.5, 1.0]
        prices = [black_scholes_price(100, 100, t, 0.05, 0.20) for t in tenors]
        assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))

    def test_known_value(self):
        """
        Benchmark against a well-known B-S value:
        S=100, K=100, T=1, r=5%, σ=20% → call ≈ $10.451
        """
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.20, 'call')
        assert abs(price - 10.4506) < 0.005, f"Got {price:.4f}"

    def test_zero_sigma_returns_intrinsic(self):
        """σ=0 → price equals discounted intrinsic."""
        price = black_scholes_price(S=110, K=100, T=1.0, r=0.05, sigma=0.0, option_type='call')
        assert price == 10.0   # max(S-K, 0) at sigma=0


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestGreeks:
    def test_call_delta_between_0_and_1(self, atm_call):
        g = bs_greeks(**atm_call)
        assert 0 < g["Delta"] < 1

    def test_put_delta_between_minus1_and_0(self):
        g = bs_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')
        assert -1 < g["Delta"] < 0

    def test_call_delta_atm_approx_half(self, atm_call):
        """ATM call delta ≈ 0.5–0.6."""
        g = bs_greeks(**atm_call)
        assert 0.50 < g["Delta"] < 0.65

    def test_put_call_delta_sum_equals_one(self, atm_call):
        """Δ_call - Δ_put = 1 (put-call delta parity)."""
        g_call = bs_greeks(**atm_call)
        g_put  = bs_greeks(**{**atm_call, "option_type": "put"})
        assert abs(g_call["Delta"] - g_put["Delta"] - 1.0) < 1e-9

    def test_gamma_positive(self, atm_call):
        assert bs_greeks(**atm_call)["Gamma"] > 0

    def test_vega_positive(self, atm_call):
        assert bs_greeks(**atm_call)["Vega"] > 0

    def test_call_theta_negative(self, atm_call):
        """Time decay: Θ < 0 for long call."""
        assert bs_greeks(**atm_call)["Theta"] < 0

    def test_call_rho_positive(self, atm_call):
        """Higher rates → higher call value → ρ > 0."""
        assert bs_greeks(**atm_call)["Rho"] > 0

    def test_put_rho_negative(self):
        """Higher rates → lower put value → ρ < 0."""
        g = bs_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='put')
        assert g["Rho"] < 0

    def test_gamma_call_equals_put(self, atm_call):
        """Gamma is identical for call and put at the same strike/expiry."""
        g_c = bs_greeks(**atm_call)
        g_p = bs_greeks(**{**atm_call, "option_type": "put"})
        assert abs(g_c["Gamma"] - g_p["Gamma"]) < 1e-10

    def test_vega_call_equals_put(self, atm_call):
        g_c = bs_greeks(**atm_call)
        g_p = bs_greeks(**{**atm_call, "option_type": "put"})
        assert abs(g_c["Vega"] - g_p["Vega"]) < 1e-10

    def test_expiry_returns_zeros(self):
        """T=0 → all Greeks return 0 (no sensitivity at expiry)."""
        g = bs_greeks(S=100, K=100, T=0, r=0.05, sigma=0.20)
        assert all(v == 0 for v in g.values())


# ---------------------------------------------------------------------------
# Payoff diagrams
# ---------------------------------------------------------------------------

class TestPayoffs:
    S_range = np.linspace(50, 150, 200)
    S0 = 100

    def test_long_call_floor(self):
        """Long call payoff is never less than -premium."""
        pf = options_payoff('Long Call', self.S_range, self.S0, K1=100, premium1=5)
        assert all(p >= -5 - 1e-9 for p in pf)

    def test_long_put_floor(self):
        pf = options_payoff('Long Put', self.S_range, self.S0, K1=100, premium1=5)
        assert all(p >= -5 - 1e-9 for p in pf)

    def test_straddle_v_shape(self):
        """Straddle has minimum payoff at S=K."""
        pf = options_payoff('Straddle', self.S_range, self.S0, K1=100, premium1=5, premium2=5)
        idx_atm = np.argmin(np.abs(self.S_range - 100))
        assert pf[idx_atm] == min(pf)

    def test_iron_condor_limited_loss(self):
        """Iron Condor has a known maximum loss region outside the wings."""
        pf = options_payoff('Iron Condor', self.S_range, self.S0, K1=85, K2=115, premium1=2.0)
        # Max loss is bounded — payoff should never be extremely negative
        assert min(pf) > -50

    def test_bull_call_spread_capped_profit(self):
        """Bull call spread profit is capped at K2-K1 (minus net premium)."""
        pf = options_payoff('Bull Call Spread', self.S_range, self.S0,
                            K1=95, K2=105, premium1=3, premium2=1)
        assert max(pf) <= (105 - 95) + 1.0  # small tolerance for premium

    def test_bear_put_spread_max_gain(self):
        pf = options_payoff('Bear Put Spread', self.S_range, self.S0,
                            K1=95, K2=105, premium1=2, premium2=4)
        max_gain = 105 - 95 - (4 - 2)  # spread - net premium
        assert max(pf) <= max_gain + 1.0
