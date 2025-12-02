# app.py
# Advanced Portfolio Analytics & Bubble Detection Platform
# pip install streamlit yfinance pandas numpy plotly scipy statsmodels scikit-learn ta xlsxwriter

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize, curve_fit
from scipy.signal import periodogram
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import traceback
import warnings
from datetime import datetime, timedelta
import xlsxwriter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ta
import time  # Added for auto-refresh loop

# ========================================================================
# SYSTEM CONFIGURATION
# ========================================================================
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="QuantLab - Advanced Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

# Initialize Session State
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'portfolio_weights' not in st.session_state:
    st.session_state.portfolio_weights = None
# Initialize timestamp state
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = "Initializing..."

# ========================================================================
# ENHANCED CSS & UI STYLING
# ========================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* Enhanced Dark Theme */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #151932 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* Animated Gradient Headers */
        .gradient-text {
            background: linear-gradient(90deg, #00f2ea, #ff0080, #00f2ea);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient 3s ease infinite;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Enhanced Metrics Cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(0,242,234,0.1) 0%, rgba(255,0,128,0.1) 100%);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,242,234,0.3);
        }
        
        /* Premium Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        
        /* Live Indicator */
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 242, 234, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(0, 242, 234, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 242, 234, 0); }
        }
        
        .live-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00f2ea;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
    </style>
    """, unsafe_allow_html=True)

# ========================================================================
# DATA UTILITIES
# ========================================================================

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    """Fetches the latest 10-Year Treasury Yield"""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            latest_yield = hist['Close'].iloc[-1]
            return float(latest_yield) / 100
        else:
            return 0.045
    except Exception:
        return 0.045

@st.cache_data(ttl=60) # Reduced TTL to 60 seconds for live updates
def fetch_market_data(tickers, start_date, end_date):
    """Robust data fetching using yfinance"""
    if not tickers:
        return pd.DataFrame()
    
    data = yf.download(tickers, start=start_date, end=end_date, 
                       group_by='ticker', auto_adjust=True, progress=False)
    
    prices = pd.DataFrame()
    
    if len(tickers) == 1:
        ticker = tickers[0]
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns:
                 prices[ticker] = data['Close']
            else:
                 prices[ticker] = data[ticker]['Close']
        elif 'Close' in data.columns:
            prices[ticker] = data['Close']
        elif ticker in data.columns:
             prices[ticker] = data[ticker]
    else:
        for t in tickers:
            try:
                if t in data.columns.levels[0]:
                    prices[t] = data[t]['Close']
            except:
                if (t, 'Close') in data.columns:
                    prices[t] = data[(t, 'Close')]
                elif t in data.columns: # fallback
                    prices[t] = data[t]
                
    prices.dropna(inplace=True)
    return prices

# ========================================================================
# ENHANCED VALUATION MODELS
# ========================================================================

class EnhancedValuationMetrics:
    """Comprehensive valuation metrics including DCF, CAPM, Fama-French, APT"""
    
    @staticmethod
    def calculate_wacc(ticker, rf_rate):
        """Calculate Weighted Average Cost of Capital"""
        try:
            info = yf.Ticker(ticker).info
            
            # Get financial data
            market_cap = info.get('marketCap', 1e9)
            total_debt = info.get('totalDebt', 0)
            tax_rate = 0.21  # Corporate tax rate
            
            # Cost of equity (CAPM)
            beta = info.get('beta', 1.0) or 1.0
            market_premium = 0.08  # Historical market risk premium
            cost_of_equity = rf_rate + beta * market_premium
            
            # Cost of debt
            if total_debt > 0:
                interest_expense = info.get('interestExpense', 0)
                cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.04
            else:
                cost_of_debt = 0.04
            
            # WACC calculation
            total_value = market_cap + total_debt
            wacc = (market_cap/total_value * cost_of_equity + 
                   total_debt/total_value * cost_of_debt * (1 - tax_rate))
            
            return wacc
        except:
            return 0.10  # Default WACC
    
    @staticmethod
    def calculate_dcf_value(ticker, rf_rate):
        """Calculate DCF Enterprise Value"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get cash flows
            cf_statement = stock.cashflow
            if cf_statement.empty:
                return None
            
            # Get free cash flow (simplified)
            if 'Free Cash Flow' in cf_statement.index:
                fcf = cf_statement.loc['Free Cash Flow'].iloc[0]
            else:
                # Attempt to calculate manually if not present
                try:
                    operating_cf = cf_statement.loc['Total Cash From Operating Activities'].iloc[0]
                    capex = abs(cf_statement.loc['Capital Expenditures'].iloc[0])
                    fcf = operating_cf - capex
                except:
                    return None
            
            # Growth assumptions
            growth_rate = 0.05  # Conservative growth
            terminal_growth = 0.02
            
            # Calculate WACC
            wacc = EnhancedValuationMetrics.calculate_wacc(ticker, rf_rate)
            
            # Project cash flows (5 years)
            projected_cf = []
            for year in range(1, 6):
                cf = fcf * (1 + growth_rate) ** year
                pv = cf / (1 + wacc) ** year
                projected_cf.append(pv)
            
            # Terminal value
            terminal_cf = fcf * (1 + growth_rate) ** 5 * (1 + terminal_growth)
            terminal_value = terminal_cf / (wacc - terminal_growth)
            pv_terminal = terminal_value / (1 + wacc) ** 5
            
            # Enterprise value
            enterprise_value = sum(projected_cf) + pv_terminal
            
            return enterprise_value
        except:
            return None
    
    @staticmethod
    def calculate_capm_return(ticker, rf_rate):
        """Calculate expected return using CAPM"""
        try:
            info = yf.Ticker(ticker).info
            beta = info.get('beta', 1.0) or 1.0
            market_premium = 0.08
            return rf_rate + beta * market_premium
        except:
            return rf_rate + 0.08
    
    @staticmethod
    def calculate_fama_french_return(ticker, prices, rf_rate):
        """Calculate expected return using Fama-French 3-factor model"""
        try:
            info = yf.Ticker(ticker).info
            
            # Factor loadings (simplified)
            beta_market = info.get('beta', 1.0) or 1.0
            
            # Size factor (SMB) - based on market cap
            market_cap = info.get('marketCap', 1e9)
            if market_cap < 2e9:  # Small cap
                beta_smb = 0.5
            elif market_cap < 10e9:  # Mid cap
                beta_smb = 0.2
            else:  # Large cap
                beta_smb = -0.1
            
            # Value factor (HML) - based on P/B ratio
            pb_ratio = info.get('priceToBook', 3.0) or 3.0
            if pb_ratio < 1:  # Value
                beta_hml = 0.4
            elif pb_ratio < 3:  # Blend
                beta_hml = 0.1
            else:  # Growth
                beta_hml = -0.2
            
            # Factor premiums (historical averages)
            market_premium = 0.08
            smb_premium = 0.02
            hml_premium = 0.04
            
            # Fama-French expected return
            expected_return = (rf_rate + 
                              beta_market * market_premium +
                              beta_smb * smb_premium +
                              beta_hml * hml_premium)
            
            return expected_return
        except:
            return rf_rate + 0.10
    
    @staticmethod
    def calculate_apt_return(ticker, prices, rf_rate):
        """Calculate expected return using Arbitrage Pricing Theory"""
        try:
            returns = prices.pct_change().dropna()
            
            # Economic factors (simplified)
            # Factor 1: Market risk
            market_factor = returns.mean().mean() * 252
            
            # Factor 2: Volatility risk
            vol_factor = returns.std().mean() * np.sqrt(252)
            
            # Factor 3: Momentum
            if len(prices) > 60:
                momentum_factor = (prices.iloc[-20:].mean() / prices.iloc[-60:-20].mean()).mean() - 1
            else:
                momentum_factor = 0.05
            
            # Factor sensitivities (betas) - simplified
            beta_market = 1.0
            beta_vol = -0.5 if vol_factor > 0.3 else 0.2
            beta_momentum = 0.3 if momentum_factor > 0 else -0.1
            
            # Risk premiums
            market_premium = 0.08
            vol_premium = 0.03
            momentum_premium = 0.02
            
            # APT expected return
            expected_return = (rf_rate +
                              beta_market * market_premium +
                              beta_vol * vol_premium +
                              beta_momentum * momentum_premium)
            
            return expected_return
        except:
            return rf_rate + 0.09
    
    @staticmethod
    def calculate_bubble_burst_impact(ticker, prices, bubble_score):
        """Estimate potential loss in a bubble burst scenario"""
        try:
            returns = prices.pct_change().dropna()
            
            # Historical worst drawdowns
            cumulative = (1 + returns).cumprod()
            drawdowns = (cumulative - cumulative.cummax()) / cumulative.cummax()
            worst_dd = drawdowns.min()
            
            # Adjust for bubble score
            if bubble_score > 0.7:
                impact_multiplier = 1.5
            elif bubble_score > 0.4:
                impact_multiplier = 1.2
            else:
                impact_multiplier = 1.0
            
            # Estimated impact
            bubble_burst_impact = worst_dd * impact_multiplier
            
            # Add volatility adjustment
            if len(returns) > 20:
                current_vol = returns.iloc[-20:].std() * np.sqrt(252)
                historical_vol = returns.std() * np.sqrt(252)
                vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
            else:
                vol_ratio = 1.0
            
            # Final impact estimate
            final_impact = bubble_burst_impact * max(vol_ratio, 1.0)
            
            return abs(final_impact)  # Return as positive percentage
        except:
            return 0.30  # Default 30% potential loss

# ========================================================================
# ENHANCED PORTFOLIO OPTIMIZATION
# ========================================================================

class EnhancedPortfolioOptimizer:
    def __init__(self, prices, bubble_scores=None, rf_rate=0.045):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252
        self.bubble_scores = bubble_scores or {}
        self.rf_rate = rf_rate
        self.n_assets = len(prices.columns)
        
        # Calculate additional risk metrics
        self.semi_cov = self._calculate_semi_covariance()
        self.cvar_matrix = self._calculate_cvar_matrix()
        
    def _calculate_semi_covariance(self):
        """Calculate semi-covariance matrix (downside risk)"""
        downside_returns = self.returns.copy()
        downside_returns[downside_returns > 0] = 0
        return downside_returns.cov() * 252
    
    def _calculate_cvar_matrix(self, alpha=0.05):
        """Calculate CVaR covariance matrix"""
        threshold = self.returns.quantile(alpha)
        # Handle each column individually to find tail returns
        # This is a simplified approximation for the matrix
        return self.cov_matrix # Returning standard cov for stability in this snippet
    
    def _bubble_penalty(self, weights, penalty_factor=0.5):
        """Apply penalty to weights based on bubble scores"""
        if not self.bubble_scores:
            return 0
        
        penalty = 0
        for i, ticker in enumerate(self.prices.columns):
            if ticker in self.bubble_scores:
                penalty += penalty_factor * self.bubble_scores[ticker] * weights[i]**2
        return penalty
    
    def maximum_sharpe(self, bubble_aware=False, penalty_factor=0.5):
        """Maximum Sharpe Ratio Portfolio"""
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, self.mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = (portfolio_return - self.rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            if bubble_aware:
                sharpe -= self._bubble_penalty(weights, penalty_factor)
            
            return -sharpe
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_guess
    
    def minimum_variance(self, bubble_aware=False, penalty_factor=0.5):
        """Minimum Variance Portfolio"""
        def objective(weights):
            portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            if bubble_aware:
                portfolio_var += self._bubble_penalty(weights, penalty_factor)
            return portfolio_var
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_guess
    
    def risk_parity(self, bubble_aware=False, penalty_factor=0.5):
        """Risk Parity Portfolio"""
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = portfolio_vol / self.n_assets
            mse = np.sum((contrib - target_contrib)**2)
            
            if bubble_aware:
                for i, ticker in enumerate(self.prices.columns):
                    if ticker in self.bubble_scores:
                        adjustment = 1 - (self.bubble_scores[ticker] * penalty_factor)
                        mse += (contrib[i] - target_contrib * adjustment)**2
            
            return mse
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(self.n_assets)) # Avoid 0 weights for risk parity
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_guess
    
    def minimum_cvar(self, alpha=0.05, bubble_aware=False, penalty_factor=0.5):
        """Minimum CVaR Portfolio"""
        def objective(weights):
            portfolio_returns = self.returns.dot(weights)
            var = np.percentile(portfolio_returns, alpha * 100)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            
            if bubble_aware:
                cvar -= self._bubble_penalty(weights, penalty_factor)
            
            return -cvar  # Minimize negative CVaR
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_guess
    
    def maximum_diversification(self, bubble_aware=False, penalty_factor=0.5):
        """Maximum Diversification Portfolio"""
        def negative_div_ratio(weights):
            weighted_avg_vol = np.dot(weights, np.sqrt(np.diag(self.cov_matrix)))
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
            
            if bubble_aware:
                div_ratio -= self._bubble_penalty(weights, penalty_factor)
            
            return -div_ratio
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(negative_div_ratio, initial_guess, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_guess
    
    def kelly_criterion(self, leverage_limit=1.0):
        """Kelly Criterion Portfolio"""
        # Simplified Kelly for multiple assets
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except:
            inv_cov = np.linalg.pinv(self.cov_matrix) # Psuedo-inverse if singular
            
        excess_returns = self.mean_returns - self.rf_rate
        
        # Raw Kelly weights
        raw_weights = np.dot(inv_cov, excess_returns)
        
        # Apply leverage limit
        total_weight = np.sum(np.abs(raw_weights))
        if total_weight > leverage_limit:
            raw_weights = raw_weights * (leverage_limit / total_weight)
        
        # Convert to long-only if needed
        weights = np.maximum(raw_weights, 0)
        sum_weights = np.sum(weights)
        weights = weights / sum_weights if sum_weights > 0 else np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def black_litterman(self, market_cap_weights=None, views=None, view_confidence=None):
        """Black-Litterman Portfolio"""
        if market_cap_weights is None:
            market_cap_weights = np.ones(self.n_assets) / self.n_assets
        
        # Market implied returns
        lambda_param = (self.mean_returns.mean() - self.rf_rate) / self.cov_matrix.diagonal().mean()
        pi = lambda_param * np.dot(self.cov_matrix, market_cap_weights)
        
        if views is None or view_confidence is None:
            # No views, return market weights
            return market_cap_weights
        
        # Incorporate views (simplified)
        tau = 0.05
        P = np.eye(self.n_assets)  # Identity for absolute views
        Q = views
        omega = np.diag(view_confidence)
        
        # Black-Litterman formula
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except:
            inv_cov = np.linalg.pinv(self.cov_matrix)

        bl_returns = pi + tau * self.cov_matrix @ P.T @ np.linalg.inv(
            P @ tau * self.cov_matrix @ P.T + omega
        ) @ (Q - P @ pi)
        
        # Optimal weights
        weights = np.dot(inv_cov, bl_returns)
        sum_w = np.sum(weights)
        weights = weights / sum_w if sum_w != 0 else weights
        
        return np.maximum(weights, 0)  # Long-only
    
    def hierarchical_risk_parity(self):
        """Hierarchical Risk Parity (HRP) Portfolio"""
        # Calculate correlation matrix
        corr_matrix = self.returns.corr()
        
        # Distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        condensed_dist = squareform(dist_matrix)
        
        # Hierarchical clustering
        link = linkage(condensed_dist, 'single')
        
        # Get quasi-diagonal matrix
        def get_quasi_diag(link):
            link = link.astype(int)
            sort_idx = []
            for i in range(link.shape[0]):
                if link[i, 0] < self.n_assets:
                    sort_idx.append(link[i, 0])
                if link[i, 1] < self.n_assets:
                    sort_idx.append(link[i, 1])
            
            remaining = set(range(self.n_assets)) - set(sort_idx)
            sort_idx.extend(list(remaining))
            
            return sort_idx[:self.n_assets]
        
        sort_idx = get_quasi_diag(link)
        
        # Recursive bisection
        def recursive_bisection(cov, sort_idx):
            weights = np.ones(len(sort_idx))
            clusters = [sort_idx]
            
            while len(clusters) > 0:
                clusters = [c[1:] for c in clusters if len(c) > 1]
                
                for i in range(0, len(clusters), 2):
                    if i + 1 < len(clusters):
                        cluster1 = clusters[i]
                        cluster2 = clusters[i + 1]
                        
                        # Calculate cluster variances
                        var1 = cov[np.ix_(cluster1, cluster1)].sum()
                        var2 = cov[np.ix_(cluster2, cluster2)].sum()
                        
                        # Allocate inversely to variance
                        alpha = var2 / (var1 + var2)
                        
                        weights[cluster1] *= alpha
                        weights[cluster2] *= (1 - alpha)
            
            return weights / weights.sum()
        
        # Get HRP weights
        weights = recursive_bisection(self.cov_matrix.values, sort_idx)
        
        # Reorder to original asset order
        final_weights = np.zeros(self.n_assets)
        for i, idx in enumerate(sort_idx):
            if idx < self.n_assets:
                final_weights[idx] = weights[i]
        
        return final_weights
    
    def equal_weight(self):
        """Equal Weight Portfolio"""
        return np.ones(self.n_assets) / self.n_assets
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate comprehensive portfolio metrics"""
        portfolio_return = np.dot(weights, self.mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (portfolio_return - self.rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Downside deviation
        portfolio_returns = self.returns.dot(weights)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_dev = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        sortino = (portfolio_return - self.rf_rate) / downside_dev if downside_dev > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        max_dd = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Calmar ratio
        calmar = portfolio_return / abs(max_dd) if max_dd != 0 else 0
        
        # CVaR (95%)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() * 252 if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else var_95 * 252
        
        # Diversification ratio
        weighted_avg_vol = np.dot(weights, np.sqrt(np.diag(self.cov_matrix)))
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Effective number of assets (Herfindahl)
        herfindahl = np.sum(weights**2)
        eff_n = 1 / herfindahl if herfindahl > 0 else self.n_assets
        
        return {
            'Expected Return': portfolio_return,
            'Volatility': portfolio_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'CVaR (95%)': cvar_95,
            'Downside Deviation': downside_dev,
            'Diversification Ratio': div_ratio,
            'Effective N Assets': eff_n
        }
    
    def efficient_frontier(self, n_portfolios=100):
        """Generate efficient frontier"""
        target_returns = np.linspace(self.mean_returns.min(), self.mean_returns.max(), n_portfolios)
        
        frontier_weights = []
        frontier_vol = []
        frontier_return = []
        
        for target in target_returns:
            def objective(weights):
                return np.dot(weights.T, np.dot(self.cov_matrix, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - target}
            ]
            
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            initial_guess = np.array([1/self.n_assets] * self.n_assets)
            
            result = minimize(objective, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                frontier_weights.append(weights)
                frontier_vol.append(np.sqrt(objective(weights)))
                frontier_return.append(target)
        
        return frontier_return, frontier_vol, frontier_weights
    
    def backtest_portfolio(self, weights, rebalance_freq='M'):
        """Backtest portfolio performance"""
        portfolio_returns = self.returns.dot(weights)
        
        # Calculate rolling metrics
        rolling_window = 252  # 1 year
        
        if len(portfolio_returns) > rolling_window:
            rolling_returns = portfolio_returns.rolling(window=rolling_window)
            rolling_sharpe = (rolling_returns.mean() * 252 - self.rf_rate) / (rolling_returns.std() * np.sqrt(252))
        else:
            rolling_sharpe = pd.Series([np.nan] * len(portfolio_returns), index=portfolio_returns.index)
        
        # Cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Drawdown analysis
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return {
            'returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'drawdown': drawdown,
            'rolling_sharpe': rolling_sharpe
        }

# ========================================================================
# PORTFOLIO OPTIMIZATION TAB RENDERER
# ========================================================================

def render_portfolio_optimization_tab(data):
    """Enhanced Portfolio Optimization Tab"""
    st.markdown("### 💼 Advanced Portfolio Optimization")
    
    # Configuration sidebar
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        optimization_method = st.selectbox(
            "Optimization Strategy",
            [
                "Maximum Sharpe Ratio",
                "Minimum Variance",
                "Risk Parity",
                "Minimum CVaR",
                "Maximum Diversification",
                "Kelly Criterion",
                "Black-Litterman",
                "Hierarchical Risk Parity",
                "Equal Weight"
            ]
        )
    
    with col2:
        bubble_aware = st.checkbox("Bubble-Aware Optimization", value=True)
        if bubble_aware:
            penalty_factor = st.slider("Bubble Penalty Factor", 0.0, 1.0, 0.5, 0.1)
        else:
            penalty_factor = 0.0
    
    with col3:
        show_efficient_frontier = st.checkbox("Show Efficient Frontier", value=False)
        show_backtest = st.checkbox("Show Backtest Results", value=False)
    
    # Initialize optimizer
    optimizer = EnhancedPortfolioOptimizer(
        data['prices'], 
        data['bubble_scores'], 
        data['rf_rate']
    )
    
    # Get optimal weights based on selected method
    if optimization_method == "Maximum Sharpe Ratio":
        weights = optimizer.maximum_sharpe(bubble_aware, penalty_factor)
    elif optimization_method == "Minimum Variance":
        weights = optimizer.minimum_variance(bubble_aware, penalty_factor)
    elif optimization_method == "Risk Parity":
        weights = optimizer.risk_parity(bubble_aware, penalty_factor)
    elif optimization_method == "Minimum CVaR":
        weights = optimizer.minimum_cvar(bubble_aware=bubble_aware, penalty_factor=penalty_factor)
    elif optimization_method == "Maximum Diversification":
        weights = optimizer.maximum_diversification(bubble_aware, penalty_factor)
    elif optimization_method == "Kelly Criterion":
        leverage_limit = st.slider("Leverage Limit", 0.5, 2.0, 1.0, 0.1)
        weights = optimizer.kelly_criterion(leverage_limit)
    elif optimization_method == "Black-Litterman":
        st.info("Using equal market cap weights. Custom views can be added.")
        weights = optimizer.black_litterman()
    elif optimization_method == "Hierarchical Risk Parity":
        weights = optimizer.hierarchical_risk_parity()
    else:  # Equal Weight
        weights = optimizer.equal_weight()
    
    # Calculate metrics
    metrics = optimizer.calculate_portfolio_metrics(weights)
    
    # Display results in columns
    st.markdown("#### Portfolio Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Expected Return", f"{metrics['Expected Return']:.2%}")
        st.metric("Volatility", f"{metrics['Volatility']:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.2f}")
    
    with col4:
        st.metric("CVaR (95%)", f"{metrics['CVaR (95%)']:.2%}")
        st.metric("Downside Dev", f"{metrics['Downside Deviation']:.2%}")
    
    with col5:
        st.metric("Div. Ratio", f"{metrics['Diversification Ratio']:.2f}")
        st.metric("Effective N", f"{metrics['Effective N Assets']:.1f}")
    
    # Portfolio allocation visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced pie chart
        fig = go.Figure(data=[go.Pie(
            labels=data['tickers'],
            values=weights,
            hole=0.4,
            marker=dict(colors=px.colors.qualitative.Set3),
            textposition='auto',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>' +
                          'Weight: %{percent}<br>' +
                          'Value: %{value:.4f}<br>' +
                          '<extra></extra>'
        )])
        
        fig.update_layout(
            title=f"{optimization_method} Portfolio Allocation",
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(text=f'Sharpe: {metrics["Sharpe Ratio"]:.2f}', 
                     x=0.5, y=0.5, font_size=16, showarrow=False)
            ]
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Weights table with additional info
        weights_df = pd.DataFrame({
            'Asset': data['tickers'],
            'Weight': weights,
            'Contribution': weights * metrics['Expected Return'],
            'Risk Contrib': weights * np.sqrt(np.diag(optimizer.cov_matrix)),
            'Bubble Score': [data['bubble_scores'].get(t, 0) for t in data['tickers']]
        })
        
        st.dataframe(
            weights_df.style.format({
                'Weight': '{:.2%}',
                'Contribution': '{:.2%}',
                'Risk Contrib': '{:.2%}',
                'Bubble Score': '{:.2%}'
            }).background_gradient(subset=['Weight'], cmap='RdYlGn'),
            width='stretch'
        )
    
    # Efficient Frontier
    if show_efficient_frontier:
        st.markdown("#### Efficient Frontier")
        
        with st.spinner("Generating efficient frontier..."):
            frontier_return, frontier_vol, frontier_weights = optimizer.efficient_frontier()
            
            # Create frontier plot
            fig = go.Figure()
            
            # Efficient frontier line
            fig.add_trace(go.Scatter(
                x=frontier_vol,
                y=frontier_return,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='cyan', width=2)
            ))
            
            # Current portfolio
            fig.add_trace(go.Scatter(
                x=[metrics['Volatility']],
                y=[metrics['Expected Return']],
                mode='markers',
                name='Current Portfolio',
                marker=dict(size=15, color='red', symbol='star')
            ))
            
            # Individual assets
            for ticker in data['tickers']:
                idx = list(data['tickers']).index(ticker)
                fig.add_trace(go.Scatter(
                    x=[np.sqrt(optimizer.cov_matrix.iloc[idx, idx])],
                    y=[optimizer.mean_returns.iloc[idx]],
                    mode='markers+text',
                    name=ticker,
                    text=[ticker],
                    textposition='top center',
                    marker=dict(size=10)
                ))
            
            # Capital Market Line
            if len(frontier_return) > 0 and len(frontier_vol) > 0:
                max_sharpe_idx = np.argmax([(r - data['rf_rate']) / v for r, v in zip(frontier_return, frontier_vol)])
                cml_x = [0, frontier_vol[max_sharpe_idx] * 2]
                cml_y = [data['rf_rate'], data['rf_rate'] + (frontier_return[max_sharpe_idx] - data['rf_rate']) / frontier_vol[max_sharpe_idx] * cml_x[1]]
                
                fig.add_trace(go.Scatter(
                    x=cml_x,
                    y=cml_y,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='yellow', width=1, dash='dash')
                ))
            
            fig.update_layout(
                title="Efficient Frontier Analysis",
                xaxis_title="Volatility (Annual)",
                yaxis_title="Expected Return (Annual)",
                template='plotly_dark',
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='closest'
            )
            
            st.plotly_chart(fig, width='stretch')
    
    # Backtest Results
    if show_backtest:
        st.markdown("#### Backtest Results")
        
        backtest_results = optimizer.backtest_portfolio(weights)
        
        # Create subplots for backtest visualization
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Cumulative Returns', 'Drawdown', 'Rolling Sharpe Ratio')
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=backtest_results['cumulative_returns'].index,
                y=backtest_results['cumulative_returns'].values,
                name='Portfolio',
                line=dict(color='cyan', width=2)
            ),
            row=1, col=1
        )
        
        # Benchmark (equal weight)
        equal_weights = optimizer.equal_weight()
        benchmark_returns = optimizer.returns.dot(equal_weights)
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                name='Equal Weight Benchmark',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=backtest_results['drawdown'].index,
                y=backtest_results['drawdown'].values,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=backtest_results['rolling_sharpe'].index,
                y=backtest_results['rolling_sharpe'].values,
                name='Rolling Sharpe',
                line=dict(color='green', width=1)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=800,
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Performance statistics
        st.markdown("#### Backtest Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_return = (backtest_results['cumulative_returns'].iloc[-1] - 1) * 100
        annual_return = (backtest_results['cumulative_returns'].iloc[-1] ** (252/len(backtest_results['returns'])) - 1) * 100
        win_rate = (backtest_results['returns'] > 0).mean() * 100
        best_day = backtest_results['returns'].max() * 100
        worst_day = backtest_results['returns'].min() * 100
        
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")
            st.metric("Annualized Return", f"{annual_return:.2f}%")
        
        with col2:
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Best Day", f"{best_day:.2f}%")
        
        with col3:
            st.metric("Worst Day", f"{worst_day:.2f}%")
            st.metric("Avg Drawdown", f"{backtest_results['drawdown'].mean():.2%}")
        
        with col4:
            st.metric("Max Drawdown", f"{backtest_results['drawdown'].min():.2%}")
            st.metric("Recovery Days", f"{(backtest_results['drawdown'] < 0).sum()}")
    
    # Comparison table for multiple strategies
    st.markdown("#### Strategy Comparison")
    
    strategies = ["Maximum Sharpe Ratio", "Minimum Variance", "Risk Parity", 
                 "Maximum Diversification", "Equal Weight"]
    
    comparison_data = []
    
    for strategy in strategies:
        if strategy == "Maximum Sharpe Ratio":
            w = optimizer.maximum_sharpe(bubble_aware, penalty_factor)
        elif strategy == "Minimum Variance":
            w = optimizer.minimum_variance(bubble_aware, penalty_factor)
        elif strategy == "Risk Parity":
            w = optimizer.risk_parity(bubble_aware, penalty_factor)
        elif strategy == "Maximum Diversification":
            w = optimizer.maximum_diversification(bubble_aware, penalty_factor)
        else:
            w = optimizer.equal_weight()
        
        m = optimizer.calculate_portfolio_metrics(w)
        comparison_data.append({
            'Strategy': strategy,
            'Return': m['Expected Return'],
            'Volatility': m['Volatility'],
            'Sharpe': m['Sharpe Ratio'],
            'Max DD': m['Max Drawdown'],
            'Div Ratio': m['Diversification Ratio']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.dataframe(
        comparison_df.style.format({
            'Return': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max DD': '{:.2%}',
            'Div Ratio': '{:.2f}'
        }).background_gradient(subset=['Sharpe'], cmap='RdYlGn'),
        width='stretch'
    )
    
    # Export portfolio data
    st.markdown("#### Export Portfolio Data")
    
    export_data = pd.DataFrame({
        'Asset': data['tickers'],
        'Weight': weights,
        'Expected Return': optimizer.mean_returns,
        'Volatility': np.sqrt(np.diag(optimizer.cov_matrix)),
        'Bubble Score': [data['bubble_scores'].get(t, 0) for t in data['tickers']]
    })
    
    csv = export_data.to_csv(index=False)
    st.download_button(
        "📥 Download Portfolio Weights (CSV)",
        csv,
        f"portfolio_{optimization_method.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        key='download-portfolio-csv'
    )

# ========================================================================
# BUBBLE DETECTION CLASSES
# ========================================================================

class MetcalfeLawAdvanced:
    """Advanced Metcalfe's Law implementation based on Bitcoin paper"""
    
    def __init__(self):
        self.models = {
            'generalized': {'gamma': 1.166, 'A0': 6.053e-07}
        }
    
    def calculate_network_value(self, users, model='generalized'):
        """Calculate network value using Metcalfe's Law"""
        if model == 'generalized':
            A0 = self.models['generalized']['A0']
            gamma = self.models['generalized']['gamma']
            network_pairs = users * (users - 1) / 2
            return A0 * (network_pairs ** gamma)
        return users
    
    def calculate_mmv_ratio(self, market_price, network_value):
        """Calculate Market-to-Metcalfe Value ratio"""
        with np.errstate(divide='ignore', invalid='ignore'):
            mmv = np.where(network_value > 0, market_price / network_value, np.nan)
        return mmv
    
    def detect_bubble_regime(self, mmv_ratio):
        """Classify bubble regime based on MMV ratio"""
        if np.isnan(mmv_ratio):
            return 'Unknown'
        
        if mmv_ratio > 1.25:
            return 'Extreme Bubble'
        elif mmv_ratio > 1.15:
            return 'Bubble Formation'
        elif 0.95 <= mmv_ratio <= 1.05:
            return 'Fair Value'
        elif mmv_ratio < 0.85:
            return 'Undervalued'
        else:
            return 'Transition'

class LongMemoryEstimators:
    """Long-memory parameter estimation using GPH method"""
    
    @staticmethod
    def gph_estimator(returns, bandwidth=None):
        """GPH estimator for fractional differencing parameter d"""
        returns = np.asarray(returns)
        n = len(returns)
        
        if bandwidth is None:
            bandwidth = int(n ** 0.5)
        
        freqs, psd = periodogram(returns - np.mean(returns))
        
        m = min(bandwidth, len(freqs) // 2)
        low_freqs = freqs[1:m+1]
        low_psd = psd[1:m+1]
        
        X = np.log(2 * np.sin(low_freqs / 2))
        Y = np.log(low_psd + 1e-10)
        
        X_with_const = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]
        
        d = -beta[1] / 2
        
        residuals = Y - X_with_const @ beta
        se = np.sqrt(np.sum(residuals**2) / (len(X) - 2)) / np.sqrt(np.sum((X - np.mean(X))**2))
        
        return d, se

class BubbleDetector:
    """Comprehensive bubble detection using multiple methods"""
    
    def __init__(self):
        self.metcalfe = MetcalfeLawAdvanced()
        self.long_memory = LongMemoryEstimators()
    
    def detect_bubbles(self, prices, returns):
        """Detect bubbles using multiple indicators"""
        results = {}
        
        # 1. Generate proxy network data
        users = np.sqrt(prices.values) * 1000 * (1 + np.random.normal(0, 0.1, len(prices)))
        users = np.abs(users)
        
        # 2. Metcalfe's Law analysis
        network_values = self.metcalfe.calculate_network_value(users)
        mmv_ratios = self.metcalfe.calculate_mmv_ratio(prices.values, network_values)
        
        current_mmv = mmv_ratios[-1] if len(mmv_ratios) > 0 else np.nan
        results['mmv_ratio'] = current_mmv
        results['mmv_regime'] = self.metcalfe.detect_bubble_regime(current_mmv)
        
        # 3. Long memory analysis
        if len(returns) > 50:
            d_gph, se_gph = self.long_memory.gph_estimator(returns.values)
            results['d_parameter'] = d_gph
            results['d_se'] = se_gph
        else:
            results['d_parameter'] = 0
            results['d_se'] = 0
        
        # 4. Statistical measures
        results['skewness'] = stats.skew(returns)
        results['kurtosis'] = stats.kurtosis(returns)
        
        # 5. Volatility clustering test
        try:
            squared_returns = returns ** 2
            lb_test = acorr_ljungbox(squared_returns, lags=10, return_df=True)
            results['has_vol_clustering'] = any(lb_test['lb_pvalue'] < 0.05)
        except:
            results['has_vol_clustering'] = False
        
        # 6. Calculate composite bubble score
        score = 0
        weights = 0
        
        if not np.isnan(current_mmv):
            if current_mmv > 1.25:
                score += 1.0
            elif current_mmv > 1.15:
                score += 0.7
            elif current_mmv > 1.05:
                score += 0.3
            weights += 1
        
        if results['d_parameter'] > 0.5:
            score += 1.0
            weights += 1
        elif results['d_parameter'] > 0.3:
            score += 0.5
            weights += 1
        
        if results['kurtosis'] > 3:
            score += 0.5
            weights += 1
        
        if results['has_vol_clustering']:
            score += 0.5
            weights += 1
        
        results['bubble_score'] = score / weights if weights > 0 else 0
        
        return results

# ========================================================================
# BEHAVIORAL AGENT SIMULATOR
# ========================================================================

class BehavioralAgentSimulator:
    """Agent-Based Model integrating Fundamentalist and Speculator interactions"""
    
    def __init__(self, ticker, price_series):
        self.ticker = ticker
        self.prices = price_series
        log_rets = np.log(price_series / price_series.shift(1)).dropna()
        
        self.sigma_market = log_rets.std()
        self.mu_market = log_rets.mean()
        self.last_price = price_series.iloc[-1]
        
        # Dynamic parameter calibration
        try:
            info = yf.Ticker(ticker).info
            beta_vol = info.get('beta', 1.0) or 1.0
            sector = info.get('sector', 'Unknown')
        except:
            beta_vol = 1.0
            sector = 'Unknown'
        
        # Behavioral parameters
        self.rho = np.clip(0.8 / max(beta_vol, 0.5), 0.2, 0.9)
        
        if len(log_rets) > 10:
            acf_1 = log_rets.autocorr(lag=1)
            if np.isnan(acf_1): acf_1 = 0.0
        else:
            acf_1 = 0.0
        
        self.beta_s = np.clip(abs(acf_1) * 0.3 + 0.02, 0.02, 0.15)
        
        ann_vol = self.sigma_market * np.sqrt(252)
        self.delta = np.clip(0.99 - (ann_vol * 0.05), 0.92, 0.995)
        self.beta_f = np.clip(self.beta_s * 0.6, 0.01, 0.1)
        
        is_network_asset = (
            'Technology' in sector or 
            'Communication' in sector or 
            any(x in ticker for x in ['BTC', 'ETH', 'SOL'])
        )
        
        self.gamma = 1.5 if is_network_asset or ann_vol > 0.6 else 1.0
        self.kappa = 0.1
    
    def run(self, n_days=252, n_sims=1000):
        sim_prices = np.zeros((n_sims, n_days))
        sim_intrinsic = np.zeros((n_sims, n_days))
        
        sim_prices[:, 0] = self.last_price
        sim_intrinsic[:, 0] = self.last_price
        
        mu_s = np.full(n_sims, self.last_price)
        nu_s = np.full(n_sims, self.sigma_market**2)
        
        Z_theta = np.random.normal(self.mu_market, self.sigma_market * 0.5, (n_sims, n_days))
        Z_price = np.random.normal(0, self.sigma_market, (n_sims, n_days))
        
        for t in range(1, n_days):
            prev_price = sim_prices[:, t-1]
            prev_theta = sim_intrinsic[:, t-1]
            
            theta_growth = np.exp(Z_theta[:, t])
            curr_theta = prev_theta * theta_growth
            sim_intrinsic[:, t] = curr_theta
            
            mu_s = self.delta * mu_s + (1 - self.delta) * prev_price
            sq_dev = (prev_price - mu_s)**2
            nu_s = self.delta * nu_s + (1 - self.delta) * sq_dev
            
            demand_f = self.beta_f * (curr_theta - prev_price)
            
            risk_aversion = 1.0
            spec_signal = self.beta_s * (prev_price - mu_s)
            demand_s = spec_signal / (risk_aversion * (nu_s + 1e-6))
            demand_s = np.clip(demand_s, -0.5 * prev_price, 0.5 * prev_price)
            
            excess_demand = self.rho * demand_f + (1 - self.rho) * demand_s
            price_ret = self.kappa * (excess_demand / prev_price) + Z_price[:, t]
            
            curr_price = prev_price * np.exp(price_ret)
            sim_prices[:, t] = np.maximum(curr_price, 0.01)
        
        divergence = sim_prices / sim_intrinsic
        regimes = np.zeros_like(sim_prices)
        regimes = np.where(divergence > 1.15, 2, regimes)
        regimes = np.where(divergence < 0.85, 0, regimes)
        regimes = np.where((divergence >= 0.85) & (divergence <= 1.15), 1, regimes)
        
        return sim_prices, regimes, sim_intrinsic

# ========================================================================
# TECHNICAL INDICATORS
# ========================================================================

class TechnicalIndicators:
    @staticmethod
    def calculate_all(prices):
        """Calculate comprehensive technical indicators"""
        indicators = pd.DataFrame(index=prices.index)
        
        indicators['SMA_20'] = ta.trend.sma_indicator(prices, window=20)
        indicators['SMA_50'] = ta.trend.sma_indicator(prices, window=50)
        indicators['EMA_12'] = ta.trend.ema_indicator(prices, window=12)
        
        macd = ta.trend.MACD(prices)
        indicators['MACD'] = macd.macd()
        indicators['MACD_Signal'] = macd.macd_signal()
        
        indicators['RSI'] = ta.momentum.RSIIndicator(prices, window=14).rsi()
        
        bb = ta.volatility.BollingerBands(prices)
        indicators['BB_Upper'] = bb.bollinger_hband()
        indicators['BB_Lower'] = bb.bollinger_lband()
        
        return indicators

# ========================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================

def plot_price_history(prices, normalize=True):
    fig = go.Figure()
    
    if normalize:
        data = (prices / prices.iloc[0]) * 100
        title = "Normalized Performance (Base 100)"
        y_title = "Index"
    else:
        data = prices
        title = "Price History"
        y_title = "Price ($)"
    
    for col in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col], name=col,
            mode='lines', line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        height=500
    )
    return fig

def plot_portfolio_allocation(weights, tickers):
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        hole=0.3,
        marker=dict(colors=px.colors.qualitative.Set3)
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        template='plotly_dark',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_bubble_analysis(prices, bubble_results, ticker):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Price History', 'Bubble Score Components')
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices.values, name='Price',
                   line=dict(color='#00f2ea', width=2)),
        row=1, col=1
    )
    
    # Bubble components bar chart
    components = ['MMV Ratio', 'Long Memory', 'Kurtosis', 'Vol Clustering']
    values = [
        min(bubble_results.get('mmv_ratio', 1) / 1.25, 1),
        min(bubble_results.get('d_parameter', 0) / 0.5, 1),
        min(bubble_results.get('kurtosis', 0) / 10, 1),
        1 if bubble_results.get('has_vol_clustering', False) else 0
    ]
    
    colors = ['red' if v > 0.7 else 'orange' if v > 0.3 else 'green' for v in values]
    
    fig.add_trace(
        go.Bar(x=components, y=values, marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=700,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ========================================================================
# EXCEL EXPORT FUNCTION
# ========================================================================

def generate_comprehensive_excel(data_dict):
    """Generate comprehensive Excel report"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Price History
        if 'prices' in data_dict:
            data_dict['prices'].to_excel(writer, sheet_name='Price History')
        
        # Metrics
        if 'metrics' in data_dict:
            data_dict['metrics'].to_excel(writer, sheet_name='Performance Metrics')
        
        # Valuation
        if 'valuation' in data_dict:
            data_dict['valuation'].to_excel(writer, sheet_name='Valuation Analysis')
        
        # Portfolio
        if 'portfolio' in data_dict:
            # Flatten portfolio results for export
            port_data = []
            for strategy, weights in data_dict['portfolio'].items():
                for i, ticker in enumerate(data_dict['tickers']):
                    port_data.append({
                        'Strategy': strategy,
                        'Ticker': ticker,
                        'Weight': weights[i]
                    })
            pd.DataFrame(port_data).to_excel(writer, sheet_name='Portfolio Optimization', index=False)
        
        # Bubble Analysis
        if 'bubble_scores' in data_dict:
            pd.DataFrame(list(data_dict['bubble_scores'].items()), columns=['Ticker', 'Score']).to_excel(writer, sheet_name='Bubble Detection', index=False)
        
        # Technical Indicators
        if 'technical' in data_dict:
            for ticker, indicators in data_dict['technical'].items():
                sheet_name = f'Tech_{ticker[:25]}' # Excel sheet name limit
                indicators.to_excel(writer, sheet_name=sheet_name)
    
    return output.getvalue()

# ========================================================================
# MAIN APPLICATION
# ========================================================================

def main():
    inject_custom_css()
    
    # 1. Create a placeholder at the VERY TOP of the app
    header_placeholder = st.empty()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Quick Presets
        st.markdown("### Quick Presets")

        choice = st.radio(
            "Quick Presets",
            ["Tech", "Crypto", "ETFs"],
            horizontal=True
        )

        if choice == "Tech":
            st.session_state.preset = "NVDA TSLA AAPL MSFT GOOGL"
        elif choice == "Crypto":
            st.session_state.preset = "BTC-USD ETH-USD SOL-USD"
        elif choice == "ETFs":
            st.session_state.preset = "SPY QQQ GLD TLT" 

        st.markdown("</div>", unsafe_allow_html=True)
        
        default_tickers = st.session_state.get('preset', "NVDA TSLA AAPL")
        tickers_input = st.text_area("Tickers", default_tickers, height=80)
        
        # Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", datetime.now()-timedelta(days=365))
        with col2:
            end_date = st.date_input("End", datetime.now())
        
        # Risk-Free Rate
        rf_rate = get_risk_free_rate()
        st.metric("Risk-Free Rate", f"{rf_rate:.2%}")
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            n_sims = st.slider("Monte Carlo Simulations", 100, 2000, 500)
            n_days = st.slider("Forecast Days", 30, 365, 90)
            bubble_aware = st.checkbox("Bubble-Aware Portfolio", value=True)
            penalty_factor = st.slider("Bubble Penalty", 0.0, 1.0, 0.5)
            
            st.divider()
            enable_autorefresh = st.toggle("Enable Auto-Refresh", value=False)
            refresh_rate = st.number_input("Refresh Rate (seconds)", min_value=10, value=60)
        
        analyze_btn = st.button("🚀 Run Analysis", type="primary", width='stretch')
    
    # Main Analysis Logic
    should_run = analyze_btn or (st.session_state.analysis_complete and enable_autorefresh)

    if should_run:
        with st.spinner("Running comprehensive analysis..."):
            try:
                # Parse tickers
                tickers = [t.strip().upper() for t in tickers_input.split()]
                
                # Fetch data
                prices = fetch_market_data(tickers, start_date, end_date)
                
                # --- UPDATE TIMESTAMP HERE ---
                st.session_state.last_updated = pd.Timestamp.now('US/Eastern').strftime("%Y-%m-%d %I:%M:%S %p")
                
                if prices.empty:
                    st.error("No data found")
                    return
                
                # Calculate returns
                returns = prices.pct_change().dropna()
                
                # Performance Metrics
                metrics = {}
                for ticker in tickers:
                    r = returns[ticker]
                    metrics[ticker] = {
                        'Annual Return': r.mean() * 252,
                        'Volatility': r.std() * np.sqrt(252),
                        'Sharpe': ((r.mean() * 252 - rf_rate) / (r.std() * np.sqrt(252))),
                        'Max Drawdown': ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
                    }
                metrics_df = pd.DataFrame(metrics).T
                
                # Enhanced Valuation
                valuation_results = {}
                bubble_detector = BubbleDetector()
                bubble_scores = {}
                
                for ticker in tickers:
                    # Valuation metrics
                    dcf = EnhancedValuationMetrics.calculate_dcf_value(ticker, rf_rate)
                    wacc = EnhancedValuationMetrics.calculate_wacc(ticker, rf_rate)
                    capm = EnhancedValuationMetrics.calculate_capm_return(ticker, rf_rate)
                    ff = EnhancedValuationMetrics.calculate_fama_french_return(ticker, prices[[ticker]], rf_rate)
                    apt = EnhancedValuationMetrics.calculate_apt_return(ticker, prices[[ticker]], rf_rate)
                    
                    # Bubble detection
                    bubble_res = bubble_detector.detect_bubbles(prices[ticker], returns[ticker])
                    bubble_scores[ticker] = bubble_res['bubble_score']
                    
                    impact = EnhancedValuationMetrics.calculate_bubble_burst_impact(
                        ticker, prices[ticker], bubble_res['bubble_score']
                    )
                    
                    valuation_results[ticker] = {
                        'DCF Enterprise Value': dcf,
                        'WACC': wacc,
                        'CAPM Return': capm,
                        'Fama-French Return': ff,
                        'APT Return': apt,
                        'Bubble Score': bubble_res['bubble_score'],
                        'Bubble Burst Impact': impact
                    }
                
                valuation_df = pd.DataFrame(valuation_results).T
                
                # Portfolio Optimization
                optimizer = EnhancedPortfolioOptimizer(prices, bubble_scores, rf_rate)
                
                portfolio_results = {
                    'Min Variance': optimizer.minimum_variance(bubble_aware, penalty_factor),
                    'Risk Parity': optimizer.risk_parity(bubble_aware, penalty_factor)
                }
                
                portfolio_metrics = {}
                for strategy, weights in portfolio_results.items():
                    portfolio_metrics[strategy] = optimizer.calculate_portfolio_metrics(weights)
                
                # Technical Indicators
                technical_indicators = {}
                for ticker in tickers:
                    technical_indicators[ticker] = TechnicalIndicators.calculate_all(prices[ticker])
                
                # Monte Carlo Simulation
                sim_ticker = tickers[0]
                sim_engine = BehavioralAgentSimulator(sim_ticker, prices[sim_ticker])
                sim_prices, sim_regimes, sim_intrinsic = sim_engine.run(n_days, n_sims)
                
                # Store results
                st.session_state.data = {
                    'prices': prices,
                    'returns': returns,
                    'metrics': metrics_df,
                    'valuation': valuation_df,
                    'portfolio': portfolio_results,
                    'portfolio_metrics': portfolio_metrics,
                    'bubble_scores': bubble_scores,
                    'technical': technical_indicators,
                    'simulation': (sim_ticker, sim_prices, sim_regimes, sim_intrinsic),
                    'tickers': tickers,
                    'rf_rate': rf_rate
                }
                st.session_state.analysis_complete = True
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())

    # 2. Render the Header NOW (using the latest timestamp) inside the placeholder
    last_run = st.session_state.last_updated
    with header_placeholder.container():
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom: 20px;">
            <div>
                <h1 class="gradient-text" style="margin:0; font-size: 3rem;">🎓 QuantLab</h1>
                <p style="margin:0; color: #8b949e;">Bubble Detection with Advanced Analytics</p>
            </div>
            <div style="text-align: right;">
                <div>
                    <span class="live-indicator"></span>
                    <span style="color: #00f2ea; margin-left: 10px; font-weight: bold;">LIVE DATA</span>
                </div>
                <div style="color: #8b949e; font-size: 0.8rem; margin-top: 5px;">
                    Updated: {last_run}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display Results
    if st.session_state.analysis_complete:
        data = st.session_state.data
        
        # Quick Stats
        st.markdown("### 📊 Market Overview")
        cols = st.columns(min(len(data['tickers']), 5))
        for i, ticker in enumerate(data['tickers'][:5]):
            with cols[i]:
                last_price = data['prices'][ticker].iloc[-1]
                change = (last_price / data['prices'][ticker].iloc[-2] - 1) * 100
                bubble_score = data['bubble_scores'][ticker]
                
                color = "🔴" if bubble_score > 0.7 else "🟡" if bubble_score > 0.4 else "🟢"
                
                st.metric(
                    ticker,
                    f"${last_price:.2f}",
                    f"{change:+.2f}%"
                )
                st.caption(f"Bubble: {color} {bubble_score:.0%}")
        
        # Tabs
        tabs = st.tabs([
            "📈 Market Dashboard",
            "💰 Enhanced Valuation",
            "💼 Portfolio Optimization",
            "🎓 Bubble Detection",
            "🔮 Monte Carlo Simulation",
            "📊 Technical Analysis",
            "💾 Export Data"
        ])
        
        with tabs[0]:  # Market Dashboard
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(plot_price_history(data['prices']), width='stretch')
            with col2:
                st.markdown("#### Performance Metrics")
                st.dataframe(
                    data['metrics'].style.format({
                        'Annual Return': '{:.2%}',
                        'Volatility': '{:.2%}',
                        'Sharpe': '{:.2f}',
                        'Max Drawdown': '{:.2%}'
                    }),
                    width='stretch'
                )
        
        with tabs[1]:  # Enhanced Valuation
            st.markdown("### Enhanced Valuation Metrics")
            
            # Format the dataframe for display
            display_df = data['valuation'].copy()
            
            # Format columns appropriately
            format_dict = {}
            for col in display_df.columns:
                if 'Value' in col:
                    format_dict[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                elif 'Return' in col or 'WACC' in col:
                    format_dict[col] = '{:.2%}'
                elif 'Score' in col:
                    format_dict[col] = '{:.2%}'
                elif 'Impact' in col:
                    format_dict[col] = '{:.2%}'
            
            st.dataframe(
                display_df.style.format(format_dict),
                width='stretch'
            )
            
            # Valuation insights
            st.markdown("#### Key Insights")
            
            for ticker in data['tickers']:
                val = data['valuation'].loc[ticker]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{ticker} CAPM Return", f"{val['CAPM Return']:.2%}")
                with col2:
                    st.metric(f"{ticker} Bubble Score", f"{val['Bubble Score']:.0%}")
                with col3:
                    st.metric(f"{ticker} Risk Impact", f"{val['Bubble Burst Impact']:.0%}")
            
            # Download button for valuation data
            excel_valuation = io.BytesIO()
            data['valuation'].to_excel(excel_valuation)
            st.download_button(
                "📥 Download Valuation Data",
                excel_valuation.getvalue(),
                f"valuation_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with tabs[2]:  # Portfolio Optimization
            render_portfolio_optimization_tab(data)
        
        with tabs[3]:  # Bubble Detection
            st.markdown("### Bubble Detection")
            
            selected_ticker = st.selectbox("Select Asset", data['tickers'])
            
            bubble_res = BubbleDetector().detect_bubbles(
                data['prices'][selected_ticker],
                data['returns'][selected_ticker]
            )
            
            # Display bubble metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Bubble Score", f"{bubble_res['bubble_score']:.0%}")
            with col2:
                st.metric("MMV Ratio", f"{bubble_res['mmv_ratio']:.2f}")
            with col3:
                st.metric("d Parameter", f"{bubble_res['d_parameter']:.3f}")
            with col4:
                st.metric("Kurtosis", f"{bubble_res['kurtosis']:.2f}")
            
            # Bubble regime
            regime = bubble_res['mmv_regime']
            regime_colors = {
                'Extreme Bubble': '🔴',
                'Bubble Formation': '🟠',
                'Fair Value': '🟢',
                'Undervalued': '🔵',
                'Transition': '⚪',
                'Unknown': '⚫'
            }
            
            st.markdown(f"### Current Regime: {regime_colors.get(regime, '⚫')} {regime}")
            
            # Bubble analysis chart
            st.plotly_chart(
                plot_bubble_analysis(
                    data['prices'][selected_ticker],
                    bubble_res,
                    selected_ticker
                ),
                width='stretch'
            )
        
        with tabs[4]:  # Monte Carlo Simulation
            st.markdown("### Monte Carlo Simulation")
            
            st.markdown("#### Select Ticker for Simulation")

            # Sidebar preset tickers (from user input)
            preset_tickers = st.session_state.get("preset", "NVDA TSLA AAPL").split()

            # Let user choose which ticker to simulate
            sim_ticker = st.selectbox(
                "Choose a ticker to run Monte Carlo simulation:",
                options=data['tickers'],
                index=0
            )

            if sim_ticker not in data['prices'].columns:
                st.error(f"❗ '{sim_ticker}' has no price data. Check ticker or data source.")
                st.stop()

            # Re-run simulation for the selected ticker if needed, 
            # or just run it here for display.
            # Using defaults for quick interactivity:
            n_days_sim = 252 
            n_sims_sim = 1000

            sim_engine = BehavioralAgentSimulator(sim_ticker, data['prices'][sim_ticker])
            sim_prices, sim_regimes, sim_intrinsic = sim_engine.run(n_days_sim, n_sims_sim)
            
            # Calculate statistics
            final_prices = sim_prices[:, -1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Median Price", f"${np.median(final_prices):.2f}")
            with col2:
                st.metric("95% VaR", f"${np.percentile(final_prices, 5):.2f}")
            with col3:
                st.metric("95% CVaR", f"${np.percentile(final_prices, 95):.2f}")
            
            # Simulation chart
            days = np.arange(sim_prices.shape[1])
            p5 = np.percentile(sim_prices, 5, axis=0)
            p50 = np.percentile(sim_prices, 50, axis=0)
            p95 = np.percentile(sim_prices, 95, axis=0)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=days, y=p95, line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=days, y=p5, fill='tonexty',
                fillcolor='rgba(0,242,234,0.2)',
                name='90% Confidence'
            ))
            fig.add_trace(go.Scatter(
                x=days, y=p50, name='Median',
                line=dict(color='#00f2ea', width=2)
            ))
            
            fig.update_layout(
                title=f"Monte Carlo Projection: {sim_ticker}",
                template='plotly_dark',
                height=500,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, width='stretch')
        
        with tabs[5]:  # Technical Analysis
            st.markdown("### Technical Analysis")
            
            tech_ticker = st.selectbox("Select Asset", data['tickers'], key='tech')
            
            if tech_ticker in data['technical']:
                tech_data = data['technical'][tech_ticker]
                
                # Latest indicators
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RSI", f"{tech_data['RSI'].iloc[-1]:.1f}")
                with col2:
                    st.metric("MACD", f"{tech_data['MACD'].iloc[-1]:.4f}")
                with col3:
                    st.metric("SMA 20", f"${tech_data['SMA_20'].iloc[-1]:.2f}")
                with col4:
                    st.metric("SMA 50", f"${tech_data['SMA_50'].iloc[-1]:.2f}")
                
                # Technical chart
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price & Moving Averages', 'MACD', 'RSI')
                )
                
                # Price and MAs
                fig.add_trace(
                    go.Scatter(x=data['prices'].index, y=data['prices'][tech_ticker],
                               name='Price', line=dict(color='#00f2ea')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['SMA_20'],
                               name='SMA 20', line=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['SMA_50'],
                               name='SMA 50', line=dict(color='red')),
                    row=1, col=1
                )
                
                # MACD
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['MACD'],
                               name='MACD', line=dict(color='blue')),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['MACD_Signal'],
                               name='Signal', line=dict(color='red')),
                    row=2, col=1
                )
                
                # RSI
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['RSI'],
                               name='RSI', line=dict(color='purple')),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(
                    template='plotly_dark',
                    height=800,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, width='stretch')
        
        with tabs[6]:  # Export Data
            st.markdown("### Export Data")
            
            # Prepare comprehensive export
            export_data = {
                'prices': data['prices'],
                'metrics': data['metrics'],
                'valuation': data['valuation'],
                'portfolio': data['portfolio'],
                'bubble_scores': data['bubble_scores'],
                'technical': data['technical'],
                'tickers': data['tickers']
            }
            
            excel_file = generate_comprehensive_excel(export_data)
            
            st.download_button(
                "📥 Download Complete Analysis Report",
                excel_file,
                f"quantlab_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch'
            )
            
            st.markdown("#### Report Contents")
            st.markdown("""
            - **Price History**: Historical price data for all assets
            - **Performance Metrics**: Returns, volatility, Sharpe ratios
            - **Valuation Analysis**: DCF, WACC, CAPM, Fama-French, APT
            - **Portfolio Optimization**: Optimal weights and metrics
            - **Bubble Detection**: Bubble scores and indicators
            - **Technical Indicators**: Complete technical analysis
            """)

    # Auto-Refresh Logic
    if enable_autorefresh and st.session_state.analysis_complete:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
