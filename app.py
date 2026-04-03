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
import urllib.request
from datetime import datetime, timedelta
import xlsxwriter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from scipy.stats import norm as scipy_norm
import ta
import time
import logging
import re
from functools import wraps

# ========================================================================
# SYSTEM CONFIGURATION
# ========================================================================
matplotlib.use('Agg')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
_logger = logging.getLogger('quantlab')

# ========================================================================
# CUSTOM EXCEPTIONS & ERROR HANDLING
# ========================================================================

class QuantLabError(Exception):
    """Base exception for QuantLab errors."""
    def __init__(self, message: str, user_message: str = None, recovery_hint: str = None):
        self.message = message
        self.user_message = user_message or message
        self.recovery_hint = recovery_hint
        super().__init__(self.message)

class DataFetchError(QuantLabError):
    """Raised when market data cannot be fetched."""
    pass

class ValidationError(QuantLabError):
    """Raised when input validation fails."""
    pass

class CalculationError(QuantLabError):
    """Raised when a financial calculation fails."""
    pass

class ExportError(QuantLabError):
    """Raised when report generation fails."""
    pass

ERROR_MESSAGES = {
    'no_data': {
        'icon': '📊', 'title': 'No Data Found',
        'message': 'Could not retrieve market data for the selected tickers and date range.',
        'hints': [
            'Check that ticker symbols are spelled correctly (e.g. AAPL, BTC-USD, EURUSD=X, GC=F)',
            'Verify the date range includes trading days',
            'Some tickers may not have data for older dates',
            'Try a shorter date range or fewer tickers',
        ]
    },
    'network_error': {
        'icon': '🌐', 'title': 'Network Error',
        'message': 'Could not connect to the market data provider (Yahoo Finance).',
        'hints': [
            'Check your internet connection',
            'Yahoo Finance may be temporarily unavailable — try again in a moment',
        ]
    },
    'invalid_ticker': {
        'icon': '❌', 'title': 'Invalid Ticker',
        'message': 'One or more ticker symbols were not recognised.',
        'hints': [
            'Stocks: AAPL, MSFT, GOOGL',
            'Crypto: BTC-USD, ETH-USD',
            'Forex: EURUSD=X, GBPUSD=X',
            'Commodities: GC=F (Gold), CL=F (Oil)',
            'Separate multiple tickers with spaces or commas',
        ]
    },
    'calculation_error': {
        'icon': '⚠️', 'title': 'Calculation Error',
        'message': 'An error occurred during analysis.',
        'hints': [
            'Try a different date range',
            'Some calculations require a minimum number of data points (≥ 30 days recommended)',
            'Remove any tickers with very short or incomplete history',
        ]
    },
    'rate_limit': {
        'icon': '⏱️', 'title': 'Rate Limited',
        'message': 'Too many requests to the data provider.',
        'hints': [
            'Wait 30–60 seconds before trying again',
            'Reduce the number of tickers',
        ]
    },
    'export_error': {
        'icon': '📄', 'title': 'Export Failed',
        'message': 'Could not generate the requested report.',
        'hints': [
            'Try a different export format',
            'Reduce the number of tickers',
            'Make sure analysis has completed successfully first',
        ]
    },
    'options_unavailable': {
        'icon': '📈', 'title': 'Options Data Unavailable',
        'message': 'No options chain found for this ticker.',
        'hints': [
            'Not all tickers have listed options',
            'Try major stocks like AAPL, MSFT, TSLA, SPY',
            'Forex and most crypto do not have options chains via Yahoo Finance',
        ]
    },
    'insufficient_data': {
        'icon': '📉', 'title': 'Insufficient Data',
        'message': 'Not enough historical data to complete this calculation.',
        'hints': [
            'Extend the date range (at least 60 trading days recommended)',
            'ML models require at least 1 year of data',
            'HRP and clustering require at least 2 tickers',
        ]
    },
}


def show_error(error_key: str, details: str = None, inline: bool = False):
    """Display a user-friendly error with recovery hints.
    
    Args:
        error_key: Key into ERROR_MESSAGES dict.
        details: Optional technical detail (shown in expander if debug mode on).
        inline: If True, use st.warning instead of st.error (for non-fatal issues).
    """
    info = ERROR_MESSAGES.get(error_key, ERROR_MESSAGES['calculation_error'])
    fn = st.warning if inline else st.error
    fn(f"{info['icon']}  **{info['title']}** — {info['message']}")
    if info.get('hints'):
        st.markdown('**💡 Suggestions:**')
        for h in info['hints']:
            st.markdown(f'- {h}')
    if details and st.session_state.get('debug_mode', False):
        with st.expander('Technical Details'):
            st.code(details)


def handle_error(func):
    """Decorator: catches QuantLabError subclasses and generic exceptions,
    displays a friendly message, and returns None so callers degrade gracefully."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DataFetchError as e:
            show_error('no_data', e.message)
            return None
        except ValidationError as e:
            show_error('invalid_ticker', e.message)
            return None
        except CalculationError as e:
            show_error('calculation_error', e.message)
            return None
        except ExportError as e:
            show_error('export_error', e.message)
            return None
        except Exception as e:
            _logger.error('Unexpected error in %s: %s', func.__name__, traceback.format_exc())
            show_error('calculation_error',
                       traceback.format_exc() if st.session_state.get('debug_mode') else str(e))
            return None
    return wrapper


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
if 'last_updated' not in st.session_state:
    st.session_state.last_updated = "Initializing..."
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# ========================================================================
# ENHANCED CSS & UI STYLING
# ========================================================================
def inject_custom_css(theme='light'):
    is_dark = theme == 'dark'

    if is_dark:
        root_vars = """
            --bg-primary: #040d1a;
            --bg-secondary: #0a1628;
            --bg-tertiary: #0f1f3d;
            --bg-card: rgba(15,31,61,0.8);
            --bg-input: #0f1f3d;
            --text-primary: #f0f4ff;
            --text-secondary: #dce2f0;
            --text-muted: #b8c4dc;
            --text-faint: #8898c0;
            --teal: #22d3ee;
            --teal-label: #5ce0f0;
            --teal-dim: #0096b7;
            --teal-glow: rgba(0,180,216,0.15);
            --gold: #ffd700;
            --gold-dim: #e5c100;
            --gold-glow: rgba(255,215,0,0.10);
            --green: #00d084;
            --red: #ff4d6d;
            --orange: #ff9a00;
            --border: rgba(255,255,255,0.08);
            --border-accent: rgba(0,180,216,0.3);
            --surface: rgba(15,31,61,0.8);
            --shadow: 0 4px 24px rgba(0,0,0,0.4);
            --shadow-lg: 0 8px 32px rgba(0,0,0,0.6);
            --bg-gradient: linear-gradient(160deg, #040d1a 0%, #0a1628 50%, #040d1a 100%);
            --hover-tint: rgba(0,180,216,0.06);
            --bubble-bar-bg: #0f1f3d;
            --btn-text: #ffffff;
            --select-text: #f0f4ff;
        """
    else:
        root_vars = """
            --bg-primary: #ffffff;
            --bg-secondary: #f9fafb;
            --bg-tertiary: #f3f4f6;
            --bg-card: rgba(255,255,255,0.95);
            --bg-input: #f3f4f6;
            --text-primary: #0a0a14;
            --text-secondary: #111827;
            --text-muted: #1f2937;
            --text-faint: #374151;
            --teal: #0090b5;
            --teal-label: #006f8f;
            --teal-dim: #006f8f;
            --teal-glow: rgba(0,144,181,0.12);
            --gold: #b8860b;
            --gold-dim: #a07008;
            --gold-glow: rgba(184,134,11,0.08);
            --green: #059669;
            --red: #dc2626;
            --orange: #d97706;
            --border: rgba(0,0,0,0.12);
            --border-accent: rgba(0,144,181,0.3);
            --surface: rgba(255,255,255,0.95);
            --shadow: 0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06);
            --shadow-lg: 0 4px 12px rgba(0,0,0,0.1);
            --bg-gradient: #ffffff;
            --hover-tint: rgba(0,144,181,0.04);
            --bubble-bar-bg: #e5e7eb;
            --btn-text: #ffffff;
            --select-text: #0a0a14;
        """

    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        /* ===== 1. ROOT VARIABLES ===== */
        :root {{
            {root_vars}
            --radius: 8px;
            --radius-lg: 12px;
            --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Consolas', monospace;
        }}

        /* ===== 2. GLOBAL BODY / APP ===== */
        html, body {{
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            font-family: var(--font-body) !important;
        }}
        .stApp {{
            background: var(--bg-gradient) !important;
            font-family: var(--font-body) !important;
            color: var(--text-primary) !important;
        }}
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"],
        .main .block-container {{
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }}
        .stApp p, .stApp span, .stApp label, .stApp div,
        .stApp li, .stApp td, .stApp th {{
            color: inherit;
        }}

        /* ===== 3. STREAMLIT HEADER BAR ===== */
        header {{
            background-color: var(--bg-secondary) !important;
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border) !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            {'color-scheme: dark !important;' if is_dark else 'color-scheme: light !important;'}
        }}
        header[data-testid="stHeader"] {{
            background-color: var(--bg-secondary) !important;
            background: var(--bg-secondary) !important;
        }}
        header * {{
            color: var(--text-muted) !important;
        }}
        header::before,
        header::after,
        header[data-testid="stHeader"]::before,
        header[data-testid="stHeader"]::after {{
            background: var(--bg-secondary) !important;
            background-color: var(--bg-secondary) !important;
            display: none !important;
        }}
        [data-testid="stToolbar"] {{
            background: transparent !important;
        }}
        [data-testid="stToolbar"] button {{
            color: var(--text-muted) !important;
        }}
        [data-testid="stToolbar"] button svg,
        header button svg,
        header a svg {{
            fill: var(--text-muted) !important;
            stroke: var(--text-muted) !important;
        }}
        [data-testid="stAppDeployButton"],
        [data-testid="stAppDeployButton"] *,
        [data-testid="stStatusWidget"],
        .stDeployButton,
        .stDeployButton * {{
            color: var(--text-muted) !important;
            background: transparent !important;
        }}
        [data-testid="stDecoration"] {{
            background-image: none !important;
            background: var(--bg-secondary) !important;
        }}

        /* ===== 4. SIDEBAR ===== */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebar"] section,
        [data-testid="stSidebarContent"],
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-right: 1px solid var(--border) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h2 {{
            font-family: var(--font-body) !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdown"] h3 {{
            font-family: var(--font-mono) !important;
            font-size: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: var(--teal-label) !important;
            margin-top: 16px !important;
            margin-bottom: 4px !important;
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stTextArea label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {{
            color: var(--text-muted) !important;
            font-size: 13px !important;
        }}
        /* Sidebar metric card — dark-mode aware */
        [data-testid="stSidebar"] [data-testid="stMetric"] {{
            background: var(--teal-glow) !important;
            border: 1px solid var(--border-accent) !important;
            border-radius: var(--radius) !important;
            padding: 12px 14px !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {{
            font-family: var(--font-mono) !important;
            color: var(--teal) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMetricValue"] div {{
            color: var(--teal) !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] p,
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] div {{
            color: var(--text-muted) !important;
        }}

        /* ===== 5. METRIC CARDS ===== */
        div[data-testid="stMetric"] {{
            background: var(--bg-card) !important;
            padding: 18px 20px !important;
            border-radius: var(--radius-lg) !important;
            border: 1px solid var(--border) !important;
            box-shadow: var(--shadow) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-3px) !important;
            box-shadow: var(--shadow-lg) !important;
            border-color: var(--border-accent) !important;
        }}
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] div,
        [data-testid="stMetricLabel"] label {{
            font-family: var(--font-mono) !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.08em !important;
            text-transform: uppercase !important;
            color: var(--text-muted) !important;
        }}
        [data-testid="stMetricValue"] div {{
            font-family: var(--font-mono) !important;
            font-size: 24px !important;
            font-weight: 800 !important;
            color: var(--text-primary) !important;
        }}
        [data-testid="stMetricDelta"] div,
        [data-testid="stMetricDelta"] span {{
            font-family: var(--font-mono) !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }}

        /* ===== 6. TABS ===== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px !important;
            background: transparent !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: var(--bg-tertiary) !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            color: var(--text-muted) !important;
            border: 1px solid var(--border) !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: var(--teal-glow) !important;
            color: var(--text-primary) !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: var(--teal) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            border-color: var(--teal) !important;
        }}
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {{
            display: none !important;
        }}

        /* ===== 7. SELECTBOX / INPUTS / TEXTAREA ===== */
        [data-baseweb="select"],
        [data-baseweb="select"] > div,
        [data-baseweb="select"] [data-baseweb="tag"],
        [data-baseweb="input"],
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {{
            background-color: var(--bg-input) !important;
            color: var(--select-text) !important;
            border-color: var(--border) !important;
        }}
        [data-baseweb="select"] [data-baseweb="icon"] svg {{
            fill: var(--text-muted) !important;
        }}
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"],
        [data-baseweb="menu"] {{
            background-color: var(--bg-secondary) !important;
        }}
        [data-baseweb="menu"] li,
        [data-baseweb="popover"] li {{
            color: var(--text-primary) !important;
        }}
        [data-baseweb="menu"] li:hover,
        [data-baseweb="popover"] li:hover {{
            background-color: var(--teal-glow) !important;
        }}
        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input {{
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            color: var(--text-primary) !important;
            font-family: var(--font-mono) !important;
        }}

        /* ===== 8. DATAFRAMES / TABLES (glide-data-grid) ===== */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrameResizable"],
        .stDataFrame {{
            border-radius: var(--radius-lg) !important;
            overflow: hidden !important;
            border: 1px solid var(--border) !important;
            background-color: var(--bg-card) !important;
        }}
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrame"] > div > div,
        [data-testid="stDataFrame"] > div > div > div {{
            background-color: var(--bg-card) !important;
        }}
        [data-testid="stDataFrame"] .dvn-scroller {{
            background-color: var(--bg-card) !important;
        }}
        [data-testid="stDataFrame"] canvas {{
            background-color: transparent !important;
        }}
        [data-testid="stDataFrame"] [data-testid="glide-data-grid-canvas"] {{
            background-color: transparent !important;
        }}
        /* Glide-data-grid uses CSS custom properties for theming */
        [data-testid="stDataFrame"] {{
            --gdg-bg-cell: {'#0a1628' if is_dark else '#ffffff'} !important;
            --gdg-bg-header: {'#0f1f3d' if is_dark else '#f3f4f6'} !important;
            --gdg-bg-header-has-focus: {'#0f1f3d' if is_dark else '#f3f4f6'} !important;
            --gdg-text-dark: {'#ffffff' if is_dark else '#000000'} !important;
            --gdg-text-medium: {'#e2e8f0' if is_dark else '#1a202c'} !important;
            --gdg-text-light: {'#cbd5e0' if is_dark else '#2d3748'} !important;
            --gdg-text-header: {'#e2e8f0' if is_dark else '#1a202c'} !important;
            --gdg-border-color: {'rgba(255,255,255,0.08)' if is_dark else 'rgba(0,0,0,0.12)'} !important;
            --gdg-bg-cell-medium: {'#0f1f3d' if is_dark else '#f9fafb'} !important;
            --gdg-accent-color: var(--teal) !important;
            --gdg-accent-light: var(--teal-glow) !important;
        }}
        [data-testid="stDataFrame"] table {{
            font-family: var(--font-mono) !important;
            font-size: 12px !important;
            background-color: var(--bg-card) !important;
        }}
        [data-testid="stDataFrame"] thead th {{
            background: var(--bg-tertiary) !important;
            color: {'#e2e8f0' if is_dark else '#000000'} !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
        }}
        [data-testid="stDataFrame"] tbody td {{
            color: {'#ffffff' if is_dark else '#000000'} !important;
            font-family: var(--font-mono) !important;
            background-color: var(--bg-card) !important;
        }}
        [data-testid="stDataFrame"] tbody tr:hover {{
            background: var(--hover-tint) !important;
        }}

        /* ===== 9. PLOTLY CHART CONTAINERS ===== */
        [data-testid="stPlotlyChart"] {{
            border-radius: var(--radius-lg) !important;
            overflow: hidden !important;
            background-color: var(--bg-primary) !important;
        }}
        [data-testid="stPlotlyChart"] > div {{
            background-color: var(--bg-primary) !important;
        }}
        .js-plotly-plot,
        .plot-container,
        .svg-container {{
            background-color: var(--bg-primary) !important;
        }}
        .js-plotly-plot .modebar {{
            background: transparent !important;
        }}
        .js-plotly-plot .modebar-btn path {{
            fill: var(--text-muted) !important;
        }}
        .js-plotly-plot .modebar-btn:hover path {{
            fill: var(--teal) !important;
        }}

        /* ===== 10. BUTTONS ===== */
        div.stButton > button,
        [data-testid="stBaseButton-primary"] {{
            background: var(--teal) !important;
            color: var(--btn-text) !important;
            font-family: var(--font-body) !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            border: 1px solid var(--teal) !important;
            padding: 10px 20px !important;
            border-radius: 6px !important;
            transition: all 0.18s ease !important;
        }}
        div.stButton > button:hover {{
            background: var(--teal-dim) !important;
            border-color: var(--teal-dim) !important;
        }}
        div.stDownloadButton > button {{
            background: var(--gold) !important;
            color: #0a0a0a !important;
            font-weight: 700 !important;
            border: 1px solid var(--gold) !important;
            border-radius: 6px !important;
            font-size: 12px !important;
            padding: 8px 16px !important;
        }}
        div.stDownloadButton > button:hover {{
            background: var(--gold-dim) !important;
        }}

        /* ===== 11. EXPANDER ===== */
        details[data-testid="stExpander"] summary,
        .streamlit-expanderHeader {{
            background: var(--bg-tertiary) !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
            font-weight: 600 !important;
        }}
        details[data-testid="stExpander"] > div {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-top: none !important;
        }}

        /* ===== 12. PROGRESS BARS ===== */
        [data-testid="stProgress"] > div > div {{
            background-color: var(--teal) !important;
        }}
        [data-testid="stProgress"] > div {{
            background-color: var(--bg-tertiary) !important;
        }}

        /* ===== 13. CAPTION ===== */
        [data-testid="stCaptionContainer"] p {{
            font-family: var(--font-mono) !important;
            font-size: 11px !important;
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
        }}

        /* ===== 14. SCROLLBAR ===== */
        ::-webkit-scrollbar {{ width: 6px !important; height: 6px !important; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-secondary) !important; }}
        ::-webkit-scrollbar-thumb {{ background: var(--text-faint) !important; border-radius: 3px !important; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted) !important; }}

        /* ===== 15. MARKDOWN HEADINGS ===== */
        [data-testid="stMarkdown"] h1 {{
            color: var(--text-primary) !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMarkdown"] h2 {{
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMarkdown"] h3 {{
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            font-size: 18px !important;
        }}
        [data-testid="stMarkdown"] h4 {{
            color: var(--text-secondary) !important;
            font-weight: 600 !important;
            font-size: 15px !important;
        }}
        [data-testid="stMarkdown"] p {{
            color: var(--text-secondary) !important;
        }}

        /* ===== 16. RADIO / CHECKBOX / TOGGLE LABELS ===== */
        .stRadio > div > label,
        .stCheckbox > label,
        [data-testid="stWidgetLabel"] p {{
            color: var(--text-secondary) !important;
            font-weight: 500 !important;
        }}

        /* ===== 17. ALERTS ===== */
        [data-testid="stAlert"] {{
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
        }}
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] div {{
            color: var(--text-primary) !important;
        }}

        /* ===== 18. DIVIDERS ===== */
        hr {{
            border-color: var(--border) !important;
        }}

        /* ===== 19. CUSTOM CLASSES USED IN THE APP ===== */

        /* Section headers */
        .section-header {{ margin-bottom: 20px !important; }}
        .section-label {{
            font-family: var(--font-mono) !important;
            font-size: 10px !important;
            font-weight: 700 !important;
            letter-spacing: 0.15em !important;
            text-transform: uppercase !important;
            color: var(--teal-label) !important;
            margin-bottom: 4px !important;
        }}
        .section-title {{
            font-size: 24px !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.01em !important;
        }}
        .section-subtitle {{
            font-size: 13px !important;
            color: var(--text-muted) !important;
            margin-top: 4px !important;
        }}

        /* App header */
        .ql-header {{
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            padding: 16px 0 24px 0 !important;
            border-bottom: 1px solid var(--border) !important;
            margin-bottom: 20px !important;
        }}
        .ql-header-left {{
            display: flex !important;
            align-items: center !important;
            gap: 14px !important;
        }}
        .ql-logo-mark {{ width: 40px !important; height: 40px !important; }}
        .ql-title {{
            font-size: 28px !important;
            font-weight: 800 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.02em !important;
            line-height: 1.1 !important;
        }}
        .ql-title span {{ color: var(--teal) !important; }}
        .ql-subtitle {{
            font-size: 12px !important;
            color: var(--text-muted) !important;
            margin-top: 2px !important;
            font-family: var(--font-mono) !important;
            letter-spacing: 0.04em !important;
        }}
        .ql-header-right {{
            display: flex !important;
            align-items: center !important;
            gap: 16px !important;
        }}
        .ql-live-badge {{
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            background: rgba(0,208,132,0.10) !important;
            border: 1px solid rgba(0,208,132,0.30) !important;
            border-radius: 4px !important;
            padding: 4px 10px !important;
            font-family: var(--font-mono) !important;
            font-size: 10px !important;
            font-weight: 700 !important;
            color: var(--green) !important;
            letter-spacing: 0.1em !important;
        }}
        .ql-timestamp {{
            font-family: var(--font-mono) !important;
            font-size: 11px !important;
            color: var(--text-faint) !important;
        }}

        /* Pulse animation for live dot */
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(0.8); }}
        }}
        .live-dot {{
            display: inline-block !important;
            width: 8px !important;
            height: 8px !important;
            background: var(--green) !important;
            border-radius: 50% !important;
            animation: pulse 2s infinite !important;
            margin-right: 6px !important;
        }}

        /* Badges */
        .badge {{
            display: inline-flex !important;
            align-items: center !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            padding: 3px 9px !important;
            border-radius: 4px !important;
            font-family: var(--font-mono) !important;
        }}
        .badge-buy, .badge-safe {{
            background: rgba(0,208,132,0.12) !important;
            color: var(--green) !important;
            border: 1px solid rgba(0,208,132,0.30) !important;
        }}
        .badge-neutral, .badge-caution {{
            background: rgba(255,154,0,0.12) !important;
            color: var(--orange) !important;
            border: 1px solid rgba(255,154,0,0.30) !important;
        }}
        .badge-sell, .badge-danger {{
            background: rgba(255,77,109,0.12) !important;
            color: var(--red) !important;
            border: 1px solid rgba(255,77,109,0.30) !important;
        }}
        .badge-info {{
            background: var(--teal-glow) !important;
            color: var(--teal) !important;
            border: 1px solid var(--border-accent) !important;
        }}

        /* Spinner */
        .stSpinner > div {{ border-top-color: var(--teal) !important; }}
    </style>
    """, unsafe_allow_html=True)

# ========================================================================
# STYLED TABLE RENDERER (bypasses glide-data-grid canvas issues)
# ========================================================================

def render_styled_table(df, format_dict=None, highlight_col=None, theme=None):
    """Render a DataFrame as a styled HTML table with proper text colors.
    
    This bypasses Streamlit's glide-data-grid which renders on canvas
    and ignores CSS custom properties for text color in light mode.
    
    Args:
        df: pandas DataFrame to render
        format_dict: dict of column -> format string (e.g., '{:.2%}')
        highlight_col: column name to apply background gradient
        theme: 'light' or 'dark' (auto-detected from session state if None)
    """
    if theme is None:
        theme = st.session_state.get('theme', 'light')
    
    is_dark = theme == 'dark'
    
    # Theme colors
    bg_header = '#0f1f3d' if is_dark else '#f3f4f6'
    bg_cell = '#0a1628' if is_dark else '#ffffff'
    bg_cell_alt = '#0c1e38' if is_dark else '#f9fafb'
    text_color = '#f0f4ff' if is_dark else '#000000'
    text_header = '#e2e8f0' if is_dark else '#000000'
    border_color = 'rgba(255,255,255,0.08)' if is_dark else 'rgba(0,0,0,0.12)'
    hover_bg = 'rgba(0,180,216,0.06)' if is_dark else 'rgba(0,144,181,0.04)'
    
    # Apply formatting
    formatted_df = df.copy()
    if format_dict:
        for col, fmt in format_dict.items():
            if col in formatted_df.columns:
                if callable(fmt):
                    formatted_df[col] = formatted_df[col].apply(lambda x: fmt(x) if pd.notna(x) else 'N/A')
                else:
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: fmt.format(x) if pd.notna(x) and isinstance(x, (int, float)) else (str(x) if pd.notna(x) else 'N/A')
                    )
    
    # Compute highlight gradient if requested
    highlight_styles = {}
    if highlight_col and highlight_col in df.columns:
        col_vals = df[highlight_col].astype(float)
        vmin, vmax = col_vals.min(), col_vals.max()
        for idx in df.index:
            val = col_vals[idx]
            if vmax > vmin:
                normed = (val - vmin) / (vmax - vmin)
            else:
                normed = 0.5
            # RdYlGn-like: red(low) -> yellow(mid) -> green(high)
            if normed < 0.5:
                r = int(220 + (255 - 220) * (1 - normed * 2))
                g = int(50 + (220 - 50) * normed * 2)
                b = 50
            else:
                r = int(255 - (255 - 50) * (normed - 0.5) * 2)
                g = int(220 + (180 - 220) * (normed - 0.5) * 2)
                b = int(50 + (80 - 50) * (normed - 0.5) * 2)
            highlight_styles[idx] = f'background-color: rgba({r},{g},{b},0.35);'
    
    # Build HTML
    html = f'''
    <div style="overflow-x:auto; border-radius:8px; border:1px solid {border_color};">
    <table style="width:100%; border-collapse:collapse; font-family:'JetBrains Mono','Fira Code',monospace; font-size:13px;">
    <thead>
    <tr style="background:{bg_header};">
    '''
    for col in formatted_df.columns:
        html += f'<th style="padding:10px 14px; color:{text_header}; font-size:11px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; text-align:left; border-bottom:2px solid {border_color};">{col}</th>'
    html += '</tr></thead><tbody>'
    
    for i, (idx, row) in enumerate(formatted_df.iterrows()):
        row_bg = bg_cell_alt if i % 2 == 1 else bg_cell
        html += f'<tr style="background:{row_bg};" onmouseover="this.style.background=\'{hover_bg}\'" onmouseout="this.style.background=\'{row_bg}\'">'  
        for col in formatted_df.columns:
            cell_style = f'padding:9px 14px; color:{text_color}; border-bottom:1px solid {border_color}; text-align:left;'
            if highlight_col and col == highlight_col and idx in highlight_styles:
                cell_style += highlight_styles[idx]
            html += f'<td style="{cell_style}">{row[col]}</td>'
        html += '</tr>'
    
    html += '</tbody></table></div>'
    
    st.markdown(html, unsafe_allow_html=True)

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

@st.cache_data(ttl=3600)
def get_benchmark_data(start_date, end_date, ticker="^GSPC"):
    """Fetches benchmark data for beta calculations"""
    try:
        benchmark = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # Handle multi-index columns if they exist
        if isinstance(benchmark.columns, pd.MultiIndex):
            benchmark = benchmark["Close"]
        elif "Close" in benchmark.columns:
            benchmark = benchmark["Close"]
        else:
            benchmark = benchmark.iloc[:, 0]
        
        # Flatten if it's a dataframe with one column
        if isinstance(benchmark, pd.DataFrame):
            benchmark = benchmark.iloc[:, 0]
            
        return benchmark
    except:
        return pd.Series()

@st.cache_data(ttl=60)
def fetch_market_data(tickers, start_date, end_date):
    """Robust data fetching using yfinance. Returns (prices, volumes).
    
    Raises:
        DataFetchError: when data cannot be retrieved or is empty.
    """
    if not tickers:
        raise DataFetchError(
            'No tickers provided',
            user_message='Please enter at least one ticker symbol.',
        )

    try:
        data = yf.download(
            tickers, start=start_date, end=end_date,
            group_by='ticker', auto_adjust=True, progress=False,
        )
    except Exception as e:
        err_str = str(e).lower()
        if 'connection' in err_str or 'network' in err_str or 'timeout' in err_str:
            raise DataFetchError(
                f'Network error: {e}',
                user_message='Could not connect to Yahoo Finance.',
                recovery_hint='Check your internet connection and try again.',
            )
        if 'rate' in err_str or '429' in err_str:
            raise DataFetchError(
                f'Rate limited: {e}',
                user_message='Too many requests to Yahoo Finance.',
                recovery_hint='Wait 30\u201360 seconds before retrying.',
            )
        raise DataFetchError(f'Unexpected fetch error: {e}')

    if data is None or (hasattr(data, 'empty') and data.empty):
        raise DataFetchError(
            f'No data returned for {tickers}',
            user_message=f'No market data found for: {", ".join(tickers)}',
            recovery_hint='Check ticker symbols and date range.',
        )

    prices = pd.DataFrame()
    volumes = pd.DataFrame()

    if len(tickers) == 1:
        ticker = tickers[0]
        cols = data.columns
        if isinstance(cols, pd.MultiIndex):
            if 'Close' in cols:
                prices[ticker] = data['Close']
            else:
                prices[ticker] = data[ticker]['Close']
            if 'Volume' in cols:
                volumes[ticker] = data['Volume']
            else:
                volumes[ticker] = data[ticker].get('Volume', pd.Series(dtype=float))
        elif 'Close' in cols:
            prices[ticker] = data['Close']
            if 'Volume' in cols:
                volumes[ticker] = data['Volume']
        elif ticker in cols:
            prices[ticker] = data[ticker]
    else:
        for t in tickers:
            try:
                if hasattr(data.columns, 'levels') and t in data.columns.levels[0]:
                    prices[t] = data[t]['Close']
                    if 'Volume' in data[t].columns:
                        volumes[t] = data[t]['Volume']
                elif (t, 'Close') in data.columns:
                    prices[t] = data[(t, 'Close')]
                    if (t, 'Volume') in data.columns:
                        volumes[t] = data[(t, 'Volume')]
                elif t in data.columns:
                    prices[t] = data[t]
            except Exception:
                _logger.warning('Could not extract data for ticker %s', t)

    prices.dropna(how='all', inplace=True)
    volumes = volumes.reindex(prices.index).fillna(0)

    if prices.empty:
        raise DataFetchError(
            'Price extraction yielded empty DataFrame',
            user_message=(
                f'No price data could be extracted for: {", ".join(tickers)}. '
                'Ticker symbols may be incorrect or delisted.'
            ),
        )

    # Warn about any tickers that came back empty
    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        _logger.warning('No data for tickers: %s', missing)

    return prices, volumes


# ========================================================================
# MACRO DATA FETCHING (via yfinance — works on Streamlit Cloud)
# ========================================================================

# Macro ticker mapping: yfinance symbols for key economic indicators
_MACRO_TICKERS = {
    'TNX': '^TNX',    # 10-Year Treasury Yield
    'FVX': '^FVX',    # 5-Year Treasury Yield
    'TYX': '^TYX',    # 30-Year Treasury Yield
    'IRX': '^IRX',    # 13-Week T-Bill (proxy for Fed Funds)
    'VIX': '^VIX',    # CBOE Volatility Index
    'GSPC': '^GSPC',  # S&P 500
}


@st.cache_data(ttl=3600)
def fetch_all_macro_data():
    """Fetch macro indicators via yfinance (reliable on cloud)."""
    macro = {}
    start = datetime.now() - timedelta(days=730)  # 2 years history
    try:
        symbols = list(_MACRO_TICKERS.values())
        raw = yf.download(symbols, start=start, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            for label, sym in _MACRO_TICKERS.items():
                col = ('Close', sym)
                if col in raw.columns:
                    s = raw[col].dropna()
                    macro[label] = pd.DataFrame({label: s.values}, index=s.index)
        else:
            # Single ticker fallback
            macro['TNX'] = pd.DataFrame({'TNX': raw['Close'].dropna()})
    except Exception:
        pass
    return macro


@st.cache_data(ttl=3600)
def fetch_yield_curve_data():
    """Fetch yield curve maturities for current and 1-year-ago comparison."""
    yield_map = {
        '^IRX': 0.25,   # 3-month
        '^FVX': 5,      # 5-year
        '^TNX': 10,     # 10-year
        '^TYX': 30,     # 30-year
    }
    current_yields = {}
    year_ago_yields = {}
    try:
        symbols = list(yield_map.keys())
        start = datetime.now() - timedelta(days=400)
        raw = yf.download(symbols, start=start, progress=False)
        for sym, maturity in yield_map.items():
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    col = ('Close', sym)
                    if col not in raw.columns:
                        continue
                    series = raw[col].dropna()
                else:
                    series = raw['Close'].dropna()
                if len(series) > 0:
                    current_yields[maturity] = float(series.iloc[-1])
                    one_yr_ago = series.index[-1] - pd.DateOffset(years=1)
                    mask = series.index <= one_yr_ago
                    if mask.any():
                        year_ago_yields[maturity] = float(series.loc[mask].iloc[-1])
            except Exception:
                continue
    except Exception:
        pass
    return current_yields, year_ago_yields


# ========================================================================
# ML MODEL TRAINING (ML Predictions Tab)
# ========================================================================

@st.cache_data(ttl=3600)
def train_ml_models(prices_series, volumes_series=None):
    """Train ML models on stock price data and return predictions + metrics.

    Args:
        prices_series: pd.Series of closing prices (indexed by date)
        volumes_series: pd.Series of volumes (optional)

    Returns:
        dict with keys: models_info, feature_importance, predictions, actuals,
              feature_names, scaler, train_features (for simulator)
    """
    df = pd.DataFrame({'Close': prices_series})
    if volumes_series is not None:
        df['Volume'] = volumes_series
    else:
        df['Volume'] = 0

    # Feature engineering
    df['returns_1d'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(5)
    df['returns_21d'] = df['Close'].pct_change(21)
    df['vol_21d'] = df['returns_1d'].rolling(21).std()
    df['sma_20_ratio'] = df['Close'] / df['Close'].rolling(20).mean()
    df['sma_50_ratio'] = df['Close'] / df['Close'].rolling(50).mean()
    df['sma_200_ratio'] = df['Close'] / df['Close'].rolling(200).mean()
    vol_ma = df['Volume'].rolling(20).mean()
    if vol_ma.gt(0).any():
        df['volume_ratio'] = df['Volume'] / vol_ma.replace(0, np.nan)
    else:
        df['volume_ratio'] = 1.0  # no volume data — use neutral ratio

    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Target: forward 21-day return
    df['target'] = df['Close'].shift(-21) / df['Close'] - 1

    # Drop NaN rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < 60:
        return None

    feature_cols = [
        'returns_1d', 'returns_5d', 'returns_21d', 'vol_21d',
        'sma_20_ratio', 'sma_50_ratio', 'sma_200_ratio',
        'volume_ratio', 'rsi'
    ]
    # Only use features that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df['target'].values

    # Train/test split (80/20 chronological)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    }

    results = {}
    all_predictions = {}
    feature_importance = {}

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        results[name] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'model': model,
        }
        all_predictions[name] = preds

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance[name] = dict(zip(feature_cols, np.abs(model.coef_)))

    return {
        'models_info': results,
        'feature_importance': feature_importance,
        'predictions': all_predictions,
        'actuals': y_test,
        'test_dates': df.index[split_idx:],
        'feature_names': feature_cols,
        'scaler': scaler,
        'last_features': scaler.transform(X[-1:]),
        'last_price': df['Close'].iloc[-1],
    }


# ========================================================================
# ENHANCED VALUATION MODELS
# ========================================================================

class EnhancedValuationMetrics:
    """Comprehensive valuation metrics including DCF, CAPM, Fama-French, APT"""
    
    @staticmethod
    def calculate_beta(ticker_prices, benchmark_prices):
        """Calculates Beta dynamically using price history"""
        try:
            # Align dates
            common_dates = ticker_prices.index.intersection(benchmark_prices.index)
            if len(common_dates) < 30: return 1.0
            
            asset_ret = ticker_prices.loc[common_dates].pct_change().dropna()
            bench_ret = benchmark_prices.loc[common_dates].pct_change().dropna()
            
            # Align again after pct_change dropna
            common_dates = asset_ret.index.intersection(bench_ret.index)
            asset_ret = asset_ret.loc[common_dates]
            bench_ret = bench_ret.loc[common_dates]
            
            covariance = np.cov(asset_ret, bench_ret)[0][1]
            variance = np.var(bench_ret, ddof=1)
            
            if variance == 0: return 1.0
            return covariance / variance
        except:
            return 1.0

    @staticmethod
    def calculate_wacc(ticker, rf_rate, beta):
        """Calculate Weighted Average Cost of Capital using dynamic Beta"""
        try:
            info = yf.Ticker(ticker).info
            
            # Get financial data (Defaults for ETFs)
            market_cap = info.get('marketCap', 1e9)
            total_debt = info.get('totalDebt', 0)
            tax_rate = 0.21
            
            # Cost of equity (CAPM)
            market_premium = 0.057  # More realistic ERP (5.7%)
            cost_of_equity = rf_rate + beta * market_premium
            
            # Cost of debt
            if total_debt > 0:
                interest_expense = abs(info.get('interestExpense', 0) or 0)
                cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.045
            else:
                cost_of_debt = 0.0
            
            # WACC calculation
            total_value = market_cap + total_debt
            if total_value == 0: return cost_of_equity
            
            wacc = (market_cap/total_value * cost_of_equity + 
                   total_debt/total_value * cost_of_debt * (1 - tax_rate))
            
            return wacc
        except:
            return 0.10

    @staticmethod
    def calculate_dcf_value(ticker, rf_rate, beta):
        """Calculate DCF. Returns None for ETFs/Crypto if no cash flow found"""
        try:
            stock = yf.Ticker(ticker)
            # ETFs generally don't have this data
            if stock.info.get('quoteType') == 'ETF':
                return None 
            
            cf_statement = stock.cashflow
            if cf_statement.empty:
                return None
            
            # Get free cash flow (simplified)
            if 'Free Cash Flow' in cf_statement.index:
                fcf = cf_statement.loc['Free Cash Flow'].iloc[0]
            else:
                try:
                    operating_cf = cf_statement.loc['Total Cash From Operating Activities'].iloc[0]
                    capex = abs(cf_statement.loc['Capital Expenditures'].iloc[0])
                    fcf = operating_cf - capex
                except:
                    return None
            
            # Growth assumptions
            growth_rate = 0.05
            terminal_growth = 0.02
            
            # Calculate WACC with BETA
            wacc = EnhancedValuationMetrics.calculate_wacc(ticker, rf_rate, beta)
            
            # Project cash flows (5 years)
            projected_cf = []
            for year in range(1, 6):
                cf = fcf * (1 + growth_rate) ** year
                pv = cf / (1 + wacc) ** year
                projected_cf.append(pv)
            
            # Terminal value
            terminal_cf = fcf * (1 + growth_rate) ** 5 * (1 + terminal_growth)
            if wacc <= terminal_growth:
                return None
            terminal_value = terminal_cf / (wacc - terminal_growth)
            pv_terminal = terminal_value / (1 + wacc) ** 5
            
            # Enterprise value
            enterprise_value = sum(projected_cf) + pv_terminal
            
            return enterprise_value
        except:
            return None
    
    @staticmethod
    def calculate_capm_return(rf_rate, beta):
        """Calculate expected return using CAPM with dynamic Beta"""
        market_premium = 0.057
        return rf_rate + beta * market_premium
    
    @staticmethod
    def calculate_fama_french_return(ticker, prices, rf_rate, beta):
        """Calculate expected return using Fama-French 3-factor model with inferred factors"""
        # Factor premiums
        market_premium = 0.057
        smb_premium = 0.02 # Size premium
        hml_premium = 0.04 # Value premium
        
        # Estimate factors based on Beta if fundamentals missing (Common for ETFs)
        if beta > 1.2:
            beta_smb = 0.5
            beta_hml = -0.3 # Growth
        elif beta < 0.8:
            beta_smb = -0.2 # Large cap
            beta_hml = 0.4 # Value
        else:
            beta_smb = 0.0
            beta_hml = 0.0
            
        expected_return = (rf_rate + 
                          beta * market_premium +
                          beta_smb * smb_premium +
                          beta_hml * hml_premium)
        
        return expected_return
    
    @staticmethod
    def calculate_apt_return(ticker, prices, rf_rate):
        """Calculate expected return using Arbitrage Pricing Theory"""
        try:
            returns = prices.pct_change().dropna()
            
            # Economic factors (simplified)
            market_factor = returns.mean().mean() * 252
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
            market_premium = 0.057
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
    def calculate_bubble_burst_impact(ticker, prices, bubble_score, beta):
        """Estimate potential loss using Beta and Bubble Score"""
        try:
            # Higher bubble score + Higher Beta = Higher Risk
            base_risk = 0.15 # 15% correction baseline
            
            # Beta multiplier (High beta falls harder)
            beta_multiplier = max(beta, 0.5) 
            
            # Bubble multiplier (0% to 100% score)
            bubble_multiplier = 1 + bubble_score
            
            impact = base_risk * beta_multiplier * bubble_multiplier
            return min(impact, 0.80) # Cap at 80% loss
        except:
            return 0.30

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
        downside_mask = (self.returns < 0).any(axis=1)
        downside_rows = self.returns[downside_mask]
        return downside_rows.cov() * 252 if len(downside_rows) > 1 else self.returns.cov() * 252
    
    def _calculate_cvar_matrix(self, alpha=0.05):
        """Calculate CVaR covariance matrix (REAL LOGIC)"""
        ew_returns = self.returns.mean(axis=1)
        tail_dates = ew_returns <= ew_returns.quantile(alpha)
        tail_rets = self.returns.loc[tail_dates]
        return tail_rets.cov() * 252 if len(tail_rets) > 1 else self.returns.cov() * 252
    
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
            portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_var if portfolio_var > 0 else np.ones(self.n_assets) / self.n_assets
            target = 1.0 / self.n_assets
            mse = np.sum((risk_contrib - target)**2)
            
            if bubble_aware:
                for i, ticker in enumerate(self.prices.columns):
                    if ticker in self.bubble_scores:
                        adjustment = 1 - (self.bubble_scores[ticker] * penalty_factor)
                        mse += (risk_contrib[i] - target * adjustment)**2
            
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
            # Use the actual CVaR matrix calculated in init
            portfolio_risk = np.dot(weights.T, np.dot(self.cvar_matrix, weights))
            
            if bubble_aware:
                portfolio_risk += self._bubble_penalty(weights, penalty_factor)
            
            return portfolio_risk
        
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
        
        # Convert to long-only
        weights = np.maximum(raw_weights, 0)
        sum_weights = np.sum(weights)
        weights = weights / sum_weights if sum_weights > 0 else np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def black_litterman(self, market_cap_weights=None, views=None, view_confidence=None):
        """Black-Litterman Portfolio"""
        if market_cap_weights is None:
            market_cap_weights = np.ones(self.n_assets) / self.n_assets
        
        # Market implied returns
        mkt_weights = np.ones(self.n_assets) / self.n_assets
        mkt_var = float(np.dot(mkt_weights.T, np.dot(self.cov_matrix, mkt_weights)))
        lambda_param = (self.mean_returns.mean() - self.rf_rate) / mkt_var if mkt_var > 0 else 2.5
        pi = lambda_param * np.dot(self.cov_matrix, market_cap_weights)
        
        if views is None or view_confidence is None:
            return market_cap_weights
        
        # Incorporate views (simplified)
        tau = 0.05
        P = np.eye(self.n_assets)
        Q = views
        omega = np.diag(view_confidence)
        
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
        except:
            inv_cov = np.linalg.pinv(self.cov_matrix)

        bl_returns = pi + tau * self.cov_matrix @ P.T @ np.linalg.inv(
            P @ tau * self.cov_matrix @ P.T + omega
        ) @ (Q - P @ pi)
        
        weights = np.dot(inv_cov, bl_returns)
        sum_w = np.sum(weights)
        weights = weights / sum_w if sum_w != 0 else weights
        
        return np.maximum(weights, 0)
    
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
            sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            while sort_idx.max() >= num_items:
                sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
                df0 = sort_idx[sort_idx >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_idx[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_idx = pd.concat([sort_idx, df0]).sort_index()
                sort_idx = sort_idx.drop_duplicates()
            return sort_idx.tolist()
        
        sort_idx = get_quasi_diag(link)
        
        # Recursive bisection
        def recursive_bisection(cov, sort_idx):
            w = pd.Series(1.0, index=sort_idx)
            clusters = [sort_idx]
            while len(clusters) > 0:
                new_clusters = []
                for c in clusters:
                    if len(c) > 1:
                        half = len(c) // 2
                        c0, c1 = c[:half], c[half:]
                        new_clusters.extend([c0, c1])
                        v0 = np.diag(cov[np.ix_(c0, c0)]).sum()
                        v1 = np.diag(cov[np.ix_(c1, c1)]).sum()
                        alpha = 1 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
                        w[c0] *= alpha
                        w[c1] *= (1 - alpha)
                clusters = [c for c in new_clusters if len(c) > 1]
            return w.values / w.sum()
        
        # Get HRP weights
        weights = recursive_bisection(self.cov_matrix.values, sort_idx)
        
        # Reorder to original asset order
        final_weights = np.zeros(self.n_assets)
        for i, idx in enumerate(sort_idx):
            if idx < self.n_assets:
                final_weights[idx] = weights[i]
        final_weights = final_weights / final_weights.sum() if final_weights.sum() > 0 else np.ones(self.n_assets) / self.n_assets
        
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
        downside_diff = np.minimum(portfolio_returns, 0)
        downside_dev = np.sqrt(np.mean(downside_diff**2)) * np.sqrt(252)
        
        # Sortino ratio
        sortino = (portfolio_return - self.rf_rate) / downside_dev if downside_dev > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        max_dd = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Calmar ratio
        calmar = portfolio_return / abs(max_dd) if max_dd != 0 else 0
        
        # CVaR (95%)
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else var_95
        
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
    st.markdown("""
    <div class="section-header">
        <div class="section-label">SECTION 03</div>
        <div class="section-title">Portfolio Optimization</div>
        <div class="section-subtitle">Multi-strategy optimization with bubble-aware risk management</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Understanding Portfolio Optimization"):
        st.markdown("**Modern Portfolio Theory (MPT)** — Harry Markowitz's framework for constructing portfolios that maximize return for a given risk level.")
        st.markdown("**Maximum Sharpe Ratio** — Finds the tangency portfolio on the efficient frontier:")
        st.latex(r"\max_w \frac{w^T \mu - R_f}{\sqrt{w^T \Sigma w}}")
        st.markdown("**Minimum Variance** — Minimizes total portfolio risk:")
        st.latex(r"\min_w \quad w^T \Sigma w")
        st.markdown("**Risk Parity** — Equalizes each asset's contribution to total portfolio risk:")
        st.latex(r"RC_i = w_i \cdot \frac{(\Sigma w)_i}{w^T \Sigma w} = \frac{1}{N}")
        st.markdown("**Minimum CVaR (Conditional Value-at-Risk)** — Minimizes expected loss in the worst-case tail:")
        st.latex(r"CVaR_\alpha = E[L \mid L > VaR_\alpha]")
        st.markdown("**Maximum Diversification** — Maximizes the diversification ratio:")
        st.latex(r"DR = \frac{w^T \sigma}{\sqrt{w^T \Sigma w}}")
        st.markdown("**Kelly Criterion** — Optimal bet sizing for maximum long-term growth:")
        st.latex(r"f^* = \frac{\mu - R_f}{\sigma^2}")
        st.markdown("**Black-Litterman** — Combines market equilibrium with investor views:")
        st.latex(r"\mu_{BL} = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \pi + P^T \Omega^{-1} Q]")
        st.markdown("**Hierarchical Risk Parity (HRP)** — Uses hierarchical clustering on the correlation matrix to build a diversified portfolio without requiring covariance matrix inversion.")
        st.markdown("""
        **Bubble-Aware Optimization** — Penalizes allocations to assets with high bubble scores by adjusting expected returns downward:
        """)
        st.latex(r"\mu_{adjusted,i} = \mu_i \times (1 - \lambda \cdot BubbleScore_i)")
        st.markdown("Where lambda is the bubble penalty factor (0 = ignore bubbles, 1 = full penalty).")

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
        _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
        fig = go.Figure(data=[go.Pie(
            labels=data['tickers'],
            values=weights,
            hole=0.4,
            marker=dict(colors=_clrs[:len(data['tickers'])]),
            textposition='auto',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>' +
                          'Weight: %{percent}<br>' +
                          'Value: %{value:.4f}<br>' +
                          '<extra></extra>'
        )])

        _gold = '#ffd700' if st.session_state.get('theme') == 'dark' else '#b8860b'
        fig.update_layout(
            title=dict(text=f"{optimization_method} Portfolio Allocation", font=dict(color=_fc)),
            template=_tmpl,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color=_fc)),
            font=dict(family="Inter, system-ui, sans-serif", color=_fc),
            annotations=[
                dict(text=f'Sharpe: {metrics["Sharpe Ratio"]:.2f}',
                     x=0.5, y=0.5, font_size=16, showarrow=False,
                     font=dict(color=_gold, family='JetBrains Mono, monospace'))
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weights table with additional info
        weights_df = pd.DataFrame({
            'Asset': data['tickers'],
            'Weight': weights,
            'Contribution': weights * metrics['Expected Return'],
            'Risk Contrib': weights * np.sqrt(np.diag(optimizer.cov_matrix)),
            'Bubble Score': [data['bubble_scores'].get(t, 0) for t in data['tickers']]
        })
        
        render_styled_table(
            weights_df,
            format_dict={
                'Weight': '{:.2%}',
                'Contribution': '{:.2%}',
                'Risk Contrib': '{:.2%}',
                'Bubble Score': '{:.2%}'
            },
            highlight_col='Weight'
        )
    
    # Efficient Frontier
    if show_efficient_frontier:
        st.markdown("#### Efficient Frontier")
        
        with st.spinner("Generating efficient frontier..."):
            frontier_return, frontier_vol, frontier_weights = optimizer.efficient_frontier()
            
            # Create frontier plot
            _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
            _gold = '#ffd700' if st.session_state.get('theme') == 'dark' else '#b8860b'
            fig = go.Figure()

            # Efficient frontier line
            fig.add_trace(go.Scatter(
                x=frontier_vol,
                y=frontier_return,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color=_clrs[0], width=2)
            ))

            # Current portfolio
            fig.add_trace(go.Scatter(
                x=[metrics['Volatility']],
                y=[metrics['Expected Return']],
                mode='markers',
                name='Current Portfolio',
                marker=dict(size=15, color=_gold, symbol='star')
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
                    line=dict(color=_gold, width=1, dash='dash')
                ))

            fig.update_layout(
                title=dict(text="Efficient Frontier Analysis", font=dict(color=_fc)),
                xaxis_title="Volatility (Annual)",
                yaxis_title="Expected Return (Annual)",
                template=_tmpl,
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode='closest',
                legend=dict(font=dict(color=_fc)),
                font=dict(family="Inter, system-ui, sans-serif", color=_fc)
            )
            fig.update_xaxes(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc))
            fig.update_yaxes(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc))
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Backtest Results
    if show_backtest:
        st.markdown("#### Backtest Results")
        
        backtest_results = optimizer.backtest_portfolio(weights)

        # Create subplots for backtest visualization
        _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
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
                line=dict(color=_clrs[0], width=2)
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
                line=dict(color=_clrs[5], width=1, dash='dash')
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
                fillcolor='rgba(255,77,109,0.2)' if st.session_state.get('theme') == 'dark' else 'rgba(220,38,38,0.15)',
                line=dict(color=_clrs[3], width=1)
            ),
            row=2, col=1
        )

        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=backtest_results['rolling_sharpe'].index,
                y=backtest_results['rolling_sharpe'].values,
                name='Rolling Sharpe',
                line=dict(color=_clrs[2], width=1)
            ),
            row=3, col=1
        )

        fig.update_layout(
            template=_tmpl,
            height=800,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(font=dict(color=_fc)),
            font=dict(family="Inter, system-ui, sans-serif", color=_fc)
        )
        fig.update_xaxes(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc))
        fig.update_yaxes(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc))

        st.plotly_chart(fig, use_container_width=True)

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
    
    render_styled_table(
        comparison_df,
        format_dict={
            'Return': '{:.2%}',
            'Volatility': '{:.2%}',
            'Sharpe': '{:.2f}',
            'Max DD': '{:.2%}',
            'Div Ratio': '{:.2f}'
        },
        highlight_col='Sharpe'
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
        "Download Portfolio Weights (CSV)",
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
        
        X = np.log(2 * np.sin(np.pi * low_freqs))
        Y = np.log(low_psd + 1e-10)
        
        X_with_const = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X_with_const, Y, rcond=None)[0]
        
        d = -beta[1] / 2
        
        residuals = Y - X_with_const @ beta
        se = np.pi / np.sqrt(24 * len(X))
        
        return d, se

class BubbleDetector:
    """Comprehensive bubble detection using multiple methods"""
    
    def __init__(self):
        self.metcalfe = MetcalfeLawAdvanced()
        self.long_memory = LongMemoryEstimators()
    
    def detect_bubbles(self, prices, returns, volumes=None):
        """Detect bubbles using multiple indicators with Better Logic"""
        np.random.seed(42) # Remove jitter
        results = {}
        
        # 1. Generate proxy network data
        # BETTER: Use Volume as a proxy for activity/users if available
        if volumes is not None and not volumes.empty:
            # Smooth volume to represent "Active Users"
            users = volumes.rolling(window=30).mean().bfill()
            # Normalize to avoid scale issues (Proxy users)
            if users.iloc[0] > 0:
                users = users / users.iloc[0] * 1000
            else:
                users = np.log(prices.values) * 1000
        else:
            # Fallback: Use Log Price (less circular than Sqrt Price)
            users = np.log(prices.values) * 1000
        
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
        
        # 6. Calculate GRANULAR composite bubble score
        score = 0.0
        
        # MMV Score (0.0 to 0.4)
        if not np.isnan(current_mmv):
            # Sigmoid-like scaling for MMV deviation from 1.0
            deviation = max(current_mmv - 1.0, 0)
            mmv_score = 1 / (1 + np.exp(-deviation * 5)) - 0.5 
            score += max(mmv_score * 0.8, 0)
        
        # Long Memory Score (0.0 to 0.3)
        # d > 0.5 implies non-stationary bubble behavior
        d_val = max(min(results.get('d_parameter', 0), 1.0), 0)
        score += d_val * 0.3
        
        # Kurtosis Score (0.0 to 0.2)
        # Fat tails indicate bubble risk
        kurt = min(max(results.get('kurtosis', 0), 0), 10) / 10
        score += kurt * 0.2
        
        # Volatility Clustering (0.0 to 0.1)
        if results.get('has_vol_clustering', False):
            score += 0.1
            
        results['bubble_score'] = min(score, 1.0)
        
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
        indicators['MACD_Histogram'] = indicators['MACD'] - indicators['MACD_Signal']
        
        indicators['RSI'] = ta.momentum.RSIIndicator(prices, window=14).rsi()
        
        bb = ta.volatility.BollingerBands(prices)
        indicators['BB_Upper'] = bb.bollinger_hband()
        indicators['BB_Lower'] = bb.bollinger_lband()
        
        return indicators

# ========================================================================
# OPTIONS PRICING HELPERS (Black-Scholes & Greeks)
# ========================================================================

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return intrinsic
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * scipy_norm.cdf(d1) - K * np.exp(-r * T) * scipy_norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * scipy_norm.cdf(-d2) - S * scipy_norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0:
        return {'Delta': 0, 'Gamma': 0, 'Theta': 0, 'Vega': 0, 'Rho': 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = scipy_norm.pdf(d1)
    if option_type == 'call':
        delta = scipy_norm.cdf(d1)
        theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * scipy_norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * scipy_norm.cdf(d2) / 100
    else:
        delta = scipy_norm.cdf(d1) - 1
        theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * scipy_norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * scipy_norm.cdf(-d2) / 100
    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega = S * nd1 * np.sqrt(T) / 100
    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}


@st.cache_data(ttl=3600)
def fetch_options_chain(ticker, expiration):
    """Fetch options chain for a ticker and expiration date."""
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiration)
        return chain.calls, chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_options_expirations(ticker):
    """Fetch available expiration dates for a ticker."""
    try:
        t = yf.Ticker(ticker)
        return list(t.options[:12])
    except Exception:
        return []


def options_payoff(strategy, S_range, S0, K1, K2=None, premium1=0, premium2=0):
    """Calculate options strategy payoff at expiration."""
    payoff = np.zeros_like(S_range, dtype=float)
    if strategy == 'Long Call':
        payoff = np.maximum(S_range - K1, 0) - premium1
    elif strategy == 'Long Put':
        payoff = np.maximum(K1 - S_range, 0) - premium1
    elif strategy == 'Bull Call Spread':
        payoff = np.maximum(S_range - K1, 0) - np.maximum(S_range - K2, 0) - premium1 + premium2
    elif strategy == 'Bear Put Spread':
        payoff = np.maximum(K2 - S_range, 0) - np.maximum(K1 - S_range, 0) - premium2 + premium1
    elif strategy == 'Straddle':
        payoff = np.maximum(S_range - K1, 0) + np.maximum(K1 - S_range, 0) - premium1 - premium2
    elif strategy == 'Iron Condor':
        # Buy OTM put (K1), sell ATM put (K1+5), sell ATM call (K2-5), buy OTM call (K2)
        k_pl = K1
        k_ps = K1 + (K2 - K1) * 0.33
        k_cs = K2 - (K2 - K1) * 0.33
        k_ch = K2
        payoff = (np.maximum(k_pl - S_range, 0) - np.maximum(k_ps - S_range, 0)
                  - np.maximum(S_range - k_cs, 0) + np.maximum(S_range - k_ch, 0)
                  + premium1)  # net credit
    return payoff


# ========================================================================
# RISK & GEOPOLITICS HELPERS
# ========================================================================

@st.cache_data(ttl=3600)
def fetch_risk_data():
    """Fetch risk-related market data from yfinance."""
    symbols = {
        'VIX': '^VIX',
        'DXY': 'DX-Y.NYB',
        'Gold': 'GC=F',
        'Oil': 'CL=F',
        'TNX': '^TNX',
        'IRX': '^IRX',
        'SPX': '^GSPC',
    }
    result = {}
    end = datetime.now()
    start = end - timedelta(days=730)  # 2 years
    for name, sym in symbols.items():
        try:
            df = yf.download(sym, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
                if isinstance(df, pd.DataFrame):
                    df = df.iloc[:, 0]
                result[name] = df.dropna()
            elif 'Close' in df.columns:
                result[name] = df['Close'].dropna()
            else:
                result[name] = pd.Series(dtype=float)
        except Exception:
            result[name] = pd.Series(dtype=float)
    return result


def calculate_composite_risk_score(risk_data):
    """Calculate a composite risk score from 0-100."""
    score = 0.0
    # VIX component (0-25): normalize VIX 10-50
    vix = risk_data.get('VIX', pd.Series(dtype=float))
    if len(vix) > 0:
        vix_val = float(vix.iloc[-1])
        score += min(max((vix_val - 10) / 40 * 25, 0), 25)

    # Yield curve component (0-25): inversion = 25
    tnx = risk_data.get('TNX', pd.Series(dtype=float))
    irx = risk_data.get('IRX', pd.Series(dtype=float))
    if len(tnx) > 0 and len(irx) > 0:
        spread = float(tnx.iloc[-1]) - float(irx.iloc[-1])
        if spread < 0:
            score += min(abs(spread) / 2 * 25, 25)
        else:
            score += max(0, 10 - spread * 5)

    # Safe haven component (0-25): gold 30d return
    gold = risk_data.get('Gold', pd.Series(dtype=float))
    if len(gold) > 30:
        gold_ret_30d = (float(gold.iloc[-1]) / float(gold.iloc[-30]) - 1) * 100
        if gold_ret_30d > 10:
            score += 25
        elif gold_ret_30d > 5:
            score += 20
        elif gold_ret_30d > 2:
            score += 10

    # Volatility regime (0-25): VIX 21d avg vs 252d avg
    if len(vix) > 252:
        vix_21 = float(vix.iloc[-21:].mean())
        vix_252 = float(vix.iloc[-252:].mean())
        ratio = vix_21 / vix_252 if vix_252 > 0 else 1
        score += min(max((ratio - 1) * 50, 0), 25)

    return min(score, 100)


# ========================================================================
# SENTIMENT ANALYSIS HELPERS
# ========================================================================

POSITIVE_WORDS = {'surge', 'rally', 'gain', 'profit', 'beat', 'upgrade', 'bullish', 'growth',
                  'strong', 'record', 'outperform', 'buy', 'positive', 'boom', 'soar', 'high',
                  'innovation', 'breakthrough', 'momentum', 'recovery', 'optimistic', 'revenue',
                  'earnings', 'up', 'rises', 'jumps', 'climbs', 'wins', 'success'}
NEGATIVE_WORDS = {'crash', 'plunge', 'loss', 'miss', 'downgrade', 'bearish', 'decline',
                  'weak', 'fall', 'drop', 'sell', 'negative', 'risk', 'warning', 'fear',
                  'recession', 'bankruptcy', 'layoff', 'investigation', 'lawsuit', 'cut',
                  'slump', 'tumble', 'deficit', 'debt', 'crisis', 'downturn', 'loses', 'fails'}


def simple_sentiment_score(text):
    """Score sentiment of text using keyword matching. Returns -1 to +1."""
    words = text.lower().split()
    pos = sum(1 for w in words if w.strip('.,!?;:()') in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip('.,!?;:()') in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


@st.cache_data(ttl=3600)
def fetch_ticker_news(ticker):
    """Fetch news for a ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if news is None:
            return []
        return news
    except Exception:
        return []


# ========================================================================
# ML CLUSTERING HELPERS
# ========================================================================

def compute_asset_features(prices_df):
    """Compute features for clustering: return, vol, skew, kurtosis, Sharpe, max drawdown."""
    returns = prices_df.pct_change().dropna()
    features = {}
    for col in returns.columns:
        r = returns[col]
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        rf_annual = 0.045
        sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0
        cum = (1 + r).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        features[col] = {
            'Ann Return': ann_ret,
            'Volatility': ann_vol,
            'Skewness': float(r.skew()),
            'Kurtosis': float(r.kurtosis()),
            'Sharpe': sharpe,
            'Max Drawdown': max_dd,
        }
    return pd.DataFrame(features).T


# ========================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================



# ========================================================================
# NEW MODULES — Backtesting, Fundamentals, Fixed Income, Factor Model,
#               Options Builder, Risk Suite
# ========================================================================

# ─── BACKTEST ───
"""
module_backtest.py
==================
Backtesting Engine module for QuantLab.

Provides:
  - BacktestEngine: Event-driven portfolio backtester with walk-forward
    rebalancing and four optimization strategies.
  - calculate_backtest_metrics: Comprehensive risk/return metric suite.
  - run_benchmark_comparison: Side-by-side comparison of all strategies.
  - render_backtesting_tab: Full Streamlit UI for the Backtesting tab.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
_STRATEGIES = ["Equal Weight", "Max Sharpe", "Min Variance", "Risk Parity"]
_FREQUENCIES = ["Monthly", "Quarterly", "Annually", "Buy & Hold"]


# ---------------------------------------------------------------------------
# Helpers – thin wrappers so the engine doesn't need a live app instance
# ---------------------------------------------------------------------------


def _equal_weight(n: int) -> np.ndarray:
    """Return equal-weight vector of length *n*."""
    return np.full(n, 1.0 / n)


def _max_sharpe_weights(
    returns: pd.DataFrame, rf_rate: float = 0.045
) -> np.ndarray:
    """
    Mean-variance optimisation: maximise Sharpe ratio.

    Falls back to equal weight if optimisation fails.
    """
    n = returns.shape[1]
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.cov() * TRADING_DAYS
    rf_daily = rf_rate / TRADING_DAYS

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, mu))
        port_vol = float(np.sqrt(w @ sigma.values @ w))
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - rf_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = _equal_weight(n)

    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return w0


def _min_variance_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Minimum-variance optimisation.

    Falls back to equal weight if optimisation fails.
    """
    n = returns.shape[1]
    sigma = returns.cov() * TRADING_DAYS

    def port_var(w: np.ndarray) -> float:
        return float(w @ sigma.values @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = _equal_weight(n)

    res = minimize(
        port_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return w0


def _risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Equal-risk-contribution (Risk Parity) weights via gradient descent.

    Falls back to equal weight if optimisation fails.
    """
    n = returns.shape[1]
    sigma = returns.cov().values * TRADING_DAYS
    w0 = _equal_weight(n)

    def objective(w: np.ndarray) -> float:
        port_var = float(w @ sigma @ w)
        if port_var < 1e-14:
            return 0.0
        mrc = sigma @ w  # marginal risk contributions
        rc = w * mrc / np.sqrt(port_var)  # risk contributions
        # Minimise sum of squared pairwise differences
        target = np.sqrt(port_var) / n
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-4, 1.0)] * n

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return w0


def _get_weights(
    strategy: str,
    hist_returns: pd.DataFrame,
    rf_rate: float,
    bubble_scores: Optional[dict],
) -> np.ndarray:
    """
    Dispatch to the correct optimiser and optionally tilt by bubble scores.

    *hist_returns* must contain only history **prior** to the rebalance date.
    """
    n = hist_returns.shape[1]
    if n == 0:
        return np.array([])

    if strategy == "Equal Weight":
        w = _equal_weight(n)
    elif strategy == "Max Sharpe":
        w = _max_sharpe_weights(hist_returns, rf_rate)
    elif strategy == "Min Variance":
        w = _min_variance_weights(hist_returns)
    elif strategy == "Risk Parity":
        w = _risk_parity_weights(hist_returns)
    else:
        w = _equal_weight(n)

    # Optional bubble-score tilt: down-weight assets with high bubble scores
    if bubble_scores:
        tickers = hist_returns.columns.tolist()
        tilt = np.array(
            [1.0 - 0.5 * float(bubble_scores.get(t, 0.0)) for t in tickers]
        )
        tilt = np.clip(tilt, 0.05, 1.0)
        w = w * tilt
        total = w.sum()
        if total > 0:
            w /= total

    return w


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Return the rebalance dates for a given frequency.

    For 'Buy & Hold' only the first date is returned.
    """
    if freq == "Buy & Hold":
        return pd.DatetimeIndex([index[0]])

    freq_map = {"Monthly": "MS", "Quarterly": "QS", "Annually": "YS"}
    offset = freq_map[freq]
    # All month/quarter/year starts that fall within the index range
    cal = pd.date_range(start=index[0], end=index[-1], freq=offset)
    # Snap each calendar date to the nearest available trading day (forward)
    dates = []
    for d in cal:
        mask = index >= d
        if mask.any():
            dates.append(index[mask][0])
    return pd.DatetimeIndex(sorted(set(dates)))


# ---------------------------------------------------------------------------
# 1. BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Event-driven portfolio backtester.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices, rows = dates, columns = tickers.
        Must have a DatetimeIndex.
    rf_rate : float
        Annual risk-free rate used for Sharpe optimisation (default 4.5 %).
    """

    def __init__(self, prices: pd.DataFrame, rf_rate: float = 0.045):
        if prices.empty:
            raise ValueError("prices DataFrame must not be empty.")
        self.prices: pd.DataFrame = prices.sort_index().dropna(axis=1, how="all")
        self.rf_rate: float = rf_rate
        self.tickers: list[str] = self.prices.columns.tolist()

    # ------------------------------------------------------------------
    def run(
        self,
        strategy: str = "Equal Weight",
        rebalance_freq: str = "Monthly",
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        bubble_scores: Optional[dict] = None,
    ) -> dict:
        """
        Run a walk-forward backtest.

        Parameters
        ----------
        strategy : str
            One of 'Equal Weight', 'Max Sharpe', 'Min Variance', 'Risk Parity'.
        rebalance_freq : str
            One of 'Monthly', 'Quarterly', 'Annually', 'Buy & Hold'.
        initial_capital : float
            Starting portfolio value in USD.
        transaction_cost : float
            Round-trip cost as a fraction (e.g. 0.001 = 0.1 %).
        bubble_scores : dict, optional
            {ticker: score ∈ [0, 1]} used to tilt weights away from
            overvalued assets.

        Returns
        -------
        dict with keys:
          'equity_curve'    : pd.Series  – portfolio value at each date
          'returns'         : pd.Series  – daily portfolio returns
          'trades'          : list[dict] – trade log
          'metrics'         : dict       – performance metrics
          'weights_history' : pd.DataFrame – target weights at each rebalance
        """
        prices = self.prices
        index = prices.index
        n_assets = len(self.tickers)

        # ---- minimum lookback for covariance estimation ----
        min_lookback = max(n_assets * 2, 60)

        rb_dates = _rebalance_dates(index, rebalance_freq)

        # State
        cash: float = initial_capital
        holdings: np.ndarray = np.zeros(n_assets)  # shares held per ticker
        equity_curve: list[float] = []
        trades: list[dict] = []
        weights_records: list[dict] = []
        last_weights: np.ndarray = np.zeros(n_assets)

        for i, date in enumerate(index):
            current_prices = prices.loc[date].values.astype(float)

            # ---- rebalance? ----
            if date in rb_dates:
                # Build history strictly before today
                hist = prices.loc[prices.index < date]

                if len(hist) >= min_lookback:
                    hist_ret = hist.pct_change().dropna()
                    # Drop tickers with no variance
                    valid = hist_ret.std() > 1e-8
                    hist_ret = hist_ret.loc[:, valid]
                    valid_tickers = hist_ret.columns.tolist()
                else:
                    # Not enough history – fall back to equal weight
                    hist_ret = pd.DataFrame()
                    valid_tickers = self.tickers

                if len(valid_tickers) == 0:
                    valid_tickers = self.tickers
                    hist_ret = pd.DataFrame()

                if hist_ret.empty:
                    raw_w = _equal_weight(len(valid_tickers))
                else:
                    raw_w = _get_weights(
                        strategy, hist_ret, self.rf_rate, bubble_scores
                    )

                # Map back to full ticker list
                target_weights = np.zeros(n_assets)
                for j, t in enumerate(valid_tickers):
                    idx_t = self.tickers.index(t)
                    target_weights[idx_t] = raw_w[j]
                if target_weights.sum() > 0:
                    target_weights /= target_weights.sum()

                # Portfolio value before rebalance
                port_value = cash + float(np.dot(holdings, current_prices))

                # Target dollar amounts
                target_values = target_weights * port_value

                for j, ticker in enumerate(self.tickers):
                    p = current_prices[j]
                    if p <= 0 or not np.isfinite(p):
                        continue
                    target_shares = target_values[j] / p
                    delta_shares = target_shares - holdings[j]

                    if abs(delta_shares) < 1e-6:
                        continue

                    trade_value = abs(delta_shares * p)
                    cost = trade_value * transaction_cost
                    action = "BUY" if delta_shares > 0 else "SELL"

                    if action == "BUY":
                        total_cost = trade_value + cost
                        if cash >= total_cost:
                            holdings[j] += delta_shares
                            cash -= total_cost
                        else:
                            # Partial fill with available cash
                            affordable = (cash) / (p * (1 + transaction_cost))
                            if affordable > 0:
                                holdings[j] += affordable
                                cash -= affordable * p * (1 + transaction_cost)
                    else:
                        proceeds = trade_value - cost
                        holdings[j] += delta_shares  # delta is negative
                        cash += proceeds

                    trades.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": ticker,
                            "action": action,
                            "shares": round(abs(delta_shares), 4),
                            "price": round(p, 4),
                            "value": round(trade_value, 2),
                        }
                    )

                # Record weights
                weights_records.append(
                    {
                        "date": date,
                        **{t: round(target_weights[k], 6) for k, t in enumerate(self.tickers)},
                    }
                )
                last_weights = target_weights.copy()

            # ---- mark-to-market ----
            port_value = cash + float(np.dot(holdings, current_prices))
            equity_curve.append(port_value)

        equity_series = pd.Series(equity_curve, index=index, name="Portfolio")
        daily_returns = equity_series.pct_change().dropna()

        weights_df = (
            pd.DataFrame(weights_records).set_index("date")
            if weights_records
            else pd.DataFrame()
        )

        metrics = calculate_backtest_metrics(equity_series, daily_returns, self.rf_rate)

        return {
            "equity_curve": equity_series,
            "returns": daily_returns,
            "trades": trades,
            "metrics": metrics,
            "weights_history": weights_df,
        }


# ---------------------------------------------------------------------------
# 2. Performance metrics
# ---------------------------------------------------------------------------


def calculate_backtest_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    rf_rate: float,
) -> dict:
    """
    Compute a comprehensive set of risk and return metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio value at each date.
    returns : pd.Series
        Daily portfolio returns (fractional, e.g. 0.01 = 1 %).
    rf_rate : float
        Annual risk-free rate.

    Returns
    -------
    dict with keys: CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio,
    Calmar Ratio, Max Drawdown, Max Drawdown Duration (days), Win Rate,
    Avg Win, Avg Loss, Profit Factor, Total Return, Best Day, Worst Day,
    VaR 95%, CVaR 95%.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {}

    ret = returns.dropna()
    equity = equity_curve.dropna()

    # ---- return metrics ----
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    n_days = (equity.index[-1] - equity.index[0]).days
    years = max(n_days / 365.25, 1 / 365.25)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)

    ann_vol = float(ret.std() * np.sqrt(TRADING_DAYS)) if len(ret) > 1 else 0.0

    rf_daily = rf_rate / TRADING_DAYS
    excess = ret - rf_daily
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))
        if excess.std() > 0
        else 0.0
    )

    downside_ret = ret[ret < rf_daily]
    downside_std = float(downside_ret.std() * np.sqrt(TRADING_DAYS)) if len(downside_ret) > 1 else 1e-10
    sortino = float((cagr - rf_rate) / downside_std) if downside_std > 0 else 0.0

    # ---- drawdown ----
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Max drawdown duration
    in_drawdown = drawdown < 0
    max_dd_duration = 0
    current_dd_start: Optional[pd.Timestamp] = None
    for date, val in in_drawdown.items():
        if val and current_dd_start is None:
            current_dd_start = date
        elif not val and current_dd_start is not None:
            dur = (date - current_dd_start).days
            max_dd_duration = max(max_dd_duration, dur)
            current_dd_start = None
    if current_dd_start is not None:
        dur = (in_drawdown.index[-1] - current_dd_start).days
        max_dd_duration = max(max_dd_duration, dur)

    calmar = float(cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # ---- trade statistics ----
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    win_rate = float(len(wins) / len(ret)) if len(ret) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    best_day = float(ret.max()) if len(ret) > 0 else 0.0
    worst_day = float(ret.min()) if len(ret) > 0 else 0.0

    # ---- risk metrics ----
    var_95 = float(np.percentile(ret, 5)) if len(ret) > 0 else 0.0
    cvar_95 = float(ret[ret <= var_95].mean()) if len(ret[ret <= var_95]) > 0 else var_95

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Max Drawdown": max_drawdown,
        "Max Drawdown Duration (days)": max_dd_duration,
        "Win Rate": win_rate,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Profit Factor": profit_factor,
        "Best Day": best_day,
        "Worst Day": worst_day,
        "VaR 95%": var_95,
        "CVaR 95%": cvar_95,
    }


# ---------------------------------------------------------------------------
# 3. Benchmark comparison
# ---------------------------------------------------------------------------


def run_benchmark_comparison(
    prices: pd.DataFrame,
    rf_rate: float = 0.045,
    initial_capital: float = 100_000.0,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Run all four strategies plus a pure buy-and-hold benchmark.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices.
    rf_rate : float
        Annual risk-free rate.
    initial_capital : float
        Starting capital for each strategy.
    transaction_cost : float
        Round-trip transaction cost fraction.

    Returns
    -------
    dict of {strategy_name: metrics_dict}
    """
    engine = BacktestEngine(prices, rf_rate=rf_rate)
    results: dict[str, dict] = {}

    # ---- four optimised strategies ----
    for strat in _STRATEGIES:
        try:
            res = engine.run(
                strategy=strat,
                rebalance_freq="Monthly",
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
            )
            results[strat] = res["metrics"]
        except Exception:
            results[strat] = {}

    # ---- buy-and-hold equal weight benchmark ----
    try:
        bh_res = engine.run(
            strategy="Equal Weight",
            rebalance_freq="Buy & Hold",
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
        )
        results["Buy & Hold"] = bh_res["metrics"]
    except Exception:
        results["Buy & Hold"] = {}

    return results


# ---------------------------------------------------------------------------
# 4. Streamlit tab rendering
# ---------------------------------------------------------------------------


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_ratio(v: float) -> str:
    return f"{v:.4f}"


def _fmt_days(v: float) -> str:
    return f"{int(v):,} days"


def _build_equity_chart(
    equity_curve: pd.Series,
    benchmark_curve: pd.Series,
    theme: str,
) -> go.Figure:
    """Equity curve vs buy-and-hold benchmark."""
    plotly_theme = _get_plotly_theme(theme)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            name="Strategy",
            line=dict(color="#00D4FF", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    if benchmark_curve is not None and not benchmark_curve.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_curve.index,
                y=benchmark_curve.values,
                name="Buy & Hold",
                line=dict(color="#FF6B6B", width=2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        **plotly_theme,
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_drawdown_chart(equity_curve: pd.Series, theme: str) -> go.Figure:
    """Underwater equity (drawdown) chart."""
    plotly_theme = _get_plotly_theme(theme)
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.3)",
            line=dict(color="#FF6B6B", width=1),
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_theme,
        title="Drawdown (Underwater Equity Curve)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    return fig


def _build_rolling_sharpe_chart(
    returns: pd.Series, rf_rate: float, theme: str
) -> go.Figure:
    """Rolling 252-day Sharpe ratio chart."""
    plotly_theme = _get_plotly_theme(theme)
    rf_daily = rf_rate / TRADING_DAYS
    excess = returns - rf_daily
    roll_mean = excess.rolling(TRADING_DAYS).mean()
    roll_std = excess.rolling(TRADING_DAYS).std()
    rolling_sharpe = roll_mean / roll_std * np.sqrt(TRADING_DAYS)
    rolling_sharpe = rolling_sharpe.dropna()

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=1, line_dash="dot", line_color="#00D4FF", opacity=0.4)

    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            name="Rolling Sharpe (252d)",
            line=dict(color="#7C3AED", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **plotly_theme,
        title="Rolling 252-Day Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
    )
    return fig


def _get_plotly_theme(theme: str) -> dict:
    """
    Return Plotly layout kwargs matching the app theme.

    Falls back gracefully if the app helper is unavailable.
    """
    try:
        # Attempt to call the app-level helper if it exists in the module scope
        from app import _get_plotly_theme as app_theme  # type: ignore
        return app_theme(theme)
    except ImportError:
        pass

    is_dark = theme == "dark"
    bg = "#0E1117" if is_dark else "#FFFFFF"
    paper = "#161B22" if is_dark else "#F8F9FA"
    font_color = "#FAFAFA" if is_dark else "#1A1A2E"
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"

    return dict(
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor=bg,
        paper_bgcolor=paper,
        font=dict(color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True),
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )


def _render_styled_table(df: pd.DataFrame, key: str = "") -> None:
    """
    Render a styled DataFrame table.

    Uses the app helper if available; falls back to st.dataframe.
    """
    try:
        from app import render_styled_table  # type: ignore
        render_styled_table(df, key=key)
        return
    except ImportError:
        pass
    st.dataframe(df, use_container_width=True)


def _show_error(msg: str) -> None:
    """Display an error using the app helper or st.error."""
    try:
        from app import show_error  # type: ignore
        show_error(msg)
        return
    except ImportError:
        pass
    st.error(msg)


def _metrics_to_display_df(metrics: dict) -> pd.DataFrame:
    """Format a metrics dict into a two-column display DataFrame."""
    rows = []
    formatters = {
        "Total Return": _fmt_pct,
        "CAGR": _fmt_pct,
        "Annualized Volatility": _fmt_pct,
        "Sharpe Ratio": _fmt_ratio,
        "Sortino Ratio": _fmt_ratio,
        "Calmar Ratio": _fmt_ratio,
        "Max Drawdown": _fmt_pct,
        "Max Drawdown Duration (days)": _fmt_days,
        "Win Rate": _fmt_pct,
        "Avg Win": _fmt_pct,
        "Avg Loss": _fmt_pct,
        "Profit Factor": _fmt_ratio,
        "Best Day": _fmt_pct,
        "Worst Day": _fmt_pct,
        "VaR 95%": _fmt_pct,
        "CVaR 95%": _fmt_pct,
    }
    for k, v in metrics.items():
        fmt = formatters.get(k, str)
        try:
            rows.append({"Metric": k, "Value": fmt(v)})
        except Exception:
            rows.append({"Metric": k, "Value": str(v)})
    return pd.DataFrame(rows)


def _comparison_df(comparison: dict) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame from run_benchmark_comparison."""
    all_metrics_keys = [
        "Total Return", "CAGR", "Annualized Volatility",
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Max Drawdown", "Max Drawdown Duration (days)",
        "Win Rate", "Profit Factor", "Best Day", "Worst Day",
        "VaR 95%", "CVaR 95%",
    ]
    formatters = {
        "Total Return": _fmt_pct,
        "CAGR": _fmt_pct,
        "Annualized Volatility": _fmt_pct,
        "Sharpe Ratio": _fmt_ratio,
        "Sortino Ratio": _fmt_ratio,
        "Calmar Ratio": _fmt_ratio,
        "Max Drawdown": _fmt_pct,
        "Max Drawdown Duration (days)": _fmt_days,
        "Win Rate": _fmt_pct,
        "Profit Factor": _fmt_ratio,
        "Best Day": _fmt_pct,
        "Worst Day": _fmt_pct,
        "VaR 95%": _fmt_pct,
        "CVaR 95%": _fmt_pct,
    }
    data = {"Metric": all_metrics_keys}
    for strat_name, m in comparison.items():
        col = []
        for k in all_metrics_keys:
            v = m.get(k, None)
            if v is None:
                col.append("—")
            else:
                fmt = formatters.get(k, str)
                try:
                    col.append(fmt(v))
                except Exception:
                    col.append(str(v))
        data[strat_name] = col
    return pd.DataFrame(data)


def render_backtesting_tab(data: dict) -> None:
    """
    Render the full Backtesting tab UI in Streamlit.

    Parameters
    ----------
    data : dict
        Must contain 'prices' (pd.DataFrame with DatetimeIndex) and optionally
        'rf_rate' (float), 'bubble_scores' (dict).
    """
    theme = st.session_state.get("theme", "light")

    st.markdown("## Backtesting Engine")
    st.markdown(
        "Walk-forward event-driven backtester with four optimisation strategies. "
        "No look-ahead bias — each rebalance uses only prior price history."
    )

    # ---- input validation ----
    prices: pd.DataFrame = data.get("prices", pd.DataFrame())
    if prices is None or prices.empty:
        _show_error("No price data available. Please load a portfolio first.")
        return

    rf_rate: float = float(data.get("rf_rate", 0.045))
    bubble_scores: Optional[dict] = data.get("bubble_scores", None)

    # ---- sidebar / control panel ----
    st.markdown("### Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        strategy = st.selectbox(
            "Strategy",
            options=_STRATEGIES,
            index=0,
            key="bt_strategy",
            help="Portfolio optimisation method applied at each rebalance date.",
        )
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            options=_FREQUENCIES,
            index=0,
            key="bt_freq",
            help="How often the portfolio is rebalanced.",
        )

    with col2:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=10_000,
            max_value=10_000_000,
            value=100_000,
            step=10_000,
            key="bt_capital",
            help="Starting portfolio value in USD.",
        )
        transaction_cost = st.slider(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f%%",
            key="bt_tc",
            help="Round-trip cost per trade as a percentage.",
        ) / 100.0

    with col3:
        st.markdown("**Selected Tickers**")
        tickers_list = prices.columns.tolist()
        st.markdown(
            ", ".join(f"`{t}`" for t in tickers_list[:20])
            + (" ..." if len(tickers_list) > 20 else "")
        )
        st.markdown(f"**Date Range:** {prices.index[0].date()} → {prices.index[-1].date()}")
        st.markdown(f"**Trading Days:** {len(prices):,}")

    st.markdown("---")
    run_btn = st.button(
        "▶ Run Backtest",
        key="bt_run_btn",
        type="primary",
        use_container_width=False,
    )

    # ---- session-state key for storing results ----
    bt_key = "bt_results"
    bh_key = "bt_bh_results"
    cmp_key = "bt_comparison_results"

    if run_btn:
        with st.spinner("Running backtest…"):
            try:
                engine = BacktestEngine(prices, rf_rate=rf_rate)

                # Primary strategy
                result = engine.run(
                    strategy=strategy,
                    rebalance_freq=rebalance_freq,
                    initial_capital=float(initial_capital),
                    transaction_cost=transaction_cost,
                    bubble_scores=bubble_scores,
                )
                st.session_state[bt_key] = result
                st.session_state["bt_strategy_label"] = strategy
                st.session_state["bt_freq_label"] = rebalance_freq

                # Buy-and-hold benchmark (equal weight, no rebalance)
                bh_result = engine.run(
                    strategy="Equal Weight",
                    rebalance_freq="Buy & Hold",
                    initial_capital=float(initial_capital),
                    transaction_cost=transaction_cost,
                )
                st.session_state[bh_key] = bh_result

                # Strategy comparison
                with st.spinner("Running strategy comparison…"):
                    comparison = run_benchmark_comparison(
                        prices,
                        rf_rate=rf_rate,
                        initial_capital=float(initial_capital),
                        transaction_cost=transaction_cost,
                    )
                    st.session_state[cmp_key] = comparison

                st.success("Backtest complete.")
            except Exception as exc:
                _show_error(f"Backtest failed: {exc}")
                return

    # ---- display stored results ----
    if bt_key not in st.session_state:
        st.info("Configure and run the backtest above to see results.")
        return

    result = st.session_state[bt_key]
    bh_result = st.session_state.get(bh_key, None)
    comparison = st.session_state.get(cmp_key, None)
    strategy_label = st.session_state.get("bt_strategy_label", strategy)
    freq_label = st.session_state.get("bt_freq_label", rebalance_freq)

    equity_curve: pd.Series = result["equity_curve"]
    daily_returns: pd.Series = result["returns"]
    metrics: dict = result["metrics"]
    trades: list = result["trades"]

    bh_equity: Optional[pd.Series] = (
        bh_result["equity_curve"] if bh_result else None
    )

    # ---- KPI cards ----
    st.markdown(f"### Results — {strategy_label} ({freq_label})")

    kpi_keys = ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Sortino Ratio"]
    kpi_cols = st.columns(len(kpi_keys))
    kpi_fmts = {
        "Total Return": _fmt_pct,
        "CAGR": _fmt_pct,
        "Sharpe Ratio": _fmt_ratio,
        "Max Drawdown": _fmt_pct,
        "Win Rate": _fmt_pct,
        "Sortino Ratio": _fmt_ratio,
    }
    kpi_labels = {
        "Total Return": "Total Return",
        "CAGR": "CAGR",
        "Sharpe Ratio": "Sharpe",
        "Max Drawdown": "Max Drawdown",
        "Win Rate": "Win Rate",
        "Sortino Ratio": "Sortino",
    }
    for col, key in zip(kpi_cols, kpi_keys):
        val = metrics.get(key, None)
        display_val = kpi_fmts[key](val) if val is not None else "—"
        col.metric(kpi_labels[key], display_val)

    st.markdown("---")

    # ---- Equity curve chart ----
    st.markdown("#### Equity Curve")
    fig_equity = _build_equity_chart(equity_curve, bh_equity, theme)
    st.plotly_chart(fig_equity, use_container_width=True)

    # ---- Drawdown chart ----
    st.markdown("#### Drawdown")
    fig_dd = _build_drawdown_chart(equity_curve, theme)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ---- Rolling Sharpe chart ----
    if len(daily_returns) >= TRADING_DAYS + 5:
        st.markdown("#### Rolling 252-Day Sharpe Ratio")
        fig_sharpe = _build_rolling_sharpe_chart(daily_returns, rf_rate, theme)
        st.plotly_chart(fig_sharpe, use_container_width=True)
    else:
        st.info(
            "At least 253 trading days of returns are needed to display the "
            "rolling Sharpe chart."
        )

    st.markdown("---")

    # ---- Performance metrics table ----
    st.markdown("#### Performance Metrics")
    metrics_df = _metrics_to_display_df(metrics)
    _render_styled_table(metrics_df, key="bt_metrics_table")

    st.markdown("---")

    # ---- Trade log ----
    st.markdown("#### Trade Log")
    if trades:
        trades_df = pd.DataFrame(trades)
        # Show summary stats above the table
        n_buys = int((trades_df["action"] == "BUY").sum())
        n_sells = int((trades_df["action"] == "SELL").sum())
        total_traded = float(trades_df["value"].sum())
        t1, t2, t3 = st.columns(3)
        t1.metric("Total Trades", f"{len(trades_df):,}")
        t2.metric("Buys / Sells", f"{n_buys} / {n_sells}")
        t3.metric("Total Value Traded", f"${total_traded:,.0f}")

        display_trades = trades_df[["date", "ticker", "action", "shares", "price", "value"]].copy()
        display_trades.columns = ["Date", "Ticker", "Action", "Shares", "Price ($)", "Value ($)"]
        _render_styled_table(display_trades, key="bt_trade_log")
    else:
        st.info("No trades were executed during this backtest.")

    # ---- Strategy comparison table ----
    if comparison:
        st.markdown("---")
        st.markdown("#### Strategy Comparison")
        st.caption("All strategies run with Monthly rebalancing on the same date range.")
        cmp_df = _comparison_df(comparison)
        _render_styled_table(cmp_df, key="bt_comparison_table")

        # Visual comparison bar chart (key metrics)
        st.markdown("##### Key Metrics Comparison")
        bar_metrics = ["CAGR", "Sharpe Ratio", "Max Drawdown", "Annualized Volatility"]
        plot_theme = _get_plotly_theme(theme)
        colors = ["#00D4FF", "#7C3AED", "#FF6B6B", "#00C49A", "#F59E0B"]
        fig_bar = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=bar_metrics,
        )
        strat_names = list(comparison.keys())
        for i, metric_name in enumerate(bar_metrics):
            row = (i // 2) + 1
            col_idx = (i % 2) + 1
            vals = []
            names = []
            for j, s in enumerate(strat_names):
                v = comparison[s].get(metric_name, None)
                if v is not None:
                    # For display scale pct values as %-points
                    if metric_name in ("CAGR", "Max Drawdown", "Annualized Volatility"):
                        vals.append(v * 100)
                    else:
                        vals.append(v)
                    names.append(s)
            fig_bar.add_trace(
                go.Bar(
                    x=names,
                    y=vals,
                    marker_color=[colors[k % len(colors)] for k in range(len(names))],
                    showlegend=False,
                    hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=col_idx,
            )
        fig_bar.update_layout(
            **{k: v for k, v in plot_theme.items() if k not in ("xaxis", "yaxis")},
            title_text="Strategy Comparison — Key Metrics",
            height=500,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.caption(
        "Disclaimer: Past performance does not guarantee future results. "
        "This tool is for educational and research purposes only."
    )


# ─── FUNDAMENTALS ───
"""
module_fundamentals.py
======================
Fundamental Data Panel module for QuantLab.

Provides:
  - FundamentalDataFetcher: Fetch and process fundamental data via yfinance.
  - calculate_roic: NOPAT / Invested Capital.
  - calculate_fcf_yield: FCF / Market Cap.
  - calculate_ev_ebitda: (Market Cap + Debt - Cash) / EBITDA.
  - render_fundamentals_tab: Full Streamlit UI for the Fundamentals tab.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLORS = {
    "primary": "#00D4FF",
    "secondary": "#7C3AED",
    "positive": "#00C49A",
    "negative": "#FF6B6B",
    "warning": "#F59E0B",
    "neutral": "#94A3B8",
}

# Ratio thresholds: (metric_key, good_if_below, threshold, good_if_above, threshold2)
# Each entry: (key, direction, threshold)
# direction = "below" means green when value < threshold
# direction = "above" means green when value > threshold
_RATIO_THRESHOLDS: dict[str, tuple[str, float]] = {
    "P/E": ("below", 25.0),
    "P/B": ("below", 3.0),
    "EV/EBITDA": ("below", 15.0),
    "ROE": ("above", 15.0),
    "ROIC": ("above", 10.0),
    "D/E": ("below", 1.5),
    "FCF Yield": ("above", 3.0),
    "Gross Margin": ("above", 30.0),
    "Net Margin": ("above", 10.0),
    "Current Ratio": ("above", 1.5),
}


# ---------------------------------------------------------------------------
# Helper – theme / table / error wrappers (same pattern as module_backtest.py)
# ---------------------------------------------------------------------------


def _get_plotly_theme(theme: str) -> dict:
    """
    Return Plotly layout kwargs matching the app theme.

    Falls back gracefully if the app helper is unavailable.
    """
    try:
        from app import _get_plotly_theme as app_theme  # type: ignore
        return app_theme(theme)
    except ImportError:
        pass

    is_dark = theme == "dark"
    bg = "#0E1117" if is_dark else "#FFFFFF"
    paper = "#161B22" if is_dark else "#F8F9FA"
    font_color = "#FAFAFA" if is_dark else "#1A1A2E"
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"

    return dict(
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor=bg,
        paper_bgcolor=paper,
        font=dict(color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True),
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )


def _render_styled_table(df: pd.DataFrame, key: str = "") -> None:
    """
    Render a styled DataFrame table.

    Uses the app helper if available; falls back to st.dataframe.
    """
    try:
        from app import render_styled_table  # type: ignore
        render_styled_table(df, key=key)
        return
    except ImportError:
        pass
    st.dataframe(df, use_container_width=True)


def _show_error(msg: str) -> None:
    """Display an error using the app helper or st.error."""
    try:
        from app import show_error  # type: ignore
        show_error(msg)
        return
    except ImportError:
        pass
    st.error(msg)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _safe_get(d: dict, *keys, default=np.nan):
    """Safely retrieve a value from a nested dict."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, {})
    return d if d not in (None, {}) else default


def _safe_val(series_or_val, default=np.nan):
    """Return a scalar from a Series, array, or scalar safely."""
    if series_or_val is None:
        return default
    if isinstance(series_or_val, pd.Series):
        if series_or_val.empty:
            return default
        v = series_or_val.dropna()
        return float(v.iloc[0]) if not v.empty else default
    if isinstance(series_or_val, (int, float)):
        return float(series_or_val) if np.isfinite(series_or_val) else default
    try:
        v = float(series_or_val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _fmt_large(val: float) -> str:
    """Format large financial values with B/M/K suffixes."""
    if np.isnan(val) or not np.isfinite(val):
        return "N/A"
    abs_val = abs(val)
    if abs_val >= 1e12:
        return f"${val / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"${val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.0f}"


def _fmt_pct(val: float, decimals: int = 1) -> str:
    if np.isnan(val) or not np.isfinite(val):
        return "N/A"
    return f"{val:.{decimals}f}%"


def _fmt_ratio(val: float, decimals: int = 2) -> str:
    if np.isnan(val) or not np.isfinite(val):
        return "N/A"
    return f"{val:.{decimals}f}x"


def _extract_row(df: pd.DataFrame, *candidates) -> pd.Series:
    """
    Try multiple candidate index labels and return the first match as a Series.
    Returns empty Series if none found.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    idx = df.index.astype(str).str.lower()
    for cand in candidates:
        matches = [i for i, label in enumerate(idx) if cand.lower() in label]
        if matches:
            return pd.to_numeric(df.iloc[matches[0]], errors="coerce")
    return pd.Series(dtype=float)


def _annual_columns(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """
    Return the last *n* annual columns from a yfinance statement DataFrame.
    yfinance returns columns as Timestamps; we keep the most recent *n*.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance annual DataFrames: columns are year-end Timestamps, newest first
    cols = df.columns[:n]
    sub = df[cols].copy()
    # Rename columns to year strings
    sub.columns = [str(c.year) if hasattr(c, "year") else str(c) for c in sub.columns]
    return sub


# ---------------------------------------------------------------------------
# Key ratio calculation helpers
# ---------------------------------------------------------------------------


def calculate_roic(ticker_obj: yf.Ticker) -> float:
    """
    ROIC = NOPAT / Invested Capital

    NOPAT  = EBIT × (1 - effective_tax_rate)
    Invested Capital = Total Equity + Total Debt - Cash & Equivalents
    """
    try:
        info = ticker_obj.info or {}
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt

        # EBIT
        ebit = _safe_val(_extract_row(inc, "ebit", "operating income").iloc[0]
                         if not _extract_row(inc, "ebit", "operating income").empty else np.nan)

        # Tax rate
        pretax = _safe_val(_extract_row(inc, "pretax income", "income before tax").iloc[0]
                           if not _extract_row(inc, "pretax income", "income before tax").empty else np.nan)
        tax_provision = _safe_val(_extract_row(inc, "tax provision", "income tax expense").iloc[0]
                                  if not _extract_row(inc, "tax provision", "income tax expense").empty else np.nan)
        if pretax and not np.isnan(pretax) and pretax != 0 and not np.isnan(tax_provision):
            tax_rate = max(0.0, min(0.5, tax_provision / pretax))
        else:
            tax_rate = 0.21  # default US corporate rate

        nopat = ebit * (1 - tax_rate) if not np.isnan(ebit) else np.nan

        # Invested Capital
        equity = _safe_val(_extract_row(bs, "stockholders equity", "total equity", "common stock equity").iloc[0]
                           if not _extract_row(bs, "stockholders equity", "total equity", "common stock equity").empty else np.nan)
        total_debt = _safe_val(_extract_row(bs, "total debt", "long term debt", "current debt").iloc[0]
                               if not _extract_row(bs, "total debt", "long term debt", "current debt").empty else np.nan)
        cash = _safe_val(_extract_row(bs, "cash and cash equivalents", "cash equivalents", "cash").iloc[0]
                         if not _extract_row(bs, "cash and cash equivalents", "cash equivalents", "cash").empty else np.nan)

        if np.isnan(equity):
            equity = 0.0
        if np.isnan(total_debt):
            total_debt = 0.0
        if np.isnan(cash):
            cash = 0.0

        invested_capital = equity + total_debt - cash
        if np.isnan(nopat) or invested_capital <= 0:
            return np.nan

        return (nopat / invested_capital) * 100.0
    except Exception:
        return np.nan


def calculate_fcf_yield(ticker_obj: yf.Ticker) -> float:
    """
    FCF Yield = (Operating Cash Flow - CapEx) / Market Cap × 100
    """
    try:
        info = ticker_obj.info or {}
        market_cap = _safe_val(info.get("marketCap"))
        if np.isnan(market_cap) or market_cap <= 0:
            return np.nan

        cf = ticker_obj.cashflow
        op_cf = _safe_val(_extract_row(cf, "operating cash flow", "cash from operations",
                                       "total cash from operating activities").iloc[0]
                          if not _extract_row(cf, "operating cash flow", "cash from operations",
                                              "total cash from operating activities").empty else np.nan)
        capex = _safe_val(_extract_row(cf, "capital expenditure", "capex",
                                       "purchase of property plant and equipment").iloc[0]
                          if not _extract_row(cf, "capital expenditure", "capex",
                                              "purchase of property plant and equipment").empty else np.nan)
        if np.isnan(op_cf):
            return np.nan

        # CapEx is typically negative in yfinance; FCF = op_cf + capex (capex already negative)
        capex_val = capex if not np.isnan(capex) else 0.0
        fcf = op_cf + capex_val  # capex is negative → subtraction
        return (fcf / market_cap) * 100.0
    except Exception:
        return np.nan


def calculate_ev_ebitda(ticker_obj: yf.Ticker) -> float:
    """
    EV/EBITDA = (Market Cap + Total Debt - Cash) / EBITDA
    """
    try:
        info = ticker_obj.info or {}
        market_cap = _safe_val(info.get("marketCap"))

        bs = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt

        total_debt = _safe_val(_extract_row(bs, "total debt", "long term debt").iloc[0]
                               if not _extract_row(bs, "total debt", "long term debt").empty else np.nan)
        cash = _safe_val(_extract_row(bs, "cash and cash equivalents", "cash equivalents", "cash").iloc[0]
                         if not _extract_row(bs, "cash and cash equivalents", "cash equivalents", "cash").empty else np.nan)

        ebitda_row = _extract_row(inc, "ebitda", "normalized ebitda")
        if ebitda_row.empty:
            # Compute EBITDA = EBIT + D&A
            ebit_row = _extract_row(inc, "ebit", "operating income")
            da_row = _extract_row(inc, "depreciation", "depreciation amortization")
            ebit_val = _safe_val(ebit_row.iloc[0]) if not ebit_row.empty else np.nan
            da_val = _safe_val(da_row.iloc[0]) if not da_row.empty else np.nan
            # D&A from cash flow statement
            if np.isnan(da_val):
                cf = ticker_obj.cashflow
                da_row2 = _extract_row(cf, "depreciation", "depreciation amortization",
                                       "depreciation and amortization")
                da_val = _safe_val(da_row2.iloc[0]) if not da_row2.empty else 0.0
            ebitda = (ebit_val if not np.isnan(ebit_val) else 0.0) + abs(da_val if not np.isnan(da_val) else 0.0)
        else:
            ebitda = _safe_val(ebitda_row.iloc[0])

        if np.isnan(market_cap) or market_cap <= 0 or np.isnan(ebitda) or ebitda <= 0:
            return np.nan

        debt_val = total_debt if not np.isnan(total_debt) else 0.0
        cash_val = cash if not np.isnan(cash) else 0.0
        ev = market_cap + debt_val - cash_val
        return ev / ebitda
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# 1. FundamentalDataFetcher
# ---------------------------------------------------------------------------


class FundamentalDataFetcher:
    """Fetch and process fundamental data from yfinance."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_financials(ticker: str) -> dict:
        """
        Fetch fundamental data for a single ticker via yfinance.

        Returns
        -------
        dict with keys:
          - 'income_stmt'    : pd.DataFrame
          - 'balance_sheet'  : pd.DataFrame
          - 'cash_flow'      : pd.DataFrame
          - 'key_ratios'     : dict
          - 'earnings_history': pd.DataFrame
          - 'info'           : dict
          - 'error'          : str | None
        """
        result: dict = {
            "income_stmt": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
            "key_ratios": {},
            "earnings_history": pd.DataFrame(),
            "info": {},
            "error": None,
        }
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            result["info"] = {
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": _safe_val(info.get("marketCap")),
                "employees": info.get("fullTimeEmployees", "N/A"),
                "description": info.get("longBusinessSummary", ""),
                "name": info.get("longName", info.get("shortName", ticker)),
                "website": info.get("website", ""),
                "currency": info.get("financialCurrency", "USD"),
            }

            # ----------------------------------------------------------------
            # Income Statement
            # ----------------------------------------------------------------
            inc_raw = t.income_stmt
            inc_4yr = _annual_columns(inc_raw, 4)
            if not inc_4yr.empty:
                revenue = _extract_row(inc_raw, "total revenue", "revenue", "net revenue")
                gross_profit = _extract_row(inc_raw, "gross profit")
                ebit = _extract_row(inc_raw, "ebit", "operating income")
                net_income = _extract_row(inc_raw, "net income", "net income common stockholders")
                basic_eps = _extract_row(inc_raw, "basic eps", "diluted eps", "eps")

                rows = {
                    "Revenue": revenue,
                    "Gross Profit": gross_profit,
                    "EBIT": ebit,
                    "Net Income": net_income,
                    "EPS": basic_eps,
                }
                inc_df = pd.DataFrame(rows).T
                # Align to annual columns
                inc_df.columns = inc_df.columns.astype(str)
                cols_4 = [str(c.year) if hasattr(c, "year") else str(c)
                          for c in list(inc_raw.columns[:4])]
                try:
                    inc_df.columns = cols_4
                except ValueError:
                    pass
                result["income_stmt"] = inc_df.apply(pd.to_numeric, errors="coerce")

            # ----------------------------------------------------------------
            # Balance Sheet
            # ----------------------------------------------------------------
            bs_raw = t.balance_sheet
            if bs_raw is not None and not bs_raw.empty:
                total_assets = _extract_row(bs_raw, "total assets")
                total_debt = _extract_row(bs_raw, "total debt", "long term debt")
                equity = _extract_row(bs_raw, "stockholders equity", "total equity",
                                      "common stock equity")
                cash = _extract_row(bs_raw, "cash and cash equivalents", "cash equivalents",
                                    "cash financial")

                rows_bs = {
                    "Total Assets": total_assets,
                    "Total Debt": total_debt,
                    "Equity": equity,
                    "Cash": cash,
                }
                bs_df = pd.DataFrame(rows_bs).T
                cols_4_bs = [str(c.year) if hasattr(c, "year") else str(c)
                             for c in list(bs_raw.columns[:4])]
                try:
                    bs_df.columns = cols_4_bs
                except ValueError:
                    pass
                result["balance_sheet"] = bs_df.apply(pd.to_numeric, errors="coerce")

            # ----------------------------------------------------------------
            # Cash Flow Statement
            # ----------------------------------------------------------------
            cf_raw = t.cashflow
            if cf_raw is not None and not cf_raw.empty:
                op_cf = _extract_row(cf_raw, "operating cash flow", "total cash from operating",
                                     "cash from operations")
                capex = _extract_row(cf_raw, "capital expenditure", "capex",
                                     "purchase of property plant and equipment")
                dividends = _extract_row(cf_raw, "common stock dividend", "dividends paid",
                                         "payment of dividends", "cash dividends paid")

                # Free Cash Flow = Operating CF + CapEx (capex is negative)
                free_cf = op_cf + capex.reindex(op_cf.index).fillna(0)

                rows_cf = {
                    "Operating CF": op_cf,
                    "CapEx": capex,
                    "Free CF": free_cf,
                    "Dividends": dividends,
                }
                cf_df = pd.DataFrame(rows_cf).T
                cols_4_cf = [str(c.year) if hasattr(c, "year") else str(c)
                             for c in list(cf_raw.columns[:4])]
                try:
                    cf_df.columns = cols_4_cf
                except ValueError:
                    pass
                result["cash_flow"] = cf_df.apply(pd.to_numeric, errors="coerce")

            # ----------------------------------------------------------------
            # Key Ratios
            # ----------------------------------------------------------------
            pe = _safe_val(info.get("trailingPE") or info.get("forwardPE"))
            pb = _safe_val(info.get("priceToBook"))
            roe = _safe_val(info.get("returnOnEquity"))
            if not np.isnan(roe):
                roe = roe * 100.0  # convert fraction to %

            # Margins from info
            gross_margin = _safe_val(info.get("grossMargins"))
            net_margin = _safe_val(info.get("profitMargins"))
            if not np.isnan(gross_margin):
                gross_margin = gross_margin * 100.0
            if not np.isnan(net_margin):
                net_margin = net_margin * 100.0

            # D/E from info or balance sheet
            de = _safe_val(info.get("debtToEquity"))
            if not np.isnan(de):
                de = de / 100.0  # yfinance returns as percentage

            # Current ratio
            current_ratio = _safe_val(info.get("currentRatio"))

            # Calculated ratios
            ev_ebitda = calculate_ev_ebitda(t)
            # Try yfinance first
            if np.isnan(ev_ebitda):
                ev_ebitda = _safe_val(info.get("enterpriseToEbitda"))

            roic = calculate_roic(t)
            fcf_yield = calculate_fcf_yield(t)
            # Try yfinance fallback for EV/EBITDA
            if np.isnan(ev_ebitda):
                ev_ebitda = _safe_val(info.get("enterpriseToEbitda"))

            result["key_ratios"] = {
                "P/E": pe,
                "P/B": pb,
                "EV/EBITDA": ev_ebitda,
                "ROE": roe,
                "ROIC": roic,
                "D/E": de,
                "FCF Yield": fcf_yield,
                "Gross Margin": gross_margin,
                "Net Margin": net_margin,
                "Current Ratio": current_ratio,
            }

            # ----------------------------------------------------------------
            # Earnings History
            # ----------------------------------------------------------------
            try:
                eh = t.earnings_history
                if eh is not None and not eh.empty:
                    eh = eh.reset_index()
                    # Standardize column names
                    col_map = {}
                    for col in eh.columns:
                        cl = col.lower().replace(" ", "_")
                        if "date" in cl or cl == "index":
                            col_map[col] = "Date"
                        elif "estimate" in cl or cl == "epsestimate":
                            col_map[col] = "EPS Estimate"
                        elif "actual" in cl or cl == "epsactual":
                            col_map[col] = "EPS Actual"
                        elif "surprise" in cl and "percent" not in cl:
                            col_map[col] = "Surprise %"
                        elif "surprisepct" in cl or ("surprise" in cl and "pct" in cl):
                            col_map[col] = "Surprise %"
                    eh = eh.rename(columns=col_map)

                    # Compute Surprise % if not present
                    if "EPS Estimate" in eh.columns and "EPS Actual" in eh.columns:
                        if "Surprise %" not in eh.columns:
                            est = pd.to_numeric(eh["EPS Estimate"], errors="coerce")
                            act = pd.to_numeric(eh["EPS Actual"], errors="coerce")
                            # Surprise % = (actual - estimate) / |estimate| * 100
                            with np.errstate(divide="ignore", invalid="ignore"):
                                surprise = np.where(
                                    est.abs() > 0,
                                    ((act - est) / est.abs()) * 100.0,
                                    np.nan,
                                )
                            eh["Surprise %"] = surprise
                        else:
                            # Convert to percentage if stored as fraction
                            sp = pd.to_numeric(eh["Surprise %"], errors="coerce")
                            if sp.abs().median() < 1.0:
                                eh["Surprise %"] = sp * 100.0

                    keep_cols = [c for c in ["Date", "EPS Estimate", "EPS Actual", "Surprise %"]
                                 if c in eh.columns]
                    result["earnings_history"] = eh[keep_cols].dropna(
                        subset=["EPS Estimate", "EPS Actual"]
                    ).reset_index(drop=True)
            except Exception:
                result["earnings_history"] = pd.DataFrame()

        except Exception as exc:
            result["error"] = str(exc)

        return result


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_income_chart(inc_df: pd.DataFrame, theme: str) -> go.Figure:
    """Bar chart: Revenue bars + Net Income line overlay."""
    plot_theme = _get_plotly_theme(theme)
    years = inc_df.columns.tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "Revenue" in inc_df.index:
        rev = inc_df.loc["Revenue"].values / 1e9
        fig.add_trace(
            go.Bar(
                x=years,
                y=rev,
                name="Revenue (B)",
                marker_color=_COLORS["primary"],
                opacity=0.85,
            ),
            secondary_y=False,
        )

    if "Net Income" in inc_df.index:
        ni = inc_df.loc["Net Income"].values / 1e9
        fig.add_trace(
            go.Scatter(
                x=years,
                y=ni,
                name="Net Income (B)",
                mode="lines+markers",
                line=dict(color=_COLORS["positive"], width=2),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )

    if "Gross Profit" in inc_df.index:
        gp = inc_df.loc["Gross Profit"].values / 1e9
        fig.add_trace(
            go.Bar(
                x=years,
                y=gp,
                name="Gross Profit (B)",
                marker_color=_COLORS["secondary"],
                opacity=0.65,
            ),
            secondary_y=False,
        )

    fig.update_layout(
        title="Revenue & Profitability",
        barmode="group",
        **plot_theme,
    )
    fig.update_yaxes(title_text="$ Billions (Revenue / GP)", secondary_y=False)
    fig.update_yaxes(title_text="$ Billions (Net Income)", secondary_y=True)
    return fig


def _build_balance_chart(bs_df: pd.DataFrame, theme: str) -> go.Figure:
    """Stacked bar: Assets vs Liabilities vs Equity."""
    plot_theme = _get_plotly_theme(theme)
    years = bs_df.columns.tolist()

    fig = go.Figure()

    # Total Liabilities = Total Assets - Equity
    assets_vals = bs_df.loc["Total Assets"].values / 1e9 if "Total Assets" in bs_df.index else np.zeros(len(years))
    equity_vals = bs_df.loc["Equity"].values / 1e9 if "Equity" in bs_df.index else np.zeros(len(years))
    debt_vals = bs_df.loc["Total Debt"].values / 1e9 if "Total Debt" in bs_df.index else np.zeros(len(years))
    cash_vals = bs_df.loc["Cash"].values / 1e9 if "Cash" in bs_df.index else np.zeros(len(years))

    # Approximate liabilities from assets - equity
    liabilities_vals = assets_vals - equity_vals

    fig.add_trace(go.Bar(
        x=years, y=equity_vals, name="Equity",
        marker_color=_COLORS["positive"],
    ))
    fig.add_trace(go.Bar(
        x=years, y=liabilities_vals, name="Liabilities",
        marker_color=_COLORS["negative"], opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=years, y=cash_vals, name="Cash (line)",
        mode="lines+markers",
        line=dict(color=_COLORS["primary"], width=2),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title="Balance Sheet Composition",
        barmode="stack",
        **plot_theme,
    )
    fig.update_yaxes(title_text="$ Billions")
    return fig


def _build_cashflow_chart(cf_df: pd.DataFrame, theme: str) -> go.Figure:
    """Waterfall-style grouped bar: Operating CF, CapEx, Free CF."""
    plot_theme = _get_plotly_theme(theme)
    years = cf_df.columns.tolist()

    fig = go.Figure()

    if "Operating CF" in cf_df.index:
        op_cf = cf_df.loc["Operating CF"].values / 1e9
        fig.add_trace(go.Bar(
            x=years, y=op_cf, name="Operating CF",
            marker_color=_COLORS["primary"],
        ))

    if "CapEx" in cf_df.index:
        capex = cf_df.loc["CapEx"].values / 1e9
        fig.add_trace(go.Bar(
            x=years, y=capex, name="CapEx",
            marker_color=_COLORS["negative"], opacity=0.8,
        ))

    if "Free CF" in cf_df.index:
        fcf = cf_df.loc["Free CF"].values / 1e9
        colors_fcf = [_COLORS["positive"] if v >= 0 else _COLORS["negative"] for v in fcf]
        fig.add_trace(go.Bar(
            x=years, y=fcf, name="Free CF",
            marker_color=colors_fcf, opacity=0.9,
        ))

    fig.update_layout(
        title="Cash Flow Waterfall",
        barmode="group",
        **plot_theme,
    )
    fig.update_yaxes(title_text="$ Billions")
    return fig


def _build_earnings_chart(eh: pd.DataFrame, theme: str) -> go.Figure:
    """EPS Actual vs Estimate bar chart, colored by beat/miss."""
    plot_theme = _get_plotly_theme(theme)

    dates = eh["Date"].astype(str).tolist() if "Date" in eh.columns else list(range(len(eh)))
    estimates = pd.to_numeric(eh.get("EPS Estimate", pd.Series()), errors="coerce").tolist()
    actuals = pd.to_numeric(eh.get("EPS Actual", pd.Series()), errors="coerce").tolist()

    beat_colors = [
        _COLORS["positive"] if (a is not None and e is not None and not np.isnan(a) and not np.isnan(e) and a >= e)
        else _COLORS["negative"]
        for a, e in zip(actuals, estimates)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=estimates, name="EPS Estimate",
        marker_color=_COLORS["neutral"], opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=dates, y=actuals, name="EPS Actual",
        marker_color=beat_colors, opacity=0.9,
    ))
    fig.update_layout(
        title="EPS: Actual vs Estimate",
        barmode="group",
        **plot_theme,
    )
    fig.update_yaxes(title_text="EPS ($)")
    return fig


def _build_revenue_growth_chart(inc_df: pd.DataFrame, theme: str) -> go.Figure:
    """YoY Revenue growth trend."""
    plot_theme = _get_plotly_theme(theme)

    if "Revenue" not in inc_df.index:
        fig = go.Figure()
        fig.update_layout(title="Revenue Growth (YoY)", **plot_theme)
        return fig

    rev = inc_df.loc["Revenue"].astype(float)
    # columns are sorted newest-first; reverse for chronological
    rev = rev.iloc[::-1]
    growth = rev.pct_change() * 100.0
    growth = growth.dropna()

    colors = [_COLORS["positive"] if v >= 0 else _COLORS["negative"] for v in growth.values]

    fig = go.Figure(go.Bar(
        x=growth.index.tolist(),
        y=growth.values.tolist(),
        marker_color=colors,
        name="YoY Growth %",
    ))
    fig.update_layout(title="Revenue Growth (YoY %)", **plot_theme)
    fig.update_yaxes(title_text="Growth (%)")
    return fig


def _build_margin_trend_chart(inc_df: pd.DataFrame, theme: str) -> go.Figure:
    """Gross margin and net margin trend lines."""
    plot_theme = _get_plotly_theme(theme)

    years = list(reversed(inc_df.columns.tolist()))  # chronological
    fig = go.Figure()

    if "Revenue" in inc_df.index:
        rev = inc_df.loc["Revenue"].astype(float)
        rev_chron = rev.iloc[::-1]

        if "Gross Profit" in inc_df.index:
            gp = inc_df.loc["Gross Profit"].astype(float).iloc[::-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                gm = np.where(rev_chron != 0, gp / rev_chron * 100.0, np.nan)
            fig.add_trace(go.Scatter(
                x=years, y=gm, name="Gross Margin %",
                mode="lines+markers",
                line=dict(color=_COLORS["primary"], width=2),
                marker=dict(size=8),
            ))

        if "Net Income" in inc_df.index:
            ni = inc_df.loc["Net Income"].astype(float).iloc[::-1]
            with np.errstate(divide="ignore", invalid="ignore"):
                nm = np.where(rev_chron != 0, ni / rev_chron * 100.0, np.nan)
            fig.add_trace(go.Scatter(
                x=years, y=nm, name="Net Margin %",
                mode="lines+markers",
                line=dict(color=_COLORS["positive"], width=2, dash="dash"),
                marker=dict(size=8),
            ))

    fig.update_layout(title="Margin Trend", **plot_theme)
    fig.update_yaxes(title_text="Margin (%)")
    return fig


def _build_pe_context_chart(inc_df: pd.DataFrame, ratios: dict, theme: str) -> go.Figure:
    """Current P/E vs trailing average P/E context."""
    plot_theme = _get_plotly_theme(theme)

    current_pe = ratios.get("P/E", np.nan)

    fig = go.Figure()

    # Show current P/E as a single gauge-like bar
    pe_valid = not np.isnan(current_pe) if isinstance(current_pe, float) else current_pe is not None
    if pe_valid:
        fig.add_trace(go.Bar(
            x=["Current P/E"],
            y=[current_pe],
            marker_color=_COLORS["primary"] if current_pe < 25 else _COLORS["warning"],
            name="Current P/E",
            width=0.4,
            text=[f"{current_pe:.1f}x"],
            textposition="outside",
        ))
        # Add reference lines
        fig.add_hline(y=25, line_dash="dash", line_color=_COLORS["warning"],
                      annotation_text="Fair Value (~25x)", opacity=0.7)
        fig.add_hline(y=15, line_dash="dot", line_color=_COLORS["positive"],
                      annotation_text="Value Zone (~15x)", opacity=0.7)

    fig.update_layout(title="Valuation Context — P/E", **plot_theme)
    fig.update_yaxes(title_text="P/E Ratio")
    return fig


# ---------------------------------------------------------------------------
# Ratio color helper
# ---------------------------------------------------------------------------


def _ratio_color(label: str, value: float) -> str:
    """Return green/red/neutral CSS color string based on ratio threshold."""
    if np.isnan(value) or not np.isfinite(value):
        return _COLORS["neutral"]
    config = _RATIO_THRESHOLDS.get(label)
    if not config:
        return _COLORS["neutral"]
    direction, threshold = config
    if direction == "below":
        return _COLORS["positive"] if value < threshold else _COLORS["negative"]
    else:  # above
        return _COLORS["positive"] if value > threshold else _COLORS["negative"]


def _format_ratio_value(label: str, value: float) -> str:
    """Format a ratio value for display."""
    if np.isnan(value) or not np.isfinite(value):
        return "N/A"
    pct_metrics = {"ROE", "ROIC", "FCF Yield", "Gross Margin", "Net Margin"}
    if label in pct_metrics:
        return f"{value:.1f}%"
    if label == "Current Ratio":
        return f"{value:.2f}x"
    return f"{value:.2f}x"


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------


def _format_stmt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format financial statement DataFrame for display (values in billions)."""
    if df is None or df.empty:
        return pd.DataFrame()
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].apply(
            lambda v: _fmt_large(float(v)) if pd.notna(v) else "N/A"
        )
    return display


# ---------------------------------------------------------------------------
# 3. render_fundamentals_tab
# ---------------------------------------------------------------------------


def render_fundamentals_tab(data: dict) -> None:
    """
    Render the full Fundamental Data Panel UI in Streamlit.

    Parameters
    ----------
    data : dict
        Must contain 'tickers' (list of ticker strings).
        Optional: 'theme' string override (falls back to session state).
    """
    theme = data.get("theme", st.session_state.get("theme", "light"))

    st.markdown("## Fundamental Data Panel")
    st.markdown(
        "Company financials, key ratios, earnings history, and valuation context "
        "sourced from public filings via yfinance."
    )

    # ---- Ticker selector ----
    tickers: list[str] = data.get("tickers", [])
    if not tickers:
        _show_error("No tickers available. Please load a portfolio first.")
        return

    selected_ticker = st.selectbox(
        "Select Ticker",
        options=tickers,
        key="fund_ticker_select",
        help="Choose a ticker to view its fundamental data.",
    )
    if not selected_ticker:
        return

    # ---- Fetch data ----
    with st.spinner(f"Loading fundamental data for {selected_ticker}…"):
        fin = FundamentalDataFetcher.get_financials(selected_ticker)

    if fin.get("error"):
        _show_error(f"calculation_error: {fin['error']}")
        return

    info = fin.get("info", {})
    ratios = fin.get("key_ratios", {})
    inc_df = fin.get("income_stmt", pd.DataFrame())
    bs_df = fin.get("balance_sheet", pd.DataFrame())
    cf_df = fin.get("cash_flow", pd.DataFrame())
    eh = fin.get("earnings_history", pd.DataFrame())

    # ---- Company Header ----
    st.markdown("---")
    company_name = info.get("name", selected_ticker)
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    mkt_cap = info.get("market_cap", np.nan)
    employees = info.get("employees", "N/A")

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("Company", company_name)
    col_h2.metric("Sector", sector)
    col_h3.metric("Market Cap", _fmt_large(mkt_cap) if not np.isnan(mkt_cap) else "N/A")
    col_h4.metric("Employees", f"{employees:,}" if isinstance(employees, int) else str(employees))
    st.caption(f"Industry: {industry}")

    description = info.get("description", "")
    if description:
        with st.expander("Company Description"):
            st.write(description)

    # ====================================================================
    # Section 1 — Key Ratios
    # ====================================================================
    st.markdown("### Key Ratios")

    ratio_labels = ["P/E", "P/B", "EV/EBITDA", "ROE", "ROIC",
                    "D/E", "FCF Yield", "Gross Margin", "Net Margin", "Current Ratio"]

    cols = st.columns(5)
    for i, label in enumerate(ratio_labels):
        val = ratios.get(label, np.nan)
        if isinstance(val, float) and (np.isnan(val) or not np.isfinite(val)):
            display_val = "N/A"
            delta_str = None
            delta_color = "off"
        else:
            val_f = float(val) if val is not None else np.nan
            display_val = _format_ratio_value(label, val_f)
            color = _ratio_color(label, val_f)
            # Use delta to encode color: positive delta = green, negative = red
            is_good = color == _COLORS["positive"]
            delta_str = "✓ Favorable" if is_good else "✗ Elevated"
            delta_color = "normal" if is_good else "inverse"

        with cols[i % 5]:
            st.metric(
                label=label,
                value=display_val,
                delta=delta_str,
                delta_color=delta_color,
            )

    # ====================================================================
    # Section 2 — Financial Statements
    # ====================================================================
    st.markdown("---")
    st.markdown("### Financial Statements")

    fs_tab1, fs_tab2, fs_tab3 = st.tabs(
        ["Income Statement", "Balance Sheet", "Cash Flow"]
    )

    with fs_tab1:
        if inc_df.empty:
            st.info("Income statement data not available.")
        else:
            fig_inc = _build_income_chart(inc_df, theme)
            st.plotly_chart(fig_inc, use_container_width=True, key="fund_inc_chart")
            st.markdown("**Annual Income Statement** (USD)")
            _render_styled_table(_format_stmt_df(inc_df), key="fund_inc_table")

    with fs_tab2:
        if bs_df.empty:
            st.info("Balance sheet data not available.")
        else:
            fig_bs = _build_balance_chart(bs_df, theme)
            st.plotly_chart(fig_bs, use_container_width=True, key="fund_bs_chart")
            st.markdown("**Annual Balance Sheet** (USD)")
            _render_styled_table(_format_stmt_df(bs_df), key="fund_bs_table")

    with fs_tab3:
        if cf_df.empty:
            st.info("Cash flow data not available.")
        else:
            fig_cf = _build_cashflow_chart(cf_df, theme)
            st.plotly_chart(fig_cf, use_container_width=True, key="fund_cf_chart")
            st.markdown("**Annual Cash Flow Statement** (USD)")
            _render_styled_table(_format_stmt_df(cf_df), key="fund_cf_table")

    # ====================================================================
    # Section 3 — Earnings History
    # ====================================================================
    st.markdown("---")
    st.markdown("### Earnings History")

    if eh.empty:
        st.info("Earnings history data not available.")
    else:
        # Beat rate
        if "EPS Estimate" in eh.columns and "EPS Actual" in eh.columns:
            est = pd.to_numeric(eh["EPS Estimate"], errors="coerce")
            act = pd.to_numeric(eh["EPS Actual"], errors="coerce")
            beats = (act >= est).sum()
            total_valid = (~est.isna() & ~act.isna()).sum()
            beat_rate = (beats / total_valid * 100.0) if total_valid > 0 else np.nan
        else:
            beat_rate = np.nan

        col_er1, col_er2 = st.columns([3, 1])

        with col_er1:
            fig_eh = _build_earnings_chart(eh, theme)
            st.plotly_chart(fig_eh, use_container_width=True, key="fund_eh_chart")

        with col_er2:
            st.markdown("##### Beat Rate")
            if not np.isnan(beat_rate):
                color_class = "normal" if beat_rate >= 70 else "inverse"
                st.metric(
                    "EPS Beat Rate",
                    f"{beat_rate:.0f}%",
                    delta=f"{beats}/{total_valid} beats",
                    delta_color=color_class,
                )
            else:
                st.metric("EPS Beat Rate", "N/A")

        # Earnings table
        display_eh = eh.copy()
        if "EPS Estimate" in display_eh.columns:
            display_eh["EPS Estimate"] = pd.to_numeric(display_eh["EPS Estimate"],
                                                        errors="coerce").apply(
                lambda v: f"${v:.2f}" if pd.notna(v) else "N/A"
            )
        if "EPS Actual" in display_eh.columns:
            display_eh["EPS Actual"] = pd.to_numeric(display_eh["EPS Actual"],
                                                      errors="coerce").apply(
                lambda v: f"${v:.2f}" if pd.notna(v) else "N/A"
            )
        if "Surprise %" in display_eh.columns:
            display_eh["Surprise %"] = pd.to_numeric(display_eh["Surprise %"],
                                                      errors="coerce").apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"
            )
        _render_styled_table(display_eh, key="fund_eh_table")

    # ====================================================================
    # Section 4 — Valuation Context
    # ====================================================================
    st.markdown("---")
    st.markdown("### Valuation Context")

    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        fig_pe = _build_pe_context_chart(inc_df, ratios, theme)
        st.plotly_chart(fig_pe, use_container_width=True, key="fund_pe_chart")

    with col_v2:
        if not inc_df.empty:
            fig_rev_growth = _build_revenue_growth_chart(inc_df, theme)
            st.plotly_chart(fig_rev_growth, use_container_width=True, key="fund_revgrowth_chart")
        else:
            st.info("Revenue growth data unavailable.")

    with col_v3:
        if not inc_df.empty:
            fig_margins = _build_margin_trend_chart(inc_df, theme)
            st.plotly_chart(fig_margins, use_container_width=True, key="fund_margins_chart")
        else:
            st.info("Margin trend data unavailable.")

    # Summary callout
    st.markdown("---")
    pe_val = ratios.get("P/E", np.nan)
    roe_val = ratios.get("ROE", np.nan)
    fcf_yield_val = ratios.get("FCF Yield", np.nan)
    gm_val = ratios.get("Gross Margin", np.nan)

    summary_parts = []
    if not np.isnan(pe_val) and np.isfinite(pe_val):
        pe_label = "trading at a discount" if pe_val < 15 else ("fairly valued" if pe_val < 25 else "premium-priced")
        summary_parts.append(f"P/E of **{pe_val:.1f}x** ({pe_label})")
    if not np.isnan(roe_val) and np.isfinite(roe_val):
        roe_label = "strong" if roe_val > 15 else "below average"
        summary_parts.append(f"ROE of **{roe_val:.1f}%** ({roe_label})")
    if not np.isnan(fcf_yield_val) and np.isfinite(fcf_yield_val):
        summary_parts.append(f"FCF Yield of **{fcf_yield_val:.1f}%**")
    if not np.isnan(gm_val) and np.isfinite(gm_val):
        summary_parts.append(f"Gross Margin of **{gm_val:.1f}%**")

    if summary_parts:
        st.info(
            f"**{company_name}** ({selected_ticker}) — " + " · ".join(summary_parts) + "."
        )
    else:
        st.info(f"Fundamental snapshot loaded for **{company_name}** ({selected_ticker}).")


# ─── FIXED_INCOME ───
"""
module_fixed_income.py
======================
Fixed Income & Macro Analytics module for QuantLab (Streamlit quantitative finance app).

Provides:
  - BondPricer       : price, macaulay_duration, modified_duration, convexity, dv01, price_change
  - fetch_full_yield_curve()   : current yield curve as DataFrame
  - fetch_historical_curves()  : historical curve snapshots as {label: pd.Series}
  - render_fixed_income_tab(data) : complete Streamlit UI (3 sections)
"""

import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Colour palette (matches the rest of the QuantLab app)
# ---------------------------------------------------------------------------

_COLORS = {
    "primary":   "#00D4FF",
    "secondary": "#7C3AED",
    "positive":  "#00C49A",
    "negative":  "#FF6B6B",
    "warning":   "#F59E0B",
    "neutral":   "#94A3B8",
    "accent1":   "#F472B6",
    "accent2":   "#34D399",
}

# Yield-curve tickers available through yfinance
# ^IRX = 13-week T-Bill (proxy for short end)
# ^FVX = 5-Year Treasury
# ^TNX = 10-Year Treasury
# ^TYX = 30-Year Treasury
_CURVE_SYMBOLS = ["^IRX", "^FVX", "^TNX", "^TYX"]

# Tenor label → (symbol, years)  — 1Y is synthesised via interpolation below
_TENOR_TABLE: list[tuple[str, str, float]] = [
    ("3M",  "^IRX",  0.25),
    ("5Y",  "^FVX",  5.0),
    ("10Y", "^TNX", 10.0),
    ("30Y", "^TYX", 30.0),
]

# ---------------------------------------------------------------------------
# Helper shims – fall back gracefully when run outside the main app
# ---------------------------------------------------------------------------


def _get_plotly_theme(theme: str = "dark") -> dict:
    """Return Plotly layout kwargs matching the app theme."""
    try:
        from app import _get_plotly_theme as _app_theme  # type: ignore
        return _app_theme(theme)
    except ImportError:
        pass
    is_dark = theme == "dark"
    bg          = "#0E1117" if is_dark else "#FFFFFF"
    paper       = "#161B22" if is_dark else "#F8F9FA"
    font_color  = "#FAFAFA" if is_dark else "#1A1A2E"
    grid_color  = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"
    return dict(
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor=bg,
        paper_bgcolor=paper,
        font=dict(color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True),
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )




# ---------------------------------------------------------------------------
# BondPricer
# ---------------------------------------------------------------------------


class BondPricer:
    """
    Static methods for fixed-income analytics.

    All cash-flow arithmetic uses the per-period discount factor
    ``r = ytm / frequency`` and ``t`` in units of coupon periods.
    Duration metrics are returned in *years*.
    """

    @staticmethod
    def price(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """
        Present value of all bond cash flows (dirty/full price).

        Parameters
        ----------
        face_value  : Par value of the bond (e.g. 1000).
        coupon_rate : Annual coupon rate as a decimal (e.g. 0.05 for 5%).
        ytm         : Yield to maturity as a decimal (e.g. 0.04 for 4%).
        periods     : Total number of coupon periods = years_to_maturity * frequency.
        frequency   : Coupon payments per year (2 = semi-annual, 1 = annual).

        Returns
        -------
        float  Full (dirty) price of the bond.

        Formula
        -------
        P = C * [1 - (1+r)^(-n)] / r  +  F * (1+r)^(-n)
        where C = face * coupon_rate / frequency,  r = ytm / frequency,  n = periods.
        """
        if frequency <= 0:
            raise ValueError("frequency must be a positive integer")
        if periods < 1:
            raise ValueError("periods must be >= 1")
        coupon = face_value * coupon_rate / frequency
        r = ytm / frequency
        if abs(r) < 1e-12:
            # Zero-yield edge case: PV is just the sum of undiscounted cash flows
            pv = coupon * periods + face_value
        else:
            pv = (
                coupon * (1.0 - (1.0 + r) ** (-periods)) / r
                + face_value * (1.0 + r) ** (-periods)
            )
        return pv

    @staticmethod
    def macaulay_duration(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """
        Macaulay duration in years.

        Computed as the present-value-weighted average time to receipt of
        each cash flow:

            D_mac = (1/P) * Σ_{t=1}^{n}  (t / frequency) * CF_t / (1 + r)^t

        where r = ytm / frequency.

        Parameters
        ----------
        face_value  : Par value.
        coupon_rate : Annual coupon rate (decimal).
        ytm         : Yield to maturity (decimal).
        periods     : Total coupon periods.
        frequency   : Payments per year.

        Returns
        -------
        float  Macaulay duration in years.
        """
        if frequency <= 0:
            raise ValueError("frequency must be a positive integer")
        coupon = face_value * coupon_rate / frequency
        r = ytm / frequency
        p = BondPricer.price(face_value, coupon_rate, ytm, periods, frequency)
        if p == 0.0:
            return 0.0
        weighted_time = 0.0
        for t in range(1, periods + 1):
            cf = coupon if t < periods else coupon + face_value
            pv_cf = cf / (1.0 + r) ** t if abs(r) > 1e-12 else cf
            # time in years for period t is t / frequency
            weighted_time += (t / frequency) * pv_cf
        return weighted_time / p

    @staticmethod
    def modified_duration(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """
        Modified duration in years.

        Modified duration scales Macaulay duration by the periodic discount
        factor, giving the (negative) percentage price change per unit change
        in yield:

            D_mod = D_mac / (1 + ytm / frequency)

        Parameters
        ----------
        face_value  : Par value.
        coupon_rate : Annual coupon rate (decimal).
        ytm         : Yield to maturity (decimal).
        periods     : Total coupon periods.
        frequency   : Payments per year.

        Returns
        -------
        float  Modified duration in years.
        """
        mac = BondPricer.macaulay_duration(
            face_value, coupon_rate, ytm, periods, frequency
        )
        return mac / (1.0 + ytm / frequency)

    @staticmethod
    def convexity(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """
        Convexity of the bond (in years²).

        Second derivative of price with respect to yield, normalised by price:

            C = [1 / (P * (1+r)^2)] * Σ_{t=1}^{n}  [t*(t+1) * CF_t / (1+r)^t]
                / frequency^2

        The division by frequency² converts from period units to year units.

        Parameters
        ----------
        face_value  : Par value.
        coupon_rate : Annual coupon rate (decimal).
        ytm         : Yield to maturity (decimal).
        periods     : Total coupon periods.
        frequency   : Payments per year.

        Returns
        -------
        float  Convexity in years².
        """
        if frequency <= 0:
            raise ValueError("frequency must be a positive integer")
        coupon = face_value * coupon_rate / frequency
        r = ytm / frequency
        p = BondPricer.price(face_value, coupon_rate, ytm, periods, frequency)
        if p == 0.0:
            return 0.0
        n = periods
        conv_sum = 0.0
        for t in range(1, n + 1):
            cf = coupon if t < n else coupon + face_value
            # t*(t+1)*CF / (1+r)^(t+2)
            if abs(r) < 1e-12:
                conv_sum += t * (t + 1) * cf
            else:
                conv_sum += t * (t + 1) * cf / (1.0 + r) ** (t + 2)
        # Normalise: divide by price and convert period² → year²
        return conv_sum / (p * frequency ** 2)

    @staticmethod
    def dv01(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """
        Dollar Value of 01 (DV01): price change for a 1 basis-point (0.01%)
        increase in yield.

            DV01 = Modified Duration × Price × 0.0001

        A positive number represents the *dollar loss* for a 1-bp rise in rates.

        Parameters
        ----------
        face_value  : Par value.
        coupon_rate : Annual coupon rate (decimal).
        ytm         : Yield to maturity (decimal).
        periods     : Total coupon periods.
        frequency   : Payments per year.

        Returns
        -------
        float  DV01 in dollars (magnitude of price decline per +1bp).
        """
        mod_dur = BondPricer.modified_duration(
            face_value, coupon_rate, ytm, periods, frequency
        )
        p = BondPricer.price(face_value, coupon_rate, ytm, periods, frequency)
        return mod_dur * p * 0.0001

    @staticmethod
    def price_change(
        mod_dur: float,
        convexity: float,
        delta_y: float,
        current_price: float,
    ) -> float:
        """
        Estimate the dollar price change using the second-order Taylor
        (duration + convexity) approximation:

            ΔP ≈ -D_mod * Δy * P  +  0.5 * C * Δy² * P

        Parameters
        ----------
        mod_dur       : Modified duration (years).
        convexity     : Convexity (years²).
        delta_y       : Yield change in decimal (e.g. +0.01 = +100 bps).
        current_price : Current full price of the bond.

        Returns
        -------
        float  Estimated dollar price change (positive = price increase).
        """
        dp = -mod_dur * delta_y * current_price
        dp += 0.5 * convexity * delta_y ** 2 * current_price
        return dp


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _last_close(ticker: str, period: str = "5d") -> Optional[float]:
    """Return the most recent closing value for a yfinance ticker, or None."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return None
        val = hist["Close"].dropna()
        return float(val.iloc[-1]) if not val.empty else None
    except Exception:
        return None


def _close_series(ticker: str, period: str = "2y") -> pd.Series:
    """Return a daily Close series for a yfinance ticker (empty on failure)."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return pd.Series(dtype=float, name=ticker)
        s = hist["Close"].dropna()
        s.name = ticker
        return s
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def fetch_full_yield_curve() -> pd.DataFrame:
    """
    Fetch the current US Treasury yield curve from yfinance.

    Tickers used:
      ^IRX  = 13-week T-Bill  → proxies for 3M and 1Y short end
      ^FVX  = 5-Year Treasury
      ^TNX  = 10-Year Treasury
      ^TYX  = 30-Year Treasury

    A synthetic 1Y point is interpolated from IRX and FVX where available.

    Returns
    -------
    pd.DataFrame with columns:
        tenor_label  : str    e.g. "3M", "1Y", "5Y", "10Y", "30Y"
        tenor_years  : float  e.g. 0.25, 1.0, 5.0, 10.0, 30.0
        yield_pct    : float  annualised yield in percent (e.g. 4.35)
        date         : str    ISO date of the latest observation
    """
    rows: list[dict] = []
    today_str = datetime.today().strftime("%Y-%m-%d")

    # Fetch each symbol once
    raw: dict[str, Optional[float]] = {}
    for sym in _CURVE_SYMBOLS:
        raw[sym] = _last_close(sym, period="5d")

    for label, sym, years in _TENOR_TABLE:
        val = raw.get(sym)
        if val is None:
            continue
        rows.append(
            {
                "tenor_label": label,
                "tenor_years": years,
                "yield_pct": round(val, 4),
                "date": today_str,
            }
        )

    # Synthetic 1Y proxy: interpolate between IRX (0.25y) and FVX (5y)
    # using log-linear interpolation in tenor space
    irx = raw.get("^IRX")
    fvx = raw.get("^FVX")
    if irx is not None and fvx is not None:
        # simple linear interp: weight = (1 - 0.25) / (5 - 0.25)
        w = (1.0 - 0.25) / (5.0 - 0.25)
        y1_proxy = irx + w * (fvx - irx)
        rows.append(
            {
                "tenor_label": "1Y",
                "tenor_years": 1.0,
                "yield_pct": round(y1_proxy, 4),
                "date": today_str,
            }
        )
    elif irx is not None:
        rows.append(
            {
                "tenor_label": "1Y",
                "tenor_years": 1.0,
                "yield_pct": round(irx, 4),
                "date": today_str,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["tenor_label", "tenor_years", "yield_pct", "date"])

    df = pd.DataFrame(rows).sort_values("tenor_years").reset_index(drop=True)
    # Drop duplicate tenor_years, keeping the first (more specific) entry
    df = df.drop_duplicates(subset=["tenor_years"], keep="first").reset_index(drop=True)
    return df


def fetch_historical_curves() -> dict:
    """
    Return historical yield curve snapshots using ^TNX as the rate proxy.

    Uses 2 years of daily history for ^TNX (10Y Treasury) to derive
    three snapshots:
      - "Current"    : most recent close
      - "1M Ago"     : close ~21 trading days ago
      - "1Y Ago"     : close ~252 trading days ago

    Because yfinance only provides ^IRX, ^FVX, ^TNX, ^TYX freely, each
    snapshot is built for those four tenors (0.25y, 5y, 10y, 30y).

    Returns
    -------
    dict[str, pd.Series]
        Keys are "Current", "1M Ago", "1Y Ago".
        Each value is a pd.Series indexed by tenor_years (float) with
        yield values in percent.
    """
    result: dict[str, pd.Series] = {}

    try:
        # Pull ~2 years of history for all four curve symbols
        histories: dict[str, pd.Series] = {}
        for sym in _CURVE_SYMBOLS:
            s = _close_series(sym, period="2y")
            if not s.empty:
                histories[sym] = s

        if not histories:
            return result

        # Tenor → symbol mapping (same as _TENOR_TABLE)
        tenor_sym_map: list[tuple[float, str]] = [
            (0.25,  "^IRX"),
            (5.0,   "^FVX"),
            (10.0,  "^TNX"),
            (30.0,  "^TYX"),
        ]

        def _snapshot(offset: int) -> pd.Series:
            """Extract a single curve snapshot at `offset` days from end."""
            vals: dict[float, float] = {}
            for tenor_yrs, sym in tenor_sym_map:
                if sym not in histories:
                    continue
                s = histories[sym]
                if len(s) > offset:
                    vals[tenor_yrs] = float(s.iloc[-(offset + 1)])
                elif len(s) > 0:
                    vals[tenor_yrs] = float(s.iloc[0])
            if not vals:
                return pd.Series(dtype=float)
            return pd.Series(vals).sort_index()

        result["Current"] = _snapshot(0)
        result["1M Ago"]  = _snapshot(21)
        result["1Y Ago"]  = _snapshot(252)

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Streamlit UI — render_fixed_income_tab
# ---------------------------------------------------------------------------


def render_fixed_income_tab(data: dict) -> None:
    """
    Render the Fixed Income & Macro Analytics tab in QuantLab.

    Parameters
    ----------
    data : dict
        Shared app-state dictionary.  Keys used:
          "theme"            : str  "dark" | "light"
          "returns"          : pd.DataFrame  (date-indexed log/pct returns per ticker)
          "tickers"          : list[str]
          "weights"          : dict[str, float] | None
          "portfolio_value"  : float | None
    """
    theme = data.get("theme", "dark")
    tpt = _get_plotly_theme(theme)

    st.markdown("## Fixed Income & Macro Analytics")
    st.caption(
        "Bond pricing analytics, US Treasury yield curve, and portfolio rate sensitivity."
    )

    # =========================================================================
    # SECTION 1 — Bond Pricing Calculator
    # =========================================================================
    st.markdown("---")
    st.markdown("### Bond Pricing Calculator")

    # Input columns
    col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1, 1.2])
    with col1:
        face_value = st.number_input(
            "Face Value ($)",
            min_value=100.0,
            max_value=10_000_000.0,
            value=1_000.0,
            step=100.0,
            key="fi_face_value",
        )
    with col2:
        coupon_rate_pct = st.number_input(
            "Coupon Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=5.0,
            step=0.25,
            key="fi_coupon_rate",
        )
    with col3:
        ytm_pct = st.number_input(
            "YTM (%)",
            min_value=0.01,
            max_value=30.0,
            value=5.0,
            step=0.25,
            key="fi_ytm",
        )
    with col4:
        years_to_maturity = st.number_input(
            "Years to Maturity",
            min_value=0.5,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key="fi_years",
        )
    with col5:
        freq_label = st.selectbox(
            "Payment Frequency",
            ["Semi-Annual", "Annual"],
            index=0,
            key="fi_frequency",
        )

    freq_map = {"Semi-Annual": 2, "Annual": 1}
    frequency = freq_map[freq_label]
    coupon_rate = coupon_rate_pct / 100.0
    ytm = ytm_pct / 100.0
    periods = max(1, int(round(years_to_maturity * frequency)))

    calc_button = st.button("Calculate Bond", key="fi_calc_btn", type="primary")

    # Always compute on load or when button is clicked (Streamlit reruns on interaction)
    try:
        p0 = BondPricer.price(face_value, coupon_rate, ytm, periods, frequency)
        mac_dur = BondPricer.macaulay_duration(
            face_value, coupon_rate, ytm, periods, frequency
        )
        mod_dur = BondPricer.modified_duration(
            face_value, coupon_rate, ytm, periods, frequency
        )
        conv = BondPricer.convexity(
            face_value, coupon_rate, ytm, periods, frequency
        )
        dv01_val = BondPricer.dv01(
            face_value, coupon_rate, ytm, periods, frequency
        )
        calc_error: Optional[str] = None
    except Exception as exc:
        calc_error = str(exc)
        p0 = mac_dur = mod_dur = conv = dv01_val = 0.0

    if calc_error:
        show_error(f"Bond calculation error: {calc_error}")
    else:
        # ── Metrics row ──────────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "Price ($)",
            f"${p0:,.2f}",
            help="Full (dirty) bond price = PV of all cash flows.",
        )
        m2.metric(
            "Macaulay Duration",
            f"{mac_dur:.3f} yr",
            help="PV-weighted average time to cash flow receipt.",
        )
        m3.metric(
            "Modified Duration",
            f"{mod_dur:.3f} yr",
            help="Macaulay duration / (1 + YTM / frequency). ΔP/P ≈ −D_mod · Δy.",
        )
        m4.metric(
            "Convexity",
            f"{conv:.3f}",
            help="Second derivative of price w.r.t. yield (yr²). Positive for vanilla bonds.",
        )
        m5.metric(
            "DV01 ($)",
            f"${dv01_val:,.4f}",
            help="Dollar price change for a 1 bp increase in yield.",
        )

        # ── Price sensitivity table ───────────────────────────────────────────
        st.markdown("#### Price Sensitivity to Yield Shifts")
        shock_bps_list = [-200, -100, -50, 0, 50, 100, 200]
        sens_rows = []
        for sbps in shock_bps_list:
            dy = sbps / 10_000.0
            new_ytm = ytm + dy
            if new_ytm <= 0.0:
                sens_rows.append(
                    {
                        "Yield Change": f"{sbps:+d} bps",
                        "New YTM (%)": "—",
                        "New Price ($)": "—",
                        "% Change": "—",
                        "Est. Change (D+C)": "—",
                    }
                )
                continue
            try:
                new_price = BondPricer.price(
                    face_value, coupon_rate, new_ytm, periods, frequency
                )
                pct_chg = (new_price - p0) / p0 * 100.0
                est_dp = BondPricer.price_change(mod_dur, conv, dy, p0)
                est_p = p0 + est_dp
                sens_rows.append(
                    {
                        "Yield Change": f"{sbps:+d} bps",
                        "New YTM (%)": f"{new_ytm * 100:.2f}%",
                        "New Price ($)": f"${new_price:,.2f}",
                        "% Change": f"{pct_chg:+.3f}%",
                        "Est. Change (D+C)": f"${est_dp:+,.2f}  (≈${est_p:,.2f})",
                    }
                )
            except Exception:
                sens_rows.append(
                    {
                        "Yield Change": f"{sbps:+d} bps",
                        "New YTM (%)": "error",
                        "New Price ($)": "error",
                        "% Change": "error",
                        "Est. Change (D+C)": "error",
                    }
                )

        sens_df = pd.DataFrame(sens_rows)
        render_styled_table(sens_df, key="fi_sens_table")

        # ── Price–Yield curve ─────────────────────────────────────────────────
        st.markdown("#### Price–Yield Curve")
        ytm_range = np.linspace(0.005, 0.15, 300)  # 0.5% → 15%
        prices_curve = []
        for y in ytm_range:
            try:
                prices_curve.append(
                    BondPricer.price(face_value, coupon_rate, y, periods, frequency)
                )
            except Exception:
                prices_curve.append(np.nan)

        fig_py = go.Figure()
        fig_py.add_trace(
            go.Scatter(
                x=ytm_range * 100,
                y=prices_curve,
                mode="lines",
                name="Bond Price",
                line=dict(color=_COLORS["primary"], width=2.5),
            )
        )
        # Mark current YTM
        fig_py.add_trace(
            go.Scatter(
                x=[ytm * 100],
                y=[p0],
                mode="markers",
                name=f"Current YTM ({ytm_pct:.2f}%)",
                marker=dict(color=_COLORS["warning"], size=10, symbol="circle"),
            )
        )
        fig_py.update_layout(
            **tpt,
            title="Bond Price vs Yield to Maturity",
            xaxis_title="YTM (%)",
            yaxis_title="Price ($)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_py, use_container_width=True)

    # =========================================================================
    # SECTION 2 — US Treasury Yield Curve
    # =========================================================================
    st.markdown("---")
    st.markdown("### US Treasury Yield Curve")
    st.caption(
        "Current curve from yfinance (^IRX · ^FVX · ^TNX · ^TYX). "
        "Historical overlays show the curve 1 month and 1 year ago."
    )

    with st.spinner("Fetching yield curve data…"):
        try:
            curve_df = fetch_full_yield_curve()
            hist_curves = fetch_historical_curves()
            curve_fetch_error: Optional[str] = None
        except Exception as exc:
            curve_fetch_error = str(exc)
            curve_df = pd.DataFrame()
            hist_curves = {}

    if curve_fetch_error:
        show_error(f"Yield curve fetch error: {curve_fetch_error}")
    elif curve_df.empty:
        show_error("No yield curve data available from yfinance at this time.")
    else:
        # ── Plotly chart with historical overlays ─────────────────────────────
        fig_yc = go.Figure()

        # Current curve
        fig_yc.add_trace(
            go.Scatter(
                x=curve_df["tenor_years"].tolist(),
                y=curve_df["yield_pct"].tolist(),
                mode="lines+markers",
                name="Current",
                line=dict(color=_COLORS["primary"], width=2.5),
                marker=dict(size=7),
                text=curve_df["tenor_label"].tolist(),
                hovertemplate="%{text}: %{y:.2f}%<extra></extra>",
            )
        )

        # Historical overlay lines
        hist_styles = {
            "1M Ago": dict(color=_COLORS["secondary"], dash="dash", width=1.8),
            "1Y Ago": dict(color=_COLORS["neutral"],   dash="dot",  width=1.5),
        }
        for label, style in hist_styles.items():
            if label in hist_curves and not hist_curves[label].empty:
                s = hist_curves[label]
                fig_yc.add_trace(
                    go.Scatter(
                        x=s.index.tolist(),
                        y=s.values.tolist(),
                        mode="lines+markers",
                        name=label,
                        line=style,
                        marker=dict(size=5),
                        hovertemplate=f"{label}: %{{y:.2f}}%<extra></extra>",
                    )
                )

        fig_yc.update_layout(
            **tpt,
            title="US Treasury Yield Curve",
            xaxis=dict(
                title="Tenor (Years)",
                tickvals=curve_df["tenor_years"].tolist(),
                ticktext=curve_df["tenor_label"].tolist(),
                gridcolor=tpt.get("xaxis", {}).get("gridcolor", "rgba(255,255,255,0.08)"),
                showgrid=True,
            ),
            yaxis_title="Yield (%)",
            height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_yc, use_container_width=True)

        # ── Spread metrics ────────────────────────────────────────────────────
        yield_map: dict[str, float] = dict(
            zip(curve_df["tenor_label"], curve_df["yield_pct"])
        )

        # 3M–10Y spread
        spread_3m10y: Optional[float] = None
        if "3M" in yield_map and "10Y" in yield_map:
            spread_3m10y = yield_map["10Y"] - yield_map["3M"]

        # 2Y–10Y spread (use 5Y as proxy per task spec)
        spread_2s10s: Optional[float] = None
        if "5Y" in yield_map and "10Y" in yield_map:
            spread_2s10s = yield_map["10Y"] - yield_map["5Y"]

        c1, c2, c3 = st.columns(3)
        if spread_3m10y is not None:
            c1.metric(
                "3M–10Y Spread",
                f"{spread_3m10y:+.2f}%",
                delta_color="inverse",
                help="Negative = inverted curve; historically elevated recession risk.",
            )
        if spread_2s10s is not None:
            c2.metric(
                "5Y–10Y Spread (2Y proxy)",
                f"{spread_2s10s:+.2f}%",
                delta_color="inverse",
                help="5Y used as proxy for 2Y. Negative = partial inversion.",
            )
        if "30Y" in yield_map and "10Y" in yield_map:
            spread_10s30s = yield_map["30Y"] - yield_map["10Y"]
            c3.metric(
                "10Y–30Y Spread",
                f"{spread_10s30s:+.2f}%",
                delta_color="normal",
            )

        # Inversion warning
        inverted = (
            (spread_3m10y is not None and spread_3m10y < 0)
            or (spread_2s10s is not None and spread_2s10s < 0)
        )
        if inverted:
            st.warning(
                "Yield curve inversion detected: short-term rates exceed long-term rates. "
                "Historically associated with elevated near-term recession risk."
            )

    # =========================================================================
    # SECTION 3 — Rate Sensitivity for Portfolio
    # =========================================================================
    st.markdown("---")
    st.markdown("### Portfolio Rate Sensitivity")
    st.caption(
        "Rolling OLS beta of each asset's daily returns against ^TNX (10-Year Treasury yield) "
        "changes. Negative beta = price rises when rates rise (rate-beneficiary); "
        "positive beta = price falls when rates rise (rate-sensitive)."
    )

    returns: pd.DataFrame = data.get("returns", pd.DataFrame())
    tickers: list = data.get("tickers", [])
    weights: dict = data.get("weights", {}) or {}

    # Also try to get returns from 'prices' if 'returns' not present
    if (returns is None or (isinstance(returns, pd.DataFrame) and returns.empty)):
        prices = data.get("prices", pd.DataFrame())
        if prices is not None and not (isinstance(prices, pd.DataFrame) and prices.empty):
            try:
                if isinstance(prices, pd.DataFrame):
                    returns = prices.pct_change().dropna()
            except Exception:
                returns = pd.DataFrame()

    if returns is None or (isinstance(returns, pd.DataFrame) and returns.empty) or not tickers:
        st.info(
            "Load a portfolio (tickers + price history) to see rate sensitivity analysis."
        )
    else:
        with st.spinner("Computing rate betas…"):
            try:
                tnx_series = _close_series("^TNX", period="2y")

                # Align tickers we have returns for
                available_tickers = [
                    t for t in tickers
                    if isinstance(returns, pd.DataFrame) and t in returns.columns
                ]

                if not available_tickers or tnx_series.empty:
                    st.info("Insufficient data for rate sensitivity analysis.")
                else:
                    # Yield changes (level differences, not pct — yield is already %)
                    tnx_changes = tnx_series.diff().dropna()

                    stk_rets = returns[available_tickers].copy()
                    # Ensure timezone-naive index for alignment
                    if hasattr(stk_rets.index, "tz") and stk_rets.index.tz is not None:
                        stk_rets.index = stk_rets.index.tz_localize(None)
                    if hasattr(tnx_changes.index, "tz") and tnx_changes.index.tz is not None:
                        tnx_changes.index = tnx_changes.index.tz_localize(None)

                    # Align on common dates
                    common_idx = stk_rets.index.intersection(tnx_changes.index)
                    stk_a = stk_rets.loc[common_idx]
                    tnx_a = tnx_changes.loc[common_idx]

                    # Compute OLS rate beta for each ticker
                    rate_betas: dict[str, float] = {}
                    for tkr in available_tickers:
                        y_vals = stk_a[tkr].values
                        x_vals = tnx_a.values
                        valid = ~(np.isnan(y_vals) | np.isnan(x_vals))
                        if valid.sum() < 30:
                            rate_betas[tkr] = np.nan
                            continue
                        yv = y_vals[valid]
                        xv = x_vals[valid]
                        # beta = cov(y, x) / var(x)
                        cov_mat = np.cov(yv, xv)
                        var_x = cov_mat[1, 1]
                        rate_betas[tkr] = cov_mat[0, 1] / var_x if var_x != 0.0 else np.nan

                    # Portfolio weights (equal weight fallback)
                    if weights:
                        total_w = sum(weights.get(t, 0) for t in available_tickers)
                        norm_w = {
                            t: weights.get(t, 0) / total_w if total_w > 0 else 1.0 / len(available_tickers)
                            for t in available_tickers
                        }
                    else:
                        norm_w = {t: 1.0 / len(available_tickers) for t in available_tickers}

                    # Portfolio rate beta (weighted)
                    port_beta = sum(
                        norm_w.get(t, 0) * rate_betas.get(t, 0)
                        for t in available_tickers
                        if not np.isnan(rate_betas.get(t, np.nan))
                    )

                    # ── Per-ticker table ──────────────────────────────────────
                    ticker_rows = []
                    for tkr in available_tickers:
                        b = rate_betas.get(tkr, np.nan)
                        # Estimated impact = beta * Δy (in % of portfolio weight)
                        # +50 bps = +0.50 percentage point yield change
                        impact_50  = b * 0.50 * 100.0 if not np.isnan(b) else np.nan
                        impact_100 = b * 1.00 * 100.0 if not np.isnan(b) else np.nan
                        ticker_rows.append(
                            {
                                "Ticker": tkr,
                                "Rate Beta (vs ^TNX)": f"{b:+.4f}" if not np.isnan(b) else "N/A",
                                "Est. Impact +50bps (%)": f"{impact_50:+.2f}%" if not np.isnan(impact_50) else "N/A",
                                "Est. Impact +100bps (%)": f"{impact_100:+.2f}%" if not np.isnan(impact_100) else "N/A",
                            }
                        )

                    ticker_df = pd.DataFrame(ticker_rows)
                    render_styled_table(ticker_df, key="fi_rate_beta_table")

                    # ── Portfolio-level shock summary ─────────────────────────
                    st.markdown("#### Portfolio-Level Rate Shock Estimates")
                    notional = data.get("portfolio_value", 100_000.0) or 100_000.0
                    shock_rows = []
                    for sbps in [50, 100]:
                        dy_pct = sbps / 100.0          # e.g. 0.50 for +50bps
                        est_pct_impact = port_beta * dy_pct * 100.0
                        est_dollar_impact = port_beta * dy_pct * notional
                        shock_rows.append(
                            {
                                "Rate Shock": f"+{sbps} bps",
                                "Portfolio Beta (^TNX)": f"{port_beta:+.4f}",
                                "Est. Portfolio Impact (%)": f"{est_pct_impact:+.2f}%",
                                f"Est. $ Impact (${notional:,.0f} notional)": f"${est_dollar_impact:+,.0f}",
                            }
                        )
                    render_styled_table(pd.DataFrame(shock_rows), key="fi_port_shock_table")

                    # ── Bar chart — rate beta by ticker ───────────────────────
                    valid_betas = {
                        t: b for t, b in rate_betas.items() if not np.isnan(b)
                    }
                    if valid_betas:
                        fig_beta = go.Figure(
                            go.Bar(
                                x=list(valid_betas.keys()),
                                y=list(valid_betas.values()),
                                marker_color=[
                                    _COLORS["negative"] if v > 0 else _COLORS["positive"]
                                    for v in valid_betas.values()
                                ],
                                text=[f"{v:+.3f}" for v in valid_betas.values()],
                                textposition="outside",
                                name="Rate Beta",
                            )
                        )
                        fig_beta.update_layout(
                            **tpt,
                            title="Rate Beta by Ticker (vs ^TNX Yield Changes)",
                            xaxis_title="Ticker",
                            yaxis_title="Beta (return per 1% yield change)",
                            height=380,
                            showlegend=False,
                        )
                        st.plotly_chart(fig_beta, use_container_width=True)
                        st.caption(
                            "Negative beta (green) = asset benefits from rising rates. "
                            "Positive beta (red) = asset is rate-sensitive (price falls as rates rise). "
                            "Computed via OLS of daily returns vs daily ^TNX level changes."
                        )

            except Exception as exc:
                show_error(f"Rate sensitivity analysis failed: {exc}")


# ─── FACTOR_MODEL ───
"""
module_factor_model.py
======================
Multi-Factor Alpha Model module for QuantLab (Streamlit quantitative finance app).

Provides:
  - FactorModel : factor construction, exposure, timing, and attribution.
  - render_factor_model_tab(data) : full Streamlit UI (5 sections).
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
RF_RATE = 0.045  # 4.5% annual risk-free rate

_STRATEGIES = ["Equal Weight", "Max Sharpe", "Min Variance", "Risk Parity"]

_COLORS = {
    "primary":   "#00D4FF",
    "secondary": "#7C3AED",
    "positive":  "#00C49A",
    "negative":  "#FF6B6B",
    "warning":   "#F59E0B",
    "neutral":   "#94A3B8",
    "accent1":   "#F472B6",
    "accent2":   "#34D399",
}

# ---------------------------------------------------------------------------
# Shim helpers – fall back gracefully when run outside the main app
# ---------------------------------------------------------------------------


def _get_plotly_theme(theme: str = "dark") -> dict:
    """Return Plotly layout kwargs matching the app theme."""
    try:
        from app import _get_plotly_theme as _app_theme  # type: ignore
        return _app_theme(theme)
    except ImportError:
        pass
    is_dark = theme == "dark"
    bg         = "#0E1117" if is_dark else "#FFFFFF"
    paper      = "#161B22" if is_dark else "#F8F9FA"
    font_color = "#FAFAFA" if is_dark else "#1A1A2E"
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"
    return dict(
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor=bg,
        paper_bgcolor=paper,
        font=dict(color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True),
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )




# ---------------------------------------------------------------------------
# Portfolio optimisation helpers (for strategy comparison)
# ---------------------------------------------------------------------------


def _equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _max_sharpe_weights(returns: pd.DataFrame, rf_rate: float = RF_RATE) -> np.ndarray:
    n = returns.shape[1]
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.cov() * TRADING_DAYS
    w0 = _equal_weight(n)

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(np.dot(w, mu))
        port_vol = float(np.sqrt(w @ sigma.values @ w))
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - rf_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return w0


def _min_variance_weights(returns: pd.DataFrame) -> np.ndarray:
    n = returns.shape[1]
    sigma = returns.cov() * TRADING_DAYS
    w0 = _equal_weight(n)

    def port_var(w: np.ndarray) -> float:
        return float(w @ sigma.values @ w)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(
        port_var, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return w0


def _risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
    n = returns.shape[1]
    sigma = returns.cov().values * TRADING_DAYS
    w0 = _equal_weight(n)

    def objective(w: np.ndarray) -> float:
        port_var_val = float(w @ sigma @ w)
        if port_var_val < 1e-14:
            return 0.0
        mrc = sigma @ w
        rc = w * mrc / np.sqrt(port_var_val)
        target = np.sqrt(port_var_val) / n
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-4, 1.0)] * n
    res = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, 1.0)
        return w / w.sum()
    return w0


def _get_strategy_weights(strategy: str, returns: pd.DataFrame) -> np.ndarray:
    n = returns.shape[1]
    if n == 0:
        return np.array([])
    if strategy == "Equal Weight":
        return _equal_weight(n)
    elif strategy == "Max Sharpe":
        return _max_sharpe_weights(returns)
    elif strategy == "Min Variance":
        return _min_variance_weights(returns)
    elif strategy == "Risk Parity":
        return _risk_parity_weights(returns)
    return _equal_weight(n)


# ---------------------------------------------------------------------------
# FactorModel
# ---------------------------------------------------------------------------


class FactorModel:
    """Multi-factor alpha model: factor construction, exposure, timing, attribution."""

    FACTORS = ["Momentum", "Value", "Quality", "Low Vol", "Size"]

    # ------------------------------------------------------------------
    # 1. Factor score computation
    # ------------------------------------------------------------------

    def compute_factor_scores(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectionally z-scored factor scores per ticker.

        Factors:
          - Momentum  : 12M-1M return  (252-day minus 21-day)
          - Value     : -1 × trailing 12M return  (contrarian proxy)
          - Quality   : annualised Sharpe ratio  (rf = 4.5 %)
          - Low Vol   : -1 × annualised 252-day volatility
          - Size      : -1 × log(last_price × 1e6)  (smaller = higher score)

        All factors are z-scored cross-sectionally.

        Returns
        -------
        pd.DataFrame
            shape [tickers × factors]
        """
        if prices is None or prices.empty or len(prices) < 22:
            return pd.DataFrame(
                0.0,
                index=prices.columns if prices is not None else [],
                columns=self.FACTORS,
            )

        tickers = prices.columns.tolist()
        returns = prices.pct_change().dropna(how="all")

        raw: dict[str, pd.Series] = {}

        # ---- Momentum: 12M-1M return ----
        mom_vals = []
        for t in tickers:
            p = prices[t].dropna()
            if len(p) >= 252:
                ret_12m = (p.iloc[-1] / p.iloc[-252]) - 1.0
                ret_1m  = (p.iloc[-1] / p.iloc[-21])  - 1.0 if len(p) >= 21 else 0.0
                mom_vals.append(ret_12m - ret_1m)
            elif len(p) >= 22:
                ret_full = (p.iloc[-1] / p.iloc[0]) - 1.0
                ret_1m   = (p.iloc[-1] / p.iloc[-21]) - 1.0 if len(p) >= 21 else 0.0
                mom_vals.append(ret_full - ret_1m)
            else:
                mom_vals.append(np.nan)
        raw["Momentum"] = pd.Series(mom_vals, index=tickers)

        # ---- Value: -1 × trailing 12M return ----
        val_vals = []
        for t in tickers:
            p = prices[t].dropna()
            if len(p) >= 252:
                val_vals.append(-((p.iloc[-1] / p.iloc[-252]) - 1.0))
            elif len(p) >= 2:
                val_vals.append(-((p.iloc[-1] / p.iloc[0]) - 1.0))
            else:
                val_vals.append(np.nan)
        raw["Value"] = pd.Series(val_vals, index=tickers)

        # ---- Quality: annualised Sharpe ----
        qual_vals = []
        rf_daily = RF_RATE / TRADING_DAYS
        for t in tickers:
            r = returns[t].dropna()
            if len(r) < 20:
                qual_vals.append(np.nan)
                continue
            excess = r - rf_daily
            vol = excess.std()
            if vol < 1e-10:
                qual_vals.append(0.0)
            else:
                sharpe = (excess.mean() / vol) * np.sqrt(TRADING_DAYS)
                qual_vals.append(sharpe)
        raw["Quality"] = pd.Series(qual_vals, index=tickers)

        # ---- Low Vol: -1 × annualised 252-day vol ----
        lv_vals = []
        for t in tickers:
            r = returns[t].dropna()
            if len(r) < 20:
                lv_vals.append(np.nan)
                continue
            vol = r.std() * np.sqrt(TRADING_DAYS)
            lv_vals.append(-vol)
        raw["Low Vol"] = pd.Series(lv_vals, index=tickers)

        # ---- Size: -1 × log(last_price × 1e6) ----
        size_vals = []
        for t in tickers:
            p = prices[t].dropna()
            if p.empty:
                size_vals.append(np.nan)
                continue
            last = p.iloc[-1]
            if last <= 0:
                size_vals.append(np.nan)
            else:
                size_vals.append(-np.log(last * 1e6))
        raw["Size"] = pd.Series(size_vals, index=tickers)

        # ---- Cross-sectional z-score ----
        factor_df = pd.DataFrame(raw, index=tickers)

        def _zscore(s: pd.Series) -> pd.Series:
            s = s.dropna()
            mu, sigma = s.mean(), s.std()
            if sigma < 1e-10:
                return pd.Series(0.0, index=s.index)
            return (s - mu) / sigma

        zscored = pd.DataFrame(index=tickers, columns=self.FACTORS, dtype=float)
        for col in self.FACTORS:
            z = _zscore(factor_df[col])
            zscored[col] = z
            # Clip to [-3, 3] for robustness
            zscored[col] = zscored[col].clip(-3.0, 3.0)

        zscored = zscored.fillna(0.0)
        return zscored

    # ------------------------------------------------------------------
    # 2. Factor attribution
    # ------------------------------------------------------------------

    def factor_attribution(
        self,
        weights: np.ndarray,
        tickers: list,
        factor_scores: pd.DataFrame,
    ) -> dict:
        """
        Compute weighted factor exposures for a portfolio.

        Returns
        -------
        dict
            {factor_name: float}  — portfolio-level weighted exposure per factor.
        """
        result = {}
        w = np.asarray(weights, dtype=float)
        # Ensure weights sum to 1
        total = w.sum()
        if total > 1e-10:
            w = w / total

        for factor in self.FACTORS:
            if factor not in factor_scores.columns:
                result[factor] = 0.0
                continue
            exposure = 0.0
            for i, t in enumerate(tickers):
                if i >= len(w):
                    break
                score = factor_scores.loc[t, factor] if t in factor_scores.index else 0.0
                if np.isnan(score):
                    score = 0.0
                exposure += w[i] * score
            result[factor] = float(exposure)
        return result

    # ------------------------------------------------------------------
    # 3. Factor timing
    # ------------------------------------------------------------------

    def factor_timing(
        self,
        prices: pd.DataFrame,
        lookback_days: int = 126,
    ) -> pd.DataFrame:
        """
        For each factor, compute long top-quartile / short bottom-quartile portfolio
        statistics over the lookback window.

        Returns
        -------
        pd.DataFrame
            Index = factor names.
            Columns = ["return", "sharpe", "signal"].
        """
        if prices is None or prices.empty or len(prices) < max(lookback_days, 22) + 5:
            return pd.DataFrame(
                {"return": 0.0, "sharpe": 0.0, "signal": "Neutral"},
                index=self.FACTORS,
            )

        # Use full price history for scoring, but only lookback window for returns
        scores = self.compute_factor_scores(prices)
        # Factor portfolio returns use lookback slice
        factor_rets = self.build_factor_portfolios(prices)

        rows = []
        for factor in self.FACTORS:
            if factor not in factor_rets.columns:
                rows.append({"return": 0.0, "sharpe": 0.0, "signal": "Neutral"})
                continue

            fr = factor_rets[factor].dropna()
            if len(fr) == 0:
                rows.append({"return": 0.0, "sharpe": 0.0, "signal": "Neutral"})
                continue

            # Slice to lookback
            fr = fr.iloc[-lookback_days:] if len(fr) > lookback_days else fr

            cumret = float((1 + fr).prod() - 1)
            vol = float(fr.std())
            rf_period = RF_RATE * len(fr) / TRADING_DAYS
            if vol < 1e-10:
                sharpe = 0.0
            else:
                ann_vol = vol * np.sqrt(TRADING_DAYS)
                ann_ret = (1 + cumret) ** (TRADING_DAYS / max(len(fr), 1)) - 1
                sharpe = (ann_ret - RF_RATE) / ann_vol

            if cumret > 0.01:
                signal = "Positive"
            elif cumret < -0.01:
                signal = "Negative"
            else:
                signal = "Neutral"

            rows.append({"return": cumret, "sharpe": sharpe, "signal": signal})

        df = pd.DataFrame(rows, index=self.FACTORS)
        return df

    # ------------------------------------------------------------------
    # 4. Alpha attribution (OLS regression)
    # ------------------------------------------------------------------

    def alpha_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> dict:
        """
        OLS regression: portfolio_excess = alpha + Σ(β_i × factor_i) + ε

        Returns
        -------
        dict
            alpha_annualized, betas (dict), r_squared, residual_vol, information_ratio
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            return {
                "alpha_annualized": 0.0,
                "betas": {f: 0.0 for f in self.FACTORS},
                "r_squared": 0.0,
                "residual_vol": 0.0,
                "information_ratio": 0.0,
            }

        # Align all series
        aligned = pd.concat(
            [portfolio_returns, benchmark_returns, factor_returns],
            axis=1,
            join="inner",
        ).dropna()

        if aligned.shape[0] < 10:
            return {
                "alpha_annualized": 0.0,
                "betas": {f: 0.0 for f in self.FACTORS},
                "r_squared": 0.0,
                "residual_vol": 0.0,
                "information_ratio": 0.0,
            }

        port_col = aligned.columns[0]
        bm_col   = aligned.columns[1]
        fac_cols = aligned.columns[2:]

        excess = aligned[port_col] - aligned[bm_col]

        X = aligned[fac_cols].copy()
        X = sm.add_constant(X)
        y = excess

        try:
            model = sm.OLS(y, X).fit()
        except Exception:
            return {
                "alpha_annualized": 0.0,
                "betas": {f: 0.0 for f in self.FACTORS},
                "r_squared": 0.0,
                "residual_vol": 0.0,
                "information_ratio": 0.0,
            }

        # Extract alpha (intercept) and annualise
        alpha_daily = float(model.params.get("const", model.params.iloc[0]))
        alpha_annualized = alpha_daily * TRADING_DAYS

        # Betas for each factor
        betas = {}
        for f in self.FACTORS:
            if f in model.params.index:
                betas[f] = float(model.params[f])
            else:
                betas[f] = 0.0

        r_squared = float(model.rsquared)
        resid = model.resid
        residual_vol = float(resid.std() * np.sqrt(TRADING_DAYS))

        # Information ratio = annualised alpha / residual vol
        if residual_vol > 1e-10:
            information_ratio = alpha_annualized / residual_vol
        else:
            information_ratio = 0.0

        return {
            "alpha_annualized": alpha_annualized,
            "betas": betas,
            "r_squared": r_squared,
            "residual_vol": residual_vol,
            "information_ratio": information_ratio,
        }

    # ------------------------------------------------------------------
    # 5. Build factor portfolios
    # ------------------------------------------------------------------

    def build_factor_portfolios(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        For each factor, compute daily returns of a long top-quartile /
        short bottom-quartile portfolio.

        Returns
        -------
        pd.DataFrame
            Daily factor portfolio returns; columns = factor names.
        """
        if prices is None or prices.empty or len(prices) < 25:
            return pd.DataFrame(columns=self.FACTORS)

        returns = prices.pct_change().dropna(how="all")

        # Score using all available data (latest snapshot)
        scores = self.compute_factor_scores(prices)

        factor_port_returns: dict[str, pd.Series] = {}

        for factor in self.FACTORS:
            if factor not in scores.columns:
                continue

            factor_scores_series = scores[factor].dropna().sort_values(ascending=False)
            n = len(factor_scores_series)
            if n < 4:
                continue

            q1_boundary = int(np.ceil(n * 0.25))
            q3_boundary = int(np.floor(n * 0.75))

            long_tickers  = factor_scores_series.index[:q1_boundary].tolist()
            short_tickers = factor_scores_series.index[q3_boundary:].tolist()

            # Filter to tickers that exist in returns
            long_tickers  = [t for t in long_tickers  if t in returns.columns]
            short_tickers = [t for t in short_tickers if t in returns.columns]

            if not long_tickers or not short_tickers:
                continue

            # Equal-weight long and short legs
            long_ret  = returns[long_tickers].mean(axis=1)
            short_ret = returns[short_tickers].mean(axis=1)
            port_ret  = long_ret - short_ret

            factor_port_returns[factor] = port_ret

        if not factor_port_returns:
            return pd.DataFrame(columns=self.FACTORS)

        return pd.DataFrame(factor_port_returns).dropna(how="all")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_factor_model_tab(data: dict) -> None:
    """
    Render the full Factor Model tab UI in Streamlit.

    Parameters
    ----------
    data : dict
        Must contain:
          - 'prices'   : pd.DataFrame  (DatetimeIndex, columns = tickers)
          - 'tickers'  : list[str]
          - 'portfolio': dict  { strategy_name: np.ndarray of weights }
                         (first available strategy is used for sections 2 & 4)
        Optional:
          - 'theme'    : str  "dark" | "light"
          - 'rf_rate'  : float
    """
    theme    = data.get("theme", st.session_state.get("theme", "dark"))
    rf_rate  = float(data.get("rf_rate", RF_RATE))
    plotly_t = _get_plotly_theme(theme)

    is_dark    = theme == "dark"
    pos_color  = _COLORS["positive"]
    neg_color  = _COLORS["negative"]
    neu_color  = _COLORS["neutral"]
    pri_color  = _COLORS["primary"]

    st.markdown("## Multi-Factor Alpha Model")
    st.markdown(
        "Factor construction, portfolio exposure, factor timing, alpha attribution, "
        "and cross-strategy comparison across **Momentum · Value · Quality · Low Vol · Size**."
    )

    # ---- validate inputs ----
    prices: pd.DataFrame = data.get("prices", pd.DataFrame())
    if prices is None or prices.empty:
        show_error("No price data available. Please load a portfolio first.")
        return

    tickers: list[str] = data.get("tickers", prices.columns.tolist())
    if not tickers:
        show_error("No tickers found in the data dictionary.")
        return

    # Restrict prices to the requested tickers
    available = [t for t in tickers if t in prices.columns]
    if not available:
        show_error("None of the provided tickers are present in the price data.")
        return
    prices = prices[available].copy()
    tickers = available

    # Retrieve portfolio weights dict
    portfolio_dict: dict = data.get("portfolio", {})

    # Build returns for optimisers
    returns = prices.pct_change().dropna(how="all")

    # Instantiate model
    model = FactorModel()

    # =========================================================================
    # Section 1 – Factor Score Heatmap
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 1 — Factor Score Heatmap")
    st.markdown(
        "Cross-sectionally z-scored factor scores (clipped to ±3). "
        "Green = high score (favourable), red = low score (unfavourable)."
    )

    try:
        with st.spinner("Computing factor scores…"):
            factor_scores = model.compute_factor_scores(prices)

        if factor_scores.empty:
            show_error("Insufficient price history to compute factor scores (need ≥ 22 days).")
        else:
            # --- Heatmap ---
            z_data    = factor_scores[model.FACTORS].values.T.tolist()
            x_labels  = factor_scores.index.tolist()
            y_labels  = model.FACTORS

            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=z_data,
                    x=x_labels,
                    y=y_labels,
                    zmin=-2,
                    zmax=2,
                    colorscale=[
                        [0.0,  neg_color],   # -2  → red
                        [0.5,  "#FFFFFF"],   #  0  → white
                        [1.0,  pos_color],   # +2  → green
                    ],
                    colorbar=dict(
                        title="Z-Score",
                        tickvals=[-2, -1, 0, 1, 2],
                    ),
                    hovertemplate="Ticker: %{x}<br>Factor: %{y}<br>Score: %{z:.2f}<extra></extra>",
                    text=[[f"{v:.2f}" for v in row] for row in z_data],
                    texttemplate="%{text}",
                )
            )
            fig_heat.update_layout(
                **plotly_t,
                title="Factor Score Heatmap (Z-Scored)",
                xaxis=dict(
                    title="Ticker",
                    tickangle=-45,
                    gridcolor=plotly_t.get("xaxis", {}).get("gridcolor", "rgba(255,255,255,0.08)"),
                ),
                yaxis=dict(
                    title="Factor",
                    gridcolor=plotly_t.get("yaxis", {}).get("gridcolor", "rgba(255,255,255,0.08)"),
                ),
                height=400,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # --- Raw scores table ---
            st.markdown("#### Raw Factor Scores")
            display_scores = factor_scores[model.FACTORS].copy()
            display_scores.index.name = "Ticker"
            for col in display_scores.columns:
                display_scores[col] = display_scores[col].map("{:.3f}".format)
            render_styled_table(display_scores, key="factor_scores_table")

    except Exception as exc:
        show_error(f"Factor Score section error: {exc}")

    # =========================================================================
    # Section 2 – Portfolio Factor Exposure
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 2 — Portfolio Factor Exposure")
    st.markdown(
        "Weighted factor exposures for the selected strategy versus Equal Weight."
    )

    try:
        # Determine selected strategy
        strategy_names = list(portfolio_dict.keys()) if portfolio_dict else []
        if strategy_names:
            default_idx = 0
            selected_strategy = st.selectbox(
                "Strategy",
                options=strategy_names,
                index=default_idx,
                key="fm_strategy_select",
            )
            strat_weights = np.asarray(portfolio_dict[selected_strategy], dtype=float)
        else:
            # Fall back: compute equal weight
            selected_strategy = "Equal Weight"
            strat_weights = _equal_weight(len(tickers))

        n = len(tickers)
        ew_weights = _equal_weight(n)

        # Align weights to tickers length
        if len(strat_weights) != n:
            strat_weights = _equal_weight(n)

        # Compute exposures
        if "factor_scores" not in dir():
            factor_scores = model.compute_factor_scores(prices)

        ew_exposure   = model.factor_attribution(ew_weights,    tickers, factor_scores)
        strat_exposure = model.factor_attribution(strat_weights, tickers, factor_scores)

        factors_list = model.FACTORS
        ew_vals      = [ew_exposure.get(f, 0.0)    for f in factors_list]
        strat_vals   = [strat_exposure.get(f, 0.0) for f in factors_list]

        # Bar colors: positive = teal, negative = red (per strategy)
        strat_colors = [pos_color if v >= 0 else neg_color for v in strat_vals]
        ew_colors    = [_COLORS["secondary"] if v >= 0 else _COLORS["warning"] for v in ew_vals]

        fig_exp = go.Figure()
        fig_exp.add_trace(go.Bar(
            name="Equal Weight",
            x=factors_list,
            y=ew_vals,
            marker_color=ew_colors,
            opacity=0.75,
            hovertemplate="Factor: %{x}<br>EW Exposure: %{y:.3f}<extra></extra>",
        ))
        fig_exp.add_trace(go.Bar(
            name=selected_strategy,
            x=factors_list,
            y=strat_vals,
            marker_color=strat_colors,
            opacity=0.95,
            hovertemplate="Factor: %{x}<br>Exposure: %{y:.3f}<extra></extra>",
        ))
        fig_exp.update_layout(
            **plotly_t,
            title=f"Factor Exposure: Equal Weight vs {selected_strategy}",
            xaxis_title="Factor",
            yaxis_title="Weighted Exposure (Z-Score units)",
            barmode="group",
            height=420,
        )
        fig_exp.add_hline(y=0, line_dash="dash", line_color=neu_color, line_width=1)
        st.plotly_chart(fig_exp, use_container_width=True)

        # Exposure summary table
        exp_table = pd.DataFrame(
            {
                "Factor": factors_list,
                "Equal Weight": [f"{v:.3f}" for v in ew_vals],
                selected_strategy: [f"{v:.3f}" for v in strat_vals],
            }
        ).set_index("Factor")
        render_styled_table(exp_table, key="factor_exposure_table")

    except Exception as exc:
        show_error(f"Portfolio Exposure section error: {exc}")

    # =========================================================================
    # Section 3 – Factor Timing
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 3 — Factor Timing (Which Factors Are Working?)")
    st.markdown(
        "Long top-quartile / short bottom-quartile factor portfolio performance "
        "over the selected lookback window."
    )

    try:
        lookback_label = st.selectbox(
            "Lookback Period",
            options=["1M (21 days)", "3M (63 days)", "6M (126 days)", "1Y (252 days)"],
            index=2,
            key="fm_lookback",
        )
        lookback_map = {
            "1M (21 days)":   21,
            "3M (63 days)":   63,
            "6M (126 days)": 126,
            "1Y (252 days)": 252,
        }
        lookback_days = lookback_map[lookback_label]

        with st.spinner("Computing factor timing…"):
            timing_df = model.factor_timing(prices, lookback_days=lookback_days)

        # --- Signal table ---
        st.markdown("#### Factor Performance Table")

        def _signal_badge(sig: str) -> str:
            if sig == "Positive":
                return "🟢 POSITIVE"
            elif sig == "Negative":
                return "🔴 NEGATIVE"
            return "⚪ NEUTRAL"

        timing_display = pd.DataFrame(
            {
                "Factor":        timing_df.index.tolist(),
                "Recent Return": [f"{v*100:.2f}%" for v in timing_df["return"]],
                "Sharpe Ratio":  [f"{v:.2f}"      for v in timing_df["sharpe"]],
                "Signal":        [_signal_badge(s) for s in timing_df["signal"]],
            }
        ).set_index("Factor")
        render_styled_table(timing_display, key="factor_timing_table")

        # --- Bar chart of factor returns ---
        bar_colors = []
        for sig in timing_df["signal"]:
            if sig == "Positive":
                bar_colors.append(pos_color)
            elif sig == "Negative":
                bar_colors.append(neg_color)
            else:
                bar_colors.append(neu_color)

        fig_timing = go.Figure(
            go.Bar(
                x=timing_df.index.tolist(),
                y=(timing_df["return"] * 100).tolist(),
                marker_color=bar_colors,
                hovertemplate="Factor: %{x}<br>Return: %{y:.2f}%<extra></extra>",
                text=[f"{v*100:.2f}%" for v in timing_df["return"]],
                textposition="outside",
            )
        )
        fig_timing.update_layout(
            **plotly_t,
            title=f"Factor Returns over {lookback_label}",
            xaxis_title="Factor",
            yaxis_title="Return (%)",
            height=420,
        )
        fig_timing.add_hline(y=0, line_dash="dash", line_color=neu_color, line_width=1)
        st.plotly_chart(fig_timing, use_container_width=True)

    except Exception as exc:
        show_error(f"Factor Timing section error: {exc}")

    # =========================================================================
    # Section 4 – Alpha Attribution
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 4 — Alpha Attribution")
    st.markdown(
        "OLS regression of portfolio excess returns on factor portfolio returns. "
        "Decomposes performance into alpha, systematic factor exposures, and residual."
    )

    try:
        with st.spinner("Building factor portfolios and running regression…"):
            factor_port_rets = model.build_factor_portfolios(prices)

        if factor_port_rets.empty:
            show_error("Insufficient data to build factor portfolios (need ≥ 25 days and ≥ 4 tickers).")
        else:
            # Portfolio returns (use selected strategy weights from Section 2)
            if len(strat_weights) == n:
                port_ret_series = (returns @ strat_weights)
            else:
                port_ret_series = returns.mean(axis=1)

            # Benchmark: equal weight
            bm_ret_series = returns.mean(axis=1)

            # Align
            common_idx = port_ret_series.dropna().index.intersection(
                bm_ret_series.dropna().index
            ).intersection(factor_port_rets.dropna(how="all").index)

            if len(common_idx) < 10:
                show_error("Insufficient overlapping data for regression (need ≥ 10 trading days).")
            else:
                port_aligned = port_ret_series.loc[common_idx]
                bm_aligned   = bm_ret_series.loc[common_idx]
                fac_aligned  = factor_port_rets.loc[common_idx]

                attr = model.alpha_attribution(port_aligned, fac_aligned, bm_aligned)

                # ---- Metrics display ----
                col1, col2, col3, col4, col5 = st.columns(5)
                alpha_pct = attr["alpha_annualized"] * 100
                r2        = attr["r_squared"]
                res_vol   = attr["residual_vol"] * 100
                ir        = attr["information_ratio"]

                alpha_color = pos_color if alpha_pct >= 0 else neg_color
                ir_color    = pos_color if ir >= 0 else neg_color

                with col1:
                    st.metric(
                        "Alpha (Ann.)",
                        f"{alpha_pct:+.2f}%",
                        delta=None,
                    )
                with col2:
                    st.metric("R²", f"{r2:.3f}")
                with col3:
                    st.metric("Residual Vol (Ann.)", f"{res_vol:.2f}%")
                with col4:
                    st.metric("Information Ratio", f"{ir:.3f}")
                with col5:
                    betas = attr["betas"]
                    max_beta_factor = max(betas, key=lambda k: abs(betas[k])) if betas else "N/A"
                    st.metric("Largest Beta Factor", max_beta_factor)

                # ---- Betas table ----
                st.markdown("#### Factor Betas")
                betas_df = pd.DataFrame(
                    {
                        "Factor": list(attr["betas"].keys()),
                        "Beta":   [f"{v:.4f}" for v in attr["betas"].values()],
                    }
                ).set_index("Factor")
                render_styled_table(betas_df, key="alpha_betas_table")

                # ---- Return decomposition bar chart ----
                st.markdown("#### Return Decomposition")

                # Factor contributions = beta_i × mean_factor_return × 252
                decomp_labels = ["Alpha"]
                decomp_values = [attr["alpha_annualized"] * 100]
                decomp_colors = [pos_color if attr["alpha_annualized"] >= 0 else neg_color]

                for factor in model.FACTORS:
                    if factor in fac_aligned.columns and factor in betas:
                        beta  = betas[factor]
                        fmean = fac_aligned[factor].mean() * TRADING_DAYS
                        contrib = beta * fmean * 100  # in percent
                        decomp_labels.append(factor)
                        decomp_values.append(contrib)
                        decomp_colors.append(pos_color if contrib >= 0 else neg_color)

                fig_decomp = go.Figure(
                    go.Bar(
                        x=decomp_labels,
                        y=decomp_values,
                        marker_color=decomp_colors,
                        hovertemplate="%{x}<br>Contribution: %{y:.2f}%<extra></extra>",
                        text=[f"{v:.2f}%" for v in decomp_values],
                        textposition="outside",
                    )
                )
                fig_decomp.update_layout(
                    **plotly_t,
                    title="Annualised Return Decomposition: Alpha + Factor Contributions",
                    xaxis_title="Component",
                    yaxis_title="Annualised Contribution (%)",
                    height=420,
                )
                fig_decomp.add_hline(y=0, line_dash="dash", line_color=neu_color, line_width=1)
                st.plotly_chart(fig_decomp, use_container_width=True)

    except Exception as exc:
        show_error(f"Alpha Attribution section error: {exc}")

    # =========================================================================
    # Section 5 – Strategy Comparison
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 5 — Strategy Comparison")
    st.markdown(
        "Factor exposures across all four optimisation strategies side-by-side."
    )

    try:
        if "factor_scores" not in dir() or factor_scores is None or factor_scores.empty:
            factor_scores = model.compute_factor_scores(prices)

        strategy_exposures: dict[str, dict] = {}
        for strat_name in _STRATEGIES:
            try:
                w = _get_strategy_weights(strat_name, returns)
                if len(w) != len(tickers):
                    w = _equal_weight(len(tickers))
                exp = model.factor_attribution(w, tickers, factor_scores)
                strategy_exposures[strat_name] = exp
            except Exception:
                strategy_exposures[strat_name] = {f: 0.0 for f in model.FACTORS}

        # Grouped bar chart: factors on x-axis, one bar per strategy
        strategy_colors_list = [
            pri_color,
            _COLORS["secondary"],
            _COLORS["accent2"],
            _COLORS["warning"],
        ]

        fig_cmp = go.Figure()
        for i, strat_name in enumerate(_STRATEGIES):
            exp = strategy_exposures[strat_name]
            y_vals = [exp.get(f, 0.0) for f in model.FACTORS]
            fig_cmp.add_trace(go.Bar(
                name=strat_name,
                x=model.FACTORS,
                y=y_vals,
                marker_color=strategy_colors_list[i % len(strategy_colors_list)],
                hovertemplate=f"{strat_name}<br>Factor: %{{x}}<br>Exposure: %{{y:.3f}}<extra></extra>",
            ))

        fig_cmp.update_layout(
            **plotly_t,
            title="Factor Exposures Across Strategies",
            xaxis_title="Factor",
            yaxis_title="Weighted Exposure (Z-Score units)",
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=460,
        )
        fig_cmp.add_hline(y=0, line_dash="dash", line_color=neu_color, line_width=1)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Comparison table
        cmp_table = pd.DataFrame(
            {strat: {f: f"{strategy_exposures[strat].get(f, 0.0):.3f}" for f in model.FACTORS}
             for strat in _STRATEGIES}
        )
        cmp_table.index.name = "Factor"
        render_styled_table(cmp_table, key="strategy_comparison_table")

    except Exception as exc:
        show_error(f"Strategy Comparison section error: {exc}")


# ─── OPTIONS_BUILDER ───
"""
QuantLab — Options Strategy Builder Module
==========================================
Provides:
  • OptionsStrategyBuilder  – multi-leg strategy construction, payoff analysis
  • screen_options_chain()  – yfinance-backed options screener
  • render_options_builder_tab() – full Streamlit UI (two sections)
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers expected from app.py – fall-back stubs so module works standalone
# ---------------------------------------------------------------------------

def _get_plotly_theme() -> dict:
    """Return a minimal Plotly layout dict compatible with QuantLab theme."""
    try:
        from app import _get_plotly_theme as _app_theme  # type: ignore
        return _app_theme()
    except Exception:
        return {
            "template": "plotly_dark",
            "paper_bgcolor": "#0e1117",
            "plot_bgcolor": "#0e1117",
            "font": {"color": "#fafafa", "family": "Inter, sans-serif"},
        }




# ---------------------------------------------------------------------------
# Black-Scholes pricing & Greeks  (self-contained, no app.py dependency)
# ---------------------------------------------------------------------------

_SQRT_2PI = math.sqrt(2 * math.pi)
_MIN_SIGMA = 1e-8
_MIN_T = 1e-8


def _options_bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    opt_type: str = "call",
) -> float:
    """
    Black-Scholes European option price (options builder variant).

    Parameters
    ----------
    S        : current underlying price
    K        : strike price
    T        : time to expiry in years
    r        : risk-free rate (annualised, decimal)
    sigma    : implied volatility (annualised, decimal)
    opt_type : 'call' or 'put'

    Returns
    -------
    float option price (0.0 for stock legs)
    """
    if opt_type == "stock":
        return float(S)
    T = max(T, _MIN_T)
    sigma = max(sigma, _MIN_SIGMA)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(float(price), 0.0)


def _options_bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    opt_type: str = "call",
) -> Dict[str, float]:
    """
    Black-Scholes Greeks (options builder variant).

    Returns dict with keys: delta, gamma, theta, vega, rho.
    Stock legs return {delta:1, gamma:0, theta:0, vega:0, rho:0}.
    """
    if opt_type == "stock":
        return {"delta": 1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    T = max(T, _MIN_T)
    sigma = max(sigma, _MIN_SIGMA)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    nd1 = norm.pdf(d1)
    discount = math.exp(-r * T)

    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * nd1 * sqrt_T / 100  # per 1% move in vol

    if opt_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * nd1 * sigma) / (2 * sqrt_T)
            - r * K * discount * norm.cdf(d2)
        ) / 365
        rho = K * T * discount * norm.cdf(d2) / 100
    else:  # put
        delta = norm.cdf(d1) - 1.0
        theta = (
            -(S * nd1 * sigma) / (2 * sqrt_T)
            + r * K * discount * norm.cdf(-d2)
        ) / 365
        rho = -K * T * discount * norm.cdf(-d2) / 100

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }


# ---------------------------------------------------------------------------
# OptionsStrategyBuilder
# ---------------------------------------------------------------------------

class OptionsStrategyBuilder:
    """
    Builds and analyses multi-leg options strategies.

    Templates map strategy names to lists of
        (option_type, action, strike_offset_from_spot)
    where strike = spot * (1 + offset).
    """

    TEMPLATES: Dict[str, List[Tuple[str, str, float]]] = {
        "Bull Call Spread": [("call", "buy", -0.05), ("call", "sell", 0.05)],
        "Bear Put Spread": [("put", "buy", 0.05), ("put", "sell", -0.05)],
        "Straddle": [("call", "buy", 0.0), ("put", "buy", 0.0)],
        "Strangle": [("call", "buy", 0.05), ("put", "buy", -0.05)],
        "Iron Condor": [
            ("put", "buy", -0.10),
            ("put", "sell", -0.05),
            ("call", "sell", 0.05),
            ("call", "buy", 0.10),
        ],
        "Covered Call": [("stock", "buy", 0.0), ("call", "sell", 0.05)],
        "Protective Put": [("stock", "buy", 0.0), ("put", "buy", -0.05)],
        "Butterfly": [
            ("call", "buy", -0.05),
            ("call", "sell", 0.0),
            ("call", "sell", 0.0),
            ("call", "buy", 0.05),
        ],
    }

    # Default market assumptions when not supplied by caller
    DEFAULT_R: float = 0.05
    DEFAULT_SIGMA: float = 0.25

    # ------------------------------------------------------------------
    # Core leg builder
    # ------------------------------------------------------------------

    def build_leg(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        opt_type: str,
        action: str,
        qty: int = 1,
    ) -> Dict:
        """
        Compute price and Greeks for a single leg.

        Returns dict with keys:
            type, action, K, price, delta, gamma, theta, vega, qty
        """
        price = _options_bs_price(S, K, T, r, sigma, opt_type)
        greeks = _options_bs_greeks(S, K, T, r, sigma, opt_type)
        return {
            "type": opt_type,
            "action": action,
            "K": round(K, 4),
            "price": round(price, 4),
            "delta": round(greeks["delta"], 4),
            "gamma": round(greeks["gamma"], 6),
            "theta": round(greeks["theta"], 4),
            "vega": round(greeks["vega"], 4),
            "qty": qty,
        }

    # ------------------------------------------------------------------
    # Template instantiation
    # ------------------------------------------------------------------

    def build_from_template(
        self,
        template_name: str,
        S: float,
        T: float,
        r: float = DEFAULT_R,
        sigma: float = DEFAULT_SIGMA,
    ) -> List[Dict]:
        """
        Instantiate all legs for a named template.

        Returns list of leg dicts from build_leg().
        """
        legs_spec = self.TEMPLATES[template_name]
        legs = []
        for idx, (opt_type, action, offset) in enumerate(legs_spec):
            K = S * (1.0 + offset)
            leg = self.build_leg(S, K, T, r, sigma, opt_type, action, qty=1)
            leg["leg_num"] = idx + 1
            legs.append(leg)
        return legs

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def aggregate_greeks(self, legs: List[Dict]) -> Dict[str, float]:
        """
        Net delta / gamma / theta / vega across all legs.
        buy legs count as +1, sell legs as -1.
        """
        net = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for leg in legs:
            sign = 1 if leg["action"] == "buy" else -1
            qty = leg.get("qty", 1)
            for greek in net:
                net[greek] += sign * qty * leg[greek]
        return {k: round(v, 6) for k, v in net.items()}

    def net_premium(self, legs: List[Dict]) -> float:
        """
        Net cost of the strategy (positive = debit, negative = credit).
        """
        total = 0.0
        for leg in legs:
            sign = 1 if leg["action"] == "buy" else -1
            total += sign * leg["qty"] * leg["price"]
        return round(total, 4)

    # ------------------------------------------------------------------
    # Payoff / P&L at expiry
    # ------------------------------------------------------------------

    def _leg_payoff(self, leg: Dict, S_T: float) -> float:
        """Intrinsic value of a single leg at expiry for underlying price S_T."""
        K = leg["K"]
        opt_type = leg["type"]
        action = leg["action"]
        qty = leg.get("qty", 1)
        sign = 1 if action == "buy" else -1

        if opt_type == "call":
            intrinsic = max(S_T - K, 0.0)
        elif opt_type == "put":
            intrinsic = max(K - S_T, 0.0)
        else:  # stock
            intrinsic = S_T  # return full price; cost subtracted via premium

        return sign * qty * intrinsic

    def payoff_at_expiry(self, legs: List[Dict], S_range: np.ndarray) -> np.ndarray:
        """
        Strategy P&L at expiry for each price in S_range.

        P&L = sum(leg intrinsic values) − net_premium
        (net_premium is the upfront cost; positive debit reduces P&L).
        """
        premium = self.net_premium(legs)
        payoffs = np.array(
            [sum(self._leg_payoff(leg, float(s)) for leg in legs) for s in S_range],
            dtype=float,
        )
        return payoffs - premium

    def max_profit_loss(
        self, legs: List[Dict], S_range: np.ndarray
    ) -> Dict[str, float]:
        """
        Return max_profit, max_loss and their respective underlying prices.
        """
        pnl = self.payoff_at_expiry(legs, S_range)
        max_idx = int(np.argmax(pnl))
        min_idx = int(np.argmin(pnl))
        return {
            "max_profit": float(pnl[max_idx]),
            "max_loss": float(pnl[min_idx]),
            "max_profit_price": float(S_range[max_idx]),
            "max_loss_price": float(S_range[min_idx]),
        }

    def breakevens(self, legs: List[Dict], S_range: np.ndarray) -> List[float]:
        """
        Identify breakeven prices (where P&L crosses zero).
        Uses linear interpolation between adjacent sign-change points.
        """
        pnl = self.payoff_at_expiry(legs, S_range)
        bks: List[float] = []
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i + 1] <= 0 and not (pnl[i] == 0 and pnl[i + 1] == 0):
                # Linear interpolation
                x0, x1 = float(S_range[i]), float(S_range[i + 1])
                y0, y1 = float(pnl[i]), float(pnl[i + 1])
                if y1 != y0:
                    bk = x0 - y0 * (x1 - x0) / (y1 - y0)
                    bks.append(round(bk, 4))
        return bks

    # ------------------------------------------------------------------
    # Greeks vs underlying price (for Greeks chart)
    # ------------------------------------------------------------------

    def greeks_vs_price(
        self,
        legs_spec: List[Dict],
        S_range: np.ndarray,
        T: float,
        r: float,
        sigma: float,
    ) -> pd.DataFrame:
        """
        Recompute net delta and gamma for each S in S_range.

        legs_spec must contain keys: type, action, K, qty.
        Returns DataFrame with columns: S, delta, gamma.
        """
        records = []
        for s in S_range:
            net_d = 0.0
            net_g = 0.0
            for leg in legs_spec:
                sign = 1 if leg["action"] == "buy" else -1
                qty = leg.get("qty", 1)
                g = _options_bs_greeks(float(s), leg["K"], T, r, sigma, leg["type"])
                net_d += sign * qty * g["delta"]
                net_g += sign * qty * g["gamma"]
            records.append({"S": float(s), "delta": net_d, "gamma": net_g})
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Options Screener
# ---------------------------------------------------------------------------

def screen_options_chain(
    ticker: str,
    option_type: str = "both",
    min_volume: int = 10,
    min_oi: int = 100,
    moneyness_min: float = 0.8,
    moneyness_max: float = 1.2,
) -> pd.DataFrame:
    """
    Fetch and filter an options chain via yfinance.

    Parameters
    ----------
    ticker         : equity ticker symbol
    option_type    : 'calls', 'puts', or 'both'
    min_volume     : minimum daily volume
    min_oi         : minimum open interest
    moneyness_min  : K/S lower bound (e.g. 0.8 = 20% OTM puts)
    moneyness_max  : K/S upper bound (e.g. 1.2 = 20% OTM calls)

    Returns
    -------
    pd.DataFrame with columns including strike, expiration, type,
    lastPrice, bid, ask, impliedVolatility, volume, openInterest,
    moneyness, iv_rank, volume_spike.
    """
    tk = yf.Ticker(ticker)
    spot = None
    try:
        info = tk.fast_info
        spot = float(info.get("lastPrice") or info.get("last_price") or 0)
    except Exception:
        pass
    if not spot or spot <= 0:
        hist = tk.history(period="2d")
        if not hist.empty:
            spot = float(hist["Close"].iloc[-1])
    if not spot or spot <= 0:
        raise ValueError(f"Could not retrieve current price for {ticker}")

    expirations = tk.options
    if not expirations:
        raise ValueError(f"No options data available for {ticker}")

    frames: List[pd.DataFrame] = []
    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        if option_type in ("calls", "both"):
            df_c = chain.calls.copy()
            df_c["type"] = "call"
            df_c["expiration"] = exp
            frames.append(df_c)
        if option_type in ("puts", "both"):
            df_p = chain.puts.copy()
            df_p["type"] = "put"
            df_p["expiration"] = exp
            frames.append(df_p)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric types
    for col in ["volume", "openInterest", "impliedVolatility", "lastPrice", "bid", "ask"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived columns
    df["moneyness"] = df["strike"] / spot
    df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # Filters
    vol_col = "volume" if "volume" in df.columns else None
    oi_col = "openInterest" if "openInterest" in df.columns else None

    if vol_col:
        df = df[df[vol_col] >= min_volume]
    if oi_col:
        df = df[df[oi_col] >= min_oi]
    df = df[(df["moneyness"] >= moneyness_min) & (df["moneyness"] <= moneyness_max)]

    if df.empty:
        return df

    # IV Rank: percentile of current IV relative to all IVs in the filtered chain
    if "impliedVolatility" in df.columns:
        iv_vals = df["impliedVolatility"].replace(0, np.nan).dropna()
        if len(iv_vals) > 1:
            iv_pct = df["impliedVolatility"].rank(pct=True)
            df["iv_rank"] = (iv_pct * 100).round(1)
        else:
            df["iv_rank"] = 50.0
    else:
        df["iv_rank"] = np.nan

    # Volume spike: z-score of volume within each type
    if vol_col:
        df["vol_mean"] = df.groupby("type")[vol_col].transform("mean")
        df["vol_std"] = df.groupby("type")[vol_col].transform("std").fillna(1)
        df["volume_zscore"] = ((df[vol_col] - df["vol_mean"]) / df["vol_std"]).round(2)
        df["volume_spike"] = df["volume_zscore"] > 2.0
        df.drop(columns=["vol_mean", "vol_std", "volume_zscore"], inplace=True)
    else:
        df["volume_spike"] = False

    # Clean up and reorder
    keep_cols = [
        c for c in [
            "expiration", "type", "strike", "moneyness",
            "lastPrice", "bid", "ask", "mid",
            "impliedVolatility", "iv_rank",
            "volume", "openInterest", "volume_spike",
            "contractSymbol",
        ]
        if c in df.columns
    ]
    df = df[keep_cols].reset_index(drop=True)
    df = df.rename(columns={
        "lastPrice": "Last",
        "impliedVolatility": "IV",
        "openInterest": "OI",
        "volume": "Volume",
        "contractSymbol": "Contract",
        "moneyness": "Moneyness",
        "iv_rank": "IV Rank",
        "volume_spike": "Vol Spike",
    })
    df["spot"] = spot
    return df


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------

def _sigma_from_ticker(ticker: str) -> float:
    """Estimate IV from recent historical volatility (30-day window)."""
    try:
        hist = yf.Ticker(ticker).history(period="3mo")
        if len(hist) > 5:
            log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            return float(np.std(log_ret) * np.sqrt(252))
    except Exception:
        pass
    return 0.25


def _spot_from_ticker(ticker: str) -> float:
    """Get current spot price."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        price = float(info.get("lastPrice") or info.get("last_price") or 0)
        if price > 0:
            return price
        hist = tk.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 100.0


def _get_expirations(ticker: str) -> List[str]:
    """Return available option expiration dates."""
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


def _t_from_expiry(expiry_str: str) -> float:
    """Convert YYYY-MM-DD expiry string to years from today."""
    from datetime import date, datetime
    try:
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = date.today()
        days = (exp_date - today).days
        return max(days / 365.0, 1 / 365.0)
    except Exception:
        return 30 / 365.0


def _iv_from_chain(ticker: str, expiry: str, spot: float) -> float:
    """Rough ATM IV from options chain for a given expiry."""
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
        calls = chain.calls
        if calls.empty:
            return 0.25
        calls["dist"] = abs(calls["strike"] - spot)
        atm = calls.nsmallest(3, "dist")
        iv = atm["impliedVolatility"].median()
        if iv and iv > 0:
            return float(iv)
    except Exception:
        pass
    return 0.25


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_options_builder_tab(data: dict) -> None:  # noqa: C901
    """
    Render the full Options Builder tab in QuantLab.

    Parameters
    ----------
    data : dict  (passed from main app; may contain 'ticker' key as default)
    """
    builder = OptionsStrategyBuilder()
    theme = _get_plotly_theme()

    st.markdown("## Options Strategy Lab")
    tab1, tab2 = st.tabs(["📐 Strategy Builder", "🔍 Options Screener"])

    # ==================================================================
    # TAB 1 — Multi-Leg Strategy Builder
    # ==================================================================
    with tab1:
        st.markdown("### Multi-Leg Strategy Builder")

        # ---- Controls row ----
        col_ticker, col_exp, col_price = st.columns([2, 2, 2])
        with col_ticker:
            default_ticker = data.get("ticker", "SPY") if data else "SPY"
            ticker_sb = st.text_input(
                "Ticker", value=default_ticker, key="ob_ticker"
            ).upper().strip()

        # Fetch expirations
        with st.spinner("Loading options chain…"):
            expirations = _get_expirations(ticker_sb)

        with col_exp:
            if expirations:
                exp_choice = st.selectbox(
                    "Expiration", expirations, key="ob_expiry"
                )
            else:
                exp_choice = st.text_input(
                    "Expiration (YYYY-MM-DD)",
                    value=pd.Timestamp.today().strftime("%Y-%m-%d"),
                    key="ob_expiry_manual",
                )

        spot = _spot_from_ticker(ticker_sb)
        with col_price:
            spot = st.number_input(
                "Spot Price ($)", value=round(spot, 2), step=0.01, key="ob_spot"
            )

        # ---- Strategy template ----
        template_names = ["Custom"] + list(builder.TEMPLATES.keys())
        strategy_name = st.selectbox(
            "Strategy Template", template_names, key="ob_template"
        )

        T = _t_from_expiry(exp_choice if expirations else exp_choice)
        r = st.number_input(
            "Risk-Free Rate (%)", value=5.0, step=0.1, min_value=0.0, max_value=20.0,
            key="ob_rfr"
        ) / 100.0

        # IV: try to get from chain, else use historical
        with st.spinner("Estimating implied volatility…"):
            if expirations:
                sigma = _iv_from_chain(ticker_sb, exp_choice, spot)
            else:
                sigma = _sigma_from_ticker(ticker_sb)
        sigma = st.number_input(
            "Implied Volatility (%)",
            value=round(sigma * 100, 1),
            step=0.5,
            min_value=1.0,
            max_value=300.0,
            key="ob_sigma",
        ) / 100.0

        # ---- Build initial legs from template ----
        if strategy_name != "Custom":
            template_legs = builder.build_from_template(
                strategy_name, spot, T, r, sigma
            )
        else:
            template_legs = []

        # Session state for custom legs
        if "ob_custom_legs" not in st.session_state:
            st.session_state["ob_custom_legs"] = []

        # Reset custom legs when template changes
        if st.session_state.get("ob_last_template") != strategy_name:
            st.session_state["ob_custom_legs"] = []
            st.session_state["ob_last_template"] = strategy_name

        all_legs = template_legs + st.session_state["ob_custom_legs"]

        # ---- Legs display ----
        st.markdown("#### Strategy Legs")
        if all_legs:
            legs_df = pd.DataFrame(
                [
                    {
                        "Leg": i + 1,
                        "Type": leg["type"].capitalize(),
                        "Action": leg["action"].capitalize(),
                        "Strike": f"${leg['K']:.2f}",
                        "Qty": leg.get("qty", 1),
                        "Price": f"${leg['price']:.3f}",
                        "Delta": f"{leg['delta']:+.4f}",
                        "Gamma": f"{leg['gamma']:.6f}",
                        "Theta": f"{leg['theta']:+.4f}",
                        "Vega": f"{leg['vega']:.4f}",
                    }
                    for i, leg in enumerate(all_legs)
                ]
            )
            render_styled_table(legs_df)
        else:
            st.info("Select a template or add custom legs below.")

        # ---- Add Custom Leg ----
        with st.expander("➕ Add Custom Leg"):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                custom_type = st.selectbox(
                    "Type", ["call", "put", "stock"], key="ob_cleg_type"
                )
            with c2:
                custom_action = st.selectbox(
                    "Action", ["buy", "sell"], key="ob_cleg_action"
                )
            with c3:
                custom_strike = st.number_input(
                    "Strike ($)", value=round(spot, 2), step=0.5, key="ob_cleg_strike"
                )
            with c4:
                custom_qty = st.number_input(
                    "Qty", value=1, min_value=1, step=1, key="ob_cleg_qty"
                )
            with c5:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add Leg", key="ob_add_leg"):
                    new_leg = builder.build_leg(
                        spot, custom_strike, T, r, sigma,
                        custom_type, custom_action, qty=int(custom_qty)
                    )
                    new_leg["leg_num"] = len(all_legs) + 1
                    st.session_state["ob_custom_legs"].append(new_leg)
                    st.rerun()

            if st.button("Clear Custom Legs", key="ob_clear_legs"):
                st.session_state["ob_custom_legs"] = []
                st.rerun()

        # ---- Build Strategy ----
        if st.button("⚡ Build Strategy", type="primary", key="ob_build"):
            if not all_legs:
                show_error("No legs defined. Select a template or add custom legs.")
            else:
                _render_strategy_results(
                    builder, all_legs, spot, T, r, sigma, theme
                )

        # Auto-render if template was selected
        if all_legs and strategy_name != "Custom":
            _render_strategy_results(
                builder, all_legs, spot, T, r, sigma, theme
            )

    # ==================================================================
    # TAB 2 — Options Screener
    # ==================================================================
    with tab2:
        st.markdown("### Options Screener")

        sc1, sc2, sc3, sc4 = st.columns([2, 1, 1, 1])
        with sc1:
            screen_ticker = st.text_input(
                "Ticker", value=data.get("ticker", "SPY") if data else "SPY",
                key="screen_ticker"
            ).upper().strip()
        with sc2:
            screen_type = st.selectbox(
                "Option Type", ["both", "calls", "puts"], key="screen_type"
            )
        with sc3:
            min_vol = st.number_input(
                "Min Volume", value=100, min_value=0, step=10, key="screen_minvol"
            )
        with sc4:
            min_oi = st.number_input(
                "Min Open Interest", value=500, min_value=0, step=50, key="screen_minoi"
            )

        mc1, mc2 = st.columns(2)
        with mc1:
            money_min = st.slider(
                "Moneyness Min (K/S)", 0.70, 1.0, 0.85, step=0.01, key="screen_mmin"
            )
        with mc2:
            money_max = st.slider(
                "Moneyness Max (K/S)", 1.0, 1.30, 1.15, step=0.01, key="screen_mmax"
            )

        if st.button("🔍 Fetch & Screen", type="primary", key="screen_fetch"):
            with st.spinner(f"Fetching options chain for {screen_ticker}…"):
                try:
                    df_screen = screen_options_chain(
                        screen_ticker,
                        option_type=screen_type,
                        min_volume=int(min_vol),
                        min_oi=int(min_oi),
                        moneyness_min=money_min,
                        moneyness_max=money_max,
                    )
                except Exception as exc:
                    show_error(f"Could not fetch options: {exc}")
                    df_screen = pd.DataFrame()

            if not df_screen.empty:
                _render_screener_results(df_screen, theme)
            else:
                st.warning("No options matched the current filters.")


# ---------------------------------------------------------------------------
# Strategy results renderer (called from Tab 1)
# ---------------------------------------------------------------------------

def _render_strategy_results(
    builder: OptionsStrategyBuilder,
    legs: List[Dict],
    spot: float,
    T: float,
    r: float,
    sigma: float,
    theme: dict,
) -> None:
    """Compute and display all strategy analytics."""

    S_range = np.linspace(spot * 0.5, spot * 1.5, 500)
    net_prem = builder.net_premium(legs)
    metrics = builder.max_profit_loss(legs, S_range)
    greeks = builder.aggregate_greeks(legs)
    bks = builder.breakevens(legs, S_range)
    pnl = builder.payoff_at_expiry(legs, S_range)

    # ---- KPI row ----
    st.markdown("---")
    st.markdown("#### Strategy Summary")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Net Premium",
            f"{'−' if net_prem < 0 else ''}${abs(net_prem):.3f}",
            help="Negative = credit received; Positive = debit paid",
        )
    with k2:
        mp = metrics["max_profit"]
        st.metric(
            "Max Profit",
            f"${mp:.3f}" if not math.isinf(mp) else "Unlimited",
            delta=f"@ ${metrics['max_profit_price']:.2f}",
        )
    with k3:
        ml = metrics["max_loss"]
        st.metric(
            "Max Loss",
            f"${abs(ml):.3f}" if not math.isinf(ml) else "Unlimited",
            delta=f"@ ${metrics['max_loss_price']:.2f}",
            delta_color="inverse",
        )
    with k4:
        # Profit probability approximation via net delta (bounded 0–100%)
        net_delta = greeks["delta"]
        p_profit = min(max(abs(net_delta) * 100, 5.0), 95.0)
        st.metric("Est. Profit Prob.", f"{p_profit:.1f}%", help="Delta-based approximation")

    # ---- Greeks KPIs ----
    st.markdown("#### Aggregated Greeks")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Net Δ Delta", f"{greeks['delta']:+.4f}")
    g2.metric("Net Γ Gamma", f"{greeks['gamma']:+.6f}")
    g3.metric("Net Θ Theta", f"{greeks['theta']:+.4f}")
    g4.metric("Net ν Vega", f"{greeks['vega']:+.4f}")

    if bks:
        st.markdown(
            "**Breakeven(s):** " + " · ".join(f"${b:.2f}" for b in bks)
        )

    # ---- Payoff Diagram ----
    st.markdown("#### Payoff Diagram at Expiry")
    fig_payoff = _build_payoff_chart(S_range, pnl, spot, bks, theme)
    st.plotly_chart(fig_payoff, use_container_width=True)

    # ---- Greeks vs Price ----
    st.markdown("#### Greeks vs Underlying Price")
    greeks_df = builder.greeks_vs_price(legs, S_range, T, r, sigma)
    fig_greeks = _build_greeks_chart(greeks_df, spot, theme)
    st.plotly_chart(fig_greeks, use_container_width=True)


def _build_payoff_chart(
    S_range: np.ndarray,
    pnl: np.ndarray,
    spot: float,
    bks: List[float],
    theme: dict,
) -> go.Figure:
    """Build the payoff/P&L diagram with profit/loss colouring."""
    fig = go.Figure()

    # Profit zone (green fill)
    profit_mask = pnl >= 0
    loss_mask = pnl < 0

    # Profit fill
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=np.where(profit_mask, pnl, 0),
            fill="tozeroy",
            fillcolor="rgba(0,200,100,0.15)",
            line={"width": 0},
            name="Profit Zone",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Loss fill
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=np.where(loss_mask, pnl, 0),
            fill="tozeroy",
            fillcolor="rgba(220,50,50,0.15)",
            line={"width": 0},
            name="Loss Zone",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # P&L line coloured by sign
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=pnl,
            mode="lines",
            line={"color": "#00c864", "width": 2.5},
            name="P&L at Expiry",
            hovertemplate="Price: $%{x:.2f}<br>P&L: $%{y:.3f}<extra></extra>",
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(200,200,200,0.4)", line_width=1)

    # Current spot
    fig.add_vline(
        x=spot,
        line_dash="dot",
        line_color="#4da6ff",
        line_width=1.5,
        annotation_text=f"Spot ${spot:.2f}",
        annotation_position="top right",
        annotation_font_color="#4da6ff",
    )

    # Breakevens
    for bk in bks:
        fig.add_vline(
            x=bk,
            line_dash="dash",
            line_color="#ffaa00",
            line_width=1,
            annotation_text=f"BE ${bk:.2f}",
            annotation_position="top left",
            annotation_font_color="#ffaa00",
        )

    fig.update_layout(
        **theme,
        xaxis_title="Underlying Price ($)",
        yaxis_title="P&L ($) per Share",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin={"t": 40, "b": 40, "l": 60, "r": 30},
        height=420,
    )
    return fig


def _build_greeks_chart(
    greeks_df: pd.DataFrame,
    spot: float,
    theme: dict,
) -> go.Figure:
    """Build Delta and Gamma vs underlying price chart."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Net Delta vs Price", "Net Gamma vs Price"),
        horizontal_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(
            x=greeks_df["S"],
            y=greeks_df["delta"],
            mode="lines",
            line={"color": "#4da6ff", "width": 2},
            name="Net Delta",
            hovertemplate="$%{x:.2f} → Δ %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=greeks_df["S"],
            y=greeks_df["gamma"],
            mode="lines",
            line={"color": "#ff7f0e", "width": 2},
            name="Net Gamma",
            hovertemplate="$%{x:.2f} → Γ %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    for col_idx in (1, 2):
        fig.add_vline(
            x=spot,
            line_dash="dot",
            line_color="rgba(100,180,255,0.5)",
            line_width=1,
            row=1,  # type: ignore[call-arg]
            col=col_idx,  # type: ignore[call-arg]
        )

    fig.update_xaxes(title_text="Underlying Price ($)")
    fig.update_yaxes(title_text="Net Delta", row=1, col=1)
    fig.update_yaxes(title_text="Net Gamma", row=1, col=2)

    fig.update_layout(
        **theme,
        showlegend=False,
        margin={"t": 40, "b": 40, "l": 60, "r": 30},
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Screener results renderer (called from Tab 2)
# ---------------------------------------------------------------------------

def _render_screener_results(df: pd.DataFrame, theme: dict) -> None:
    """Display screened options chain with highlights and IV histogram."""
    spot = df["spot"].iloc[0] if "spot" in df.columns else None
    display_df = df.drop(columns=["spot"], errors="ignore").copy()

    st.markdown(f"**{len(display_df)} contracts matched** the filters.")

    # --- Highlight logic ---
    iv_80th = display_df["IV"].quantile(0.8) if "IV" in display_df.columns else None
    vol_spike_col = "Vol Spike" if "Vol Spike" in display_df.columns else None
    moneyness_col = "Moneyness" if "Moneyness" in display_df.columns else None

    def _row_style(row):
        styles = [""] * len(row)
        idx_map = {c: i for i, c in enumerate(row.index)}
        # IV > 80th pct → orange
        if iv_80th and "IV" in idx_map and row["IV"] > iv_80th:
            styles = ["background-color: rgba(255,140,0,0.25)"] * len(row)
        # Volume spike → yellow (overrides orange)
        if vol_spike_col and "Vol Spike" in idx_map and row["Vol Spike"]:
            styles = ["background-color: rgba(255,220,0,0.20)"] * len(row)
        # Near ATM (moneyness 0.98–1.02) → blue
        if moneyness_col and "Moneyness" in idx_map:
            m = row["Moneyness"]
            if 0.98 <= m <= 1.02:
                styles = ["background-color: rgba(77,166,255,0.20)"] * len(row)
        return styles

    # Format display columns
    fmt_df = display_df.copy()
    for col in ["IV", "Moneyness"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    for col in ["Last", "bid", "ask", "mid"]:
        if col in fmt_df.columns:
            fmt_df[col] = fmt_df[col].map(lambda x: f"${x:.3f}" if pd.notnull(x) else "")

    styled = fmt_df.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, height=400)

    # Legend
    st.markdown(
        '<span style="background:rgba(255,140,0,0.4);padding:2px 8px;border-radius:3px">■</span> '
        "IV > 80th pct &nbsp;&nbsp;"
        '<span style="background:rgba(255,220,0,0.4);padding:2px 8px;border-radius:3px">■</span> '
        "Volume Spike &nbsp;&nbsp;"
        '<span style="background:rgba(77,166,255,0.4);padding:2px 8px;border-radius:3px">■</span> '
        "Near ATM",
        unsafe_allow_html=True,
    )

    # ---- IV Distribution Histogram ----
    if "IV" in display_df.columns:
        st.markdown("#### IV Distribution")
        iv_vals = pd.to_numeric(display_df["IV"], errors="coerce").dropna()
        if len(iv_vals) > 1:
            fig_hist = go.Figure()
            fig_hist.add_trace(
                go.Histogram(
                    x=iv_vals,
                    nbinsx=40,
                    marker_color="#4da6ff",
                    opacity=0.8,
                    name="IV",
                    hovertemplate="IV: %{x:.3f}<br>Count: %{y}<extra></extra>",
                )
            )
            if iv_80th:
                fig_hist.add_vline(
                    x=iv_80th,
                    line_dash="dash",
                    line_color="#ff8c00",
                    annotation_text=f"80th pct: {iv_80th:.3f}",
                    annotation_position="top right",
                    annotation_font_color="#ff8c00",
                )
            fig_hist.update_layout(
                **theme,
                xaxis_title="Implied Volatility",
                yaxis_title="Count",
                showlegend=False,
                margin={"t": 30, "b": 40, "l": 60, "r": 30},
                height=300,
            )
            st.plotly_chart(fig_hist, use_container_width=True)


# ─── RISK_SUITE ───
"""
module_risk_suite.py
====================
Portfolio Risk Management Suite for QuantLab (Streamlit quant finance app).

Provides:
  - StressTester  : historical scenarios, custom shocks, factor VaR, correlation breakdown.
  - render_risk_suite_tab(data) : full Streamlit UI (4 sections).
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS = 252
RF_RATE = 0.045  # 4.5% annual risk-free rate

_COLORS = {
    "primary":   "#00D4FF",
    "secondary": "#7C3AED",
    "positive":  "#00C49A",
    "negative":  "#FF6B6B",
    "warning":   "#F59E0B",
    "neutral":   "#94A3B8",
    "accent1":   "#F472B6",
    "accent2":   "#34D399",
}

HISTORICAL_SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Lehman collapse — S&P 500 -55%, VIX peaked at 80.",
        "equity_shock": -0.55,
        "vol_multiplier": 3.0,
        "rate_shock": -0.03,
        "period": ("2008-09-01", "2009-03-31"),
    },
    "COVID Crash (Mar 2020)": {
        "description": "Pandemic sell-off — S&P 500 -34% in 33 days.",
        "equity_shock": -0.34,
        "vol_multiplier": 5.0,
        "rate_shock": -0.015,
        "period": ("2020-02-19", "2020-03-23"),
    },
    "2022 Rate Shock": {
        "description": "Fed raised rates 425bps — S&P 500 -25%, bonds -20%.",
        "equity_shock": -0.25,
        "vol_multiplier": 1.5,
        "rate_shock": 0.04,
        "period": ("2022-01-01", "2022-12-31"),
    },
    "Dot-com Crash (2000-02)": {
        "description": "Tech bubble burst — NASDAQ -78%, S&P 500 -49%.",
        "equity_shock": -0.49,
        "vol_multiplier": 2.0,
        "rate_shock": -0.02,
        "period": ("2000-03-10", "2002-10-09"),
    },
    "Black Monday 1987": {
        "description": "Single-day crash — -22.6% on Oct 19, 1987.",
        "equity_shock": -0.23,
        "vol_multiplier": 8.0,
        "rate_shock": 0.01,
        "period": None,  # parametric only
    },
    "Flash Crash 2010": {
        "description": "May 6, 2010 — -10% intraday, recovered same day.",
        "equity_shock": -0.10,
        "vol_multiplier": 4.0,
        "rate_shock": 0.0,
        "period": None,
    },
}

# ---------------------------------------------------------------------------
# Shim helpers – fall back gracefully when run outside the main app
# ---------------------------------------------------------------------------


def _get_plotly_theme(theme: str = "dark") -> dict:
    """Return Plotly layout kwargs matching the app theme."""
    try:
        from app import _get_plotly_theme as _app_theme  # type: ignore
        return _app_theme(theme)
    except ImportError:
        pass
    is_dark = theme == "dark"
    bg         = "#0E1117" if is_dark else "#FFFFFF"
    paper      = "#161B22" if is_dark else "#F8F9FA"
    font_color = "#FAFAFA" if is_dark else "#1A1A2E"
    grid_color = "rgba(255,255,255,0.08)" if is_dark else "rgba(0,0,0,0.08)"
    return dict(
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor=bg,
        paper_bgcolor=paper,
        font=dict(color=font_color, size=12),
        xaxis=dict(gridcolor=grid_color, showgrid=True),
        yaxis=dict(gridcolor=grid_color, showgrid=True),
        margin=dict(l=60, r=20, t=50, b=50),
        height=420,
    )




# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _equal_weight(n: int) -> np.ndarray:
    """Return equal-weight vector of length *n*."""
    return np.full(n, 1.0 / n)


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-like pct_change returns, NaN rows dropped."""
    return prices.pct_change().dropna(how="all")


def _estimate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """OLS beta of asset_returns on market_returns (aligned)."""
    combined = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(combined) < 10:
        return 1.0
    x = combined.iloc[:, 1].values
    y = combined.iloc[:, 0].values
    cov_xy = np.cov(x, y, ddof=1)
    var_x  = np.var(x, ddof=1)
    if var_x < 1e-12:
        return 1.0
    return float(cov_xy[0, 1] / var_x)


def _compute_max_drawdown(returns_series: pd.Series) -> float:
    """Maximum drawdown from a series of period returns."""
    cum = (1 + returns_series).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min()) if not dd.empty else 0.0


def _historical_var(portfolio_returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical-simulation VaR (negative number means loss)."""
    if portfolio_returns.empty:
        return 0.0
    return float(np.percentile(portfolio_returns.dropna(), (1 - confidence) * 100))


# ---------------------------------------------------------------------------
# StressTester
# ---------------------------------------------------------------------------


class StressTester:
    """
    Portfolio stress testing suite:
      - Historical scenario replay (or parametric fallback).
      - Custom scenario estimation (equity + rate + vol shocks).
      - Factor VaR decomposition (market vs idiosyncratic).
      - Correlation breakdown detector (crisis vs normal).
    """

    # ------------------------------------------------------------------
    # 1.  run_historical_scenario
    # ------------------------------------------------------------------

    def run_historical_scenario(
        self,
        prices: pd.DataFrame,
        weights: np.ndarray,
        scenario_name: str,
    ) -> dict:
        """
        Run a named historical stress scenario on the portfolio.

        If the scenario has a ``period`` **and** the prices DataFrame spans
        that period, actual asset returns from the window are used
        (method='historical').  Otherwise the parametric equity / rate shocks
        are applied via each asset's estimated beta (method='parametric').

        Returns
        -------
        dict with keys:
          portfolio_return, max_drawdown, volatility, var_95,
          per_ticker {ticker: estimated_return}, method
        """
        scenario = HISTORICAL_SCENARIOS.get(scenario_name)
        if scenario is None:
            return {
                "portfolio_return": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "var_95": 0.0,
                "per_ticker": {},
                "method": "parametric",
            }

        tickers = prices.columns.tolist()
        n = len(tickers)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n:
            weights = _equal_weight(n)

        period = scenario.get("period")
        equity_shock: float = scenario["equity_shock"]
        rate_shock: float = scenario["rate_shock"]
        vol_mult: float = scenario["vol_multiplier"]

        method = "parametric"
        per_ticker: dict[str, float] = {}

        # ---- Try historical window ----
        if period is not None:
            start_str, end_str = period
            try:
                start_dt = pd.Timestamp(start_str)
                end_dt   = pd.Timestamp(end_str)
                price_index = prices.index
                if (
                    price_index.min() <= start_dt
                    and price_index.max() >= end_dt
                ):
                    window = prices.loc[start_dt:end_dt]
                    if len(window) >= 5:
                        window_returns = _compute_returns(window)
                        # Cumulative return per ticker over the window
                        for t in tickers:
                            if t in window_returns.columns:
                                cum = (1 + window_returns[t].dropna()).prod() - 1.0
                                per_ticker[t] = float(cum)
                            else:
                                per_ticker[t] = 0.0
                        method = "historical"
            except Exception:
                pass  # fall through to parametric

        # ---- Parametric fallback ----
        if method == "parametric":
            returns = _compute_returns(prices)
            # Build a proxy "market" return series: equal-weight portfolio
            mkt_returns = returns.mean(axis=1)

            for t in tickers:
                if t not in returns.columns:
                    per_ticker[t] = equity_shock
                    continue
                asset_ret = returns[t].dropna()
                beta = _estimate_beta(asset_ret, mkt_returns)
                # Equity component
                equity_component = beta * equity_shock
                # Rate component: rolling correlation with a synthetic rate proxy
                # (we approximate rate sensitivity as negative corr with market × rate_shock)
                rate_sensitivity = -beta * 0.3  # conservative heuristic
                estimated = equity_component + rate_sensitivity * rate_shock
                # Clip to realistic bounds
                estimated = float(np.clip(estimated, -0.99, 2.0))
                per_ticker[t] = estimated

        # ---- Portfolio-level aggregation ----
        ticker_rets = np.array([per_ticker.get(t, 0.0) for t in tickers])
        portfolio_return = float(np.dot(weights, ticker_rets))

        # Simulate a daily path to get drawdown / vol metrics
        # Use historical vol if available, else approximate
        returns = _compute_returns(prices)
        if not returns.empty:
            port_daily = returns[tickers].fillna(0.0).dot(weights)
            # Scale the daily path by the scenario return
            scaling = (1 + portfolio_return) / max(
                float((1 + port_daily).prod()), 1e-6
            )
            # Volatility during scenario: scale up by vol_multiplier
            base_vol = float(port_daily.std()) * np.sqrt(TRADING_DAYS)
            scenario_vol = base_vol * vol_mult

            # Max drawdown: simulate worst-case under scenario shock
            max_drawdown = min(portfolio_return, _compute_max_drawdown(port_daily) * vol_mult)
            max_drawdown = float(np.clip(max_drawdown, -0.99, 0.0))

            # VaR 95% under scenario (parametric, adjusted)
            mu_daily = portfolio_return / max(len(port_daily), 1)
            sigma_daily = float(port_daily.std()) * vol_mult
            var_95 = float(mu_daily - 1.645 * sigma_daily)
        else:
            scenario_vol = abs(portfolio_return) * vol_mult
            max_drawdown = portfolio_return
            var_95 = portfolio_return * 1.1

        return {
            "portfolio_return": portfolio_return,
            "max_drawdown": max_drawdown,
            "volatility": scenario_vol,
            "var_95": var_95,
            "per_ticker": per_ticker,
            "method": method,
        }

    # ------------------------------------------------------------------
    # 2.  run_custom_scenario
    # ------------------------------------------------------------------

    def run_custom_scenario(
        self,
        prices: pd.DataFrame,
        weights: np.ndarray,
        equity_change: float,
        rate_change: float,
        vol_multiplier: float,
    ) -> dict:
        """
        Estimate portfolio impact of user-specified macro shocks.

        equity_change   : fractional shock, e.g. -0.20 for -20 %
        rate_change     : fractional shock in decimal, e.g. 0.01 for +100 bps
        vol_multiplier  : multiplier on realised vol, e.g. 2.0

        Returns
        -------
        dict with keys:
          portfolio_impact, per_ticker {ticker: estimated_return},
          volatility_estimate, var_95_estimate
        """
        tickers = prices.columns.tolist()
        n = len(tickers)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n:
            weights = _equal_weight(n)

        returns = _compute_returns(prices)
        if returns.empty:
            return {
                "portfolio_impact": equity_change,
                "per_ticker": {t: equity_change for t in tickers},
                "volatility_estimate": abs(equity_change) * vol_multiplier,
                "var_95_estimate": equity_change * 1.1,
            }

        # Market proxy: equal-weight portfolio of available assets
        mkt_returns = returns.mean(axis=1)

        per_ticker: dict[str, float] = {}
        for t in tickers:
            if t not in returns.columns:
                per_ticker[t] = equity_change
                continue

            asset_ret = returns[t].dropna()
            if len(asset_ret) < 10:
                per_ticker[t] = equity_change
                continue

            # Equity beta
            equity_beta = _estimate_beta(asset_ret, mkt_returns)

            # Rate sensitivity: rolling correlation with simulated rate changes
            # We proxy 10Y yield changes with the NEGATIVE of a long-duration bond
            # approximation: -duration × Δrate. For a simple proxy, use -corr(asset, mkt)
            # to simulate interest-rate sensitivity direction.
            aligned = pd.concat([asset_ret, mkt_returns], axis=1).dropna()
            if len(aligned) > 20:
                # Rate sensitivity via direct correlation estimation
                # More negative beta assets tend to be bond-like (rate sensitive)
                rate_corr = float(aligned.iloc[:, 0].rolling(min(63, len(aligned))).corr(
                    aligned.iloc[:, 1]
                ).dropna().iloc[-1]) if len(aligned) >= 5 else 0.0
                # Heuristic: bond-like assets (negative equity corr) amplify rate impact
                rate_sensitivity = -rate_corr * 0.5
            else:
                rate_sensitivity = -equity_beta * 0.3

            estimated = (
                equity_beta * equity_change
                + rate_sensitivity * rate_change
            )
            estimated = float(np.clip(estimated, -0.99, 2.0))
            per_ticker[t] = estimated

        ticker_impacts = np.array([per_ticker.get(t, equity_change) for t in tickers])
        portfolio_impact = float(np.dot(weights, ticker_impacts))

        # Adjusted portfolio volatility
        port_daily = returns[tickers].fillna(0.0).dot(weights)
        base_vol = float(port_daily.std()) * np.sqrt(TRADING_DAYS)
        vol_estimate = base_vol * vol_multiplier

        # VaR 95% estimate under scenario
        mu_daily = float(port_daily.mean())
        sigma_daily = float(port_daily.std()) * vol_multiplier
        var_95 = float(mu_daily - 1.645 * sigma_daily)

        return {
            "portfolio_impact": portfolio_impact,
            "per_ticker": per_ticker,
            "volatility_estimate": vol_estimate,
            "var_95_estimate": var_95,
        }

    # ------------------------------------------------------------------
    # 3.  factor_var
    # ------------------------------------------------------------------

    def factor_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
    ) -> dict:
        """
        Decompose portfolio VaR into market and idiosyncratic components.

        Uses a single-factor market model where the factor is the equal-weight
        portfolio return.  Residuals from this model are treated as idiosyncratic.

        Returns
        -------
        dict with keys:
          total_var_95, total_var_99,
          market_var_95, idio_var_95,
          per_ticker_marginal {ticker: marginal_var_contribution}
        """
        tickers = returns.columns.tolist()
        n = len(tickers)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n:
            weights = _equal_weight(n)

        rets = returns[tickers].fillna(0.0)
        port_returns = rets.dot(weights)

        # ---- Total VaR (historical simulation) ----
        total_var_95 = _historical_var(port_returns, 0.95)
        total_var_99 = _historical_var(port_returns, 0.99)

        # ---- Market factor: equal-weight portfolio ----
        mkt_factor = rets.mean(axis=1)

        # ---- Beta of portfolio vs market factor ----
        port_beta = _estimate_beta(port_returns, mkt_factor)

        # ---- Market VaR (via beta-scaled market returns) ----
        mkt_var_95 = float(np.percentile(mkt_factor.dropna(), 5)) * port_beta

        # ---- Idiosyncratic VaR: residual from market model ----
        aligned = pd.concat([port_returns, mkt_factor], axis=1).dropna()
        if len(aligned) > 10:
            x = aligned.iloc[:, 1].values
            y = aligned.iloc[:, 0].values
            # OLS residuals
            coeffs = np.polyfit(x, y, 1)
            residuals = y - (coeffs[0] * x + coeffs[1])
            idio_var_95 = float(np.percentile(residuals, 5))
        else:
            idio_var_95 = total_var_95 - mkt_var_95

        # Ensure idio_var_95 is sensible
        if abs(mkt_var_95) + abs(idio_var_95) < 1e-10:
            mkt_var_95 = total_var_95 * 0.7
            idio_var_95 = total_var_95 * 0.3

        # ---- Per-ticker marginal VaR contribution ----
        # Marginal VaR: partial derivative of portfolio VaR w.r.t. weight i
        # Approximated as: weight_i × Cov(r_i, r_port) / std(r_port) × z-score
        cov_matrix = rets.cov().values
        port_var_scalar = float(weights @ cov_matrix @ weights)
        port_std = np.sqrt(max(port_var_scalar, 1e-12))

        z_95 = 1.645
        marginal_var: dict[str, float] = {}
        for i, t in enumerate(tickers):
            cov_i_port = float(cov_matrix[i] @ weights)
            # Scaled contribution
            marginal_var[t] = float(weights[i] * cov_i_port / port_std * z_95)

        # Normalise contributions to sum to |total_var_95|
        total_mvar = sum(abs(v) for v in marginal_var.values())
        if total_mvar > 1e-10:
            scale = abs(total_var_95) / total_mvar
            marginal_var = {t: v * scale for t, v in marginal_var.items()}

        return {
            "total_var_95": total_var_95,
            "total_var_99": total_var_99,
            "market_var_95": mkt_var_95,
            "idio_var_95": idio_var_95,
            "per_ticker_marginal": marginal_var,
        }

    # ------------------------------------------------------------------
    # 4.  correlation_breakdown
    # ------------------------------------------------------------------

    def correlation_breakdown(
        self,
        prices: pd.DataFrame,
        lookback_normal: int = 252,
        lookback_crisis: int = 21,
    ) -> dict:
        """
        Detect correlation regime changes between the normal period and
        the most recent crisis window.

        Parameters
        ----------
        lookback_normal : days for the 'normal' correlation baseline (default 252)
        lookback_crisis : days for the 'current' correlation window (default 21)

        Returns
        -------
        dict with keys:
          normal_corr     : pd.DataFrame  (full lookback_normal window)
          current_corr    : pd.DataFrame  (last lookback_crisis days)
          avg_normal      : float
          avg_current     : float
          spike_pairs     : list of (t1, t2, normal_corr, current_corr, change)
          alert           : bool (True if avg correlation rose > 0.2)
        """
        returns = _compute_returns(prices)
        tickers = returns.columns.tolist()

        if len(returns) < lookback_crisis + 2:
            empty_corr = pd.DataFrame(
                np.eye(len(tickers)), index=tickers, columns=tickers
            )
            return {
                "normal_corr": empty_corr,
                "current_corr": empty_corr,
                "avg_normal": 0.0,
                "avg_current": 0.0,
                "spike_pairs": [],
                "alert": False,
            }

        # Normal period: last lookback_normal days
        n_rows = len(returns)
        normal_window = returns.iloc[-min(lookback_normal, n_rows):]
        crisis_window = returns.iloc[-min(lookback_crisis, n_rows):]

        # Compute correlation matrices
        normal_corr = normal_window.corr()
        current_corr = crisis_window.corr()

        # Replace NaN with 0 (single-asset edge case)
        normal_corr = normal_corr.fillna(0.0)
        current_corr = current_corr.fillna(0.0)

        # Average off-diagonal correlations
        n = len(tickers)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            avg_normal  = float(normal_corr.values[mask].mean())
            avg_current = float(current_corr.values[mask].mean())
        else:
            avg_normal = 0.0
            avg_current = 0.0

        # Spike pairs: |change| > 0.3
        spike_pairs: list[tuple] = []
        for i, t1 in enumerate(tickers):
            for j, t2 in enumerate(tickers):
                if j <= i:
                    continue
                nc = float(normal_corr.loc[t1, t2])
                cc = float(current_corr.loc[t1, t2])
                change = cc - nc
                if abs(change) > 0.3:
                    spike_pairs.append((t1, t2, nc, cc, change))

        # Sort by absolute change, largest first
        spike_pairs.sort(key=lambda x: abs(x[4]), reverse=True)

        alert = (avg_current - avg_normal) > 0.2

        return {
            "normal_corr": normal_corr,
            "current_corr": current_corr,
            "avg_normal": avg_normal,
            "avg_current": avg_current,
            "spike_pairs": spike_pairs,
            "alert": alert,
        }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_risk_suite_tab(data: dict) -> None:
    """
    Render the full Portfolio Risk Management Suite tab in Streamlit.

    Parameters
    ----------
    data : dict
        Must contain:
          - 'prices'    : pd.DataFrame  (DatetimeIndex, columns = tickers)
          - 'tickers'   : list[str]
          - 'portfolio' : dict  { strategy_name: np.ndarray of weights }
        Optional:
          - 'theme'     : str  "dark" | "light"
          - 'rf_rate'   : float
    """
    theme    = data.get("theme", st.session_state.get("theme", "dark"))
    rf_rate  = float(data.get("rf_rate", RF_RATE))
    plotly_t = _get_plotly_theme(theme)

    is_dark   = theme == "dark"
    neg_color = _COLORS["negative"]
    pos_color = _COLORS["positive"]
    pri_color = _COLORS["primary"]
    neu_color = _COLORS["neutral"]
    warn_color = _COLORS["warning"]

    st.markdown("## Portfolio Risk Management Suite")
    st.markdown(
        "Stress testing, scenario analysis, factor VaR decomposition, "
        "and correlation regime monitoring."
    )

    # ---- Validate inputs ----
    prices: pd.DataFrame = data.get("prices", pd.DataFrame())
    if prices is None or prices.empty:
        show_error("No price data available. Please load a portfolio first.")
        return

    tickers: list[str] = data.get("tickers", prices.columns.tolist())
    if not tickers:
        show_error("No tickers found in the data dictionary.")
        return

    available = [t for t in tickers if t in prices.columns]
    if not available:
        show_error("None of the provided tickers are present in the price data.")
        return
    prices  = prices[available].copy()
    tickers = available

    portfolio_dict: dict = data.get("portfolio", {})
    returns = _compute_returns(prices)

    # Determine base weights (first strategy or equal weight)
    if portfolio_dict:
        first_strategy = next(iter(portfolio_dict))
        base_weights = np.asarray(portfolio_dict[first_strategy], dtype=float)
        if len(base_weights) != len(tickers):
            base_weights = _equal_weight(len(tickers))
    else:
        first_strategy = "Equal Weight"
        base_weights = _equal_weight(len(tickers))

    tester = StressTester()

    # =========================================================================
    # Section 1 — Historical Stress Tests
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 1 — Historical Stress Tests")
    st.markdown(
        "Estimated portfolio impact under six historical market crises. "
        "Where price history covers the event window, actual returns are used; "
        "otherwise parametric beta-scaling is applied."
    )

    try:
        with st.spinner("Running all historical scenarios…"):
            scenario_results: list[dict] = []
            for sname, sdata in HISTORICAL_SCENARIOS.items():
                result = tester.run_historical_scenario(prices, base_weights, sname)
                scenario_results.append(
                    {
                        "scenario_name": sname,
                        "description": sdata["description"],
                        "portfolio_return": result["portfolio_return"],
                        "method": result["method"],
                        "details": result,
                    }
                )

        # ---- Summary table ----
        st.markdown("#### Scenario Summary")
        table_df = pd.DataFrame(
            [
                {
                    "Scenario": r["scenario_name"],
                    "Description": r["description"],
                    "Est. Portfolio Return": f"{r['portfolio_return']:.1%}",
                    "Method": r["method"].capitalize(),
                }
                for r in scenario_results
            ]
        )
        render_styled_table(table_df, key="stress_summary_table")

        # ---- Horizontal bar chart ----
        scenario_names  = [r["scenario_name"] for r in scenario_results]
        port_returns    = [r["portfolio_return"] for r in scenario_results]
        bar_colors      = [neg_color if v < 0 else pos_color for v in port_returns]
        pct_labels      = [f"{v:.1%}" for v in port_returns]

        fig_stress = go.Figure(
            go.Bar(
                y=scenario_names,
                x=port_returns,
                orientation="h",
                marker_color=bar_colors,
                text=pct_labels,
                textposition="outside",
                hovertemplate="%{y}<br>Portfolio Return: %{x:.1%}<extra></extra>",
            )
        )
        fig_stress.update_layout(
            **plotly_t,
            title="Estimated Portfolio Return by Scenario",
            xaxis=dict(
                title="Estimated Portfolio Return",
                tickformat=".0%",
                gridcolor=plotly_t.get("xaxis", {}).get("gridcolor", "rgba(255,255,255,0.08)"),
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
                gridcolor=plotly_t.get("yaxis", {}).get("gridcolor", "rgba(255,255,255,0.08)"),
            ),
            height=380,
            margin=dict(l=220, r=80, t=60, b=50),
        )
        fig_stress.add_vline(x=0, line_color=neu_color, line_width=1, line_dash="dash")
        st.plotly_chart(fig_stress, use_container_width=True)

        # ---- Per-ticker impact expander for worst scenario ----
        worst_idx = int(np.argmin(port_returns))
        worst     = scenario_results[worst_idx]
        with st.expander(
            f"Per-Ticker Impact — Worst Scenario: {worst['scenario_name']} "
            f"({worst['portfolio_return']:.1%})"
        ):
            pt = worst["details"]["per_ticker"]
            if pt:
                pt_df = pd.DataFrame(
                    [
                        {
                            "Ticker": t,
                            "Estimated Return": f"{v:.1%}",
                            "Impact Direction": "▼ Loss" if v < 0 else "▲ Gain",
                        }
                        for t, v in sorted(pt.items(), key=lambda x: x[1])
                    ]
                )
                render_styled_table(pt_df, key="worst_scenario_per_ticker")

                fig_pt = go.Figure(
                    go.Bar(
                        x=[t for t in sorted(pt, key=pt.get)],
                        y=sorted(pt.values()),
                        marker_color=[
                            neg_color if v < 0 else pos_color
                            for v in sorted(pt.values())
                        ],
                        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
                    )
                )
                fig_pt.update_layout(
                    **plotly_t,
                    title=f"Per-Ticker Impact: {worst['scenario_name']}",
                    xaxis_title="Ticker",
                    yaxis=dict(title="Estimated Return", tickformat=".0%"),
                    height=340,
                )
                fig_pt.add_hline(y=0, line_color=neu_color, line_width=1, line_dash="dash")
                st.plotly_chart(fig_pt, use_container_width=True)
            else:
                st.info("No per-ticker data available for this scenario.")

    except Exception as exc:
        show_error(f"Historical Stress Test section error: {exc}")

    # =========================================================================
    # Section 2 — Custom Scenario (Real-Time)
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 2 — Custom Scenario")
    st.markdown(
        "Adjust macro shocks with the sliders below and instantly see the "
        "estimated portfolio impact."
    )

    try:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            equity_pct = st.slider(
                "Equity Market Change (%)",
                min_value=-50,
                max_value=50,
                value=-20,
                step=5,
                key="rs_equity_slider",
            )
        with col_s2:
            rate_bps = st.slider(
                "Rate Change (bps)",
                min_value=-300,
                max_value=300,
                value=100,
                step=25,
                key="rs_rate_slider",
            )
        with col_s3:
            vol_mult = st.slider(
                "Volatility Multiplier",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="rs_vol_slider",
        )

        equity_change = equity_pct / 100.0
        rate_change   = rate_bps / 10_000.0  # bps → decimal

        with st.spinner("Estimating portfolio impact…"):
            custom_result = tester.run_custom_scenario(
                prices, base_weights, equity_change, rate_change, vol_mult
            )

        portfolio_impact = custom_result["portfolio_impact"]
        per_ticker_custom = custom_result["per_ticker"]
        vol_estimate      = custom_result["volatility_estimate"]
        var_95_est        = custom_result["var_95_estimate"]

        # ---- Live output ----
        m1, m2, m3 = st.columns(3)
        impact_color = neg_color if portfolio_impact < 0 else pos_color
        with m1:
            st.metric(
                "Portfolio Impact",
                f"{portfolio_impact:.2%}",
                delta=None,
                help="Estimated total portfolio return under the scenario.",
            )
        with m2:
            st.metric(
                "Scenario Volatility (Ann.)",
                f"{vol_estimate:.2%}",
                help="Annualised portfolio volatility under the vol multiplier.",
            )
        with m3:
            st.metric(
                "VaR 95% (Daily Est.)",
                f"{var_95_est:.2%}",
                help="Estimated daily VaR at 95% confidence under scenario vol.",
            )

        # Gauge chart for portfolio impact
        gauge_max = max(abs(portfolio_impact) * 1.5, 0.05)
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=portfolio_impact * 100,
                delta={"reference": 0, "valueformat": ".1f", "suffix": "%"},
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {
                        "range": [-gauge_max * 100, gauge_max * 100],
                        "ticksuffix": "%",
                    },
                    "bar": {"color": impact_color},
                    "steps": [
                        {"range": [-gauge_max * 100, 0], "color": "rgba(255,107,107,0.15)"},
                        {"range": [0, gauge_max * 100], "color": "rgba(0,196,154,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": warn_color, "width": 3},
                        "thickness": 0.75,
                        "value": portfolio_impact * 100,
                    },
                },
                title={"text": "Portfolio Impact"},
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor=plotly_t.get("paper_bgcolor", "#161B22"),
            font=dict(color=plotly_t.get("font", {}).get("color", "#FAFAFA"), size=13),
            height=300,
            margin=dict(l=30, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Per-ticker table
        if per_ticker_custom:
            st.markdown("#### Per-Ticker Impact")
            pt_custom_df = pd.DataFrame(
                [
                    {
                        "Ticker": t,
                        "Estimated Return": f"{v:.2%}",
                        "Equity Shock": f"{equity_pct:+d}%",
                        "Rate Shock": f"{rate_bps:+d} bps",
                    }
                    for t, v in sorted(per_ticker_custom.items(), key=lambda x: x[1])
                ]
            )
            render_styled_table(pt_custom_df, key="custom_per_ticker_table")

        # ---- What-if mode ----
        st.markdown("---")
        st.markdown("#### What-If Comparison (3 Scenarios Side-by-Side)")
        st.caption("Define up to 3 custom scenarios to compare simultaneously.")

        wi_cols = st.columns(3)
        whatif_scenarios: list[dict] = []
        default_params = [
            {"eq": -20, "rate": 100, "vol": 2.0, "label": "Bear Case"},
            {"eq": -10, "rate": 50,  "vol": 1.5, "label": "Mild Stress"},
            {"eq": 10,  "rate": -50, "vol": 0.8, "label": "Recovery"},
        ]
        for i, (col, dp) in enumerate(zip(wi_cols, default_params)):
            with col:
                st.markdown(f"**Scenario {i + 1}**")
                wi_label = st.text_input(
                    "Label", value=dp["label"], key=f"wi_label_{i}"
                )
                wi_eq = st.number_input(
                    "Equity (%)", value=dp["eq"], step=5,
                    min_value=-100, max_value=100, key=f"wi_eq_{i}"
                )
                wi_rate = st.number_input(
                    "Rate (bps)", value=dp["rate"], step=25,
                    min_value=-500, max_value=500, key=f"wi_rate_{i}"
                )
                wi_vol = st.number_input(
                    "Vol Mult", value=dp["vol"], step=0.5,
                    min_value=0.1, max_value=10.0, key=f"wi_vol_{i}"
                )
                whatif_scenarios.append(
                    {
                        "label": wi_label,
                        "equity": wi_eq / 100.0,
                        "rate": wi_rate / 10_000.0,
                        "vol": wi_vol,
                    }
                )

        wi_results: list[dict] = []
        for ws in whatif_scenarios:
            r = tester.run_custom_scenario(
                prices, base_weights, ws["equity"], ws["rate"], ws["vol"]
            )
            wi_results.append({"label": ws["label"], **r})

        wi_labels  = [r["label"] for r in wi_results]
        wi_impacts = [r["portfolio_impact"] for r in wi_results]
        wi_vols    = [r["volatility_estimate"] for r in wi_results]
        wi_vars    = [r["var_95_estimate"] for r in wi_results]

        fig_wi = go.Figure()
        fig_wi.add_trace(
            go.Bar(
                name="Portfolio Impact",
                x=wi_labels,
                y=wi_impacts,
                marker_color=[neg_color if v < 0 else pos_color for v in wi_impacts],
                text=[f"{v:.1%}" for v in wi_impacts],
                textposition="outside",
            )
        )
        fig_wi.add_trace(
            go.Scatter(
                name="VaR 95%",
                x=wi_labels,
                y=wi_vars,
                mode="markers+lines",
                marker=dict(color=warn_color, size=10),
                line=dict(color=warn_color, dash="dot"),
                yaxis="y",
            )
        )
        fig_wi.update_layout(
            **plotly_t,
            title="What-If Scenario Comparison",
            yaxis=dict(title="Return / VaR", tickformat=".0%"),
            barmode="group",
            height=380,
        )
        fig_wi.add_hline(y=0, line_color=neu_color, line_width=1, line_dash="dash")
        st.plotly_chart(fig_wi, use_container_width=True)

    except Exception as exc:
        show_error(f"Custom Scenario section error: {exc}")

    # =========================================================================
    # Section 3 — Factor VaR Decomposition
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 3 — Factor VaR Decomposition")
    st.markdown(
        "Portfolio VaR decomposed into systematic (market) and idiosyncratic risk. "
        "Marginal contribution per ticker identifies the largest risk drivers."
    )

    try:
        if returns.empty or len(returns) < 20:
            show_error("Insufficient return history for VaR decomposition (need ≥ 20 days).")
        else:
            with st.spinner("Decomposing portfolio VaR…"):
                var_result = tester.factor_var(returns[tickers], base_weights)

            total_var_95 = var_result["total_var_95"]
            total_var_99 = var_result["total_var_99"]
            mkt_var      = var_result["market_var_95"]
            idio_var     = var_result["idio_var_95"]
            per_ticker_mvar = var_result["per_ticker_marginal"]

            # ---- Large metric cards ----
            v1, v2, v3, v4 = st.columns(4)
            with v1:
                st.metric("Portfolio VaR 95%", f"{total_var_95:.2%}",
                          help="Historical simulation VaR at 95% confidence.")
            with v2:
                st.metric("Portfolio VaR 99%", f"{total_var_99:.2%}",
                          help="Historical simulation VaR at 99% confidence.")
            with v3:
                st.metric("Market VaR 95%", f"{mkt_var:.2%}",
                          help="Systematic (market-factor) component of VaR.")
            with v4:
                st.metric("Idiosyncratic VaR", f"{idio_var:.2%}",
                          help="Residual component not explained by the market factor.")

            # ---- Donut chart: Market vs Idiosyncratic ----
            mkt_abs  = abs(mkt_var)
            idio_abs = abs(idio_var)
            total_abs = mkt_abs + idio_abs if (mkt_abs + idio_abs) > 1e-10 else 1.0

            fig_donut = go.Figure(
                go.Pie(
                    labels=["Market VaR", "Idiosyncratic VaR"],
                    values=[mkt_abs / total_abs, idio_abs / total_abs],
                    hole=0.55,
                    marker=dict(colors=[pri_color, _COLORS["accent1"]]),
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{percent}<extra></extra>",
                )
            )
            fig_donut.update_layout(
                paper_bgcolor=plotly_t.get("paper_bgcolor", "#161B22"),
                font=dict(
                    color=plotly_t.get("font", {}).get("color", "#FAFAFA"), size=12
                ),
                title="VaR Decomposition: Market vs Idiosyncratic",
                height=360,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(orientation="h", y=-0.05),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # ---- Per-ticker marginal VaR table + bar chart ----
            if per_ticker_mvar:
                sorted_tickers = sorted(
                    per_ticker_mvar.keys(),
                    key=lambda t: abs(per_ticker_mvar[t]),
                    reverse=True,
                )
                sorted_vals = [per_ticker_mvar[t] for t in sorted_tickers]

                st.markdown("#### Per-Ticker Marginal VaR Contribution")
                mvar_df = pd.DataFrame(
                    [
                        {
                            "Ticker": t,
                            "Marginal VaR": f"{per_ticker_mvar[t]:.4%}",
                            "Share of Total VaR": (
                                f"{abs(per_ticker_mvar[t]) / max(abs(total_var_95), 1e-10):.1%}"
                            ),
                        }
                        for t in sorted_tickers
                    ]
                )
                render_styled_table(mvar_df, key="mvar_table")

                fig_mvar = go.Figure(
                    go.Bar(
                        y=sorted_tickers,
                        x=sorted_vals,
                        orientation="h",
                        marker_color=[
                            neg_color if v < 0 else pos_color for v in sorted_vals
                        ],
                        text=[f"{v:.3%}" for v in sorted_vals],
                        textposition="outside",
                        hovertemplate="%{y}<br>Marginal VaR: %{x:.3%}<extra></extra>",
                    )
                )
                # Annotate largest contributors
                if sorted_tickers:
                    largest_ticker = sorted_tickers[0]
                    largest_val    = sorted_vals[0]
                    fig_mvar.add_annotation(
                        x=largest_val,
                        y=largest_ticker,
                        text="Largest Risk Contributor",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=warn_color,
                        font=dict(color=warn_color, size=11),
                        ax=60 if largest_val < 0 else -60,
                        ay=0,
                    )

                fig_mvar.update_layout(
                    **plotly_t,
                    title="Per-Ticker Marginal VaR (sorted by abs contribution)",
                    xaxis=dict(
                        title="Marginal VaR",
                        tickformat=".2%",
                        gridcolor=plotly_t.get("xaxis", {}).get(
                            "gridcolor", "rgba(255,255,255,0.08)"
                        ),
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        gridcolor=plotly_t.get("yaxis", {}).get(
                            "gridcolor", "rgba(255,255,255,0.08)"
                        ),
                    ),
                    height=max(320, len(sorted_tickers) * 36 + 80),
                    margin=dict(l=120, r=100, t=60, b=50),
                )
                fig_mvar.add_vline(x=0, line_color=neu_color, line_width=1, line_dash="dash")
                st.plotly_chart(fig_mvar, use_container_width=True)

    except Exception as exc:
        show_error(f"Factor VaR section error: {exc}")

    # =========================================================================
    # Section 4 — Correlation Breakdown Detector
    # =========================================================================
    st.markdown("---")
    st.markdown("### Section 4 — Correlation Breakdown Detector")
    st.markdown(
        "Compares the 252-day baseline correlation matrix with the most recent "
        "21-day window. Rising correlations signal diversification breakdown "
        "— assets tend to move together during crises."
    )

    try:
        if len(returns) < 25:
            show_error(
                "Insufficient return history for correlation analysis (need ≥ 25 days)."
            )
        else:
            with st.spinner("Computing correlation matrices…"):
                corr_result = tester.correlation_breakdown(
                    prices, lookback_normal=252, lookback_crisis=21
                )

            normal_corr  = corr_result["normal_corr"]
            current_corr = corr_result["current_corr"]
            avg_normal   = corr_result["avg_normal"]
            avg_current  = corr_result["avg_current"]
            spike_pairs  = corr_result["spike_pairs"]
            alert        = corr_result["alert"]

            # ---- Alert banner ----
            if alert:
                st.warning(
                    f"⚠️ Correlation Spike Detected: Average correlation rose from "
                    f"{avg_normal:.2f} (252d baseline) to {avg_current:.2f} (21d current) "
                    f"— a change of {avg_current - avg_normal:+.2f}. "
                    "Diversification benefits are eroding."
                )

            # Summary metrics
            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                st.metric(
                    "Avg Correlation (252d)",
                    f"{avg_normal:.3f}",
                    help="Mean off-diagonal pairwise correlation over the past 252 trading days.",
                )
            with cm2:
                delta_corr = avg_current - avg_normal
                st.metric(
                    "Avg Correlation (21d)",
                    f"{avg_current:.3f}",
                    delta=f"{delta_corr:+.3f}",
                    delta_color="inverse",
                    help="Mean off-diagonal pairwise correlation over the past 21 trading days.",
                )
            with cm3:
                diversif_status = "BREAKING DOWN" if alert else "HOLDING"
                status_color = neg_color if alert else pos_color
                st.markdown(
                    f"**Diversification:**  "
                    f"<span style='color:{status_color};font-weight:bold'>{diversif_status}</span>",
                    unsafe_allow_html=True,
                )

            # ---- Side-by-side heatmaps ----
            hm_col1, hm_col2 = st.columns(2)
            corr_colorscale = [
                [0.0, "#1E40AF"],   # blue  = -1
                [0.5, "#FFFFFF"],   # white =  0
                [1.0, "#B91C1C"],   # red   = +1
            ]

            def _build_heatmap(corr_df: pd.DataFrame, title: str) -> go.Figure:
                labels = corr_df.columns.tolist()
                fig = go.Figure(
                    go.Heatmap(
                        z=corr_df.values,
                        x=labels,
                        y=labels,
                        colorscale=corr_colorscale,
                        zmin=-1,
                        zmax=1,
                        colorbar=dict(
                            title="Corr",
                            tickvals=[-1, -0.5, 0, 0.5, 1],
                            thickness=12,
                        ),
                        hovertemplate="%{y} × %{x}<br>Corr: %{z:.3f}<extra></extra>",
                        text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
                        texttemplate="%{text}",
                        textfont=dict(size=9),
                    )
                )
                fig.update_layout(
                    paper_bgcolor=plotly_t.get("paper_bgcolor", "#161B22"),
                    plot_bgcolor=plotly_t.get("plot_bgcolor", "#0E1117"),
                    font=dict(
                        color=plotly_t.get("font", {}).get("color", "#FAFAFA"), size=11
                    ),
                    title=title,
                    xaxis=dict(tickangle=-45, side="bottom"),
                    height=420,
                    margin=dict(l=60, r=20, t=60, b=60),
                )
                return fig

            with hm_col1:
                st.plotly_chart(
                    _build_heatmap(normal_corr, "Normal Period (252d)"),
                    use_container_width=True,
                )
            with hm_col2:
                st.plotly_chart(
                    _build_heatmap(current_corr, "Current Period (21d)"),
                    use_container_width=True,
                )

            # ---- Spike pairs table ----
            if spike_pairs:
                st.markdown("#### Correlation Spike Pairs ( |Δcorr| > 0.30 )")
                spike_df = pd.DataFrame(
                    [
                        {
                            "Ticker 1": p[0],
                            "Ticker 2": p[1],
                            "Normal Corr": f"{p[2]:.3f}",
                            "Current Corr": f"{p[3]:.3f}",
                            "Change": f"{p[4]:+.3f}",
                            "Direction": "▲ Rising" if p[4] > 0 else "▼ Falling",
                        }
                        for p in spike_pairs
                    ]
                )
                render_styled_table(spike_df, key="spike_pairs_table")

                # Bar chart of correlation changes for spike pairs
                spike_labels = [f"{p[0]}–{p[1]}" for p in spike_pairs]
                spike_changes = [p[4] for p in spike_pairs]
                fig_spike = go.Figure(
                    go.Bar(
                        x=spike_labels,
                        y=spike_changes,
                        marker_color=[
                            neg_color if c > 0 else pos_color for c in spike_changes
                        ],
                        text=[f"{c:+.3f}" for c in spike_changes],
                        textposition="outside",
                        hovertemplate="%{x}<br>Δ Corr: %{y:+.3f}<extra></extra>",
                    )
                )
                fig_spike.update_layout(
                    **plotly_t,
                    title="Correlation Changes — Spike Pairs",
                    xaxis=dict(title="Pair", tickangle=-30),
                    yaxis=dict(title="Δ Correlation"),
                    height=340,
                )
                fig_spike.add_hline(y=0, line_color=neu_color, line_width=1, line_dash="dash")
                fig_spike.add_hline(
                    y=0.3, line_color=warn_color, line_width=1, line_dash="dot",
                    annotation_text="+0.30 threshold", annotation_position="right",
                )
                fig_spike.add_hline(
                    y=-0.3, line_color=warn_color, line_width=1, line_dash="dot",
                    annotation_text="-0.30 threshold", annotation_position="right",
                )
                st.plotly_chart(fig_spike, use_container_width=True)
            else:
                st.success(
                    "No significant correlation spikes detected "
                    "(no pair changed by more than 0.30)."
                )

            # Final diversification verdict
            st.markdown("---")
            verdict_text = (
                "**Portfolio diversification is "
                f"{'BREAKING DOWN ⚠️' if alert else 'HOLDING ✓'}.**  "
            )
            if alert:
                verdict_text += (
                    f"Average cross-asset correlation has risen by "
                    f"{avg_current - avg_normal:.2f} over the past 21 days, "
                    "reducing the portfolio's ability to absorb idiosyncratic shocks."
                )
            else:
                verdict_text += (
                    "Pairwise correlations are within normal historical ranges. "
                    "Diversification benefits remain intact."
                )
            st.markdown(verdict_text)

    except Exception as exc:
        show_error(f"Correlation Breakdown section error: {exc}")


PLOTLY_COLORS_DARK = ['#00b4d8', '#ffd700', '#00d084', '#ff4d6d', '#ff9a00', '#a0adc8', '#9055c8']
PLOTLY_COLORS_LIGHT = ['#0090b5', '#b8860b', '#059669', '#dc2626', '#d97706', '#6b7280', '#7c3aed']

def _get_plotly_theme():
    """Return plotly template, colors, grid color, and font color based on current theme."""
    theme = st.session_state.get('theme', 'light')
    if theme == 'dark':
        return 'plotly_dark', PLOTLY_COLORS_DARK, 'rgba(255,255,255,0.10)', '#ffffff'
    return 'plotly_white', PLOTLY_COLORS_LIGHT, 'rgba(0,0,0,0.12)', '#000000'

PLOTLY_COLORS = PLOTLY_COLORS_DARK  # kept for backward compat in non-chart code

def plot_price_history(prices, normalize=True):
    template, colors, grid_color, font_color = _get_plotly_theme()
    fig = go.Figure()

    if normalize:
        data = (prices / prices.iloc[0]) * 100
        title = "Normalized Performance (Base 100)"
        y_title = "Index"
    else:
        data = prices
        title = "Price History"
        y_title = "Price ($)"

    for i, col in enumerate(data.columns):
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col], name=col,
            mode='lines', line=dict(width=2, color=colors[i % len(colors)])
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=font_color)),
        yaxis_title=y_title,
        template=template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color)),
        yaxis=dict(gridcolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color)),
        legend=dict(font=dict(color=font_color)),
        hovermode="x unified",
        height=500,
        font=dict(family="Inter, system-ui, sans-serif", color=font_color)
    )
    return fig

def plot_portfolio_allocation(weights, tickers):
    template, colors, grid_color, font_color = _get_plotly_theme()
    fig = go.Figure(data=[go.Pie(
        labels=tickers,
        values=weights,
        hole=0.3,
        marker=dict(colors=colors[:len(tickers)])
    )])

    fig.update_layout(
        title=dict(text="Portfolio Allocation", font=dict(color=font_color)),
        template=template,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color=font_color)),
        font=dict(family="Inter, system-ui, sans-serif", color=font_color)
    )
    return fig

def plot_bubble_analysis(prices, bubble_results, ticker):
    template, colors_list, grid_color, font_color = _get_plotly_theme()
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{ticker} Price History', 'Bubble Score Components')
    )

    # Price chart
    fig.add_trace(
        go.Scatter(x=prices.index, y=prices.values, name='Price',
                   line=dict(color=colors_list[0], width=2)),
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

    theme = st.session_state.get('theme', 'light')
    if theme == 'dark':
        bar_colors = ['#ff4d6d' if v > 0.7 else '#ff9a00' if v > 0.3 else '#00d084' for v in values]
    else:
        bar_colors = ['#dc2626' if v > 0.7 else '#d97706' if v > 0.3 else '#059669' for v in values]

    fig.add_trace(
        go.Bar(x=components, y=values, marker_color=bar_colors),
        row=2, col=1
    )

    fig.update_layout(
        template=template,
        height=700,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, system-ui, sans-serif", color=font_color)
    )
    fig.update_xaxes(gridcolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color))
    fig.update_yaxes(gridcolor=grid_color, tickfont=dict(color=font_color), title_font=dict(color=font_color))

    return fig

# ========================================================================
# EXPORT FUNCTIONS — PDF, Slides, Excel
# ========================================================================

from fpdf import FPDF

# --------------- colour constants for exports ---------------
_NAVY = (10, 22, 40)
_TEAL = (0, 144, 181)
_WHITE = (255, 255, 255)
_LIGHT_GRAY = (245, 245, 245)
_DARK_GRAY = (80, 80, 80)
_GREEN = (0, 168, 84)
_RED = (220, 53, 69)
_GOLD = (184, 134, 11)

# --------------- helpers ---------------

def _mpl_normalized_chart(prices):
    """Render a normalized performance line chart to PNG bytes via matplotlib."""
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    normed = (prices / prices.iloc[0]) * 100
    palette = ['#0090b5', '#ffd700', '#00d084', '#ff4d6d', '#ff9a00', '#8b5cf6', '#ec4899']
    for i, col in enumerate(normed.columns):
        ax.plot(normed.index, normed[col], label=col, color=palette[i % len(palette)], linewidth=1.4)
    ax.set_title('Normalized Performance (Base 100)', fontsize=10, weight='bold', color='#0a1628')
    ax.set_ylabel('Index', fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def _mpl_pie_chart(labels, sizes):
    """Render a portfolio allocation pie chart to PNG bytes."""
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=150)
    palette = ['#0090b5', '#ffd700', '#00d084', '#ff4d6d', '#ff9a00', '#8b5cf6', '#ec4899']
    colors = [palette[i % len(palette)] for i in range(len(labels))]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', colors=colors,
        textprops={'fontsize': 7}, startangle=140
    )
    for t in autotexts:
        t.set_fontsize(6)
    ax.set_title('Optimal Portfolio Allocation', fontsize=9, weight='bold')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


# --------------- PDF Report (A4 portrait) ---------------

class _QuantLabPDF(FPDF):
    """Custom FPDF subclass with header/footer branding."""

    def header(self):
        if self.page_no() == 1:
            return  # cover page has no header
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(*_TEAL)
        self.cell(130, 6, 'QuantLab Analysis Report', align='L')
        self.set_text_color(*_DARK_GRAY)
        self.set_font('Helvetica', '', 8)
        self.cell(0, 6, f'Page {self.page_no()}', align='R', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(*_TEAL)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', '', 7)
        self.set_text_color(*_DARK_GRAY)
        self.cell(0, 8, f'Generated by QuantLab  |  Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(*_NAVY)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(*_TEAL)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def add_table(self, headers, rows, col_widths=None):
        """Draw a formatted table with alternating row colours."""
        n = len(headers)
        if col_widths is None:
            avail = 190
            col_widths = [avail / n] * n
        # header row
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(*_NAVY)
        self.set_text_color(*_WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, str(h), border=1, fill=True, align='C')
        self.ln()
        # data rows
        self.set_font('Helvetica', '', 7)
        for r_idx, row in enumerate(rows):
            if r_idx % 2 == 0:
                self.set_fill_color(*_LIGHT_GRAY)
            else:
                self.set_fill_color(*_WHITE)
            self.set_text_color(*_DARK_GRAY)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, fill=True, align='C')
            self.ln()


def generate_pdf_report(data_dict):
    """Generate a multi-page A4 PDF research report and return bytes."""
    tickers = data_dict.get('tickers', [])
    prices = data_dict.get('prices', pd.DataFrame())
    metrics_df = data_dict.get('metrics', pd.DataFrame())
    valuation_df = data_dict.get('valuation', pd.DataFrame())
    portfolio = data_dict.get('portfolio', {})
    portfolio_metrics = data_dict.get('portfolio_metrics', {})
    bubble_scores = data_dict.get('bubble_scores', {})
    technical = data_dict.get('technical', {})
    monte_carlo = data_dict.get('monte_carlo', {})

    pdf = _QuantLabPDF('P', 'mm', 'A4')
    pdf.set_auto_page_break(auto=True, margin=18)

    # ---- Page 1: Cover ----
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font('Helvetica', 'B', 32)
    pdf.set_text_color(*_NAVY)
    pdf.cell(0, 14, 'QuantLab', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(*_TEAL)
    pdf.cell(0, 10, 'Analysis Report', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(8)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(*_DARK_GRAY)
    pdf.cell(0, 8, f"Tickers: {', '.join(tickers)}", align='C', new_x='LMARGIN', new_y='NEXT')
    if not prices.empty:
        pdf.cell(0, 8, f"Date Range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align='C', new_x='LMARGIN', new_y='NEXT')

    # ---- Page 2: Executive Summary ----
    pdf.add_page()
    pdf.section_title('Executive Summary')
    if not metrics_df.empty:
        headers = ['Ticker', 'Annual Return', 'Volatility', 'Sharpe', 'Max Drawdown']
        rows = []
        for ticker in metrics_df.index:
            row = metrics_df.loc[ticker]
            rows.append([
                ticker,
                f"{row.get('Annual Return', 0):.2%}",
                f"{row.get('Volatility', 0):.2%}",
                f"{row.get('Sharpe', 0):.2f}",
                f"{row.get('Max Drawdown', 0):.2%}"
            ])
        pdf.add_table(headers, rows, col_widths=[25, 40, 40, 40, 45])

    # ---- Page 3: Performance Analysis ----
    pdf.add_page()
    pdf.section_title('Performance Analysis')
    if not prices.empty:
        try:
            chart_buf = _mpl_normalized_chart(prices)
            pdf.image(chart_buf, x=15, w=180)
            pdf.ln(4)
        except Exception:
            pdf.set_font('Helvetica', 'I', 9)
            pdf.cell(0, 8, '[Chart generation failed]', new_x='LMARGIN', new_y='NEXT')
    if not metrics_df.empty:
        pdf.ln(2)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(*_NAVY)
        pdf.cell(0, 8, 'Performance Metrics', new_x='LMARGIN', new_y='NEXT')
        headers = ['Ticker', 'Annual Return', 'Volatility', 'Sharpe', 'Max Drawdown']
        rows = []
        for ticker in metrics_df.index:
            row = metrics_df.loc[ticker]
            rows.append([
                ticker,
                f"{row.get('Annual Return', 0):.2%}",
                f"{row.get('Volatility', 0):.2%}",
                f"{row.get('Sharpe', 0):.2f}",
                f"{row.get('Max Drawdown', 0):.2%}"
            ])
        pdf.add_table(headers, rows, col_widths=[25, 40, 40, 40, 45])

    # ---- Page 4: Valuation Analysis ----
    if not valuation_df.empty:
        pdf.add_page()
        pdf.section_title('Valuation Analysis')
        val_cols = ['DCF Enterprise Value', 'WACC', 'CAPM Return', 'Fama-French Return', 'APT Return', 'Beta']
        present_cols = [c for c in val_cols if c in valuation_df.columns]
        headers = ['Ticker'] + present_cols
        rows = []
        for ticker in valuation_df.index:
            row_vals = [ticker]
            for c in present_cols:
                v = valuation_df.loc[ticker, c]
                if 'Value' in c:
                    row_vals.append(f"${v:,.0f}" if pd.notna(v) else 'N/A')
                elif 'Beta' in c:
                    row_vals.append(f"{v:.2f}" if pd.notna(v) else 'N/A')
                else:
                    row_vals.append(f"{v:.2%}" if pd.notna(v) else 'N/A')
            rows.append(row_vals)
        w = 190 / len(headers)
        pdf.add_table(headers, rows, col_widths=[w] * len(headers))

    # ---- Page 5: Portfolio Optimization ----
    if portfolio:
        pdf.add_page()
        pdf.section_title('Portfolio Optimization')
        # Use first strategy for pie chart
        first_strat = list(portfolio.keys())[0]
        first_weights = portfolio[first_strat]
        valid_mask = [w > 0.01 for w in first_weights]
        pie_labels = [t for t, m in zip(tickers, valid_mask) if m]
        pie_sizes = [w for w, m in zip(first_weights, valid_mask) if m]
        if pie_labels:
            try:
                pie_buf = _mpl_pie_chart(pie_labels, pie_sizes)
                pdf.image(pie_buf, x=40, w=120)
                pdf.ln(4)
            except Exception:
                pass
        # Portfolio metrics table
        if portfolio_metrics:
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(*_NAVY)
            pdf.cell(0, 8, 'Portfolio Metrics by Strategy', new_x='LMARGIN', new_y='NEXT')
            headers = ['Strategy', 'Exp. Return', 'Volatility', 'Sharpe', 'Max DD']
            rows = []
            for strat, m in portfolio_metrics.items():
                rows.append([
                    strat,
                    f"{m.get('Expected Return', 0):.2%}",
                    f"{m.get('Volatility', 0):.2%}",
                    f"{m.get('Sharpe Ratio', 0):.2f}",
                    f"{m.get('Max Drawdown', 0):.2%}"
                ])
            pdf.add_table(headers, rows, col_widths=[45, 35, 35, 35, 40])

    # ---- Page 6: Strategy Comparison ----
    if portfolio:
        pdf.add_page()
        pdf.section_title('Strategy Comparison')
        headers = ['Strategy'] + [t for t in tickers]
        rows = []
        for strat, weights in portfolio.items():
            rows.append([strat] + [f"{w:.1%}" for w in weights])
        w = 190 / len(headers)
        pdf.add_table(headers, rows, col_widths=[w] * len(headers))

    # ---- Page 7: Bubble Detection ----
    if bubble_scores:
        pdf.add_page()
        pdf.section_title('Bubble Detection')
        headers = ['Ticker', 'Bubble Score', 'Risk Level']
        rows = []
        for ticker, score in bubble_scores.items():
            level = 'HIGH RISK' if score > 0.7 else ('CAUTION' if score > 0.4 else 'NORMAL')
            rows.append([ticker, f"{score:.2%}", level])
        pdf.add_table(headers, rows, col_widths=[50, 70, 70])

    # ---- Page 8: Technical Analysis ----
    if technical:
        pdf.add_page()
        pdf.section_title('Technical Analysis')
        for ticker in tickers:
            if ticker not in technical:
                continue
            tech_df = technical[ticker]
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(*_TEAL)
            pdf.cell(0, 8, ticker, new_x='LMARGIN', new_y='NEXT')
            # Get the latest row
            if tech_df.empty:
                continue
            latest = tech_df.iloc[-1]
            items = []
            for col in ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'EMA_20']:
                if col in tech_df.columns:
                    v = latest[col]
                    items.append([col, f"{v:.2f}" if pd.notna(v) else 'N/A'])
            if items:
                pdf.add_table(['Indicator', 'Value'], items, col_widths=[60, 60])
                pdf.ln(3)
            # Check if we need a new page
            if pdf.get_y() > 240:
                pdf.add_page()
                pdf.section_title('Technical Analysis (cont.)')

    # ---- Page 9: Monte Carlo ----
    if monte_carlo:
        pdf.add_page()
        pdf.section_title('Monte Carlo Simulation')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*_DARK_GRAY)
        if isinstance(monte_carlo, dict):
            for k, v in monte_carlo.items():
                pdf.cell(0, 6, f"{k}: {v}", new_x='LMARGIN', new_y='NEXT')
        else:
            pdf.cell(0, 6, 'Simulation data available in Excel export.', new_x='LMARGIN', new_y='NEXT')

    # ---- Page 10: Macroeconomic Dashboard ----
    macro_data = data_dict.get('macro_data', {})
    if macro_data:
        pdf.add_page()
        pdf.section_title('Macroeconomic Dashboard')
        headers = ['Indicator', 'Latest Value']
        rows = []
        label_map = {
            'IRX': '3M T-Bill Rate',
            'TNX': '10Y Treasury',
            'FVX': '5Y Treasury',
            'TYX': '30Y Treasury',
            'VIX': 'VIX Volatility',
            'GSPC': 'S&P 500',
        }
        for sid, label in label_map.items():
            df = macro_data.get(sid, pd.DataFrame())
            if not df.empty:
                df = df.dropna()
                if len(df) > 0:
                    val = df.iloc[-1].values[0]
                    fmt = f"{val:,.2f}" if sid == 'GSPC' else f"{val:.2f}"
                    rows.append([label, fmt])
        if rows:
            pdf.add_table(headers, rows, col_widths=[95, 95])

    # ---- Page 11: ML Predictions ----
    ml_results = data_dict.get('ml_results')
    ml_ticker = data_dict.get('ml_ticker', '')
    if ml_results:
        pdf.add_page()
        pdf.section_title(f'ML Predictions ({ml_ticker})')
        headers = ['Model', 'R-squared', 'RMSE', 'MAE']
        rows = []
        for name, info in ml_results['models_info'].items():
            rows.append([
                name,
                f"{info['r2']:.4f}",
                f"{info['rmse']:.4f}",
                f"{info['mae']:.4f}",
            ])
        pdf.add_table(headers, rows, col_widths=[55, 45, 45, 45])
        pdf.ln(4)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*_DARK_GRAY)
        pdf.cell(0, 6, 'Models: Linear Regression, Random Forest (n=100, depth=8), Gradient Boosting (n=100, lr=0.1)', new_x='LMARGIN', new_y='NEXT')
        pdf.cell(0, 6, 'Target: Forward 21-day return. Features: returns, volatility, SMA ratios, volume ratio, RSI.', new_x='LMARGIN', new_y='NEXT')

    # ---- Page 12: Options Pricing ----
    options_data = data_dict.get('options', {})
    if options_data and options_data.get('calls') is not None and not options_data['calls'].empty:
        pdf.add_page()
        pdf.section_title(f"Options Pricing ({options_data.get('ticker', '')} - {options_data.get('expiration', '')})")
        calls = options_data['calls']
        headers = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV']
        rows = []
        for _, row in calls.head(15).iterrows():
            rows.append([
                f"{row.get('strike', 0):.0f}",
                f"{row.get('lastPrice', 0):.2f}",
                f"{row.get('bid', 0):.2f}",
                f"{row.get('ask', 0):.2f}",
                f"{row.get('volume', 0):.0f}" if pd.notna(row.get('volume')) else 'N/A',
                f"{row.get('openInterest', 0):.0f}" if pd.notna(row.get('openInterest')) else 'N/A',
                f"{row.get('impliedVolatility', 0):.2%}" if pd.notna(row.get('impliedVolatility')) else 'N/A',
            ])
        w = 190 / len(headers)
        pdf.add_table(headers, rows, col_widths=[w] * len(headers))

    # ---- Page 13: Risk & Geopolitics ----
    risk_export = data_dict.get('risk', {})
    if risk_export:
        pdf.add_page()
        pdf.section_title('Risk & Geopolitics Dashboard')
        risk_d = risk_export.get('risk_data', {})
        score = risk_export.get('composite_score', 0)
        headers = ['Indicator', 'Latest Value']
        rows = []
        for name in ['VIX', 'DXY', 'Gold', 'Oil', 'TNX']:
            s = risk_d.get(name, pd.Series(dtype=float))
            if len(s) > 0:
                val = float(s.iloc[-1])
                rows.append([name, f"{val:.2f}"])
        rows.append(['Composite Risk Score', f"{score:.0f}/100"])
        pdf.add_table(headers, rows, col_widths=[95, 95])

    # ---- Page 14: ML Clustering ----
    clustering_data = data_dict.get('clustering', {})
    if clustering_data and 'features' in clustering_data:
        pdf.add_page()
        pdf.section_title('ML Clustering')
        feat = clustering_data['features']
        headers = ['Asset', 'Cluster', 'Return', 'Vol', 'Sharpe']
        rows = []
        for idx in feat.index:
            rows.append([
                str(idx),
                str(int(feat.loc[idx, 'Cluster'])),
                f"{feat.loc[idx, 'Ann Return']:.2%}",
                f"{feat.loc[idx, 'Volatility']:.2%}",
                f"{feat.loc[idx, 'Sharpe']:.2f}",
            ])
        pdf.add_table(headers, rows, col_widths=[35, 25, 45, 45, 40])

    # ---- Page 15: Sentiment Analysis ----
    sentiment_data = data_dict.get('sentiment', {})
    if sentiment_data and 'articles' in sentiment_data:
        pdf.add_page()
        pdf.section_title(f"Sentiment Analysis ({sentiment_data.get('ticker', '')})")
        articles = sentiment_data['articles']
        headers = ['Headline', 'Score']
        rows = []
        for _, row in articles.head(15).iterrows():
            # Truncate headline for PDF (ASCII only)
            headline = str(row.get('Headline', '')).encode('ascii', 'replace').decode('ascii')[:60]
            rows.append([headline, f"{row.get('Score', 0):+.2f}"])
        pdf.add_table(headers, rows, col_widths=[145, 45])

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# --------------- Presentation Slides (landscape PDF) ---------------

class _QuantLabSlides(FPDF):
    """Landscape 16:9 slide deck."""

    _PW = 297   # page width mm
    _PH = 167   # page height mm
    _PM = 10    # margin mm
    _CW = _PW - 2 * _PM  # 277 content width

    def footer(self):
        self.set_y(-10)
        self.set_font('Helvetica', '', 7)
        self.set_text_color(*_DARK_GRAY)
        self.cell(0, 6, f'Slide {self.page_no()}', align='R')

    def slide_title_bar(self, title):
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, self._PW, 25, 'F')
        self.set_xy(self._PM, 5)
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(*_WHITE)
        self.cell(self._CW, 15, title)
        self.set_xy(self._PM, 30)

    def add_table(self, headers, rows, col_widths=None, x_offset=10):
        n = len(headers)
        if col_widths is None:
            avail = self._CW
            col_widths = [avail / n] * n
        else:
            # Clamp col_widths so total does not exceed content width
            total = sum(col_widths)
            if total > self._CW:
                scale = self._CW / total
                col_widths = [w * scale for w in col_widths]
        self.set_x(x_offset)
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(*_NAVY)
        self.set_text_color(*_WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, str(h), border=1, fill=True, align='C')
        self.ln()
        self.set_font('Helvetica', '', 7)
        for r_idx, row in enumerate(rows):
            self.set_x(x_offset)
            if r_idx % 2 == 0:
                self.set_fill_color(*_LIGHT_GRAY)
            else:
                self.set_fill_color(*_WHITE)
            self.set_text_color(*_DARK_GRAY)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, fill=True, align='C')
            self.ln()


def generate_slides(data_dict):
    """Generate landscape PDF slides and return bytes."""
    tickers = data_dict.get('tickers', [])
    prices = data_dict.get('prices', pd.DataFrame())
    metrics_df = data_dict.get('metrics', pd.DataFrame())
    valuation_df = data_dict.get('valuation', pd.DataFrame())
    portfolio = data_dict.get('portfolio', {})
    portfolio_metrics = data_dict.get('portfolio_metrics', {})
    bubble_scores = data_dict.get('bubble_scores', {})
    technical = data_dict.get('technical', {})

    _SW = 297  # slide width mm
    _SH = 167  # slide height mm
    _SM = 10   # slide margin mm
    _SCW = _SW - 2 * _SM  # content width = 277
    pdf = _QuantLabSlides('P', 'mm', (_SW, _SH))
    pdf.set_auto_page_break(auto=False)

    # ---- Slide 1: Title ----
    pdf.add_page()
    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, _SW, _SH, 'F')
    pdf.set_xy(_SM, 40)
    pdf.set_font('Helvetica', 'B', 36)
    pdf.set_text_color(*_WHITE)
    pdf.cell(_SCW, 20, 'QuantLab', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 18)
    pdf.set_text_color(*_TEAL)
    pdf.cell(_SCW, 12, 'Portfolio Analytics & Research', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(180, 190, 210)
    pdf.cell(_SCW, 8, f"Tickers: {', '.join(tickers)}", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(_SCW, 8, datetime.now().strftime('%B %d, %Y'), align='C', new_x='LMARGIN', new_y='NEXT')

    # ---- Slide 2: Portfolio Overview ----
    pdf.add_page()
    pdf.slide_title_bar('Portfolio Overview')
    if not prices.empty:
        headers = ['Ticker', 'Last Price', 'Change %']
        rows = []
        for t in tickers:
            if t in prices.columns:
                last = prices[t].iloc[-1]
                prev = prices[t].iloc[-2] if len(prices[t]) > 1 else last
                chg = ((last / prev) - 1) * 100
                rows.append([t, f"${last:.2f}", f"{chg:+.2f}%"])
        pdf.add_table(headers, rows)

    # ---- Slide 3: Performance Chart ----
    pdf.add_page()
    pdf.slide_title_bar('Performance')
    if not prices.empty:
        try:
            chart_buf = _mpl_normalized_chart(prices)
            pdf.image(chart_buf, x=_SM, y=32, w=_SCW)
        except Exception:
            pass

    # ---- Slide 4: Performance Metrics ----
    pdf.add_page()
    pdf.slide_title_bar('Performance Metrics')
    if not metrics_df.empty:
        headers = ['Ticker', 'Annual Return', 'Volatility', 'Sharpe', 'Max Drawdown']
        rows = []
        for t in metrics_df.index:
            row = metrics_df.loc[t]
            rows.append([
                t, f"{row.get('Annual Return',0):.2%}",
                f"{row.get('Volatility',0):.2%}",
                f"{row.get('Sharpe',0):.2f}",
                f"{row.get('Max Drawdown',0):.2%}"
            ])
        pdf.add_table(headers, rows)

    # ---- Slide 5: Valuation Summary ----
    if not valuation_df.empty:
        pdf.add_page()
        pdf.slide_title_bar('Valuation Summary')
        val_cols = [c for c in ['DCF Enterprise Value', 'WACC', 'CAPM Return', 'Beta'] if c in valuation_df.columns]
        headers = ['Ticker'] + val_cols
        rows = []
        for t in valuation_df.index:
            rv = [t]
            for c in val_cols:
                v = valuation_df.loc[t, c]
                if 'Value' in c:
                    rv.append(f"${v:,.0f}" if pd.notna(v) else 'N/A')
                elif 'Beta' in c:
                    rv.append(f"{v:.2f}" if pd.notna(v) else 'N/A')
                else:
                    rv.append(f"{v:.2%}" if pd.notna(v) else 'N/A')
            rows.append(rv)
        w = 277 / len(headers)
        pdf.add_table(headers, rows, col_widths=[w] * len(headers))

    # ---- Slide 6: Portfolio Allocation ----
    if portfolio:
        pdf.add_page()
        pdf.slide_title_bar('Portfolio Allocation')
        first_strat = list(portfolio.keys())[0]
        first_weights = portfolio[first_strat]
        valid_mask = [w > 0.01 for w in first_weights]
        pie_labels = [t for t, m in zip(tickers, valid_mask) if m]
        pie_sizes = [w for w, m in zip(first_weights, valid_mask) if m]
        if pie_labels:
            try:
                pie_buf = _mpl_pie_chart(pie_labels, pie_sizes)
                pdf.image(pie_buf, x=15, y=32, w=130)
            except Exception:
                pass
        # Metrics on the right side
        if portfolio_metrics:
            pdf.set_xy(155, 35)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(*_NAVY)
            for strat, m in portfolio_metrics.items():
                pdf.set_x(155)
                pdf.set_font('Helvetica', 'B', 8)
                pdf.set_text_color(*_TEAL)
                pdf.cell(130, 7, strat, new_x='LMARGIN', new_y='NEXT')
                pdf.set_font('Helvetica', '', 7)
                pdf.set_text_color(*_DARK_GRAY)
                for k in ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']:
                    v = m.get(k, 0)
                    fmt = f"{v:.2%}" if 'Return' in k or 'Volatility' in k or 'Drawdown' in k else f"{v:.2f}"
                    pdf.set_x(155)
                    pdf.cell(130, 5, f"  {k}: {fmt}", new_x='LMARGIN', new_y='NEXT')
                pdf.ln(2)

    # ---- Slide 7: Strategy Comparison ----
    if portfolio:
        pdf.add_page()
        pdf.slide_title_bar('Strategy Comparison')
        headers = ['Strategy'] + list(tickers)
        rows = []
        for strat, weights in portfolio.items():
            rows.append([strat] + [f"{w:.1%}" for w in weights])
        w = 277 / len(headers)
        pdf.add_table(headers, rows, col_widths=[w] * len(headers))

    # ---- Slide 8: Bubble Detection ----
    if bubble_scores:
        pdf.add_page()
        pdf.slide_title_bar('Bubble Detection')
        headers = ['Ticker', 'Bubble Score', 'Risk Level']
        rows = []
        for t, s in bubble_scores.items():
            level = 'HIGH RISK' if s > 0.7 else ('CAUTION' if s > 0.4 else 'NORMAL')
            rows.append([t, f"{s:.2%}", level])
        pdf.add_table(headers, rows, col_widths=[70, 100, 107])

    # ---- Slide 9: Technical Signals ----
    if technical:
        pdf.add_page()
        pdf.slide_title_bar('Technical Signals')
        headers = ['Ticker', 'RSI', 'MACD', 'SMA 20', 'SMA 50']
        rows = []
        for t in tickers:
            if t not in technical or technical[t].empty:
                continue
            latest = technical[t].iloc[-1]
            rows.append([
                t,
                f"{latest.get('RSI', 0):.1f}" if pd.notna(latest.get('RSI')) else 'N/A',
                f"{latest.get('MACD', 0):.2f}" if pd.notna(latest.get('MACD')) else 'N/A',
                f"{latest.get('SMA_20', 0):.2f}" if pd.notna(latest.get('SMA_20')) else 'N/A',
                f"{latest.get('SMA_50', 0):.2f}" if pd.notna(latest.get('SMA_50')) else 'N/A',
            ])
        pdf.add_table(headers, rows)

    # ---- Slide 10: Macro Indicators ----
    macro_data = data_dict.get('macro_data', {})
    if macro_data:
        pdf.add_page()
        pdf.slide_title_bar('Macro Indicators')
        headers = ['Indicator', 'Latest Value']
        rows = []
        label_map = {
            'IRX': '3M T-Bill Rate',
            'TNX': '10Y Treasury',
            'FVX': '5Y Treasury',
            'TYX': '30Y Treasury',
            'VIX': 'VIX Volatility',
            'GSPC': 'S&P 500',
        }
        for sid, label in label_map.items():
            df = macro_data.get(sid, pd.DataFrame())
            if not df.empty:
                df = df.dropna()
                if len(df) > 0:
                    val = df.iloc[-1].values[0]
                    fmt = f"{val:,.2f}" if sid == 'GSPC' else f"{val:.2f}"
                    rows.append([label, fmt])
        if rows:
            pdf.add_table(headers, rows, col_widths=[140, 137])

    # ---- Slide 11: ML Model Performance ----
    ml_results = data_dict.get('ml_results')
    ml_ticker = data_dict.get('ml_ticker', '')
    if ml_results:
        pdf.add_page()
        pdf.slide_title_bar(f'ML Predictions ({ml_ticker})')
        headers = ['Model', 'R-squared', 'RMSE', 'MAE']
        rows = []
        for name, info in ml_results['models_info'].items():
            rows.append([name, f"{info['r2']:.4f}", f"{info['rmse']:.4f}", f"{info['mae']:.4f}"])
        pdf.add_table(headers, rows, col_widths=[80, 65, 65, 67])

    # ---- Slide 12: Options Pricing ----
    options_data = data_dict.get('options', {})
    if options_data and options_data.get('calls') is not None and not options_data.get('calls', pd.DataFrame()).empty:
        pdf.add_page()
        pdf.slide_title_bar(f"Options Pricing ({options_data.get('ticker', '')})")
        calls = options_data['calls']
        headers = ['Strike', 'Last', 'Bid', 'Ask', 'Vol', 'OI', 'IV']
        rows = []
        for _, row in calls.head(10).iterrows():
            rows.append([
                f"{row.get('strike', 0):.0f}",
                f"{row.get('lastPrice', 0):.2f}",
                f"{row.get('bid', 0):.2f}",
                f"{row.get('ask', 0):.2f}",
                f"{row.get('volume', 0):.0f}" if pd.notna(row.get('volume')) else '-',
                f"{row.get('openInterest', 0):.0f}" if pd.notna(row.get('openInterest')) else '-',
                f"{row.get('impliedVolatility', 0):.1%}" if pd.notna(row.get('impliedVolatility')) else '-',
            ])
        pdf.add_table(headers, rows)

    # ---- Slide 13: Risk Dashboard ----
    risk_export = data_dict.get('risk', {})
    if risk_export:
        pdf.add_page()
        pdf.slide_title_bar('Risk & Geopolitics Dashboard')
        risk_d = risk_export.get('risk_data', {})
        score = risk_export.get('composite_score', 0)
        headers = ['Indicator', 'Value']
        rows = []
        for name in ['VIX', 'DXY', 'Gold', 'Oil', 'TNX']:
            s = risk_d.get(name, pd.Series(dtype=float))
            if len(s) > 0:
                rows.append([name, f"{float(s.iloc[-1]):.2f}"])
        rows.append(['Risk Score', f"{score:.0f}/100"])
        pdf.add_table(headers, rows, col_widths=[140, 137])

    # ---- Slide 14: ML Clustering ----
    clustering_data = data_dict.get('clustering', {})
    if clustering_data and 'features' in clustering_data:
        pdf.add_page()
        pdf.slide_title_bar('ML Clustering')
        feat = clustering_data['features']
        headers = ['Asset', 'Cluster', 'Return', 'Vol', 'Sharpe']
        rows = []
        for idx in feat.index:
            rows.append([
                str(idx), str(int(feat.loc[idx, 'Cluster'])),
                f"{feat.loc[idx, 'Ann Return']:.2%}",
                f"{feat.loc[idx, 'Volatility']:.2%}",
                f"{feat.loc[idx, 'Sharpe']:.2f}",
            ])
        pdf.add_table(headers, rows)

    # ---- Slide 15: Sentiment Analysis ----
    sentiment_data = data_dict.get('sentiment', {})
    if sentiment_data and 'articles' in sentiment_data:
        pdf.add_page()
        pdf.slide_title_bar(f"Sentiment ({sentiment_data.get('ticker', '')})")
        articles = sentiment_data['articles']
        headers = ['Headline', 'Score']
        rows = []
        for _, row in articles.head(10).iterrows():
            headline = str(row.get('Headline', '')).encode('ascii', 'replace').decode('ascii')[:55]
            rows.append([headline, f"{row.get('Score', 0):+.2f}"])
        pdf.add_table(headers, rows, col_widths=[220, 57])

    # ---- Slide 16: Key Takeaways ----
    pdf.add_page()
    pdf.slide_title_bar('Key Takeaways')
    pdf.set_xy(15, 38)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(*_DARK_GRAY)
    takeaways = []
    if not metrics_df.empty:
        best = metrics_df['Sharpe'].idxmax()
        takeaways.append(f"Highest Sharpe Ratio: {best} ({metrics_df.loc[best, 'Sharpe']:.2f})")
        best_ret = metrics_df['Annual Return'].idxmax()
        takeaways.append(f"Best Annual Return: {best_ret} ({metrics_df.loc[best_ret, 'Annual Return']:.2%})")
        lowest_vol = metrics_df['Volatility'].idxmin()
        takeaways.append(f"Lowest Volatility: {lowest_vol} ({metrics_df.loc[lowest_vol, 'Volatility']:.2%})")
    if bubble_scores:
        high_risk = [t for t, s in bubble_scores.items() if s > 0.7]
        if high_risk:
            takeaways.append(f"High Bubble Risk: {', '.join(high_risk)}")
        else:
            takeaways.append("No tickers currently in high bubble-risk territory")
    if portfolio_metrics:
        best_strat = max(portfolio_metrics.items(), key=lambda x: x[1].get('Sharpe Ratio', 0))
        takeaways.append(f"Best Strategy: {best_strat[0]} (Sharpe {best_strat[1].get('Sharpe Ratio', 0):.2f})")
    if ml_results:
        best_ml = max(ml_results['models_info'].items(), key=lambda x: x[1]['r2'])
        takeaways.append(f"Best ML Model: {best_ml[0]} (R2={best_ml[1]['r2']:.4f})")
    for i, t in enumerate(takeaways):
        pdf.set_x(20)
        pdf.multi_cell(_SCW - 10, 9, f"  -  {t}", new_x='LMARGIN', new_y='NEXT')

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# --------------- Enhanced Excel Export ---------------

def generate_comprehensive_excel(data_dict):
    """Generate formatted Excel workbook with xlsxwriter styling."""
    tickers = data_dict.get('tickers', [])
    prices = data_dict.get('prices', pd.DataFrame())
    metrics_df = data_dict.get('metrics', pd.DataFrame())
    valuation_df = data_dict.get('valuation', pd.DataFrame())
    portfolio = data_dict.get('portfolio', {})
    portfolio_metrics = data_dict.get('portfolio_metrics', {})
    bubble_scores = data_dict.get('bubble_scores', {})
    technical = data_dict.get('technical', {})

    output = io.BytesIO()
    wb = xlsxwriter.Workbook(output, {'in_memory': True, 'nan_inf_to_errors': True})

    # ---- Shared formats ----
    fmt_header_navy = wb.add_format({
        'bold': True, 'font_color': 'white', 'bg_color': '#0a1628',
        'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 10
    })
    fmt_header_teal = wb.add_format({
        'bold': True, 'font_color': 'white', 'bg_color': '#0090b5',
        'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 10
    })
    fmt_header_gold = wb.add_format({
        'bold': True, 'font_color': 'white', 'bg_color': '#b8860b',
        'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 10
    })
    fmt_pct = wb.add_format({'num_format': '0.00%', 'align': 'center', 'border': 1})
    fmt_pct_green = wb.add_format({'num_format': '0.00%', 'align': 'center', 'border': 1, 'font_color': '#00a854'})
    fmt_pct_red = wb.add_format({'num_format': '0.00%', 'align': 'center', 'border': 1, 'font_color': '#dc3545'})
    fmt_currency = wb.add_format({'num_format': '$#,##0', 'align': 'center', 'border': 1})
    fmt_number = wb.add_format({'num_format': '0.00', 'align': 'center', 'border': 1})
    fmt_text = wb.add_format({'align': 'center', 'border': 1})
    fmt_text_bold = wb.add_format({'bold': True, 'align': 'center', 'border': 1})
    fmt_row_alt = wb.add_format({'bg_color': '#f5f5f5', 'border': 1, 'align': 'center'})
    fmt_row_alt_pct = wb.add_format({'bg_color': '#f5f5f5', 'border': 1, 'align': 'center', 'num_format': '0.00%'})
    fmt_row_alt_cur = wb.add_format({'bg_color': '#f5f5f5', 'border': 1, 'align': 'center', 'num_format': '$#,##0'})
    fmt_row_alt_num = wb.add_format({'bg_color': '#f5f5f5', 'border': 1, 'align': 'center', 'num_format': '0.00'})
    fmt_date = wb.add_format({'num_format': 'yyyy-mm-dd', 'align': 'center', 'border': 1})
    fmt_date_alt = wb.add_format({'num_format': 'yyyy-mm-dd', 'align': 'center', 'border': 1, 'bg_color': '#f5f5f5'})
    fmt_title = wb.add_format({
        'bold': True, 'font_size': 14, 'font_color': '#0a1628', 'bottom': 2
    })

    def _write_df_sheet(ws, df, header_fmt, start_row=1, col_formats=None):
        """Write a DataFrame to a worksheet with header formatting and alternating rows."""
        # Write headers
        for c, col_name in enumerate(df.columns):
            ws.write(start_row, c + 1, col_name, header_fmt)
        if df.index.name or True:
            ws.write(start_row, 0, df.index.name or 'Ticker', header_fmt)
        # Write data
        for r, idx in enumerate(df.index):
            row_num = start_row + 1 + r
            is_alt = r % 2 == 0
            ws.write(row_num, 0, str(idx), fmt_row_alt if is_alt else fmt_text)
            for c, col_name in enumerate(df.columns):
                val = df.iloc[r, c]
                if col_formats and col_name in col_formats:
                    fmt_normal, fmt_alt = col_formats[col_name]
                    ws.write(row_num, c + 1, val, fmt_alt if is_alt else fmt_normal)
                else:
                    if isinstance(val, (int, float)) and pd.notna(val):
                        ws.write_number(row_num, c + 1, val, fmt_row_alt_num if is_alt else fmt_number)
                    else:
                        ws.write(row_num, c + 1, str(val) if pd.notna(val) else 'N/A', fmt_row_alt if is_alt else fmt_text)
        # Freeze header row
        ws.freeze_panes(start_row + 1, 0)
        # Auto-fit column widths (approximate)
        for c in range(len(df.columns) + 1):
            max_len = 12
            if c == 0:
                max_len = max(max_len, max((len(str(i)) for i in df.index), default=8) + 2)
            else:
                col_name = df.columns[c - 1]
                max_len = max(max_len, len(col_name) + 2)
            ws.set_column(c, c, min(max_len, 25))

    # ==== Sheet 1: Summary Dashboard ====
    ws = wb.add_worksheet('Summary Dashboard')
    ws.set_tab_color('#0a1628')
    ws.write(0, 0, 'QuantLab Analysis Summary', fmt_title)
    ws.write(0, 3, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", wb.add_format({'italic': True, 'font_color': '#666666'}))
    if not metrics_df.empty:
        _write_df_sheet(ws, metrics_df, fmt_header_navy, start_row=2, col_formats={
            'Annual Return': (fmt_pct, fmt_row_alt_pct),
            'Volatility': (fmt_pct, fmt_row_alt_pct),
            'Sharpe': (fmt_number, fmt_row_alt_num),
            'Max Drawdown': (fmt_pct, fmt_row_alt_pct),
        })
        # Conditional formatting for Annual Return column
        ret_col = list(metrics_df.columns).index('Annual Return') + 1 if 'Annual Return' in metrics_df.columns else None
        if ret_col is not None:
            data_start = 3
            data_end = 2 + len(metrics_df)
            ws.conditional_format(data_start, ret_col, data_end, ret_col, {
                'type': 'cell', 'criteria': '>=', 'value': 0,
                'format': wb.add_format({'font_color': '#00a854', 'num_format': '0.00%', 'align': 'center', 'border': 1})
            })
            ws.conditional_format(data_start, ret_col, data_end, ret_col, {
                'type': 'cell', 'criteria': '<', 'value': 0,
                'format': wb.add_format({'font_color': '#dc3545', 'num_format': '0.00%', 'align': 'center', 'border': 1})
            })

    # ==== Sheet 2: Price History ====
    if not prices.empty:
        ws = wb.add_worksheet('Price History')
        ws.set_tab_color('#0090b5')
        ws.write(0, 0, 'Historical Prices', fmt_title)
        # Write date + price columns
        ws.write(1, 0, 'Date', fmt_header_teal)
        for c, col in enumerate(prices.columns):
            ws.write(1, c + 1, col, fmt_header_teal)
        for r in range(len(prices)):
            is_alt = r % 2 == 0
            ws.write_datetime(r + 2, 0, prices.index[r].to_pydatetime(), fmt_date_alt if is_alt else fmt_date)
            for c, col in enumerate(prices.columns):
                val = prices.iloc[r, c]
                if pd.notna(val):
                    ws.write_number(r + 2, c + 1, val,
                        fmt_row_alt_cur if is_alt else fmt_currency)
                else:
                    ws.write(r + 2, c + 1, '', fmt_row_alt if is_alt else fmt_text)
        ws.freeze_panes(2, 0)
        ws.set_column(0, 0, 14)
        for c in range(len(prices.columns)):
            ws.set_column(c + 1, c + 1, 14)

    # ==== Sheet 3: Performance Metrics ====
    if not metrics_df.empty:
        ws = wb.add_worksheet('Performance Metrics')
        ws.set_tab_color('#00d084')
        ws.write(0, 0, 'Performance Metrics', fmt_title)
        _write_df_sheet(ws, metrics_df, fmt_header_teal, start_row=2, col_formats={
            'Annual Return': (fmt_pct, fmt_row_alt_pct),
            'Volatility': (fmt_pct, fmt_row_alt_pct),
            'Sharpe': (fmt_number, fmt_row_alt_num),
            'Max Drawdown': (fmt_pct, fmt_row_alt_pct),
        })

    # ==== Sheet 4: Valuation Analysis ====
    if not valuation_df.empty:
        ws = wb.add_worksheet('Valuation Analysis')
        ws.set_tab_color('#b8860b')
        ws.write(0, 0, 'Valuation Analysis', fmt_title)
        col_fmts = {}
        for col in valuation_df.columns:
            if 'Value' in col:
                col_fmts[col] = (fmt_currency, fmt_row_alt_cur)
            elif any(x in col for x in ['Return', 'WACC', 'Score', 'Impact']):
                col_fmts[col] = (fmt_pct, fmt_row_alt_pct)
            elif 'Beta' in col:
                col_fmts[col] = (fmt_number, fmt_row_alt_num)
        _write_df_sheet(ws, valuation_df, fmt_header_gold, start_row=2, col_formats=col_fmts)

    # ==== Sheet 5: Portfolio Optimization ====
    if portfolio:
        ws = wb.add_worksheet('Portfolio Optimization')
        ws.set_tab_color('#0090b5')
        ws.write(0, 0, 'Portfolio Weights by Strategy', fmt_title)
        # Build a strategy x ticker DataFrame
        port_df = pd.DataFrame({strat: weights for strat, weights in portfolio.items()}, index=tickers).T
        port_df.index.name = 'Strategy'
        _write_df_sheet(ws, port_df, fmt_header_teal, start_row=2, col_formats={
            t: (fmt_pct, fmt_row_alt_pct) for t in tickers
        })

    # ==== Sheet 6: Strategy Comparison ====
    if portfolio_metrics:
        ws = wb.add_worksheet('Strategy Comparison')
        ws.set_tab_color('#ff9a00')
        ws.write(0, 0, 'Strategy Comparison', fmt_title)
        strat_df = pd.DataFrame(portfolio_metrics).T
        strat_df.index.name = 'Strategy'
        col_fmts = {}
        for col in strat_df.columns:
            if 'Return' in col or 'Volatility' in col or 'Drawdown' in col:
                col_fmts[col] = (fmt_pct, fmt_row_alt_pct)
            else:
                col_fmts[col] = (fmt_number, fmt_row_alt_num)
        _write_df_sheet(ws, strat_df, fmt_header_navy, start_row=2, col_formats=col_fmts)
        # Highlight best Sharpe
        if 'Sharpe Ratio' in strat_df.columns:
            sr_col = list(strat_df.columns).index('Sharpe Ratio') + 1
            data_start = 3
            data_end = 2 + len(strat_df)
            ws.conditional_format(data_start, sr_col, data_end, sr_col, {
                'type': 'top', 'value': 1,
                'format': wb.add_format({'bold': True, 'font_color': '#00a854', 'num_format': '0.00', 'align': 'center', 'border': 1, 'bg_color': '#e6ffe6'})
            })

    # ==== Sheet 7: Bubble Detection ====
    if bubble_scores:
        ws = wb.add_worksheet('Bubble Detection')
        ws.set_tab_color('#ff4d6d')
        ws.write(0, 0, 'Bubble Detection Scores', fmt_title)
        ws.write(2, 0, 'Ticker', fmt_header_navy)
        ws.write(2, 1, 'Bubble Score', fmt_header_navy)
        ws.write(2, 2, 'Risk Level', fmt_header_navy)
        for r, (ticker, score) in enumerate(bubble_scores.items()):
            is_alt = r % 2 == 0
            ws.write(r + 3, 0, ticker, fmt_row_alt if is_alt else fmt_text)
            ws.write_number(r + 3, 1, score, fmt_row_alt_pct if is_alt else fmt_pct)
            level = 'HIGH RISK' if score > 0.7 else ('CAUTION' if score > 0.4 else 'NORMAL')
            if score > 0.7:
                level_fmt = wb.add_format({'font_color': 'white', 'bg_color': '#dc3545', 'bold': True, 'align': 'center', 'border': 1})
            elif score > 0.4:
                level_fmt = wb.add_format({'font_color': '#856404', 'bg_color': '#fff3cd', 'bold': True, 'align': 'center', 'border': 1})
            else:
                level_fmt = wb.add_format({'font_color': 'white', 'bg_color': '#00a854', 'bold': True, 'align': 'center', 'border': 1})
            ws.write(r + 3, 2, level, level_fmt)
        ws.freeze_panes(3, 0)
        ws.set_column(0, 0, 12)
        ws.set_column(1, 1, 15)
        ws.set_column(2, 2, 15)

    # ==== Sheet 8: Technical Indicators (one per ticker) ====
    if technical:
        for ticker in tickers:
            if ticker not in technical:
                continue
            tech_df = technical[ticker]
            if tech_df.empty:
                continue
            sheet_name = f'Tech_{ticker[:25]}'
            ws = wb.add_worksheet(sheet_name)
            ws.set_tab_color('#8b5cf6')
            ws.write(0, 0, f'{ticker} Technical Indicators', fmt_title)
            # Write headers
            ws.write(2, 0, 'Date', fmt_header_teal)
            cols_to_write = [c for c in tech_df.columns if c not in ['Close']][:10]  # Limit columns
            for c, col_name in enumerate(cols_to_write):
                ws.write(2, c + 1, col_name, fmt_header_teal)
            # Write data (last 60 rows to keep manageable)
            display_df = tech_df.tail(60)
            for r in range(len(display_df)):
                is_alt = r % 2 == 0
                idx = display_df.index[r]
                if hasattr(idx, 'to_pydatetime'):
                    ws.write_datetime(r + 3, 0, idx.to_pydatetime(), fmt_date_alt if is_alt else fmt_date)
                else:
                    ws.write(r + 3, 0, str(idx), fmt_row_alt if is_alt else fmt_text)
                for c, col_name in enumerate(cols_to_write):
                    val = display_df.iloc[r][col_name] if col_name in display_df.columns else None
                    if pd.notna(val) and isinstance(val, (int, float)):
                        ws.write_number(r + 3, c + 1, val, fmt_row_alt_num if is_alt else fmt_number)
                    else:
                        ws.write(r + 3, c + 1, '', fmt_row_alt if is_alt else fmt_text)
            ws.freeze_panes(3, 0)
            ws.set_column(0, 0, 14)
            for c in range(len(cols_to_write)):
                ws.set_column(c + 1, c + 1, 12)

    # ==== Sheet 9: Charts ====
    if not prices.empty:
        ws = wb.add_worksheet('Charts')
        ws.set_tab_color('#0a1628')
        ws.write(0, 0, 'Price Charts', fmt_title)

        # Line chart for prices
        chart = wb.add_chart({'type': 'line'})
        chart.set_title({'name': 'Price History'})
        chart.set_size({'width': 720, 'height': 400})
        chart.set_style(10)

        # Write price data to a hidden helper sheet
        helper_ws = wb.add_worksheet('_chart_data')
        helper_ws.hide()
        helper_ws.write(0, 0, 'Date')
        for c, col in enumerate(prices.columns):
            helper_ws.write(0, c + 1, col)
        for r in range(len(prices)):
            helper_ws.write(r + 1, 0, prices.index[r].strftime('%Y-%m-%d'))
            for c in range(len(prices.columns)):
                val = prices.iloc[r, c]
                if pd.notna(val):
                    helper_ws.write_number(r + 1, c + 1, val)
        n_rows = len(prices)
        for c, col in enumerate(prices.columns):
            chart.add_series({
                'name': col,
                'categories': ['_chart_data', 1, 0, n_rows, 0],
                'values': ['_chart_data', 1, c + 1, n_rows, c + 1],
            })
        ws.insert_chart('A3', chart)

    # ==== Sheet: Macro Data ====
    macro_data = data_dict.get('macro_data', {})
    if macro_data:
        ws = wb.add_worksheet('Macro Data')
        ws.set_tab_color('#ff9a00')
        ws.write(0, 0, 'Macroeconomic Data', fmt_title)
        row_idx = 2
        for sid, df in macro_data.items():
            if df.empty:
                continue
            df = df.dropna()
            if df.empty:
                continue
            ws.write(row_idx, 0, sid, fmt_header_navy)
            ws.write(row_idx, 1, 'Date', fmt_header_navy)
            ws.write(row_idx, 2, 'Value', fmt_header_navy)
            # Write last 24 data points
            display = df.tail(24)
            for r in range(len(display)):
                is_alt = r % 2 == 0
                ws.write(row_idx + 1 + r, 0, sid, fmt_row_alt if is_alt else fmt_text)
                idx = display.index[r]
                if hasattr(idx, 'to_pydatetime'):
                    ws.write_datetime(row_idx + 1 + r, 1, idx.to_pydatetime(), fmt_date_alt if is_alt else fmt_date)
                else:
                    ws.write(row_idx + 1 + r, 1, str(idx), fmt_row_alt if is_alt else fmt_text)
                val = display.iloc[r].values[0]
                if pd.notna(val):
                    ws.write_number(row_idx + 1 + r, 2, val, fmt_row_alt_num if is_alt else fmt_number)
            row_idx += len(display) + 3
        ws.set_column(0, 0, 14)
        ws.set_column(1, 1, 14)
        ws.set_column(2, 2, 14)

    # ==== Sheet: ML Analysis ====
    ml_results = data_dict.get('ml_results')
    ml_ticker = data_dict.get('ml_ticker', '')
    if ml_results:
        ws = wb.add_worksheet('ML Analysis')
        ws.set_tab_color('#8b5cf6')
        ws.write(0, 0, f'ML Model Results ({ml_ticker})', fmt_title)
        ws.write(2, 0, 'Model', fmt_header_navy)
        ws.write(2, 1, 'R-squared', fmt_header_navy)
        ws.write(2, 2, 'RMSE', fmt_header_navy)
        ws.write(2, 3, 'MAE', fmt_header_navy)
        for r, (name, info) in enumerate(ml_results['models_info'].items()):
            is_alt = r % 2 == 0
            ws.write(r + 3, 0, name, fmt_row_alt if is_alt else fmt_text)
            ws.write_number(r + 3, 1, info['r2'], fmt_row_alt_num if is_alt else fmt_number)
            ws.write_number(r + 3, 2, info['rmse'], fmt_row_alt_num if is_alt else fmt_number)
            ws.write_number(r + 3, 3, info['mae'], fmt_row_alt_num if is_alt else fmt_number)
        ws.freeze_panes(3, 0)
        ws.set_column(0, 0, 22)
        ws.set_column(1, 3, 14)

        # Feature importance section
        fi_row = 3 + len(ml_results['models_info']) + 2
        ws.write(fi_row, 0, 'Feature Importance', fmt_title)
        fi_row += 1
        fi_headers = ['Feature'] + list(ml_results['feature_importance'].keys())
        for c, h in enumerate(fi_headers):
            ws.write(fi_row, c, h, fmt_header_teal)
        for r, feat in enumerate(ml_results['feature_names']):
            is_alt = r % 2 == 0
            ws.write(fi_row + 1 + r, 0, feat, fmt_row_alt if is_alt else fmt_text)
            for c, (model_name, imp_dict) in enumerate(ml_results['feature_importance'].items()):
                val = imp_dict.get(feat, 0)
                ws.write_number(fi_row + 1 + r, c + 1, val, fmt_row_alt_num if is_alt else fmt_number)

    # ==== Sheet: Options Chain ====
    options_data = data_dict.get('options', {})
    if options_data and options_data.get('calls') is not None and not options_data.get('calls', pd.DataFrame()).empty:
        ws = wb.add_worksheet('Options Chain')
        ws.set_tab_color('#ff9a00')
        ws.write(0, 0, f"Options Chain ({options_data.get('ticker', '')} - {options_data.get('expiration', '')})", fmt_title)
        # Calls
        calls = options_data['calls']
        ws.write(2, 0, 'CALLS', fmt_header_navy)
        cols_to_export = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        for c, col_name in enumerate(cols_to_export):
            ws.write(3, c, col_name, fmt_header_teal)
        for r in range(min(len(calls), 30)):
            is_alt = r % 2 == 0
            for c, col_name in enumerate(cols_to_export):
                val = calls.iloc[r].get(col_name)
                if pd.notna(val) and isinstance(val, (int, float)):
                    ws.write_number(r + 4, c, val, fmt_row_alt_num if is_alt else fmt_number)
                else:
                    ws.write(r + 4, c, str(val) if pd.notna(val) else 'N/A', fmt_row_alt if is_alt else fmt_text)
        ws.freeze_panes(4, 0)
        for c in range(len(cols_to_export)):
            ws.set_column(c, c, 14)
        # Puts
        puts = options_data.get('puts', pd.DataFrame())
        if not puts.empty:
            put_start = min(len(calls), 30) + 6
            ws.write(put_start, 0, 'PUTS', fmt_header_navy)
            for c, col_name in enumerate(cols_to_export):
                ws.write(put_start + 1, c, col_name, fmt_header_teal)
            for r in range(min(len(puts), 30)):
                is_alt = r % 2 == 0
                for c, col_name in enumerate(cols_to_export):
                    val = puts.iloc[r].get(col_name)
                    if pd.notna(val) and isinstance(val, (int, float)):
                        ws.write_number(put_start + 2 + r, c, val, fmt_row_alt_num if is_alt else fmt_number)
                    else:
                        ws.write(put_start + 2 + r, c, str(val) if pd.notna(val) else 'N/A', fmt_row_alt if is_alt else fmt_text)

    # ==== Sheet: Risk Dashboard ====
    risk_export = data_dict.get('risk', {})
    if risk_export:
        ws = wb.add_worksheet('Risk Dashboard')
        ws.set_tab_color('#ff4d6d')
        ws.write(0, 0, 'Risk & Geopolitics Dashboard', fmt_title)
        risk_d = risk_export.get('risk_data', {})
        score = risk_export.get('composite_score', 0)
        ws.write(2, 0, 'Indicator', fmt_header_navy)
        ws.write(2, 1, 'Latest Value', fmt_header_navy)
        row_idx = 3
        for name in ['VIX', 'DXY', 'Gold', 'Oil', 'TNX', 'IRX', 'SPX']:
            s = risk_d.get(name, pd.Series(dtype=float))
            if len(s) > 0:
                is_alt = (row_idx - 3) % 2 == 0
                ws.write(row_idx, 0, name, fmt_row_alt if is_alt else fmt_text)
                ws.write_number(row_idx, 1, float(s.iloc[-1]), fmt_row_alt_num if is_alt else fmt_number)
                row_idx += 1
        ws.write(row_idx, 0, 'Composite Risk Score', fmt_text_bold)
        ws.write_number(row_idx, 1, score, fmt_number)
        ws.freeze_panes(3, 0)
        ws.set_column(0, 0, 22)
        ws.set_column(1, 1, 16)

    # ==== Sheet: ML Clustering ====
    clustering_data = data_dict.get('clustering', {})
    if clustering_data and 'features' in clustering_data:
        ws = wb.add_worksheet('ML Clustering')
        ws.set_tab_color('#9055c8')
        ws.write(0, 0, 'ML Clustering Results', fmt_title)
        feat = clustering_data['features']
        _write_df_sheet(ws, feat, fmt_header_navy, start_row=2, col_formats={
            'Ann Return': (fmt_pct, fmt_row_alt_pct),
            'Volatility': (fmt_pct, fmt_row_alt_pct),
            'Sharpe': (fmt_number, fmt_row_alt_num),
            'Max Drawdown': (fmt_pct, fmt_row_alt_pct),
        })

    # ==== Sheet: Sentiment ====
    sentiment_data = data_dict.get('sentiment', {})
    if sentiment_data and 'articles' in sentiment_data:
        ws = wb.add_worksheet('Sentiment')
        ws.set_tab_color('#00d084')
        ws.write(0, 0, f"Sentiment Analysis ({sentiment_data.get('ticker', '')})", fmt_title)
        articles = sentiment_data['articles']
        ws.write(2, 0, 'Headline', fmt_header_navy)
        ws.write(2, 1, 'Score', fmt_header_navy)
        for r in range(min(len(articles), 30)):
            is_alt = r % 2 == 0
            ws.write(r + 3, 0, str(articles.iloc[r].get('Headline', '')), fmt_row_alt if is_alt else fmt_text)
            score_val = articles.iloc[r].get('Score', 0)
            if pd.notna(score_val):
                ws.write_number(r + 3, 1, float(score_val), fmt_row_alt_num if is_alt else fmt_number)
            else:
                ws.write(r + 3, 1, 'N/A', fmt_row_alt if is_alt else fmt_text)
        ws.freeze_panes(3, 0)
        ws.set_column(0, 0, 60)
        ws.set_column(1, 1, 12)

    wb.close()
    return output.getvalue()

# ========================================================================
# MAIN APPLICATION
# ========================================================================

def main():
    # 1. Create a placeholder at the VERY TOP of the app
    header_placeholder = st.empty()

    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")

        # Theme Toggle
        dark_mode = st.toggle("Dark Mode", value=(st.session_state.theme == 'dark'))
        new_theme = 'dark' if dark_mode else 'light'
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

    # Inject CSS AFTER theme state is resolved
    inject_custom_css(st.session_state.theme)

    # Sidebar (continued)
    with st.sidebar:

        # Asset Class Selector
        st.markdown("### Asset Class")
        PRESET_TICKERS = {
            "Stocks & ETFs": "NVDA TSLA AAPL MSFT GOOGL",
            "Forex": "EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCHF=X",
            "Commodities": "GC=F SI=F CL=F NG=F HG=F",
            "Crypto": "BTC-USD ETH-USD SOL-USD XRP-USD ADA-USD",
            "Options": "AAPL MSFT NVDA TSLA AMZN",
        }
        asset_class = st.selectbox("Select Asset Class", list(PRESET_TICKERS.keys()), key='asset_class_selector')

        # Quick Presets
        st.markdown("### Quick Presets")

        QUICK_PRESETS = {
            "— None —": "",
            # Stocks
            "Tech Giants": "AAPL MSFT GOOGL AMZN META NVDA",
            "Semiconductor": "NVDA AMD INTC TSM AVGO QCOM",
            "EV & Clean Energy": "TSLA RIVN LCID NIO ENPH FSLR",
            "FAANG+": "META AAPL AMZN NFLX GOOGL MSFT",
            "Banking & Finance": "JPM BAC GS MS WFC C",
            "Healthcare & Pharma": "JNJ PFE UNH ABBV MRK LLY",
            "Energy & Oil": "XOM CVX COP SLB EOG OXY",
            "Mega Cap": "AAPL MSFT GOOGL AMZN NVDA META TSLA BRK-B",
            "Dividend Aristocrats": "JNJ PG KO PEP MMM ABT EMR",
            "Growth Stocks": "SHOP SNOW CRWD DDOG NET PLTR",
            "Value Stocks": "BRK-B JPM XOM JNJ PG BAC",
            # Crypto
            "Crypto Majors": "BTC-USD ETH-USD SOL-USD XRP-USD ADA-USD AVAX-USD",
            "Crypto DeFi": "UNI7083-USD AAVE-USD MKR-USD LINK-USD SNX-USD",
            # ETFs
            "Index ETFs": "SPY QQQ DIA IWM VTI VOO",
            "Bond ETFs": "TLT IEF SHY BND AGG LQD",
            "Sector ETFs": "XLK XLF XLE XLV XLI XLP",
            "Commodity ETFs": "GLD SLV USO UNG DBA DBC",
            # Forex
            "Forex Majors": "EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCHF=X NZDUSD=X",
            "Forex Emerging": "USDMXN=X USDBRL=X USDINR=X USDTRY=X USDZAR=X",
            # Commodities
            "Precious Metals": "GC=F SI=F PA=F PL=F",
            "Energy Futures": "CL=F NG=F RB=F HO=F",
            "Agricultural": "ZC=F ZW=F ZS=F KC=F SB=F CC=F",
        }

        preset_choice = st.selectbox(
            "Quick Presets",
            list(QUICK_PRESETS.keys()),
            key='quick_preset_selector'
        )

        if preset_choice != "— None —":
            st.session_state.preset = QUICK_PRESETS[preset_choice]

        # Determine default tickers: preset overrides asset class
        if preset_choice != "— None —":
            default_tickers = QUICK_PRESETS[preset_choice]
        else:
            default_tickers = PRESET_TICKERS.get(asset_class, "NVDA TSLA AAPL")
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
            # --- Monte Carlo & Simulation ---
            st.markdown("**Monte Carlo & Simulation**")
            n_sims = st.slider("Monte Carlo Simulations", 100, 2000, 500)
            n_days = st.slider("Forecast Days", 30, 365, 90)
            confidence_level = st.select_slider("Confidence Level", options=[90, 95, 99], value=95)
            sim_method = st.selectbox("Simulation Method", ["Geometric Brownian Motion", "Behavioral Agent Model"], index=1)

            st.divider()

            # --- Portfolio Optimization ---
            st.markdown("**Portfolio Optimization**")
            bubble_aware = st.checkbox("Bubble-Aware Portfolio", value=True)
            penalty_factor = st.slider("Bubble Penalty", 0.0, 1.0, 0.5)
            benchmark_options = {
                "S&P 500 (^GSPC)": "^GSPC",
                "NASDAQ (^IXIC)": "^IXIC",
                "Dow Jones (^DJI)": "^DJI",
                "Russell 2000 (^RUT)": "^RUT",
                "None": None,
            }
            benchmark_label = st.selectbox("Benchmark", list(benchmark_options.keys()))
            benchmark_ticker = benchmark_options[benchmark_label]
            use_custom_rf = st.checkbox("Use Custom Risk-Free Rate", value=False)
            if use_custom_rf:
                custom_rf = st.number_input("Risk-Free Rate Override (%)", min_value=0.0, max_value=20.0, value=4.5, step=0.1)
            else:
                custom_rf = None
            rebalancing = st.selectbox("Rebalancing Frequency", ["None (Buy & Hold)", "Monthly", "Quarterly", "Annually"])

            st.divider()

            # --- ML & Analysis ---
            st.markdown("**ML & Analysis**")
            ml_training_years = st.slider("ML Training Period (years)", 1, 5, 3)
            clustering_method = st.selectbox("Clustering Method", ["K-Means", "Gaussian Mixture"])
            anomaly_sensitivity = st.slider("Anomaly Sensitivity", 0.01, 0.15, 0.05, 0.01, help="Lower = fewer anomalies detected")

            st.divider()

            # --- Display & Export ---
            st.markdown("**Display & Export**")
            enable_autorefresh = st.toggle("Enable Auto-Refresh", value=False)
            refresh_rate = st.number_input("Refresh Rate (seconds)", min_value=10, value=60)
            chart_height = st.slider("Chart Height (px)", 300, 800, 500, 50)
            table_row_limit = st.number_input("Max Table Rows", min_value=10, max_value=500, value=50)
            export_sections = st.multiselect(
                "Include in Reports",
                ["Price Charts", "Portfolio Weights", "Bubble Scores", "Technical Indicators",
                 "Macro Data", "ML Predictions", "Options Data", "Risk Dashboard", "Clustering", "Sentiment"],
                default=["Price Charts", "Portfolio Weights", "Bubble Scores", "Technical Indicators",
                         "Macro Data", "ML Predictions"]
            )

            st.divider()
            st.markdown("**Developer Options**")
            debug_mode = st.checkbox(
                "Debug Mode",
                value=st.session_state.get('debug_mode', False),
                help="Show full technical error details and stack traces for troubleshooting."
            )
            st.session_state.debug_mode = debug_mode
            if debug_mode:
                st.caption("🔧 Debug mode ON — full error traces will be shown in error messages.")

        # Override risk-free rate if custom is set
        if use_custom_rf and custom_rf is not None:
            rf_rate = custom_rf / 100.0

        analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    # Main Analysis Logic
    should_run = analyze_btn or (st.session_state.analysis_complete and enable_autorefresh)

    if should_run:
        _progress = st.progress(0, text='Initializing...')
        try:
            # ── Step 1: Parse & validate tickers ──────────────────────────
            _progress.progress(5, text='Validating tickers...')
            tickers = [t for t in re.split(r'[\s,;]+', tickers_input.strip()) if t]
            tickers = [re.sub(r'[^A-Za-z0-9.=^-]', '', t).upper() for t in tickers if t.strip()]
            if not tickers:
                _progress.empty()
                show_error('invalid_ticker')
                return

            # ── Step 2: Fetch market data ──────────────────────────────────
            _progress.progress(15, text=f'Fetching market data for {len(tickers)} ticker(s)...')
            try:
                prices, volumes = fetch_market_data(tickers, start_date, end_date)
            except DataFetchError as e:
                _progress.empty()
                show_error('no_data', e.message)
                return

            # Warn about any partially missing tickers (non-fatal)
            fetched = [t for t in tickers if t in prices.columns and not prices[t].isna().all()]
            missing_tickers = [t for t in tickers if t not in fetched]
            if missing_tickers:
                st.warning(
                    f'⚠️  No data found for: **{", ".join(missing_tickers)}**. '
                    'These tickers have been skipped. Check spelling or try a different date range.'
                )
            tickers = fetched  # continue with valid tickers only
            if not tickers:
                _progress.empty()
                show_error('no_data')
                return

            st.session_state.last_updated = pd.Timestamp.now('America/New_York').strftime('%Y-%m-%d %I:%M:%S %p')

            # ── Step 3: Returns & performance metrics ─────────────────────
            _progress.progress(25, text='Computing performance metrics...')
            try:
                returns = prices.pct_change().bfill().dropna(how='all')
                metrics = {}
                for ticker in tickers:
                    if ticker not in returns.columns:
                        continue
                    r = returns[ticker].dropna()
                    if r.empty:
                        continue
                    ann_vol = r.std() * np.sqrt(252)
                    metrics[ticker] = {
                        'Annual Return': r.mean() * 252,
                        'Volatility': ann_vol,
                        'Sharpe': (r.mean() * 252 - rf_rate) / ann_vol if ann_vol > 0 else 0.0,
                        'Max Drawdown': ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(),
                    }
                metrics_df = pd.DataFrame(metrics).T
            except Exception as e:
                _progress.empty()
                show_error('calculation_error', traceback.format_exc())
                return

            # ── Step 4: Valuation & bubble detection ──────────────────────
            _progress.progress(40, text='Running valuation models & bubble detection...')
            valuation_results = {}
            bubble_detector = BubbleDetector()
            bubble_scores = {}
            if benchmark_ticker:
                benchmark_prices = get_benchmark_data(start_date, end_date, benchmark_ticker)
            else:
                benchmark_prices = pd.Series()

            for ticker in tickers:
                try:
                    beta = (
                        EnhancedValuationMetrics.calculate_beta(prices[ticker], benchmark_prices)
                        if not benchmark_prices.empty and ticker in prices.columns
                        else 1.0
                    )
                    wacc = EnhancedValuationMetrics.calculate_wacc(ticker, rf_rate, beta)
                    capm = EnhancedValuationMetrics.calculate_capm_return(rf_rate, beta)
                    ff   = EnhancedValuationMetrics.calculate_fama_french_return(ticker, prices[[ticker]], rf_rate, beta)
                    apt  = EnhancedValuationMetrics.calculate_apt_return(ticker, prices[[ticker]], rf_rate)
                    vol_data = volumes[ticker] if ticker in volumes.columns else None
                    bubble_res = bubble_detector.detect_bubbles(prices[ticker], returns[ticker], vol_data)
                    bubble_scores[ticker] = bubble_res['bubble_score']
                    impact = EnhancedValuationMetrics.calculate_bubble_burst_impact(
                        ticker, prices[ticker], bubble_res['bubble_score'], beta
                    )
                    dcf = EnhancedValuationMetrics.calculate_dcf_value(ticker, rf_rate, beta)
                    valuation_results[ticker] = {
                        'DCF Enterprise Value': dcf, 'WACC': wacc, 'CAPM Return': capm,
                        'Fama-French Return': ff, 'APT Return': apt,
                        'Bubble Score': bubble_res['bubble_score'],
                        'Bubble Burst Impact': impact, 'Beta': beta,
                    }
                except Exception as e:
                    _logger.warning('Valuation failed for %s: %s', ticker, e)
                    st.warning(f'⚠️  Valuation models could not be computed for **{ticker}** — skipping. ({e})')
                    bubble_scores.setdefault(ticker, 0.0)

            valuation_df = pd.DataFrame(valuation_results).T

            # ── Step 5: Portfolio optimisation ────────────────────────────
            _progress.progress(60, text='Optimising portfolio...')
            try:
                optimizer = EnhancedPortfolioOptimizer(prices, bubble_scores, rf_rate)
                portfolio_results = {
                    'Min Variance': optimizer.minimum_variance(bubble_aware, penalty_factor),
                    'Risk Parity':  optimizer.risk_parity(bubble_aware, penalty_factor),
                    'Min CVaR':     optimizer.minimum_cvar(bubble_aware=bubble_aware, penalty_factor=penalty_factor),
                }
                portfolio_metrics = {
                    s: optimizer.calculate_portfolio_metrics(w)
                    for s, w in portfolio_results.items()
                }
            except Exception as e:
                _logger.warning('Portfolio optimisation failed: %s', e)
                st.warning(f'⚠️  Portfolio optimisation encountered an issue — falling back to equal-weight. ({e})')
                portfolio_results  = {'Equal Weight': optimizer.equal_weight()}
                portfolio_metrics  = {'Equal Weight': optimizer.calculate_portfolio_metrics(portfolio_results['Equal Weight'])}

            # ── Step 6: Technical indicators ──────────────────────────────
            _progress.progress(72, text='Computing technical indicators...')
            technical_indicators = {}
            for ticker in tickers:
                try:
                    technical_indicators[ticker] = TechnicalIndicators.calculate_all(prices[ticker])
                except Exception as e:
                    _logger.warning('Technical indicators failed for %s: %s', ticker, e)

            # ── Step 7: Monte Carlo simulation ────────────────────────────
            _progress.progress(85, text='Running Monte Carlo simulation...')
            sim_ticker = tickers[0]
            try:
                sim_engine = BehavioralAgentSimulator(sim_ticker, prices[sim_ticker])
                sim_prices, sim_regimes, sim_intrinsic = sim_engine.run(n_days, n_sims)
            except Exception as e:
                _logger.warning('Monte Carlo failed: %s', e)
                st.warning(f'⚠️  Monte Carlo simulation could not run for **{sim_ticker}**. ({e})')
                sim_prices, sim_regimes, sim_intrinsic = None, None, None

            # ── Step 8: Store results & complete ──────────────────────────
            _progress.progress(95, text='Finalising results...')
            st.session_state.data = {
                'prices': prices, 'returns': returns,
                'metrics': metrics_df, 'valuation': valuation_df,
                'portfolio': portfolio_results, 'portfolio_metrics': portfolio_metrics,
                'bubble_scores': bubble_scores, 'technical': technical_indicators,
                'simulation': (sim_ticker, sim_prices, sim_regimes, sim_intrinsic),
                'tickers': tickers, 'rf_rate': rf_rate, 'volumes': volumes,
                'confidence_level': confidence_level, 'benchmark_ticker': benchmark_ticker,
                'ml_training_years': ml_training_years, 'clustering_method': clustering_method,
                'anomaly_sensitivity': anomaly_sensitivity, 'chart_height': chart_height,
                'table_row_limit': table_row_limit, 'export_sections': export_sections,
                'rebalancing': rebalancing, 'sim_method': sim_method,
            }
            st.session_state.analysis_complete = True
            _progress.progress(100, text='Analysis complete!')
            time.sleep(0.4)
            _progress.empty()
            st.success('✅  Analysis completed successfully!')

        except QuantLabError as e:
            _progress.empty()
            show_error('calculation_error', e.message)
        except Exception as e:
            _progress.empty()
            _logger.error('Unexpected top-level error: %s', traceback.format_exc())
            show_error('calculation_error',
                       traceback.format_exc() if st.session_state.get('debug_mode') else str(e))

    # 2. Render the Header NOW (using the latest timestamp) inside the placeholder
    last_run = st.session_state.last_updated
    _logo_bg = '#0a1628' if st.session_state.theme == 'dark' else '#f3f4f6'
    _logo_teal = '#00b4d8' if st.session_state.theme == 'dark' else '#0090b5'
    _logo_gold = '#ffd700' if st.session_state.theme == 'dark' else '#b8860b'
    with header_placeholder.container():
        st.markdown(f"""
        <div class="ql-header">
            <div class="ql-header-left">
                <svg class="ql-logo-mark" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect width="40" height="40" rx="8" fill="{_logo_bg}"/>
                    <path d="M10 28L16 12h3l6 16h-3l-1.5-4.5h-6L13 28h-3zm5.5-7.5h4l-2-6-2 6z" fill="{_logo_teal}"/>
                    <circle cx="30" cy="14" r="3" fill="{_logo_gold}" opacity="0.9"/>
                </svg>
                <div>
                    <div class="ql-title">Quant<span>Lab</span></div>
                    <div class="ql-subtitle">Advanced Portfolio Analytics</div>
                </div>
            </div>
            <div class="ql-header-right">
                <div class="ql-live-badge">
                    <span class="live-dot"></span>
                    LIVE DATA
                </div>
                <div class="ql-timestamp">{last_run}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display Results
    if st.session_state.analysis_complete:
        data = st.session_state.data
        _chart_h = data.get('chart_height', 500)
        _tbl_limit = data.get('table_row_limit', 50)
        
        # Quick Stats
        st.markdown("""
        <div class="section-header">
            <div class="section-label">OVERVIEW</div>
            <div class="section-title">Market Dashboard</div>
            <div class="section-subtitle">Real-time performance metrics and price analysis</div>
        </div>
        """, unsafe_allow_html=True)
        # Quick Stats - Native Streamlit Metrics
        cols = st.columns(min(len(data['tickers']), 5))
        for i, ticker in enumerate(data['tickers'][:5]):
            with cols[i]:
                last_price = data['prices'][ticker].iloc[-1]
                change = (last_price / data['prices'][ticker].iloc[-2] - 1) * 100
                bubble_score = data['bubble_scores'][ticker]

                if bubble_score > 0.7:
                    badge_label = "HIGH RISK"
                elif bubble_score > 0.4:
                    badge_label = "CAUTION"
                else:
                    badge_label = "NORMAL"

                st.metric(
                    ticker,
                    f"${last_price:.2f}",
                    f"{change:+.2f}%"
                )
                st.caption(f"Bubble: {badge_label} ({bubble_score:.0%})")
                st.progress(min(bubble_score, 1.0))
        
        # Tabs
        tabs = st.tabs([
            "Market Dashboard",         # 0
            "Valuation",                # 1
            "Portfolio",                # 2
            "Bubble Detection",         # 3
            "Monte Carlo",              # 4
            "Technicals",               # 5
            "Options Pricing",          # 6
            "Macro Dashboard",          # 7
            "Risk & Geopolitics",       # 8
            "ML Predictions",           # 9
            "ML Clustering",            # 10
            "Sentiment Analysis",       # 11
            "Backtesting",              # 12 NEW
            "Fundamentals",             # 13 NEW
            "Fixed Income",             # 14 NEW
            "Factor Model",             # 15 NEW
            "Options Builder",          # 16 NEW
            "Risk Suite",               # 17 NEW
            "Export",                   # 18
        ])
        
        with tabs[0]:  # Market Dashboard
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(plot_price_history(data['prices']), use_container_width=True)
            with col2:
                st.markdown("#### Performance Metrics")
                render_styled_table(
                    data['metrics'].head(_tbl_limit),
                    format_dict={
                        'Annual Return': '{:.2%}',
                        'Volatility': '{:.2%}',
                        'Sharpe': '{:.2f}',
                        'Max Drawdown': '{:.2%}'
                    }
                )
            with st.expander("Understanding Performance Metrics"):
                st.markdown("""
                **Normalized Performance (Base 100)** rebases all assets to 100 at the start date,
                allowing direct comparison of percentage returns regardless of absolute price levels.
                """)
                st.latex(r"P_{normalized}(t) = \frac{P(t)}{P(0)} \times 100")
                st.markdown("""
                **Annual Return** — Compound annual growth rate (CAGR) of the asset:
                """)
                st.latex(r"R_{annual} = \left(\frac{P_{end}}{P_{start}}\right)^{\frac{252}{N_{days}}} - 1")
                st.markdown("""
                **Volatility** — Annualized standard deviation of daily returns, measuring price uncertainty:
                """)
                st.latex(r"\sigma_{annual} = \sigma_{daily} \times \sqrt{252}")
                st.markdown("""
                **Sharpe Ratio** — Risk-adjusted return (higher is better, >1 is good, >2 is excellent):
                """)
                st.latex(r"S = \frac{R_p - R_f}{\sigma_p}")
                st.markdown("""
                Where R_p is the portfolio return, R_f is the risk-free rate, and sigma_p is the portfolio volatility.

                **Max Drawdown** — The largest peak-to-trough decline, measuring worst-case loss:
                """)
                st.latex(r"MDD = \frac{Trough - Peak}{Peak}")

        with tabs[1]:  # Enhanced Valuation
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 02</div>
                <div class="section-title">Enhanced Valuation</div>
                <div class="section-subtitle">DCF, CAPM, Fama-French, and APT multi-factor models</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding Valuation Models"):
                st.markdown("**DCF Enterprise Value** — Present value of all expected future free cash flows:")
                st.latex(r"EV = \sum_{t=1}^{n} \frac{FCF_t}{(1 + WACC)^t} + \frac{TV}{(1 + WACC)^n}")
                st.markdown("**WACC (Weighted Average Cost of Capital)** — Blended cost of debt and equity financing:")
                st.latex(r"WACC = \frac{E}{V} \cdot R_e + \frac{D}{V} \cdot R_d \cdot (1 - T_c)")
                st.markdown("**CAPM (Capital Asset Pricing Model)** — Expected return based on systematic risk:")
                st.latex(r"E(R_i) = R_f + \beta_i \cdot (E(R_m) - R_f)")
                st.markdown("""
                Where Beta measures sensitivity to market movements. Beta > 1 means more volatile than market.

                **Fama-French Three-Factor Model** — Extends CAPM with size and value factors:
                """)
                st.latex(r"R_i - R_f = \alpha + \beta_M (R_m - R_f) + \beta_S \cdot SMB + \beta_V \cdot HML")
                st.markdown("""
                - **SMB** (Small Minus Big) — Size premium
                - **HML** (High Minus Low) — Value premium

                **APT (Arbitrage Pricing Theory)** — Multi-factor model allowing any number of risk factors:
                """)
                st.latex(r"E(R_i) = R_f + \sum_{j=1}^{k} \beta_{ij} \cdot \lambda_j")
                st.markdown("""
                **Bubble Score** — Composite score (0-100%) combining Metcalfe's Law deviation, long-memory analysis,
                and excess kurtosis. Higher scores indicate greater bubble risk.

                **Bubble Burst Impact** — Estimated potential loss if a bubble corrects, based on the distance
                between current price and estimated fair value.
                """)

            # Format the dataframe for display
            display_df = data['valuation'].copy()
            
            # Format columns appropriately
            format_dict = {}
            for col in display_df.columns:
                if 'Value' in col:
                    format_dict[col] = lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                elif any(x in col for x in ['Return', 'WACC', 'Score', 'Impact']):
                    format_dict[col] = '{:.2%}'
                elif 'Beta' in col:
                    format_dict[col] = '{:.2f}'
            
            render_styled_table(
                display_df.head(_tbl_limit),
                format_dict=format_dict
            )

            # Valuation insights
            st.markdown("#### Key Insights")
            
            for ticker in data['tickers']:
                val = data['valuation'].loc[ticker]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"{ticker} Beta", f"{val.get('Beta', 1.0):.2f}")
                with col2:
                    st.metric(f"{ticker} CAPM", f"{val['CAPM Return']:.2%}")
                with col3:
                    st.metric(f"{ticker} Bubble Score", f"{val['Bubble Score']:.0%}")
                with col4:
                    st.metric(f"{ticker} Risk Impact", f"{val['Bubble Burst Impact']:.0%}")
            
            # Download button for valuation data
            excel_valuation = io.BytesIO()
            data['valuation'].to_excel(excel_valuation)
            st.download_button(
                "Download Valuation Data",
                excel_valuation.getvalue(),
                f"valuation_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with tabs[2]:  # Portfolio Optimization
            render_portfolio_optimization_tab(data)
        
        with tabs[3]:  # Bubble Detection
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 04</div>
                <div class="section-title">Bubble Detection</div>
                <div class="section-subtitle">Metcalfe's Law, long-memory analysis, and composite risk scoring</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding Bubble Detection"):
                st.markdown("**Metcalfe's Law Ratio (MMV)** — Compares market cap to network-value estimates. A ratio > 1 suggests overvaluation relative to fundamental network effects:")
                st.latex(r"MMV = \frac{MarketCap}{C \cdot \log(users)^2}")
                st.markdown("**Long-Memory (d Parameter)** — Hurst exponent variant measuring persistence in returns. Values > 0.5 suggest trending/bubble behavior:")
                st.latex(r"d \in (0, 0.5) \Rightarrow \text{long memory (trend persistence)}")
                st.latex(r"d \approx 0 \Rightarrow \text{short memory (mean reverting)}")
                st.markdown("**Excess Kurtosis** — Measures the fatness of return distribution tails. Higher kurtosis means more extreme events:")
                st.latex(r"Kurt = \frac{E[(X-\mu)^4]}{\sigma^4} - 3")
                st.markdown("""
                Kurtosis > 3 (excess > 0) indicates fat tails — more frequent extreme returns than a normal distribution.

                **Composite Bubble Score** — Weighted average of multiple indicators:
                """)
                st.latex(r"BubbleScore = w_1 \cdot MMV_{norm} + w_2 \cdot d_{norm} + w_3 \cdot Kurt_{norm} + w_4 \cdot Vol_{norm}")
                st.markdown("""
                **Regime Classification:**
                - **Normal** (0-40%): No significant bubble indicators
                - **Caution** (40-70%): Elevated risk, some bubble characteristics present
                - **High Risk** (70-100%): Strong bubble indicators, significant correction risk
                """)

            selected_ticker = st.selectbox("Select Asset", data['tickers'])
            
            # NOTE: We re-calculate here for display, but ideally we use stored scores.
            # Re-fetching volume just for chart context
            vol_display = None # Simplified for display chart logic
            
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
            regime_badge_map = {
                'Extreme Bubble': 'badge-sell',
                'Bubble Formation': 'badge-caution',
                'Fair Value': 'badge-safe',
                'Undervalued': 'badge-info',
                'Transition': 'badge-neutral',
                'Unknown': 'badge-neutral'
            }
            badge_cls = regime_badge_map.get(regime, 'badge-neutral')

            st.markdown(f"""
            <div style="margin:16px 0 20px 0;">
                <span style="font-size:16px;font-weight:700;color:var(--white);margin-right:12px;">Current Regime</span>
                <span class="badge {badge_cls}" style="font-size:13px;padding:5px 14px;">{regime}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Bubble analysis chart
            st.plotly_chart(
                plot_bubble_analysis(
                    data['prices'][selected_ticker],
                    bubble_res,
                    selected_ticker
                ),
                use_container_width=True
            )
        
        with tabs[4]:  # Monte Carlo Simulation
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 05</div>
                <div class="section-title">Monte Carlo Simulation</div>
                <div class="section-subtitle">Behavioral agent-based stochastic projections</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding Monte Carlo Simulation"):
                st.markdown("**Monte Carlo simulation** generates thousands of possible future price paths using random sampling, modeling uncertainty in asset returns.")
                st.markdown("**Geometric Brownian Motion (GBM)** — The foundational model for each price path:")
                st.latex(r"S_{t+1} = S_t \cdot \exp\left[(\mu - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} \cdot Z\right]")
                st.markdown("Where Z ~ N(0,1) is a standard normal random variable.")
                st.markdown("**Behavioral Agent Extension** — Enhances GBM with regime-switching agents (momentum traders, mean-reversion traders, noise traders) that shift the drift and volatility parameters based on market conditions.")
                st.markdown("**Key Output Metrics:**")
                st.markdown("""
                - **Median Price** — The 50th percentile of all simulated final prices
                - **95% VaR (Value at Risk)** — The price level below which only 5% of simulations fall. Measures downside risk:
                """)
                st.latex(r"VaR_{95\%} = P_{5th\ percentile}")
                st.markdown("""
                - **95% CVaR (Conditional VaR)** — The expected price in the worst 5% of scenarios (more conservative than VaR):
                """)
                st.latex(r"CVaR_{95\%} = E[S_T \mid S_T \leq VaR_{95\%}]")
                st.markdown("The shaded band on the chart shows the 90% confidence interval (5th to 95th percentile) — wider bands indicate higher uncertainty.")

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
                st.error(f"'{sim_ticker}' has no price data.")
                sim_ticker = None

            if sim_ticker in data['prices'].columns:
                # Re-run simulation for the selected ticker if needed,
                # or just run it here for display.
                # Using defaults for quick interactivity:
                n_days_sim = 252
                n_sims_sim = 1000

                sim_engine = BehavioralAgentSimulator(sim_ticker, data['prices'][sim_ticker])
                sim_prices, sim_regimes, sim_intrinsic = sim_engine.run(n_days_sim, n_sims_sim)

                # Calculate statistics
                final_prices = sim_prices[:, -1]

                conf = data.get('confidence_level', 95)
                var_pct = 100 - conf  # e.g. 95 -> 5th percentile

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Median Price", f"${np.median(final_prices):.2f}")
                with col2:
                    st.metric(f"{conf}% VaR", f"${np.percentile(final_prices, var_pct):.2f}")
                with col3:
                    var_threshold = np.percentile(final_prices, var_pct)
                    tail_prices = final_prices[final_prices <= var_threshold]
                    cvar_val = np.mean(tail_prices) if len(tail_prices) > 0 else var_threshold
                    st.metric(f"{conf}% CVaR", f"${cvar_val:.2f}")

                # Simulation chart
                days = np.arange(sim_prices.shape[1])
                p5 = np.percentile(sim_prices, var_pct, axis=0)
                p50 = np.percentile(sim_prices, 50, axis=0)
                p95 = np.percentile(sim_prices, 100 - var_pct, axis=0)

                _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                _fill = 'rgba(0,180,216,0.15)' if st.session_state.get('theme') == 'dark' else 'rgba(0,144,181,0.12)'
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=days, y=p95, line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=days, y=p5, fill='tonexty',
                    fillcolor=_fill,
                    name=f'{100 - 2 * var_pct}% Confidence Interval',
                    line=dict(width=0)
                ))
                fig.add_trace(go.Scatter(
                    x=days, y=p50, name='Median',
                    line=dict(color=_clrs[0], width=2)
                ))

                _chart_h = data.get('chart_height', 500)
                fig.update_layout(
                    title=dict(text=f"Monte Carlo Projection: {sim_ticker}", font=dict(color=_fc)),
                    template=_tmpl,
                    height=_chart_h,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc)),
                    yaxis=dict(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc)),
                    legend=dict(font=dict(color=_fc)),
                    font=dict(family="Inter, system-ui, sans-serif", color=_fc)
                )

                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[5]:  # Technical Analysis
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 06</div>
                <div class="section-title">Technical Analysis</div>
                <div class="section-subtitle">RSI, MACD, Bollinger Bands, and moving average indicators</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding Technical Indicators"):
                st.markdown("**RSI (Relative Strength Index)** — Momentum oscillator measuring speed and magnitude of price changes (0-100):")
                st.latex(r"RSI = 100 - \frac{100}{1 + \frac{AvgGain_{14}}{AvgLoss_{14}}}")
                st.markdown("""
                - RSI > 70: **Overbought** — potential reversal or pullback
                - RSI < 30: **Oversold** — potential bounce or recovery
                - RSI ~ 50: Neutral momentum
                """)
                st.markdown("**MACD (Moving Average Convergence Divergence)** — Trend-following momentum indicator:")
                st.latex(r"MACD = EMA_{12} - EMA_{26}")
                st.latex(r"Signal = EMA_9(MACD)")
                st.markdown("""
                - MACD crossing above Signal: **Bullish** signal
                - MACD crossing below Signal: **Bearish** signal
                - Histogram (MACD - Signal) shows momentum strength
                """)
                st.markdown("**SMA (Simple Moving Average)** — Smooths price data over N periods:")
                st.latex(r"SMA_N = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}")
                st.markdown("""
                - **SMA 20** (short-term): Captures recent trends
                - **SMA 50** (medium-term): Identifies intermediate direction
                - **Golden Cross** (SMA 20 > SMA 50): Bullish trend signal
                - **Death Cross** (SMA 20 < SMA 50): Bearish trend signal
                """)
                st.markdown("**Bollinger Bands** — Volatility-adjusted price channels:")
                st.latex(r"Upper = SMA_{20} + 2\sigma_{20}")
                st.latex(r"Lower = SMA_{20} - 2\sigma_{20}")
                st.markdown("Prices touching the upper band may be overbought; touching the lower band may be oversold.")

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
                _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price & Moving Averages', 'MACD', 'RSI')
                )

                # Price and MAs
                fig.add_trace(
                    go.Scatter(x=data['prices'].index, y=data['prices'][tech_ticker],
                               name='Price', line=dict(color=_clrs[0], width=2)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['SMA_20'],
                               name='SMA 20', line=dict(color=_clrs[1], width=1)),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['SMA_50'],
                               name='SMA 50', line=dict(color=_clrs[4], width=1)),
                    row=1, col=1
                )

                # Bollinger Bands
                if 'BB_Upper' in tech_data.columns and 'BB_Lower' in tech_data.columns:
                    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['BB_Upper'],
                        name='BB Upper', line=dict(color='gray', width=1, dash='dot'),
                        showlegend=True), row=1, col=1)
                    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['BB_Lower'],
                        name='BB Lower', line=dict(color='gray', width=1, dash='dot'),
                        fill='tonexty', fillcolor='rgba(128,128,128,0.05)',
                        showlegend=True), row=1, col=1)

                # MACD
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['MACD'],
                               name='MACD', line=dict(color=_clrs[0], width=1.5)),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['MACD_Signal'],
                               name='Signal', line=dict(color=_clrs[3], width=1.5)),
                    row=2, col=1
                )

                # RSI
                fig.add_trace(
                    go.Scatter(x=tech_data.index, y=tech_data['RSI'],
                               name='RSI', line=dict(color=_clrs[0], width=1.5)),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color=_clrs[3], row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color=_clrs[2], row=3, col=1)

                fig.update_layout(
                    template=_tmpl,
                    height=800,
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color=_fc)),
                    font=dict(family="Inter, system-ui, sans-serif", color=_fc)
                )
                fig.update_xaxes(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc))
                fig.update_yaxes(gridcolor=_gc, tickfont=dict(color=_fc), title_font=dict(color=_fc))
                
                st.plotly_chart(fig, use_container_width=True)

        # ================================================================
        # TAB 6: OPTIONS PRICING (NEW)
        # ================================================================
        with tabs[6]:
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 07</div>
                <div class="section-title">Options Pricing</div>
                <div class="section-subtitle">Options chains, IV surface, Black-Scholes Greeks, and payoff diagrams</div>
            </div>
            """, unsafe_allow_html=True)

            if asset_class not in ("Stocks & ETFs", "Options"):
                st.info("Options Pricing is most relevant for Stocks & ETFs or Options asset class. Switch asset class in the sidebar for full functionality.")

            with st.expander("Understanding Options Pricing"):
                st.markdown("**Black-Scholes Formula** -- Theoretical price of a European call option:")
                st.latex(r"C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)")
                st.latex(r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}")
                st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
                st.markdown("""
                - **S** = Current stock price, **K** = Strike price, **T** = Time to expiration (years)
                - **r** = Risk-free rate, **sigma** = Implied volatility
                - **N(x)** = Cumulative standard normal distribution
                """)
                st.markdown("**The Greeks:**")
                st.latex(r"\Delta = \frac{\partial C}{\partial S} = N(d_1)")
                st.latex(r"\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{N'(d_1)}{S\sigma\sqrt{T}}")
                st.latex(r"\Theta = -\frac{S N'(d_1) \sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)")
                st.latex(r"\mathcal{V} = S N'(d_1) \sqrt{T}")
                st.markdown("""
                - **Delta**: Sensitivity of option price to underlying price change
                - **Gamma**: Rate of change of delta (convexity)
                - **Theta**: Time decay per day
                - **Vega**: Sensitivity to volatility changes
                """)
                st.markdown("**IV Smile/Skew** -- Implied volatility typically varies by strike. OTM puts often have higher IV (volatility skew), reflecting demand for downside protection.")

            opt_ticker = st.selectbox("Select Ticker for Options", data['tickers'], key='opt_ticker')

            try:
                expirations = fetch_options_expirations(opt_ticker)
            except Exception:
                expirations = []

            if not expirations:
                show_error('options_unavailable', inline=True)
            else:
                selected_exp = st.selectbox("Expiration Date", expirations, key='opt_exp')
                opt_view = st.radio("View", ["Calls", "Puts", "Both"], horizontal=True, key='opt_view')

                # Fetch chain
                calls_df, puts_df = fetch_options_chain(opt_ticker, selected_exp)

                if calls_df.empty and puts_df.empty:
                    show_error('options_unavailable', inline=True)
                else:
                    # Row 1: Options Chain Table
                    st.markdown("#### Options Chain")
                    display_cols = ['strike', 'lastPrice', 'bid', 'ask', 'change', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney']

                    if opt_view in ("Calls", "Both") and not calls_df.empty:
                        st.markdown("**Calls**")
                        chain_display = calls_df[[c for c in display_cols if c in calls_df.columns]].copy()
                        chain_display.columns = [c.replace('lastPrice', 'Last').replace('openInterest', 'OI').replace('impliedVolatility', 'IV').replace('inTheMoney', 'ITM').replace('strike', 'Strike') for c in chain_display.columns]
                        render_styled_table(chain_display.head(_tbl_limit), format_dict={'IV': '{:.2%}', 'Last': '${:.2f}', 'bid': '${:.2f}', 'ask': '${:.2f}'})

                    if opt_view in ("Puts", "Both") and not puts_df.empty:
                        st.markdown("**Puts**")
                        chain_display = puts_df[[c for c in display_cols if c in puts_df.columns]].copy()
                        chain_display.columns = [c.replace('lastPrice', 'Last').replace('openInterest', 'OI').replace('impliedVolatility', 'IV').replace('inTheMoney', 'ITM').replace('strike', 'Strike') for c in chain_display.columns]
                        render_styled_table(chain_display.head(_tbl_limit), format_dict={'IV': '{:.2%}', 'Last': '${:.2f}', 'bid': '${:.2f}', 'ask': '${:.2f}'})

                    # Row 2: IV Surface (3D)
                    st.markdown("#### Implied Volatility Surface")
                    try:
                        iv_data = []
                        for exp in expirations[:6]:
                            c_df, p_df = fetch_options_chain(opt_ticker, exp)
                            if not c_df.empty:
                                for _, row in c_df.iterrows():
                                    if row.get('impliedVolatility', 0) > 0:
                                        iv_data.append({
                                            'Strike': row['strike'],
                                            'Expiration': exp,
                                            'IV': row['impliedVolatility']
                                        })
                        if iv_data:
                            iv_df = pd.DataFrame(iv_data)
                            iv_pivot = iv_df.pivot_table(index='Strike', columns='Expiration', values='IV', aggfunc='mean')
                            _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                            _bg = 'rgba(0,0,0,0)' if st.session_state.get('theme', 'light') == 'dark' else '#FFFFFF'

                            fig_iv = go.Figure(data=[go.Surface(
                                z=iv_pivot.values,
                                x=list(range(len(iv_pivot.columns))),
                                y=iv_pivot.index.values,
                                colorscale='Viridis',
                                colorbar=dict(title='IV')
                            )])
                            fig_iv.update_layout(
                                template=_tmpl, height=_chart_h,
                                title='Implied Volatility Surface',
                                scene=dict(
                                    xaxis_title='Expiration',
                                    yaxis_title='Strike',
                                    zaxis_title='IV',
                                    xaxis=dict(tickvals=list(range(len(iv_pivot.columns))), ticktext=[str(c)[:10] for c in iv_pivot.columns]),
                                ),
                                plot_bgcolor=_bg, paper_bgcolor=_bg,
                                font=dict(color=_fc),
                            )
                            st.plotly_chart(fig_iv, use_container_width=True)
                        else:
                            st.info("Not enough IV data to build surface.")
                    except Exception as e:
                        st.warning(f"Could not build IV surface: {e}")

                    # Row 3: Greeks Calculator
                    st.markdown("#### Black-Scholes Greeks Calculator")
                    try:
                        spot_price = float(data['prices'][opt_ticker].iloc[-1])
                    except Exception:
                        spot_price = 100.0

                    greeks_cols = st.columns(4)
                    with greeks_cols[0]:
                        bs_strike = st.number_input("Strike Price", value=round(spot_price, 0), step=1.0, key='bs_strike')
                    with greeks_cols[1]:
                        bs_tte = st.number_input("Days to Expiry", value=30, min_value=1, max_value=365, key='bs_tte')
                    with greeks_cols[2]:
                        bs_vol = st.number_input("Implied Vol (%)", value=30.0, min_value=1.0, max_value=200.0, key='bs_vol')
                    with greeks_cols[3]:
                        bs_type = st.selectbox("Option Type", ["Call", "Put"], key='bs_type')

                    T_years = bs_tte / 365.0
                    sigma = bs_vol / 100.0
                    opt_t = 'call' if bs_type == 'Call' else 'put'
                    bs_price = black_scholes_price(spot_price, bs_strike, T_years, rf_rate, sigma, opt_t)
                    greeks = bs_greeks(spot_price, bs_strike, T_years, rf_rate, sigma, opt_t)

                    gcols = st.columns(6)
                    with gcols[0]:
                        st.metric("BS Price", f"${bs_price:.2f}")
                    with gcols[1]:
                        st.metric("Delta", f"{greeks['Delta']:.4f}")
                    with gcols[2]:
                        st.metric("Gamma", f"{greeks['Gamma']:.4f}")
                    with gcols[3]:
                        st.metric("Theta", f"{greeks['Theta']:.4f}")
                    with gcols[4]:
                        st.metric("Vega", f"{greeks['Vega']:.4f}")
                    with gcols[5]:
                        st.metric("Rho", f"{greeks['Rho']:.4f}")

                    # Row 4: Payoff Diagram
                    st.markdown("#### Options Payoff Diagram")
                    strategy = st.selectbox("Strategy", [
                        "Long Call", "Long Put", "Bull Call Spread",
                        "Bear Put Spread", "Straddle", "Iron Condor"
                    ], key='opt_strategy')

                    S_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 200)

                    if strategy in ("Long Call", "Long Put"):
                        po = options_payoff(strategy, S_range, spot_price, bs_strike, premium1=bs_price)
                    elif strategy == "Bull Call Spread":
                        K2 = bs_strike * 1.1
                        p1 = black_scholes_price(spot_price, bs_strike, T_years, rf_rate, sigma, 'call')
                        p2 = black_scholes_price(spot_price, K2, T_years, rf_rate, sigma, 'call')
                        po = options_payoff(strategy, S_range, spot_price, bs_strike, K2, p1, p2)
                    elif strategy == "Bear Put Spread":
                        K2 = bs_strike * 1.1
                        p1 = black_scholes_price(spot_price, bs_strike, T_years, rf_rate, sigma, 'put')
                        p2 = black_scholes_price(spot_price, K2, T_years, rf_rate, sigma, 'put')
                        po = options_payoff(strategy, S_range, spot_price, bs_strike, K2, p1, p2)
                    elif strategy == "Straddle":
                        p_call = black_scholes_price(spot_price, bs_strike, T_years, rf_rate, sigma, 'call')
                        p_put = black_scholes_price(spot_price, bs_strike, T_years, rf_rate, sigma, 'put')
                        po = options_payoff(strategy, S_range, spot_price, bs_strike, premium1=p_call, premium2=p_put)
                    elif strategy == "Iron Condor":
                        K_low = bs_strike * 0.9
                        K_high = bs_strike * 1.1
                        po = options_payoff(strategy, S_range, spot_price, K_low, K_high, premium1=2.0)

                    _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                    _bg = 'rgba(0,0,0,0)' if st.session_state.get('theme', 'light') == 'dark' else '#FFFFFF'
                    fig_po = go.Figure()
                    fig_po.add_trace(go.Scatter(x=S_range, y=po, mode='lines', name='P&L',
                                                line=dict(color=_clrs[0], width=2)))
                    fig_po.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
                    fig_po.add_vline(x=spot_price, line_dash='dot', line_color=_clrs[3],
                                     annotation_text=f"Spot: ${spot_price:.2f}")
                    fig_po.update_layout(
                        template=_tmpl, height=_chart_h,
                        title=f'{strategy} Payoff at Expiration',
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc),
                        xaxis=dict(gridcolor=_gc, title='Stock Price at Expiration'),
                        yaxis=dict(gridcolor=_gc, title='Profit / Loss ($)'),
                    )
                    st.plotly_chart(fig_po, use_container_width=True)

        # ================================================================
        # TAB 7: MACRO DASHBOARD
        # ================================================================
        with tabs[7]:
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 08</div>
                <div class="section-title">Macroeconomic Dashboard</div>
                <div class="section-subtitle">Live market data, yield curves, and WACC scenario analysis</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding Macroeconomic Indicators"):
                st.markdown("**CAPM (Capital Asset Pricing Model)** -- Expected return for an asset based on systematic risk:")
                st.latex(r"K_e = R_f + \beta (R_m - R_f)")
                st.markdown("""
                - Rf = Risk-free rate (Treasury yield)
                - Beta = Sensitivity to market movements
                - Rm - Rf = Equity risk premium
                """)
                st.markdown("**WACC (Weighted Average Cost of Capital)** -- Blended cost of financing:")
                st.latex(r"WACC = \frac{E}{V} K_e + \frac{D}{V} K_d (1 - T_c)")
                st.markdown("""
                - E/V = Equity weight, D/V = Debt weight
                - Ke = Cost of equity (from CAPM), Kd = Cost of debt
                - Tc = Corporate tax rate
                """)
                st.markdown("**Gordon Growth Model** -- Intrinsic value from perpetual dividend growth:")
                st.latex(r"P = \frac{FCF \times (1 + g)}{WACC - g}")
                st.markdown("""
                - FCF = Free cash flow, g = perpetual growth rate
                - Requires WACC > g for convergence
                """)
                st.markdown("**Yield Curve** -- Plots interest rates across maturities. An inverted curve (short > long) historically signals recessions.")
                st.markdown("**CPI & Monetary Policy** -- Rising CPI leads the Fed to raise rates, tightening financial conditions and affecting equity valuations.")

            # Fetch macro data
            try:
                macro_data = fetch_all_macro_data()
                macro_ok = any(not v.empty for v in macro_data.values())
            except Exception:
                macro_data = {}
                macro_ok = False

            if not macro_ok:
                st.warning("Could not fetch macro data. Check your internet connection.")
            else:
                # --- Row 1: KPI Cards ---
                def _latest_val(key):
                    df = macro_data.get(key, pd.DataFrame())
                    if df.empty:
                        return np.nan, np.nan
                    df = df.dropna()
                    if len(df) == 0:
                        return np.nan, np.nan
                    current = float(df.iloc[-1].values[0])
                    # Delta: compare to ~1 year ago
                    one_yr = df.index[-1] - pd.DateOffset(years=1)
                    mask = df.index <= one_yr
                    if mask.any():
                        prev = float(df.loc[mask].iloc[-1].values[0])
                        delta = current - prev
                    else:
                        delta = np.nan
                    return current, delta

                irx_val, irx_delta = _latest_val('IRX')      # 13-week T-Bill ~ Fed Funds proxy
                t10_val, t10_delta = _latest_val('TNX')       # 10Y Treasury
                t5_val, t5_delta = _latest_val('FVX')         # 5Y Treasury
                t30_val, t30_delta = _latest_val('TYX')       # 30Y Treasury
                vix_val, vix_delta = _latest_val('VIX')       # VIX
                sp_val, sp_delta_abs = _latest_val('GSPC')    # S&P 500
                # S&P 500 YoY % change
                sp_df = macro_data.get('GSPC', pd.DataFrame()).dropna()
                if not sp_df.empty and len(sp_df) > 200:
                    one_yr = sp_df.index[-1] - pd.DateOffset(years=1)
                    mask = sp_df.index <= one_yr
                    if mask.any():
                        sp_prev = float(sp_df.loc[mask].iloc[-1].values[0])
                        sp_pct = ((sp_val / sp_prev) - 1) * 100
                    else:
                        sp_pct = np.nan
                else:
                    sp_pct = np.nan

                # Yield curve spread (10Y - IRX)
                spread = (t10_val - irx_val) if not (np.isnan(t10_val) or np.isnan(irx_val)) else np.nan

                kpi_cols = st.columns(6)
                kpi_data = [
                    ("3M T-Bill", irx_val, irx_delta, "%", ""),
                    ("10Y Treasury", t10_val, t10_delta, "%", ""),
                    ("30Y Treasury", t30_val, t30_delta, "%", ""),
                    ("10Y-3M Spread", spread, None, "%", ""),
                    ("VIX", vix_val, vix_delta, "", "inverse"),
                    ("S&P 500", sp_val, f"{sp_pct:+.1f}% YoY" if not np.isnan(sp_pct) else None, "", ""),
                ]
                for col, (label, val, delta, suffix, delta_inv) in zip(kpi_cols, kpi_data):
                    with col:
                        if isinstance(val, float) and not np.isnan(val):
                            val_str = f"{val:,.2f}{suffix}"
                        else:
                            val_str = "N/A"
                        delta_str = None
                        if isinstance(delta, str):
                            delta_str = delta
                        elif isinstance(delta, float) and not np.isnan(delta):
                            delta_str = f"{delta:+.2f}"
                        st.metric(label, val_str, delta_str,
                                  delta_color="inverse" if delta_inv == "inverse" else "normal")

                # --- Row 2: Charts ---
                _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                is_dark = st.session_state.get('theme', 'light') == 'dark'
                _bg = 'rgba(0,0,0,0)' if is_dark else '#FFFFFF'

                chart_cols = st.columns(3)

                # Chart 1: Treasury Rates History (3M, 10Y, 30Y)
                with chart_cols[0]:
                    fig = go.Figure()
                    for key, name, color_idx in [('IRX', '3M T-Bill', 0), ('TNX', '10Y Treasury', 1), ('TYX', '30Y Treasury', 2)]:
                        _df = macro_data.get(key, pd.DataFrame()).dropna()
                        if not _df.empty:
                            fig.add_trace(go.Scatter(x=_df.index, y=_df[key],
                                                     name=name, line=dict(color=_clrs[color_idx], width=2)))
                    fig.update_layout(
                        template=_tmpl, title='Treasury Yields', height=350,
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                        xaxis=dict(gridcolor=_gc), yaxis=dict(gridcolor=_gc, title='Yield (%)'),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Chart 2: Yield Curve
                with chart_cols[1]:
                    try:
                        current_yc, year_ago_yc = fetch_yield_curve_data()
                    except Exception:
                        current_yc, year_ago_yc = {}, {}
                    fig = go.Figure()
                    if current_yc:
                        mats = sorted(current_yc.keys())
                        mat_labels = []
                        for m in mats:
                            if m < 1:
                                mat_labels.append(f"{int(m*12)}M")
                            else:
                                mat_labels.append(f"{int(m)}Y")
                        fig.add_trace(go.Scatter(
                            x=mat_labels,
                            y=[current_yc[m] for m in mats],
                            name='Current', mode='lines+markers',
                            line=dict(color=_clrs[0], width=2)))
                    if year_ago_yc:
                        mats_ago = sorted(year_ago_yc.keys())
                        mat_labels_ago = []
                        for m in mats_ago:
                            if m < 1:
                                mat_labels_ago.append(f"{int(m*12)}M")
                            else:
                                mat_labels_ago.append(f"{int(m)}Y")
                        fig.add_trace(go.Scatter(
                            x=mat_labels_ago,
                            y=[year_ago_yc[m] for m in mats_ago],
                            name='1 Year Ago', mode='lines+markers',
                            line=dict(color=_clrs[3], width=2, dash='dash')))
                    fig.update_layout(
                        template=_tmpl, title='Yield Curve', height=350,
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                        xaxis=dict(gridcolor=_gc, title='Maturity'),
                        yaxis=dict(gridcolor=_gc, title='Yield (%)'),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Chart 3: VIX + S&P 500 dual-axis
                with chart_cols[2]:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    vix_df = macro_data.get('VIX', pd.DataFrame()).dropna()
                    if not vix_df.empty:
                        fig.add_trace(go.Scatter(
                            x=vix_df.index, y=vix_df['VIX'],
                            name='VIX', fill='tozeroy',
                            line=dict(color=_clrs[2], width=1.5)),
                            secondary_y=False)
                    sp_chart_df = macro_data.get('GSPC', pd.DataFrame()).dropna()
                    if not sp_chart_df.empty:
                        fig.add_trace(go.Scatter(
                            x=sp_chart_df.index, y=sp_chart_df['GSPC'],
                            name='S&P 500',
                            line=dict(color=_clrs[4], width=2)),
                            secondary_y=True)
                    fig.update_layout(
                        template=_tmpl, title='VIX & S&P 500', height=350,
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                    )
                    fig.update_xaxes(gridcolor=_gc)
                    fig.update_yaxes(title_text='VIX', gridcolor=_gc, secondary_y=False)
                    fig.update_yaxes(title_text='S&P 500', gridcolor=_gc, secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)

                # --- Row 3: Beta Scatter ---
                st.markdown("#### Correlation: S&P 500 vs Stock Returns")
                try:
                    sp500 = yf.download('^GSPC', start=data['prices'].index[0],
                                        end=data['prices'].index[-1], progress=False)
                    if isinstance(sp500.columns, pd.MultiIndex):
                        sp500_close = sp500[('Close', '^GSPC')] if (('Close', '^GSPC') in sp500.columns) else sp500['Close']
                    else:
                        sp500_close = sp500['Close']
                    sp500_ret = sp500_close.pct_change().dropna()

                    beta_ticker = st.selectbox("Select ticker for beta scatter", data['tickers'], key='macro_beta_ticker')
                    stock_ret = data['returns'][beta_ticker].reindex(sp500_ret.index).dropna()
                    sp500_aligned = sp500_ret.reindex(stock_ret.index).dropna()
                    stock_aligned = stock_ret.reindex(sp500_aligned.index).dropna()

                    if len(stock_aligned) > 20:
                        slope, intercept, r_value, _, _ = stats.linregress(sp500_aligned, stock_aligned)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=sp500_aligned, y=stock_aligned,
                            mode='markers', name='Daily Returns',
                            marker=dict(size=4, color=_clrs[0], opacity=0.5)))
                        x_line = np.linspace(sp500_aligned.min(), sp500_aligned.max(), 50)
                        fig.add_trace(go.Scatter(
                            x=x_line, y=intercept + slope * x_line,
                            mode='lines', name=f'Beta={slope:.2f}, R2={r_value**2:.2f}',
                            line=dict(color=_clrs[3], width=2)))
                        fig.update_layout(
                            template=_tmpl, title=f'{beta_ticker} vs S&P 500 (Beta = {slope:.2f})',
                            height=_chart_h, plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                            xaxis=dict(gridcolor=_gc, title='S&P 500 Return'),
                            yaxis=dict(gridcolor=_gc, title=f'{beta_ticker} Return'),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data to compute beta scatter.")
                except Exception as e:
                    st.warning(f"Could not load S&P 500 data for beta analysis: {e}")

                # --- Row 4: WACC Calculator + DCF Sensitivity ---
                st.markdown("#### WACC Calculator & DCF Sensitivity")
                wacc_cols = st.columns([1, 2])

                with wacc_cols[0]:
                    rf_slider = st.slider("Risk-Free Rate (%)", 2.0, 6.0,
                                          min(max(t10_val if not np.isnan(t10_val) else 4.25, 2.0), 6.0), 0.05,
                                          key='macro_rf')
                    erp_slider = st.slider("Equity Risk Premium (%)", 3.0, 8.0, 5.5, 0.1, key='macro_erp')
                    beta_slider = st.slider("Beta", 0.5, 2.0, 1.0, 0.05, key='macro_beta')
                    debt_ratio = st.slider("Debt / Total Capital (%)", 0, 60, 30, 5, key='macro_debt')
                    cost_debt = st.slider("Cost of Debt (%)", 2.0, 8.0, 5.0, 0.1, key='macro_kd')
                    tax_rate = st.slider("Tax Rate (%)", 10, 35, 21, 1, key='macro_tax')
                    growth_slider = st.slider("Perpetual Growth Rate (%)", 0.0, 5.0, 2.5, 0.1, key='macro_g')

                    ke = rf_slider + beta_slider * erp_slider
                    equity_w = (100 - debt_ratio) / 100
                    debt_w = debt_ratio / 100
                    wacc_val = equity_w * ke + debt_w * cost_debt * (1 - tax_rate / 100)

                    st.metric("Cost of Equity (Ke)", f"{ke:.2f}%")
                    st.metric("WACC", f"{wacc_val:.2f}%")

                with wacc_cols[1]:
                    # DCF Sensitivity Table
                    wacc_range = np.arange(max(wacc_val - 2, 3), wacc_val + 2.5, 0.5)
                    growth_range = np.arange(max(growth_slider - 1.5, 0), min(growth_slider + 2.0, 5.1), 0.5)

                    # Use FCF=1 as a normalized base
                    base_fcf = 1.0
                    sensitivity = pd.DataFrame(index=[f"{w:.1f}%" for w in wacc_range],
                                               columns=[f"{g:.1f}%" for g in growth_range])
                    for wi, w in enumerate(wacc_range):
                        for gi, g in enumerate(growth_range):
                            if w > g:
                                implied = base_fcf * (1 + g / 100) / (w / 100 - g / 100)
                                sensitivity.iloc[wi, gi] = f"{implied:.1f}x"
                            else:
                                sensitivity.iloc[wi, gi] = "N/A"

                    sensitivity.index.name = "WACC \\ Growth"
                    st.markdown("**DCF Sensitivity (Implied Value / FCF multiple)**")
                    render_styled_table(sensitivity.reset_index())

        # ================================================================
        # TAB 8: RISK & GEOPOLITICS (NEW)
        # ================================================================
        with tabs[8]:
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 09</div>
                <div class="section-title">Risk & Geopolitics Dashboard</div>
                <div class="section-subtitle">Cross-asset risk signals, VIX regimes, and composite risk scoring</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding Risk Indicators"):
                st.markdown("**VIX (CBOE Volatility Index)** -- Measures expected 30-day volatility of the S&P 500:")
                st.latex(r"VIX = 100 \times \sqrt{\frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i) - \frac{1}{T}\left(\frac{F}{K_0}-1\right)^2}")
                st.markdown("""
                - VIX < 15: **Low volatility** (complacency)
                - VIX 15-25: **Normal** market conditions
                - VIX 25-35: **Elevated** fear
                - VIX > 35: **Extreme** fear (crisis)
                """)
                st.markdown("**Yield Curve Spread** -- Difference between long-term and short-term Treasury yields:")
                st.latex(r"\text{Spread} = Y_{10Y} - Y_{3M}")
                st.markdown("An inverted yield curve (negative spread) has preceded every US recession since the 1950s.")
                st.markdown("**Gold as Safe Haven** -- Gold tends to rally during risk-off periods. Simultaneous rises in gold and the US dollar signal extreme stress.")
                st.markdown("**Composite Risk Score** -- Weighted average of normalized VIX, yield curve, safe haven demand, and volatility regime signals (0-100 scale).")

            with st.spinner("Fetching risk data..."):
                try:
                    risk_data = fetch_risk_data()
                    risk_ok = any(len(v) > 0 for v in risk_data.values())
                except Exception:
                    risk_data = {}
                    risk_ok = False

            if not risk_ok:
                st.warning("Could not fetch risk data. Check your internet connection.")
            else:
                _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                _bg = 'rgba(0,0,0,0)' if st.session_state.get('theme', 'light') == 'dark' else '#FFFFFF'

                # Row 1: Risk KPI Cards
                vix_s = risk_data.get('VIX', pd.Series(dtype=float))
                dxy_s = risk_data.get('DXY', pd.Series(dtype=float))
                gold_s = risk_data.get('Gold', pd.Series(dtype=float))
                oil_s = risk_data.get('Oil', pd.Series(dtype=float))
                tnx_s = risk_data.get('TNX', pd.Series(dtype=float))
                irx_s = risk_data.get('IRX', pd.Series(dtype=float))

                vix_val = float(vix_s.iloc[-1]) if len(vix_s) > 0 else np.nan
                dxy_val = float(dxy_s.iloc[-1]) if len(dxy_s) > 0 else np.nan
                gold_val = float(gold_s.iloc[-1]) if len(gold_s) > 0 else np.nan
                oil_val = float(oil_s.iloc[-1]) if len(oil_s) > 0 else np.nan
                tnx_val = float(tnx_s.iloc[-1]) if len(tnx_s) > 0 else np.nan
                irx_val_r = float(irx_s.iloc[-1]) if len(irx_s) > 0 else np.nan
                spread_val = (tnx_val - irx_val_r) if not (np.isnan(tnx_val) or np.isnan(irx_val_r)) else np.nan

                # VIX interpretation
                if not np.isnan(vix_val):
                    if vix_val < 15:
                        vix_label = "Low"
                    elif vix_val < 25:
                        vix_label = "Normal"
                    elif vix_val < 35:
                        vix_label = "Elevated"
                    else:
                        vix_label = "Extreme"
                else:
                    vix_label = "N/A"

                composite_score = calculate_composite_risk_score(risk_data)

                kpi_cols = st.columns(6)
                with kpi_cols[0]:
                    st.metric("VIX", f"{vix_val:.1f}" if not np.isnan(vix_val) else "N/A", vix_label)
                with kpi_cols[1]:
                    st.metric("Dollar Index", f"{dxy_val:.2f}" if not np.isnan(dxy_val) else "N/A")
                with kpi_cols[2]:
                    gold_chg = ""
                    if len(gold_s) > 1:
                        gold_chg = f"{(float(gold_s.iloc[-1]) / float(gold_s.iloc[-2]) - 1) * 100:+.2f}%"
                    st.metric("Gold", f"${gold_val:,.0f}" if not np.isnan(gold_val) else "N/A", gold_chg)
                with kpi_cols[3]:
                    oil_chg = ""
                    if len(oil_s) > 1:
                        oil_chg = f"{(float(oil_s.iloc[-1]) / float(oil_s.iloc[-2]) - 1) * 100:+.2f}%"
                    st.metric("Oil (WTI)", f"${oil_val:.2f}" if not np.isnan(oil_val) else "N/A", oil_chg)
                with kpi_cols[4]:
                    st.metric("10Y-3M Spread", f"{spread_val:.2f}%" if not np.isnan(spread_val) else "N/A",
                              "Inverted" if (not np.isnan(spread_val) and spread_val < 0) else "Normal")
                with kpi_cols[5]:
                    score_label = "Green" if composite_score < 30 else ("Yellow" if composite_score < 60 else ("Orange" if composite_score < 80 else "Red"))
                    st.metric("Risk Score", f"{composite_score:.0f}/100", score_label)

                # Row 2: Risk Charts (3 columns)
                chart_cols = st.columns(3)

                with chart_cols[0]:
                    st.markdown("#### VIX History with Regime Bands")
                    if len(vix_s) > 0:
                        fig_vix = go.Figure()
                        fig_vix.add_trace(go.Scatter(x=vix_s.index, y=vix_s.values, name='VIX',
                                                     line=dict(color=_clrs[0], width=2)))
                        fig_vix.add_hrect(y0=0, y1=15, fillcolor='green', opacity=0.1, line_width=0)
                        fig_vix.add_hrect(y0=15, y1=25, fillcolor='yellow', opacity=0.1, line_width=0)
                        fig_vix.add_hrect(y0=25, y1=35, fillcolor='orange', opacity=0.1, line_width=0)
                        fig_vix.add_hrect(y0=35, y1=80, fillcolor='red', opacity=0.1, line_width=0)
                        fig_vix.update_layout(
                            template=_tmpl, height=350,
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc), yaxis=dict(gridcolor=_gc, title='VIX'),
                        )
                        st.plotly_chart(fig_vix, use_container_width=True)

                with chart_cols[1]:
                    st.markdown("#### Safe Haven: Gold vs Dollar")
                    if len(gold_s) > 0 and len(dxy_s) > 0:
                        fig_sh = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_sh.add_trace(go.Scatter(x=gold_s.index, y=gold_s.values, name='Gold',
                                                    line=dict(color=_clrs[1], width=2)), secondary_y=False)
                        fig_sh.add_trace(go.Scatter(x=dxy_s.index, y=dxy_s.values, name='DXY',
                                                    line=dict(color=_clrs[2], width=2)), secondary_y=True)
                        fig_sh.update_layout(
                            template=_tmpl, height=350,
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            legend=dict(font=dict(color=_fc)),
                        )
                        fig_sh.update_xaxes(gridcolor=_gc)
                        fig_sh.update_yaxes(gridcolor=_gc, title_text="Gold ($)", secondary_y=False)
                        fig_sh.update_yaxes(gridcolor=_gc, title_text="DXY", secondary_y=True)
                        st.plotly_chart(fig_sh, use_container_width=True)

                with chart_cols[2]:
                    st.markdown("#### Oil + VIX Correlation")
                    if len(oil_s) > 0 and len(vix_s) > 0:
                        fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_ov.add_trace(go.Scatter(x=oil_s.index, y=oil_s.values, name='Oil',
                                                    line=dict(color=_clrs[3], width=2)), secondary_y=False)
                        fig_ov.add_trace(go.Scatter(x=vix_s.index, y=vix_s.values, name='VIX',
                                                    line=dict(color=_clrs[0], width=2)), secondary_y=True)
                        fig_ov.update_layout(
                            template=_tmpl, height=350,
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            legend=dict(font=dict(color=_fc)),
                        )
                        fig_ov.update_xaxes(gridcolor=_gc)
                        fig_ov.update_yaxes(gridcolor=_gc, title_text="Oil ($)", secondary_y=False)
                        fig_ov.update_yaxes(gridcolor=_gc, title_text="VIX", secondary_y=True)
                        st.plotly_chart(fig_ov, use_container_width=True)

                # Row 3: Composite Risk Gauge
                st.markdown("#### Composite Risk Score")
                gauge_cols = st.columns([2, 1])
                with gauge_cols[0]:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=composite_score,
                        title={'text': "Geopolitical Risk Score", 'font': {'color': _fc}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': _fc},
                            'bar': {'color': _clrs[0]},
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(0,208,132,0.3)'},
                                {'range': [30, 60], 'color': 'rgba(255,215,0,0.3)'},
                                {'range': [60, 80], 'color': 'rgba(255,154,0,0.3)'},
                                {'range': [80, 100], 'color': 'rgba(255,77,109,0.3)'},
                            ],
                        },
                        number={'font': {'color': _fc}},
                    ))
                    fig_gauge.update_layout(
                        template=_tmpl, height=300,
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with gauge_cols[1]:
                    st.markdown("**Score Breakdown:**")
                    st.markdown(f"- VIX Component: {min(max((vix_val - 10) / 40 * 25, 0), 25):.0f}/25" if not np.isnan(vix_val) else "- VIX: N/A")
                    st.markdown(f"- Yield Curve: {'Inverted' if not np.isnan(spread_val) and spread_val < 0 else 'Normal'}")
                    if len(gold_s) > 30:
                        g30 = (float(gold_s.iloc[-1]) / float(gold_s.iloc[-30]) - 1) * 100
                        st.markdown(f"- Gold 30d Return: {g30:.1f}%")
                    st.markdown(f"- **Overall: {score_label}**")

                # Row 4: Cross-Asset Correlation Heatmap
                st.markdown("#### Cross-Asset Correlation Heatmap")
                try:
                    corr_assets = {}
                    for name in ['VIX', 'Gold', 'Oil', 'DXY', 'TNX', 'SPX']:
                        s = risk_data.get(name, pd.Series(dtype=float))
                        if len(s) > 0:
                            corr_assets[name] = s.pct_change().dropna()
                    # Add user tickers
                    for t in data['tickers'][:3]:
                        if t in data['prices'].columns:
                            corr_assets[t] = data['prices'][t].pct_change().dropna()

                    if len(corr_assets) >= 3:
                        corr_df = pd.DataFrame(corr_assets)
                        corr_df = corr_df.dropna()
                        corr_matrix = corr_df.corr()
                        fig_hm = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns.tolist(),
                            y=corr_matrix.index.tolist(),
                            colorscale='RdBu_r',
                            zmid=0,
                            text=np.round(corr_matrix.values, 2),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                        ))
                        fig_hm.update_layout(
                            template=_tmpl, height=_chart_h,
                            title='Return Correlations (Daily)',
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc), yaxis=dict(gridcolor=_gc),
                        )
                        st.plotly_chart(fig_hm, use_container_width=True)
                    else:
                        st.info("Not enough data for correlation heatmap.")
                except Exception as e:
                    st.warning(f"Could not build correlation heatmap: {e}")

        # ================================================================
        # TAB 9: ML PREDICTIONS
        # ================================================================
        with tabs[9]:
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 10</div>
                <div class="section-title">ML Price Predictions</div>
                <div class="section-subtitle">Machine learning models trained on historical price data</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding ML Models"):
                st.markdown("**Linear Regression** -- Fits a linear relationship between features and target:")
                st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \varepsilon")
                st.markdown("""
                - Coefficients show directional impact of each feature
                - Assumes linear relationship (may miss non-linear patterns)
                - Fast, interpretable baseline model
                """)
                st.markdown("**Random Forest** -- Ensemble of decision trees using bagging:")
                st.latex(r"\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)")
                st.markdown("""
                - Each tree trained on a bootstrapped sample with random feature subsets
                - Feature importance measured by Gini impurity reduction
                - Robust to outliers, handles non-linear relationships
                """)
                st.markdown("**Gradient Boosting** -- Sequential ensemble with additive residual fitting:")
                st.latex(r"F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)")
                st.markdown("""
                - Each new tree fits the residual errors of the previous ensemble
                - Learning rate (eta) controls contribution of each tree
                - Often achieves best accuracy, but more prone to overfitting
                """)
                st.markdown("**Cross-Validation & Overfitting** -- Models are evaluated on held-out test data (last 20% chronologically) to measure generalization. R-squared near 1.0 on training but low on test indicates overfitting.")
                st.markdown("**Feature Importance** -- Shows which input variables have the greatest influence on predictions. Helps interpret model behavior and identify key market drivers.")

            ml_ticker = st.selectbox("Select Asset for ML", data['tickers'], key='ml_ticker')

            # Fetch extended data for ML — SMA200 needs 200+ days
            _ml_years = data.get('ml_training_years', 3)
            with st.spinner("Fetching extended history & training ML models..."):
                try:
                    ml_end = datetime.now()
                    ml_start = ml_end - timedelta(days=_ml_years * 365)
                    ml_raw = yf.download(ml_ticker, start=ml_start, end=ml_end, progress=False)
                    if isinstance(ml_raw.columns, pd.MultiIndex):
                        prices_s = ml_raw[('Close', ml_ticker)] if ('Close', ml_ticker) in ml_raw.columns else ml_raw['Close'].iloc[:, 0]
                        vol_s = ml_raw[('Volume', ml_ticker)] if ('Volume', ml_ticker) in ml_raw.columns else ml_raw['Volume'].iloc[:, 0]
                    else:
                        prices_s = ml_raw['Close']
                        vol_s = ml_raw['Volume']
                    ml_results = train_ml_models(prices_s, vol_s)
                except Exception:
                    # Fallback to user-range data
                    prices_s = data['prices'][ml_ticker]
                    vol_s = data['volumes'].get(ml_ticker) if 'volumes' in data else None
                    ml_results = train_ml_models(prices_s, vol_s)

            if ml_results is None:
                st.warning("Not enough historical data to train ML models (need 60+ data points after feature engineering).")
            else:
                # --- Row 1: Model Overview Cards ---
                model_cols = st.columns(3)
                for col, (name, info) in zip(model_cols, ml_results['models_info'].items()):
                    with col:
                        st.metric(f"{name} R2", f"{info['r2']:.4f}")
                        st.metric(f"{name} RMSE", f"{info['rmse']:.4f}")

                _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                is_dark = st.session_state.get('theme', 'light') == 'dark'
                _bg = 'rgba(0,0,0,0)' if is_dark else '#FFFFFF'

                # --- Row 2: Interactive Prediction Simulator ---
                st.markdown("#### Prediction Simulator")
                sim_cols = st.columns([1, 2])

                with sim_cols[0]:
                    st.markdown("**Adjust scenario parameters:**")
                    sim_ret1d = st.slider("1-Day Return (%)", -5.0, 5.0, 0.0, 0.1, key='ml_ret1d') / 100
                    sim_ret5d = st.slider("5-Day Return (%)", -10.0, 10.0, 0.0, 0.5, key='ml_ret5d') / 100
                    sim_ret21d = st.slider("21-Day Return (%)", -20.0, 20.0, 0.0, 0.5, key='ml_ret21d') / 100
                    sim_vol = st.slider("21-Day Volatility (%)", 0.5, 5.0, 1.5, 0.1, key='ml_vol') / 100
                    sim_sma20 = st.slider("Price / SMA20 Ratio", 0.85, 1.15, 1.0, 0.01, key='ml_sma20')
                    sim_sma50 = st.slider("Price / SMA50 Ratio", 0.80, 1.20, 1.0, 0.01, key='ml_sma50')
                    sim_sma200 = st.slider("Price / SMA200 Ratio", 0.70, 1.30, 1.0, 0.01, key='ml_sma200')
                    sim_volr = st.slider("Volume Ratio", 0.5, 3.0, 1.0, 0.1, key='ml_volr')
                    sim_rsi = st.slider("RSI", 20.0, 80.0, 50.0, 1.0, key='ml_rsi')

                with sim_cols[1]:
                    feature_names = ml_results['feature_names']
                    sim_values = [sim_ret1d, sim_ret5d, sim_ret21d, sim_vol,
                                  sim_sma20, sim_sma50, sim_sma200, sim_volr, sim_rsi]
                    # Align to actual feature list
                    sim_input = []
                    name_to_val = dict(zip(
                        ['returns_1d', 'returns_5d', 'returns_21d', 'vol_21d',
                         'sma_20_ratio', 'sma_50_ratio', 'sma_200_ratio',
                         'volume_ratio', 'rsi'],
                        sim_values
                    ))
                    for fn in feature_names:
                        sim_input.append(name_to_val.get(fn, 0.0))

                    sim_array = ml_results['scaler'].transform([sim_input])
                    last_price = ml_results['last_price']

                    st.markdown("**Model Predictions (21-day forward return):**")
                    preds_dict = {}
                    for name, info in ml_results['models_info'].items():
                        pred = info['model'].predict(sim_array)[0]
                        preds_dict[name] = pred
                        pred_price = last_price * (1 + pred)
                        conf_lo = last_price * (1 + pred - 1.96 * info['rmse'])
                        conf_hi = last_price * (1 + pred + 1.96 * info['rmse'])
                        st.metric(f"{name}", f"{pred:+.2%} (${pred_price:.2f})",
                                  delta=f"95% CI: ${conf_lo:.2f} - ${conf_hi:.2f}")

                    # Ensemble
                    ensemble = 0.25 * preds_dict['Linear Regression'] + \
                               0.50 * preds_dict['Random Forest'] + \
                               0.25 * preds_dict['Gradient Boosting']
                    ens_price = last_price * (1 + ensemble)
                    st.metric("Ensemble Consensus", f"{ensemble:+.2%} (${ens_price:.2f})")

                    # Visual confidence bars
                    fig_conf = go.Figure()
                    for i, (name, pred) in enumerate(preds_dict.items()):
                        rmse = ml_results['models_info'][name]['rmse']
                        fig_conf.add_trace(go.Bar(
                            x=[pred * 100], y=[name], orientation='h',
                            name=name, marker_color=_clrs[i % len(_clrs)],
                            error_x=dict(type='data', array=[rmse * 1.96], visible=True)
                        ))
                    fig_conf.add_trace(go.Bar(
                        x=[ensemble * 100], y=['Ensemble'], orientation='h',
                        name='Ensemble', marker_color=_clrs[4 % len(_clrs)]
                    ))
                    fig_conf.update_layout(
                        template=_tmpl, title='Predicted 21-Day Return (%)', height=250,
                        showlegend=False,
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc),
                        xaxis=dict(gridcolor=_gc, title='Return (%)'),
                        yaxis=dict(gridcolor=_gc),
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

                # --- Row 3: Feature Importance + Sensitivity ---
                row3_cols = st.columns(2)

                with row3_cols[0]:
                    st.markdown("#### Feature Importance")
                    fig_imp = go.Figure()
                    for i, (name, imp) in enumerate(ml_results['feature_importance'].items()):
                        sorted_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)
                        fig_imp.add_trace(go.Bar(
                            y=[f[0] for f in sorted_feats],
                            x=[f[1] for f in sorted_feats],
                            name=name, orientation='h',
                            marker_color=_clrs[i % len(_clrs)]
                        ))
                    fig_imp.update_layout(
                        template=_tmpl, height=_chart_h, barmode='group',
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                        xaxis=dict(gridcolor=_gc, title='Importance'),
                        yaxis=dict(gridcolor=_gc),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                with row3_cols[1]:
                    st.markdown("#### Sensitivity Analysis")
                    sens_feature = st.selectbox("Vary feature:", feature_names, key='ml_sens_feat')
                    feat_idx = feature_names.index(sens_feature)
                    base_input = list(sim_input)

                    # Determine range
                    if 'ratio' in sens_feature:
                        sweep = np.linspace(0.8, 1.2, 30)
                    elif 'rsi' in sens_feature:
                        sweep = np.linspace(20, 80, 30)
                    elif 'vol' in sens_feature and 'ratio' not in sens_feature:
                        sweep = np.linspace(0.005, 0.05, 30)
                    else:
                        sweep = np.linspace(-0.1, 0.1, 30)

                    fig_sens = go.Figure()
                    for mi, (name, info) in enumerate(ml_results['models_info'].items()):
                        preds_sweep = []
                        for val in sweep:
                            tmp = list(base_input)
                            tmp[feat_idx] = val
                            tmp_s = ml_results['scaler'].transform([tmp])
                            preds_sweep.append(info['model'].predict(tmp_s)[0] * 100)
                        fig_sens.add_trace(go.Scatter(
                            x=sweep, y=preds_sweep, name=name,
                            line=dict(color=_clrs[mi % len(_clrs)], width=2)))
                    fig_sens.update_layout(
                        template=_tmpl, height=_chart_h,
                        title=f'Predicted Return vs {sens_feature}',
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                        xaxis=dict(gridcolor=_gc, title=sens_feature),
                        yaxis=dict(gridcolor=_gc, title='Predicted Return (%)'),
                    )
                    st.plotly_chart(fig_sens, use_container_width=True)

                # --- Row 4: Actual vs Predicted + Residuals ---
                row4_cols = st.columns(2)

                with row4_cols[0]:
                    st.markdown("#### Actual vs Predicted")
                    fig_avp = go.Figure()
                    actuals = ml_results['actuals']
                    for mi, (name, preds) in enumerate(ml_results['predictions'].items()):
                        fig_avp.add_trace(go.Scatter(
                            x=actuals, y=preds, mode='markers',
                            name=name, marker=dict(size=4, color=_clrs[mi % len(_clrs)], opacity=0.6)))
                    # Perfect prediction line
                    min_val = min(actuals.min(), min(p.min() for p in ml_results['predictions'].values()))
                    max_val = max(actuals.max(), max(p.max() for p in ml_results['predictions'].values()))
                    fig_avp.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='Perfect', line=dict(color='gray', dash='dash', width=1),
                        showlegend=False))
                    fig_avp.update_layout(
                        template=_tmpl, height=_chart_h,
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                        xaxis=dict(gridcolor=_gc, title='Actual Return'),
                        yaxis=dict(gridcolor=_gc, title='Predicted Return'),
                    )
                    st.plotly_chart(fig_avp, use_container_width=True)

                with row4_cols[1]:
                    st.markdown("#### Residual Distribution")
                    fig_res = go.Figure()
                    for mi, (name, preds) in enumerate(ml_results['predictions'].items()):
                        residuals = actuals - preds
                        fig_res.add_trace(go.Histogram(
                            x=residuals, name=name, opacity=0.6,
                            marker_color=_clrs[mi % len(_clrs)],
                            nbinsx=30))
                    fig_res.update_layout(
                        template=_tmpl, height=_chart_h, barmode='overlay',
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc), legend=dict(font=dict(color=_fc)),
                        xaxis=dict(gridcolor=_gc, title='Residual (Actual - Predicted)'),
                        yaxis=dict(gridcolor=_gc, title='Count'),
                    )
                    st.plotly_chart(fig_res, use_container_width=True)

        # ================================================================
        # TAB 10: ML CLUSTERING (NEW)
        # ================================================================
        with tabs[10]:
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 11</div>
                <div class="section-title">ML Clustering & Regime Detection</div>
                <div class="section-subtitle">K-Means clustering, market regime detection, anomaly detection, and PCA factor analysis</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding ML Clustering"):
                st.markdown("**K-Means Clustering** -- Partitions assets into k groups by minimizing within-cluster variance:")
                st.latex(r"\min_{S} \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2")
                st.markdown("""
                - Each cluster has a centroid (mean of its members)
                - Optimal k can be found using the elbow method (inertia vs k)
                """)
                st.markdown("**Gaussian Mixture Models** -- Probabilistic clustering that models data as a mixture of Gaussians:")
                st.latex(r"p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)")
                st.markdown("Used for **market regime detection**: identifies low-volatility (bull), high-volatility (bear), and transition regimes.")
                st.markdown("**PCA (Principal Component Analysis)** -- Reduces dimensionality by projecting data onto directions of maximum variance:")
                st.latex(r"Z = X W, \quad W = [w_1, w_2, \ldots, w_p]")
                st.markdown("Factor loadings show how much each original variable contributes to each principal component.")
                st.markdown("**Isolation Forest** -- Anomaly detection algorithm that isolates outliers by randomly partitioning data. Points requiring fewer partitions to isolate are more anomalous.")

            _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
            _bg = 'rgba(0,0,0,0)' if st.session_state.get('theme', 'light') == 'dark' else '#FFFFFF'

            # A. Asset Clustering (K-Means)
            st.markdown("#### Asset Clustering (K-Means)")
            if len(data['tickers']) < 3:
                st.info("Need at least 3 assets for meaningful clustering. Add more tickers in the sidebar.")
            else:
                try:
                    features_df = compute_asset_features(data['prices'])
                    n_clusters = st.slider("Number of Clusters (k)", 2, min(6, len(data['tickers'])), min(3, len(data['tickers'])), key='km_k')

                    scaler_km = StandardScaler()
                    features_scaled = scaler_km.fit_transform(features_df.values)

                    method = data.get('clustering_method', 'K-Means')
                    if method == 'Gaussian Mixture':
                        from sklearn.mixture import GaussianMixture
                        model = GaussianMixture(n_components=n_clusters, random_state=42)
                        clusters = model.fit_predict(features_scaled)
                    else:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(features_scaled)
                    features_df['Cluster'] = clusters

                    # PCA for 2D visualization
                    pca_2d = PCA(n_components=2)
                    pca_coords = pca_2d.fit_transform(features_scaled)
                    features_df['PC1'] = pca_coords[:, 0]
                    features_df['PC2'] = pca_coords[:, 1]

                    clust_cols = st.columns(2)
                    with clust_cols[0]:
                        fig_clust = go.Figure()
                        for c in range(n_clusters):
                            mask = features_df['Cluster'] == c
                            fig_clust.add_trace(go.Scatter(
                                x=features_df.loc[mask, 'PC1'],
                                y=features_df.loc[mask, 'PC2'],
                                mode='markers+text',
                                text=features_df.index[mask],
                                textposition='top center',
                                name=f'Cluster {c}',
                                marker=dict(size=12, color=_clrs[c % len(_clrs)]),
                            ))
                        fig_clust.update_layout(
                            template=_tmpl, height=450,
                            title='Asset Clusters (PCA 2D)',
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc, title=f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} var)'),
                            yaxis=dict(gridcolor=_gc, title=f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} var)'),
                            legend=dict(font=dict(color=_fc)),
                        )
                        st.plotly_chart(fig_clust, use_container_width=True)

                    with clust_cols[1]:
                        st.markdown("**Cluster Characteristics:**")
                        display_features = features_df.drop(columns=['PC1', 'PC2'], errors='ignore')
                        render_styled_table(display_features.reset_index().round(4), format_dict={
                            'Ann Return': '{:.2%}', 'Volatility': '{:.2%}',
                            'Sharpe': '{:.2f}', 'Max Drawdown': '{:.2%}',
                            'Skewness': '{:.2f}', 'Kurtosis': '{:.2f}',
                        })

                except Exception as e:
                    st.warning(f"Clustering failed: {e}")

            # B. Market Regime Detection (GMM)
            st.markdown("#### Market Regime Detection")
            regime_ticker = st.selectbox("Select Asset for Regime Detection", data['tickers'], key='regime_ticker')
            try:
                regime_returns = data['returns'][regime_ticker].dropna().values.reshape(-1, 1)
                if len(regime_returns) > 60:
                    n_regimes = st.slider("Number of Regimes", 2, 3, 2, key='n_regimes')
                    gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type='full')
                    regimes = gmm.fit_predict(regime_returns)

                    regime_s = pd.Series(regimes, index=data['returns'][regime_ticker].dropna().index)

                    # Sort regimes by mean return
                    means = [regime_returns[regimes == i].mean() for i in range(n_regimes)]
                    order = np.argsort(means)
                    regime_names = {}
                    if n_regimes == 2:
                        regime_names = {order[0]: 'Bear', order[1]: 'Bull'}
                    else:
                        regime_names = {order[0]: 'Bear', order[1]: 'Transition', order[2]: 'Bull'}

                    regime_colors = {'Bear': _clrs[3], 'Bull': _clrs[2], 'Transition': _clrs[1]}

                    reg_cols = st.columns(2)
                    with reg_cols[0]:
                        fig_reg = go.Figure()
                        price_s = data['prices'][regime_ticker].loc[regime_s.index]
                        for reg_id, reg_name in regime_names.items():
                            mask = regime_s == reg_id
                            fig_reg.add_trace(go.Scatter(
                                x=price_s.index[mask],
                                y=price_s.values[mask],
                                mode='markers',
                                name=reg_name,
                                marker=dict(size=3, color=regime_colors.get(reg_name, 'gray')),
                            ))
                        fig_reg.update_layout(
                            template=_tmpl, height=_chart_h,
                            title=f'{regime_ticker} Price Colored by Regime',
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc), yaxis=dict(gridcolor=_gc, title='Price'),
                            legend=dict(font=dict(color=_fc)),
                        )
                        st.plotly_chart(fig_reg, use_container_width=True)

                    with reg_cols[1]:
                        st.markdown("**Current Regime:**")
                        current_regime = regime_names.get(regimes[-1], 'Unknown')
                        st.metric("Detected Regime", current_regime)
                        st.markdown("**Regime Statistics:**")
                        reg_stats = {}
                        for reg_id, reg_name in regime_names.items():
                            mask = regimes == reg_id
                            reg_stats[reg_name] = {
                                'Mean Daily Return': float(regime_returns[mask].mean()),
                                'Daily Volatility': float(regime_returns[mask].std()),
                                'Days in Regime': int(mask.sum()),
                                'Pct of Time': float(mask.sum() / len(regimes)),
                            }
                        reg_stats_df = pd.DataFrame(reg_stats).T
                        reg_stats_df.index.name = 'Regime'
                        render_styled_table(reg_stats_df.reset_index(), format_dict={
                            'Mean Daily Return': '{:.4%}', 'Daily Volatility': '{:.4%}',
                            'Pct of Time': '{:.1%}',
                        })
                else:
                    st.info("Need at least 60 data points for regime detection.")
            except Exception as e:
                st.warning(f"Regime detection failed: {e}")

            # C. Anomaly Detection (Isolation Forest)
            st.markdown("#### Anomaly Detection (Isolation Forest)")
            anom_ticker = st.selectbox("Select Asset for Anomaly Detection", data['tickers'], key='anom_ticker')
            try:
                anom_returns = data['returns'][anom_ticker].dropna()
                if len(anom_returns) > 30:
                    _contam = data.get('anomaly_sensitivity', 0.05)
                    iso_forest = IsolationForest(contamination=_contam, random_state=42)
                    anom_labels = iso_forest.fit_predict(anom_returns.values.reshape(-1, 1))
                    anomalies = anom_returns.index[anom_labels == -1]

                    fig_anom = go.Figure()
                    price_anom = data['prices'][anom_ticker]
                    fig_anom.add_trace(go.Scatter(x=price_anom.index, y=price_anom.values, name='Price',
                                                  line=dict(color=_clrs[0], width=1.5)))
                    # Mark anomalies
                    anom_in_price = [d for d in anomalies if d in price_anom.index]
                    if anom_in_price:
                        fig_anom.add_trace(go.Scatter(
                            x=anom_in_price,
                            y=price_anom.loc[anom_in_price].values,
                            mode='markers', name='Anomaly',
                            marker=dict(size=8, color=_clrs[3], symbol='x'),
                        ))
                    fig_anom.update_layout(
                        template=_tmpl, height=_chart_h,
                        title=f'{anom_ticker} Price with Anomalous Days',
                        plot_bgcolor=_bg, paper_bgcolor=_bg,
                        font=dict(color=_fc),
                        xaxis=dict(gridcolor=_gc), yaxis=dict(gridcolor=_gc, title='Price'),
                        legend=dict(font=dict(color=_fc)),
                    )
                    st.plotly_chart(fig_anom, use_container_width=True)
                    st.caption(f"Detected {len(anomalies)} anomalous days out of {len(anom_returns)} ({len(anomalies)/len(anom_returns)*100:.1f}%)")
                else:
                    st.info("Need at least 30 data points for anomaly detection.")
            except Exception as e:
                st.warning(f"Anomaly detection failed: {e}")

            # D. PCA Factor Analysis
            st.markdown("#### PCA Factor Analysis")
            if len(data['tickers']) >= 2:
                try:
                    returns_df = data['returns'].dropna()
                    n_comp = min(len(data['tickers']), 5)
                    from sklearn.preprocessing import StandardScaler
                    pca_scaler = StandardScaler()
                    returns_scaled = pca_scaler.fit_transform(returns_df.values)
                    pca_full = PCA(n_components=n_comp)
                    pca_full.fit(returns_scaled)

                    pca_cols = st.columns(2)
                    with pca_cols[0]:
                        fig_var = go.Figure()
                        fig_var.add_trace(go.Bar(
                            x=[f'PC{i+1}' for i in range(n_comp)],
                            y=pca_full.explained_variance_ratio_,
                            name='Individual',
                            marker_color=_clrs[0],
                        ))
                        fig_var.add_trace(go.Scatter(
                            x=[f'PC{i+1}' for i in range(n_comp)],
                            y=np.cumsum(pca_full.explained_variance_ratio_),
                            name='Cumulative',
                            line=dict(color=_clrs[1], width=2),
                            mode='lines+markers',
                        ))
                        fig_var.update_layout(
                            template=_tmpl, height=350,
                            title='Variance Explained by Principal Components',
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc), yaxis=dict(gridcolor=_gc, title='Variance Ratio'),
                            legend=dict(font=dict(color=_fc)),
                        )
                        st.plotly_chart(fig_var, use_container_width=True)

                    with pca_cols[1]:
                        st.markdown("**Factor Loadings (Top 3 PCs):**")
                        loadings = pd.DataFrame(
                            pca_full.components_[:min(3, n_comp)].T,
                            index=returns_df.columns,
                            columns=[f'PC{i+1}' for i in range(min(3, n_comp))]
                        )
                        loadings.index.name = 'Asset'
                        render_styled_table(loadings.reset_index().round(4))
                except Exception as e:
                    st.warning(f"PCA failed: {e}")
            else:
                st.info("Need at least 2 assets for PCA factor analysis.")

        # ================================================================
        # TAB 11: SENTIMENT ANALYSIS (NEW)
        # ================================================================
        with tabs[11]:
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 12</div>
                <div class="section-title">Sentiment Analysis</div>
                <div class="section-subtitle">News-based sentiment scoring and keyword analysis</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Understanding Sentiment Analysis"):
                st.markdown("**Keyword-Based Sentiment Scoring** -- Simple NLP approach that counts positive and negative keywords:")
                st.latex(r"\text{Score} = \frac{N_{positive} - N_{negative}}{N_{positive} + N_{negative}}")
                st.markdown("""
                - Score range: **-1** (very bearish) to **+1** (very bullish)
                - Score = 0: Neutral or no sentiment keywords detected
                """)
                st.markdown("**Limitations:** Keyword-based approaches miss context, sarcasm, and complex sentiment. Transformer-based models (BERT, FinBERT) are more accurate but require additional dependencies.")
                st.markdown("**How to Interpret:**")
                st.markdown("""
                - Aggregate sentiment across many articles is more reliable than individual scores
                - Extreme sentiment (very bullish/bearish) can be a contrarian indicator
                - News sentiment tends to lag price moves but can confirm trends
                """)

            sent_ticker = st.selectbox("Select Ticker for Sentiment", data['tickers'], key='sent_ticker')

            with st.spinner(f"Fetching news for {sent_ticker}..."):
                news_items = fetch_ticker_news(sent_ticker)

            if not news_items:
                st.warning(f"No news articles found for {sent_ticker}. News may not be available for this ticker.")
            else:
                # Parse news
                articles = []
                for item in news_items:
                    title = item.get('title', item.get('content', {}).get('title', ''))
                    if not title:
                        continue
                    publisher = item.get('publisher', item.get('content', {}).get('provider', {}).get('displayName', 'Unknown'))
                    pub_time = item.get('providerPublishTime', item.get('content', {}).get('pubDate', ''))
                    if isinstance(pub_time, (int, float)):
                        pub_date = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d %H:%M')
                    elif isinstance(pub_time, str) and pub_time:
                        pub_date = pub_time[:16]
                    else:
                        pub_date = 'N/A'
                    score = simple_sentiment_score(title)
                    label = 'Bullish' if score > 0.1 else ('Bearish' if score < -0.1 else 'Neutral')
                    articles.append({
                        'Date': pub_date,
                        'Headline': title[:100],
                        'Source': str(publisher)[:20],
                        'Score': score,
                        'Sentiment': label,
                    })

                if not articles:
                    st.warning("Could not parse any news articles.")
                else:
                    articles_df = pd.DataFrame(articles)

                    # Row 1: Sentiment Overview Cards
                    avg_score = articles_df['Score'].mean()
                    n_bullish = (articles_df['Sentiment'] == 'Bullish').sum()
                    n_bearish = (articles_df['Sentiment'] == 'Bearish').sum()
                    n_neutral = (articles_df['Sentiment'] == 'Neutral').sum()

                    ov_cols = st.columns(4)
                    with ov_cols[0]:
                        st.metric("Overall Sentiment", f"{avg_score:+.2f}",
                                  'Bullish' if avg_score > 0.1 else ('Bearish' if avg_score < -0.1 else 'Neutral'))
                    with ov_cols[1]:
                        st.metric("Articles Analyzed", str(len(articles_df)))
                    with ov_cols[2]:
                        st.metric("Bullish", str(n_bullish))
                    with ov_cols[3]:
                        st.metric("Bearish", str(n_bearish))

                    # Row 2: Headlines Table
                    st.markdown("#### News Headlines")
                    render_styled_table(articles_df.head(_tbl_limit), format_dict={'Score': '{:+.2f}'})

                    # Row 3: Sentiment Charts
                    _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
                    _bg = 'rgba(0,0,0,0)' if st.session_state.get('theme', 'light') == 'dark' else '#FFFFFF'

                    sent_chart_cols = st.columns(2)
                    with sent_chart_cols[0]:
                        st.markdown("#### Sentiment Distribution")
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Bar(
                            x=['Bullish', 'Neutral', 'Bearish'],
                            y=[n_bullish, n_neutral, n_bearish],
                            marker_color=[_clrs[2], _clrs[5] if len(_clrs) > 5 else 'gray', _clrs[3]],
                        ))
                        fig_dist.update_layout(
                            template=_tmpl, height=350,
                            title='Sentiment Distribution',
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc), yaxis=dict(gridcolor=_gc, title='Count'),
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                    with sent_chart_cols[1]:
                        st.markdown("#### Sentiment by Source")
                        source_sent = articles_df.groupby('Source')['Score'].mean().sort_values()
                        fig_src = go.Figure()
                        colors = [_clrs[2] if v > 0.1 else (_clrs[3] if v < -0.1 else 'gray') for v in source_sent.values]
                        fig_src.add_trace(go.Bar(
                            x=source_sent.values,
                            y=source_sent.index,
                            orientation='h',
                            marker_color=colors,
                        ))
                        fig_src.update_layout(
                            template=_tmpl, height=350,
                            title='Average Sentiment by Source',
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc, title='Avg Score'), yaxis=dict(gridcolor=_gc),
                        )
                        st.plotly_chart(fig_src, use_container_width=True)

                    # Row 4: Keyword Frequency Analysis
                    st.markdown("#### Top Keywords in Headlines")
                    try:
                        all_words = ' '.join(articles_df['Headline'].tolist()).lower().split()
                        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for',
                                      'of', 'and', 'or', 'but', 'not', 'with', 'by', 'from', 'as', 'it', 'its',
                                      'this', 'that', 'be', 'has', 'have', 'had', 'do', 'does', 'did', 'will',
                                      'would', 'could', 'should', 'may', 'can', '-', '--', '|', 'vs', 'vs.'}
                        filtered = [w.strip('.,!?;:()[]"\'') for w in all_words if len(w) > 2 and w.strip('.,!?;:()[]"\'') not in stop_words]
                        word_freq = pd.Series(filtered).value_counts().head(20)

                        fig_kw = go.Figure()
                        fig_kw.add_trace(go.Bar(
                            x=word_freq.values[::-1],
                            y=word_freq.index[::-1],
                            orientation='h',
                            marker_color=_clrs[0],
                        ))
                        fig_kw.update_layout(
                            template=_tmpl, height=_chart_h,
                            title='Top 20 Keywords in Headlines',
                            plot_bgcolor=_bg, paper_bgcolor=_bg,
                            font=dict(color=_fc),
                            xaxis=dict(gridcolor=_gc, title='Frequency'), yaxis=dict(gridcolor=_gc),
                        )
                        st.plotly_chart(fig_kw, use_container_width=True)
                    except Exception:
                        st.info("Could not generate keyword analysis.")

        # ================================================================
        # TAB 12: EXPORT DATA
        # ================================================================

        with tabs[12]:  # Backtesting
            try:
                render_backtesting_tab(data)
            except Exception as e:
                _logger.error('Backtesting tab error: %s', traceback.format_exc())
                show_error('calculation_error', str(e))

        with tabs[13]:  # Fundamentals
            try:
                render_fundamentals_tab(data)
            except Exception as e:
                _logger.error('Fundamentals tab error: %s', traceback.format_exc())
                show_error('calculation_error', str(e))

        with tabs[14]:  # Fixed Income
            try:
                render_fixed_income_tab(data)
            except Exception as e:
                _logger.error('Fixed Income tab error: %s', traceback.format_exc())
                show_error('calculation_error', str(e))

        with tabs[15]:  # Factor Model
            try:
                render_factor_model_tab(data)
            except Exception as e:
                _logger.error('Factor Model tab error: %s', traceback.format_exc())
                show_error('calculation_error', str(e))

        with tabs[16]:  # Options Builder
            try:
                render_options_builder_tab(data)
            except Exception as e:
                _logger.error('Options Builder tab error: %s', traceback.format_exc())
                show_error('calculation_error', str(e))

        with tabs[17]:  # Risk Suite
            try:
                render_risk_suite_tab(data)
            except Exception as e:
                _logger.error('Risk Suite tab error: %s', traceback.format_exc())
                show_error('calculation_error', str(e))

        with tabs[18]:  # Export Data
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 13</div>
                <div class="section-title">Export Data</div>
                <div class="section-subtitle">Download comprehensive analysis reports in multiple formats</div>
            </div>
            """, unsafe_allow_html=True)

            # Prepare export data dict
            # Fetch macro data for export
            try:
                _export_macro = fetch_all_macro_data()
            except Exception:
                _export_macro = {}
            # Train ML for export (first ticker)
            _export_ml = None
            try:
                _ml_t = data['tickers'][0]
                _export_ml = train_ml_models(data['prices'][_ml_t], data.get('volumes', pd.DataFrame()).get(_ml_t))
            except Exception:
                pass

            # Options chain for export
            _export_options = {}
            try:
                _opt_t = data['tickers'][0]
                _opt_exps = fetch_options_expirations(_opt_t)
                if _opt_exps:
                    _c, _p = fetch_options_chain(_opt_t, _opt_exps[0])
                    _export_options = {'calls': _c, 'puts': _p, 'ticker': _opt_t, 'expiration': _opt_exps[0]}
            except Exception:
                pass

            # Risk data for export
            _export_risk = {}
            try:
                _export_risk_data = fetch_risk_data()
                _export_risk = {
                    'risk_data': _export_risk_data,
                    'composite_score': calculate_composite_risk_score(_export_risk_data),
                }
            except Exception:
                pass

            # Clustering for export
            _export_clustering = {}
            try:
                if len(data['tickers']) >= 3:
                    _feat_df = compute_asset_features(data['prices'])
                    _sc = StandardScaler()
                    _fs = _sc.fit_transform(_feat_df.values)
                    _km = KMeans(n_clusters=min(3, len(data['tickers'])), random_state=42, n_init=10)
                    _feat_df['Cluster'] = _km.fit_predict(_fs)
                    _export_clustering = {'features': _feat_df}
            except Exception:
                pass

            # Sentiment for export
            _export_sentiment = {}
            try:
                _sent_t = data['tickers'][0]
                _news = fetch_ticker_news(_sent_t)
                _sent_articles = []
                for _item in (_news or []):
                    _title = _item.get('title', _item.get('content', {}).get('title', ''))
                    if _title:
                        _score = simple_sentiment_score(_title)
                        _sent_articles.append({'Headline': _title[:80], 'Score': _score})
                if _sent_articles:
                    _export_sentiment = {'ticker': _sent_t, 'articles': pd.DataFrame(_sent_articles)}
            except Exception:
                pass

            export_data = {
                'prices': data['prices'],
                'metrics': data['metrics'],
                'valuation': data['valuation'],
                'portfolio': data['portfolio'],
                'portfolio_metrics': data.get('portfolio_metrics', {}),
                'bubble_scores': data['bubble_scores'],
                'technical': data['technical'],
                'tickers': data['tickers'],
                'monte_carlo': data.get('simulation', {}),
                'macro_data': _export_macro,
                'ml_results': _export_ml,
                'ml_ticker': _ml_t if _export_ml else '',
                'options': _export_options,
                'risk': _export_risk,
                'clustering': _export_clustering,
                'sentiment': _export_sentiment,
            }

            ts = datetime.now().strftime('%Y%m%d_%H%M')

            col1, col2, col3 = st.columns(3)
            with col1:
                with st.spinner('Generating PDF report...'):
                    try:
                        pdf_data = generate_pdf_report(export_data)
                        st.download_button(
                            "PDF Report",
                            pdf_data,
                            f"QuantLab_Report_{ts}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
            with col2:
                with st.spinner('Generating slides...'):
                    try:
                        slides_data = generate_slides(export_data)
                        st.download_button(
                            "Presentation Slides",
                            slides_data,
                            f"QuantLab_Slides_{ts}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Slides generation failed: {e}")
            with col3:
                with st.spinner('Generating Excel workbook...'):
                    try:
                        excel_data = generate_comprehensive_excel(export_data)
                        st.download_button(
                            "Excel Data",
                            excel_data,
                            f"QuantLab_Data_{ts}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Excel generation failed: {e}")

            with st.expander("Report Contents"):
                st.markdown("""
**PDF Report** -- Multi-page A4 research report with:
- Cover page, executive summary, performance charts
- Valuation analysis, portfolio allocation pie chart
- Strategy comparison, bubble detection, technical signals
- Macroeconomic dashboard summary, ML predictions overview
- Options pricing summary (chain snapshot, IV)
- Risk & Geopolitics dashboard (composite score, KPIs)
- ML Clustering (cluster assignments, regime summary)
- Sentiment analysis (top headlines with scores)

**Presentation Slides** -- Landscape slide deck with:
- Title slide, portfolio overview, performance chart
- Metrics tables, valuation summary, allocation visual
- Strategy comparison, bubble detection, key takeaways
- Macro indicators slide, ML model performance slide
- Options Pricing slide, Risk Dashboard slide
- ML Clustering slide, Sentiment Analysis slide

**Excel Workbook** -- Formatted spreadsheet with:
- Summary Dashboard with conditional formatting
- Price History, Performance Metrics, Valuation Analysis
- Portfolio Optimization weights, Strategy Comparison
- Bubble Detection with color-coded risk levels
- Technical Indicators per ticker, embedded charts
- Macro Data sheet with treasury yields, VIX, and S&P 500
- ML Analysis sheet with model metrics and predictions
- Options Chain sheet (calls and puts data)
- Risk Dashboard sheet (risk scores, cross-asset correlations)
- ML Clustering sheet (cluster assignments, PCA components)
- Sentiment sheet (news headlines with scores)
                """)

    # Auto-Refresh Logic
    if enable_autorefresh and st.session_state.analysis_complete:
        # Replace time.sleep + st.rerun with non-blocking approach
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        if time.time() - st.session_state.last_refresh > refresh_rate:
            st.session_state.last_refresh = time.time()
            st.rerun()

if __name__ == "__main__":
    main()