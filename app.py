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

@st.cache_data(ttl=60) # Reduced TTL for live data
def fetch_market_data(tickers, start_date, end_date):
    """Robust data fetching using yfinance - Returns Prices AND Volume"""
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    
    data = yf.download(tickers, start=start_date, end=end_date, 
                       group_by='ticker', auto_adjust=True, progress=False)
    
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    if len(tickers) == 1:
        ticker = tickers[0]
        # Check structure
        cols = data.columns
        # Price extraction
        if isinstance(cols, pd.MultiIndex):
            if 'Close' in cols: prices[ticker] = data['Close']
            else: prices[ticker] = data[ticker]['Close']
            
            if 'Volume' in cols: volumes[ticker] = data['Volume']
            else: volumes[ticker] = data[ticker]['Volume']
        elif 'Close' in cols:
            prices[ticker] = data['Close']
            if 'Volume' in cols: volumes[ticker] = data['Volume']
        elif ticker in cols:
             prices[ticker] = data[ticker] # Fallback
    else:
        for t in tickers:
            try:
                # Try getting Close price
                if t in data.columns.levels[0]:
                    prices[t] = data[t]['Close']
                    if 'Volume' in data[t].columns:
                        volumes[t] = data[t]['Volume']
            except:
                # Fallback flat structure
                if (t, 'Close') in data.columns:
                    prices[t] = data[(t, 'Close')]
                    if (t, 'Volume') in data.columns:
                        volumes[t] = data[(t, 'Volume')]
                elif t in data.columns:
                    prices[t] = data[t]
                
    prices.dropna(inplace=True)
    # Align volumes to prices
    volumes = volumes.reindex(prices.index).fillna(0)
    
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
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)

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
        
        # Override risk-free rate if custom is set
        if use_custom_rf and custom_rf is not None:
            rf_rate = custom_rf / 100.0

        analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    # Main Analysis Logic
    should_run = analyze_btn or (st.session_state.analysis_complete and enable_autorefresh)

    if should_run:
        with st.spinner("Running comprehensive analysis..."):
            try:
                # Parse tickers
                import re
                tickers = [t for t in re.split(r'[\s,;]+', tickers_input.strip()) if t]
                tickers = [re.sub(r'[^A-Za-z0-9.=^-]', '', t).upper() for t in tickers if t.strip()]
                
                # Fetch data
                prices, volumes = fetch_market_data(tickers, start_date, end_date) # Unpack Volumes
                
                # --- UPDATE TIMESTAMP HERE ---
                st.session_state.last_updated = pd.Timestamp.now('America/New_York').strftime("%Y-%m-%d %I:%M:%S %p")
                
                if prices.empty:
                    st.error("No data found")
                    return
                
                # Calculate returns
                returns = prices.pct_change().bfill().dropna(how='all')
                
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
                
                # Fetch Benchmark Data ONCE
                if benchmark_ticker:
                    benchmark_prices = get_benchmark_data(start_date, end_date, benchmark_ticker)
                else:
                    benchmark_prices = pd.Series()

                for ticker in tickers:
                    # 1. Calculate Dynamic Beta
                    if not benchmark_prices.empty and ticker in prices.columns:
                        beta = EnhancedValuationMetrics.calculate_beta(prices[ticker], benchmark_prices)
                    else:
                        beta = 1.0
                    
                    # 2. Pass Beta to methods
                    wacc = EnhancedValuationMetrics.calculate_wacc(ticker, rf_rate, beta)
                    capm = EnhancedValuationMetrics.calculate_capm_return(rf_rate, beta)
                    
                    # Fama-French
                    ff = EnhancedValuationMetrics.calculate_fama_french_return(ticker, prices[[ticker]], rf_rate, beta)
                    
                    # APT (Uses own logic, can stay same or be updated)
                    apt = EnhancedValuationMetrics.calculate_apt_return(ticker, prices[[ticker]], rf_rate)
                    
                    # Bubble detection (PASS VOLUME)
                    vol_data = volumes[ticker] if ticker in volumes.columns else None
                    bubble_res = bubble_detector.detect_bubbles(prices[ticker], returns[ticker], vol_data)
                    bubble_scores[ticker] = bubble_res['bubble_score']
                    
                    # Risk Impact
                    impact = EnhancedValuationMetrics.calculate_bubble_burst_impact(
                        ticker, prices[ticker], bubble_res['bubble_score'], beta
                    )
                    
                    # DCF (Pass beta/rf only)
                    dcf = EnhancedValuationMetrics.calculate_dcf_value(ticker, rf_rate, beta)

                    valuation_results[ticker] = {
                        'DCF Enterprise Value': dcf,
                        'WACC': wacc,
                        'CAPM Return': capm,
                        'Fama-French Return': ff,
                        'APT Return': apt,
                        'Bubble Score': bubble_res['bubble_score'],
                        'Bubble Burst Impact': impact,
                        'Beta': beta # Helpful to see in the table!
                    }
                
                valuation_df = pd.DataFrame(valuation_results).T
                
                # Portfolio Optimization
                optimizer = EnhancedPortfolioOptimizer(prices, bubble_scores, rf_rate)
                
                portfolio_results = {
                    'Min Variance': optimizer.minimum_variance(bubble_aware, penalty_factor),
                    'Risk Parity': optimizer.risk_parity(bubble_aware, penalty_factor),
                    'Min CVaR': optimizer.minimum_cvar(bubble_aware=bubble_aware, penalty_factor=penalty_factor)
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
                    'rf_rate': rf_rate,
                    'volumes': volumes,
                    # Advanced settings
                    'confidence_level': confidence_level,
                    'benchmark_ticker': benchmark_ticker,
                    'ml_training_years': ml_training_years,
                    'clustering_method': clustering_method,
                    'anomaly_sensitivity': anomaly_sensitivity,
                    'chart_height': chart_height,
                    'table_row_limit': table_row_limit,
                    'export_sections': export_sections,
                    'rebalancing': rebalancing,
                    'sim_method': sim_method,
                }
                st.session_state.analysis_complete = True
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())

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
            "Market Dashboard",      # 0
            "Valuation",             # 1
            "Portfolio",             # 2
            "Bubble Detection",      # 3
            "Monte Carlo",           # 4
            "Technicals",            # 5
            "Options Pricing",       # 6 NEW
            "Macro Dashboard",       # 7
            "Risk & Geopolitics",    # 8 NEW
            "ML Predictions",        # 9
            "ML Clustering",         # 10 NEW
            "Sentiment Analysis",    # 11 NEW
            "Export"                 # 12
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
                st.warning(f"No options data available for {opt_ticker}. Options chains may not exist for this ticker.")
            else:
                selected_exp = st.selectbox("Expiration Date", expirations, key='opt_exp')
                opt_view = st.radio("View", ["Calls", "Puts", "Both"], horizontal=True, key='opt_view')

                # Fetch chain
                calls_df, puts_df = fetch_options_chain(opt_ticker, selected_exp)

                if calls_df.empty and puts_df.empty:
                    st.warning("Could not fetch options chain data.")
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
        with tabs[12]:  # Export Data
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