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
def get_benchmark_data(start_date, end_date):
    """Fetches S&P 500 data for beta calculations"""
    try:
        benchmark = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
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
            variance = np.var(bench_ret)
            
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
                interest_expense = info.get('interestExpense', 0)
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
        downside_returns = self.returns.copy()
        downside_returns[downside_returns > 0] = 0
        return downside_returns.cov() * 252
    
    def _calculate_cvar_matrix(self, alpha=0.05):
        """Calculate CVaR covariance matrix (REAL LOGIC)"""
        # Filter for days where the portfolio (or market) is in the bottom tail
        # Proxy: weigh negative returns more heavily
        is_tail = self.returns < self.returns.quantile(alpha)
        
        # If sufficient data, compute covariance of tail events
        if is_tail.shape[0] > 10:
            # We use pairwise intersection for robust covariance in tail
            # Simplified: Use semi-covariance of the tail
            tail_rets = self.returns[is_tail].fillna(0)
            return tail_rets.cov() * 252
            
        # Fallback: Semicovariance
        negative_rets = self.returns.copy()
        negative_rets[negative_rets > 0] = 0
        return negative_rets.cov() * 252
    
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
        lambda_param = (self.mean_returns.mean() - self.rf_rate) / self.cov_matrix.diagonal().mean()
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
        
        indicators['RSI'] = ta.momentum.RSIIndicator(prices, window=14).rsi()
        
        bb = ta.volatility.BollingerBands(prices)
        indicators['BB_Upper'] = bb.bollinger_hband()
        indicators['BB_Lower'] = bb.bollinger_lband()
        
        return indicators

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
        self.cell(0, 6, 'QuantLab Analysis Report', align='L')
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

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# --------------- Presentation Slides (landscape PDF) ---------------

class _QuantLabSlides(FPDF):
    """Landscape 16:9 slide deck."""

    def footer(self):
        self.set_y(-10)
        self.set_font('Helvetica', '', 7)
        self.set_text_color(*_DARK_GRAY)
        self.cell(0, 6, f'Slide {self.page_no()}', align='R')

    def slide_title_bar(self, title):
        self.set_fill_color(*_NAVY)
        self.rect(0, 0, 297, 25, 'F')
        self.set_xy(10, 5)
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(*_WHITE)
        self.cell(0, 15, title)
        self.set_xy(10, 30)

    def add_table(self, headers, rows, col_widths=None, x_offset=10):
        n = len(headers)
        if col_widths is None:
            avail = 277
            col_widths = [avail / n] * n
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

    pdf = _QuantLabSlides('L', 'mm', (297, 167))
    pdf.set_auto_page_break(auto=False)

    # ---- Slide 1: Title ----
    pdf.add_page()
    pdf.set_fill_color(*_NAVY)
    pdf.rect(0, 0, 297, 167, 'F')
    pdf.set_xy(0, 40)
    pdf.set_font('Helvetica', 'B', 36)
    pdf.set_text_color(*_WHITE)
    pdf.cell(297, 20, 'QuantLab', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 18)
    pdf.set_text_color(*_TEAL)
    pdf.cell(297, 12, 'Portfolio Analytics & Research', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(180, 190, 210)
    pdf.cell(297, 8, f"Tickers: {', '.join(tickers)}", align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(297, 8, datetime.now().strftime('%B %d, %Y'), align='C', new_x='LMARGIN', new_y='NEXT')

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
            pdf.image(chart_buf, x=20, y=32, w=257)
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

    # ---- Slide 10: Key Takeaways ----
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
    for i, t in enumerate(takeaways):
        pdf.set_x(20)
        pdf.multi_cell(257, 9, f"  -  {t}", new_x='LMARGIN', new_y='NEXT')

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

        # Quick Presets
        st.markdown("### Presets")

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
        
        analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)
    
    # Main Analysis Logic
    should_run = analyze_btn or (st.session_state.analysis_complete and enable_autorefresh)

    if should_run:
        with st.spinner("Running comprehensive analysis..."):
            try:
                # Parse tickers
                tickers = [t.strip().upper() for t in tickers_input.split()]
                
                # Fetch data
                prices, volumes = fetch_market_data(tickers, start_date, end_date) # Unpack Volumes
                
                # --- UPDATE TIMESTAMP HERE ---
                st.session_state.last_updated = pd.Timestamp.now('America/New_York').strftime("%Y-%m-%d %I:%M:%S %p")
                
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
                
                # Fetch Benchmark Data ONCE
                benchmark_prices = get_benchmark_data(start_date, end_date)

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
                    'rf_rate': rf_rate
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
            "Market Dashboard",
            "Valuation",
            "Portfolio",
            "Bubble Detection",
            "Monte Carlo",
            "Technicals",
            "Export"
        ])
        
        with tabs[0]:  # Market Dashboard
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(plot_price_history(data['prices']), use_container_width=True)
            with col2:
                st.markdown("#### Performance Metrics")
                render_styled_table(
                    data['metrics'],
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
                display_df,
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
            
            _tmpl, _clrs, _gc, _fc = _get_plotly_theme()
            _fill = 'rgba(0,180,216,0.15)' if st.session_state.get('theme') == 'dark' else 'rgba(0,144,181,0.12)'
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=days, y=p95, line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=days, y=p5, fill='tonexty',
                fillcolor=_fill,
                name='90% Confidence',
                line=dict(width=0)
            ))
            fig.add_trace(go.Scatter(
                x=days, y=p50, name='Median',
                line=dict(color=_clrs[0], width=2)
            ))

            fig.update_layout(
                title=dict(text=f"Monte Carlo Projection: {sim_ticker}", font=dict(color=_fc)),
                template=_tmpl,
                height=500,
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
        
        with tabs[6]:  # Export Data
            st.markdown("""
            <div class="section-header">
                <div class="section-label">SECTION 07</div>
                <div class="section-title">Export Data</div>
                <div class="section-subtitle">Download comprehensive analysis reports in multiple formats</div>
            </div>
            """, unsafe_allow_html=True)

            # Prepare export data dict
            export_data = {
                'prices': data['prices'],
                'metrics': data['metrics'],
                'valuation': data['valuation'],
                'portfolio': data['portfolio'],
                'portfolio_metrics': data.get('portfolio_metrics', {}),
                'bubble_scores': data['bubble_scores'],
                'technical': data['technical'],
                'tickers': data['tickers'],
                'monte_carlo': data.get('monte_carlo', {}),
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
**PDF Report** — Multi-page A4 research report with:
- Cover page, executive summary, performance charts
- Valuation analysis, portfolio allocation pie chart
- Strategy comparison, bubble detection, technical signals

**Presentation Slides** — Landscape slide deck with:
- Title slide, portfolio overview, performance chart
- Metrics tables, valuation summary, allocation visual
- Strategy comparison, bubble detection, key takeaways

**Excel Workbook** — Formatted spreadsheet with:
- Summary Dashboard with conditional formatting
- Price History, Performance Metrics, Valuation Analysis
- Portfolio Optimization weights, Strategy Comparison
- Bubble Detection with color-coded risk levels
- Technical Indicators per ticker, embedded charts
                """)

    # Auto-Refresh Logic
    if enable_autorefresh and st.session_state.analysis_complete:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()