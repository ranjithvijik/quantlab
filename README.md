# QuantLab — Advanced Portfolio Analytics

[![Live App](https://img.shields.io/badge/Live%20App-rjquantlab.streamlit.app-blue?logo=streamlit)](https://rjquantlab.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![CI](https://github.com/ranjithvijik/quantlab/actions/workflows/qa.yml/badge.svg)](https://github.com/ranjithvijik/quantlab/actions/workflows/qa.yml)
[![Tests](https://img.shields.io/badge/tests-590%2B%20passing-brightgreen)](QA-REPORT.md)
[![Grade](https://img.shields.io/badge/QA%20Grade-A%2B-brightgreen)](QA-REPORT.md)
[![Tabs](https://img.shields.io/badge/tabs-36-blue)](https://rjquantlab.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

QuantLab is an institutional-grade multi-asset quantitative research platform built with Streamlit. It covers the complete investment research workflow across **36 analytical tabs** — from data ingestion, fundamental analysis, and technical indicators through portfolio optimization, backtesting, bubble detection, fixed income analytics, machine learning, factor models, options strategy building, portfolio stress testing, pairs trading, sector rotation, earnings analysis, tail risk, cross-asset correlation, ESG scoring, enhanced pairs backtesting, macro regime detection, cryptocurrency on-chain metrics, insider trading analysis, and watchlist management — all running in the browser with no local setup required.

---

## Live Demo

**[https://rjquantlab.streamlit.app/](https://rjquantlab.streamlit.app/)**

---

## Features

### Portfolio Optimization
- **9 strategies:** Maximum Sharpe Ratio, Minimum Variance, Risk Parity, Minimum CVaR, Maximum Diversification, Kelly Criterion, Black-Litterman, Hierarchical Risk Parity (HRP), Equal Weight
- Bubble-aware optimization — portfolio weights dynamically penalized by bubble detection scores
- Efficient frontier visualization
- Backtest vs. equal-weight benchmark
- Strategy comparison table

### Bubble Detection
- **Metcalfe's Law analysis** — Market-to-Metcalfe Value (MMV) ratio using volume as a network-activity proxy
- **GPH long-memory estimator** — fractional integration parameter *d*; *d* > 0.5 implies non-stationary bubble behavior
- **Kurtosis scoring** — fat-tailed return distributions flag elevated speculative risk
- **Ljung-Box volatility clustering test** — detects ARCH effects in squared returns
- Composite bubble score (0–1) with Normal / Caution / High Risk regime labels

### Valuation Models
Five models per ticker, computed from live yfinance fundamental data:
1. **CAPM** — Expected return via the Security Market Line (E(R) = Rf + β·ERP)
2. **WACC** — Blended cost of equity and after-tax debt
3. **DCF** — Enterprise value from discounted free cash flows with terminal value guard (WACC > g)
4. **Fama-French 3-Factor** — Market, SMB, and HML factor exposures
5. **APT** — Arbitrage Pricing Theory multi-factor expected return

### Technical Analysis
Computed via the `ta` library, displayed as overlays on price charts:
- SMA 20, SMA 50
- EMA 12
- MACD (EMA 12 − EMA 26), Signal (EMA 9), Histogram
- RSI 14 (Wilder's EMA smoothing)
- Bollinger Bands (SMA 20 ± 2σ)

### Monte Carlo Simulation
Two engines selectable per run:
- **Geometric Brownian Motion** — standard log-normal price paths calibrated to historical mu and sigma
- **Behavioral Agent Model** — heterogeneous agent model mixing Fundamentalist mean-reversion and Speculator momentum demand; calibrated to historical beta, autocorrelation, and sector dynamics

Outputs: median price path, VaR, CVaR, 90 % confidence band, regime labels (Bubble / Fair / Undervalued)

### Options Pricing
- Live options chain (calls & puts) from Yahoo Finance
- 3D implied volatility surface across strikes and expirations
- Black-Scholes pricing with full Greeks: Delta, Gamma, Theta, Vega, Rho
- Payoff diagrams: Long Call/Put, Bull/Bear Spread, Straddle, Strangle, Iron Condor, Covered Call

### Macro Dashboard
Live data via yfinance:
- Treasury yield curve (3-month, 5-year, 10-year, 30-year)
- VIX, S&P 500 performance
- WACC scenario analysis

### Risk & Geopolitics Dashboard
- Cross-asset signals: VIX, US Dollar Index (DXY), Gold, WTI Crude Oil
- Yield curve spread (10Y − 3M) with inversion warnings
- Composite risk score (0–100) weighted across VIX, yield curve slope, and safe-haven demand

### ML Predictions
Three models trained on configurable historical windows (1–5 years):
- **Linear Regression** — interpretable baseline
- **Random Forest** — bagged ensemble with feature importance
- **Gradient Boosting** — boosted residual fitting

Features: returns (1d, 5d, 21d), volatility, SMA ratios (P/SMA 20/50/200), volume ratio, RSI.  
Train/test split is strictly chronological (last 20 % held out). Metrics: R², RMSE, MAE.

### ML Clustering & Regime Detection
- **Asset clustering** — K-Means or Gaussian Mixture across return/volatility/Sharpe/skewness/kurtosis/drawdown features (PCA-reduced to 2D for visualization)
- **Market regime detection** — GMM-based Bull / Bear / Sideways labeling on rolling returns
- **Anomaly detection** — Isolation Forest with tunable contamination parameter

### Sentiment Analysis
- Fetches recent headlines from Yahoo Finance news API
- Keyword-based scoring: Score = (N_positive − N_negative) / (N_positive + N_negative) ∈ [−1, +1]
- Bullish / Neutral / Bearish classification per article with aggregate trend summary

### Backtesting Engine *(new)*
- Event-driven walk-forward backtester — no lookahead bias
- 4 strategies: Equal Weight, Max Sharpe, Min Variance, Risk Parity
- Rebalancing: Monthly, Quarterly, Annually, Buy & Hold
- 16 metrics: CAGR, Sharpe, Sortino, Calmar, Max Drawdown Duration, Win Rate, Profit Factor, VaR, CVaR
- Equity curve vs benchmark, drawdown chart, rolling Sharpe, trade log, strategy comparison

### Fundamental Data Panel *(new)*
- 10 key ratios: P/E, P/B, EV/EBITDA, ROE, ROIC, D/E, FCF Yield, Gross/Net Margin, Current Ratio
- Color-coded favorable/unfavorable thresholds per ratio
- Income statement, balance sheet, cash flow charts (4 years historical)
- Earnings history with EPS beat/miss tracking and beat rate calculation

### Fixed Income & Macro Analytics *(new)*
- Bond Pricing Calculator: price, Macaulay duration, modified duration, convexity, DV01
- Price sensitivity table: ±50/100/200bps yield shocks with duration+convexity estimates
- Full yield curve (3M, 1Y, 5Y, 10Y, 30Y) with current/1M ago/1Y ago overlays
- 3M-10Y and 2Y-10Y spread with inversion detection
- Portfolio rate sensitivity: per-asset rate beta and shock impact estimates

### Multi-Factor Alpha Model *(new)*
- 5 factors: Momentum (12M-1M), Value (contrarian), Quality (Sharpe-based), Low Vol, Size
- Cross-sectional z-scored factor heatmap
- Portfolio factor exposure decomposition
- Factor timing: which factors are rewarding in the current market regime
- Alpha attribution via OLS: alpha, factor betas, R², residual vol, Information Ratio

### Options Strategy Builder *(new)*
- 8 strategy templates: Bull/Bear Spread, Straddle, Strangle, Iron Condor, Covered Call, Protective Put, Butterfly
- Multi-leg custom strategy builder with add/remove legs
- Net premium, max profit/loss, breakeven calculation
- Aggregated Greeks: Net Delta, Gamma, Theta, Vega
- Payoff diagram with profit/loss zones, breakeven lines, Greeks vs stock price
- Options screener: filter by IV rank, volume, open interest, moneyness

### Portfolio Risk Suite *(new)*
- 6 historical stress tests: 2008 Crisis, COVID 2020, 2022 Rate Shock, Dot-com Crash, Black Monday, Flash Crash
- Custom scenario analysis: equity/rate/vol sliders with live portfolio impact
- Factor VaR decomposition: market beta vs idiosyncratic contribution per ticker
- Correlation breakdown detector: 252-day vs 21-day heatmaps with crisis spike alerts
- Diversification verdict: "HOLDING" vs "BREAKING DOWN"

### Pairs Trading & Statistical Arbitrage *(new)*
- Cointegration testing across all ticker pairs using the Engle-Granger test
- Spread analysis with z-score bands, hedge ratio (OLS), and rolling mean
- Half-life of mean reversion calculation
- Signal generation with configurable entry/exit z-score thresholds
- Pairs strategy backtester with equity curve, Sharpe ratio, win rate, and max drawdown

### Sector Rotation & Relative Strength *(new)*
- All 11 GICS sector ETFs tracked with multi-period momentum (1M/3M/6M/12M)
- Relative strength analysis vs S&P 500 benchmark
- Regime detection: Risk-On / Risk-Off / Neutral based on cyclical vs defensive spread
- Recommended sector tilts per regime
- Sector correlation heatmap

### Earnings & Event Analysis *(new)*
- Earnings history from yfinance with surprise %, beat/miss tracking
- Post-earnings announcement drift (PEAD) — cumulative abnormal return T-5 to T+20
- Event volatility ratio: event-period vol vs normal periods
- Earnings surprise scatter (surprise % vs next-day return)
- Monthly return seasonality: average return, win rate, and standard deviation by month

### Tail Risk & Drawdown Analysis *(new)*
- Extreme Value Theory: Generalized Pareto Distribution fit to tail losses
- Return distribution analysis: skewness, kurtosis, Jarque-Bera test, Shapiro-Wilk test
- Histogram with fitted Normal and t-distribution overlays
- QQ plot (empirical vs normal quantiles)
- Underwater drawdown chart with episode table (depth, duration, recovery)
- Lower tail dependence coefficients across portfolio assets
- Worst daily returns table

### Currency & Cross-Asset Correlation *(new)*
- Dynamic correlation heatmaps: current 63-day vs full-period comparison
- Rolling multi-window correlations (21d, 63d, 252d) for any asset pair
- Correlation regime detection: Normal / Crisis / Divergent
- Cross-asset momentum ranking across equities, bonds, gold, oil, USD, VIX
- Currency strength meter: 21-day relative performance of major FX pairs

### ESG & Alternative Data Scoring *(new)*
- ESG scores from yfinance Sustainalytics data: Environment, Social, Governance breakdown
- Radar chart per ticker with E/S/G component visualization
- Peer ranking by total ESG score with z-scores
- Controversy level monitoring
- ESG-return analysis: High ESG vs Low ESG portfolio Sharpe comparison
- ESG Score vs Sharpe Ratio scatter plot

### Enhanced Pairs Trading Backtest *(new)*
- Kalman filter for time-varying hedge ratio estimation (manual implementation, no pykalman)
- Walk-forward optimization of entry/exit z-score thresholds to avoid overfitting
- Regime-adaptive thresholds: widens bands in high-vol, tightens in low-vol
- Full backtest with equity curve, drawdown chart, trade log, monthly returns
- Performance comparison: Classic OLS vs Kalman Filter vs Regime-Adaptive

### Macro Regime Detector *(new)*
- Classifies market regimes using yield curve slope, credit spreads (HYG-TLT), momentum (SPY vs 200d SMA), VIX
- Composite regime scoring: expansion, late cycle, contraction, recovery
- Indicator dashboard with signal gauges and historical context charts
- Asset performance breakdown by regime
- Regime transition probability matrix

### Cryptocurrency On-Chain Metrics *(new)*
- CoinGecko free API integration (via urllib.request, no new dependencies)
- NVT Ratio (Network Value to Transactions) with valuation bands
- MVRV proxy using realized price approximation across multiple lookback windows
- Fear & Greed proxy score (0-100) based on volatility, momentum, volume, and ATH drawdown
- Market dominance pie chart, crypto correlation heatmap
- Supports BTC, ETH, SOL, ADA, DOT, LINK, AVAX, MATIC

### Insider Trading Tracker *(new)*
- SEC Form 4 insider transaction data via yfinance
- Insider sentiment score (-100 to +100) with C-suite weighting
- Cluster buy/sell detection: flags 3+ insiders trading same direction within 30 days
- Insider transactions overlaid on price chart (buy/sell markers)
- Forward return analysis after insider buys vs sells (30d, 60d, 90d)

### Watchlist & Alerts *(new)*
- Create and manage named watchlists stored in session state
- Watchlist snapshot: price, change%, volume, RSI, SMA 50/200, 52-week range
- Performance heatmap (tickers x timeframes, color-coded returns)
- 10 configurable alert types: price threshold, SMA cross, RSI overbought/oversold, volume spike, drawdown, earnings proximity, percent change
- Active alerts table with triggered alert notifications

### ML-Powered Top 10 Assets *(new)*
- Scans ~80 liquid assets across stocks (30), ETFs (18), crypto (10), forex (8), commodities (10)
- 20-feature vectors per asset: momentum, mean-reversion, volatility, technical, volume, risk
- 3-model ensemble: multi-factor composite z-score (40%), Random Forest classifier (30%), K-Means cluster quality (30%)
- Top 10 summary cards, category breakdown donut chart, score comparison bar chart
- Asset deep dive: price chart with SMA overlay, feature radar chart, ML explanation
- Full universe treemap color-coded by ensemble score
- Model diagnostics: feature importance, PCA cluster scatter, score distribution

### News Sentiment Analyzer *(new)*
- TF-IDF + ComplementNB classifier trained semi-supervised on lexicon-labeled headlines
- 40-word financial sentiment lexicon with lexicon (40%) + NB (60%) ensemble scoring
- Sentiment time series overlaid on price chart with dual y-axis
- Divergence detection: positive news + falling price (bullish) and vice versa
- Sentiment-price correlation analysis (concurrent, predictive, lag 1d/5d)
- Word frequency chart from recent headlines

### Smart Portfolio Constructor *(new)*
- Ledoit-Wolf covariance shrinkage for robust estimation
- Market-implied equilibrium returns via Black-Litterman framework
- ML-derived views from momentum signals with confidence weighting
- Portfolio optimization via scipy.optimize.minimize (SLSQP)
- Regime-conditional Monte Carlo fan chart with percentile bands
- Strategy comparison: Smart Portfolio vs Equal-Weight vs Min-Variance vs Max-Sharpe

### Risk Decomposition Engine *(new)*
- Factor proxy ETFs: SPY (market), IWM (size), IWD (value), MTUM (momentum), QUAL (quality), USMV (low vol)
- OLS factor regression per asset with alpha, betas, R-squared, t-stats
- Return attribution waterfall chart (factor contributions + alpha = total)
- Risk decomposition pie chart (factor risk % vs idiosyncratic %)
- Rolling factor exposure tracking with drift alerts (>2 std deviation)
- Morningstar-style box classification (size x style 3x3 grid)

### ML Price Forecaster *(new)*
- 3-model ensemble: Rolling Linear Regression (25%), Gradient Boosting (45%), ARIMA+EWMA vol (30%)
- Walk-forward validation with expanding training window
- Confidence intervals from EWMA volatility forecasting
- Model disagreement metric with High/Medium/Low confidence labels
- Backtest accuracy panel: MAE, RMSE, directional accuracy
- Feature importance from Gradient Boosting model

### Earnings Surprise Predictor *(new)*
- Pre-earnings features: momentum, volatility, RSI, volume trend, historical beat rate, consecutive beats
- RandomForestClassifier with TimeSeriesSplit cross-validation
- Beat probability gauge (0-100%) with Likely Beat / Likely Miss / Toss-Up labels
- Walk-forward backtest of prediction performance
- Feature importance analysis and historical beat rate trend

### Export System
All three formats generated on-demand in the Export tab:
- **PDF Report** — multi-page A4 research report (cover, metrics, valuation, portfolio, bubble detection, macro, risk, ML, options, clustering, sentiment)
- **Presentation Slides** — landscape PDF slide deck (297 × 167 mm) with charts and metrics tables
- **Excel Workbook** — formatted `.xlsx` with conditional formatting and separate sheets per analysis module

### Error Handling
- Typed exception hierarchy: `DataFetchError`, `ValidationError`, `CalculationError`, `ExportError`
- User-friendly error messages with recovery hints (`show_error()`)
- 8-step progress bar during analysis with per-step labels
- Graceful degradation — individual ticker/module failures produce warnings without crashing the pipeline
- Debug Mode toggle (Advanced Settings → Developer Options) for full stack traces

---

## 36 Tabs Overview

| # | Tab | Contents |
|---|-----|----------|
| 1 | **Market Dashboard** | Normalized price history, annual return, volatility, Sharpe ratio, max drawdown |
| 2 | **Valuation** | DCF, CAPM, WACC, Fama-French 3-Factor, APT models per ticker |
| 3 | **Portfolio** | 9 optimization strategies, efficient frontier, backtest, strategy comparison |
| 4 | **Bubble Detection** | Metcalfe MMV ratio, GPH *d*, kurtosis, Ljung-Box, composite score |
| 5 | **Monte Carlo** | GBM and Behavioral Agent simulations, VaR, CVaR, regime labeling |
| 6 | **Technicals** | SMA 20/50, EMA 12, MACD + histogram, RSI 14, Bollinger Bands |
| 7 | **Options Pricing** | Live chains, IV surface, Black-Scholes Greeks, payoff diagrams |
| 8 | **Macro Dashboard** | Yield curve, VIX, S&P 500, WACC scenario analysis |
| 9 | **Risk & Geopolitics** | VIX regimes, yield curve spread, Gold, Oil, DXY, composite risk score |
| 10 | **ML Predictions** | Linear Regression, Random Forest, Gradient Boosting; R², RMSE, feature importance |
| 11 | **ML Clustering** | K-Means / GMM clustering, PCA, regime detection, Isolation Forest anomalies |
| 12 | **Sentiment Analysis** | Yahoo Finance news, keyword sentiment scoring, bullish/bearish breakdown |
| 13 | **Backtesting** | Walk-forward backtest, 16 metrics, equity curve, drawdown, trade log, strategy comparison |
| 14 | **Fundamentals** | 10 key ratios, financial statements (4yr), earnings history, valuation context |
| 15 | **Fixed Income** | Bond calculator, duration/convexity/DV01, yield curve overlays, rate sensitivity |
| 16 | **Factor Model** | 5-factor scores, portfolio exposure, factor timing, alpha attribution via OLS |
| 17 | **Options Builder** | 8 strategy templates, multi-leg builder, payoff diagram, Greeks, screener |
| 18 | **Risk Suite** | 6 stress tests, custom scenarios, factor VaR, correlation breakdown detector |
| 19 | **Export** | PDF report, presentation slides, Excel workbook download |
| 20 | **Pairs Trading** ✨ | Cointegration matrix, spread z-score, half-life, entry/exit signals, pairs backtest |
| 21 | **Sector Rotation** ✨ | GICS sector ETF momentum, relative strength, rotation regime, sector correlation |
| 22 | **Earnings & Events** ✨ | Earnings drift, event volatility, surprise impact, monthly seasonality |
| 23 | **Tail Risk** ✨ | EVT/GPD tail fit, drawdown episodes, QQ plot, tail dependence, worst periods |
| 24 | **Cross-Asset** ✨ | Dynamic correlations, regime detection, cross-asset momentum, currency strength |
| 25 | **ESG Scoring** ✨ | ESG scores (E/S/G radar), peer ranking, controversy, ESG-return analysis |
| 26 | **Pairs Backtest** ✨ | Kalman filter hedge ratio, walk-forward optimization, regime-adaptive thresholds, full backtest |
| 27 | **Macro Regime** ✨ | Yield curve, credit spreads, momentum, VIX regime classification, transition matrix |
| 28 | **Crypto On-Chain** ✨ | NVT ratio, MVRV proxy, Fear & Greed, market dominance, crypto correlation |
| 29 | **Insider Trading** ✨ | Insider sentiment score, cluster detection, forward returns, transaction timeline |
| 30 | **Watchlist & Alerts** ✨ | Named watchlists, 10 alert types, snapshot table, performance heatmap |
| 31 | **ML Top 10 Assets** ✨ | Ensemble ML scoring across ~80 assets in 5 categories, top picks with explanations |
| 32 | **News Sentiment** ✨ | TF-IDF + NB sentiment analysis, divergence detection, sentiment-price correlation |
| 33 | **Smart Portfolio** ✨ | Black-Litterman with Ledoit-Wolf covariance, ML views, regime Monte Carlo |
| 34 | **Risk Decomposition** ✨ | Factor regression, return/risk attribution, rolling exposure, style box |
| 35 | **Price Forecast** ✨ | 3-model ensemble (LR + GBR + ARIMA), confidence intervals, backtest accuracy |
| 36 | **Earnings Predictor** ✨ | Random Forest beat/miss prediction, walk-forward validation, feature importance |

---

## Supported Asset Classes

| Asset Class | Example Tickers |
|-------------|-----------------|
| Stocks & ETFs | NVDA, TSLA, AAPL, MSFT, GOOGL |
| Crypto | BTC-USD, ETH-USD, SOL-USD |
| Forex | EURUSD=X, GBPUSD=X, USDJPY=X |
| Commodities | GC=F (Gold), CL=F (WTI Crude), SI=F (Silver) |
| Options | AAPL, MSFT, NVDA, TSLA, AMZN |

---

## Quick Presets

22 one-click presets organized by category:

**Stocks** — Tech Giants, Semiconductor, EV & Clean Energy, FAANG+, Banking & Finance, Healthcare & Pharma, Energy & Oil, Mega Cap, Dividend Aristocrats, Growth Stocks, Value Stocks

**Crypto** — Crypto Majors, Crypto DeFi

**ETFs** — Index ETFs, Bond ETFs, Sector ETFs, Commodity ETFs

**Forex** — Forex Majors, Forex Emerging Markets

**Commodities** — Precious Metals, Energy Futures, Agricultural

---

## Data Sources

QuantLab uses a **multi-source data layer** with cascading fallback, ensuring data availability even when individual providers are down.

### Architecture

```
DataSourceOrchestrator.fetch_prices(tickers, period)
    |
    +-- Tier 1: yfinance (DEFAULT - always available, no API key needed)
    +-- Tier 2: Alpha Vantage (opt-in, free API key)
    +-- Tier 3: Finnhub (opt-in, free API key)
    +-- Tier 4: Session cache (last successful fetch)
    +-- Tier 5: Error with diagnostic info
```

For macro economic data (treasury yields, CPI, unemployment, GDP):

```
fetch_macro_data(series_name, period)
    |
    +-- FRED (opt-in, free API key)
    +-- yfinance fallback
```

### Providers

| Provider | Data | API Key | Free Tier |
|----------|------|---------|-----------|
| **Yahoo Finance** (yfinance) | Prices, fundamentals, quotes | Not needed | Unlimited |
| **Alpha Vantage** | Prices, fundamentals, quotes | Required | 25 calls/day |
| **FRED** | Macro data (treasury yields, CPI, GDP, unemployment, VIX) | Required | Unlimited |
| **Finnhub** | Real-time quotes, company news | Required | 60 calls/min |

### Getting Free API Keys

- **Alpha Vantage**: [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
- **FRED**: [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- **Finnhub**: [finnhub.io](https://finnhub.io/)

### Configuration

API keys are entered in the sidebar under **Data Sources > Configure Data Providers**. Keys are stored in session state only (never written to disk). Each provider can be individually enabled/disabled via checkboxes.

---

## Sidebar Configuration

### Advanced Settings (5 sections)

**Monte Carlo & Simulation**
- Simulations: 100–2000 (default 500)
- Forecast Days: 30–365 (default 90)
- Confidence Level: 90 / 95 / 99 %
- Simulation Method: GBM or Behavioral Agent Model

**Portfolio Optimization**
- Bubble-Aware toggle with penalty factor slider (0.0–2.0)
- Benchmark: S&P 500, NASDAQ, Dow Jones, Russell 2000, or None
- Custom risk-free rate override
- Rebalancing frequency: None, Monthly, Quarterly, Annually

**ML & Analysis**
- ML Training Period: 1–5 years
- Clustering Method: K-Means or Gaussian Mixture
- Anomaly Sensitivity: 0.01–0.15

**Pairs Trading**
- Entry Z-Score: 1.0–3.0 (default 2.0)
- Exit Z-Score: 0.0–1.5 (default 0.5)
- Z-Score Lookback Window: 30–252 days (default 60)

**Enhanced Pairs Backtest**
- Method: Classic OLS, Kalman Filter, Regime-Adaptive (default Kalman)
- Walk-Forward Train Window: 126–504 days (default 252)
- Walk-Forward Test Window: 21–126 days (default 63)

**Macro Regime**
- Lookback Period: 1y, 2y, 5y, 10y (default 2y)

**Crypto On-Chain**
- Lookback: 30d, 90d, 180d, 365d (default 365d)

**Insider Trading**
- Lookback: 3 months, 6 months, 12 months (default 6 months)

**ML Top 10 Assets**
- Top N Assets: 5–20 (default 10)

**News Sentiment**
- Sentiment Model: Lexicon Only or Lexicon + NB Ensemble (default Ensemble)

**Smart Portfolio**
- Risk Tolerance: Conservative, Moderate, Aggressive (default Moderate)
- Max Weight per Asset: 0.05–0.50 (default 0.25)
- Monte Carlo Sims: 500–5000 (default 1000)

**Risk Decomposition**
- Attribution Window: 1 month, 3 months, 6 months, 1 year (default 3 months)

**Price Forecast**
- Forecast Horizon: 5 days, 10 days, 21 days (default 21 days)
- Forecast Train Window: 126–504 days (default 252)

**Earnings Predictor**
- Earnings Lookback: 2 years, 5 years, 10 years (default 5 years)

**Display & Export**
- Auto-Refresh toggle with configurable rate
- Chart height: 300–800 px
- Max table rows: 10–500
- Report section selector (multiselect)
- **Debug Mode** — shows full stack traces in error messages for troubleshooting

---

## Installation

```bash
git clone https://github.com/ranjithvijik/quantlab.git
cd quantlab
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Requirements

```
streamlit    yfinance     pandas       numpy
plotly       scipy        statsmodels  scikit-learn
ta           xlsxwriter   matplotlib   seaborn
tzdata       fpdf2
```

---

## Automated QA

QuantLab ships with **590+ tests** and a one-command QA orchestrator that runs the full suite and writes a formatted [`QA-REPORT.md`](QA-REPORT.md). Every push to `main` runs the suite automatically on Python 3.11 and 3.12 via GitHub Actions.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt && pip install pytest pytest-cov

# Run all tests and generate QA-REPORT.md
python run_tests.py
```

### Makefile Shortcuts

```bash
make qa            # full suite + QA-REPORT.md
make test          # pytest verbose, no report
make fast          # skip integration + frontend tests
make coverage      # HTML coverage + auto-open in browser
make t-valuation   # run only valuation tests
make t-portfolio   # run only portfolio tests
make t-options     # run only options tests
make t-frontend    # run only Streamlit UI tests
make lint          # flake8 check
make clean         # remove all test artifacts
make install       # install all deps
```

### Per-module flags

```bash
python run_tests.py --module valuation    # valuation models only
python run_tests.py --module portfolio    # portfolio optimization only
python run_tests.py --module options      # options pricing only
python run_tests.py --module bubble_ml    # bubble detection & ML
python run_tests.py --module risk_errors  # risk score & error handling
python run_tests.py --module integration  # end-to-end pipeline
python run_tests.py --module frontend     # Streamlit UI tests
python run_tests.py --fast               # skip integration & frontend
python run_tests.py --no-cov             # skip coverage (faster)
python run_tests.py --out my_qa.md       # custom output path
```

### CI/CD Pipeline

Every `git push` to `main` triggers the GitHub Actions workflow ([`.github/workflows/qa.yml`](.github/workflows/qa.yml)), running on Node.js 24-native action versions:

```
Your local dev machine
    └── git push ──────────────────────────────────────────┐
                                                           ▼
                                               GitHub Actions (CI)
                                               ├── Runs 590 tests (unit + frontend)
                                               ├── Python 3.11 & 3.12 matrix
                                               ├── Generates QA-REPORT.md
                                               ├── Commits report back to repo
                                               └── If all pass → Streamlit Cloud
                                                              auto-deploys app.py
```

The full `QA-REPORT.md` is posted to the [Actions summary tab](https://github.com/ranjithvijik/quantlab/actions) after every run — no download required.

### Test Coverage

| Layer | Module | What It Tests | Tests |
|-------|--------|---------------|-------|
| **Unit** | `unit/test_valuation.py` | CAPM, Beta (ddof=1), WACC, DCF guard, Fama-French, APT | 19 |
| **Unit** | `unit/test_portfolio.py` | 9 strategies × 3 invariants, Risk Parity, HRP, bubble penalty | 19 |
| **Unit** | `unit/test_options.py` | Black-Scholes (known value), put-call parity, all 5 Greeks, payoffs | 28 |
| **Unit** | `unit/test_bubble_ml.py` | BubbleDetector, GPH SE, RSI Wilder's EMA, MACD histogram, ML pipeline, sentiment | 37 |
| **Unit** | `unit/test_risk_and_errors.py` | Risk score, exception hierarchy, `handle_error` decorator, ticker parser | 36 |
| **Unit** | `unit/test_integration.py` | End-to-end: prices → portfolio → bubble → ML → options | 16 |
| **Unit** | `unit/test_new_modules.py` | Pairs trading, sector rotation, earnings, tail risk, cross-asset, ESG | 34 |
| **Unit** | `unit/test_data_providers.py` | Multi-source data layer, cascading fallback, Alpha Vantage, FRED, Finnhub | 69 |
| **Unit** | `unit/test_advanced_modules.py` | Enhanced pairs backtest, macro regime, crypto on-chain, insider trading, watchlist | 72 |
| **Unit** | `unit/test_ml_ranker.py` | ML Top 10 Assets ensemble scoring, feature engineering, model diagnostics | 41 |
| **Unit** | `unit/test_ml_insights.py` | News sentiment, smart portfolio, risk decomposition, price forecast, earnings predictor | 52 |
| **Frontend** | `frontend/test_frontend.py` | Streamlit widgets, presets, session state, dark mode, error handling | 143 |
| | **Total** | | **590** |

**Unit tests** run fully offline using deterministic synthetic data — no Yahoo Finance calls, no Streamlit server.

**Frontend tests** use `streamlit.testing.v1.AppTest` — drives the real Streamlit runtime without a browser. Covers all 22 Quick Presets, all Advanced Settings widgets, dark mode toggle, error states, and widget state persistence across reruns.

---

## Technical Documentation

A 50-page reference document — **[QuantLab-Technical-Documentation.pdf](QuantLab-Technical-Documentation.pdf)** — covers mathematical derivations, model assumptions, and implementation details for every module. Includes:

- Formula derivations with LaTeX-style equations (CAPM, Black-Scholes, HRP, GPH, Risk Parity)
- Code snippets showing the actual implementation
- Cross-module interaction maps (how valuation feeds portfolio, how bubble scores feed optimization, etc.)
- Purpose and relevance of each module in institutional investment workflows

---

## Architecture

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit 1.56, Plotly |
| Data | yfinance (prices, fundamentals, options chains, news) |
| Numerical | NumPy, SciPy, StatsModels |
| Machine Learning | scikit-learn (RF, GB, LR, K-Means, GMM, PCA, Isolation Forest, ComplementNB, TF-IDF, Ledoit-Wolf, TimeSeriesSplit) |
| Technical Indicators | `ta` library (RSI, MACD, Bollinger Bands, SMA, EMA) |
| Visualization | Plotly (interactive charts), Matplotlib/Seaborn (PDF/Excel exports) |
| Export | fpdf2 (PDF reports + slides), xlsxwriter (Excel) |
| Testing | pytest, streamlit.testing.v1 (590 tests, A+ grade) |
| CI/CD | GitHub Actions (Python 3.11 & 3.12, Node.js 24 native actions) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
