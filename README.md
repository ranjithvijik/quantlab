# QuantLab — Advanced Portfolio Analytics

[![Live App](https://img.shields.io/badge/Live%20App-rjquantlab.streamlit.app-blue?logo=streamlit)](https://rjquantlab.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![CI](https://github.com/ranjithvijik/quantlab/actions/workflows/qa.yml/badge.svg)](https://github.com/ranjithvijik/quantlab/actions/workflows/qa.yml)
[![Tests](https://img.shields.io/badge/tests-274%20passing-brightgreen)](QA-REPORT.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

QuantLab is a multi-asset quantitative research platform built with Streamlit. It covers the full workflow from data ingestion and technical analysis through portfolio optimization, bubble detection, Monte Carlo simulation, machine learning, and one-click report export — all running in the browser with no local setup required.

---

## Live Demo

**[https://rjquantlab.streamlit.app/](https://rjquantlab.streamlit.app/)**

---

## Features

### Portfolio Optimization
- **9 strategies:** Maximum Sharpe Ratio, Minimum Variance, Risk Parity, Minimum CVaR, Maximum Diversification, Kelly Criterion, Black-Litterman, Hierarchical Risk Parity (HRP), Equal Weight
- Bubble-aware optimization with adjustable penalty factor
- Efficient frontier visualization
- Backtest vs. equal-weight benchmark
- Strategy comparison table across five methods

### Bubble Detection
- **Metcalfe's Law analysis** — computes Market-to-Metcalfe Value (MMV) ratio using volume as a network-activity proxy
- **GPH long-memory estimator** — estimates fractional integration parameter *d*; *d* > 0.5 implies non-stationary bubble behavior
- **Kurtosis scoring** — fat-tailed return distributions flag elevated risk
- **Ljung-Box volatility clustering test** — detects ARCH effects in squared returns
- Granular composite bubble score (0–100 %) with Normal / Caution / High Risk regime labels

### Valuation Models
Five models per ticker, computed from live yfinance fundamental data:
1. **CAPM** — Expected return via the Security Market Line
2. **WACC** — Blended cost of equity and after-tax debt
3. **DCF** — Enterprise value from discounted free cash flows with terminal value
4. **Fama-French 3-Factor** — Alpha, SMB, and HML factor exposures
5. **APT** — Arbitrage Pricing Theory multi-factor expected return

### Technical Analysis
Indicators computed via the `ta` library:
- SMA 20, SMA 50
- EMA 12
- MACD (EMA 12 − EMA 26), Signal (EMA 9), Histogram
- RSI 14
- Bollinger Bands (SMA 20 ± 2σ)

### Monte Carlo Simulation
Two simulation engines selectable per run:
- **Geometric Brownian Motion** — standard log-normal price paths
- **Behavioral Agent Model** — agent-based model mixing Fundamentalist mean-reversion and Speculator momentum demand; parameters calibrated to each ticker's historical beta, autocorrelation, and sector

Outputs: median price, VaR, CVaR, confidence bands, regime labels (Bubble / Fair / Undervalued)

### Options Pricing
- Live options chain (calls & puts) from Yahoo Finance
- 3D implied volatility surface across strikes and expirations
- Black-Scholes pricing with Greeks (Delta, Gamma, Theta, Vega)
- Payoff diagrams for common strategies (Long Call, Long Put, Bull/Bear Spread, Straddle, etc.)

### Macro Dashboard
Live data via yfinance:
- Treasury yield curve (3-month, 5-year, 10-year, 30-year)
- VIX, S&P 500
- WACC scenario analysis

### Risk & Geopolitics Dashboard
- Cross-asset signals: VIX, US Dollar Index, Gold, WTI Crude Oil
- Yield curve spread (10Y − 3M) with inversion warnings
- Composite risk score (0–100) weighted across VIX, yield curve, and safe-haven demand

### ML Predictions
Three models trained on configurable historical windows (1–5 years):
- **Linear Regression** — interpretable baseline
- **Random Forest** — bagged ensemble with feature importance
- **Gradient Boosting** — boosted residual fitting

Features: returns, volatility, SMA ratios, volume ratio, RSI. Train/test split is chronological (last 20 % held out). Metrics: R², RMSE, MAE.

### ML Clustering & Regime Detection
- **Asset clustering** — K-Means or Gaussian Mixture across return/volatility/momentum features
- **PCA factor analysis** — 2D projection for cluster visualization
- **Market regime detection** — GMM-based Bull / Bear / Transition labeling
- **Anomaly detection** — Isolation Forest with tunable sensitivity

### Sentiment Analysis
- Fetches recent headlines from Yahoo Finance news
- Keyword-based sentiment scoring: Score = (N_positive − N_negative) / (N_positive + N_negative), range −1 to +1
- Bullish / Neutral / Bearish classification per article with aggregate summary

### Export System
All three formats generated on-demand in the Export tab:
- **PDF Report** — multi-page A4 research report (cover, executive summary, charts, valuation, portfolio, bubble detection, macro, ML, options, risk, clustering, sentiment)
- **Presentation Slides** — landscape PDF slide deck with charts and metrics tables
- **Excel Workbook** — formatted `.xlsx` with conditional formatting, embedded charts, and separate sheets for each analysis module

---

## 13 Tabs Overview

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
| 13 | **Export** | PDF report, presentation slides, Excel workbook download |

---

## Supported Asset Classes

Select from the sidebar dropdown; each class loads a default ticker set:

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

**Stocks**
- Tech Giants — AAPL MSFT GOOGL AMZN META NVDA
- Semiconductor — NVDA AMD INTC TSM AVGO QCOM
- EV & Clean Energy — TSLA RIVN LCID NIO ENPH FSLR
- FAANG+ — META AAPL AMZN NFLX GOOGL MSFT
- Banking & Finance — JPM BAC GS MS WFC C
- Healthcare & Pharma — JNJ PFE UNH ABBV MRK LLY
- Energy & Oil — XOM CVX COP SLB EOG OXY
- Mega Cap — AAPL MSFT GOOGL AMZN NVDA META TSLA BRK-B
- Dividend Aristocrats — JNJ PG KO PEP MMM ABT EMR
- Growth Stocks — SHOP SNOW CRWD DDOG NET PLTR
- Value Stocks — BRK-B JPM XOM JNJ PG BAC

**Crypto**
- Crypto Majors — BTC-USD ETH-USD SOL-USD XRP-USD ADA-USD AVAX-USD
- Crypto DeFi — UNI-USD AAVE-USD MKR-USD LINK-USD SNX-USD

**ETFs**
- Index ETFs — SPY QQQ DIA IWM VTI VOO
- Bond ETFs — TLT IEF SHY BND AGG LQD
- Sector ETFs — XLK XLF XLE XLV XLI XLP
- Commodity ETFs — GLD SLV USO UNG DBA DBC

**Forex**
- Forex Majors — EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCHF=X NZDUSD=X
- Forex Emerging — USDMXN=X USDBRL=X USDINR=X USDTRY=X USDZAR=X

**Commodities**
- Precious Metals — GC=F SI=F PA=F PL=F
- Energy Futures — CL=F NG=F RB=F HO=F
- Agricultural — ZC=F ZW=F ZS=F KC=F SB=F CC=F

---

## Sidebar Configuration

### Advanced Settings (4 sections)

**Monte Carlo & Simulation**
- Simulations: 100–2000 (default 500)
- Forecast Days: 30–365 (default 90)
- Confidence Level: 90 / 95 / 99 %
- Simulation Method: GBM or Behavioral Agent Model

**Portfolio Optimization**
- Bubble-Aware toggle with penalty factor slider
- Benchmark: S&P 500, NASDAQ, Dow Jones, Russell 2000, or None
- Custom risk-free rate override
- Rebalancing frequency: None, Monthly, Quarterly, Annually

**ML & Analysis**
- ML Training Period: 1–5 years
- Clustering Method: K-Means or Gaussian Mixture
- Anomaly Sensitivity: 0.01–0.15

**Display & Export**
- Auto-Refresh toggle with configurable rate
- Chart height: 300–800 px
- Max table rows: 10–500
- Report section selector (multiselect)

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
streamlit
yfinance
pandas
numpy
plotly
scipy
statsmodels
scikit-learn
ta
xlsxwriter
matplotlib
seaborn
tzdata
fpdf2
```

---

## Automated QA

QuantLab ships with **274 tests** and a one-command QA orchestrator that runs the full suite and writes a formatted [`QA-REPORT.md`](QA-REPORT.md). Every push to `main` runs the suite automatically on Python 3.11 and 3.12 via GitHub Actions.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt && pip install pytest pytest-cov

# Run all 274 tests and generate QA-REPORT.md
python run_tests.py
```

### Makefile Shortcuts

```bash
make qa            # full suite + QA-REPORT.md
make test          # pytest verbose, no report
make fast          # skip integration + frontend tests
make coverage      # HTML coverage + auto-open in browser
make t-portfolio   # run only portfolio tests
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
python run_tests.py --fast               # all except integration & frontend
python run_tests.py --no-cov             # skip coverage (faster)
```

### CI/CD Pipeline

Every `git push` to `main` triggers the GitHub Actions workflow ([`.github/workflows/qa.yml`](.github/workflows/qa.yml)):

```
Your local dev machine
    └── git push ──────────────────────────────────────────┐
                                                           ▼
                                               GitHub Actions (CI)
                                               ├── Runs 274 tests (unit + frontend)
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
| **Unit** | `unit/test_portfolio.py` | 9 strategies × 3 invariants, Risk Parity, HRP, bubble penalty | 43 |
| **Unit** | `unit/test_options.py` | Black-Scholes (known value), put-call parity, all 5 Greeks, payoffs | 28 |
| **Unit** | `unit/test_bubble_ml.py` | BubbleDetector, GPH SE, RSI Wilder's EMA, MACD histogram, ML pipeline, sentiment | 37 |
| **Unit** | `unit/test_risk_and_errors.py` | Risk score, exception hierarchy, `handle_error` decorator, ticker parser | 36 |
| **Unit** | `unit/test_integration.py` | End-to-end: prices → portfolio → bubble → ML → options | 16 |
| **Frontend** | `frontend/test_frontend.py` | Streamlit widgets, presets, session state, dark mode, error handling | 95 |
| | **Total** | | **274** |

**Unit tests** run fully offline using deterministic synthetic data — no Yahoo Finance calls, no Streamlit server.

**Frontend tests** use `streamlit.testing.v1.AppTest` to drive the real Streamlit runtime — no browser required. Covers all 22 Quick Presets, Advanced Settings sliders/toggles, dark mode, error states, and widget state persistence.

---

## Technical Documentation

A 50-page reference document — **QuantLab-Technical-Documentation.pdf** — covers the mathematical derivations, model assumptions, and implementation details for every module: portfolio optimization, bubble detection, valuation models, Monte Carlo engines, ML pipelines, and the export system.

---

## Architecture

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit, Plotly |
| Data | yfinance (price history, fundamentals, options chains, news) |
| Numerical | NumPy, SciPy, StatsModels |
| Machine Learning | scikit-learn (RF, GB, LR, KMeans, GMM, PCA, IsolationForest) |
| Technical Indicators | `ta` library |
| Visualization | Plotly (interactive), Matplotlib / Seaborn (export) |
| Export | fpdf2 (PDF), xlsxwriter (Excel) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
