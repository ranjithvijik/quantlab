# QuantLab: Advanced Portfolio Analytics & Bubble Detection Platform

**QuantLab** is a comprehensive web-based platform for advanced portfolio analytics, risk management, and market bubble detection. Built with Streamlit and powered by financial data from Yahoo Finance, QuantLab enables institutional-grade analysis with an intuitive, interactive UI.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Platform Guide](#platform-guide)
  - [Dashboard Overview](#dashboard-overview)
  - [Core Functionality](#core-functionality)
- [UI Components & Usage](#ui-components--usage)
  - [Sidebar Configuration](#sidebar-configuration)
  - [Portfolio Optimization](#portfolio-optimization)
  - [Bubble Detection](#bubble-detection)
  - [Technical Analysis](#technical-analysis)
  - [Risk Analytics](#risk-analytics)
  - [Reporting & Export](#reporting--export)
- [How to Use the UI for Analysis](#how-to-use-the-ui-for-analysis)
  - [Getting Started Workflow](#getting-started-workflow)
  - [Complete Analysis Workflows](#complete-analysis-workflows)
  - [Analysis Use Cases](#analysis-use-cases)
- [Advanced Features](#advanced-features)
- [Technical Architecture](#technical-architecture)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### 📊 **Portfolio Analytics**
- **Multi-asset portfolio construction** with real-time market data
- **Real-time price tracking** and portfolio valuation
- **Performance analytics** with comprehensive metrics
- **Correlation analysis** and asset relationship visualization

### 💼 **Advanced Portfolio Optimization**
Multiple optimization strategies including:
- **Maximum Sharpe Ratio** - optimal risk-adjusted returns
- **Minimum Variance** - lowest portfolio risk
- **Risk Parity** - equal risk contribution from each asset
- **Minimum CVaR** - tail-risk minimization (Conditional Value-at-Risk)
- **Maximum Diversification** - maximize risk-adjusted diversification
- **Kelly Criterion** - optimal position sizing with leverage control
- **Black-Litterman** - incorporate market views into optimization
- **Hierarchical Risk Parity (HRP)** - machine learning-based allocation
- **Equal Weight** - baseline allocation strategy

### 🔍 **Bubble Detection Engine**
Multi-factor bubble scoring system analyzing:
- **Price momentum** and trend deviation
- **Valuation metrics** (P/E, P/B ratios)
- **Technical indicators** (RSI, MACD, Bollinger Bands)
- **Statistical volatility** analysis
- **Volume anomalies**
- **Market sentiment** indicators
- **Historical tail risk** assessment

### 💰 **Valuation Models**
- **DCF (Discounted Cash Flow)** analysis with 5-year projections
- **CAPM (Capital Asset Pricing Model)** return calculation
- **Fama-French 3-Factor Model** for expected returns
- **APT (Arbitrage Pricing Theory)** multi-factor analysis
- **WACC (Weighted Average Cost of Capital)** computation

### 📈 **Technical Analysis**
- **150+ Technical Indicators** including:
  - Moving averages (SMA, EMA, WMA)
  - Momentum indicators (RSI, MACD, Stochastic)
  - Volatility measures (ATR, Bollinger Bands, Keltner Channels)
  - Volume analysis (OBV, VWAP)
  - Trend indicators (ADX, AROON)
  - Support/resistance levels

### ⚠️ **Risk Management**
- **Value at Risk (VaR)** calculation (95%, 99% confidence)
- **Expected Shortfall (CVaR)** analysis
- **Maximum Drawdown** tracking
- **Stress Testing** capabilities
- **Correlation breakdown** analysis
- **Bubble burst impact estimation**
- **Portfolio concentration** metrics

### 📊 **Advanced Analytics**
- **Efficient Frontier** visualization with optimal allocations
- **Correlation heatmaps** between assets
- **Rolling performance** metrics (daily, weekly, monthly)
- **Principal Component Analysis (PCA)** for factor extraction
- **Autocorrelation analysis** (ACF/PACF plots)
- **Spectral analysis** (periodogram)
- **Hierarchical clustering** of assets

### 📁 **Reporting & Export**
- **Comprehensive PDF reports** with analysis summaries
- **Excel workbooks** with detailed breakdowns
- **Real-time data export** in multiple formats
- **Custom report generation** with user selections
- **Performance attribution** analysis

### 🎯 **Interactive Dashboards**
- **Live market monitoring** with auto-refresh capabilities
- **Real-time price updates** with custom refresh intervals
- **Interactive Plotly charts** with hover details
- **Dynamic filtering** and sorting
- **Customizable watchlists**

---

## Installation

### Prerequisites
- **Python 3.8 or higher**
- **pip** (Python package manager)

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/ranjithvijik/quantlab.git
cd quantlab
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import streamlit, yfinance, pandas, plotly; print('All dependencies installed successfully!')"
```

---

## Quick Start

### Running the Application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501` in your default browser.

### First-Time Setup

1. **Enter Ticker Symbols**
   - In the sidebar, enter stock/ETF tickers separated by commas
   - Example: `AAPL,GOOGL,MSFT,BRK.B,SPY`

2. **Select Date Range**
   - Choose start and end dates for historical analysis
   - Default: Last 2 years of data

3. **Configure Analysis Parameters**
   - Set risk-free rate (auto-fetches from Treasury)
   - Enable/disable bubble detection
   - Choose analysis frequency (Daily/Weekly/Monthly)

4. **Initiate Analysis**
   - Click "Run Full Analysis" to process data
   - Platform will fetch prices and run all calculations

---

## Platform Guide

### Dashboard Overview

The main interface is organized into **7 key tabs**:

```
QuantLab Dashboard
├── Market Overview
├── Technical Analysis
├── Bubble Detection
├── Portfolio Optimization
├── Risk Analytics
├── Valuation Models
└── Report Generator
```

---

## UI Components & Usage

### 1️⃣ Sidebar Configuration

**Location:** Left sidebar of the application

**Components:**

#### Asset Selection
```
📌 Ticker Input
├─ Enter tickers: AAPL, MSFT, GOOGL
├─ Format: Comma-separated symbols
└─ Symbols are case-insensitive
```

**How to Use:**
1. Click the text input field labeled "Enter tickers (comma-separated)"
2. Type stock symbols: `AAPL, MSFT, VOO, AGG`
3. Press Enter or click outside the field
4. Maximum 20 assets recommended for performance

#### Date Range Selection
```
📅 Date Range
├─ Start Date: [Date Picker]
├─ End Date: [Date Picker]
└─ Preset Options: 1M, 3M, 6M, 1Y, 2Y, 5Y, All
```

**How to Use:**
1. Click the start date calendar icon
2. Navigate to desired month/year
3. Select the date
4. Repeat for end date
5. Minimum 60 days of data required for analysis

#### Risk-Free Rate
```
📊 Risk-Free Rate
├─ Auto-fetch from Treasury (TNX index)
├─ Manual override option
└─ Used for: CAPM, Sharpe Ratio, WACC calculations
```

**How to Use:**
1. Leave as "Auto-fetch" to get current 10-year Treasury yield
2. Or enter manual rate (e.g., 0.045 for 4.5%)
3. Updates on each run

#### Analysis Parameters
```
⚙️ Configuration
├─ Enable Bubble Detection: [Toggle]
├─ Auto-refresh Interval: [Slider] (30s - 5m)
├─ Data Cache TTL: [Slider] (30s - 1h)
└─ Portfolio Rebalance Frequency: [Dropdown]
```

**How to Use:**
1. Toggle bubble detection on/off
2. Set auto-refresh for live monitoring
3. Adjust cache settings based on latency needs
4. Choose rebalancing strategy

#### Action Buttons
```
🎯 Actions
├─ [Run Full Analysis] - Execute all calculations
├─ [Clear Cache] - Reset cached data
├─ [Export Settings] - Save configuration
└─ [Load Preset] - Use predefined portfolios
```

---

### 2️⃣ Market Overview Tab

**Purpose:** Monitor real-time portfolio status and market conditions

**Key Components:**

#### Portfolio Status Cards
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 Portfolio Value: $314,303.94
📊 Daily Change: +2.45% ($7,500.23)
📉 YTD Return: +12.34%
⏰ Last Updated: 2025-12-02 15:47 EST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**How to Use:**
- View live portfolio metrics
- Monitor daily performance
- Check data freshness timestamp

#### Asset Performance Table
```
| Ticker | Price    | Change  | % Change | Holdings | Value     |
|--------|----------|---------|----------|----------|-----------|
| AAPL   | $245.32  | +3.45   | +1.43%   | 50       | $12,266   |
| MSFT   | $430.15  | -2.10   | -0.49%   | 30       | $12,904.50|
| ...    | ...      | ...     | ...      | ...      | ...       |
```

**How to Use:**
1. Sort columns by clicking headers
2. Filter by ticker using search box
3. Click rows to see detailed asset analysis
4. Export table using "Export Data" button

#### Market Heatmap
**Visual:** Color-coded performance grid

**Colors:**
- 🟢 Green = Positive performance (>5%)
- 🟡 Yellow = Neutral (±5%)
- 🔴 Red = Negative (<-5%)

**How to Use:**
- Hover over cells for exact percentages
- Click to drill down into asset details
- Use for quick portfolio health assessment

---

### 3️⃣ Technical Analysis Tab

**Purpose:** Perform in-depth technical chart analysis with 150+ indicators

#### Chart Controls
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Asset Selection: [AAPL ▼]
📈 Chart Type: [Candlestick ▼]
   Options: Candlestick, Line, OHLC
🎨 Time Frame: [Daily ▼]
   Options: Daily, Weekly, Monthly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**How to Use:**
1. **Select Asset:** Dropdown to choose individual ticker
2. **Choose Chart Type:**
   - Candlestick: OHLC visualization (recommended)
   - Line: Closing price only
   - OHLC: Box plots of open/high/low/close
3. **Set Timeframe:** Daily, weekly, or monthly aggregation

#### Technical Indicators

**Available Indicators (150+):**

**Trend Indicators**
- **SMA (Simple Moving Average):** Select period (20, 50, 200)
  - Red line = short-term trend
  - Blue line = long-term trend
  - Crossovers indicate trend changes
- **EMA (Exponential Moving Average):** Faster reaction to price changes
- **WMA (Weighted Moving Average):** Recent prices weighted higher
- **ADX (Average Directional Index):** Trend strength (0-100)

**Momentum Indicators**
- **RSI (Relative Strength Index):** Overbought/oversold (0-100)
  - >70 = Overbought (potential sell signal)
  - <30 = Oversold (potential buy signal)
- **MACD (Moving Average Convergence Divergence):**
  - Signal line crossover = trading signals
  - Histogram = momentum strength
- **Stochastic Oscillator:** Similar to RSI, different calculation

**Volatility Indicators**
- **Bollinger Bands:** Price volatility envelope
  - Price touches upper band = possibly overbought
  - Price touches lower band = possibly oversold
- **ATR (Average True Range):** Volatility magnitude
- **Keltner Channels:** Volatility-adjusted support/resistance

**Volume Indicators**
- **OBV (On-Balance Volume):** Cumulative volume analysis
- **VWAP (Volume-Weighted Average Price):** Average price weighted by volume
- **Volume Rate of Change:** Volume momentum

**How to Use:**
1. Select indicator from multi-select dropdown
2. Set indicator parameters (periods, standard deviations, etc.)
3. Chart updates automatically
4. Hover over chart for exact values
5. Click "Save Chart" to export PNG

**Example Setup:**
```
Asset: AAPL
Indicators: SMA (50), SMA (200), RSI (14), MACD
Timeframe: Daily
Analysis: SMA 50 below SMA 200 = bearish trend
          RSI = 35 = oversold, potential reversal
          MACD crosses above signal = bullish momentum
```

#### Technical Analysis Plots

**ACF/PACF Analysis**
- **Purpose:** Detect autocorrelation in returns
- **Interpretation:** Significant spikes = mean reversion opportunities
- **How to Use:** Check for patterns in lags 1-20

**Periodogram Analysis**
- **Purpose:** Identify cyclical patterns in price data
- **Display:** Frequency spectrum of returns
- **How to Use:** Peaks indicate dominant cycles (e.g., quarterly patterns)

---

### 4️⃣ Bubble Detection Tab

**Purpose:** Identify and score market bubbles with multi-factor analysis

#### Bubble Score Display
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bubble Score Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
| Ticker | Score | Risk Level | Status      |
|--------|-------|-----------|-------------|
| AAPL   | 0.34  | 🟢 Low    | Not Bubbled |
| NVDA   | 0.78  | 🔴 High   | Bubble Risk |
| TSLA   | 0.62  | 🟡 Medium | Monitor     |
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Bubble Score Interpretation:**
- **0.0 - 0.3:** 🟢 Safe (Normal valuation)
- **0.3 - 0.5:** 🟡 Caution (Slightly elevated)
- **0.5 - 0.7:** 🟠 Warning (Elevated bubble risk)
- **0.7 - 1.0:** 🔴 Critical (High bubble probability)

#### Bubble Factor Breakdown
```
Bubble Detection Factors for NVDA (Score: 0.78)

┌─────────────────────────────────────────┐
│ Factor              | Weight | Score    │
├─────────────────────────────────────────┤
│ Price Momentum      | 0.20   | 0.92 ↑↑↑│
│ Valuation Metrics   | 0.25   | 0.85 ↑↑ │
│ Technical Signals   | 0.20   | 0.72 ↑  │
│ Volatility Stress   | 0.15   | 0.65    │
│ Volume Anomaly      | 0.10   | 0.45    │
│ Historical Tail Risk| 0.10   | 0.55    │
└─────────────────────────────────────────┘
```

**How to Interpret:**
1. **Price Momentum:** How fast price is rising relative to history
   - >0.8 = Extreme momentum (bubble risk)
   
2. **Valuation Metrics:** P/E, P/B ratios vs. historical averages
   - >0.8 = Trading above historical average (caution)
   
3. **Technical Signals:** RSI, MACD, Bollinger Band positions
   - >0.7 = Extreme technical positioning

4. **Volatility Stress:** Recent volatility vs. baseline
   - >0.7 = Elevated (potential instability)

5. **Volume Anomaly:** Unusual trading volume
   - >0.6 = Elevated volume (check for panic)

6. **Historical Tail Risk:** Probability of large downside move
   - >0.6 = Higher historical tail risk

#### Bubble Burst Impact Estimate
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estimated Decline in Bubble Burst
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NVDA: Estimated Loss: -42.3%
TSLA: Estimated Loss: -35.7%
AAPL: Estimated Loss: -18.2%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**How to Use:**
1. Review bubble scores for each asset
2. Assets with scores >0.6 should be monitored
3. Check estimated decline impact
4. Adjust portfolio weights accordingly
5. Use "Bubble-Aware Optimization" in Portfolio Optimization tab

#### Risk Mitigation Strategies
```
Recommended Actions for High-Bubble Assets:

🔴 NVDA (Score: 0.78)
├─ Reduce position size
├─ Set stop-loss at -15% to -20%
├─ Consider hedging with put options
├─ Monitor weekly for changes
└─ Exit if score exceeds 0.85

🟡 TSLA (Score: 0.62)
├─ Monitor closely
├─ Reduce if score increases
├─ Trim 5-10% on rallies
└─ Hold core position if fundamental confidence high
```

---

### 5️⃣ Portfolio Optimization Tab

**Purpose:** Generate optimal portfolio allocations based on selected strategy

#### Optimization Strategy Selection
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Choose Strategy: [Maximum Sharpe Ratio ▼]

Available Options:
├─ Maximum Sharpe Ratio (Recommended for most)
├─ Minimum Variance (Conservative)
├─ Risk Parity (Equal risk contribution)
├─ Minimum CVaR (Tail-risk focused)
├─ Maximum Diversification (Maximize diversification)
├─ Kelly Criterion (Aggressive sizing)
├─ Black-Litterman (Incorporate market views)
├─ Hierarchical Risk Parity (ML-based)
└─ Equal Weight (Baseline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Strategy Selection Guide:**

| Strategy | Use Case | Risk | Return Potential |
|----------|----------|------|-----------------|
| **Max Sharpe** | Best risk-adjusted returns | Medium | High |
| **Min Variance** | Conservative, low volatility | Low | Medium |
| **Risk Parity** | Balanced risk across assets | Medium | Medium |
| **Min CVaR** | Tail-risk protection | Low | Medium |
| **Max Diversification** | Maximize diversification benefit | Medium | Medium-High |
| **Kelly Criterion** | Optimal position sizing | High | Very High |
| **Black-Litterman** | Incorporate market views | Medium | High |
| **HRP** | ML-optimized allocation | Medium | High |
| **Equal Weight** | Simple baseline | Medium | Medium |

**How to Use:**
1. Select strategy from dropdown
2. Configure strategy-specific parameters
3. For **Bubble-Aware Optimization:**
   - Toggle on "Bubble-Aware Optimization"
   - Adjust "Bubble Penalty Factor" (0.0 - 1.0)
     - Lower value = less penalty for bubbles
     - Higher value = more conservative (reduces bubble-prone assets)
4. Click "Optimize Portfolio"

#### Strategy-Specific Parameters

**Maximum Sharpe Ratio**
```
No additional parameters - uses default risk-free rate
```

**Kelly Criterion**
```
Leverage Limit: [Slider] 0.5x to 2.0x
├─ 1.0x = Long-only (no leverage)
├─ 1.5x = 50% leverage
└─ 2.0x = 100% leverage (double capital at risk)
```

**Black-Litterman**
```
Market Cap Weighting: [Auto-calculated from holdings]
Incorporate Views: [Toggle]
├─ If ON: Configure asset-specific expected returns
└─ If OFF: Use market consensus views
```

#### Optimization Results

**Portfolio Allocation Display**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optimized Portfolio Allocation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Pie Chart Visualization**
- **Display:** Donut chart showing allocation percentages
- **Hover:** Shows exact percentage and dollar value
- **Colors:** Distinct colors for each asset
- **Click:** Drill-down to asset-specific details

**Allocation Table**
```
| Ticker | Allocation | Dollar Value | Rebalance Action |
|--------|-----------|--------------|------------------|
| AAPL   | 25.4%     | $79,827.60   | Hold             |
| MSFT   | 22.1%     | $69,421.22   | Buy (+2.1%)      |
| VOO    | 30.5%     | $95,860.97   | Sell (-5.3%)     |
| AGG    | 22.0%     | $69,147.87   | Buy (+0.2%)      |
```

#### Portfolio Metrics Display
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Row 1]
├─ Expected Return: 8.45% (annually)
├─ Volatility: 12.3% (standard deviation)
├─ Sharpe Ratio: 0.61 (risk-adjusted return)
└─ Sortino Ratio: 0.87 (downside risk focus)

[Row 2]
├─ Max Drawdown: -18.5% (worst historical decline)
├─ Calmar Ratio: 0.46 (return/max drawdown)
├─ CVaR (95%): -2.45% (worst 5% case loss/day)
└─ Downside Deviation: 8.2% (downside volatility)

[Row 3]
├─ Diversification Ratio: 1.34x (benefit of diversification)
└─ Effective Number of Assets: 3.2 (concentration metric)
```

**Metric Interpretations:**

- **Expected Return:** Average annual return expectation
- **Volatility:** Standard deviation of returns (risk measure)
- **Sharpe Ratio:** Higher = better risk-adjusted returns
  - >1.0 = Excellent
  - 0.5-1.0 = Good
  - <0.5 = Poor
- **Sortino Ratio:** Like Sharpe but only penalizes downside
- **Max Drawdown:** Largest peak-to-trough decline
- **Calmar Ratio:** Return relative to drawdown (higher better)
- **CVaR (95%):** 95% confidence worst-case daily loss
- **Diversification Ratio:** Measures diversification effectiveness
- **Effective N Assets:** Equivalent number of non-correlated assets

#### Efficient Frontier Visualization

**Option:** Enable "Show Efficient Frontier"

```
Visual: Scatter plot with curve

Y-axis: Return (%)
X-axis: Volatility (%)

🟢 Green dots: Efficient portfolios
🔵 Blue dot: Your optimized portfolio (highest Sharpe)
🔴 Red dots: Random portfolios (for comparison)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**How to Interpret:**
- **Efficient Frontier:** No portfolio to the upper left
- **Your Portfolio:** Should be on or near the frontier
- **Higher on curve:** Higher return, higher risk
- **Left side:** Conservative portfolios
- **Right side:** Aggressive portfolios

#### Backtest Results

**Option:** Enable "Show Backtest Results"

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Historical Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Line Chart: Portfolio Cumulative Returns Over Time
├─ X-axis: Historical dates
├─ Y-axis: Growth of $10,000
├─ Blue line: Portfolio value
└─ Gray area: Confidence band
```

**Additional Backtest Metrics:**
```
├─ Total Return: 125.3% (since start date)
├─ Annualized Return: 14.2% (per year)
├─ Number of Years: 5 years
├─ Win Rate: 62.4% (positive days/total days)
├─ Average Win: 0.87%
├─ Average Loss: -0.65%
└─ Profit Factor: 1.34 (avg win/avg loss)
```

---

### 6️⃣ Risk Analytics Tab

**Purpose:** Comprehensive risk analysis and stress testing

#### Risk Metrics Dashboard
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Value at Risk (VaR) Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

95% VaR (1-Day): -$4,285 (1.36% of portfolio)
└─ Interpretation: 95% chance daily loss < $4,285

99% VaR (1-Day): -$7,150 (2.27% of portfolio)
└─ Interpretation: 99% chance daily loss < $7,150

Expected Shortfall (CVaR) 95%: -$5,620 (1.79%)
└─ Interpretation: Average loss in worst 5% scenarios
```

**How to Use:**
1. Review VaR levels to understand downside risk
2. 1-day VaR = typical daily volatility risk
3. Compare across different confidence levels
4. Use for position sizing and risk limits

#### Correlation Analysis
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Asset Correlation Matrix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        AAPL    MSFT    VOO     AGG
AAPL    1.00    0.78    0.92    0.15
MSFT    0.78    1.00    0.85    0.12
VOO     0.92    0.85    1.00    0.20
AGG     0.15    0.12    0.20    1.00
```

**Heatmap Display:**
- 🔴 Red = High positive correlation (0.75 - 1.0)
- 🟡 Yellow = Moderate correlation (0.5 - 0.75)
- 🟢 Green = Low correlation (0 - 0.5)
- 🔵 Blue = Negative correlation (-1 - 0)

**How to Use:**
1. Look for low/negative correlations for diversification
2. High correlations (<0.3) = good diversifiers
3. AGG-AAPL correlation (0.15) = excellent hedge
4. VOO-AAPL correlation (0.92) = high redundancy

#### Drawdown Analysis
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Historical Drawdown Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Maximum Drawdown: -18.5% (Feb 2020 - Apr 2020)
├─ Duration: 79 days
├─ Recovery Time: 145 days
└─ Recovery: Yes (returned to peak)

Average Drawdown: -8.2%
Median Drawdown: -6.1%
Drawdown Standard Deviation: 4.3%
```

**Visualization:** Area chart showing cumulative drawdown over time

**How to Use:**
1. Understand worst historical loss scenario
2. Plan for recovery time
3. Assess psychological tolerance
4. Set stop-loss levels based on this

#### Stress Testing
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stress Test Scenarios
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scenario 1: Historical Crisis (2008-09)
├─ Portfolio Impact: -34.2%
├─ AAPL: -42%
├─ MSFT: -38%
└─ AGG: -8% (diversification helped)

Scenario 2: Sudden Rate Spike (+200bps)
├─ Portfolio Impact: -12.5%
├─ AGG: -18% (bond duration risk)
└─ AAPL/MSFT: -8% (equity moderates)

Scenario 3: Tech Collapse (-50%)
├─ Portfolio Impact: -28.3%
├─ AAPL: -50%
├─ MSFT: -50%
└─ AGG/VOO: Limited impact (diversification)

Scenario 4: Market Crash (-25%)
├─ Portfolio Impact: -22.1%
├─ VOO: -25%
└─ AGG: +2% (negative correlation)
```

**How to Use:**
1. Review portfolio behavior in each scenario
2. Identify concentrated risks
3. Adjust allocation if uncomfortable
4. Use for board/stakeholder communication

---

### 7️⃣ Valuation Models Tab

**Purpose:** Fundamental valuation analysis using institutional methods

#### DCF (Discounted Cash Flow) Analysis
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DCF Valuation Summary - AAPL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current Stock Price: $245.32
Calculated Fair Value: $218.45
Valuation Gap: +12.3% (Overvalued)
Recommendation: ⚠️ HOLD / REDUCE

DCF Components:
├─ 5-Year Projected FCF (PV): $48.2B
├─ Terminal Value (PV): $312.1B
├─ Total Enterprise Value: $360.3B
├─ Less: Net Debt: $45.2B
├─ Equity Value: $315.1B
├─ Per Share Value: $218.45
└─ Implied Margin of Safety: -12.3%
```

**How to Interpret:**
- **Fair Value > Current:** Undervalued (Buy signal)
- **Fair Value < Current:** Overvalued (Sell signal)
- **Margin of Safety:** Discount to intrinsic value
  - >15% = Good margin
  - 0-15% = Adequate
  - Negative = Overvalued

**Assumptions Used:**
- Growth Rate (5-year): 5% annually
- Terminal Growth Rate: 2% (long-term GDP growth)
- WACC: Calculated from capital structure

#### CAPM (Capital Asset Pricing Model)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected Return Calculation (CAPM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk-Free Rate (Rf): 4.50%
Beta (β): 1.24
Market Risk Premium: 8.00%

Expected Return = Rf + β(Rm - Rf)
                = 4.50% + 1.24(8.00%)
                = 4.50% + 9.92%
                = 14.42%

Interpretation:
├─ CAPM suggests 14.42% annual return
├─ Higher than risk-free rate (premium for risk)
├─ Beta > 1.0 (more volatile than market)
└─ Investment justified if return expectations > 14.42%
```

**How to Use:**
1. Compare expected return (14.42%) to historical returns
2. If historical return < CAPM → Undervalued
3. If historical return > CAPM → Overvalued
4. Use for required rate of return in DCF

#### Fama-French 3-Factor Model
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected Return (Fama-French 3-Factor)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk-Free Rate: 4.50%

Factor 1: Market Risk
├─ Beta: 1.24
├─ Premium: 8.00%
└─ Contribution: +9.92%

Factor 2: Size Premium (SMB)
├─ Beta: -0.10 (large-cap premium to small)
├─ Premium: 2.00%
└─ Contribution: -0.20%

Factor 3: Value Premium (HML)
├─ Beta: -0.20 (growth preferred over value)
├─ Premium: 4.00%
└─ Contribution: -0.80%

Total Expected Return: 4.50% + 9.92% - 0.20% - 0.80% = 13.42%
```

**Comparison to CAPM:**
- CAPM (1-factor): 14.42%
- FF3-Factor: 13.42%
- Difference: More refined estimate with size/value adjustments

#### APT (Arbitrage Pricing Theory)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected Return (Arbitrage Pricing Theory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Risk-Free Rate: 4.50%

Economic Factors:
├─ Market Risk
│  ├─ Beta: 1.20
│  ├─ Premium: 8.00%
│  └─ Contribution: +9.60%
│
├─ Volatility Risk
│  ├─ Beta: 0.20
│  ├─ Premium: 3.00%
│  └─ Contribution: +0.60%
│
└─ Momentum Risk
   ├─ Beta: 0.30
   ├─ Premium: 2.00%
   └─ Contribution: +0.60%

Total Expected Return: 4.50% + 9.60% + 0.60% + 0.60% = 15.30%
```

**Multi-Factor Advantage:**
- More factors captured = more precise estimate
- Accounts for momentum and volatility separately
- Higher estimate reflects economic momentum premium

#### WACC (Weighted Average Cost of Capital)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cost of Capital Structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Market Capitalization (Equity): $2.800T
Total Debt: $120B
Total Enterprise Value: $2.920T

Weights:
├─ Equity (E/V): 95.9%
└─ Debt (D/V): 4.1%

Cost of Equity (CAPM): 14.42%
Cost of Debt (Pre-Tax): 3.50%
Tax Rate: 21.00%
After-Tax Cost of Debt: 3.50% × (1 - 0.21) = 2.77%

WACC = (95.9% × 14.42%) + (4.1% × 2.77%)
     = 13.81% + 0.11%
     = 13.92%

Interpretation:
├─ Minimum return to satisfy both equity and debt holders
├─ Used as discount rate in DCF
└─ 13.92% = hurdle rate for capital projects
```

---

### 8️⃣ Report Generator Tab

**Purpose:** Generate comprehensive reports for analysis and distribution

#### Report Configuration
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report Builder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Report Type: [PDF Report ▼]
├─ PDF Report (Professional formatting)
├─ Excel Workbook (Detailed data)
└─ Summary Report (Quick overview)

Include Sections:
☑️ Executive Summary
☑️ Portfolio Overview
☑️ Asset-by-Asset Analysis
☑️ Technical Analysis Charts
☑️ Bubble Risk Assessment
☑️ Portfolio Optimization Results
☑️ Risk Metrics & Stress Tests
☑️ Valuation Analysis
☑️ Historical Performance
☑️ Recommendations

Report Title: [User Customizable]
Report Date: [Auto-filled: 2025-12-02]
```

**How to Use:**
1. Select report type (PDF/Excel/Summary)
2. Choose sections to include
3. Customize report title
4. Click "Generate Report"
5. Download or email directly

#### Report Contents

**Executive Summary**
```
Investment Overview
├─ Portfolio Value: $314,303.94
├─ Composition: 4 assets
├─ Performance: +12.34% YTD
├─ Risk Level: Medium
└─ Key Recommendation: Rebalance toward bonds

Top 3 Holdings
├─ VOO (S&P 500 ETF): 32.1% / $100,814
├─ AAPL (Apple Inc.): 25.4% / $79,827
└─ MSFT (Microsoft): 22.1% / $69,421

Key Metrics
├─ Expected Return: 8.45%
├─ Volatility: 12.3%
├─ Sharpe Ratio: 0.61
└─ Max Drawdown: -18.5%

Risk Summary
├─ Bubble Risk Score: 0.42 (Medium)
├─ 95% Daily VaR: -$4,285
└─ Primary Risk: Tech concentration
```

**Asset-by-Asset Analysis**

For each asset:
```
AAPL (Apple Inc.)
├─ Current Price: $245.32
├─ Allocation: 25.4% ($79,827)
├─ 52-Week High: $254.88 (+4.0%)
├─ 52-Week Low: $164.75 (+48.8%)
├─ YTD Return: +28.5%
├─ Dividend Yield: 0.42%
├─ P/E Ratio: 34.2
├─ DCF Fair Value: $218.45 (-10.9%)
├─ Bubble Score: 0.58 (Monitor)
└─ Recommendation: HOLD / Reduce on rallies
```

**Technical Analysis Charts**
- Candlestick charts with indicators
- RSI, MACD, Moving averages
- Support/resistance levels identified

**Bubble Risk Assessment**
- Bubble scores for all holdings
- Risk factor breakdown
- Estimated decline in crisis
- Hedging recommendations

**Portfolio Optimization**
- Current vs. optimized allocations
- Rebalancing recommendations
- Projected impact on metrics

---

## How to Use the UI for Analysis

This section provides step-by-step workflows for conducting different types of financial analysis using QuantLab.

### Getting Started Workflow

**Step 1: Initial Setup (5 minutes)**
```
1. Open QuantLab (streamlit run app.py)
2. In LEFT SIDEBAR:
   ├─ Enter tickers: AAPL, VOO, AGG
   ├─ Date range: Last 2 years
   ├─ Enable Bubble Detection: ON
   └─ Click "Run Full Analysis"
3. Wait for data to load (~30 seconds)
4. See results in tabs
```

**Step 2: First-Time Observations**
```
After running analysis, check:
├─ Market Overview tab
│  ├─ View portfolio value
│  ├─ Check correlation between assets
│  └─ Note any red/green indicators
│
├─ Bubble Detection tab
│  ├─ Review scores for each ticker
│  ├─ Identify high-risk assets (>0.6)
│  └─ Note factor breakdowns
│
└─ Risk Analytics tab
   ├─ Check max drawdown
   ├─ Review VaR numbers
   └─ Understand downside risk
```

---

### Complete Analysis Workflows

#### 📊 **Workflow 1: Portfolio Health Checkup (15 minutes)**

**Objective:** Get quick overview of portfolio status and identify issues

**Step-by-Step:**

1. **Check Overall Portfolio Status** (Market Overview tab)
   ```
   ├─ View current value and daily change
   ├─ Look at YTD performance
   └─ Is it positive? → Good. Negative? → Investigate
   ```

2. **Review Asset Performance** (Market Overview tab - Performance Table)
   ```
   ├─ Sort by "% Change" descending
   ├─ Identify top gainers (+) and losers (-)
   ├─ Check for balance:
   │  ├─ All positive? Portfolio is in sync
   │  ├─ Mixed? Expected, shows diversification
   │  └─ All negative? Risk event or correction
   ```

3. **Quick Risk Check** (Risk Analytics tab)
   ```
   ├─ Max Drawdown: -18.5%?
   │  ├─ >30% = Risky portfolio, consider bonds
   │  ├─ 15-30% = Moderate
   │  └─ <15% = Conservative
   │
   ├─ 95% VaR: -$4,285?
   │  ├─ Compare to acceptable loss per day
   │  └─ If too high, rebalance to bonds
   ```

4. **Bubble Check** (Bubble Detection tab)
   ```
   ├─ Review all bubble scores
   ├─ Any scores >0.7?
   │  └─ YES → RED FLAG, consider reducing position
   │  └─ NO → Proceed to next check
   ```

5. **Output Decision**
   ```
   All good? → Portfolio is healthy, continue monitoring
   Issues found? → Go to "Portfolio Rebalancing" workflow
   ```

---

#### 🎯 **Workflow 2: Portfolio Rebalancing (20 minutes)**

**Objective:** Optimize portfolio allocations based on current market conditions

**Step-by-Step:**

1. **Identify Current Allocation** (Portfolio Optimization tab)
   ```
   ├─ Look at allocation table
   ├─ Note which assets are overweight/underweight
   ├─ Compare to your target allocation
   └─ Example: Want 60/40 stocks/bonds?
   ```

2. **Choose Optimization Strategy** (Portfolio Optimization tab)
   ```
   For different situations:
   
   Conservative Portfolio:
   └─ Select: "Minimum Variance"
   
   Balanced (Most Common):
   └─ Select: "Maximum Sharpe Ratio"
   
   Aggressive:
   └─ Select: "Kelly Criterion"
       └─ Set Leverage Limit: 1.5x
   
   Risk-Aware (Bubble Concerns):
   └─ Select: "Maximum Sharpe Ratio"
       └─ Toggle: "Bubble-Aware Optimization" ON
       └─ Penalty Factor: 0.7
   ```

3. **Review Optimization Results**
   ```
   Compare Current vs. Optimized:
   ├─ AAPL: 25% → 20% (SELL 5%)
   ├─ VOO: 35% → 40% (BUY 5%)
   ├─ AGG: 40% → 40% (HOLD)
   
   Check Metrics Improvement:
   ├─ Sharpe Ratio increases? ✓ Good
   ├─ Volatility decreases? ✓ Good
   └─ Expected Return stable? ✓ Good
   ```

4. **Enable Efficient Frontier** (Optional visual check)
   ```
   ├─ Check "Show Efficient Frontier"
   ├─ Your blue dot should be near the curve
   ├─ If far from curve → Algorithm quality check
   └─ Update if portfolio was optimized
   ```

5. **Implementation Decision**
   ```
   Accept recommendations?
   ├─ YES → Execute trades per table
   ├─ MODERATE → Adjust allocation by 50% of recommendation
   └─ NO → Keep current allocation, run later
   ```

---

#### 🔍 **Workflow 3: Technical Analysis for Trading Signals (25 minutes)**

**Objective:** Identify technical patterns and support/resistance levels

**Step-by-Step:**

1. **Select Asset to Analyze** (Technical Analysis tab)
   ```
   ├─ Dropdown: Choose ticker (e.g., AAPL)
   ├─ Chart Type: Select "Candlestick"
   ├─ Timeframe: Select "Daily"
   └─ Review chart displays OHLC data
   ```

2. **Add Trend Indicators** (Technical Analysis tab - Indicators)
   ```
   ├─ In multi-select: Check "SMA (50)"
   ├─ Check "SMA (200)"
   ├─ Chart updates automatically
   
   Interpretation:
   ├─ SMA50 > SMA200 = Bullish (uptrend)
   ├─ SMA50 < SMA200 = Bearish (downtrend)
   └─ Crossover = Potential reversal
   ```

3. **Add Momentum Indicators** (Technical Analysis tab)
   ```
   ├─ Add "RSI (14)" from multi-select
   ├─ Look at values:
   │  ├─ >70 = Overbought (potential sell)
   │  ├─ 30-70 = Neutral
   │  └─ <30 = Oversold (potential buy)
   │
   ├─ Add "MACD" from multi-select
   ├─ Look for:
   │  ├─ Signal line crossover = Trading signal
   │  ├─ Histogram color = Momentum strength
   │  └─ Divergence = Strength/weakness warning
   ```

4. **Add Volatility Context** (Technical Analysis tab)
   ```
   ├─ Add "Bollinger Bands (20, 2)" from multi-select
   ├─ Price near upper band = Possible overbought
   ├─ Price near lower band = Possible oversold
   └─ Band width = Current volatility level
   ```

5. **Generate Trading Signal**
   ```
   Example Signal Setup:
   ├─ Price above SMA200 ✓
   ├─ SMA50 crosses above SMA200 ✓
   ├─ RSI = 35 (oversold, but momentum turning)
   ├─ MACD crosses above signal line ✓
   ├─ Price bouncing from lower Bollinger Band ✓
   
   CONCLUSION: BULLISH, consider BUY
   
   Alternative Signal:
   ├─ Price below SMA200 ✓
   ├─ RSI > 70 (overbought) ✓
   ├─ Price touches upper Bollinger Band ✓
   ├─ Volume increasing on down moves
   
   CONCLUSION: BEARISH, consider SELL or AVOID
   ```

---

#### 🚨 **Workflow 4: Bubble Detection & Risk Mitigation (20 minutes)**

**Objective:** Identify bubble risks and implement hedging strategies

**Step-by-Step:**

1. **Review Bubble Scores** (Bubble Detection tab)
   ```
   ├─ Look at summary table
   ├─ Identify assets with scores:
   │  ├─ >0.7 = RED FLAG (bubble likely)
   │  ├─ 0.5-0.7 = YELLOW FLAG (monitor closely)
   │  └─ <0.5 = GREEN (relatively safe)
   ```

2. **Analyze Risk Factors** (Bubble Detection tab - Factor Breakdown)
   ```
   For HIGH-RISK asset (e.g., NVDA, score 0.78):
   ├─ Price Momentum: 0.92 (EXTREME)
   │  └─ Action: Price rising too fast, reduction needed
   │
   ├─ Valuation Metrics: 0.85 (HIGH)
   │  └─ Action: P/E ratios above historical average
   │
   ├─ Technical Signals: 0.72 (HIGH)
   │  └─ Action: Overbought indicators (RSI >70)
   │
   └─ Understand which factors are the biggest concern
   ```

3. **Check Bubble Burst Impact** (Bubble Detection tab)
   ```
   ├─ NVDA: Estimated Loss -42.3%
   ├─ If you own $10,000 of NVDA: Could lose $4,230
   ├─ This is your downside scenario
   └─ Acceptable? Proceed. Unacceptable? Reduce position.
   ```

4. **Implement Risk Mitigation** (Choose ONE strategy)
   ```
   Strategy A: Reduce Position Size
   ├─ Current: 30% NVDA in portfolio
   ├─ Action: Reduce to 15% NVDA
   ├─ Execute: Sell 50% of NVDA holdings
   └─ Benefit: Limits downside exposure
   
   Strategy B: Set Stop-Loss
   ├─ Current NVDA Price: $850
   ├─ Set Stop-Loss at: $722 (-15%)
   ├─ Automatic sell if price drops
   └─ Benefit: Limits max loss to 15%
   
   Strategy C: Hedge with Puts (Advanced)
   ├─ Buy put options on NVDA
   ├─ Cost: ~2-3% of position value
   ├─ Benefit: Downside protected above hedge cost
   
   Strategy D: Diversify Away
   ├─ Reduce high-bubble assets
   ├─ Add low-bubble alternatives
   ├─ Example: Replace NVDA with SMH (chipmaker ETF)
   └─ More diversification = less individual stock risk
   ```

5. **Monitor Periodically**
   ```
   ├─ Check bubble score weekly
   ├─ If score increases further:
   │  ├─ Tighten stop-loss
   │  ├─ Reduce position more
   │  └─ Consider full exit
   │
   ├─ If score decreases:
   │  ├─ Relax restrictions
   │  └─ Consider adding back
   ```

---

#### 💰 **Workflow 5: Valuation Analysis for Investment Decisions (30 minutes)**

**Objective:** Determine if an asset is undervalued or overvalued

**Step-by-Step:**

1. **Navigate to Valuation Models Tab**
   ```
   ├─ Select your target asset (e.g., AAPL)
   └─ Review all valuation models
   ```

2. **Review DCF Analysis** (Valuation Models tab)
   ```
   ├─ Current Price: $245
   ├─ DCF Fair Value: $218
   ├─ Valuation Gap: +12.3% (OVERVALUED)
   
   Interpretation:
   ├─ Price >Fair Value → Overvalued
   │  ├─ Action: Reduce holdings or wait for pullback
   │  └─ Risk: Price correction could occur
   │
   ├─ Price <Fair Value → Undervalued
   │  ├─ Action: Accumulate on weakness
   │  └─ Opportunity: Upside potential
   ```

3. **Check Margin of Safety** (Valuation Models tab - DCF)
   ```
   ├─ Current Gap: +12.3%
   ├─ Is this acceptable?
   │  ├─ <10% = Good buying opportunity
   │  ├─ 10-20% = Fairly valued
   │  ├─ 20-30% = Moderately overvalued
   │  └─ >30% = Significantly overvalued
   
   For AAPL: 12.3% = Slight premium, acceptable
   ```

4. **Compare Multiple Valuation Models** (Valuation Models tab)
   ```
   Collect all valuations:
   ├─ CAPM Expected Return: 14.42%
   ├─ Fama-French Return: 13.42%
   ├─ APT Return: 15.30%
   
   Action: Average the estimates
   ├─ Average Expected Return: 14.4%
   ├─ Compare to historical performance
   ├─ If historical >14.4%: Undervalued (BUY)
   ├─ If historical <14.4%: Overvalued (SELL/HOLD)
   ```

5. **Decision Matrix**
   ```
   Combine DCF + Return Models:
   
   If: DCF Fair Value > Current Price AND Expected Return > Required Return
   THEN: STRONG BUY
   
   If: DCF Fair Value > Current Price OR Expected Return > Required Return
   THEN: BUY / ACCUMULATE
   
   If: DCF Fair Value ≈ Current Price AND Expected Return ≈ Required Return
   THEN: HOLD
   
   If: DCF Fair Value < Current Price OR Expected Return < Required Return
   THEN: HOLD / REDUCE
   
   If: DCF Fair Value << Current Price AND Expected Return << Required Return
   THEN: SELL / AVOID
   ```

---

#### ⚠️ **Workflow 6: Risk Assessment & Stress Testing (20 minutes)**

**Objective:** Understand portfolio downside scenarios and acceptable losses

**Step-by-Step:**

1. **Review Current Risk Metrics** (Risk Analytics tab)
   ```
   ├─ Maximum Drawdown: -18.5%
   ├─ 95% VaR: -$4,285 (daily max loss)
   ├─ CVaR: -$5,620 (worst 5% scenarios)
   
   Question: Are these acceptable to you?
   ├─ YES → Proceed to next check
   ├─ NO → Too risky, reduce equity allocation
   ```

2. **Check Correlation Breakdown** (Risk Analytics tab)
   ```
   ├─ Look at correlation matrix
   ├─ Find pairs with LOW correlation (<0.3)
   │  └─ These are good diversifiers
   ├─ Find pairs with HIGH correlation (>0.7)
   │  └─ These create redundancy, consider reducing
   
   Example from earlier README:
   ├─ AAPL-VOO: 0.92 (highly correlated)
   │  └─ Both move together, limited diversification
   ├─ AGG-AAPL: 0.15 (low correlation)
   │  └─ Bond-stock diversification benefit
   ```

3. **Review Drawdown History** (Risk Analytics tab)
   ```
   ├─ Maximum Drawdown: -18.5%
   ├─ Average Drawdown: -8.2%
   ├─ Median Drawdown: -6.1%
   
   Analysis:
   ├─ If max drawdown was in 2020: COVID, temporary
   ├─ If max drawdown was 2022: Rate shock, longer duration
   ├─ Recovery time: How many days to return to peak?
   │  ├─ <30 days = Quick recovery, low psychological impact
   │  ├─ 30-90 days = Moderate, tests discipline
   │  └─ >90 days = Significant, need strong conviction
   ```

4. **Run Stress Tests** (Risk Analytics tab - Stress Testing)
   ```
   Review each scenario:
   
   Scenario 1: Historical Crisis (-34.2% portfolio impact)
   ├─ Question: Could you withstand this?
   ├─ If NO → Too much equity risk
   
   Scenario 2: Rate Spike (-12.5% portfolio impact)
   ├─ Question: Bond duration too long?
   ├─ Check AGG impact separately
   
   Scenario 3: Tech Collapse (-28.3% portfolio impact)
   ├─ Question: Too much tech concentration?
   ├─ Compare AAPL + MSFT holdings
   
   Scenario 4: Market Crash (-22.1% portfolio impact)
   ├─ Question: Acceptable loss?
   ├─ This is near-worst case
   └─ If unacceptable, rebalance now
   ```

5. **Action Plan Based on Risk Assessment**
   ```
   If all stress tests <-30%:
   ├─ Portfolio is too aggressive
   ├─ Action: Increase bond allocation (AGG/BND)
   ├─ Target: Find allocation where all stresses <-20%
   
   If stress tests -20% to -30%:
   ├─ Portfolio is moderately aggressive
   ├─ Accept if growth goals justify risk
   
   If stress tests <-20%:
   ├─ Portfolio is conservative
   ├─ Acceptable for risk-averse investors
   ```

---

### Analysis Use Cases

#### **Use Case 1: "I want to find undervalued stocks"**

**Where to Start:**
1. Valuation Models tab → DCF Analysis section
2. Look for: Current Price > DCF Fair Value (YES = undervalued)
3. Margin of Safety > 15% (good safety margin)
4. CAPM Expected Return > Required Rate (YES = opportunity)

**Decision:**
- If undervalued AND expected return strong → Consider BUYING
- Wait for 10%+ pullback if only slightly undervalued

---

#### **Use Case 2: "My portfolio is down, should I sell?"**

**Where to Start:**
1. Bubble Detection tab → Review bubble scores
2. Risk Analytics tab → Check if positions have deteriorated
3. Valuation Models → Has fair value decreased?

**Decision:**
- If valuation improved (cheaper), hold or BUY more
- If valuation deteriorated (overvalued), consider SELLING
- Check correlation matrix: Is diversification still working?

---

#### **Use Case 3: "I want a portfolio with minimal risk"**

**Where to Start:**
1. Portfolio Optimization → Strategy: "Minimum Variance"
2. Risk Analytics → Review all metrics
3. Portfolio Optimization → Enable "Efficient Frontier"
4. Your portfolio should be far LEFT on the curve (low volatility)

**Implementation:**
- High AGG/BND allocation (60-70%)
- Low equity allocation (30-40%)
- Results: Lower returns, but much lower volatility

---

#### **Use Case 4: "Is this a bubble?"**

**Where to Start:**
1. Bubble Detection tab → Look at Bubble Score
2. Factor Breakdown → Which factors are most concerning?
3. Compare to other assets

**Decision Framework:**
- Score 0.0-0.3 = NO, not a bubble
- Score 0.3-0.5 = MAYBE, monitor closely
- Score 0.5-0.7 = LIKELY, reduce position
- Score 0.7-1.0 = PROBABLE, exit or hedge

---

#### **Use Case 5: "I want to generate a client report"**

**Where to Start:**
1. Report Generator tab
2. Select report type (PDF for clients)
3. Include all sections (Executive Summary + all analysis)
4. Click "Generate Report"
5. Download and distribute

**Report Usage:**
- Executive Summary for quick review
- Technical Analysis for detailed charts
- Risk Metrics for risk conversations
- Valuation for investment justification

---

## Advanced Features

### 🔄 Live Market Monitoring

**Auto-Refresh Feature**
- Real-time price updates every 30-300 seconds
- Market close detection (stops updating after 4 PM EST)
- Session-state preservation
- Cache management for efficiency

**Usage:**
```
Sidebar Configuration
├─ Enable Auto-Refresh: [Toggle]
├─ Refresh Interval: [30s - 5m Slider]
└─ Active Session Time: [Display]
```

### 📊 Efficient Frontier

**Visualization:**
- Scatter plot of all possible portfolios
- Efficient frontier curve (highest Sharpe at each volatility level)
- Your optimized portfolio highlighted
- Interactive hover showing exact metrics

**How to Use:**
```
1. Enable "Show Efficient Frontier"
2. Platform generates 100+ random portfolios
3. Chart displays all with efficient frontier overlay
4. Identify optimal risk-return tradeoff
5. Adjust volatility tolerance and regenerate
```

### 🎯 Correlation-Based Asset Clustering

**Hierarchical Clustering Dendrogram**
- Visual tree showing asset relationships
- Distance metric based on correlation
- Identify natural portfolio groupings

**Usage:**
```
Understanding the Dendrogram:
├─ Short branches = Highly correlated assets
├─ Long branches = Low correlation assets
├─ Natural cutoff point = Optimal number of clusters
└─ Creates natural portfolio segments
```

### 📈 Principal Component Analysis (PCA)

**Factor Extraction**
- Identifies main drivers of portfolio returns
- Reduces dimensionality to principal factors
- Explains variance contribution

**Interpretation:**
```
PC1 (Market Risk): Explains 72% of variance
├─ Loaded on: All equity holdings (AAPL, MSFT, VOO)
└─ Interpretation: Mainly equity market beta

PC2 (Diversification): Explains 18% of variance
├─ Loaded on: AGG (bonds) negative, Equities positive
└─ Interpretation: Equity/bond allocation mix

PC3 (Momentum): Explains 8% of variance
├─ Loaded on: Tech stocks (AAPL, MSFT) stronger
└─ Interpretation: Momentum effect
```

---

## Technical Architecture

### Technology Stack
- **Frontend:** Streamlit (Python web framework)
- **Data:** Yahoo Finance (yfinance)
- **Analysis:** Pandas, NumPy, SciPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Optimization:** SciPy optimize, scikit-learn
- **Technical Analysis:** TA-Lib (ta library)
- **Reporting:** XlsxWriter, Matplotlib

### Data Flow
```
┌─────────────────────────────────────────────────┐
│ User Input (Tickers, Date Range)                │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│ Yahoo Finance Data Fetch                        │
│ ├─ Price data (OHLCV)                          │
│ ├─ Financial statements                        │
│ └─ Market data (10Y Treasury, etc.)             │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│ Data Processing                                 │
│ ├─ Clean & normalize                           │
│ ├─ Calculate returns                           │
│ └─ Compute correlations                        │
└─────────────┬───────────────────────────────────┘
              │
         ┌────┴────┬──────────────┬─────────────┐
         ▼         ▼              ▼             ▼
    Technical  Bubble       Portfolio      Valuation
    Analysis   Detection    Optimization   Models
         │         │              │             │
         └────┬────┴──────────────┴─────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│ Visualization & Reporting                       │
│ ├─ Plotly Charts                               │
│ ├─ PDF/Excel Export                            │
│ └─ Dashboard Display                           │
└─────────────────────────────────────────────────┘
```

### Caching Strategy
```
Data Cache (TTL: 60 seconds)
├─ Market prices (yfinance)
├─ Financial statements
└─ Technical indicators

Analysis Cache (TTL: 3600 seconds)
├─ Risk-free rate (Treasury)
├─ Calculated metrics
└─ Optimization results
```

---

## Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 500MB disk space
- Internet connection (for market data)

### Python Dependencies
```
streamlit==1.28+
yfinance==0.2+
pandas==2.0+
numpy==1.24+
plotly==5.17+
scipy==1.11+
statsmodels==0.14+
scikit-learn==1.3+
ta==0.10+
xlsxwriter==3.1+
matplotlib==3.8+
seaborn==0.13+
```

See `requirements.txt` for exact versions.

---

## Configuration

### Running on Different Environments

**Local Development**
```bash
streamlit run app.py --logger.level=debug
```

**Production Server**
```bash
streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --logger.level=warning
```

**Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Environment Variables
```bash
# Optional configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_LOGGER_LEVEL=warning
```

---

## Troubleshooting

### Common Issues

**Issue: "No module named 'streamlit'"**
```bash
Solution: pip install -r requirements.txt
Verify: python -c "import streamlit; print(streamlit.__version__)"
```

**Issue: "Unable to fetch data for ticker"**
```bash
Causes:
├─ Invalid ticker symbol (check spelling)
├─ Network connectivity issues
├─ Yahoo Finance API temporarily unavailable
└─ Ticker delisted or renamed

Solution:
1. Verify ticker on Yahoo Finance website
2. Check internet connection
3. Wait and retry (API rate limiting)
4. Try alternative ticker (e.g., BRK.B instead of BRK/B)
```

**Issue: "Optimization failed to converge"**
```bash
Causes:
├─ Insufficient data points (<60 days)
├─ Perfect correlation between assets
├─ Singular covariance matrix
└─ Extreme outliers in returns

Solutions:
1. Increase date range
2. Remove perfectly correlated assets
3. Use different optimization method
4. Check data quality in output
```

**Issue: "Bubble detection scores not updating"**
```bash
Causes:
├─ Bubble detection disabled
├─ Cache not cleared
├─ Data stale (old date range)

Solution:
1. Toggle "Enable Bubble Detection" ON
2. Click "Clear Cache" button
3. Run full analysis
4. Check timestamp ("Last Updated")
```

### Performance Optimization

**For slow performance with many assets (>15):**
```
1. Reduce date range (use 1-2 years instead of 5)
2. Increase cache TTL in sidebar
3. Disable real-time auto-refresh
4. Run on machine with more RAM
5. Use alternative optimization (Equal Weight instead of HRP)
```

**For slow chart loading:**
```
1. Reduce number of technical indicators displayed
2. Use daily timeframe instead of intraday
3. Increase time range aggregation (weekly/monthly)
4. Disable Plotly hover tooltips
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup
```bash
git clone https://github.com/ranjithvijik/quantlab.git
cd quantlab
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Reporting Issues
1. Check existing issues first
2. Provide detailed reproduction steps
3. Include error messages and screenshots
4. Specify Python version and OS

### Submitting Pull Requests
1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open pull request with description

---

**Last Updated:** December 2, 2025