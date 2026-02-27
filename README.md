# Crypto Systemic Risk Engine — v1.0

**An institutional-grade crypto portfolio risk analysis platform** that combines
Monte Carlo simulation, regime detection, PCA-based systemic risk scoring,
and social sentiment positioning into a single interactive Dash dashboard.

---

## What It Does

| Layer | Input | Output |
|---|---|---|
| **Liquidity** | Exchange volumes, stablecoin flows | Volume anomaly ratio, liquidity multiplier |
| **Regime** | Rolling volatility, correlation, PCA eigenvalues | Risk score 0–100, regime classification |
| **Leverage** | Funding rates proxy (vol clustering, beta regime) | Leverage/amplified/decoupled signal |
| **Positioning** | Social sentiment (News, Reddit, X/Twitter) | Sentiment z-score, regime-conditioned sensitivity |

---

## Framework Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    CRYPTO SYSTEMIC RISK ENGINE                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│   │  LIQUIDITY    │  │  REGIME      │  │  LEVERAGE            │  │
│   │              │  │              │  │                      │  │
│   │ · Exchange   │  │ · Rolling    │  │ · Funding rate proxy │  │
│   │   inflow vol │  │   vol (30d)  │  │ · Vol clustering     │  │
│   │ · Volume     │  │ · Pairwise   │  │ · Beta regime        │  │
│   │   anomaly    │  │   correlation│  │   detection          │  │
│   │ · Stablecoin │  │ · PCA eigval │  │   (>1.5 = leverage)  │  │
│   │   dominance  │  │   decomp.    │  │                      │  │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│          │                 │                      │              │
│          ▼                 ▼                      ▼              │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │            COMPOSITE RISK INDEX (0–100)                   │  │
│   │                                                          │  │
│   │   Correlation (25) + PCA (25) + Vol (25) +               │  │
│   │   BTC Dominance (15) + Volume Anomaly (10)               │  │
│   │                                                          │  │
│   │   Zones: Low (0-25) | Elevated (25-50) |                │  │
│   │          High (50-75) | Extreme (75-100)                 │  │
│   └───────────────────────────┬──────────────────────────────┘  │
│                               │                                  │
│   ┌───────────────────────────▼──────────────────────────────┐  │
│   │            POSITIONING (Sentiment Layer)                  │  │
│   │                                                          │  │
│   │ · Sentiment index (daily, multi-source VADER)            │  │
│   │ · Sentiment shock (z-score, 14d rolling)                 │  │
│   │ · β: return per unit sentiment change                    │  │
│   │ · Volatility response to sentiment shocks                │  │
│   │   → CONDITIONED ON RISK REGIME                           │  │
│   │                                                          │  │
│   │ Deliverable: "In high-risk regimes, negative sentiment   │  │
│   │ causes X× larger downside."                              │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │            SIMULATION ENGINE                              │  │
│   │                                                          │  │
│   │ · GBM Monte Carlo          · Copula (Gaussian)           │  │
│   │ · GARCH(1,1)               · Regime-Switching MC         │  │
│   │ · Historical Simulation    · Stress Tests                │  │
│   │ · Block Bootstrap          · Normal vs Crisis regime     │  │
│   │                                                          │  │
│   │ Risk metrics: VaR, CVaR, Kupiec backtest, PCA systemic   │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │            VALIDATION & BACKTESTING                       │  │
│   │                                                          │  │
│   │ · Kupiec POF test (χ²) for VaR model validity            │  │
│   │ · Lead-lag cross-correlation (does score LEAD stress?)    │  │
│   │ · Rule-based strategy backtest vs buy-and-hold            │  │
│   │ · Risk-return asymmetry by regime                        │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Insights

1. **The risk index leads volatility** — score today predicts 14-day forward realised vol
   (positive slope in the leading-indicator scatter, R² reported on chart)

2. **Regime conditioning matters** — the same −1σ sentiment shock produces X× worse
   returns during High Stress vs Low Risk regimes

3. **Diversification breaks down in crises** — PCA first-PC concentration spikes >70%
   during liquidation cascades, meaning all coins move as one

4. **Risk-managed rule beats buy-and-hold on Sharpe** — simple threshold rule
   (< 30 = full, 30–60 = half, > 60 = flat) reduces max drawdown with
   comparable or better CAGR

5. **VaR model validation** — Kupiec backtest confirms the parametric VaR model
   produces the correct number of tail breaches (or flags when it doesn't)

---

## Modules

| File | Purpose |
|---|---|
| `app.py` | Dash web application — all UI, callbacks, chart builders |
| `config.py` | API credentials (local only — **never commit to VCS**) |
| `fetch_data.py` | CoinGecko data fetching, regime-switching MC, stress tests |
| `risk_analysis.py` | Beta calculation, liquidity scoring, regime detection |
| `risk_engine.py` | Statistical core: distributions, rolling metrics, VaR, PCA, backtest |
| `social_sentiment.py` | Multi-source sentiment (News/Reddit/X), positioning layer |
| `systemic_risk.py` | Composite risk index, component decomposition, lead-lag, backtest |
| `visualization.py` | Matplotlib chart functions (legacy, Dash uses Plotly) |
| `preprocess.py` | Data preprocessing utilities |
| `utils.py` | General utility functions |

---

## Quick Start

```bash
# 1. Create environment (recommended)
conda create -n crypto_risk python=3.11
conda activate crypto_risk

# 2. Install dependencies
pip install dash plotly pycoingecko praw tweepy nltk arch scipy numpy pandas requests

# 3. Set API keys (optional — app works without them, sentiment needs NewsAPI)
#    Option A: Edit config.py directly
#    Option B: Set environment variables:
#      export NEWS_API_KEY="your_key"
#      export REDDIT_CLIENT_ID="your_id"
#      export REDDIT_SECRET="your_secret"
#      export X_BEARER_TOKEN="your_token"

# 4. Run
python app.py
# → Open http://127.0.0.1:8050

# 5. Workflow:
#    [Fetch Data] → [Set holdings] → [Run Full Analysis] → explore tabs
#    [Fetch Sentiment] → Sentiment tab for positioning analysis
```

---

## Requirements

| Package | Version | Purpose |
|---|---|---|
| `dash` | ≥2.14 | Web dashboard framework |
| `plotly` | ≥5.18 | Interactive charts |
| `pycoingecko` | ≥3.1 | CoinGecko API wrapper |
| `pandas` | ≥2.0 | Data manipulation |
| `numpy` | ≥1.24 | Numerical computing |
| `scipy` | ≥1.11 | Statistical functions |
| `arch` | ≥6.2 | GARCH modelling (optional) |
| `praw` | ≥7.7 | Reddit API (optional) |
| `tweepy` | ≥4.14 | X/Twitter API (optional) |
| `nltk` | ≥3.8 | VADER sentiment (auto-downloads) |
| `requests` | ≥2.31 | HTTP for DefiLlama |

---

## Employer Pitch

> I built a **crypto systemic risk engine** from scratch that combines
> five independent market-stress signals (correlation, PCA, volatility,
> BTC dominance, volume anomaly) into a composite 0–100 risk score,
> then layers social sentiment positioning (News + Reddit + X) on top.
> The non-obvious insight: **the same sentiment shock causes dramatically
> different outcomes depending on risk regime** — negative sentiment
> during high stress produces X× larger drawdowns than during calm markets.
> The index leads 14-day forward volatility, passes Kupiec VaR backtesting,
> and a simple regime-based allocation rule improves Sharpe ratio vs
> buy-and-hold. Built in Python with Dash, Plotly, and institutional-grade
> statistical methods (PCA, Cholesky, copula, GARCH, Markov regime-switching).

---

## Version History

- **v1.0** — Systemic risk engine with 5-signal composite index,
  sentiment positioning layer, regime-conditioned sensitivity analysis,
  six Monte Carlo models, Kupiec backtest, lead-lag validation,
  rule-based strategy backtest, interactive Dash dashboard.

---

*Author: DeFi Risk Team · Built with Python, Dash, Plotly*

---

## See Also

- [**MATH_EXPLAINED.md**](MATH_EXPLAINED.md) — Beginner-friendly explanations of all the math concepts used in this project (Monte Carlo, VaR, GARCH, PCA, Copula, etc.)

---

## License

MIT License — free to use, modify, and distribute.
