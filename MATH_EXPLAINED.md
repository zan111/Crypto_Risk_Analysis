# Math Behind the Crypto Risk Engine — Beginner's Guide

This document explains **all the math concepts** used in this project in simple terms.
No PhD required — just curiosity!

---

## Table of Contents

1. [Log Returns — Why Not Just Percentages?](#1-log-returns)
2. [Volatility — Measuring "Risk"](#2-volatility)
3. [Monte Carlo Simulation — Rolling Dice 1000 Times](#3-monte-carlo-simulation)
4. [GBM — The Classic Price Model](#4-gbm-geometric-brownian-motion)
5. [GARCH — When Volatility Changes Over Time](#5-garch)
6. [VaR & CVaR — "How Bad Can It Get?"](#6-var-and-cvar)
7. [Sharpe & Sortino — Risk-Adjusted Returns](#7-sharpe-and-sortino-ratios)
8. [Maximum Drawdown — Worst Peak-to-Trough](#8-maximum-drawdown)
9. [Correlation — Do Assets Move Together?](#9-correlation)
10. [Covariance Matrix — Correlation + Scale](#10-covariance-matrix)
11. [PCA — Finding Hidden Factors](#11-pca-principal-component-analysis)
12. [Copula — Modeling Tail Dependence](#12-copula)
13. [Beta — Sensitivity to Bitcoin](#13-beta)
14. [Kupiec Test — Does Our Model Work?](#14-kupiec-test)
15. [Markov Regime-Switching — Bull vs Bear Markets](#15-markov-regime-switching)

---

## 1. Log Returns

### The Problem with Simple Returns

If BTC goes from $100 to $110, that's a **+10%** return.
If it then drops from $110 to $100, that's a **−9.09%** return (not −10%!).

Simple returns don't add up symmetrically. This causes problems in statistics.

### The Solution: Log Returns

**Log return** = ln(Price_today / Price_yesterday)

```
Price: $100 → $110
Log return = ln(110/100) = ln(1.10) = 0.0953 = +9.53%

Price: $110 → $100
Log return = ln(100/110) = ln(0.909) = −0.0953 = −9.53%
```

**Magic property**: Log returns ADD UP correctly over time!

### Why We Use Them

| Property | Simple Returns | Log Returns |
|---|---|---|
| Symmetric | ❌ No | ✅ Yes |
| Time-additive | ❌ No | ✅ Yes |
| Works with stats | ❌ Messier | ✅ Clean |

**In the code** (`fetch_data.py`):
```python
df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
```

---

## 2. Volatility

### What Is It?

**Volatility = Standard Deviation of returns**

It measures how much prices "jump around." Higher volatility = riskier asset.

### Intuition

Imagine two coins:
- **Coin A**: Daily returns of +1%, −1%, +1%, −1% → Very predictable
- **Coin B**: Daily returns of +10%, −15%, +20%, −8% → Wild swings

Coin B has **higher volatility**.

### The Formula

$$
\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (r_i - \bar{r})^2}
$$

Where:
- $r_i$ = each daily return
- $\bar{r}$ = average return
- $n$ = number of days

### Annualizing Volatility

Daily volatility × √365 = Annual volatility

Why √365? Because variance adds over time, and volatility is the square root of variance.

**Example**: 
- Daily vol = 3%
- Annual vol = 3% × √365 ≈ 57%

**In the code** (`risk_engine.py`):
```python
annualised_vol = daily_std * np.sqrt(365)
```

---

## 3. Monte Carlo Simulation

### The Idea

**Monte Carlo** = Run thousands of random simulations to see the range of possible outcomes.

Named after the Monte Carlo casino — it's like rolling dice many times to understand probabilities.

### How It Works

1. **Model the randomness** (what's the typical daily change?)
2. **Generate random paths** (simulate 1000 possible futures)
3. **Analyze the distribution** (what outcomes are likely?)

### Visual Example

```
Today: $50,000
        │
   Day 1├─── +2% → $51,000  (path 1)
        │─── −1% → $49,500  (path 2)
        │─── +3% → $51,500  (path 3)
        │─── −2% → $49,000  (path 4)
        ...
   Day 90: See where all 1000 paths ended up
```

### What We Learn

- **Median outcome** (50th percentile)
- **Best case** (95th percentile)
- **Worst case** (5th percentile)
- **Probability of hitting a target price**

**In the code** (`app.py`):
```python
for _ in range(num_sim):  # e.g., 1000 simulations
    path = [last_price]
    for day in range(horizon_days):
        random_return = np.random.normal(mu, sigma)
        path.append(path[-1] * np.exp(random_return))
    paths.append(path)
```

---

## 4. GBM (Geometric Brownian Motion)

### What Is It?

**GBM** is the most common model for asset prices. It assumes:
1. Prices can't go negative
2. Returns are normally distributed
3. Future returns are independent of past returns

### The Formula

$$
S_t = S_0 \cdot e^{(\mu - \frac{\sigma^2}{2})t + \sigma W_t}
$$

Don't panic! Let me break it down:

- $S_t$ = Price at time $t$
- $S_0$ = Starting price (today)
- $\mu$ = Average return (drift)
- $\sigma$ = Volatility
- $W_t$ = Random "Brownian motion" (fancy term for randomness)

### Intuition

Think of it as: **Tomorrow's price = Today's price × (1 + Drift + Randomness)**

The drift is the "expected" direction, and randomness adds uncertainty.

### Why Subtract σ²/2?

This is a correction because log returns and simple returns aren't the same.
Without it, simulations would drift upward too much.

**In the code** (`app.py`):
```python
drift = mean_return - 0.5 * volatility**2
daily_return = drift + volatility * np.random.normal()
new_price = old_price * np.exp(daily_return)
```

---

## 5. GARCH

### The Problem with GBM

GBM assumes volatility is **constant**. But in reality:
- After a crash, markets stay volatile for a while
- Calm periods tend to stay calm

**Volatility clusters!**

### What GARCH Does

**GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)** models volatility as **time-varying**.

Today's volatility depends on:
1. **Yesterday's volatility** (persistence)
2. **Yesterday's return shock** (reaction to news)

### The GARCH(1,1) Formula

$$
\sigma_t^2 = \omega + \alpha \cdot \epsilon_{t-1}^2 + \beta \cdot \sigma_{t-1}^2
$$

Where:
- $\sigma_t^2$ = Today's variance
- $\omega$ = Long-term baseline variance
- $\alpha$ = Reaction to recent shocks (typically 0.05–0.15)
- $\beta$ = Persistence of volatility (typically 0.80–0.95)
- $\epsilon_{t-1}$ = Yesterday's unexpected return

### Intuition

**Big moves today → Expect big moves tomorrow**

If α + β is close to 1, volatility "remembers" for a long time (persistent).

### Why It Matters

- More realistic risk estimates during volatile periods
- Better VaR during market stress

**In the code** (`app.py`):
```python
am = arch_model(log_returns, vol="Garch", p=1, q=1)
res = am.fit(disp="off")
forecasts = res.forecast(horizon=horizon_days)
```

---

## 6. VaR and CVaR

### VaR (Value at Risk)

**VaR** answers: "What's the worst loss I could expect 95% of the time?"

**Example**:
- Portfolio value: $100,000
- 1-day 95% VaR: $3,000
- **Meaning**: On 95% of days, you'll lose less than $3,000

### Visual

```
                ┌──────────────────────────────────┐
                │        Return Distribution        │
                │                                  │
Probability     │              ▓▓▓▓                │
                │            ▓▓▓▓▓▓▓▓              │
                │          ▓▓▓▓▓▓▓▓▓▓▓▓            │
                │     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓         │
                │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓       │
                └──┬───────────────────────────────┘
                   │
                 VaR (5th percentile)
                   │
         "95% of outcomes are better than this"
```

### CVaR (Conditional VaR) — Also Called "Expected Shortfall"

**CVaR** asks: "When things ARE worse than VaR, how bad on average?"

VaR tells you the threshold. CVaR tells you the **average** of the tail.

**Example**:
- 95% VaR: $3,000
- 95% CVaR: $4,500
- **Meaning**: When you do lose more than $3,000, the average loss is $4,500

### Why CVaR Is Better

- VaR says nothing about what happens in the worst 5%
- CVaR captures the "tail risk"

**In the code** (`risk_engine.py`):
```python
var_1d = -np.percentile(pnl, 5)         # 5th percentile
tail = pnl[pnl <= -var_1d]              # All losses worse than VaR
cvar_1d = -tail.mean()                  # Average of the bad days
```

---

## 7. Sharpe and Sortino Ratios

### The Question

Two investments made 20% per year. One was a rollercoaster, one was smooth.
**Which is better?**

We want **return per unit of risk**.

### Sharpe Ratio

$$
\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}
$$

Where:
- $R_p$ = Portfolio return (annualized)
- $R_f$ = Risk-free rate (we use 0% for crypto)
- $\sigma_p$ = Portfolio volatility (annualized)

**Interpretation**:
| Sharpe | Meaning |
|---|---|
| < 0 | Losing money |
| 0–1 | Subpar |
| 1–2 | Good |
| 2–3 | Very good |
| > 3 | Excellent (suspicious?) |

### Sortino Ratio — The Better Version

**Problem with Sharpe**: It penalizes upside volatility too!
If your portfolio jumps UP 10% one day, Sharpe treats that as "bad."

**Sortino only penalizes DOWNSIDE volatility**:

$$
\text{Sortino} = \frac{R_p - R_f}{\sigma_{downside}}
$$

Where $\sigma_{downside}$ = standard deviation of **negative returns only**.

### Which To Use?

- **Sharpe**: Industry standard, easier to compare
- **Sortino**: More accurate for asymmetric returns (like crypto)

**In the code** (`risk_engine.py`):
```python
sharpe = (annualised_return - risk_free) / annualised_vol

downside_returns = daily_returns[daily_returns < 0]
downside_vol = downside_returns.std() * np.sqrt(365)
sortino = (annualised_return - risk_free) / downside_vol
```

---

## 8. Maximum Drawdown

### What Is It?

**Maximum Drawdown** = Largest peak-to-trough decline before a new high.

It measures the **worst losing streak** in your investment's history.

### Example

```
Portfolio value over time:

   $120 ─────●────────────────────────●─── New Peak
             │                        │
   $100 ─────┤                        │
             │                        │
    $80 ─────┤────────────●           │
             │            │           │
    $60 ─────┤            │           │
             │            │           │
         ────┴────────────┴───────────┴────
           Peak        Trough      Recovery

Max Drawdown = ($120 - $80) / $120 = 33.3%
```

### Why It Matters

- **Real pain metric**: −50% drawdown means you need +100% to recover!
- **Survival test**: Can you emotionally handle this drop?
- **Risk management**: Position sizing based on max acceptable drawdown

### Recovery Math (Why Drawdowns Hurt)

| Drawdown | Gain Needed to Recover |
|---|---|
| −10% | +11% |
| −25% | +33% |
| −50% | +100% |
| −75% | +300% |
| −90% | +900% |

**In the code** (`risk_engine.py`):
```python
cumulative_max = prices.cummax()
drawdown = (prices - cumulative_max) / cumulative_max
max_drawdown = drawdown.min()  # Most negative = worst
```

---

## 9. Correlation

### What Is It?

**Correlation** measures how two assets move **together**.

- **+1.0** = Perfect positive (both go up/down together)
- **0.0** = No relationship
- **−1.0** = Perfect inverse (one up, other down)

### Visual

```
Correlation = +0.9 (high)      Correlation = 0 (none)
     BTC                            BTC
      │                              │
   ▲  │  ▲                        ▲  │     ▲
  /│\ │ /│\                      / \ │    / \
   │  │  │                      /   \│   /   \
  ─┼──┼──┼──                  ─/─────\─/─────\──
   │  │  │                    /       \       \
   │  │  │                   /         \       \
     ETH (follows BTC)           ETH (random)
```

### The Formula

$$
\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
$$

Where:
- Cov(X, Y) = Covariance (how they move together)
- $\sigma_X$, $\sigma_Y$ = Individual volatilities

### Why Correlation Matters for Risk

- **High correlation** → Less diversification benefit
- **Crisis behavior**: Correlations often spike to ~1.0 during crashes
- **"Only when you need diversification, it disappears"**

**In the code** (`risk_engine.py`):
```python
rolling_corr = coin1_returns.rolling(60).corr(coin2_returns)
```

---

## 10. Covariance Matrix

### What Is It?

A **covariance matrix** is a table showing:
- **Diagonal**: Each asset's variance (vol²)
- **Off-diagonal**: Covariance between pairs

### Example (3 coins)

```
          BTC      ETH      SOL
BTC    [0.0016   0.0014   0.0012]
ETH    [0.0014   0.0020   0.0015]
SOL    [0.0012   0.0015   0.0025]
```

Diagonal (0.0016, 0.0020, 0.0025) = variances
Off-diagonal = covariances

### Why We Need It

To calculate **portfolio variance**:

$$
\sigma_p^2 = w^T \Sigma w
$$

Where:
- $w$ = Vector of portfolio weights [0.5, 0.3, 0.2]
- $\Sigma$ = Covariance matrix
- $w^T$ = Transpose of weights

### Intuition

Portfolio risk isn't just the sum of individual risks!  
Correlations can **reduce** total risk (diversification) or **amplify** it.

**In the code** (`risk_engine.py`):
```python
cov_matrix = returns_df.cov() * 365  # Annualized
portfolio_var = weights @ cov_matrix @ weights  # Matrix multiplication
portfolio_vol = np.sqrt(portfolio_var)
```

---

## 11. PCA (Principal Component Analysis)

### The Problem

With 10 coins, there are 45 pairwise correlations. Hard to interpret!

### What PCA Does

PCA finds the **hidden factors** driving all the coins.

Usually:
- **PC1** (1st principal component) = "Market factor" (everything up/down together)
- **PC2** = Maybe BTC vs altcoins
- **PC3** = Maybe DeFi vs L1s

### How It Works (Simplified)

1. Build the correlation matrix
2. Find the "directions" that explain the most variance
3. Rank them by importance

### The Math (Eigenvalue Decomposition)

$$
\text{Correlation Matrix} = V \Lambda V^T
$$

Where:
- $V$ = Matrix of eigenvectors (directions)
- $\Lambda$ = Diagonal matrix of eigenvalues (importance)

### Why It Matters for Risk

If **PC1 explains > 70%** of variance → All coins move as one → **Systemic risk is HIGH!**

During normal times: PC1 might explain 40%
During crisis: PC1 spikes to 80%+ (diversification vanishes)

**In the code** (`risk_engine.py`):
```python
eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
explained_variance = eigenvalues / eigenvalues.sum()
first_pc_explains = explained_variance[-1]  # Largest eigenvalue

if first_pc_explains > 0.70:
    print("⚠️ Systemic risk! Market moving as one!")
```

---

## 12. Copula

### The Problem

Normal correlation assumes everything is "normally distributed."  
But in reality, **crypto correlations spike during crashes** (tail dependence).

### What a Copula Does

A **copula** separates:
1. **Marginal distributions** (each coin's individual behavior)
2. **Dependence structure** (how they move together)

This lets us model **fat tails** and **tail dependence** properly.

### Gaussian Copula (What We Use)

1. Transform each coin's returns to uniform [0, 1] using CDF
2. Apply the correlation structure in "Gaussian space"
3. Transform back to original scales

### Visual

```
Original returns         Uniform space          Gaussian space
(fat-tailed)            (copula magic)         (easy to correlate)
     │                      │                       │
  ▓▓▓▓▓▓▓▓               ▓▓▓▓▓▓                  ▓▓▓▓▓▓
 ▓▓▓▓▓▓▓▓▓▓     →      ▓▓▓▓▓▓▓▓▓▓      →      ▓▓▓▓▓▓▓▓▓▓
▓▓▓▓▓▓▓▓▓▓▓▓          ▓▓▓▓▓▓▓▓▓▓▓▓            ▓▓▓▓▓▓▓▓▓▓▓▓
```

### Why It Matters

- More realistic multi-asset simulations
- Captures the "everything crashes together" behavior
- Standard in professional risk systems

**In the code** (`app.py`):
```python
# Fit marginals
marginals = [stats.norm.fit(returns[:, i]) for i in range(n_coins)]

# Build correlation matrix
corr = np.corrcoef(returns.T)
cholesky = np.linalg.cholesky(corr)  # For generating correlated normals

# Generate copula samples
z = np.random.normal(size=(n_sims, n_coins))
correlated_z = z @ cholesky.T
```

---

## 13. Beta

### What Is It?

**Beta** measures sensitivity to a benchmark (usually BTC for altcoins).

- **β = 1.0** → Moves 1:1 with BTC
- **β = 1.5** → Moves 1.5x as much as BTC (amplified)
- **β = 0.5** → Moves only half as much (defensive)
- **β = 0.0** → No relationship

### The Formula

$$
\beta = \frac{\text{Cov}(r_{coin}, r_{BTC})}{\text{Var}(r_{BTC})}
$$

### Intuition

"If BTC goes up 10%, how much does this coin go up?"

| Beta | BTC +10% | BTC −10% |
|---|---|---|
| 0.5 | +5% | −5% |
| 1.0 | +10% | −10% |
| 1.5 | +15% | −15% |
| 2.0 | +20% | −20% |

### Rolling Beta

Beta changes over time! During bull markets, altcoins often have β > 1 (leverage).
During bear markets, they may decouple.

### Regime Detection

| Average Altcoin Beta | Regime |
|---|---|
| > 1.5 | Leverage (amplified moves) |
| 1.0 – 1.5 | Amplified |
| 0.5 – 1.0 | Normal |
| < 0.5 | Decoupled (altcoins doing their own thing) |

**In the code** (`risk_engine.py`):
```python
rolling_cov = coin_returns.rolling(60).cov(btc_returns)
rolling_var = btc_returns.rolling(60).var()
rolling_beta = rolling_cov / rolling_var
```

---

## 14. Kupiec Test

### The Problem

You built a VaR model that says "95% of days, losses won't exceed $3,000."
**But does it actually work?**

### The Kupiec Test

Look back at history. Count how many times losses **exceeded** VaR.

- **Expected**: 5% of days (if 95% VaR)
- **Actual**: Count the breaches

If breaches = 8% → Model is **underestimating** risk  
If breaches = 2% → Model is **overestimating** risk (too conservative)

### The Statistical Test

$$
LR = -2 \cdot \ln\left[\frac{p^n (1-p)^{T-n}}{\hat{p}^n (1-\hat{p})^{T-n}}\right]
$$

Where:
- $p$ = Expected breach rate (0.05 for 95% VaR)
- $\hat{p}$ = Actual breach rate (breaches / total days)
- $T$ = Total days
- $n$ = Number of breaches

**LR follows a Chi-squared distribution with 1 degree of freedom**

### Decision Rule

| LR | p-value | Result |
|---|---|---|
| < 3.841 | > 0.05 | ✅ Model PASSES (can't reject) |
| > 3.841 | < 0.05 | ❌ Model FAILS (rejected) |

### Why It Matters

- **Regulatory requirement** for banks
- **Credibility** — proves your model works
- **Improvement signal** — tells you when to recalibrate

**In the code** (`risk_engine.py`):
```python
n = breaches.sum()
T = len(breaches)
p_hat = n / T
p_exp = 0.05

LR = -2 * (n * np.log(p_exp) + (T-n) * np.log(1-p_exp)
           - n * np.log(p_hat) - (T-n) * np.log(1-p_hat))

kupiec_pass = LR < 3.841
```

---

## 15. Markov Regime-Switching

### The Idea

Markets have **regimes**:
- **Bull market** (low vol, trending up)
- **Bear market** (high vol, trending down)
- **Crisis** (extreme vol, panic)

A **Markov model** assumes:
1. You're always in one regime
2. Each day, you might **switch** regimes with some probability
3. Returns behave differently in each regime

### Transition Matrix

```
                To:  Normal   Crisis
From: Normal    [    0.95      0.05  ]
      Crisis    [    0.10      0.90  ]
```

Reading: "If you're in Normal, 95% chance you stay, 5% chance you switch to Crisis"

### Why "Markov"?

**Markov property** = The future only depends on **today's state**, not the history.

If you're in Crisis today, it doesn't matter how you got there — the probabilities are the same.

### In Monte Carlo

For each simulation step:
1. Check current regime
2. Roll dice to see if regime switches
3. Use **that regime's parameters** (mean, volatility) for the day
4. Generate the return

### Why It Matters

- **More realistic**: Captures volatility clustering
- **Stress scenarios**: Can simulate "stuck in crisis" paths
- **Risk management**: Shows fat tail behavior

**In the code** (`fetch_data.py`):
```python
# Simplified regime-switching MC
if current_regime == "normal":
    mu, sigma = 0.001, 0.03
    switch_prob = 0.05
else:  # crisis
    mu, sigma = -0.002, 0.08
    switch_prob = 0.10

if random.random() < switch_prob:
    current_regime = "crisis" if current_regime == "normal" else "normal"
    
daily_return = np.random.normal(mu, sigma)
```

---

## Summary — How It All Fits Together

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR CRYPTO RISK ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: Price data                                              │
│    │                                                            │
│    ▼                                                            │
│  Log Returns (clean, symmetric, additive)                       │
│    │                                                            │
│    ├──► Volatility ──► Annual risk estimate                    │
│    │                                                            │
│    ├──► Correlation / Covariance Matrix ──► Portfolio risk     │
│    │         │                                                  │
│    │         └──► PCA ──► Systemic risk score                  │
│    │                                                            │
│    ├──► Beta ──► Sensitivity to BTC                            │
│    │                                                            │
│    └──► Monte Carlo Simulations                                 │
│              │                                                  │
│              ├── GBM (constant vol)                             │
│              ├── GARCH (time-varying vol)                       │
│              ├── Historical (actual history)                    │
│              ├── Bootstrap (resampled blocks)                   │
│              ├── Copula (tail dependence)                       │
│              └── Regime-Switching (bull/bear)                   │
│                    │                                            │
│                    ▼                                            │
│              VaR / CVaR (downside risk)                         │
│                    │                                            │
│                    ▼                                            │
│              Kupiec Backtest (validation)                       │
│                                                                 │
│  OUTPUT: Risk-adjusted metrics (Sharpe, Sortino, Max DD)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Glossary

| Term | Simple Definition |
|---|---|
| **Annualize** | Scale daily metric to yearly (×√365 for vol) |
| **Backtest** | Test a model/strategy on historical data |
| **Cholesky** | Matrix decomposition for generating correlated random numbers |
| **CVaR** | Average loss when losses exceed VaR |
| **Drift** | Expected direction of returns |
| **Eigenvalue** | Importance of a principal component |
| **GARCH** | Model where volatility depends on recent shocks |
| **GBM** | Random walk model for prices |
| **Log Return** | ln(P_today / P_yesterday) |
| **Marginal** | Distribution of a single variable |
| **Monte Carlo** | Simulation using random sampling |
| **PCA** | Finding hidden factors in correlated data |
| **PSD** | Positive Semi-Definite (valid covariance matrix) |
| **Regime** | Market phase (bull/bear/crisis) |
| **Sharpe** | Return per unit of total risk |
| **Sortino** | Return per unit of downside risk |
| **Tail** | Extreme outcomes (left tail = losses) |
| **VaR** | Maximum loss at a confidence level |
| **Volatility** | Standard deviation of returns |

---

*Written to help you explain this project in interviews!*  
*"I implemented GBM, GARCH, and regime-switching Monte Carlo simulations  
to estimate portfolio VaR, validated with Kupiec backtesting..."*
