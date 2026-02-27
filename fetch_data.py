"""
DeFi Risk Analysis Tool
========================
A comprehensive toolkit for cryptocurrency portfolio risk analysis,
Monte Carlo simulations, and stress testing.

Author: DeFi Risk Team
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library
import re
import pathlib
import pickle
import time
import requests

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns

# API and data sources
from pycoingecko import CoinGeckoAPI
import praw

# NLP / Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Statistical modeling
import scipy.stats as stats
from arch import arch_model

# Local modules
from risk_analysis import calculate_betas, detect_volatility_regime, liquidity_multiplier
from config import COINS_TOP3, COINS_TOP5, COINS_TOP10, PRICE_CACHE_TTL
from visualization import (
    plot_regime_mc,
    plot_portfolio_mc_regimes,
    plot_stress_tests,
    plot_multi_coin_mc,
    plot_multi_coin_hist,
    plot_portfolio_mc_fan,
    plot_portfolio_mc_hist,
    plot_correlated_portfolio_mc_fan,
    plot_correlated_portfolio_mc_hist,
    plot_garch_simulation,
    plot_copula_simulation,
    plot_bootstrapped_simulation,
    plot_gbm_simulation,
    plot_historical_simulation
)

# =============================================================================
# CONFIGURATION
# =============================================================================

sns.set_style('whitegrid')
nltk.download("vader_lexicon", quiet=True)


# =============================================================================
# PRICE CACHE  (30-minute TTL — avoids hammering CoinGecko on every refresh)
# =============================================================================

_PRICE_CACHE_FILE = pathlib.Path("price_cache.pkl")


def _load_price_cache() -> dict | None:
    """Return cached coin_data dict if it exists and is fresh, else None."""
    if _PRICE_CACHE_FILE.exists():
        try:
            with open(_PRICE_CACHE_FILE, "rb") as fh:
                obj = pickle.load(fh)
            if time.time() - obj["ts"] < PRICE_CACHE_TTL:
                return obj["data"]
        except Exception:
            pass
    return None


def _save_price_cache(data: dict) -> None:
    try:
        with open(_PRICE_CACHE_FILE, "wb") as fh:
            pickle.dump({"ts": time.time(), "data": data}, fh)
    except Exception:
        pass


# Supported coin set presets (importable by app.py)
COIN_SETS = {
    "top3":  COINS_TOP3,
    "top5":  COINS_TOP5,
    "top10": COINS_TOP10,
}


# =============================================================================
# DATA FETCHING FUNCTIONS
# =============================================================================

def get_coin_data():
    """Fetch historical price and volume data for Ethereum."""
    cg = CoinGeckoAPI()
    eth_data = cg.get_coin_market_chart_by_id(id='ethereum', vs_currency='usd', days=365)
    
    prices_df = pd.DataFrame(eth_data["prices"], columns=["timestamp", "price"])
    prices_df['date'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
    prices_df.set_index('date', inplace=True)
    prices_df.drop('timestamp', axis=1, inplace=True)

    volume_df = pd.DataFrame(eth_data['total_volumes'], columns=['timestamp', 'volume'])
    volume_df['date'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
    volume_df.set_index('date', inplace=True)
    volume_df.drop('timestamp', axis=1, inplace=True)

    df = prices_df.merge(volume_df, left_index=True, right_index=True)
    df.sort_index(ascending=True, inplace=True)

    df["returns"] = df["price"].pct_change()
    df["volatility_7d"] = df["returns"].rolling(7).std()
    df["ma_30"] = df["price"].rolling(30).mean()
    df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
    return df


def get_multi_coin_data(coin_set: str = "top3", force_refresh: bool = False):
    """
    Fetch historical price and volume data for a set of coins.

    Parameters
    ----------
    coin_set : "top3" | "top5" | "top10"
        Which preset of coins to load.  Defaults to the original 3-coin set.
    force_refresh : bool
        When True, bypass the 30-minute disk cache and re-fetch from CoinGecko.

    Returns
    -------
    dict  {symbol: DataFrame}   same schema as before
    """
    coins = COIN_SETS.get(coin_set, COINS_TOP3)

    # ── Check disk cache ────────────────────────────────────────────────────
    if not force_refresh:
        cached = _load_price_cache()
        if cached is not None:
            # Return cache only if it covers all requested symbols
            if all(sym in cached for sym in coins):
                return {sym: cached[sym] for sym in coins}

    cg = CoinGeckoAPI()
    coin_data = {}

    for symbol, coin_id in coins.items():
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=365)
        
        prices_df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        prices_df['date'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
        prices_df.set_index('date', inplace=True)
        prices_df.drop('timestamp', axis=1, inplace=True)
        
        volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        volume_df['date'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('date', inplace=True)
        volume_df.drop('timestamp', axis=1, inplace=True)
        
        df = prices_df.merge(volume_df, left_index=True, right_index=True)
        df.sort_index(ascending=True, inplace=True)
        
        df["returns"] = df["price"].pct_change()
        df["volatility_7d"] = df["returns"].rolling(7).std()
        df["ma_30"] = df["price"].rolling(30).mean()
        df["log_returns"] = np.log(df["price"] / df["price"].shift(1))
        df["volume_pct_rank"] = df["volume"].rolling(90).rank(pct=True)

        coin_data[symbol] = df.dropna()

    # ── Persist to disk cache ────────────────────────────────────────────────
    _save_price_cache(coin_data)

    return coin_data


def get_macro_data_fred(series_id, api_key):
    """
    Fetch macro data from FRED API.
    
    Args:
        series_id: FRED series code (e.g., DFF for Fed Funds Rate)
        api_key: Your FRED API key
    
    Returns:
        Latest value as float
    """
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception("FRED API error")
    data = resp.json()
    obs = data.get("observations", [])
    if not obs:
        return None
    latest = obs[-1]
    return float(latest["value"])


# =============================================================================
# SIMULATION MODELS
# =============================================================================

def garch_simulation(coin_data, coin, horizon_days=90, num_sim=1000):
    """
    GARCH(1,1) volatility modeling for a single coin.
    
    Uses GARCH model to capture volatility clustering in simulated paths.
    """
    df = coin_data[coin]
    log_returns = df["log_returns"].dropna()
    last_price = df["price"].iloc[-1]
    
    am = arch_model(log_returns, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    
    all_sim = []
    for i in range(num_sim):
        sim_vol = res.simulate(res.params, horizon_days)
        sim_returns = sim_vol["data"]
        price_path = last_price * np.exp(np.cumsum(sim_returns))
        all_sim.append(price_path)
    
    sim_df = pd.DataFrame(all_sim).T
    sim_df["mean"] = sim_df.mean(axis=1)
    sim_df["std"] = sim_df.std(axis=1)
    percentiles = sim_df.drop(columns=["mean", "std"]).quantile([0.05, 0.5, 0.95], axis=1).T
    percentiles.columns = ["P5", "Median", "P95"]
    
    plot_garch_simulation(sim_df, percentiles, last_price, coin)
    
    final_prices = sim_df.iloc[-1].drop(["mean", "std"]).values
    median_terminal = np.median(final_prices)
    p5_terminal = np.percentile(final_prices, 5)
    p95_terminal = np.percentile(final_prices, 95)
    print(f"{coin} GARCH Simulation — Median: {median_terminal:.2f}, P5: {p5_terminal:.2f}, P95: {p95_terminal:.2f}")


def copula_correlation_simulation(coin_data, coins, horizon_days=90, num_sim=1000):
    """
    Copula-based correlation simulation for multiple coins.
    
    Uses Gaussian copula to model dependencies between assets.
    """
    returns_matrix = np.column_stack([coin_data[c]["log_returns"].dropna() for c in coins])
    last_prices = {c: coin_data[c]["price"].iloc[-1] for c in coins}
    # Fit marginal distributions
    marginals = [stats.norm.fit(returns_matrix[:,i]) for i in range(len(coins))]
    # Estimate empirical correlation
    corr = np.corrcoef(returns_matrix.T)
    # Gaussian copula
    all_sim = []
    for i in range(num_sim):
        # Simulate correlated normals
        mvnorm = np.random.multivariate_normal(np.zeros(len(coins)), corr, horizon_days)
        # Convert to uniform via CDF
        uniforms = stats.norm.cdf(mvnorm)
        # Inverse CDF to marginals
        sim_returns = np.column_stack([
            stats.norm.ppf(uniforms[:,j], loc=marginals[j][0], scale=marginals[j][1])
            for j in range(len(coins))
        ])
        # Price paths
        price_paths = {c: last_prices[c] * np.exp(np.cumsum(sim_returns[:,i])) for i,c in enumerate(coins)}
        portfolio = sum(price_paths[c] for c in coins)
        all_sim.append(portfolio)
    sim_df = pd.DataFrame(all_sim).T
    sim_df["mean"] = sim_df.mean(axis=1)
    sim_df["std"] = sim_df.std(axis=1)
    percentiles = sim_df.drop(columns=["mean","std"]).quantile([0.05,0.5,0.95], axis=1).T
    percentiles.columns = ["P5", "Median", "P95"]
    
    plot_copula_simulation(sim_df, percentiles, sum(last_prices.values()), coins)
    final_prices = sim_df.iloc[-1].drop(["mean","std"]).values
    median_terminal = np.median(final_prices)
    p5_terminal = np.percentile(final_prices,5)
    p95_terminal = np.percentile(final_prices,95)
    print(f"Copula Simulation — Median: {median_terminal:.2f}, P5: {p5_terminal:.2f}, P95: {p95_terminal:.2f}")
def bootstrapped_path_simulation(coin_data, coin, horizon_days=90, num_sim=1000, block_size=10):
    """
    Bootstrapped path simulation: resample blocks of returns to preserve volatility clustering.
    """
    df = coin_data[coin]
    returns = df["log_returns"].dropna().values
    last_price = df["price"].iloc[-1]
    n_blocks = int(np.ceil(horizon_days / block_size))
    all_sim = []
    for i in range(num_sim):
        path_returns = []
        for _ in range(n_blocks):
            start = np.random.randint(0, len(returns) - block_size)
            block = returns[start:start+block_size]
            path_returns.extend(block)
        path_returns = np.array(path_returns[:horizon_days])
        price_path = last_price * np.exp(np.cumsum(path_returns))
        all_sim.append(price_path)
    sim_df = pd.DataFrame(all_sim).T
    sim_df["mean"] = sim_df.mean(axis=1)
    sim_df["std"] = sim_df.std(axis=1)
    percentiles = sim_df.drop(columns=["mean","std"]).quantile([0.05,0.5,0.95], axis=1).T
    percentiles.columns = ["P5", "Median", "P95"]
    
    plot_bootstrapped_simulation(sim_df, percentiles, last_price, coin)
    final_prices = sim_df.iloc[-1].drop(["mean","std"]).values
    median_terminal = np.median(final_prices)
    p5_terminal = np.percentile(final_prices,5)
    p95_terminal = np.percentile(final_prices,95)
    print(f"{coin} Bootstrapped Simulation — Median: {median_terminal:.2f}, P5: {p5_terminal:.2f}, P95: {p95_terminal:.2f}")

def gbm_simulation(coin_data, coin, horizon_days=90, num_sim=1000):
    """
    Geometric Brownian Motion (GBM) simulation for a single coin.
    """
    df = coin_data[coin]
    log_returns = df["log_returns"].dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()
    last_price = df["price"].iloc[-1]
    dt = 1
    all_sim = []
    for i in range(num_sim):
        W = np.random.normal(0, np.sqrt(dt), horizon_days).cumsum()
        t = np.arange(1, horizon_days+1)
        price_path = last_price * np.exp((mu - 0.5 * sigma**2) * t * dt + sigma * W)
        all_sim.append(price_path)
    sim_df = pd.DataFrame(all_sim).T
    sim_df["mean"] = sim_df.mean(axis=1)
    sim_df["std"] = sim_df.std(axis=1)
    percentiles = sim_df.drop(columns=["mean","std"]).quantile([0.05,0.5,0.95], axis=1).T
    percentiles.columns = ["P5", "Median", "P95"]
    
    plot_gbm_simulation(sim_df, percentiles, last_price, coin)
    final_prices = sim_df.iloc[-1].drop(["mean","std"]).values
    median_terminal = np.median(final_prices)
    p5_terminal = np.percentile(final_prices,5)
    p95_terminal = np.percentile(final_prices,95)
    print(f"{coin} GBM Simulation — Median: {median_terminal:.2f}, P5: {p5_terminal:.2f}, P95: {p95_terminal:.2f}")
def historical_simulation(coin_data, coin, horizon_days=90, num_sim=1000):
    """
    Non-parametric historical simulation for a single coin.
    Resamples real historical returns for forward price paths.
    """
    df = coin_data[coin]
    historical_returns = df["log_returns"].dropna()
    last_price = df["price"].iloc[-1]
    all_sim = []
    for i in range(num_sim):
        sample_returns = historical_returns.sample(n=horizon_days, replace=True).values
        price_path = last_price * np.exp(np.cumsum(sample_returns))
        all_sim.append(price_path)
    sim_df = pd.DataFrame(all_sim).T
    sim_df["mean"] = sim_df.mean(axis=1)
    sim_df["std"] = sim_df.std(axis=1)
    percentiles = sim_df.drop(columns=["mean","std"]).quantile([0.05,0.5,0.95], axis=1).T
    percentiles.columns = ["P5", "Median", "P95"]
    
    plot_historical_simulation(sim_df, percentiles, last_price, coin)
    # Terminal day analysis
    final_prices = sim_df.iloc[-1].drop(["mean","std"]).values
    median_terminal = np.median(final_prices)
    p5_terminal = np.percentile(final_prices,5)
    p95_terminal = np.percentile(final_prices,95)
    print(f"{coin} Historical Simulation — Median: {median_terminal:.2f}, P5: {p5_terminal:.2f}, P95: {p95_terminal:.2f}")

def regime_switching_monte_carlo(
    coin_data,
    units,
    regimes=None,
    transition_matrix=None,
    horizon_days=90,
    num_sim=1000
):
    """
    Markov Regime Switching Monte Carlo simulation.
    
    Each regime has its own mean, covariance, and volatility.
    Simulation switches regimes using a Markov process.
    """
    coins = list(coin_data.keys())

    # Default regime parameters if not provided
    if regimes is None:
        regimes = {
            "bull": {
                "mean": np.array([0.0015]*len(coins)),
                "cov": np.cov(np.column_stack([coin_data[c]["log_returns"] for c in coins]).T),
                "vol": np.array([0.03]*len(coins))
            },
            "bear": {
                "mean": np.array([-0.001]*len(coins)),
                "cov": np.cov(np.column_stack([coin_data[c]["log_returns"] for c in coins]).T)*1.5,
                "vol": np.array([0.05]*len(coins))
            },
            "sideways": {
                "mean": np.array([0.0]*len(coins)),
                "cov": np.cov(np.column_stack([coin_data[c]["log_returns"] for c in coins]).T)*0.7,
                "vol": np.array([0.02]*len(coins))
            },
            "crisis": {
                "mean": np.array([-0.003]*len(coins)),
                "cov": np.cov(np.column_stack([coin_data[c]["log_returns"] for c in coins]).T)*2.5,
                "vol": np.array([0.09]*len(coins))
            }
        }

    regime_names = list(regimes.keys())
    n_regimes = len(regime_names)

    # Default Markov transition matrix (rows sum to 1)
    if transition_matrix is None:
        transition_matrix = np.array([
            [0.85, 0.05, 0.08, 0.02],  # bull
            [0.08, 0.80, 0.08, 0.04],  # bear
            [0.10, 0.10, 0.75, 0.05],  # sideways
            [0.10, 0.20, 0.10, 0.60],  # crisis
        ])

    # Initial regime probabilities
    regime_probs = np.array([0.4, 0.2, 0.3, 0.1])

    last_prices = {c: coin_data[c]["price"].iloc[-1] for c in coins}
    portfolio_paths = []

    for sim in range(num_sim):
        regime_idx = np.random.choice(n_regimes, p=regime_probs)
        regime_path = [regime_idx]
        prices = last_prices.copy()
        path = [sum(prices[c]*units.get(c,0) for c in coins)]

        for t in range(horizon_days):
            # Regime transition
            regime_idx = np.random.choice(n_regimes, p=transition_matrix[regime_idx])
            regime_path.append(regime_idx)
            regime = regimes[regime_names[regime_idx]]
            mean = regime["mean"]
            cov = regime["cov"]
            # Simulate returns for this day
            returns = np.random.multivariate_normal(mean, cov)
            for i, c in enumerate(coins):
                prices[c] *= np.exp(returns[i])
            path.append(sum(prices[c]*units.get(c,0) for c in coins))

        portfolio_paths.append(path)

    df_portfolio = pd.DataFrame(portfolio_paths).T
    df_portfolio.index.name = "Day"
    return df_portfolio


# =============================================================================
# MONTE CARLO PORTFOLIO FUNCTIONS
# =============================================================================

def worker():
    """
    Main entry point — runs the full analysis pipeline:
      1. Fetch multi-coin data
      2. Get user holdings
      3. Compute betas
      4. Run stress tests
      5. Run normal + crisis Monte Carlo
      6. Run regime-switching Monte Carlo
    """

    # Step 1: Fetch data
    print("\n" + "=" * 60)
    print("  DeFi Risk Analysis Tool")
    print("=" * 60)
    print("\nFetching market data for BTC, ETH, SOL ...")
    coin_data = get_multi_coin_data()
    print("Data loaded.\n")

    # Step 2: User holdings
    print("Enter how many units you hold per coin.")
    print("Example: ETH=2,BTC=0.5,SOL=10")
    units_input = input("Your holdings: ")
    units = {}
    for part in units_input.split(","):
        if "=" in part:
            coin, amount = part.split("=", 1)
            try:
                units[coin.strip().upper()] = float(amount.strip())
            except ValueError:
                print(f"Invalid amount for {coin.strip()}: {amount.strip()}, skipping.")

    if not units:
        print("No valid holdings entered. Exiting.")
        return coin_data

    # Step 3: Compute betas
    returns_df = pd.DataFrame({
        coin: coin_data[coin]["log_returns"]
        for coin in coin_data if "log_returns" in coin_data[coin]
    }).dropna()
    betas = calculate_betas(returns_df, benchmark="BTC")
    print(f"\nBetas (vs BTC): {betas}")

    # Step 4: Portfolio summary
    portfolio_df = pd.DataFrame({"Units": units})
    portfolio_df.index = [c.upper() for c in portfolio_df.index]
    portfolio_df["Price"] = [
        coin_data[coin]["price"].iloc[-1]
        if coin in coin_data else float("nan")
        for coin in portfolio_df.index
    ]
    portfolio_df["Value"] = portfolio_df["Units"] * portfolio_df["Price"]
    current_portfolio_value = portfolio_df["Value"].sum()
    if current_portfolio_value != 0:
        portfolio_df["Weight"] = portfolio_df["Value"] / current_portfolio_value
    else:
        portfolio_df["Weight"] = 0
    print(f"\nPortfolio value: ${current_portfolio_value:,.2f}")
    print(portfolio_df.to_string())

    # Step 5: Stress testing
    print("\nRunning stress tests ...")
    shock_scenarios = {
        "Crash -30%": -0.30,
        "Severe Crash -50%": -0.50,
        "Bull Rally +40%": 0.40,
        "Mild Correction -15%": -0.15,
    }
    df_stress = stress_test_portfolio(
        coin_data,
        portfolio_df["Units"].to_dict(),
        shock_scenarios,
        betas,
    )
    plot_stress_tests(df_stress)

    # Step 6: Normal + Crisis Monte Carlo
    print("\nRunning Normal regime Monte Carlo ...")
    df_portfolio_normal = monte_carlo_portfolio(
        coin_data, units, betas=betas, regime="normal",
        horizon_days=90, num_sim=1000,
    )
    print("\nRunning Crisis regime Monte Carlo ...")
    df_portfolio_crisis = monte_carlo_portfolio(
        coin_data, units, betas=betas, regime="crisis",
        horizon_days=90, num_sim=1000,
    )
    plot_portfolio_mc_regimes(
        df_portfolio_normal, df_portfolio_crisis, current_portfolio_value
    )

    # Step 7: Regime-switching Monte Carlo
    print("\nRunning Markov Regime Switching Monte Carlo simulation...")
    df_regime_mc = regime_switching_monte_carlo(
        coin_data, units, horizon_days=90, num_sim=500
    )
    plot_regime_mc(df_regime_mc)

    print("\nAnalysis complete.")
    return coin_data

#monte carlo portoflio creation new with volatility 
def monte_carlo_portfolio(

    coin_data,
    units,
    betas=None,
    regime="normal",
    horizon_days=90,
    num_sim=1000
):

    coins = list(coin_data.keys())

    # --- Returns matrix ---
    returns_matrix = np.column_stack([
        coin_data[c]["log_returns"].dropna()
        for c in coins
    ])


    mean_returns = returns_matrix.mean(axis=0)
    vols = returns_matrix.std(axis=0)

    # --- Macro overlay ---
    macro_drift_adjustment = {
        "rate_hikes": -0.02,
        "qe": +0.03,
        "dxy_up": -0.01,
        "liquidity_up": +0.02,
        "cpi_surprise": +0.01
    }
    # Example: user_macro_view = "rate_hikes" or "qe"
    user_macro_view = None
    # Optionally, fetch macro data and set user_macro_view
    # Example: DFF (Fed Funds Rate), DXY (Dollar Index)
    # api_key = "YOUR_FRED_API_KEY"
    # fed_rate = get_macro_data_fred("DFF", api_key)
    # dxy = get_macro_data_fred("DTWEXBGS", api_key)
    # if fed_rate > 4.0: user_macro_view = "rate_hikes"
    # if dxy > 105: user_macro_view = "dxy_up"

    # For now, let user pick macro scenario
    print("Macro scenario options: rate_hikes, qe, dxy_up, liquidity_up, cpi_surprise, none")
    macro_input = input("Choose macro scenario (or 'none'): ").strip()
    if macro_input in macro_drift_adjustment:
        mean_returns += macro_drift_adjustment[macro_input]

    drift = mean_returns - 0.5 * vols**2

    cov_matrix = np.cov(returns_matrix.T)

    # Volatility regime multiplier
    vol_regime_multiplier = {
        "low": 0.7,
        "normal": 1.0,
        "high": 1.8,
        "crisis": 2.3
    }
    cov_matrix *= vol_regime_multiplier.get(regime, 1.0)

    # --- Last prices ---
    last_prices = {
        c: coin_data[c]["price"].iloc[-1]
        for c in coins
    }

    # Compute betas if not provided
    if betas is None:
        returns_df = pd.DataFrame({
            coin: coin_data[coin]["log_returns"] for coin in coins
        }).dropna()
        betas = calculate_betas(returns_df, benchmark="BTC")

    # Liquidity multipliers
    liquidity_mults = {}
    for c in coins:
        if "volume_pct_rank" in coin_data[c]:
            score = coin_data[c]["volume_pct_rank"].iloc[-1]
            liquidity_mults[c] = liquidity_multiplier(score)
        else:
            liquidity_mults[c] = 1.0

    portfolio_paths = []

    for _ in range(num_sim):
        correlated_returns = np.random.multivariate_normal(
            drift,
            cov_matrix,
            horizon_days
        )
        # Apply BTC structural beta effect and liquidity multiplier
        btc_index = coins.index("BTC")

        for i in range(horizon_days):
            btc_return = correlated_returns[i, btc_index]
            for j, coin in enumerate(coins):
                if coin != "BTC":
                    beta = betas.get(coin, 1.0)
                    liq_mult = liquidity_mults.get(coin, 1.0)
                    # Shock propagation: BTC move * beta * liquidity
                    correlated_returns[i, j] = (
                        btc_return * beta * liq_mult
                        + 0.3 * correlated_returns[i, j]  # keep some idiosyncratic noise
                    )

        df_returns = pd.DataFrame(
            correlated_returns,
            columns=coins
        )

        price_paths = pd.DataFrame({
            c: last_prices[c] *
               np.exp(df_returns.cumsum())
            for c in coins
        })

        portfolio_price = sum(
            price_paths[c] * units[c]
            for c in coins
        )

        portfolio_paths.append(portfolio_price)

    df_portfolio = pd.DataFrame(portfolio_paths).T

    return df_portfolio


# =============================================================================
# MULTI-COIN MONTE CARLO
# =============================================================================

def multi_coin_monte_carlo(coin_data, horizon_days=90, num_sim=1000):
    """
    coin_data: dict with keys=coin symbols, values=DataFrames with ["price", "returns"]
    horizon_days: simulation days
    num_sim: number of Monte Carlo simulations per coin
    """
    
    results = {}
    
    for coin, df in coin_data.items():
        n_reps = horizon_days
        avg_return = df["log_returns"].mean()
        std_dev = df["log_returns"].std()
        last_price = df["price"].iloc[-1]

        all_sim_no_returns = []
        for i in range(num_sim):
            simulated_return = np.random.normal(avg_return, std_dev, n_reps)
            df_sim = pd.DataFrame(np.exp(simulated_return.cumsum()) * last_price, columns=[f"sim{i}"])
            all_sim_no_returns.append(df_sim)
        
        df_joined = pd.concat(all_sim_no_returns, axis=1)
        df_joined["mean"] = df_joined.mean(axis=1)
        df_joined["std"] = df_joined.std(axis=1)

        # Percentiles for fan chart
        percentiles = df_joined.drop(columns=["mean","std"]).quantile([0.05,0.5,0.95], axis=1).T
        percentiles.columns = ["P5","Median","P95"]

        # Terminal day
        final_prices = df_joined.iloc[-1].drop(["mean","std"]).values
        median_terminal = np.median(final_prices)
        p5_terminal = np.percentile(final_prices,5)
        p95_terminal = np.percentile(final_prices,95)

        plot_multi_coin_mc(df_joined, percentiles, last_price, coin)

        # Ask for target price per coin to compute probabilities
        target_price = float(input(f"Enter target price for {coin} after {horizon_days} days: "))
        prob_below_today = np.mean(final_prices < last_price)
        prob_above_target = np.mean(final_prices > target_price)

        plot_multi_coin_hist(
            final_prices, last_price, target_price,
            median_terminal, p5_terminal, p95_terminal,
            prob_below_today, prob_above_target, coin,
        )

        # VaR
        returns_terminal = (final_prices / last_price) - 1
        var_5 = np.percentile(returns_terminal, 5)
        cvar_5 = returns_terminal[returns_terminal <= var_5].mean()

        print(f"{coin} — VaR 5%: {var_5:.2%}, Expected Shortfall 5%: {cvar_5:.2%}")

        # Save results
        results[coin] = {
            "df_joined": df_joined,
            "percentiles": percentiles,
            "final_prices": final_prices,
            "prob_below_today": prob_below_today,
            "prob_above_target": prob_above_target,
            "var_5": var_5,
            "cvar_5": cvar_5,
        }

    return results


# =============================================================================
# PORTFOLIO MONTE CARLO FUNCTIONS
# =============================================================================

def portfolio_monte_carlo(coin_data, horizon_days=90, num_sim=1000):
    """
    coin_data: dict with keys=coin symbols, values=DataFrames with 'price' and 'returns'
    horizon_days: number of days to simulate
    num_sim: number of Monte Carlo simulations per coin
    """

    # --- Step 1: Ask user for portfolio weights ---
    print("Enter your portfolio allocation (sum must be 1). Example: ETH=0.5,BTC=0.3,SOL=0.2")
    allocation_input = input("Portfolio weights: ")
    allocation = {}
    for part in allocation_input.split(","):
        coin, weight = part.split("=")
        allocation[coin.strip().upper()] = float(weight.strip())

    # Verify sum of weights
    if not np.isclose(sum(allocation.values()), 1.0):
        raise ValueError("Portfolio weights must sum to 1")

    # --- Step 2: Simulate each coin ---
    simulated_prices = {}
    last_prices = {}
    for coin, df in coin_data.items():
        n_reps = horizon_days
        avg_return = df["log_returns"].mean()
        std_dev = df["log_returns"].std()
        last_price = df["price"].iloc[-1]
        last_prices[coin] = last_price

        coin_sims = []
        for i in range(num_sim):
            simulated_return = np.random.normal(avg_return, std_dev, n_reps)
            # cumulative compounded price
            df_sim = pd.DataFrame(np.exp(simulated_return.cumsum()) * last_price, columns=[f"sim{i}"])
            coin_sims.append(df_sim)
        simulated_prices[coin] = pd.concat(coin_sims, axis=1)

    # --- Step 3: Compute weighted portfolio paths ---
    portfolio_sims = []
    for i in range(num_sim):
        # sum weighted prices for simulation i
        portfolio_sim_i = sum(
            simulated_prices[coin].iloc[:, i] * allocation.get(coin, 0)
            for coin in simulated_prices
        )
        portfolio_sims.append(portfolio_sim_i)

    df_portfolio = pd.concat(portfolio_sims, axis=1)
    df_portfolio["mean"] = df_portfolio.mean(axis=1)
    df_portfolio["std"] = df_portfolio.std(axis=1)

    # --- Step 4: Portfolio percentiles ---
    percentiles = df_portfolio.drop(columns=["mean","std"]).quantile([0.05,0.5,0.95], axis=1).T
    percentiles.columns = ["P5","Median","P95"]

    plot_portfolio_mc_fan(df_portfolio, percentiles)

    # Terminal distribution & histogram
    final_prices = df_portfolio.iloc[-1].drop(["mean","std"]).values
    median_terminal = np.median(final_prices)
    p5_terminal = np.percentile(final_prices, 5)
    p95_terminal = np.percentile(final_prices, 95)

    target_price = float(input("Enter your target portfolio price after horizon: "))
    last_portfolio_price = sum(last_prices[coin]*allocation.get(coin,0) for coin in last_prices)
    prob_below_today  = np.mean(final_prices < last_portfolio_price)
    prob_above_target = np.mean(final_prices > target_price)

    plot_portfolio_mc_hist(final_prices, last_portfolio_price, target_price, median_terminal, p5_terminal, p95_terminal, prob_below_today, prob_above_target)

    # Portfolio VaR / CVaR
    returns_terminal = (final_prices / last_portfolio_price) - 1
    var_5  = np.percentile(returns_terminal, 5)
    cvar_5 = returns_terminal[returns_terminal <= var_5].mean()
    print(f"Portfolio VaR 5%: {var_5:.2%}, Expected Shortfall 5%: {cvar_5:.2%}")

    return df_portfolio, percentiles


def correlated_portfolio_monte_carlo(coin_data, horizon_days=90, num_sim=1000):
    """
    Correlated portfolio Monte Carlo simulation.
    
    Simulates portfolio value using correlated daily returns based on
    historical covariance matrix.
    """

    # --- Step 1: Portfolio weights ---
    print("Enter your portfolio allocation (sum must be 1). Example: ETH=0.5,BTC=0.3,SOL=0.2")
    allocation_input = input("Portfolio weights: ")
    allocation = {}
    for part in allocation_input.split(","):
        coin, weight = part.split("=")
        allocation[coin.strip().upper()] = float(weight.strip())

    if not np.isclose(sum(allocation.values()), 1.0):
        raise ValueError("Portfolio weights must sum to 1")

    # --- Step 2: Prepare returns matrix and correlations ---
    coins = list(allocation.keys())
    returns_matrix = np.column_stack([coin_data[coin]["log_returns"].dropna() for coin in coins])
    mean_returns = returns_matrix.mean(axis=0)
    cov_matrix = np.cov(returns_matrix.T)  # covariance matrix including correlations

    last_prices = {coin: coin_data[coin]["price"].iloc[-1] for coin in coins}

    # --- Step 3: Simulate correlated daily returns ---
    portfolio_sims = []
    for sim in range(num_sim):
        # generate correlated daily returns
        correlated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=horizon_days)
        df_sim = pd.DataFrame(correlated_returns, columns=coins)

        # compound prices per coin
        price_paths = pd.DataFrame({
            coin: last_prices[coin] *
                np.exp(df_sim[coin].cumsum())
            for coin in coins
        })

        # portfolio weighted sum
        portfolio_price = sum(price_paths[coin]*allocation[coin] for coin in coins)
        portfolio_sims.append(portfolio_price)

    df_portfolio = pd.concat(portfolio_sims, axis=1)
    df_portfolio["mean"] = df_portfolio.mean(axis=1)
    df_portfolio["std"] = df_portfolio.std(axis=1)

    # --- Step 4: Portfolio percentiles ---
    percentiles = df_portfolio.drop(columns=["mean","std"]).quantile([0.05,0.5,0.95], axis=1).T
    percentiles.columns = ["P5","Median","P95"]

    plot_correlated_portfolio_mc_fan(df_portfolio, percentiles)

    # --- Step 6: Terminal distribution & histogram ---
    final_prices = df_portfolio.iloc[-1].drop(["mean","std"]).values
    median_terminal = np.median(final_prices)
    p5_terminal = np.percentile(final_prices, 5)
    p95_terminal = np.percentile(final_prices, 95)

    target_price = float(input("Enter your target portfolio price after horizon: "))
    last_portfolio_price = sum(last_prices[coin]*allocation[coin] for coin in coins)
    prob_below_today  = np.mean(final_prices < last_portfolio_price)
    prob_above_target = np.mean(final_prices > target_price)

    plot_correlated_portfolio_mc_hist(final_prices, last_portfolio_price, target_price, median_terminal, p5_terminal, p95_terminal, prob_below_today, prob_above_target)

    # --- Step 7: Portfolio VaR / CVaR ---
    returns_terminal = (final_prices / last_portfolio_price) - 1
    var_5  = np.percentile(returns_terminal, 5)
    cvar_5 = returns_terminal[returns_terminal <= var_5].mean()
    print(f"Portfolio VaR 5%: {var_5:.2%}, Expected Shortfall 5%: {cvar_5:.2%}")

    return df_portfolio, percentiles


# =============================================================================
# STRESS TESTING
# =============================================================================

def stress_test_portfolio(coin_data, allocation, shock_scenarios, betas, horizon_days=90):
    """
    Stress test portfolio under various shock scenarios.
    
    Args:
        allocation: dict of coin -> units
        shock_scenarios: dict of scenario_name -> shock_return
        betas: dict of coin -> beta values
    """

    coins = list(allocation.keys())

    # --- Returns matrix ---
    returns_matrix = np.column_stack([
        coin_data[coin]["log_returns"].dropna()
        for coin in coins
    ])

    mean_returns = returns_matrix.mean(axis=0)
    cov_matrix = np.cov(returns_matrix.T)

    # --- Last prices ---
    last_prices = {
        coin: coin_data[coin]["price"].iloc[-1]
        for coin in coins
    }

    results = {}

    # =========================================================
    # 1️⃣ BASELINE / MEAN SCENARIO (no shock)
    # =========================================================

    correlated_returns = np.random.multivariate_normal(
        mean_returns,
        cov_matrix,
        size=horizon_days
    )

    df_returns = pd.DataFrame(correlated_returns, columns=coins)

    price_paths = pd.DataFrame({
        coin: last_prices[coin] *
            np.exp(df_returns[coin].cumsum())
        for coin in coins
    })

    portfolio_price = sum(
        price_paths[coin] * allocation[coin]
        for coin in coins
    )

    results["Baseline (Mean Path)"] = portfolio_price

    # =========================================================
    # 2️⃣ SHOCK SCENARIOS
    # =========================================================

    for scenario, shock in shock_scenarios.items():

        correlated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            size=horizon_days
        )

        df_returns = pd.DataFrame(correlated_returns, columns=coins)

        # Apply shock on Day 1
        # BTC shock propagation using betas
        for i, coin in enumerate(coins):
            if coin == "BTC":
                df_returns.iloc[0, i] = shock
            else:
                beta = betas.get(coin, 1.0)
                df_returns.iloc[0, i] = shock * beta

        price_paths = pd.DataFrame({
            coin: last_prices[coin] *
                  (1 + df_returns[coin]).cumprod()
            for coin in coins
        })

        portfolio_price = sum(
            price_paths[coin] * allocation[coin]
            for coin in coins
        )

        results[scenario] = portfolio_price

    # Combine scenarios
    df_stress = pd.concat(results, axis=1)

    return df_stress


# =============================================================================
# SINGLE COIN MONTE CARLO
# =============================================================================

def monte_carlo_sim(df):
    n_reps= 90 
    num_sim = 1000
    avg_return = df["log_returns"].mean()
    std_dev = np.std(df["log_returns"])
    last_price = df["price"].iloc[-1]

    all_sim = []
    all_sim_no_returns = []
    for i in range(num_sim): 
        simulated_return = np.random.normal(avg_return, std_dev, n_reps).round(5)
        df_sim = pd.DataFrame(index=range(n_reps),data={f"return_sim{i}":simulated_return})
        df_sim[f"price_sim{i}"] = last_price * np.exp(df_sim[f"return_sim{i}"].cumsum())
        all_sim.append(df_sim)
        df_sim.drop(columns={f"return_sim{i}"},inplace=True)
        all_sim_no_returns.append(df_sim)

    
    df_with_return =pd.concat(all_sim, axis = 1)
    df_joined = pd.concat(all_sim_no_returns, axis = 1)
    df_joined["mean"] = df_joined.mean(axis=1)
    df_joined["std"] = df_joined.std(axis=1)
    #compute percentiles 
    percentiles = df_joined.quantile([0.05, 0.5, 0.95], axis=1)
    percentiles = percentiles.T
    percentiles.columns = ["P5", "Median", "P95"]

    plot_multi_coin_mc(df_joined, percentiles, last_price, "ETH")

    #last day price analysis
    last_price_sim_row = df_joined.iloc[-1]
    last_price_sim_mean = last_price_sim_row["mean"]
    final_prices = last_price_sim_row.drop(labels=["mean", "std"])
    last_price_sim_median = final_prices.median()
    p5 = np.percentile(final_prices,5)
    p95 = np.percentile(final_prices,95)

    target_price = float(input("Where do you think the price will be after 90 days?"))
    prob_below_today = np.mean(final_prices < last_price)       # Fraction below current price
    prob_above_target = np.mean(final_prices > target_price)

    plot_multi_coin_hist(final_prices, last_price, target_price, last_price_sim_median, p5, p95, prob_below_today, prob_above_target, "ETH")

    #--- VaR analysis 

    returns_terminal = (final_prices / last_price) - 1
    #worst 5%
    var_5 = np.percentile(returns_terminal, 5)
    cvar_5 = returns_terminal[
        returns_terminal <= var_5
    ].mean()

    print(f"VaR 5%: {var_5:.2%}")
    print(f"Expected Shortfall 5%: {cvar_5:.2%}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


# =============================================================================
# DEFI ON-CHAIN DATA — DefiLlama (free, no API key)
# =============================================================================

def get_defi_tvl(top_n: int = 10) -> pd.DataFrame:
    """
    Fetch Top-N DeFi protocols by Total Value Locked from DefiLlama.

    Free, no API key required.  Returns an empty DataFrame on any error.

    Parameters
    ----------
    top_n : int  — number of protocols to return (default 10)

    Returns
    -------
    pd.DataFrame  columns: Protocol, Chain, Category, TVL ($B), Change 1d (%)
    """
    _LLAMA_URL = "https://api.llama.fi/protocols"
    try:
        resp = requests.get(_LLAMA_URL, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for p in sorted(data, key=lambda x: x.get("tvl", 0) or 0, reverse=True)[:top_n]:
            rows.append({
                "Protocol":       p.get("name", ""),
                "Chain":          p.get("chain", "Multi") or "Multi",
                "Category":       p.get("category", ""),
                "TVL ($B)":       round((p.get("tvl") or 0) / 1e9, 3),
                "Change 1d (%)":  round(p.get("change_1d") or 0, 2),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(
            columns=["Protocol", "Chain", "Category", "TVL ($B)", "Change 1d (%)"]
        )


def get_chain_tvl_history(chain: str = "Ethereum") -> pd.Series:
    """
    Fetch daily TVL history for a single chain from DefiLlama.

    Parameters
    ----------
    chain : str  — chain name as recognised by DefiLlama (e.g. 'Ethereum', 'Solana', 'BSC')

    Returns
    -------
    pd.Series  index = date, values = TVL in USD
    """
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df = df.set_index("date")["totalLiquidityUSD"]
        df.name = f"{chain}_TVL"
        return df.sort_index()
    except Exception:
        return pd.Series(dtype=float, name=f"{chain}_TVL")


if __name__ == "__main__":
    worker()
