"""
Visualization functions for cryptocurrency Monte Carlo simulations.
Uses matplotlib for standalone chart generation (legacy module).
The Dash app uses Plotly instead — see app.py for interactive charts.
"""
import matplotlib.pyplot as plt

def plot_garch_simulation(sim_df, percentiles, last_price, coin):
    plt.figure(figsize=(12,5))
    sample_cols = sim_df.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="black", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="black", alpha=0.2)
    plt.axhline(last_price, color="red", linestyle="--", label="Current Price")
    plt.title(f"GARCH Simulation {coin} Price Forecast")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

def plot_copula_simulation(sim_df, percentiles, last_portfolio_price, coins):
    plt.figure(figsize=(12,5))
    sample_cols = sim_df.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="brown", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="brown", alpha=0.2)
    plt.axhline(last_portfolio_price, color="red", linestyle="--", label="Current Portfolio")
    plt.title(f"Copula Simulation Portfolio ({', '.join(coins)})")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.show()
def plot_bootstrapped_simulation(sim_df, percentiles, last_price, coin):
    plt.figure(figsize=(12,5))
    sample_cols = sim_df.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="purple", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="purple", alpha=0.2)
    plt.axhline(last_price, color="red", linestyle="--", label="Current Price")
    plt.title(f"Bootstrapped Path Simulation {coin} Price Forecast")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

def plot_gbm_simulation(sim_df, percentiles, last_price, coin):
    plt.figure(figsize=(12,5))
    sample_cols = sim_df.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="orange", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="orange", alpha=0.2)
    plt.axhline(last_price, color="red", linestyle="--", label="Current Price")
    plt.title(f"GBM Simulation {coin} Price Forecast")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()
def plot_historical_simulation(sim_df, percentiles, last_price, coin):
    plt.figure(figsize=(12,5))
    sample_cols = sim_df.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="green", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="green", alpha=0.2)
    plt.axhline(last_price, color="red", linestyle="--", label="Current Price")
    plt.title(f"Historical Simulation {coin} Price Forecast")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

def plot_regime_mc(df_regime_mc):
    percentiles = df_regime_mc.quantile([0.05, 0.5, 0.95], axis=1).T
    percentiles.columns = ["P5", "Median", "P95"]
    x = percentiles.index
    sample_cols = df_regime_mc.sample(n=min(15, df_regime_mc.shape[1]), axis=1)
    plt.figure(figsize=(12, 5))
    plt.plot(sample_cols, color="gray", alpha=0.2)
    plt.plot(x, percentiles["Median"], color="purple", linewidth=2, label="Median")
    plt.fill_between(x, percentiles["P5"], percentiles["P95"],
                     color="purple", alpha=0.15, label="P5–P95")
    plt.title("Regime-Switching Monte Carlo — Portfolio Simulation")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_multi_coin_mc(df_joined, percentiles, last_price, coin):
    plt.figure(figsize=(12,5))
    sample_cols = df_joined.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="blue", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="blue", alpha=0.2)
    plt.axhline(last_price, color="red", linestyle="--", label="Current Price")
    plt.title(f"Monte Carlo {coin} Price Forecast")
    plt.xlabel("Day")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

def plot_multi_coin_hist(final_prices, last_price, target_price, median_terminal, p5_terminal, p95_terminal, prob_below_today, prob_above_target, coin):
    plt.figure(figsize=(10,5))
    plt.hist(final_prices, bins=60)
    plt.axvline(last_price, linestyle="--", label="Today Price")
    plt.axvline(target_price, linestyle="--", label="Target Price")
    plt.axvline(median_terminal, linestyle="-", label="Median")
    plt.axvline(p5_terminal, linestyle=":", label="P5")
    plt.axvline(p95_terminal, linestyle=":", label="P95")
    plt.text(last_price, plt.ylim()[1]*0.9, f"P(<Today)={prob_below_today:.1%}", rotation=90)
    plt.text(target_price, plt.ylim()[1]*0.9, f"P(>Target)={prob_above_target:.1%}", rotation=90)
    plt.title(f"Terminal Distribution for {coin} — Target Probability")
    plt.xlabel("Simulated Price at Horizon")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_portfolio_mc_fan(df_portfolio, percentiles):
    x = percentiles.index
    sample_cols = df_portfolio.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.figure(figsize=(12,5))
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="blue", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="blue", alpha=0.2)
    plt.title("Portfolio Monte Carlo Price Forecast")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Price (USD)")
    plt.legend()
    plt.show()

def plot_portfolio_mc_hist(final_prices, last_portfolio_price, target_price, median_terminal, p5_terminal, p95_terminal, prob_below_today, prob_above_target):
    plt.figure(figsize=(12,6))
    plt.hist(final_prices, bins=60)
    plt.axvline(last_portfolio_price, linestyle="--", label="Today Price")
    plt.axvline(target_price, linestyle="--", label="Target Price")
    plt.axvline(median_terminal, linestyle="-", label="Median")
    plt.axvline(p5_terminal, linestyle=":", label="P5")
    plt.axvline(p95_terminal, linestyle=":", label="P95")
    plt.text(last_portfolio_price, plt.ylim()[1]*0.9, f"P(<Today)={prob_below_today:.1%}", rotation=90)
    plt.text(target_price, plt.ylim()[1]*0.9, f"P(>Target)={prob_above_target:.1%}", rotation=90)
    plt.title("Portfolio Terminal Distribution — Target Probability")
    plt.xlabel("Portfolio Price at Horizon")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_correlated_portfolio_mc_fan(df_portfolio, percentiles):
    x = percentiles.index
    sample_cols = df_portfolio.drop(columns=["mean","std"]).sample(n=10, axis=1)
    plt.figure(figsize=(12,5))
    plt.plot(sample_cols, color="gray", alpha=0.3)
    plt.plot(percentiles["Median"], color="blue", label="Median")
    plt.fill_between(percentiles.index, percentiles["P5"], percentiles["P95"], color="blue", alpha=0.2)
    plt.title("Correlated Portfolio Monte Carlo Price Forecast")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Price (USD)")
    plt.legend()
    plt.show()

def plot_correlated_portfolio_mc_hist(final_prices, last_portfolio_price, target_price, median_terminal, p5_terminal, p95_terminal, prob_below_today, prob_above_target):
    plt.figure(figsize=(12,6))
    plt.hist(final_prices, bins=60)
    plt.axvline(last_portfolio_price, linestyle="--", label="Today Price")
    plt.axvline(target_price, linestyle="--", label="Target Price")
    plt.axvline(median_terminal, linestyle="-", label="Median")
    plt.axvline(p5_terminal, linestyle=":", label="P5")
    plt.axvline(p95_terminal, linestyle=":", label="P95")
    plt.text(last_portfolio_price, plt.ylim()[1]*0.9, f"P(<Today)={prob_below_today:.1%}", rotation=90)
    plt.text(target_price, plt.ylim()[1]*0.9, f"P(>Target)={prob_above_target:.1%}", rotation=90)
    plt.title("Portfolio Terminal Distribution — Target Probability")
    plt.xlabel("Portfolio Price at Horizon")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_portfolio_mc_regimes(df_normal, df_crisis, current_value):
    p_normal = df_normal.quantile([0.05,0.5,0.95], axis=1).T
    p_crisis = df_crisis.quantile([0.05,0.5,0.95], axis=1).T
    p_normal.columns = ["P5","Median","P95"]
    p_crisis.columns = ["P5","Median","P95"]
    x = p_normal.index
    plt.figure(figsize=(12,6))
    plt.plot(x, p_normal["Median"], label="Normal Regime")
    plt.plot(x, p_crisis["Median"], linestyle="--", label="Crisis Regime")
    plt.fill_between(x, p_normal["P5"], p_normal["P95"], alpha=0.2)
    plt.fill_between(x, p_crisis["P5"], p_crisis["P95"], alpha=0.15)
    plt.axhline(current_value, linestyle=":", label="Current Portfolio")
    plt.title("Portfolio Monte Carlo — Volatility Regimes")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.show()

def plot_stress_tests(df_stress):
    plt.figure(figsize=(12,6))
    for scenario in df_stress.columns:
        plt.plot(df_stress.index, df_stress[scenario], label=scenario)
    plt.title("Portfolio Stress Test Scenarios")
    plt.xlabel("Day")
    plt.ylabel("Portfolio Price")
    plt.legend()
    plt.show()
