
import numpy as np
import pandas as pd

def detect_volatility_regime(returns, window=30):
	"""
	Detect volatility regime based on rolling standard deviation.
	Returns: 'low', 'normal', or 'crisis' for each period.
	"""
	vol = returns.rolling(window).std()
	thresholds = {
		'low': vol.quantile(0.33),
		'normal': vol.quantile(0.66)
	}
	regime = pd.Series(index=returns.index, dtype='object')
	regime[vol <= thresholds['low']] = 'low'
	regime[(vol > thresholds['low']) & (vol <= thresholds['normal'])] = 'normal'
	regime[vol > thresholds['normal']] = 'crisis'
	return regime

def calculate_betas(returns_df, benchmark='BTC'):
	"""
	Calculate beta of each asset to the benchmark (default: BTC).
	returns_df: DataFrame with columns as coins, rows as returns.
	Returns: dict of {coin: beta}
	"""
	betas = {}
	if benchmark not in returns_df:
		raise ValueError(f"Benchmark {benchmark} not in returns_df")
	benchmark_returns = returns_df[benchmark]
	var_bench = np.var(benchmark_returns)
	for coin in returns_df.columns:
		if coin == benchmark:
			betas[coin] = 1.0
		else:
			cov = np.cov(returns_df[coin], benchmark_returns)[0, 1]
			betas[coin] = cov / var_bench if var_bench != 0 else np.nan
	return betas

def liquidity_score(volume_series, window=90):
	"""
	Calculate rolling liquidity score as percentile rank of volume.
	Returns a Series of percentile scores (0=illiquid, 1=liquid).
	"""
	return volume_series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) else np.nan)

def liquidity_multiplier(score):
	"""
	Map liquidity score to a multiplier for stress scenarios.
	Lower liquidity = higher multiplier (crashes harder).
	"""
	if score < 0.2:
		return 1.5
	elif score < 0.4:
		return 1.2
	elif score < 0.7:
		return 1.0
	else:
		return 0.8
