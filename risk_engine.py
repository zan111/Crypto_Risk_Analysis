"""
Risk Engine — Statistical core for institutional-grade portfolio analytics.
==========================================================================
Implements:
  1. Distribution statistics (mean, std, skew, kurtosis)
  2. Rolling metrics (volatility, correlation, beta)
  3. Portfolio risk decomposition  (σ² = wᵀΣw)
  4. Correlation shock / stressed covariance model
  5. Rolling VaR (parametric)
  6. PCA / eigenvalue systemic-risk analysis
  7. Beta regime detection

All functions accept ``coin_data`` as returned by ``get_multi_coin_data()``
(dict of {symbol: DataFrame with 'price', 'log_returns', 'volume', …}).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

_ANNUAL = 365  # crypto markets trade 365 days/year


# =============================================================================
# 1 — DISTRIBUTION STATISTICS
# =============================================================================

def compute_return_stats(coin_data):
    """
    Compute distribution statistics for each asset's log returns.

    Returns
    -------
    pd.DataFrame
        Rows = coins, columns = mean_daily, std_daily, skew, kurtosis,
        annualised_vol, annualised_return, n_obs.
    """
    rows = {}
    for coin, df in coin_data.items():
        lr = df["log_returns"].dropna()
        rows[coin] = {
            "mean_daily": float(lr.mean()),
            "std_daily": float(lr.std()),
            "skew": float(lr.skew()),
            "kurtosis": float(lr.kurtosis()),          # excess kurtosis
            "annualised_vol": float(lr.std() * np.sqrt(365)),
            "annualised_return": float(lr.mean() * 365),
            "n_obs": int(len(lr)),
        }
    return pd.DataFrame(rows).T


# =============================================================================
# 2 — ROLLING METRICS
# =============================================================================

def rolling_volatility(coin_data, windows=(30, 60, 90)):
    """Rolling annualised volatility for each coin at multiple windows."""
    result = {}
    for coin, df in coin_data.items():
        lr = df["log_returns"].dropna()
        for w in windows:
            result[f"{coin}_vol_{w}d"] = lr.rolling(w).std() * np.sqrt(365)
    return pd.DataFrame(result).dropna()


def rolling_correlation(coin_data, window=60):
    """Rolling pairwise correlation (returns DataFrame, one col per pair)."""
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coin_data}).dropna()
    corrs = {}
    coins = list(rdf.columns)
    for i, c1 in enumerate(coins):
        for c2 in coins[i + 1:]:
            corrs[f"{c1}—{c2}"] = rdf[c1].rolling(window).corr(rdf[c2])
    return pd.DataFrame(corrs).dropna()


def rolling_beta(coin_data, benchmark="BTC", window=60):
    """Rolling beta of each altcoin vs benchmark."""
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coin_data}).dropna()
    if benchmark not in rdf.columns:
        raise ValueError(f"Benchmark '{benchmark}' not in coin_data")
    bench = rdf[benchmark]
    betas = {}
    for coin in rdf.columns:
        if coin == benchmark:
            continue
        cov_roll = rdf[coin].rolling(window).cov(bench)
        var_roll = bench.rolling(window).var()
        betas[f"{coin}_beta"] = cov_roll / var_roll
    return pd.DataFrame(betas).dropna()


# =============================================================================
# 3 — PORTFOLIO RISK DECOMPOSITION  (σ² = wᵀΣw)
# =============================================================================

def portfolio_risk_decomposition(coin_data, units):
    """
    Compute portfolio volatility, marginal and component risk contributions.

    Parameters
    ----------
    coin_data : dict
    units : dict  {coin: number_of_units_held}

    Returns
    -------
    dict with keys: coins, weights, portfolio_vol_annual, marginal_rc,
         component_rc, pct_risk_contribution.
    """
    coins = [c for c in coin_data if units.get(c, 0) > 0]
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coins}).dropna()

    # Annualised covariance matrix
    cov = rdf.cov().values * 365

    # Dollar values → weights
    prices = np.array([float(coin_data[c]["price"].iloc[-1]) for c in coins])
    u = np.array([units[c] for c in coins], dtype=float)
    dollar = prices * u
    total = dollar.sum()
    w = dollar / total

    # Portfolio variance & vol
    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(port_var))

    # Marginal risk contribution: Σw / σ_p
    sigma_w = cov @ w
    marginal_rc = sigma_w / port_vol

    # Component risk contribution: w_i × marginal_rc_i
    component_rc = w * marginal_rc

    # Percentage contribution (sums to ~1.0)
    pct_rc = component_rc / port_vol

    return {
        "coins": coins,
        "weights": dict(zip(coins, w)),
        "portfolio_vol_annual": port_vol,
        "marginal_rc": dict(zip(coins, marginal_rc)),
        "component_rc": dict(zip(coins, component_rc)),
        "pct_risk_contribution": dict(zip(coins, pct_rc)),
    }


# =============================================================================
# 4 — CORRELATION SHOCK / STRESSED COVARIANCE
# =============================================================================

def stressed_covariance(coin_data, vol_multiplier=2.0, corr_floor=0.85):
    """
    Build a crisis covariance matrix by inflating volatilities and pushing
    pairwise correlations toward ``corr_floor``.

    Returns
    -------
    (normal_cov_annual, stressed_cov_annual, coins)
    """
    coins = list(coin_data.keys())
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coins}).dropna()
    cov_daily = rdf.cov().values
    std_daily = np.sqrt(np.diag(cov_daily))

    # Normal (annualised)
    cov_annual = cov_daily * 365

    # Stressed
    stressed_std = std_daily * vol_multiplier
    n = len(coins)
    stressed_corr = np.full((n, n), corr_floor)
    np.fill_diagonal(stressed_corr, 1.0)
    D = np.diag(stressed_std)
    stressed_cov_annual = D @ stressed_corr @ D * 365

    return cov_annual, stressed_cov_annual, coins


def stressed_portfolio_vol(coin_data, units, vol_multiplier=2.0, corr_floor=0.85):
    """
    Compare portfolio vol under normal vs stressed covariance.

    Returns
    -------
    dict with normal_vol, stressed_vol, vol_increase_pct
    """
    coins = [c for c in coin_data if units.get(c, 0) > 0]
    cov_n, cov_s, _ = stressed_covariance(
        {c: coin_data[c] for c in coins}, vol_multiplier, corr_floor
    )
    prices = np.array([float(coin_data[c]["price"].iloc[-1]) for c in coins])
    u = np.array([units[c] for c in coins], dtype=float)
    dollar = prices * u
    w = dollar / dollar.sum()

    vol_n = float(np.sqrt(w @ cov_n @ w))
    vol_s = float(np.sqrt(w @ cov_s @ w))
    return {
        "normal_vol": vol_n,
        "stressed_vol": vol_s,
        "vol_increase_pct": (vol_s / vol_n - 1) if vol_n > 0 else 0.0,
    }


# =============================================================================
# 5 — ROLLING VaR (Parametric)
# =============================================================================

def rolling_var(coin_data, units, window=60, confidence=0.05):
    """
    Compute rolling 1-day parametric portfolio VaR (normal assumption).

    Returns a Series of dollar VaR estimates at the ``confidence`` level.
    """
    coins = [c for c in coin_data if units.get(c, 0) > 0]
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coins}).dropna()
    prices_df = pd.DataFrame(
        {c: coin_data[c]["price"] for c in coins}
    ).reindex(rdf.index)

    z = norm.ppf(confidence)  # e.g. -1.645
    u = np.array([units[c] for c in coins], dtype=float)

    var_vals = []
    var_idx = []
    for i in range(window, len(rdf)):
        chunk = rdf.iloc[i - window:i]
        mu_vec = chunk.mean().values
        cov_mat = chunk.cov().values

        p = prices_df.iloc[i].values
        dollar = p * u
        total = dollar.sum()
        w = dollar / total

        port_mu = float(w @ mu_vec)
        port_sig = float(np.sqrt(w @ cov_mat @ w))

        # 1-day VaR (dollar loss)
        var_vals.append(-(port_mu + z * port_sig) * total)
        var_idx.append(rdf.index[i])

    return pd.Series(var_vals, index=var_idx, name="VaR_1d_5pct")


# =============================================================================
# 6 — PCA / EIGENVALUE ANALYSIS
# =============================================================================

def pca_risk(coin_data):
    """
    Eigenvalue decomposition of the return correlation matrix.

    Returns
    -------
    dict with eigenvalues, explained_variance_ratio, cumulative_explained,
         first_pc_explains, first_pc_loadings, systemic_flag.
    """
    coins = list(coin_data.keys())
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coins}).dropna()
    corr = rdf.corr().values

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained = eigenvalues / eigenvalues.sum()

    return {
        "coins": coins,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained,
        "cumulative_explained": np.cumsum(explained),
        "first_pc_explains": float(explained[0]),
        "first_pc_loadings": dict(zip(coins, eigenvectors[:, 0])),
        "systemic_flag": float(explained[0]) > 0.70,
    }


# =============================================================================
# 7 — BETA REGIME DETECTION
# =============================================================================

def beta_regime_summary(coin_data, benchmark="BTC", window=60):
    """
    Compute average altcoin rolling beta and flag leverage regimes.

    Returns
    -------
    pd.DataFrame with columns avg_alt_beta, regime.
    """
    rb = rolling_beta(coin_data, benchmark, window)
    if rb.empty:
        return pd.DataFrame(columns=["avg_alt_beta", "regime"])
    avg = rb.mean(axis=1).rename("avg_alt_beta")
    regime = pd.Series("normal", index=avg.index)
    regime[avg > 1.5] = "leverage"
    regime[(avg > 1.0) & (avg <= 1.5)] = "amplified"
    regime[(avg > 0.5) & (avg <= 1.0)] = "normal"
    regime[avg <= 0.5] = "decoupled"
    return pd.DataFrame({"avg_alt_beta": avg, "regime": regime})


# =============================================================================
# 8 — PSD VALIDATION
# =============================================================================

def ensure_psd(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix onto the Positive Semi-Definite cone.

    Uses spectral decomposition (eigh) and clips negative eigenvalues to
    *epsilon*, then reconstructs.  This is the standard Higham-style fix
    used by risk systems when sample covariance matrices are near-singular.

    Parameters
    ----------
    matrix  : square symmetric numpy array
    epsilon : minimum eigenvalue floor (default 1e-8)

    Returns
    -------
    np.ndarray — PSD matrix of the same shape
    """
    mat = (matrix + matrix.T) / 2            # enforce exact symmetry
    eigvals, eigvecs = np.linalg.eigh(mat)   # eigh assumes symmetric
    eigvals = np.clip(eigvals, epsilon, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def covariance_matrix(coin_data: dict, annualise: bool = True) -> tuple:
    """
    Compute the sample covariance matrix of daily log-returns, validate PSD,
    and optionally annualise by multiplying by *_ANNUAL*.

    Returns
    -------
    (cov_df: pd.DataFrame, was_psd: bool)
        *was_psd* is True if the raw matrix was already PSD (no fix needed).
    """
    rets = pd.DataFrame(
        {s: df["returns"] for s, df in coin_data.items()}
    ).dropna()
    cov_raw = rets.cov().values

    # Check PSD before any fix
    eigvals_raw = np.linalg.eigvalsh(cov_raw)
    was_psd = bool(np.all(eigvals_raw >= -1e-8))

    cov_fixed = ensure_psd(cov_raw)
    if annualise:
        cov_fixed = cov_fixed * _ANNUAL

    coins = list(coin_data.keys())
    cov_df = pd.DataFrame(cov_fixed, index=coins, columns=coins)
    return cov_df, was_psd


# =============================================================================
# 9 — HISTORICAL (NON-PARAMETRIC) VAR / CVAR
# =============================================================================

def historical_var_cvar(coin_data: dict, units: dict,
                        confidence: float = 0.05) -> dict:
    """
    Non-parametric historical VaR and CVaR from the actual P&L distribution.

    Constructs the dollar P&L series as:
        P&L_t = Σ_i  return_{i,t} × price_{i,t} × units_i

    Then reads off empirical quantiles — no distributional assumptions.

    Returns
    -------
    dict with keys:
        var_1d, cvar_1d, var_10d, var_90d, cvar_10d, cvar_90d, pnl_series
    All VaR / CVaR values are *positive* numbers (losses expressed as positive).
    """
    coins = list(coin_data.keys())
    rets   = pd.DataFrame({s: coin_data[s]["returns"] for s in coins}).dropna()
    prices = pd.DataFrame({s: coin_data[s]["price"]   for s in coins}).reindex(rets.index).ffill()

    pnl = pd.Series(0.0, index=rets.index)
    for coin in coins:
        if coin in units:
            pnl += rets[coin] * prices[coin] * units[coin]

    var_1d  = float(-np.percentile(pnl, confidence * 100))
    tail    = pnl[pnl <= -var_1d]
    cvar_1d = float(-tail.mean()) if not tail.empty else var_1d

    # Square-root-of-time scaling (assumes iid daily returns)
    return {
        "var_1d":   var_1d,
        "cvar_1d":  cvar_1d,
        "var_10d":  var_1d  * np.sqrt(10),
        "var_90d":  var_1d  * np.sqrt(90),
        "cvar_10d": cvar_1d * np.sqrt(10),
        "cvar_90d": cvar_1d * np.sqrt(90),
        "pnl_series": pnl,
    }


# =============================================================================
# 10 — THREE-WAY VAR COMPARISON: NORMAL MC vs CRISIS MC vs HISTORICAL
# =============================================================================

def var_comparison(coin_data: dict, units: dict,
                   confidence: float = 0.05,
                   vol_multiplier: float = 2.0,
                   corr_floor: float = 0.85,
                   horizon_days: int = 90,
                   num_sim: int = 5_000) -> dict:
    """
    Side-by-side Monte Carlo VaR comparison:
        1. Normal regime  — sample covariance (PSD-fixed, annualised)
        2. Crisis regime  — stressed covariance (vol ×2, corr → 0.85)
        3. Historical     — empirical P&L quantiles (1-day + √t scaled)

    Returns
    -------
    dict with keys:
        normal_var, normal_cvar, crisis_var, crisis_cvar,
        hist_var_1d, hist_cvar_1d, hist_var_90d, hist_cvar_90d,
        normal_pnl, crisis_pnl, hist_pnl (arrays / Series),
        crisis_normal_ratio, horizon_days, port_val
    """
    coins = list(coin_data.keys())
    u = np.array([units.get(c, 1.0) for c in coins])
    last_prices = np.array([float(coin_data[c]["price"].iloc[-1]) for c in coins])
    port_val = float(np.dot(u, last_prices))
    w = (u * last_prices) / port_val          # weight vector

    # --- covariance matrices (daily) ---
    cov_df, _ = covariance_matrix(coin_data, annualise=True)
    cov_normal_daily = cov_df.values / _ANNUAL

    _, cov_crisis_annual, _ = stressed_covariance(coin_data, vol_multiplier, corr_floor)
    cov_crisis_daily = ensure_psd(cov_crisis_annual) / _ANNUAL

    rng = np.random.default_rng(42)

    def _mc(daily_cov: np.ndarray) -> tuple:
        L = np.linalg.cholesky(ensure_psd(daily_cov))
        z = rng.standard_normal((num_sim, horizon_days, len(coins)))
        shocks = z @ L.T                          # (num_sim, horizon, n)
        port_daily = shocks @ w                   # (num_sim, horizon)
        port_cum = port_daily.sum(axis=1)         # cumulative log-ret
        pnl = port_cum * port_val
        var  = float(-np.percentile(pnl, confidence * 100))
        tail = pnl[pnl <= -var]
        cvar = float(-tail.mean()) if len(tail) > 0 else var
        return var, cvar, pnl

    normal_var, normal_cvar, normal_pnl = _mc(cov_normal_daily)
    crisis_var, crisis_cvar, crisis_pnl = _mc(cov_crisis_daily)
    hist = historical_var_cvar(coin_data, units, confidence)

    return {
        "normal_var":   normal_var,
        "normal_cvar":  normal_cvar,
        "crisis_var":   crisis_var,
        "crisis_cvar":  crisis_cvar,
        "hist_var_1d":  hist["var_1d"],
        "hist_cvar_1d": hist["cvar_1d"],
        "hist_var_90d": hist["var_90d"],
        "hist_cvar_90d":hist["cvar_90d"],
        "normal_pnl":   normal_pnl,
        "crisis_pnl":   crisis_pnl,
        "hist_pnl":     hist["pnl_series"],
        "crisis_normal_ratio": crisis_var / normal_var if normal_var > 0 else float("nan"),
        "horizon_days": horizon_days,
        "port_val":     port_val,
    }


# =============================================================================
# 11 — VAR BACKTEST — KUPIEC PROPORTION OF FAILURES (POF) TEST
# =============================================================================

def backtest_var(coin_data: dict, units: dict,
                 confidence: float = 0.05,
                 window: int = 60) -> dict:
    """
    Rolling 1-day parametric VaR backtest with Kupiec (1995) POF test.

    At each date t, estimate a 60-day rolling parametric VaR using the
    portfolio P&L distribution, then check whether the *next* actual P&L
    exceeds (in loss) the VaR estimate.

    Kupiec LR statistic
    -------------------
        LR = -2 · ln[ p_exp^n (1-p_exp)^{T-n} / p_hat^n (1-p_hat)^{T-n} ]
        Under H₀ (model correct): LR ~ χ²(1)
        Critical value at 5 % level: 3.841
        Pass if LR < 3.841  (model not rejected)

    Returns
    -------
    dict with keys:
        breaches (bool Series), breach_rate, expected_rate,
        kupiec_lr, kupiec_pvalue, kupiec_pass,
        rolling_var (Series), actual_pnl (Series), T, n
    """
    coins = list(coin_data.keys())
    rets   = pd.DataFrame({s: coin_data[s]["returns"] for s in coins}).dropna()
    prices = (pd.DataFrame({s: coin_data[s]["price"] for s in coins})
              .reindex(rets.index).ffill())

    # Actual daily portfolio P&L
    pnl = pd.Series(0.0, index=rets.index)
    for coin in coins:
        if coin in units:
            pnl += rets[coin] * prices[coin] * units[coin]

    # Rolling parametric VaR (loss expressed as positive)
    rolling_var_s = pd.Series(np.nan, index=rets.index)
    for i in range(window, len(rets)):
        w_rets   = rets.iloc[i - window:i]
        w_prices = prices.iloc[i - 1]
        w_pnl = pd.Series(0.0, index=w_rets.index)
        for coin in coins:
            if coin in units:
                w_pnl += w_rets[coin] * w_prices[coin] * units[coin]
        rolling_var_s.iloc[i] = float(-norm.ppf(confidence) * w_pnl.std())

    valid      = rolling_var_s.dropna().index
    actual     = pnl.reindex(valid)
    var_est    = rolling_var_s.reindex(valid)
    breaches   = actual < -var_est      # True = actual loss > VaR estimate

    T    = int(len(breaches))
    n    = int(breaches.sum())
    p_hat = n / T if T > 0 else 0.0
    p_exp = confidence

    eps = 1e-10
    lr = -2.0 * (
        n       * np.log(p_exp  + eps) + (T - n) * np.log(1 - p_exp  + eps)
        - n     * np.log(p_hat  + eps) - (T - n) * np.log(1 - p_hat  + eps)
    )
    pval        = float(1 - chi2.cdf(lr, df=1))
    kupiec_pass = bool(lr < 3.841)

    return {
        "breaches":       breaches,
        "breach_rate":    p_hat,
        "expected_rate":  p_exp,
        "kupiec_lr":      float(lr),
        "kupiec_pvalue":  pval,
        "kupiec_pass":    kupiec_pass,
        "rolling_var":    var_est,
        "actual_pnl":     actual,
        "T":              T,
        "n":              n,
    }

# =============================================================================
# 12 — PERFORMANCE METRICS: SHARPE, SORTINO
# =============================================================================

def compute_sharpe_sortino(coin_data: dict, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Compute annualised Sharpe and Sortino ratios for each coin.

    Parameters
    ----------
    coin_data       : {symbol: DataFrame}
    risk_free_rate  : annualised risk-free rate (default 0.0 — crypto basis)

    Returns
    -------
    pd.DataFrame  — index = coins, columns = sharpe, sortino,
                    annualised_return, annualised_vol
    """
    rows = {}
    daily_rfr = risk_free_rate / _ANNUAL
    for coin, df in coin_data.items():
        lr = df["log_returns"].dropna()
        mu_ann  = float(lr.mean()) * _ANNUAL
        vol_ann = float(lr.std()) * np.sqrt(_ANNUAL)
        excess  = float(lr.mean() - daily_rfr) * _ANNUAL

        # Downside deviation — semi-deviation below 0
        downside = lr[lr < 0]
        dd_ann = (float(downside.std()) * np.sqrt(_ANNUAL)
                  if len(downside) > 1 else vol_ann)

        rows[coin] = {
            "annualised_return": mu_ann,
            "annualised_vol":    vol_ann,
            "sharpe":  excess / vol_ann  if vol_ann > 0 else float("nan"),
            "sortino": excess / dd_ann   if dd_ann  > 0 else float("nan"),
        }
    return pd.DataFrame(rows).T


# =============================================================================
# 13 — MAX DRAWDOWN
# =============================================================================

def compute_max_drawdown(coin_data: dict) -> pd.DataFrame:
    """
    Compute maximum drawdown and its duration for each coin.

    Uses the full price history available in coin_data.

    Returns
    -------
    pd.DataFrame  — index = coins, columns = max_drawdown (fraction, negative),
                    max_dd_duration_days
    """
    rows = {}
    for coin, df in coin_data.items():
        prices = df["price"].dropna()
        rolling_max = prices.cummax()
        drawdown    = (prices - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        # Longest consecutive streak where price was below its prior peak
        in_dd   = (drawdown < -1e-6).values
        max_dur = cur_dur = 0
        for flag in in_dd:
            if flag:
                cur_dur += 1
                max_dur  = max(max_dur, cur_dur)
            else:
                cur_dur = 0

        rows[coin] = {
            "max_drawdown":          max_dd,
            "max_dd_duration_days":  max_dur,
        }
    return pd.DataFrame(rows).T