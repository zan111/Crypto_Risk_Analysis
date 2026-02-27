"""
DeFi Risk Analysis Tool — Dash Web Application
================================================
Interactive dashboard for cryptocurrency portfolio risk analysis,
Monte Carlo simulations (GBM, GARCH, Historical, Bootstrapped),
Copula correlation modelling, stress testing, and regime-switching MC.

Usage
-----
    pip install dash plotly
    conda activate geni2024
    python app.py
    → Open http://127.0.0.1:8050
"""

# =============================================================================
# IMPORTS
# =============================================================================

import dash
from dash import dcc, html, dash_table, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import traceback

# Local modules — pure functions only (no input / no plt.show)
from fetch_data import (
    get_multi_coin_data,
    regime_switching_monte_carlo,
    stress_test_portfolio,
    COIN_SETS,
    get_defi_tvl,
    get_chain_tvl_history,
)
from risk_analysis import calculate_betas, liquidity_multiplier

# Risk engine — institutional analytics
from risk_engine import (
    compute_return_stats,
    rolling_volatility,
    rolling_correlation,
    rolling_beta,
    portfolio_risk_decomposition,
    stressed_covariance,
    stressed_portfolio_vol,
    rolling_var,
    pca_risk,
    beta_regime_summary,
    covariance_matrix,
    historical_var_cvar,
    var_comparison,
    backtest_var,
    compute_sharpe_sortino,
    compute_max_drawdown,
)

# Social sentiment
from social_sentiment import (
    get_sentiment, sentiment_vs_returns,
    sentiment_index_daily, sentiment_sensitivity,
)

# API keys and runtime constants (local only — never commit config.py)
from config import (
    NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_SECRET, X_BEARER_TOKEN,
    N_SIMS_QUICK, N_SIMS_FULL, ALL_COIN_SYMBOLS,
)

# Systemic risk index
from systemic_risk import (
    compute_systemic_index,
    component_history,
    regime_conditioned_stats,
    leading_indicator_series,
    lead_lag_correlations,
    regime_strategy_backtest,
    ZONES, REGIMES, COMPONENT_META, STRESS_EVENTS,
)

# Advanced models
import scipy.stats as stats

try:
    from arch import arch_model
    GARCH_OK = True
except ImportError:
    GARCH_OK = False


# =============================================================================
# GLOBAL CACHE  (single-user local app — safe for localhost)
# =============================================================================

_cache: dict = {}


# =============================================================================
# DASH-COMPATIBLE SIMULATION FUNCTIONS
# (Re-implemented without input() or matplotlib calls so they work in Dash)
# =============================================================================

def run_mc_portfolio(coin_data, units, betas=None, regime="normal",
                     macro_scenario="none", horizon_days=90, num_sim=1000):
    """Monte Carlo portfolio simulation with beta & liquidity adjustments."""
    coins = list(coin_data.keys())
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coins}).dropna()
    ret_matrix = rdf.values
    mean_ret = ret_matrix.mean(axis=0).copy()
    vols = ret_matrix.std(axis=0)

    macro_drift = {
        "rate_hikes": -0.02, "qe": 0.03, "dxy_up": -0.01,
        "liquidity_up": 0.02, "cpi_surprise": 0.01,
    }
    if macro_scenario in macro_drift:
        mean_ret += macro_drift[macro_scenario]

    drift = mean_ret - 0.5 * vols ** 2
    cov = np.cov(ret_matrix.T)
    vol_mult = {"low": 0.7, "normal": 1.0, "high": 1.8, "crisis": 2.3}
    cov = cov * vol_mult.get(regime, 1.0)

    last_prices = {c: float(coin_data[c]["price"].iloc[-1]) for c in coins}

    if betas is None:
        betas = calculate_betas(rdf, benchmark="BTC")

    liq = {}
    for c in coins:
        if "volume_pct_rank" in coin_data[c].columns:
            liq[c] = liquidity_multiplier(float(coin_data[c]["volume_pct_rank"].iloc[-1]))
        else:
            liq[c] = 1.0

    btc_idx = coins.index("BTC") if "BTC" in coins else None
    paths = []
    for _ in range(num_sim):
        cr = np.random.multivariate_normal(drift, cov, horizon_days)
        if btc_idx is not None:
            for i in range(horizon_days):
                br = cr[i, btc_idx]
                for j, c in enumerate(coins):
                    if c != "BTC":
                        cr[i, j] = (
                            br * betas.get(c, 1.0) * liq.get(c, 1.0)
                            + 0.3 * cr[i, j]
                        )
        df_r = pd.DataFrame(cr, columns=coins)
        pp = pd.DataFrame({c: last_prices[c] * np.exp(df_r[c].cumsum()) for c in coins})
        paths.append(sum(pp[c] * units.get(c, 0) for c in coins))

    return pd.DataFrame(paths).T


def run_multi_coin_mc(coin_data, horizon_days=90, num_sim=1000):
    """Per-coin Monte Carlo (returns dict of results per coin)."""
    results = {}
    for coin, df in coin_data.items():
        mu = float(df["log_returns"].mean())
        sig = float(df["log_returns"].std())
        lp = float(df["price"].iloc[-1])
        sims = []
        for _ in range(num_sim):
            r = np.random.normal(mu, sig, horizon_days)
            sims.append(lp * np.exp(np.cumsum(r)))
        sim_df = pd.DataFrame(sims).T
        sim_df["mean"] = sim_df.mean(axis=1)
        sim_df["std"] = sim_df.std(axis=1)
        pct = sim_df.drop(columns=["mean", "std"]).quantile([0.05, 0.5, 0.95], axis=1).T
        pct.columns = ["P5", "Median", "P95"]
        fp = sim_df.iloc[-1].drop(["mean", "std"]).values.astype(float)
        results[coin] = dict(
            sim_df=sim_df, percentiles=pct, final_prices=fp,
            last_price=lp,
            median=float(np.median(fp)),
            p5=float(np.percentile(fp, 5)),
            p95=float(np.percentile(fp, 95)),
        )
    return results


# --------------- single-coin model runners ---------------

def _pack(sims, lp):
    """Turn list-of-arrays into sim_df + percentiles."""
    sim_df = pd.DataFrame(sims).T
    sim_df["mean"] = sim_df.mean(axis=1)
    sim_df["std"] = sim_df.std(axis=1)
    pct = sim_df.drop(columns=["mean", "std"]).quantile([0.05, 0.5, 0.95], axis=1).T
    pct.columns = ["P5", "Median", "P95"]
    return sim_df, pct, lp


def run_gbm(coin_data, coin, horizon_days=90, num_sim=1000):
    lr = coin_data[coin]["log_returns"].dropna()
    mu, sig = float(lr.mean()), float(lr.std())
    lp = float(coin_data[coin]["price"].iloc[-1])
    sims = []
    for _ in range(num_sim):
        W = np.random.normal(0, 1, horizon_days).cumsum()
        t = np.arange(1, horizon_days + 1)
        sims.append(lp * np.exp((mu - 0.5 * sig ** 2) * t + sig * W))
    return _pack(sims, lp)


def run_garch(coin_data, coin, horizon_days=90, num_sim=500):
    if not GARCH_OK:
        raise ImportError("arch package is not installed — cannot run GARCH.")
    lr = coin_data[coin]["log_returns"].dropna()
    lp = float(coin_data[coin]["price"].iloc[-1])
    am = arch_model(lr, vol="Garch", p=1, q=1)
    res = am.fit(disp="off")
    sims = []
    for _ in range(num_sim):
        sv = res.simulate(res.params, horizon_days)
        sims.append(lp * np.exp(np.cumsum(sv["data"].values.flatten())))
    return _pack(sims, lp)


def run_historical(coin_data, coin, horizon_days=90, num_sim=1000):
    hr = coin_data[coin]["log_returns"].dropna()
    lp = float(coin_data[coin]["price"].iloc[-1])
    sims = []
    for _ in range(num_sim):
        sims.append(lp * np.exp(np.cumsum(hr.sample(n=horizon_days, replace=True).values)))
    return _pack(sims, lp)


def run_bootstrapped(coin_data, coin, horizon_days=90, num_sim=1000, block_size=10):
    rets = coin_data[coin]["log_returns"].dropna().values
    lp = float(coin_data[coin]["price"].iloc[-1])
    nb = int(np.ceil(horizon_days / block_size))
    sims = []
    for _ in range(num_sim):
        pr = []
        for _ in range(nb):
            s = np.random.randint(0, max(len(rets) - block_size, 1))
            pr.extend(rets[s:s + block_size])
        sims.append(lp * np.exp(np.cumsum(np.array(pr[:horizon_days]))))
    return _pack(sims, lp)


def run_copula(coin_data, coins, horizon_days=90, num_sim=1000):
    rdf = pd.DataFrame({c: coin_data[c]["log_returns"] for c in coins}).dropna()
    rm = rdf.values
    lp = {c: float(coin_data[c]["price"].iloc[-1]) for c in coins}
    marg = [stats.norm.fit(rm[:, i]) for i in range(len(coins))]
    corr = np.corrcoef(rm.T)
    # ensure positive-definite
    eigvals = np.linalg.eigvalsh(corr)
    if np.any(eigvals <= 0):
        corr += np.eye(len(coins)) * 1e-6
    sims = []
    for _ in range(num_sim):
        mv = np.random.multivariate_normal(np.zeros(len(coins)), corr, horizon_days)
        u = stats.norm.cdf(mv)
        sr = np.column_stack([
            stats.norm.ppf(u[:, j], loc=marg[j][0], scale=marg[j][1])
            for j in range(len(coins))
        ])
        pp = {c: lp[c] * np.exp(np.cumsum(sr[:, i])) for i, c in enumerate(coins)}
        sims.append(sum(pp[c] for c in coins))
    return _pack(sims, sum(lp.values()))


# =============================================================================
# PLOTLY FIGURE BUILDERS
# =============================================================================

_RGB = {
    "blue": "33,150,243", "red": "244,67,54", "green": "76,175,80",
    "orange": "255,152,0", "purple": "156,39,176", "brown": "121,85,72",
    "black": "66,66,66", "teal": "0,150,136",
}


def fig_fan(sim_df, pct, last_price, title, color="blue", n_paths=25):
    """Fan chart — sample paths + median + P5-P95 band."""
    rgb = _RGB.get(color, _RGB["blue"])
    fig = go.Figure()
    ncols = sim_df.shape[1] - 2
    sample = sim_df.drop(columns=["mean", "std"]).sample(n=min(n_paths, ncols), axis=1)
    for col in sample.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(sample))), y=sample[col], mode="lines",
            line=dict(color="rgba(158,158,158,0.25)", width=0.7),
            showlegend=False, hoverinfo="skip",
        ))
    x = list(range(len(pct)))
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(pct["P95"]) + list(pct["P5"])[::-1],
        fill="toself", fillcolor=f"rgba({rgb},0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="P5 – P95", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pct["Median"], mode="lines",
        line=dict(color=f"rgb({rgb})", width=2.5), name="Median",
        hovertemplate="Day %{x}  ·  $%{y:,.0f}<extra>Median</extra>",
    ))
    # Add hover info to P5 and P95 lines so users can inspect the bands
    fig.add_trace(go.Scatter(
        x=x, y=pct["P5"], mode="lines",
        line=dict(color=f"rgba({rgb},0.4)", width=1, dash="dot"),
        showlegend=False,
        hovertemplate="Day %{x}  ·  P5: $%{y:,.0f}<extra>5th percentile</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pct["P95"], mode="lines",
        line=dict(color=f"rgba({rgb},0.4)", width=1, dash="dot"),
        showlegend=False,
        hovertemplate="Day %{x}  ·  P95: $%{y:,.0f}<extra>95th percentile</extra>",
    ))
    fig.add_hline(
        y=last_price, line_dash="dash", line_color="red",
        annotation_text=f"Current ${last_price:,.0f}",
    )
    fig.update_layout(
        title=title, xaxis_title="Day", yaxis_title="Price (USD)",
        template="plotly_white", height=480,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def fig_hist(final_prices, last_price, target_price, title):
    """Terminal-distribution histogram with key levels."""
    med = float(np.median(final_prices))
    p5 = float(np.percentile(final_prices, 5))
    p95 = float(np.percentile(final_prices, 95))
    pb = float(np.mean(final_prices < last_price))
    pa = float(np.mean(final_prices > target_price)) if target_price else 0.0

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_prices, nbinsx=60,
        marker_color="steelblue", opacity=0.75,
    ))
    for val, lbl, dash_style, clr in [
        (last_price, "Current", "dash", "red"),
        (med, "Median", "solid", "black"),
        (p5, "P5", "dashdot", "gray"),
        (p95, "P95", "dashdot", "gray"),
    ]:
        fig.add_vline(x=val, line_dash=dash_style, line_color=clr,
                      annotation_text=f"{lbl} ${val:,.0f}")
    if target_price and target_price > 0:
        fig.add_vline(x=target_price, line_dash="dot", line_color="green",
                      annotation_text=f"Target ${target_price:,.0f}")

    sub = f"P(&lt;Today) = {pb:.1%}"
    if target_price and target_price > 0:
        sub += f" &nbsp;|&nbsp; P(&gt;Target) = {pa:.1%}"
    fig.update_layout(
        title=f"{title}<br><sup>{sub}</sup>",
        xaxis_title="Simulated Price at Horizon", yaxis_title="Frequency",
        template="plotly_white", height=420,
        margin=dict(l=60, r=30, t=80, b=40),
    )
    return fig


def fig_stress(df_stress):
    palette = ["#2196F3", "#f44336", "#4CAF50", "#FF9800", "#9C27B0", "#607D8B"]
    fig = go.Figure()
    for i, col in enumerate(df_stress.columns):
        fig.add_trace(go.Scatter(
            x=list(df_stress.index), y=df_stress[col], mode="lines",
            name=col, line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig.update_layout(
        title="Portfolio Stress Test Scenarios",
        xaxis_title="Day", yaxis_title="Portfolio Value (USD)",
        template="plotly_white", height=500,
    )
    return fig


def fig_regimes(df_n, df_c, cv):
    """Normal-vs-Crisis regime comparison."""
    pn = df_n.quantile([0.05, 0.5, 0.95], axis=1).T
    pc = df_c.quantile([0.05, 0.5, 0.95], axis=1).T
    pn.columns = pc.columns = ["P5", "Median", "P95"]
    x = list(pn.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=list(pn["P95"]) + list(pn["P5"])[::-1],
        fill="toself", fillcolor="rgba(33,150,243,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="Normal 90 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=pn["Median"], mode="lines",
        line=dict(color="#2196F3", width=2), name="Normal Median"))
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=list(pc["P95"]) + list(pc["P5"])[::-1],
        fill="toself", fillcolor="rgba(244,67,54,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="Crisis 90 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=pc["Median"], mode="lines",
        line=dict(color="#f44336", width=2, dash="dash"), name="Crisis Median"))
    fig.add_hline(y=cv, line_dash="dot", line_color="black",
                  annotation_text=f"Current ${cv:,.0f}")
    fig.update_layout(
        title="Portfolio Monte Carlo — Normal vs Crisis Regime",
        xaxis_title="Day", yaxis_title="Portfolio Value (USD)",
        template="plotly_white", height=500,
    )
    return fig


def fig_regime_switch(df_rs):
    """Markov regime-switching MC fan chart."""
    pct = df_rs.quantile([0.05, 0.5, 0.95], axis=1).T
    pct.columns = ["P5", "Median", "P95"]
    x = list(pct.index)
    fig = go.Figure()
    ns = min(30, df_rs.shape[1])
    for col in df_rs.sample(n=ns, axis=1).columns:
        fig.add_trace(go.Scatter(
            x=x, y=df_rs[col], mode="lines",
            line=dict(color="rgba(158,158,158,0.2)", width=0.6),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=list(pct["P95"]) + list(pct["P5"])[::-1],
        fill="toself", fillcolor="rgba(156,39,176,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="P5 – P95", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=pct["Median"], mode="lines",
        line=dict(color="#9C27B0", width=2.5), name="Median"))
    fig.update_layout(
        title="Markov Regime-Switching MC — Portfolio",
        xaxis_title="Day", yaxis_title="Portfolio Value (USD)",
        template="plotly_white", height=500,
    )
    return fig


# =============================================================================
# DASH APPLICATION
# =============================================================================

app = dash.Dash(
    __name__,
    title="DeFi Risk Analysis",
    suppress_callback_exceptions=True,
)

# ---- styles ----------------------------------------------------------------

SIDEBAR = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "320px", "padding": "20px 18px", "overflowY": "auto",
    "backgroundColor": "#f8f9fa", "borderRight": "1px solid #dee2e6",
}
CONTENT = {"marginLeft": "340px", "padding": "24px 28px"}
BTN_PRIMARY = {
    "width": "100%", "padding": "10px", "border": "none", "borderRadius": "5px",
    "backgroundColor": "#1976D2", "color": "white", "cursor": "pointer",
    "fontWeight": "bold", "fontSize": "14px", "marginTop": "6px",
}
BTN_SUCCESS = {**BTN_PRIMARY, "backgroundColor": "#388E3C", "fontSize": "15px",
               "padding": "12px"}
BTN_ORANGE = {**BTN_PRIMARY, "backgroundColor": "#F57C00"}
LABEL = {"fontWeight": "bold", "marginTop": "10px", "display": "block"}
INPUT_FULL = {"width": "100%", "marginBottom": "6px"}
TAB_STYLE = {
    "padding": "10px 16px", "fontWeight": "600", "fontSize": "13px",
    "borderBottom": "2px solid transparent",
}
TAB_SEL = {**TAB_STYLE, "borderTop": "3px solid #1976D2", "color": "#1976D2",
           "backgroundColor": "white"}

# ---- sidebar ----------------------------------------------------------------

sidebar = html.Div([
    html.H3("DeFi Risk Tool", style={"color": "#1a237e", "marginBottom": "4px"}),
    html.P("v1.0 · Crypto Systemic Risk Engine", style={"color": "#666", "fontSize": "12px"}),
    html.Hr(),

    # 1 — Fetch data
    html.Label("1 · Load Market Data", style=LABEL),
    html.Label("Coin set", style={"fontSize": "12px"}),
    dcc.Dropdown(
        id="in-coin-set", value="top3", clearable=False,
        options=[
            {"label": "Top 3  (BTC · ETH · SOL)",              "value": "top3"},
            {"label": "Top 5  (+ BNB · XRP)",                 "value": "top5"},
            {"label": "Top 10 (+ ADA · DOGE · AVAX · DOT · LINK)", "value": "top10"},
        ],
        style={"marginBottom": "6px"},
    ),
    html.Button("Fetch Data", id="btn-fetch", n_clicks=0, style=BTN_PRIMARY),
    dcc.Loading(html.Div(id="data-status",
                         style={"fontSize": "12px", "marginTop": "4px", "minHeight": "18px"}),
                type="circle"),
    html.Hr(),

    # 2 — Holdings
    html.Label("2 · Portfolio Holdings", style=LABEL),
    html.Div([
        # — always visible (Top 3 core)
        html.Label("BTC units", style={"fontSize": "12px"}),
        dcc.Input(id="in-btc", type="number", value=0.5, step=0.01, style=INPUT_FULL),
        html.Label("ETH units", style={"fontSize": "12px"}),
        dcc.Input(id="in-eth", type="number", value=2.0, step=0.01, style=INPUT_FULL),
        html.Label("SOL units", style={"fontSize": "12px"}),
        dcc.Input(id="in-sol", type="number", value=10.0, step=0.1, style=INPUT_FULL),
        # — shown for Top 5 and Top 10
        html.Div(id="holdings-top5", children=[
            html.Label("BNB units", style={"fontSize": "12px"}),
            dcc.Input(id="in-bnb", type="number", value=0, step=0.1, style=INPUT_FULL),
            html.Label("XRP units", style={"fontSize": "12px"}),
            dcc.Input(id="in-xrp", type="number", value=0, step=1.0, style=INPUT_FULL),
        ], style={"display": "none"}),
        # — shown for Top 10 only
        html.Div(id="holdings-top10", children=[
            html.Label("ADA units",  style={"fontSize": "12px"}),
            dcc.Input(id="in-ada",  type="number", value=0, step=1.0,  style=INPUT_FULL),
            html.Label("DOGE units", style={"fontSize": "12px"}),
            dcc.Input(id="in-doge", type="number", value=0, step=10.0, style=INPUT_FULL),
            html.Label("AVAX units", style={"fontSize": "12px"}),
            dcc.Input(id="in-avax", type="number", value=0, step=0.1,  style=INPUT_FULL),
            html.Label("DOT units",  style={"fontSize": "12px"}),
            dcc.Input(id="in-dot",  type="number", value=0, step=0.1,  style=INPUT_FULL),
            html.Label("LINK units", style={"fontSize": "12px"}),
            dcc.Input(id="in-link", type="number", value=0, step=0.1,  style=INPUT_FULL),
        ], style={"display": "none"}),
    ]),
    html.Hr(),

    # 3 — Parameters
    html.Label("3 · Simulation Parameters", style=LABEL),
    html.Label("Horizon (days)", style={"fontSize": "12px"}),
    dcc.Input(id="in-horizon", type="number", value=90, min=7, max=365, style=INPUT_FULL),
    html.Label("Simulations", style={"fontSize": "12px"}),
    dcc.Dropdown(id="in-nsim",
                 options=[{"label": str(n), "value": n} for n in [200, 500, 1000, 2000]],
                 value=500, clearable=False, style={"marginBottom": "4px"}),
    dcc.Checklist(
        id="in-quick-mode",
        options=[{"label": " Quick Mode (300 sims — faster with 10 coins)", "value": "quick"}],
        value=[],
        style={"fontSize": "12px", "marginBottom": "6px"},
    ),
    html.Label("Macro scenario", style={"fontSize": "12px"}),
    dcc.Dropdown(id="in-macro", value="none", clearable=False,
                 options=[
                     {"label": "None", "value": "none"},
                     {"label": "Rate Hikes", "value": "rate_hikes"},
                     {"label": "Quantitative Easing", "value": "qe"},
                     {"label": "USD Strength (DXY ↑)", "value": "dxy_up"},
                     {"label": "Liquidity ↑", "value": "liquidity_up"},
                     {"label": "CPI Surprise", "value": "cpi_surprise"},
                 ], style={"marginBottom": "10px"}),
    html.Button("▶  Run Full Analysis", id="btn-run", n_clicks=0, style=BTN_SUCCESS),
    dcc.Loading(html.Div(id="run-status",
                         style={"fontSize": "12px", "marginTop": "4px", "minHeight": "18px"}),
                type="circle"),
    html.Hr(),

    # 4 — Single-coin simulation
    html.Label("4 · Single-Coin Simulation", style=LABEL),
    html.Label("Coin", style={"fontSize": "12px"}),
    dcc.Dropdown(id="in-coin", value="ETH", clearable=False,
                 options=[{"label": c, "value": c} for c in ["BTC", "ETH", "SOL"]],
                 placeholder="Select coin (fetch data first)",
                 style={"marginBottom": "6px"}),
    html.Label("Model", style={"fontSize": "12px"}),
    dcc.Dropdown(id="in-model", value="gbm", clearable=False,
                 options=[
                     {"label": "Geometric Brownian Motion", "value": "gbm"},
                     {"label": "GARCH(1,1)", "value": "garch"},
                     {"label": "Historical Simulation", "value": "historical"},
                     {"label": "Bootstrapped Paths", "value": "bootstrapped"},
                 ], style={"marginBottom": "6px"}),
    html.Label("Target price ($)", style={"fontSize": "12px"}),
    dcc.Input(id="in-target", type="number", value=0, min=0, step=100, style=INPUT_FULL),
    html.Button("▶  Run Single-Coin Sim", id="btn-single", n_clicks=0, style=BTN_ORANGE),
    dcc.Loading(html.Div(id="single-status",
                         style={"fontSize": "12px", "marginTop": "4px", "minHeight": "18px"}),
                type="circle"),
    html.Hr(),

    # 5 — Social Sentiment
    html.Label("5 · Social Sentiment", style=LABEL),
    html.P("Keys are read from environment variables:\nNEWS_API_KEY · REDDIT_CLIENT_ID · REDDIT_SECRET · X_BEARER_TOKEN",
           style={"fontSize": "11px", "color": "#888", "whiteSpace": "pre-line",
                  "marginBottom": "6px"}),
    dcc.Input(id="in-news-key", type="text", placeholder="NewsAPI key (optional)",
              debounce=True, style={**INPUT_FULL, "fontSize": "11px"}),
    html.Button("📰  Fetch Sentiment", id="btn-sentiment", n_clicks=0,
                style={**BTN_PRIMARY, "backgroundColor": "#6A1B9A"}),
    dcc.Loading(html.Div(id="sentiment-status",
                         style={"fontSize": "12px", "marginTop": "4px", "minHeight": "18px"}),
                type="circle"),
], style=SIDEBAR)

# ---- main content -----------------------------------------------------------

content = html.Div([
    dcc.Tabs(id="tabs", value="tab-overview", children=[
        dcc.Tab(label="Overview",        value="tab-overview",    style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="Stress Tests",    value="tab-stress",     style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="MC Regimes",      value="tab-regimes",    style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="Regime Switch",   value="tab-rswitch",    style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="Multi-Coin MC",   value="tab-multicoin",  style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="Single-Coin Sim", value="tab-single",     style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="Copula",          value="tab-copula",     style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="Risk Engine",     value="tab-riskengine",  style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="Sentiment",        value="tab-sentiment",   style=TAB_STYLE, selected_style=TAB_SEL),
        dcc.Tab(label="⚠ Systemic Risk",  value="tab-systemic",    style=TAB_STYLE, selected_style=TAB_SEL),
    ], style={"marginBottom": "16px"}),

    dcc.Loading(html.Div(id="tab-content"), type="default"),

    # hidden trigger stores
    dcc.Store(id="trg-analysis", data=0),
    dcc.Store(id="trg-single", data=0),
    dcc.Store(id="trg-sentiment", data=0),
], style=CONTENT)

app.layout = html.Div([sidebar, content])


# =============================================================================
# HELPER
# =============================================================================

def _placeholder(msg):
    return html.Div(msg, style={
        "padding": "80px 20px", "textAlign": "center",
        "color": "#aaa", "fontSize": "17px",
    })


def _var_cvar(final_prices, last_price):
    ret = (final_prices / last_price) - 1
    var5 = float(np.percentile(ret, 5))
    cvar5 = float(ret[ret <= var5].mean()) if np.any(ret <= var5) else var5
    return var5, cvar5


# =============================================================================
# CALLBACKS
# =============================================================================

# ---- 1. Fetch data ----------------------------------------------------------

@app.callback(
    Output("data-status", "children"),
    Input("btn-fetch", "n_clicks"),
    State("in-coin-set", "value"),
    prevent_initial_call=True,
)
def cb_fetch(_n, coin_set):
    try:
        _cache["coin_data"] = get_multi_coin_data(coin_set=coin_set or "top3")
        _cache["coin_set"]  = coin_set or "top3"
        coins = list(_cache["coin_data"].keys())
        prices = ", ".join(
            f"{c} ${_cache['coin_data'][c]['price'].iloc[-1]:,.0f}" for c in coins
        )
        return html.Span(f"✓ Loaded ({len(coins)} coins): {prices}", style={"color": "green"})
    except Exception as e:
        return html.Span(f"✗ {e}", style={"color": "red"})


# ---- 2. Run full analysis ---------------------------------------------------

@app.callback(
    [Output("run-status", "children"),
     Output("trg-analysis", "data")],
    Input("btn-run", "n_clicks"),
    [State("in-btc",  "value"), State("in-eth",  "value"), State("in-sol",  "value"),
     State("in-bnb",  "value"), State("in-xrp",  "value"), State("in-ada",  "value"),
     State("in-doge", "value"), State("in-avax", "value"), State("in-dot",  "value"),
     State("in-link", "value"),
     State("in-horizon", "value"), State("in-nsim", "value"),
     State("in-macro", "value"), State("in-quick-mode", "value")],
    prevent_initial_call=True,
)
def cb_run(_n, btc, eth, sol, bnb, xrp, ada, doge, avax, dot, link,
           horizon, nsim, macro, quick_mode):
    if "coin_data" not in _cache:
        return html.Span("✗ Fetch data first!", style={"color": "red"}), no_update

    try:
        cd    = _cache["coin_data"]
        coins = list(cd.keys())
        all_vals = dict(
            BTC=btc, ETH=eth, SOL=sol, BNB=bnb, XRP=xrp,
            ADA=ada, DOGE=doge, AVAX=avax, DOT=dot, LINK=link,
        )
        units = {
            sym: float(val)
            for sym, val in all_vals.items()
            if sym in coins and val and float(val) > 0
        }
        if not units:
            return html.Span("✗ Enter at least one holding > 0", style={"color": "red"}), no_update

        horizon = int(horizon or 90)
        # Quick Mode overrides the simulations dropdown
        if quick_mode:
            nsim = N_SIMS_QUICK
        else:
            nsim = int(nsim or 500)

        # betas
        rdf = pd.DataFrame({c: cd[c]["log_returns"] for c in coins}).dropna()
        betas = calculate_betas(rdf, benchmark="BTC")

        # portfolio summary
        portfolio = {}
        for c in coins:
            p = float(cd[c]["price"].iloc[-1])
            u = units.get(c, 0)
            portfolio[c] = {"units": u, "price": p, "value": u * p}
        total_val = sum(v["value"] for v in portfolio.values())
        for c in portfolio:
            portfolio[c]["weight"] = portfolio[c]["value"] / total_val if total_val else 0

        # stress test
        shocks = {
            "Crash −30 %": -0.30, "Severe Crash −50 %": -0.50,
            "Bull Rally +40 %": 0.40, "Mild Correction −15 %": -0.15,
        }
        df_stress = stress_test_portfolio(cd, units, shocks, betas, horizon)

        # MC normal + crisis
        df_normal = run_mc_portfolio(cd, units, betas, "normal", macro, horizon, nsim)
        df_crisis = run_mc_portfolio(cd, units, betas, "crisis", macro, horizon, nsim)

        # regime-switching MC
        df_rs = regime_switching_monte_carlo(cd, units, horizon_days=horizon, num_sim=nsim)

        # multi-coin MC
        mc_results = run_multi_coin_mc(cd, horizon, nsim)

        # copula
        cop_sim, cop_pct, cop_lp = run_copula(cd, coins, horizon, nsim)

        _cache["results"] = dict(
            portfolio=portfolio, total_value=total_val, betas=betas,
            df_stress=df_stress, df_normal=df_normal, df_crisis=df_crisis,
            df_rs=df_rs, mc=mc_results, units=units,
            copula=dict(sim_df=cop_sim, pct=cop_pct, lp=cop_lp),
            horizon=horizon, nsim=nsim,
        )
        return (
            html.Span(f"✓ Done — {nsim} sims × {horizon} days", style={"color": "green"}),
            _n,
        )
    except Exception:
        return html.Span(f"✗ {traceback.format_exc()[:200]}", style={"color": "red"}), no_update


# ---- 3. Run single-coin simulation -----------------------------------------

@app.callback(
    [Output("single-status", "children"),
     Output("trg-single", "data")],
    Input("btn-single", "n_clicks"),
    [State("in-coin", "value"), State("in-model", "value"),
     State("in-target", "value"), State("in-horizon", "value"),
     State("in-nsim", "value")],
    prevent_initial_call=True,
)
def cb_single(_n, coin, model, target, horizon, nsim):
    if "coin_data" not in _cache:
        return html.Span("✗ Fetch data first!", style={"color": "red"}), no_update
    try:
        cd = _cache["coin_data"]
        horizon = int(horizon or 90)
        nsim = int(nsim or 500)
        runners = {
            "gbm": run_gbm, "garch": run_garch,
            "historical": run_historical, "bootstrapped": run_bootstrapped,
        }
        sim_df, pct, lp = runners[model](cd, coin, horizon, nsim)
        fp = sim_df.iloc[-1].drop(["mean", "std"]).values.astype(float)
        var5, cvar5 = _var_cvar(fp, lp)
        color_map = {"gbm": "orange", "garch": "black",
                     "historical": "green", "bootstrapped": "purple"}
        _cache["single"] = dict(
            sim_df=sim_df, pct=pct, lp=lp, fp=fp, coin=coin, model=model,
            color=color_map.get(model, "blue"), target=float(target or 0),
            median=float(np.median(fp)),
            p5=float(np.percentile(fp, 5)),
            p95=float(np.percentile(fp, 95)),
            var5=var5, cvar5=cvar5,
        )
        return (
            html.Span(f"✓ {model.upper()} for {coin} done", style={"color": "green"}),
            _n,
        )
    except Exception:
        return html.Span(f"✗ {traceback.format_exc()[:200]}", style={"color": "red"}), no_update


# ---- 4. Fetch sentiment -----------------------------------------------------

@app.callback(
    [Output("sentiment-status", "children"),
     Output("trg-sentiment", "data")],
    Input("btn-sentiment", "n_clicks"),
    State("in-news-key", "value"),
    prevent_initial_call=True,
)
def cb_sentiment(_n, news_key):
    try:
        coins = list(_cache["coin_data"].keys()) if "coin_data" in _cache else ["BTC", "ETH", "SOL"]
        result = get_sentiment(
            coins=coins,
            force_refresh=True,
            news_api_key=news_key or NEWS_API_KEY or None,
            reddit_client_id=REDDIT_CLIENT_ID or None,
            reddit_secret=REDDIT_SECRET or None,
            x_bearer_token=X_BEARER_TOKEN or None,
        )
        _cache["sentiment"] = result
        sources_found = set()
        for v in result.values():
            sources_found.update(v.get("sources", []))
        if sources_found:
            return (
                html.Span(f"✓ Fetched from: {', '.join(sorted(sources_found))}",
                          style={"color": "green"}),
                _n,
            )
        errors = [v.get("error") for v in result.values() if v.get("error")]
        hint = errors[0] if errors else "No API keys set — add NEWS_API_KEY env var."
        return html.Span(f"⚠ No data. {hint}", style={"color": "#F57C00"}), no_update
    except Exception:
        return html.Span(f"✗ {traceback.format_exc()[:200]}", style={"color": "red"}), no_update


# ---- 5. Render tab content --------------------------------------------------

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "value"),
     Input("trg-analysis", "data"),
     Input("trg-single", "data"),
     Input("trg-sentiment", "data")],
)
def cb_tab(tab, _a, _s, _sent):

    # -- Overview --
    if tab == "tab-overview":
        if "results" not in _cache:
            return _placeholder("Click ▶ Run Full Analysis to start.")
        r  = _cache["results"]
        cd = _cache.get("coin_data", {})
        rows = []
        for c, d in r["portfolio"].items():
            rows.append({
                "Coin": c,
                "Units": f"{d['units']:.4f}",
                "Price": f"${d['price']:,.2f}",
                "Value": f"${d['value']:,.2f}",
                "Weight": f"{d['weight']:.1%}",
                "Beta (vs BTC)": f"{r['betas'].get(c, 'N/A'):.3f}" if isinstance(r["betas"].get(c), (int, float)) else "N/A",
            })
        # portfolio VaR from normal MC
        df_n = r["df_normal"]
        fp_port = df_n.iloc[-1].values.astype(float)
        var5, cvar5 = _var_cvar(fp_port, r["total_value"])

        # Performance metrics: Sharpe, Sortino, Max Drawdown
        perf_rows = []
        if cd:
            try:
                perf_df = compute_sharpe_sortino(cd)
                mdd_df  = compute_max_drawdown(cd)
                for coin in list(r["portfolio"].keys()):
                    if coin in perf_df.index and coin in mdd_df.index:
                        perf_rows.append({
                            "Coin":            coin,
                            "Ann. Return":     f"{perf_df.loc[coin, 'annualised_return']:.1%}",
                            "Ann. Vol":        f"{perf_df.loc[coin, 'annualised_vol']:.1%}",
                            "Sharpe":          f"{perf_df.loc[coin, 'sharpe']:.2f}",
                            "Sortino":         f"{perf_df.loc[coin, 'sortino']:.2f}",
                            "Max Drawdown":    f"{mdd_df.loc[coin, 'max_drawdown']:.1%}",
                            "DD Duration (d)": str(int(mdd_df.loc[coin, 'max_dd_duration_days'])),
                        })
            except Exception:
                pass

        # DefiLlama on-chain TVL (free, no key required)
        defi_df = get_defi_tvl(10)

        return html.Div([
            html.H4("Portfolio Overview", style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "Your current portfolio composition, dollar weights, betas vs BTC, "
                "and tail-risk statistics (VaR / CVaR) from Normal Monte Carlo. "
                "This is the starting point — understand what you hold before "
                "you stress-test it.",
                style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
            ),
            html.H4(f"Portfolio Value: ${r['total_value']:,.2f}",
                     style={"marginBottom": "12px"}),
            dash_table.DataTable(
                columns=[{"name": k, "id": k} for k in rows[0]],
                data=rows,
                style_table={"width": "780px", "marginBottom": "18px"},
                style_header={"backgroundColor": "#1a237e", "color": "white",
                              "fontWeight": "bold", "fontSize": "13px"},
                style_cell={"textAlign": "center", "padding": "8px",
                            "fontSize": "13px"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f5f5f5"}
                ],
            ),
            html.Div([
                html.Span("Portfolio Risk (Normal MC)  ·  ",
                           style={"fontWeight": "bold"}),
                html.Span(f"VaR 5 %: {var5:.2%}"),
                html.Span("  ·  "),
                html.Span(f"Expected Shortfall 5 %: {cvar5:.2%}"),
            ], style={"padding": "8px 0", "fontSize": "14px", "color": "#333"}),
            html.P(f"Horizon: {r['horizon']} days  |  Simulations: {r['nsim']}",
                   style={"color": "#888", "fontSize": "12px"}),

            # ── Performance Metrics (Sharpe, Sortino, Max Drawdown) ─────────
            *([html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "16px"}),
               html.H5("Performance Metrics — Sharpe · Sortino · Max Drawdown",
                       style={"color": "#1a237e", "marginTop": "12px"}),
               html.P(
                   "Sharpe = annualised excess return ÷ vol.  "
                   "Sortino = annualised excess return ÷ downside deviation (penalises losses only).  "
                   "Max Drawdown = worst peak-to-trough fall over full history.",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "8px"},
               ),
               dash_table.DataTable(
                   columns=[{"name": k, "id": k} for k in perf_rows[0]],
                   data=perf_rows,
                   style_table={"width": "820px", "marginBottom": "18px"},
                   style_header={"backgroundColor": "#1a237e", "color": "white",
                                 "fontWeight": "bold", "fontSize": "12px"},
                   style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
                   style_data_conditional=[
                       {"if": {"row_index": "odd"}, "backgroundColor": "#f5f5f5"},
                       {"if": {"column_id": "Max Drawdown"},
                        "color": "#d32f2f", "fontWeight": "bold"},
                       {"if": {"column_id": "Sharpe"},
                        "color": "#1976D2", "fontWeight": "bold"},
                   ],
               )] if perf_rows else []),

            # ── DefiLlama On-Chain TVL ───────────────────────────────────────
            *([html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "16px"}),
               html.H5("⛓  DeFi On-Chain — Top 10 Protocols by TVL",
                       style={"color": "#1a237e", "marginTop": "12px"}),
               html.P(
                   "Source: DefiLlama (free, no API key).  "
                   "TVL = Total Value Locked — a proxy for DeFi ecosystem health.",
                   style={"color": "#888", "fontSize": "12px", "marginBottom": "8px"},
               ),
               dash_table.DataTable(
                   columns=[{"name": k, "id": k} for k in defi_df.columns],
                   data=defi_df.to_dict("records"),
                   style_table={"width": "820px", "marginBottom": "18px"},
                   style_header={"backgroundColor": "#388E3C", "color": "white",
                                 "fontWeight": "bold", "fontSize": "12px"},
                   style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
                   style_data_conditional=[
                       {"if": {"row_index": "odd"}, "backgroundColor": "#f5f5f5"},
                       {"if": {"filter_query": "{Change 1d (%)} > 0", "column_id": "Change 1d (%)"},
                        "color": "#388E3C", "fontWeight": "bold"},
                       {"if": {"filter_query": "{Change 1d (%)} < 0", "column_id": "Change 1d (%)"},
                        "color": "#d32f2f", "fontWeight": "bold"},
                   ],
               )] if not defi_df.empty else []),
        ])

    # -- Stress Tests --
    if tab == "tab-stress":
        if "results" not in _cache:
            return _placeholder("Run Full Analysis first.")
        return html.Div([
            html.H4("Stress Test Scenarios", style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "Deterministic shock scenarios applied to your portfolio: "
                "Crash (−30%), Severe Crash (−50%), Bull Rally (+40%), "
                "Mild Correction (−15%). Each scenario scales returns by the "
                "shock factor, adjusted for individual coin betas vs BTC and "
                "liquidity conditions. Helps answer: \"How bad can it get?\" "
                "under known macro triggers.",
                style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
            ),
            dcc.Graph(figure=fig_stress(_cache["results"]["df_stress"])),
        ])

    # -- MC Regimes (Normal vs Crisis) --
    if tab == "tab-regimes":
        if "results" not in _cache:
            return _placeholder("Run Full Analysis first.")
        r = _cache["results"]
        return html.Div([
            html.H4("Monte Carlo — Normal vs Crisis Regime", style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "Two parallel Monte Carlo simulations of your portfolio: "
                "Normal (historical vol) and Crisis (vol ×2.3, correlations pushed to 0.85). "
                "The blue fan is what you expect in calm markets; the red fan is what history "
                "says happens during liquidation cascades. The gap between them is your "
                "\"regime risk\" — the loss you don’t see coming until correlations spike.",
                style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
            ),
            dcc.Graph(figure=fig_regimes(r["df_normal"], r["df_crisis"], r["total_value"])),
            html.P("Blue = Normal volatility · Red = Crisis volatility. "
                   "Bands show 5th–95th percentile.",
                   style={"color": "#666", "fontSize": "12px", "paddingLeft": "12px"}),
        ])

    # -- Regime Switching --
    if tab == "tab-rswitch":
        if "results" not in _cache:
            return _placeholder("Run Full Analysis first.")
        return html.Div([
            html.H4("Markov Regime-Switching Monte Carlo", style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "A four-state Markov chain (Bull / Sideways / Bear / Crisis) drives "
                "the drift and volatility of each simulation step. The model detects "
                "which regime the market is currently in and simulates realistic "
                "transitions — including sudden jumps from calm to crisis. This is more "
                "realistic than a single-volatility MC because crypto regimes cluster.",
                style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
            ),
            dcc.Graph(figure=fig_regime_switch(_cache["results"]["df_rs"])),
            html.P("Markov regime-switching model with Bull / Bear / Sideways / Crisis states.",
                   style={"color": "#666", "fontSize": "12px", "paddingLeft": "12px"}),
        ])

    # -- Multi-Coin MC --
    if tab == "tab-multicoin":
        if "results" not in _cache:
            return _placeholder("Run Full Analysis first.")
        children = [
            html.H4("Multi-Coin Monte Carlo", style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "Independent GBM-based Monte Carlo for each coin (BTC, ETH, SOL). "
                "Fan charts show the median path with a 5th–95th percentile band. "
                "Terminal distributions show the probability of ending above or below "
                "key levels. VaR (5%) = worst loss you should expect 95% of the time; "
                "CVaR = average loss in the worst 5% of scenarios.",
                style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
            ),
        ]
        for coin, d in _cache["results"]["mc"].items():
            var5, cvar5 = _var_cvar(d["final_prices"], d["last_price"])
            target = d["last_price"] * 1.2  # default target = +20 %
            children.extend([
                html.H5(f"{coin}", style={"marginTop": "16px", "color": "#1a237e"}),
                dcc.Graph(figure=fig_fan(d["sim_df"], d["percentiles"], d["last_price"],
                                        f"Monte Carlo — {coin} Price Forecast", "blue")),
                dcc.Graph(figure=fig_hist(d["final_prices"], d["last_price"], target,
                                         f"Terminal Distribution — {coin} (target +20 %)")),
                html.P(
                    f"Median: ${d['median']:,.0f}  ·  "
                    f"P5: ${d['p5']:,.0f}  ·  P95: ${d['p95']:,.0f}  ·  "
                    f"VaR 5 %: {var5:.2%}  ·  CVaR 5 %: {cvar5:.2%}",
                    style={"fontWeight": "bold", "fontSize": "13px", "padding": "4px 12px"},
                ),
                html.Hr(),
            ])
        return html.Div(children)

    # -- Single-Coin Sim --
    if tab == "tab-single":
        if "single" not in _cache:
            return _placeholder("Select a coin & model in the sidebar, then click ▶ Run.")
        s = _cache["single"]
        return html.Div([
            html.H4("Single-Coin Simulation", style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "Run one simulation model on a single coin. Choose between: "
                "GBM (classic log-normal random walk), GARCH(1,1) (volatility-clustering model), "
                "Historical Simulation (resample actual past returns), or "
                "Bootstrapped Paths (block-bootstrap to preserve autocorrelation). "
                "Compare models to see which assumptions matter most for your forecast.",
                style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
            ),
            html.H5(f"{s['model'].upper()} Simulation — {s['coin']}",
                     style={"color": "#1a237e"}),
            dcc.Graph(figure=fig_fan(s["sim_df"], s["pct"], s["lp"],
                                    f"{s['model'].upper()} — {s['coin']} Price Forecast",
                                    s["color"])),
            html.P(
                f"Median: ${s['median']:,.0f}  ·  P5: ${s['p5']:,.0f}  ·  "
                f"P95: ${s['p95']:,.0f}  ·  VaR 5 %: {s['var5']:.2%}  ·  "
                f"CVaR 5 %: {s['cvar5']:.2%}",
                style={"fontWeight": "bold", "fontSize": "14px", "padding": "6px 12px"},
            ),
            dcc.Graph(figure=fig_hist(s["fp"], s["lp"], s["target"],
                                     f"Terminal Distribution — {s['coin']} ({s['model'].upper()})")),
        ])

    # -- Copula --
    if tab == "tab-copula":
        if "results" not in _cache:
            return _placeholder("Run Full Analysis first.")
        c = _cache["results"]["copula"]
        fp = c["sim_df"].iloc[-1].drop(["mean", "std"]).values.astype(float)
        var5, cvar5 = _var_cvar(fp, c["lp"])
        return html.Div([
            html.H4("Gaussian Copula Simulation", style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "The Copula model captures non-linear tail dependencies between assets "
                "that regular correlation matrices miss. We fit marginal distributions "
                "per coin, then use a Gaussian copula to link them. This means extreme "
                "moves (crashes) can be more correlated than calm periods — a critical "
                "feature for portfolio risk that standard MC ignores.",
                style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
            ),
            html.H5("Gaussian Copula — Portfolio Simulation", style={"color": "#1a237e"}),
            dcc.Graph(figure=fig_fan(c["sim_df"], c["pct"], c["lp"],
                                    "Copula Correlation Simulation — Portfolio", "brown")),
            html.P(
                f"Median: ${float(np.median(fp)):,.0f}  ·  "
                f"P5: ${float(np.percentile(fp, 5)):,.0f}  ·  "
                f"P95: ${float(np.percentile(fp, 95)):,.0f}  ·  "
                f"VaR 5 %: {var5:.2%}  ·  CVaR 5 %: {cvar5:.2%}",
                style={"fontWeight": "bold", "fontSize": "14px", "padding": "6px 12px"},
            ),
            dcc.Graph(figure=fig_hist(fp, c["lp"], 0,
                                     "Copula Portfolio — Terminal Distribution")),
        ])

    # -- Risk Engine --
    if tab == "tab-riskengine":
        if "coin_data" not in _cache:
            return _placeholder("Fetch data first.")
        cd = _cache["coin_data"]
        r  = _cache.get("results")
        units = r["units"] if r else {"BTC": 0.5, "ETH": 2.0, "SOL": 10.0}

        # ---- shared table style helper ----
        def _tbl(rows, width=None):
            style = {"marginBottom": "18px"}
            if width:
                style["maxWidth"] = width
            return dash_table.DataTable(
                columns=[{"name": k, "id": k} for k in rows[0]],
                data=rows,
                style_header={
                    "backgroundColor": "#1a237e", "color": "white",
                    "fontWeight": "bold", "fontSize": "12px",
                },
                style_cell={
                    "textAlign": "center", "padding": "6px",
                    "fontSize": "12px",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#f5f7ff"},
                ],
                style_table=style,
            )

        # ---- badge helper ----
        def _badge(label, color):
            return html.Span(
                label,
                style={
                    "backgroundColor": color, "color": "white",
                    "borderRadius": "4px", "padding": "3px 10px",
                    "fontSize": "12px", "fontWeight": "bold",
                    "marginLeft": "10px", "verticalAlign": "middle",
                },
            )

        try:
            # ==============================================================
            # DATA LAYER
            # ==============================================================

            # 1 — distribution statistics
            stats_df = compute_return_stats(cd)
            stat_rows = []
            for coin in stats_df.index:
                row = {"Coin": coin}
                for col in stats_df.columns:
                    v = stats_df.loc[coin, col]
                    row[col] = f"{v:.4f}" if isinstance(v, float) else str(v)
                stat_rows.append(row)

            # performance snapshot (Sharpe, Sortino, Max DD)
            _perf_df = compute_sharpe_sortino(cd)
            _mdd_df  = compute_max_drawdown(cd)
            perf_snap_rows = []
            for _c in _perf_df.index:
                if _c in _mdd_df.index:
                    perf_snap_rows.append({
                        "Coin":            _c,
                        "Ann. Return":     f"{_perf_df.loc[_c, 'annualised_return']:.1%}",
                        "Ann. Vol":        f"{_perf_df.loc[_c, 'annualised_vol']:.1%}",
                        "Sharpe":          f"{_perf_df.loc[_c, 'sharpe']:.2f}",
                        "Sortino":         f"{_perf_df.loc[_c, 'sortino']:.2f}",
                        "Max Drawdown":    f"{_mdd_df.loc[_c, 'max_drawdown']:.1%}",
                        "DD Duration (d)": str(int(_mdd_df.loc[_c, 'max_dd_duration_days'])),
                    })

            # 2 — covariance matrix (PSD-validated)
            cov_df, was_psd = covariance_matrix(cd, annualise=True)
            cov_rows = []
            for coin_row in cov_df.index:
                row = {"": coin_row}
                for coin_col in cov_df.columns:
                    row[coin_col] = f"{cov_df.loc[coin_row, coin_col]:.6f}"
                cov_rows.append(row)

            # 3 — portfolio risk decomposition
            rd = portfolio_risk_decomposition(cd, units)
            rc_rows = []
            rc_sum = sum(rd["pct_risk_contribution"].values())
            for c in rd["coins"]:
                rc_rows.append({
                    "Coin": c,
                    "Weight": f"{rd['weights'][c]:.1%}",
                    "Marginal RC": f"{rd['marginal_rc'][c]:.4f}",
                    "Component RC": f"{rd['component_rc'][c]:.4f}",
                    "% Risk": f"{rd['pct_risk_contribution'][c]:.1%}",
                })

            # 4 — correlation shock model
            sv = stressed_portfolio_vol(cd, units)

            # 5 — PCA systemic risk
            pca = pca_risk(cd)
            pca_rows = []
            for i, coin in enumerate(pca["coins"]):
                pca_rows.append({
                    "PC": f"PC{i+1}",
                    "Eigenvalue": f"{pca['eigenvalues'][i]:.4f}",
                    "Var Explained": f"{pca['explained_variance_ratio'][i]:.1%}",
                    "Cumulative": f"{pca['cumulative_explained'][i]:.1%}",
                    "PC1 Loading": f"{list(pca['first_pc_loadings'].values())[i]:+.4f}",
                })

            # 6 — VaR comparison (Normal MC vs Crisis MC vs Historical)
            vc = var_comparison(cd, units, num_sim=5_000)
            vc_rows = [
                {"Scenario": "Normal MC",
                 "VaR (90 d, 5 %)": f"${vc['normal_var']:,.0f}",
                 "CVaR (90 d, 5 %)": f"${vc['normal_cvar']:,.0f}"},
                {"Scenario": "Crisis MC (vol ×2, corr 0.85)",
                 "VaR (90 d, 5 %)": f"${vc['crisis_var']:,.0f}",
                 "CVaR (90 d, 5 %)": f"${vc['crisis_cvar']:,.0f}"},
                {"Scenario": "Historical (1 d)",
                 "VaR (90 d, 5 %)": f"${vc['hist_var_1d']:,.0f}",
                 "CVaR (90 d, 5 %)": f"${vc['hist_cvar_1d']:,.0f}"},
                {"Scenario": "Historical (90 d scaled)",
                 "VaR (90 d, 5 %)": f"${vc['hist_var_90d']:,.0f}",
                 "CVaR (90 d, 5 %)": f"${vc['hist_cvar_90d']:,.0f}"},
            ]

            # 7 — Kupiec backtest
            bt = backtest_var(cd, units, confidence=0.05, window=60)

            # ==============================================================
            # CHART LAYER
            # ==============================================================

            # VaR comparison histogram overlay
            fig_vc = go.Figure()
            fig_vc.add_trace(go.Histogram(
                x=vc["normal_pnl"], nbinsx=80, name="Normal MC",
                marker_color="rgba(33,150,243,0.55)",
                autobinx=False,
                xbins=dict(size=abs(vc["normal_pnl"].max() - vc["normal_pnl"].min()) / 80),
            ))
            fig_vc.add_trace(go.Histogram(
                x=vc["crisis_pnl"], nbinsx=80, name="Crisis MC",
                marker_color="rgba(244,67,54,0.55)",
                autobinx=False,
                xbins=dict(size=abs(vc["crisis_pnl"].max() - vc["crisis_pnl"].min()) / 80),
            ))
            fig_vc.add_vline(x=-vc["normal_var"], line_dash="dash",
                             line_color="#1565C0",
                             annotation_text=f"Normal VaR ${vc['normal_var']:,.0f}",
                             annotation_font_size=11)
            fig_vc.add_vline(x=-vc["crisis_var"], line_dash="dot",
                             line_color="#b71c1c",
                             annotation_text=f"Crisis VaR ${vc['crisis_var']:,.0f}",
                             annotation_font_size=11)
            fig_vc.update_layout(
                title=f"P&L Distribution — Normal vs Crisis MC ({vc['horizon_days']}-day horizon)",
                xaxis_title="Portfolio P&L ($)", yaxis_title="Frequency",
                barmode="overlay", template="plotly_white", height=420,
            )

            # Kupiec breach chart
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Bar(
                x=bt["actual_pnl"].index,
                y=bt["actual_pnl"].values,
                name="Daily P&L",
                marker_color=np.where(bt["breaches"].reindex(bt["actual_pnl"].index).fillna(False),
                                      "#f44336", "#90CAF9"),
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt["rolling_var"].index,
                y=-bt["rolling_var"].values,
                mode="lines", name="−VaR (5 %)",
                line=dict(color="#1a237e", width=2, dash="dash"),
            ))
            fig_bt.update_layout(
                title=(
                    f"VaR Backtest — Breaches: {bt['n']}/{bt['T']}  "
                    f"({bt['breach_rate']:.1%} vs {bt['expected_rate']:.1%} expected)"
                ),
                xaxis_title="Date", yaxis_title="P&L ($)",
                template="plotly_white", height=420,
            )

            # Rolling volatility
            rvol = rolling_volatility(cd, (30, 60, 90))
            fig_rvol = go.Figure()
            _pal = ["#2196F3", "#f44336", "#4CAF50", "#FF9800", "#9C27B0",
                    "#607D8B", "#795548", "#009688", "#E91E63"]
            for i, col in enumerate(rvol.columns):
                fig_rvol.add_trace(go.Scatter(
                    x=rvol.index, y=rvol[col], mode="lines",
                    name=col, line=dict(color=_pal[i % len(_pal)]),
                ))
            fig_rvol.update_layout(
                title="Rolling Annualised Volatility (30 / 60 / 90 d)",
                xaxis_title="Date", yaxis_title="Annualised Vol",
                template="plotly_white", height=400,
            )

            # Rolling beta
            rb = rolling_beta(cd, "BTC", 60)
            fig_rb = go.Figure()
            for col in rb.columns:
                fig_rb.add_trace(go.Scatter(
                    x=rb.index, y=rb[col], mode="lines",
                    name=col.replace("_beta", ""),
                ))
            fig_rb.add_hline(y=1.0, line_dash="dash", line_color="gray",
                             annotation_text="\u03b2 = 1")
            fig_rb.add_hline(y=1.5, line_dash="dot", line_color="red",
                             annotation_text="Leverage regime")
            fig_rb.update_layout(
                title="Rolling 60-Day Beta vs BTC",
                xaxis_title="Date", yaxis_title="Beta",
                template="plotly_white", height=400,
            )

            # Beta regime
            br = beta_regime_summary(cd, "BTC", 60)
            _regime_clr = {
                "leverage": "#f44336", "amplified": "#FF9800",
                "normal": "#4CAF50", "decoupled": "#2196F3",
            }
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(
                x=br.index, y=br["avg_alt_beta"], mode="lines",
                line=dict(color="#333", width=2), name="Avg Alt Beta",
            ))
            for regime_name, clr in _regime_clr.items():
                mask = br["regime"] == regime_name
                if mask.any():
                    fig_br.add_trace(go.Scatter(
                        x=br.index[mask], y=br["avg_alt_beta"][mask],
                        mode="markers", marker=dict(color=clr, size=5),
                        name=regime_name.capitalize(),
                    ))
            fig_br.add_hline(y=1.5, line_dash="dot", line_color="red",
                             annotation_text="Leverage threshold")
            fig_br.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig_br.update_layout(
                title="Average Altcoin Beta — Regime Detection",
                xaxis_title="Date", yaxis_title="Avg Beta vs BTC",
                template="plotly_white", height=400,
            )

            # Rolling correlation
            rc = rolling_correlation(cd, 60)
            fig_rc = go.Figure()
            for col in rc.columns:
                fig_rc.add_trace(go.Scatter(
                    x=rc.index, y=rc[col], mode="lines", name=col,
                ))
            fig_rc.update_layout(
                title="Rolling 60-Day Pairwise Correlation",
                xaxis_title="Date", yaxis_title="Correlation",
                template="plotly_white", height=400,
            )

            # Static correlation heatmap
            _rdf_corr = pd.DataFrame({co: cd[co]["log_returns"] for co in cd}).dropna()
            _corr_mat = _rdf_corr.corr()
            _cns      = list(_corr_mat.columns)
            fig_heatmap = go.Figure(go.Heatmap(
                z=_corr_mat.values.tolist(),
                x=_cns,
                y=_cns,
                colorscale="RdBu_r",
                zmin=-1, zmax=1,
                texttemplate="%{z:.2f}",
                hovertemplate="%{y} — %{x}: %{z:.3f}<extra></extra>",
            ))
            fig_heatmap.update_layout(
                title="Full-Period (365 d) Return Correlation Heatmap",
                template="plotly_white",
                height=max(350, 55 * len(_cns)),
            )

            # Rolling parametric VaR
            rv = rolling_var(cd, units, window=60)
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(
                x=rv.index, y=rv.values, mode="lines",
                line=dict(color="#f44336", width=2), name="1-Day VaR (5 %)",
            ))
            fig_rv.update_layout(
                title="Rolling 1-Day Portfolio VaR (5 %, Parametric)",
                xaxis_title="Date", yaxis_title="VaR ($)",
                template="plotly_white", height=400,
            )

            # ==============================================================
            # TEXT / BADGE LAYER
            # ==============================================================

            psd_badge = (_badge("PSD ✓", "#388E3C") if was_psd
                         else _badge("PSD fixed ✓", "#F57C00"))

            systemic_text = (
                f"SYSTEMIC: First PC explains {pca['first_pc_explains']:.0%} of variance — "
                f"market is highly correlated."
                if pca["systemic_flag"] else
                f"First PC explains {pca['first_pc_explains']:.0%} "
                f"— moderate diversification present."
            )
            sys_color = "#d32f2f" if pca["systemic_flag"] else "#388E3C"

            kupiec_badge = (
                _badge("PASS ✓", "#388E3C") if bt["kupiec_pass"]
                else _badge("FAIL ✗", "#d32f2f")
            )

            # ==============================================================
            # LAYOUT ASSEMBLY
            # ==============================================================

            _section = {"marginTop": "28px", "color": "#1a237e"}
            _hint    = {"color": "#666", "fontSize": "12px", "marginBottom": "8px"}

            return html.Div([
                html.H4("Risk Engine",
                        style={"color": "#1a237e", "marginBottom": "4px"}),
                html.P("Institutional-grade portfolio risk analytics · "
                       "Parametric · Historical · PCA · Kupiec backtest",
                       style={"color": "#888", "fontSize": "12px",
                              "marginBottom": "20px"}),

                # divider
                html.Hr(style={"borderColor": "#e8eaf6"}),

                # ── 1 · Distribution Statistics ──────────────────────────
                html.H5("1 · Distribution Statistics", style=_section),
                html.P("Log-return shape: annualised return, annualised vol, "
                       "skewness, excess kurtosis, observation count.",
                       style=_hint),
                _tbl(stat_rows),

                # ── 2 · Covariance Matrix ────────────────────────────────
                html.H5([
                    "2 · Annualised Covariance Matrix",
                    psd_badge,
                ], style=_section),
                html.P(
                    "Sample covariance × 365.  "
                    "Eigenvalue-clipping applied if matrix was not PSD.",
                    style=_hint,
                ),
                _tbl(cov_rows, width="700px"),

                # ── 3 · Portfolio Risk Contribution ──────────────────────
                html.H5("3 · Portfolio Risk Contribution  (wᵀΣw)",
                        style=_section),
                html.Div([
                    html.Span(
                        f"Portfolio Annual Vol: {rd['portfolio_vol_annual']:.2%}",
                        style={"fontWeight": "bold", "fontSize": "14px",
                               "marginRight": "20px"},
                    ),
                    html.Span(
                        f"RC sum: {rc_sum:.4f}",
                        style={"fontSize": "12px", "color": "#888"},
                    ),
                ], style={"marginBottom": "8px"}),
                _tbl(rc_rows),

                # ── 4 · Correlation Shock Model ──────────────────────────
                html.H5("4 · Correlation Shock Model", style=_section),
                html.P("Stressed scenario: vol ×2, all pairwise correlations "
                       "floored at 0.85.", style=_hint),
                html.Div([
                    html.Span(f"Normal Vol: {sv['normal_vol']:.2%}",
                              style={"fontWeight": "bold",
                                     "marginRight": "24px"}),
                    html.Span(f"Stressed Vol: {sv['stressed_vol']:.2%}",
                              style={"fontWeight": "bold", "color": "#d32f2f",
                                     "marginRight": "24px"}),
                    html.Span(f"Increase: +{sv['vol_increase_pct']:.0%}",
                              style={"fontWeight": "bold",
                                     "color": "#F57C00"}),
                ], style={"padding": "8px 0", "fontSize": "14px",
                          "marginBottom": "18px"}),

                # ── 5 · PCA Systemic Risk ────────────────────────────────
                html.H5("5 · PCA — Systemic Risk Index", style=_section),
                html.P(systemic_text,
                       style={"fontWeight": "bold", "fontSize": "14px",
                              "color": sys_color, "marginBottom": "8px"}),
                _tbl(pca_rows, width="680px"),

                html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

                # ── 6 · VaR Comparison ───────────────────────────────────
                html.H5(
                    f"6 · VaR Comparison  "
                    f"(confidence 5 %, {vc['horizon_days']}-day horizon, "
                    f"portfolio ${vc['port_val']:,.0f})",
                    style=_section,
                ),
                html.P(
                    f"Crisis/Normal VaR ratio: "
                    f"{vc['crisis_normal_ratio']:.2f}×  — "
                    "how much worse a crisis scenario makes your tail risk.",
                    style={**_hint, "fontWeight": "bold", "color": "#b71c1c"},
                ),
                _tbl(vc_rows, width="700px"),
                dcc.Graph(figure=fig_vc),

                html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

                # ── 7 · Kupiec VaR Backtest ──────────────────────────────
                html.H5(["7 · VaR Backtest — Kupiec POF Test", kupiec_badge],
                        style=_section),
                html.P(
                    "Proportion-of-Failures test: LR ~ χ²(1).  "
                    "Critical value 3.841 (5 %).  "
                    "Red bars = VaR breach days.",
                    style=_hint,
                ),
                _tbl([{
                    "Obs (T)":        bt["T"],
                    "Breaches (n)":   bt["n"],
                    "Breach rate":    f"{bt['breach_rate']:.2%}",
                    "Expected rate":  f"{bt['expected_rate']:.2%}",
                    "Kupiec LR":      f"{bt['kupiec_lr']:.3f}",
                    "p-value":        f"{bt['kupiec_pvalue']:.4f}",
                    "Decision":       "Pass — model valid" if bt["kupiec_pass"]
                                      else "Fail — model underestimates risk",
                }], width="900px"),
                dcc.Graph(figure=fig_bt),

                html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

                # ── 8 · Rolling Risk Metrics ─────────────────────────────
                html.H5("8 · Rolling Risk Metrics", style=_section),

                html.H6("Rolling Annualised Volatility (30 / 60 / 90 d)",
                        style={"color": "#37474f", "marginTop": "16px"}),
                dcc.Graph(figure=fig_rvol),

                html.H6("Rolling 60-Day Beta vs BTC",
                        style={"color": "#37474f", "marginTop": "16px"}),
                dcc.Graph(figure=fig_rb),

                html.H6("Beta Regime Detection",
                        style={"color": "#37474f", "marginTop": "16px"}),
                html.P("leverage > 1.5  ·  amplified 1.0–1.5  ·  "
                       "normal 0.5–1.0  ·  decoupled < 0.5",
                       style=_hint),
                dcc.Graph(figure=fig_br),

                html.H6("Rolling 60-Day Pairwise Correlation",
                        style={"color": "#37474f", "marginTop": "16px"}),
                dcc.Graph(figure=fig_rc),

                html.H6("Correlation Heatmap (Full 365-Day Period)",
                        style={"color": "#37474f", "marginTop": "16px"}),
                html.P("Static full-period heatmap.  "
                       "Red = strong positive correlation, Blue = negative / uncorrelated.",
                       style={"color": "#888", "fontSize": "12px", "marginBottom": "6px"}),
                dcc.Graph(figure=fig_heatmap),

                html.H6("Rolling 1-Day Portfolio VaR (5 %, Parametric)",
                        style={"color": "#37474f", "marginTop": "16px"}),
                dcc.Graph(figure=fig_rv),

                # ── section 9: performance snapshot ────────────────────────────────
                html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),
                html.H5("9 · Performance Snapshot — Sharpe · Sortino · Max Drawdown",
                        style={"color": "#1a237e"}),
                html.P(
                    "Sharpe = annualised excess return ÷ volatility.  "
                    "Sortino uses downside deviation only.  "
                    "Max Drawdown = worst peak-to-trough decline over full price history.",
                    style={"color": "#666", "fontSize": "12px", "marginBottom": "8px"},
                ),
                *([dash_table.DataTable(
                    columns=[{"name": k, "id": k} for k in perf_snap_rows[0]],
                    data=perf_snap_rows,
                    style_table={"width": "820px", "marginBottom": "18px"},
                    style_header={"backgroundColor": "#1a237e", "color": "white",
                                  "fontWeight": "bold", "fontSize": "12px"},
                    style_cell={"textAlign": "center", "padding": "6px",
                                "fontSize": "12px"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#f5f7ff"},
                        {"if": {"column_id": "Max Drawdown"},
                         "color": "#d32f2f", "fontWeight": "bold"},
                        {"if": {"column_id": "Sharpe"},
                         "color": "#1976D2", "fontWeight": "bold"},
                    ],
                )] if perf_snap_rows else
                [html.P("No performance data.", style={"color": "#aaa"})]),

            ], style={"padding": "0 8px 40px 8px"})

        except Exception:
            return _placeholder(f"Risk Engine error:\n{traceback.format_exc()[:400]}")

    # -- Social Sentiment --
    if tab == "tab-sentiment":
        if "sentiment" not in _cache:
            return _placeholder("Click 📰 Fetch Sentiment in the sidebar first.")

        cd = _cache.get("coin_data", {})
        sent = _cache["sentiment"]
        children = []
        children.append(html.H4("Social Sentiment — Positioning Layer", style={"color": "#6A1B9A", "marginBottom": "4px"}))
        children.append(html.P(
            "Aggregated VADER sentiment from NewsAPI, Reddit, and X/Twitter. "
            "Scores range from −1 (bearish) to +1 (bullish). "
            "Beyond raw direction, we compute sentiment z-scores (shock magnitude) "
            "and measure price sensitivity per unit sentiment change — conditioned "
            "on the systemic risk regime. Scroll down for the Positioning Layer analysis.",
            style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
        ))

        # ---- Summary cards ----
        score_cards = []
        for coin, s in sent.items():
            score = s["current"]
            clr  = "#388E3C" if score > 0.05 else ("#d32f2f" if score < -0.05 else "#F57C00")
            label = "Bullish" if score > 0.05 else ("Bearish" if score < -0.05 else "Neutral")
            src_txt = ", ".join(s["sources"]) if s["sources"] else "no data"
            score_cards.append(
                html.Div([
                    html.H5(coin, style={"margin": "0 0 4px 0", "color": "#333"}),
                    html.Div(f"{score:+.3f}",
                             style={"fontSize": "28px", "fontWeight": "bold", "color": clr}),
                    html.Div(label, style={"fontSize": "13px", "color": clr}),
                    html.Div(f"Sources: {src_txt}",
                             style={"fontSize": "11px", "color": "#aaa", "marginTop": "4px"}),
                ], style={
                    "border": f"2px solid {clr}", "borderRadius": "8px",
                    "padding": "16px 24px", "minWidth": "150px", "textAlign": "center",
                    "backgroundColor": "#fafafa",
                })
            )
        children.append(
            html.Div(score_cards, style={"display": "flex", "gap": "16px",
                                          "flexWrap": "wrap", "marginBottom": "24px"})
        )

        # ---- Sentiment over time chart ----
        fig_sent_time = go.Figure()
        palette_sent = {"BTC": "#FF9800", "ETH": "#2196F3", "SOL": "#9C27B0"}
        for coin, s in sent.items():
            bd = s["by_day"]
            if bd.empty:
                continue
            clr = palette_sent.get(coin, "#333")
            fig_sent_time.add_trace(go.Scatter(
                x=bd["ts"], y=bd["sentiment"], mode="lines+markers",
                name=coin, line=dict(color=clr, width=2),
                marker=dict(size=5),
                error_y=dict(type="data", array=bd["std"].tolist(),
                             visible=True, color="rgba(0,0,0,0.15)"),
            ))
        fig_sent_time.add_hline(y=0, line_dash="dash", line_color="gray",
                                annotation_text="Neutral")
        fig_sent_time.add_hrect(y0=0.05, y1=1,  fillcolor="rgba(76,175,80,0.05)",
                                line_width=0, annotation_text="Bullish zone")
        fig_sent_time.add_hrect(y0=-1,  y1=-0.05, fillcolor="rgba(244,67,54,0.05)",
                                line_width=0, annotation_text="Bearish zone")
        fig_sent_time.update_layout(
            title="Daily Sentiment Score Over Time",
            xaxis_title="Date", yaxis_title="Sentiment (VADER compound)",
            yaxis=dict(range=[-1, 1]),
            template="plotly_white", height=400,
        )
        children.append(html.H5("Sentiment Over Time", style={"color": "#6A1B9A"}))
        children.append(dcc.Graph(figure=fig_sent_time))

        # ---- Mention volume bar chart ----
        fig_vol = go.Figure()
        for coin, s in sent.items():
            bd = s["by_day"]
            if bd.empty:
                continue
            fig_vol.add_trace(go.Bar(
                x=bd["ts"], y=bd["volume"], name=coin,
                marker_color=palette_sent.get(coin, "#888"), opacity=0.75,
            ))
        fig_vol.update_layout(
            title="Daily Mention Volume",
            xaxis_title="Date", yaxis_title="Mentions",
            barmode="group", template="plotly_white", height=320,
        )
        children.append(html.H5("Mention Volume", style={"color": "#6A1B9A", "marginTop": "16px"}))
        children.append(dcc.Graph(figure=fig_vol))

        # ---- Sentiment vs Returns scatter ----
        if cd:
            try:
                svr = sentiment_vs_returns(sent, cd)
                if not svr.empty:
                    fig_scatter = go.Figure()
                    for coin in svr["coin"].unique():
                        sub = svr[svr["coin"] == coin]
                        fig_scatter.add_trace(go.Scatter(
                            x=sub["sentiment"], y=sub["return_1d"] * 100,
                            mode="markers",
                            marker=dict(color=palette_sent.get(coin, "#888"),
                                        size=7, opacity=0.7),
                            name=coin,
                            text=sub["date"].astype(str),
                            hovertemplate="%{text}<br>Sentiment: %{x:.3f}<br>Return: %{y:.2f}%",
                        ))
                    # Trendline per coin
                    for coin in svr["coin"].unique():
                        sub = svr[svr["coin"] == coin].dropna()
                        if len(sub) > 2:
                            z = np.polyfit(sub["sentiment"], sub["return_1d"] * 100, 1)
                            xs = np.linspace(sub["sentiment"].min(), sub["sentiment"].max(), 50)
                            fig_scatter.add_trace(go.Scatter(
                                x=xs, y=np.polyval(z, xs), mode="lines",
                                line=dict(color=palette_sent.get(coin, "#888"),
                                          width=1.5, dash="dot"),
                                name=f"{coin} trend", showlegend=False,
                            ))
                    fig_scatter.update_layout(
                        title="Sentiment vs Next-Day Return",
                        xaxis_title="Sentiment Score", yaxis_title="Return (%)",
                        template="plotly_white", height=400,
                    )
                    children.append(html.H5("Sentiment vs Returns",
                                            style={"color": "#6A1B9A", "marginTop": "16px"}))
                    children.append(html.P(
                        "Each point is one day. Dotted lines are linear trendlines.",
                        style={"color": "#888", "fontSize": "12px"}))
                    children.append(dcc.Graph(figure=fig_scatter))

                    # Pearson correlation table
                    corr_rows = []
                    for coin in svr["coin"].unique():
                        sub = svr[svr["coin"] == coin].dropna()
                        if len(sub) > 3:
                            r_val = float(np.corrcoef(sub["sentiment"], sub["return_1d"])[0, 1])
                            corr_rows.append({"Coin": coin, "Pearson r": f"{r_val:+.3f}",
                                              "N": len(sub)})
                    if corr_rows:
                        children.append(html.H5("Sentiment–Return Correlation",
                                                style={"color": "#6A1B9A", "marginTop": "16px"}))
                        children.append(dash_table.DataTable(
                            columns=[{"name": k, "id": k} for k in corr_rows[0]],
                            data=corr_rows,
                            style_header={"backgroundColor": "#6A1B9A", "color": "white",
                                          "fontWeight": "bold", "fontSize": "12px"},
                            style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
                            style_table={"width": "360px", "marginBottom": "18px"},
                        ))
            except Exception:
                pass

        # ---- Latest headlines ----
        all_headlines = []
        for coin, s in sent.items():
            for h in s.get("headlines", []):
                all_headlines.append({"Coin": coin, **h})
        if all_headlines:
            hl_rows = [{
                "Coin": h["Coin"],
                "Score": f"{h.get('score', 0):+.3f}",
                "Headline": h.get("title", "")[:90],
                "Link": h.get("url", ""),
            } for h in all_headlines]
            children.append(html.H5("Latest Headlines (NewsAPI)",
                                    style={"color": "#6A1B9A", "marginTop": "16px"}))
            children.append(dash_table.DataTable(
                columns=[
                    {"name": "Coin",     "id": "Coin"},
                    {"name": "Score",    "id": "Score"},
                    {"name": "Headline", "id": "Headline"},
                    {"name": "Link",     "id": "Link", "presentation": "markdown"},
                ],
                data=[{**r, "Link": f"[open]({r['Link']})"
                        if r["Link"] else ""} for r in hl_rows],
                style_header={"backgroundColor": "#6A1B9A", "color": "white",
                              "fontWeight": "bold", "fontSize": "12px"},
                style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px",
                            "maxWidth": "500px", "overflow": "hidden",
                            "textOverflow": "ellipsis"},
                style_table={"marginBottom": "18px"},
                markdown_options={"html": True},
            ))

        # ====================================================================
        # POSITIONING LAYER — Sentiment Sensitivity Analysis
        # ====================================================================
        if cd:
            try:
                # Get risk index if available (for regime conditioning)
                risk_idx = None
                try:
                    sr_data = compute_systemic_index(cd)
                    risk_idx = sr_data.get("history")
                except Exception:
                    pass

                ss = sentiment_sensitivity(sent, cd, risk_index=risk_idx)

                children.append(html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}))
                children.append(html.H4("Positioning Layer — Sentiment Sensitivity",
                                        style={"color": "#4a148c", "marginTop": "24px"}))
                children.append(html.P(
                    "Not direction — sensitivity. We measure how price and volatility "
                    "respond to sentiment shocks, conditioned on risk regime. "
                    "A useful signal is not 'sentiment is positive', it's "
                    "'negative sentiment in a high-risk regime causes X× larger losses'.",
                    style={"color": "#888", "fontSize": "12px", "marginBottom": "16px"},
                ))

                # β: return per sentiment z-score
                beta = ss["price_per_sentiment"]
                if not np.isnan(beta):
                    children.append(html.Div([
                        html.Span("β (return per sentiment z-score): ",
                                  style={"fontWeight": "bold", "fontSize": "14px"}),
                        html.Span(f"{beta:+.3f}% per σ",
                                  style={"fontSize": "16px", "fontWeight": "bold",
                                         "color": "#388E3C" if beta > 0 else "#d32f2f"}),
                    ], style={"padding": "8px 0", "marginBottom": "8px"}))

                # Regime impact deliverable
                if ss["regime_impact"]:
                    children.append(html.Div(
                        ss["regime_impact"],
                        style={
                            "backgroundColor": "#f3e5f5", "borderLeft": "4px solid #7B1FA2",
                            "padding": "12px 16px", "fontSize": "14px", "fontWeight": "bold",
                            "color": "#4a148c", "marginBottom": "16px", "borderRadius": "4px",
                        },
                    ))

                # Vol-response table by regime
                vr = ss["vol_response"]
                if not vr.empty:
                    children.append(html.H5("Volatility Response to Negative Sentiment Shocks",
                                            style={"color": "#6A1B9A", "marginTop": "16px"}))
                    children.append(dash_table.DataTable(
                        columns=[{"name": k, "id": k} for k in vr.columns],
                        data=vr.to_dict("records"),
                        style_header={"backgroundColor": "#6A1B9A", "color": "white",
                                      "fontWeight": "bold", "fontSize": "12px"},
                        style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
                        style_table={"maxWidth": "700px", "marginBottom": "18px"},
                    ))

                # ── KILLER CHART: Sentiment Shock vs Drawdown by Regime ──
                dd = ss["daily_data"]
                if not dd.empty and "regime" in dd.columns:
                    _regime_colors = {
                        "Low Risk": "#388E3C",
                        "Transition": "#F9A825",
                        "High Stress": "#d32f2f",
                    }
                    fig_killer = go.Figure()

                    for regime in ["Low Risk", "Transition", "High Stress"]:
                        sub = dd[dd["regime"] == regime]
                        if sub.empty:
                            continue
                        clr = _regime_colors.get(regime, "#888")
                        fig_killer.add_trace(go.Scatter(
                            x=sub["sentiment_z"],
                            y=sub["return_1d"] * 100,
                            mode="markers",
                            marker=dict(
                                color=clr, size=8, opacity=0.65,
                                line=dict(color="white", width=0.5),
                            ),
                            name=regime,
                            hovertemplate=(
                                f"{regime}<br>"
                                "Sentiment shock (z): %{x:.2f}<br>"
                                "Return: %{y:.2f}%<extra></extra>"
                            ),
                        ))
                        # Trendline per regime
                        if len(sub) > 3:
                            z_fit = np.polyfit(sub["sentiment_z"], sub["return_1d"] * 100, 1)
                            xs = np.linspace(sub["sentiment_z"].min(),
                                             sub["sentiment_z"].max(), 50)
                            fig_killer.add_trace(go.Scatter(
                                x=xs, y=np.polyval(z_fit, xs), mode="lines",
                                line=dict(color=clr, width=2.5, dash="dash"),
                                name=f"{regime} trend",
                                showlegend=False,
                            ))

                    fig_killer.add_vline(x=0, line_dash="dash",
                                        line_color="gray", line_width=1)
                    fig_killer.add_hline(y=0, line_dash="dash",
                                        line_color="gray", line_width=1)
                    fig_killer.add_vrect(
                        x0=-3, x1=-0.5,
                        fillcolor="rgba(244,67,54,0.05)", line_width=0,
                        annotation_text="Negative shocks",
                        annotation_position="top left",
                        annotation_font_size=10, annotation_font_color="#d32f2f",
                    )

                    fig_killer.update_layout(
                        title=(
                            "<b>Sentiment Shock vs Return — Conditioned on Risk Regime</b><br>"
                            "<sup>Same negative sentiment → different outcome depending on "
                            "systemic stress. Steeper red slope = regime amplifies sentiment.</sup>"
                        ),
                        xaxis_title="Sentiment Shock (z-score)",
                        yaxis_title="Next-Day Portfolio Return (%)",
                        template="plotly_white", height=520,
                        legend=dict(orientation="h", y=-0.15),
                    )

                    children.append(html.H5(
                        "★ Killer Chart — Sentiment Shock × Risk Regime → Return",
                        style={"color": "#4a148c", "marginTop": "24px"},
                    ))
                    children.append(html.P(
                        "Each dot is one trading day. X-axis = how extreme the sentiment "
                        "shock was (z-score); Y-axis = what the portfolio actually did. "
                        "Colors separate risk regimes. A steeper red slope means High Stress "
                        "regimes amplify the effect of bad sentiment — the non-obvious insight.",
                        style={"color": "#888", "fontSize": "12px", "marginBottom": "8px"},
                    ))
                    children.append(dcc.Graph(figure=fig_killer))

                elif not dd.empty:
                    # No regime data, show unconditional version
                    fig_killer_uc = go.Figure()
                    fig_killer_uc.add_trace(go.Scatter(
                        x=dd["sentiment_z"], y=dd["return_1d"] * 100,
                        mode="markers",
                        marker=dict(
                            color=dd["sentiment_z"],
                            colorscale=[[0, "#d32f2f"], [0.5, "#F9A825"], [1, "#388E3C"]],
                            size=7, opacity=0.7, showscale=True,
                            colorbar=dict(title="z-score", thickness=12),
                        ),
                        hovertemplate="z: %{x:.2f}<br>Return: %{y:.2f}%<extra></extra>",
                    ))
                    if len(dd) > 3:
                        z_fit = np.polyfit(dd["sentiment_z"], dd["return_1d"] * 100, 1)
                        xs = np.linspace(dd["sentiment_z"].min(), dd["sentiment_z"].max(), 50)
                        fig_killer_uc.add_trace(go.Scatter(
                            x=xs, y=np.polyval(z_fit, xs), mode="lines",
                            line=dict(color="#1a237e", width=2, dash="dash"),
                            name=f"β = {z_fit[0]:+.3f}% / σ",
                        ))
                    fig_killer_uc.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig_killer_uc.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_killer_uc.update_layout(
                        title="Sentiment Shock (z-score) vs Next-Day Return",
                        xaxis_title="Sentiment Shock (z-score)",
                        yaxis_title="Return (%)",
                        template="plotly_white", height=450,
                    )
                    children.append(html.H5(
                        "Sentiment Shock vs Return",
                        style={"color": "#6A1B9A", "marginTop": "24px"},
                    ))
                    children.append(dcc.Graph(figure=fig_killer_uc))

            except Exception:
                pass  # Graceful fallback — sentiment basics still render

        return html.Div(children)

    # -- Systemic Risk Index --
    if tab == "tab-systemic":
        if "coin_data" not in _cache:
            return _placeholder("Fetch data first (▶ Run Full Analysis).")
        cd = _cache["coin_data"]

        try:
            sr = compute_systemic_index(cd)
        except Exception:
            return _placeholder(
                f"Systemic Risk error:\n{traceback.format_exc()[:400]}"
            )

        score = sr["score"]
        label = sr["label"]
        color = sr["color"]
        comps = sr["components"]
        hist  = sr["history"]

        _section = {"marginTop": "28px", "color": "#1a237e"}
        _hint    = {"color": "#666", "fontSize": "12px", "marginBottom": "8px"}

        tbl_hdr  = {"backgroundColor": "#1a237e", "color": "white",
                    "fontWeight": "bold", "fontSize": "12px"}
        tbl_cell = {"textAlign": "center", "padding": "7px 10px",
                    "fontSize": "12px"}

        def _tbl(rows, width=None):
            s = {"marginBottom": "18px"}
            if width:
                s["maxWidth"] = width
            return dash_table.DataTable(
                columns=[{"name": k, "id": k} for k in rows[0]
                         if k != "color"],
                data=[{k: v for k, v in r.items() if k != "color"}
                      for r in rows],
                style_header=tbl_hdr,
                style_cell=tbl_cell,
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#f5f7ff"},
                ],
                style_table=s,
            )

        # ── 1. GAUGE ──────────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 48, "color": color}, "suffix": " / 100"},
            title={"text": f"<b>Crypto Systemic Risk</b><br>"
                           f"<span style='font-size:18px;color:{color}'>"
                           f"{label}</span>",
                   "font": {"size": 20}},
            gauge={
                "axis":  {"range": [0, 100], "nticks": 11},
                "bar":   {"color": color, "thickness": 0.30},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#ccc",
                "steps": [
                    {"range": [  0,  25], "color": "rgba(56,142,60,0.15)"},
                    {"range": [ 25,  50], "color": "rgba(249,168,37,0.15)"},
                    {"range": [ 50,  75], "color": "rgba(230,74,25,0.15)"},
                    {"range": [ 75, 100], "color": "rgba(183,28,28,0.15)"},
                ],
                "threshold": {"line": {"color": color, "width": 4},
                              "thickness": 0.85, "value": score},
            },
        ))
        fig_gauge.update_layout(
            height=320, margin=dict(t=40, b=20, l=40, r=40),
            paper_bgcolor="white",
        )

        # ── 2. HISTORY WITH STRESS EVENT ANNOTATIONS ──────────────────────
        fig_hist_line = go.Figure()
        if not hist.empty:
            for lo_z, hi_z, fill_lbl, fill_clr in [
                (0,  30,  "Low",        "rgba(56,142,60,0.07)"),
                (30, 60,  "Transition", "rgba(249,168,37,0.07)"),
                (60, 100, "High Stress","rgba(183,28,28,0.07)"),
            ]:
                fig_hist_line.add_hrect(
                    y0=lo_z, y1=hi_z, fillcolor=fill_clr, line_width=0,
                    annotation_text=fill_lbl,
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="#888",
                )

            fig_hist_line.add_trace(go.Scatter(
                x=hist.index, y=hist.values,
                mode="lines",
                line=dict(color="#1a237e", width=2.5),
                name="Systemic Risk Index",
                fill="tozeroy",
                fillcolor="rgba(26,35,126,0.06)",
                hovertemplate="%{x|%Y-%m-%d}  Score: %{y:.1f}<extra></extra>",
            ))
            fig_hist_line.add_trace(go.Scatter(
                x=[hist.index[-1]], y=[hist.values[-1]],
                mode="markers",
                marker=dict(color=color, size=12,
                            line=dict(color="white", width=2)),
                name="Current",
                showlegend=False,
                hovertemplate=f"Today: {score:.1f}<extra></extra>",
            ))

            # Known stress event vertical lines
            for evt_name, evt_date in STRESS_EVENTS.items():
                ed = pd.to_datetime(evt_date)
                if hist.index.min() <= ed <= hist.index.max():
                    fig_hist_line.add_vline(
                        x=ed.timestamp() * 1000,
                        line_dash="dot",
                        line_color="rgba(180,0,0,0.55)",
                        line_width=1.5,
                        annotation_text=evt_name,
                        annotation_textangle=-90,
                        annotation_font_size=9,
                        annotation_font_color="rgba(180,0,0,0.8)",
                        annotation_position="top right",
                    )

        fig_hist_line.update_layout(
            title="Score History with Known Stress Events (Sanity Check)",
            xaxis_title="Date",
            yaxis=dict(title="Score (0–100)", range=[0, 100]),
            template="plotly_white", height=420,
            showlegend=False,
            margin=dict(t=48, b=24, r=80),
        )

        # ── 3. STACKED AREA — DECOMPOSITION OVER TIME ─────────────────────
        comp_hist_df = pd.DataFrame()
        try:
            comp_hist_df = component_history(
                cd, sr["btc_dominance"], sr["pca_pc1"]
            )
        except Exception:
            pass

        _comp_colors = {
            "Avg Correlation":   "#9C27B0",
            "Volatility Stress": "#f44336",
            "PCA Concentration": "#FF9800",
            "BTC Dominance":     "#2196F3",
            "Volume Anomaly":    "#4CAF50",
        }
        fig_decomp = go.Figure()
        if not comp_hist_df.empty:
            comp_cols = [c for c in _comp_colors if c in comp_hist_df.columns]
            for col in comp_cols:
                fig_decomp.add_trace(go.Scatter(
                    x=comp_hist_df.index,
                    y=comp_hist_df[col],
                    mode="lines",
                    name=col,
                    stackgroup="one",
                    fillcolor=f"rgba({','.join(str(int(c,16)) for c in [_comp_colors[col][1:3], _comp_colors[col][3:5], _comp_colors[col][5:7]])},0.60)",
                    line=dict(color=_comp_colors[col], width=0.5),
                    hovertemplate=f"{col}: %{{y:.2f}} pts<extra></extra>",
                ))
        else:
            fig_decomp.add_annotation(
                text="Run analysis first to see decomposition",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#aaa", size=14),
            )
        fig_decomp.update_layout(
            title="Signal Decomposition — What Is Driving the Score?",
            xaxis_title="Date",
            yaxis=dict(title="Points contributed", range=[0, 100]),
            template="plotly_white", height=400,
            legend=dict(orientation="h", y=-0.22),
            margin=dict(t=48, b=80),
        )

        # ── 4. REGIME-CONDITIONED STATS ────────────────────────────────────
        regime_df = pd.DataFrame()
        try:
            regime_df = regime_conditioned_stats(hist, cd)
        except Exception:
            pass

        # Colour each row to match its regime
        reg_cond_style = []
        if not regime_df.empty:
            for i, row in regime_df.iterrows():
                clr = row.get("color", "#333")
                reg_cond_style.append({
                    "if": {"row_index": i},
                    "color": clr,
                    "fontWeight": "bold",
                })

        # ── 5. COMPONENT BREAKDOWN BAR ────────────────────────────────────
        comp_names   = list(comps.keys())
        comp_scores  = [comps[c]["score"]  for c in comp_names]
        comp_weights = [comps[c]["weight"] for c in comp_names]
        bar_colors = []
        for cs, cw in zip(comp_scores, comp_weights):
            frac = cs / cw if cw > 0 else 0
            if frac < 0.33:   bar_colors.append("#388E3C")
            elif frac < 0.66: bar_colors.append("#F9A825")
            elif frac < 0.85: bar_colors.append("#E64A19")
            else:             bar_colors.append("#b71c1c")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=comp_scores, y=comp_names, orientation="h",
            marker=dict(color=bar_colors,
                        line=dict(color="white", width=1)),
            text=[f"{s:.1f} / {w}" for s, w in zip(comp_scores, comp_weights)],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f} pts<extra></extra>",
        ))
        fig_bar.add_trace(go.Bar(
            x=comp_weights, y=comp_names, orientation="h",
            marker=dict(color="rgba(200,200,200,0.25)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig_bar.update_layout(
            title="Component Scores Today  (bar = actual, grey = max capacity)",
            xaxis=dict(title="Points", range=[0, 28]),
            barmode="overlay", template="plotly_white", height=320,
            margin=dict(t=48, b=24, l=160, r=80),
        )

        # ── 6. LEADING INDICATOR ─────────────────────────────────────────
        lead_df = pd.DataFrame()
        fig_lead = go.Figure()
        try:
            lead_df = leading_indicator_series(hist, cd, lead_days=14)
        except Exception:
            pass

        if not lead_df.empty:
            fig_lead.add_trace(go.Scatter(
                x=lead_df["score"],
                y=lead_df["fwd_vol"] * 100,
                mode="markers",
                marker=dict(
                    color=lead_df["score"],
                    colorscale=[[0, "#388E3C"],
                                [0.3, "#F9A825"],
                                [0.6, "#E64A19"],
                                [1.0, "#b71c1c"]],
                    size=5, opacity=0.65,
                    showscale=True,
                    colorbar=dict(title="Score", thickness=12, len=0.8),
                ),
                hovertemplate="Score: %{x:.1f}<br>14d fwd vol: %{y:.1f}%<extra></extra>",
                name="(score today, vol in 14d)",
            ))
            # Trend line
            m, b = np.polyfit(lead_df["score"], lead_df["fwd_vol"] * 100, 1)
            xs = np.linspace(lead_df["score"].min(), lead_df["score"].max(), 80)
            fig_lead.add_trace(go.Scatter(
                x=xs, y=m * xs + b, mode="lines",
                line=dict(color="#1a237e", width=2, dash="dash"),
                name=f"Trend  (slope {m:+.3f}% vol / pt)",
            ))
            r2 = np.corrcoef(lead_df["score"], lead_df["fwd_vol"])[0, 1] ** 2
            fig_lead.update_layout(
                title=(f"Leading Indicator — Does today's score predict "
                       f"14-day forward volatility?  R² = {r2:.3f}"),
                xaxis_title="Systemic Risk Score (today)",
                yaxis_title="Realised Portfolio Vol 14d forward (%)",
                template="plotly_white", height=420,
                legend=dict(orientation="h", y=-0.18),
            )
        else:
            fig_lead.add_annotation(
                text="Not enough history — run analysis after data is loaded",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#aaa", size=13),
            )
            fig_lead.update_layout(
                title="Leading Indicator (14-day forward vol)",
                template="plotly_white", height=380,
            )

        # ── SIGNAL DETAIL TABLE ───────────────────────────────────────────
        detail_rows = []
        for cname, cv in comps.items():
            meta = COMPONENT_META.get(cname, {})
            detail_rows.append({
                "Signal":      cname,
                "Current":     f"{cv['raw']} {cv['unit']}",
                "Percentile":  f"{cv['pct']} %",
                "Score":       f"{cv['score']:.2f}",
                "Max":         str(cv["weight"]),
                "Description": meta.get("description", ""),
            })

        # ── CORRELATION + VOL DRIVER ──────────────────────────────────────
        fig_sup = go.Figure()
        try:
            from risk_engine import rolling_correlation as _rc_fn, rolling_volatility as _rv_fn
            avg_corr = _rc_fn(cd, 30).mean(axis=1).rename("Avg Corr (30d)")
            rvol_df  = _rv_fn(cd, (30,))
            fig_sup.add_trace(go.Scatter(
                x=avg_corr.index, y=avg_corr.values,
                mode="lines", name="Avg Pairwise Corr",
                line=dict(color="#9C27B0", width=2), yaxis="y1",
            ))
            for col in rvol_df.columns:
                if "30" in col:
                    fig_sup.add_trace(go.Scatter(
                        x=rvol_df.index, y=rvol_df[col],
                        mode="lines", name=col,
                        line=dict(color="#FF9800", width=1.5, dash="dot"),
                        yaxis="y2",
                    ))
                    break
            fig_sup.update_layout(
                title="Correlation + Volatility Drivers",
                yaxis=dict(title="Avg Correlation", side="left", range=[0, 1]),
                yaxis2=dict(title="Ann. Vol", overlaying="y", side="right",
                            tickformat=".0%"),
                template="plotly_white", height=360,
                legend=dict(orientation="h", y=-0.18),
            )
        except Exception:
            fig_sup.update_layout(title="Correlation + Volatility: no data",
                                  height=300)

        # ── 8. LEAD-LAG CROSS-CORRELATION ─────────────────────────────────
        lag_data = {}
        fig_lag  = go.Figure()
        try:
            lag_data = lead_lag_correlations(hist, cd, max_lag=15)
        except Exception:
            pass

        if lag_data:
            _lag_meta = [
                ("vol",         "Realised 5d Vol",     "#E64A19"),
                ("drawdown",    "10d Drawdown",         "#b71c1c"),
                ("correlation", "Avg Pairwise Corr",    "#9C27B0"),
            ]
            for key, lname, lcol in _lag_meta:
                df_lag = lag_data.get(key)
                if df_lag is not None and not df_lag.empty:
                    fig_lag.add_trace(go.Scatter(
                        x=df_lag["lag"], y=df_lag["corr"],
                        mode="lines+markers",
                        marker=dict(size=5),
                        line=dict(color=lcol, width=2),
                        name=lname,
                        hovertemplate="Lag %{x}d: corr=%{y:.3f}<extra>" + lname + "</extra>",
                    ))
            fig_lag.add_vline(x=0, line_dash="dash", line_color="#aaa", line_width=1)
            fig_lag.add_annotation(
                x=0, y=1.04, xref="x", yref="paper",
                text="lag = 0", showarrow=False,
                font=dict(size=10, color="#aaa"),
            )
            fig_lag.update_layout(
                title="Lead-Lag: Does risk score LEAD market stress?  "
                      "(peak at positive lag ⟹ score leads)",
                xaxis_title="Lag (days)     ← target leads · score leads →",
                yaxis_title="Pearson Correlation",
                yaxis=dict(range=[-1, 1]),
                template="plotly_white", height=420,
                legend=dict(orientation="h", y=-0.18),
            )
        else:
            fig_lag.add_annotation(
                text="Not enough history for lead-lag analysis",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#aaa", size=13),
            )
            fig_lag.update_layout(height=280)

        # ── 9. RISK-RETURN ASYMMETRY CHART ────────────────────────────────
        fig_asym = go.Figure()
        try:
            _coins_a = list(cd.keys())
            _rets_a  = pd.DataFrame({s: cd[s]["returns"] for s in _coins_a}).dropna()
            _port_a  = _rets_a.mean(axis=1)
            _comm_a  = hist.index.intersection(_port_a.index)
            if len(_comm_a) > 20:
                _sc_a  = hist.reindex(_comm_a)
                _rt_a  = _port_a.reindex(_comm_a)
                _cum_a = (1 + _rt_a).cumprod()
                _r_labels, _r_colors = [], []
                _avg_rets, _ann_vols, _max_dds = [], [], []
                for lo, hi, label, clr in REGIMES:
                    mask = (_sc_a >= lo) & (_sc_a < hi) if hi < 100 else _sc_a >= lo
                    _r_labels.append(label)
                    _r_colors.append(clr)
                    if mask.sum() == 0:
                        _avg_rets.append(0.0)
                        _ann_vols.append(0.0)
                        _max_dds.append(0.0)
                        continue
                    r_in = _rt_a[mask]
                    _avg_rets.append(float(r_in.mean()) * 365 * 100)
                    _ann_vols.append(float(r_in.std() * np.sqrt(365)) * 100)
                    _dd_vals = []
                    for d in _comm_a[mask]:
                        loc = _comm_a.get_loc(d)
                        w_end = min(loc + 30, len(_cum_a) - 1)
                        _w = _cum_a.iloc[loc: w_end + 1]
                        _pk = _w.cummax()
                        _dd_vals.append(float(((_w - _pk) / _pk).min()))
                    _max_dds.append(float(min(_dd_vals)) * 100 if _dd_vals else 0.0)

                fig_asym.add_trace(go.Bar(
                    x=_r_labels, y=_avg_rets,
                    name="Annualised Return (%)",
                    marker_color=_r_colors, opacity=0.85,
                    text=[f"{v:+.1f}%" for v in _avg_rets],
                    textposition="outside",
                ))
                fig_asym.add_trace(go.Bar(
                    x=_r_labels, y=_max_dds,
                    name="Max 30d Drawdown (%) — more negative = worse",
                    marker_color="rgba(0,0,0,0)",
                    marker_line=dict(color=_r_colors, width=2),
                    text=[f"{v:.1f}%" for v in _max_dds],
                    textposition="outside",
                ))
                fig_asym.update_layout(
                    title="Risk-Return Asymmetry by Regime  "
                          "(High Stress → negative return + large DD ⟹ index has edge)",
                    xaxis_title="Risk Regime",
                    yaxis_title="% (annualised return / drawdown)",
                    barmode="group", template="plotly_white", height=400,
                    legend=dict(orientation="h", y=-0.18),
                )
        except Exception:
            fig_asym.add_annotation(
                text="Could not compute asymmetry chart",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#aaa", size=13),
            )
            fig_asym.update_layout(height=280)

        # ── 10. STRATEGY BACKTEST ─────────────────────────────────────────
        _bt       = {}
        fig_bt    = go.Figure()
        _bt_stats = []
        try:
            _bt = regime_strategy_backtest(hist, cd)
        except Exception:
            pass

        if _bt and "equity" in _bt:
            _eq = _bt["equity"]
            _st = _bt["stats"]
            fig_bt.add_trace(go.Scatter(
                x=_eq.index, y=_eq["strategy"],
                mode="lines", name="Risk-Managed Strategy",
                line=dict(color="#1a237e", width=2.5),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}×<extra>Strategy</extra>",
            ))
            fig_bt.add_trace(go.Scatter(
                x=_eq.index, y=_eq["buyhold"],
                mode="lines", name="Buy & Hold",
                line=dict(color="#888", width=1.5, dash="dot"),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}×<extra>Buy & Hold</extra>",
            ))
            fig_bt.add_trace(go.Scatter(
                x=_eq.index, y=_eq["weight"],
                mode="lines", name="Exposure Weight",
                line=dict(color="#F9A825", width=1, dash="dash"),
                yaxis="y2", opacity=0.55,
                hovertemplate="%{x|%Y-%m-%d}: exp=%{y:.1f}<extra>Exposure</extra>",
            ))
            fig_bt.update_layout(
                title=(f"Rule Backtest  —  "
                       f"Strategy CAGR {_st['CAGR_strat']:+.1%}  |  "
                       f"Buy&Hold CAGR {_st['CAGR_bh']:+.1%}  |  "
                       f"Sharpe {_st['Sharpe_strat']:.2f} vs {_st['Sharpe_bh']:.2f}  |  "
                       f"Invested {_st['pct_invested']:.0%} of time"),
                xaxis_title="Date",
                yaxis=dict(title="Portfolio Growth (×1)", side="left"),
                yaxis2=dict(title="Exposure", overlaying="y", side="right",
                            range=[0, 1.6], tickformat=".1f"),
                template="plotly_white", height=440,
                legend=dict(orientation="h", y=-0.18),
            )
            _bt_stats = [
                {"Metric": "CAGR",           "Strategy": f"{_st['CAGR_strat']:+.2%}",  "Buy & Hold": f"{_st['CAGR_bh']:+.2%}"},
                {"Metric": "Max Drawdown",    "Strategy": f"{_st['MaxDD_strat']:.2%}",  "Buy & Hold": f"{_st['MaxDD_bh']:.2%}"},
                {"Metric": "Sharpe Ratio",    "Strategy": f"{_st['Sharpe_strat']:.3f}", "Buy & Hold": f"{_st['Sharpe_bh']:.3f}"},
                {"Metric": "Avg Exposure",    "Strategy": f"{_st['pct_invested']:.0%}", "Buy & Hold": "100%"},
                {"Metric": "Days Backtested", "Strategy": str(_st['n_days']),            "Buy & Hold": str(_st['n_days'])},
            ]
        else:
            fig_bt.add_annotation(
                text="Not enough history for backtest",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#aaa", size=13),
            )
            fig_bt.update_layout(height=280)

        # ── ZONE LEGEND CARDS ─────────────────────────────────────────────
        zone_cards = []
        for lo_z, hi_z, zlabel, zclr in ZONES:
            is_cur = (lo_z <= score < hi_z) or (hi_z == 100 and score >= 75)
            zone_cards.append(html.Div([
                html.Div(zlabel, style={"fontWeight": "bold",
                                        "color": zclr, "fontSize": "15px"}),
                html.Div(f"{lo_z}–{hi_z}",
                         style={"fontSize": "12px", "color": "#999"}),
            ], style={
                "border": f"{'3px' if is_cur else '1px'} solid {zclr}",
                "borderRadius": "8px", "padding": "10px 18px",
                "minWidth": "100px", "textAlign": "center",
                "backgroundColor": "#fafafa",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.08)" if is_cur else "none",
            }))

        # ── ASSEMBLE ──────────────────────────────────────────────────────
        return html.Div([
            html.H4("Crypto Systemic Risk Index",
                    style={"color": "#1a237e", "marginBottom": "4px"}),
            html.P(
                "Composite 0–100 score across correlation · vol · PCA · "
                "BTC dominance · volume anomaly.  "
                "Stress events annotated for sanity validation.",
                style={"color": "#888", "fontSize": "12px",
                       "marginBottom": "16px"},
            ),
            html.Div(zone_cards,
                     style={"display": "flex", "gap": "12px",
                             "flexWrap": "wrap", "marginBottom": "20px"}),
            html.Hr(style={"borderColor": "#e8eaf6"}),

            # Current score gauge
            html.H5("Current Score", style=_section),
            dcc.Graph(figure=fig_gauge),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # History + stress annotations (sanity check)
            html.H5("1 · Score History — Stress Event Validation", style=_section),
            html.P(
                "Score should spike around FTX (Nov '22), Luna (May '22), "
                "and 3AC/Celsius (Jun '22). Flat during calm mid-2023 periods.",
                style={**_hint, "color": "#b71c1c", "fontWeight": "bold"},
            ),
            dcc.Graph(figure=fig_hist_line),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # Decomposition over time
            html.H5("2 · Signal Decomposition Over Time", style=_section),
            html.P(
                "Stacked contributions per signal.  "
                "Shows WHY the score moved — not just that it did.",
                style=_hint,
            ),
            dcc.Graph(figure=fig_decomp),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # Regime-conditioned stats (the non-obvious insight)
            html.H5("3 · Regime-Conditioned Portfolio Behaviour", style=_section),
            html.P(
                "Avg daily return, annualised vol, max 30-day drawdown, "
                "and average 5-day forward return — split by risk regime.  "
                "A high-stress regime should show negative fwd returns and "
                "larger drawdowns.",
                style=_hint,
            ),
            (dash_table.DataTable(
                columns=[{"name": k, "id": k}
                         for k in ["Regime", "Days", "Avg Daily Ret",
                                   "Ann. Vol", "Max Drawdown", "Fwd 5d Ret"]],
                data=regime_df[["Regime", "Days", "Avg Daily Ret",
                                "Ann. Vol", "Max Drawdown",
                                "Fwd 5d Ret"]].to_dict("records")
                if not regime_df.empty else [],
                style_header=tbl_hdr,
                style_cell={**tbl_cell, "textAlign": "center"},
                style_data_conditional=reg_cond_style,
                style_table={"marginBottom": "18px", "maxWidth": "800px"},
            ) if not regime_df.empty else html.P("No regime history yet.",
                                                  style=_hint)),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # Component bar
            html.H5("4 · Component Breakdown (Today)", style=_section),
            dcc.Graph(figure=fig_bar),

            # Signal detail table
            html.H5("5 · Signal Detail", style={**_section, "marginTop": "24px"}),
            dash_table.DataTable(
                columns=[{"name": k, "id": k} for k in detail_rows[0]],
                data=detail_rows,
                style_header=tbl_hdr,
                style_cell={**tbl_cell, "textAlign": "left"},
                style_cell_conditional=[
                    {"if": {"column_id": "Description"}, "maxWidth": "360px"},
                    {"if": {"column_id": "Score"}, "fontWeight": "bold"},
                ],
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f5f7ff"},
                ],
                style_table={"marginBottom": "18px"},
            ),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # Leading indicator (non-obvious finding)
            html.H5("6 · Leading Indicator — Score vs 14-Day Forward Volatility",
                    style=_section),
            html.P(
                "Each point = one trading day.  X-axis = today's risk score.  "
                "Y-axis = realised portfolio vol over the next 14 days.  "
                "A positive slope means the index leads volatility — "
                "the non-obvious employer-grade insight.",
                style=_hint,
            ),
            dcc.Graph(figure=fig_lead),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # Correlation + vol driver
            html.H5("7 · Correlation & Volatility Drivers", style=_section),
            html.P(
                "Correlation (left) and 30-day ann. vol (right).  "
                "Rising together = systemic risk building.",
                style=_hint,
            ),
            dcc.Graph(figure=fig_sup),

            # Lead-lag test
            html.H5("8 · Lead-Lag Test — Does the Score LEAD Market Stress?",
                    style=_section),
            html.P(
                "Cross-correlation between risk score and three stress targets at lags "
                "−15 … +15 days.  "
                "A peak at positive lag → risk score rises BEFORE the stress indicator — "
                "that is genuine predictive power, not noise.",
                style={**_hint, "color": "#4a148c", "fontWeight": "bold"},
            ),
            dcc.Graph(figure=fig_lag),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # Asymmetry chart
            html.H5("9 · Risk-Return Asymmetry by Regime", style=_section),
            html.P(
                "Annualised return (bars) and worst 30-day drawdown (outlined) per regime.  "
                "High Stress → negative return + deep drawdown = real edge.",
                style={**_hint, "color": "#b71c1c", "fontWeight": "bold"},
            ),
            dcc.Graph(figure=fig_asym),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            # Strategy backtest
            html.H5("10 · Backtest: Risk-Managed Rule vs Buy & Hold", style=_section),
            html.P(
                "Rule (no look-ahead, 1-day lag): score < 30 → full · 30–60 → half · ≥ 60 → cash.  "
                "Better Sharpe or lower max-drawdown at comparable CAGR → "
                "the index earns its keep. Then you go ML.",
                style={**_hint, "color": "#1a237e", "fontWeight": "bold"},
            ),
            dcc.Graph(figure=fig_bt),
            (dash_table.DataTable(
                columns=[{"name": k, "id": k}
                         for k in ["Metric", "Strategy", "Buy & Hold"]],
                data=_bt_stats,
                style_header=tbl_hdr,
                style_cell={**tbl_cell, "textAlign": "center"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#f5f7ff"},
                    {"if": {"column_id": "Strategy"},
                     "fontWeight": "bold", "color": "#1a237e"},
                ],
                style_table={"marginBottom": "18px", "maxWidth": "580px"},
            ) if _bt_stats else html.Div()),
            html.Hr(style={"borderColor": "#e8eaf6", "marginTop": "24px"}),

            html.P(
                f"BTC dominance: {sr['btc_dominance']:.1%}  ·  "
                f"Portfolio vol (30d): {sr['current_vol']:.1%}  ·  "
                f"Avg corr (30d): {sr['current_corr']:.3f}  ·  "
                f"Volume ratio: {sr['current_vr']:.2f}×  ·  "
                f"PCA PC1: {sr['pca_pc1']:.1%}",
                style={"color": "#aaa", "fontSize": "11px",
                       "marginTop": "24px", "marginBottom": "40px"},
            ),
        ], style={"padding": "0 8px 40px 8px"})

    return _placeholder("Select a tab.")


# ---- 6. Show / hide extra coin holdings based on coin-set selection ----------

@app.callback(
    [Output("holdings-top5",  "style"),
     Output("holdings-top10", "style")],
    Input("in-coin-set", "value"),
)
def cb_toggle_holdings(coin_set):
    show = {"display": "block"}
    hide = {"display": "none"}
    if coin_set == "top10":
        return show, show
    if coin_set == "top5":
        return show, hide
    return hide, hide


# ---- 7. Keep single-coin dropdown in sync with loaded coins -----------------

@app.callback(
    [Output("in-coin", "options"),
     Output("in-coin", "value")],
    Input("data-status", "children"),
)
def cb_update_coin_options(_status):
    """Refresh the single-coin sim dropdown whenever new data is fetched."""
    if "coin_data" in _cache:
        coins = list(_cache["coin_data"].keys())
        opts  = [{"label": c, "value": c} for c in coins]
        val   = coins[1] if len(coins) > 1 else coins[0]
        return opts, val
    # Fallback before any data is loaded
    default = ["BTC", "ETH", "SOL"]
    return [{"label": c, "value": c} for c in default], "ETH"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n  DeFi Risk Analysis — Dash App")
    print("  Open http://127.0.0.1:8050 in your browser\n")
    app.run(debug=True, port=8050)
