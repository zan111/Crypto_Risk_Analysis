"""
Crypto Systemic Risk Index
==========================

Composite risk score (0–100) that measures how stressed the crypto
market is *right now*, built from five independent signals:

    Signal                    Max score   Source
    ─────────────────────────────────────────────────────────────
    1  Avg pairwise correlation    25     rolling_correlation()
    2  PCA first-PC concentration  25     pca_risk()
    3  Portfolio volatility level  25     rolling returns
    4  BTC dominance               15     CoinGecko /global
    5  Volume anomaly ratio        10     CoinGecko price chart
    ─────────────────────────────────────────────────────────────
    Total                         100

Risk zones
    0–25   Low       (green)   — normal market conditions
    25–50  Elevated  (amber)   — correlations rising, watch closely
    50–75  High      (orange)  — diversification breaking down
    75–100 Extreme   (red)     — systemic event in progress

All CoinGecko "global" calls are cached for 1 hour in
``sysrisk_cache.pkl`` so the app stays responsive between refreshes.
"""

from __future__ import annotations

import pathlib
import pickle
import time

import numpy as np
import pandas as pd
from pycoingecko import CoinGeckoAPI

from risk_engine import pca_risk, rolling_correlation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ANNUAL      = 365
_CACHE_FILE  = pathlib.Path("sysrisk_cache.pkl")
_CACHE_TTL   = 3_600      # seconds — 1 hour for CoinGecko global data

# Risk zones ─────────────────────────────────────────────────────────────────
ZONES = [
    (0,  25,  "Low",      "#388E3C"),
    (25, 50,  "Elevated", "#F9A825"),
    (50, 75,  "High",     "#E64A19"),
    (75, 100, "Extreme",  "#b71c1c"),
]

# 3-zone regime classification (used for regime-conditioned stats)
REGIMES = [
    (0,  30, "Low Risk",    "#388E3C"),
    (30, 60, "Transition",  "#F9A825"),
    (60, 100,"High Stress", "#b71c1c"),
]

# Known crypto stress events for sanity-check annotations
STRESS_EVENTS = {
    "Luna Collapse":  "2022-05-09",
    "3AC / Celsius":  "2022-06-12",
    "FTX Collapse":   "2022-11-08",
    "BTC -30% (Jan)": "2022-01-21",
    "USDC Depeg":     "2023-03-10",
    "Aug '23 dump":   "2023-08-17",
}

COMPONENT_META = {
    "Avg Correlation":   {"weight": 25, "description": "Rolling 30-day average pairwise correlation"},
    "PCA Concentration": {"weight": 25, "description": "% variance explained by first principal component"},
    "Volatility Stress": {"weight": 25, "description": "Current portfolio vol vs 1-year percentile"},
    "BTC Dominance":     {"weight": 15, "description": "BTC market-cap share (flight-to-safety signal)"},
    "Volume Anomaly":    {"weight": 10, "description": "Current volume vs 30-day average (leverage proxy)"},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_cache() -> dict | None:
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as fh:
                obj = pickle.load(fh)
            if time.time() - obj["ts"] < _CACHE_TTL:
                return obj["data"]
        except Exception:
            pass
    return None


def _save_cache(data: dict) -> None:
    try:
        with open(_CACHE_FILE, "wb") as fh:
            pickle.dump({"ts": time.time(), "data": data}, fh)
    except Exception:
        pass


def _fetch_btc_dominance() -> float:
    """
    Return BTC market-cap dominance as a fraction [0, 1].
    Falls back to 0.50 on any network error.
    """
    try:
        cg     = CoinGeckoAPI()
        g      = cg.get_global()
        return g["market_cap_percentage"].get("btc", 50.0) / 100.0
    except Exception:
        return 0.50


def _pct_rank(series: pd.Series, value: float) -> float:
    """Empirical percentile of *value* in *series* ∈ [0, 1]."""
    if series.empty:
        return 0.5
    return float((series <= value).mean())


def _normalise(value: float, lo: float, hi: float) -> float:
    """Clip-normalise *value* to [0, 1] given expected range [lo, hi]."""
    return float(np.clip((value - lo) / (hi - lo + 1e-10), 0.0, 1.0))


def _score_label(score: float) -> tuple[str, str]:
    for lo, hi, label, color in ZONES:
        if score < hi or hi == 100:
            return label, color
    return "Extreme", "#b71c1c"


# ---------------------------------------------------------------------------
# Fast vectorised rolling index for history chart
# ---------------------------------------------------------------------------

def _rolling_index_series(
    avg_corr:   pd.Series,
    port_vol:   pd.Series,
    vol_ratio:  pd.Series,
    pc1_val:    float,
    btc_dom:    float,
) -> pd.Series:
    """
    Reconstruct the Systemic Risk Index for every day in the data.

    Components 2 (PCA) and 4 (BTC dominance) are expensive to recalculate
    per day, so we hold them fixed at their current values — acceptable for
    a monitoring chart.  Components 1, 3, 5 are walk-forward percentile
    scored so the history reflects the information available at each date.
    """
    dates = (avg_corr.dropna().index
             .intersection(port_vol.dropna().index)
             .intersection(vol_ratio.dropna().index))
    if dates.empty:
        return pd.Series(dtype=float, name="systemic_risk_index")

    ac  = avg_corr.reindex(dates)
    pv  = port_vol.reindex(dates)
    vr  = vol_ratio.reindex(dates)

    c2_fixed  = _normalise(pc1_val, 0.40, 0.95) * 25
    c4_fixed  = _normalise(btc_dom, 0.40, 0.65) * 15

    # Vectorised rolling percentile rank — eliminates the O(T²) Python loop.
    # For each position i we need:  (arr[:i+1] <= arr[i]).mean()
    # Build a lower-triangular comparison matrix (n×n) once, then reduce.
    n        = len(dates)
    ac_arr   = ac.values
    pv_arr   = pv.values
    vr_arr   = vr.values
    counts   = np.arange(1, n + 1, dtype=float)    # [1, 2, …, n]

    tril_mask = np.tril(np.ones((n, n), dtype=bool))  # shape (n, n)

    # ac_arr[np.newaxis, :] broadcasts to (n, n): row i, col j = ac_arr[j]
    ac_le     = (ac_arr[np.newaxis, :] <= ac_arr[:, np.newaxis]) & tril_mask
    c1_scores = (ac_le.sum(axis=1) / counts) * 25

    pv_le     = (pv_arr[np.newaxis, :] <= pv_arr[:, np.newaxis]) & tril_mask
    c3_scores = (pv_le.sum(axis=1) / counts) * 25

    c5_scores = np.clip((vr_arr - 0.5) / (3.0 - 0.5 + 1e-10), 0.0, 1.0) * 10

    scores = c1_scores + c3_scores + c5_scores + c2_fixed + c4_fixed

    return pd.Series(scores, index=dates, name="systemic_risk_index")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_systemic_index(
    coin_data:     dict,
    force_refresh: bool = False,
) -> dict:
    """
    Compute the Crypto Systemic Risk Index.

    Parameters
    ----------
    coin_data     : {symbol: DataFrame}  from ``get_multi_coin_data()``
    force_refresh : bypass the 1-hour CoinGecko cache

    Returns
    -------
    dict
        score        float               0–100
        label        str                 "Low" / "Elevated" / "High" / "Extreme"
        color        str                 hex colour for the current zone
        components   dict                per-component breakdown
        history      pd.Series           rolling daily index (date → score)
        btc_dominance float
        current_vol  float               annualised portfolio volatility
        current_corr float               average 30-day pairwise correlation
        current_vr   float               volume anomaly ratio
    """
    # ── component 1: average pairwise correlation ───────────────────────────
    rc_df        = rolling_correlation(coin_data, window=30)
    avg_corr_ser = rc_df.mean(axis=1).dropna()
    current_corr = float(avg_corr_ser.iloc[-1]) if not avg_corr_ser.empty else 0.5
    corr_pct     = _pct_rank(avg_corr_ser, current_corr)
    c1_score     = corr_pct * 25

    # ── component 2: PCA first-PC concentration ─────────────────────────────
    pca      = pca_risk(coin_data)
    pc1_val  = float(pca["first_pc_explains"])
    c2_score = _normalise(pc1_val, 0.40, 0.95) * 25

    # ── component 3: portfolio volatility percentile ─────────────────────────
    rets       = pd.DataFrame(
        {s: df["returns"] for s, df in coin_data.items()}
    ).dropna()
    port_ret   = rets.mean(axis=1)
    vol_30_ser = (port_ret.rolling(30).std() * np.sqrt(_ANNUAL)).dropna()
    current_vol = float(vol_30_ser.iloc[-1]) if not vol_30_ser.empty else 0.5
    vol_pct     = _pct_rank(vol_30_ser, current_vol)
    c3_score    = vol_pct * 25

    # ── component 4: BTC dominance ───────────────────────────────────────────
    cached = None if force_refresh else _load_cache()
    btc_dom = cached.get("btc_dom", None) if cached else None
    if btc_dom is None:
        btc_dom = _fetch_btc_dominance()
        _save_cache({"btc_dom": btc_dom})
    btc_dom = float(btc_dom)
    # high dominance (>55 %) → investors flee alts → market stress
    c4_score = _normalise(btc_dom, 0.40, 0.65) * 15

    # ── component 5: volume anomaly (leverage / panic proxy) ─────────────────
    vols_raw    = pd.DataFrame(
        {s: df["volume"] for s, df in coin_data.items()}
    ).dropna()
    vol_ratio_ser = (
        (vols_raw / vols_raw.rolling(30).mean()).mean(axis=1).dropna()
    )
    current_vr    = float(vol_ratio_ser.iloc[-1]) if not vol_ratio_ser.empty else 1.0
    c5_score      = _normalise(current_vr, 0.5, 3.0) * 10

    # ── composite ────────────────────────────────────────────────────────────
    score          = c1_score + c2_score + c3_score + c4_score + c5_score
    label, color   = _score_label(score)

    # ── rolling history ───────────────────────────────────────────────────────
    history = _rolling_index_series(
        avg_corr_ser, vol_30_ser, vol_ratio_ser, pc1_val, btc_dom
    )

    return {
        "score":        round(score, 2),
        "label":        label,
        "color":        color,
        "components": {
            "Avg Correlation": {
                "score":   round(c1_score, 2),
                "weight":  25,
                "raw":     round(current_corr, 4),
                "pct":     round(corr_pct * 100, 1),
                "unit":    "",
            },
            "PCA Concentration": {
                "score":   round(c2_score, 2),
                "weight":  25,
                "raw":     round(pc1_val * 100, 1),
                "pct":     round(pc1_val * 100, 1),
                "unit":    "%",
            },
            "Volatility Stress": {
                "score":   round(c3_score, 2),
                "weight":  25,
                "raw":     round(current_vol * 100, 1),
                "pct":     round(vol_pct * 100, 1),
                "unit":    "% ann.",
            },
            "BTC Dominance": {
                "score":   round(c4_score, 2),
                "weight":  15,
                "raw":     round(btc_dom * 100, 1),
                "pct":     round(_normalise(btc_dom, 0.40, 0.65) * 100, 1),
                "unit":    "%",
            },
            "Volume Anomaly": {
                "score":   round(c5_score, 2),
                "weight":  10,
                "raw":     round(current_vr, 3),
                "pct":     round(_normalise(current_vr, 0.5, 3.0) * 100, 1),
                "unit":    "× avg",
            },
        },
        "history":       history,
        "btc_dominance": btc_dom,
        "current_vol":   current_vol,
        "current_corr":  current_corr,
        "current_vr":    current_vr,
        "pca_pc1":       pc1_val,
    }


# =============================================================================
# COMPONENT DECOMPOSITION  — per-day contribution of each signal
# =============================================================================

def component_history(coin_data: dict, btc_dom: float, pc1_val: float) -> pd.DataFrame:
    """
    Return a daily DataFrame with each component's *point contribution*
    to the Systemic Risk Index.

    Columns:
        Avg Correlation, PCA Concentration, Volatility Stress,
        BTC Dominance, Volume Anomaly, Total

    Components 2 (PCA) and 4 (BTC dominance) are held at their current
    values — same approximation used in _rolling_index_series.

    Parameters
    ----------
    coin_data : {symbol: DataFrame}
    btc_dom   : current BTC dominance fraction  (from compute_systemic_index)
    pc1_val   : current PCA first-PC fraction   (from compute_systemic_index)
    """
    from risk_engine import rolling_correlation as _rc

    rets = pd.DataFrame(
        {s: df["returns"] for s, df in coin_data.items()}
    ).dropna()
    port_ret = rets.mean(axis=1)
    vol_ser  = (port_ret.rolling(30).std() * np.sqrt(_ANNUAL)).dropna()

    rc_df        = _rc(coin_data, window=30)
    avg_corr_ser = rc_df.mean(axis=1).dropna()

    vols_raw = pd.DataFrame(
        {s: df["volume"] for s, df in coin_data.items()}
    ).dropna()
    vol_ratio_ser = (
        (vols_raw / vols_raw.rolling(30).mean()).mean(axis=1).dropna()
    )

    dates = (avg_corr_ser.index
             .intersection(vol_ser.index)
             .intersection(vol_ratio_ser.index))
    if dates.empty:
        return pd.DataFrame()

    ac = avg_corr_ser.reindex(dates)
    pv = vol_ser.reindex(dates)
    vr = vol_ratio_ser.reindex(dates)

    c2_fixed = _normalise(pc1_val, 0.40, 0.95) * 25
    c4_fixed = _normalise(btc_dom, 0.40, 0.65) * 15

    rows: list[dict] = []
    for i in range(len(dates)):
        c1 = _pct_rank(ac.iloc[: i + 1], ac.iloc[i]) * 25
        c3 = _pct_rank(pv.iloc[: i + 1], pv.iloc[i]) * 25
        c5 = _normalise(vr.iloc[i], 0.5, 3.0)         * 10
        rows.append({
            "date":              dates[i],
            "Avg Correlation":   round(c1,       2),
            "PCA Concentration": round(c2_fixed, 2),
            "Volatility Stress": round(c3,       2),
            "BTC Dominance":     round(c4_fixed, 2),
            "Volume Anomaly":    round(c5,       2),
            "Total":             round(c1 + c2_fixed + c3 + c4_fixed + c5, 2),
        })

    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# REGIME-CONDITIONED STATISTICS — how portfolio behaves in each risk regime
# =============================================================================

def _regime_of(score: float) -> str:
    for lo, hi, label, _ in REGIMES:
        if score < hi or hi == 100:
            return label
    return "High Stress"


def regime_conditioned_stats(
    index_history: pd.Series,
    coin_data: dict,
) -> pd.DataFrame:
    """
    For each risk regime (Low / Transition / High Stress) compute:
        - Days in regime
        - Avg daily portfolio return
        - Avg daily portfolio volatility (30d ann.)
        - Max drawdown experienced while in regime
        - Avg next-5-day forward return

    Parameters
    ----------
    index_history : pd.Series  (date → systemic score) from compute_systemic_index
    coin_data     : {symbol: DataFrame} with "returns" and "price" columns

    Returns
    -------
    pd.DataFrame indexed by regime label
    """
    if index_history.empty:
        return pd.DataFrame()

    coins  = list(coin_data.keys())
    rets   = pd.DataFrame({s: coin_data[s]["returns"] for s in coins}).dropna()
    port_r = rets.mean(axis=1)

    # Align
    common = index_history.index.intersection(port_r.index)
    if common.empty:
        return pd.DataFrame()

    scores  = index_history.reindex(common)
    portret = port_r.reindex(common)

    # Portfolio cumulative value (for drawdown)
    port_cum = (1 + portret).cumprod()

    records = []
    for lo, hi, label, color in REGIMES:
        mask = (scores >= lo) & (scores < hi) if hi < 100 else (scores >= lo)
        n    = int(mask.sum())
        if n == 0:
            records.append({
                "Regime": label, "Days": 0,
                "Avg Daily Ret": "—", "Ann. Vol": "—",
                "Max Drawdown": "—", "Fwd 5d Ret": "—", "color": color,
            })
            continue

        r_in      = portret[mask]
        avg_ret   = float(r_in.mean())
        ann_vol   = float(r_in.std() * np.sqrt(_ANNUAL))

        # Max drawdown inside regime windows
        dd_vals = []
        in_dates = common[mask]
        for d in in_dates:
            loc = common.get_loc(d)
            window_end = min(loc + 30, len(port_cum) - 1)
            window = port_cum.iloc[loc: window_end + 1]
            peak   = window.cummax()
            dd     = ((window - peak) / peak).min()
            dd_vals.append(float(dd))
        max_dd = float(min(dd_vals)) if dd_vals else float("nan")

        # Avg forward 5-day return (is regime predictive of near-term direction?)
        fwd_rets = []
        for d in in_dates:
            loc = common.get_loc(d)
            if loc + 5 < len(portret):
                fwd_rets.append(float(portret.iloc[loc + 1: loc + 6].sum()))
        avg_fwd = float(np.mean(fwd_rets)) if fwd_rets else float("nan")

        records.append({
            "Regime":       label,
            "Days":         n,
            "Avg Daily Ret": f"{avg_ret:+.3%}",
            "Ann. Vol":      f"{ann_vol:.1%}",
            "Max Drawdown":  f"{max_dd:.1%}",
            "Fwd 5d Ret":    f"{avg_fwd:+.3%}",
            "color":         color,
        })

    return pd.DataFrame(records)


# =============================================================================
# LEADING-INDICATOR SERIES — does a high score predict higher vol N days later?
# =============================================================================

def leading_indicator_series(
    index_history: pd.Series,
    coin_data: dict,
    lead_days: int = 14,
) -> pd.DataFrame:
    """
    Compute the realized portfolio volatility *lead_days* forward from each date,
    aligned with today's risk score.

    Returns a DataFrame with columns: score, fwd_vol.
    — Used to answer: "Does a high score today predict higher volatility ahead?"
    """
    if index_history.empty:
        return pd.DataFrame()

    coins  = list(coin_data.keys())
    rets   = pd.DataFrame({s: coin_data[s]["returns"] for s in coins}).dropna()
    port_r = rets.mean(axis=1)

    common = index_history.index.intersection(port_r.index)
    if len(common) < lead_days + 5:
        return pd.DataFrame()

    scores  = index_history.reindex(common)
    portret = port_r.reindex(common)

    rows = []
    for i in range(len(common) - lead_days):
        fwd   = portret.iloc[i + 1: i + lead_days + 1]
        fvol  = float(fwd.std() * np.sqrt(_ANNUAL))
        rows.append({"score": float(scores.iloc[i]), "fwd_vol": fvol,
                     "date": common[i]})
    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# LEAD-LAG CORRELATIONS — does the risk score lead market-stress indicators?
# =============================================================================

def lead_lag_correlations(
    index_history: pd.Series,
    coin_data: dict,
    max_lag: int = 15,
) -> dict:
    """
    Compute cross-correlations between the risk score and three market-stress
    targets at integer lags  –max_lag … +max_lag.

    A *positive* lag k means: corr(score(t), target(t+k))
    → positive peak at lag > 0  ⟹  risk score LEADS the stress indicator.

    Targets
    -------
    vol         – rolling 5-day realised portfolio volatility
    drawdown    – rolling 10-day portfolio drawdown (positive fraction)
    correlation – rolling 10-day average pairwise correlation

    Returns
    -------
    dict  {'vol': DataFrame, 'drawdown': DataFrame, 'correlation': DataFrame}
    Each DataFrame has columns: lag (int), corr (float)
    """
    if index_history.empty or not coin_data:
        return {}

    coins  = list(coin_data.keys())
    rets   = pd.DataFrame({s: coin_data[s]["returns"] for s in coins}).dropna()
    port_r = rets.mean(axis=1)

    common = index_history.index.intersection(port_r.index)
    if len(common) < 2 * max_lag + 20:
        return {}

    score  = index_history.reindex(common).values.astype(float)
    pr     = port_r.reindex(common).values.astype(float)

    # --- Target 1: rolling 5-day realised volatility ----------------------
    vol_ser = pd.Series(pr).rolling(5).std().bfill().values

    # --- Target 2: rolling 10-day max drawdown (positive value) ----------
    cum_val = np.cumprod(1 + np.nan_to_num(pr))
    dd_ser  = np.zeros(len(cum_val))
    for i in range(1, len(cum_val)):
        window_peak = float(np.max(cum_val[max(0, i - 10): i + 1]))
        dd_ser[i]   = max(0.0, (window_peak - cum_val[i]) / window_peak
                          if window_peak > 0 else 0.0)

    # --- Target 3: rolling 10-day avg pairwise correlation ----------------
    pair_series = []
    for i, s1 in enumerate(coins):
        for j, s2 in enumerate(coins):
            if j > i:
                pair_r = pd.DataFrame(
                    {s1: coin_data[s1]["returns"], s2: coin_data[s2]["returns"]}
                ).dropna()
                if len(pair_r) > 0:
                    pair_series.append(
                        pair_r[s1].rolling(10).corr(pair_r[s2])
                    )

    if pair_series:
        avg_corr_ts = pd.concat(pair_series, axis=1).mean(axis=1)
        corr_ser = avg_corr_ts.reindex(common).ffill().fillna(0).values.astype(float)
    else:
        corr_ser = np.zeros(len(score))

    def _xcorr(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Pearson corr(x(t), y(t+lag)) for each lag in -max_lag…+max_lag."""
        n    = len(x)
        rows = []
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                xs, ys = x[:n - lag], y[lag:] if lag > 0 else y
            else:
                xs, ys = x[-lag:], y[:n + lag]
            if len(xs) < 10:
                rows.append({"lag": lag, "corr": float("nan")})
                continue
            c = np.corrcoef(xs, ys)[0, 1]
            rows.append({"lag": lag, "corr": float(c)})
        return pd.DataFrame(rows)

    return {
        "vol":         _xcorr(score, vol_ser),
        "drawdown":    _xcorr(score, dd_ser),
        "correlation": _xcorr(score, corr_ser),
    }


# =============================================================================
# STRATEGY BACKTEST — simple risk-managed rule vs buy-and-hold
# =============================================================================

def regime_strategy_backtest(
    index_history: pd.Series,
    coin_data: dict,
) -> dict:
    """
    Backtest a rule-based, risk-managed strategy vs equal-weight buy-and-hold.

    Rule (executed one day after score observation — no look-ahead):
        score < 30          →  full exposure   (weight = 1.0)
        30 ≤ score < 60     →  half exposure   (weight = 0.5)
        score ≥ 60          →  cash / flat     (weight = 0.0)

    Returns
    -------
    dict with
        'equity'  – pd.DataFrame indexed by date:
                        strategy (cumulative), buyhold (cumulative), weight
        'stats'   – dict:
                        CAGR_strat, CAGR_bh, MaxDD_strat, MaxDD_bh,
                        Sharpe_strat, Sharpe_bh, n_days, pct_invested
    """
    if index_history.empty or not coin_data:
        return {}

    coins  = list(coin_data.keys())
    rets   = pd.DataFrame({s: coin_data[s]["returns"] for s in coins}).dropna()
    port_r = rets.mean(axis=1)

    common = index_history.index.intersection(port_r.index)
    if len(common) < 30:
        return {}

    scores  = index_history.reindex(common)
    portret = port_r.reindex(common)

    def _w(s: float) -> float:
        if s < 30:  return 1.0
        if s < 60:  return 0.5
        return 0.0

    weights   = scores.apply(_w).shift(1).fillna(1.0)
    strat_ret = portret * weights
    bh_ret    = portret

    def _cum(r: pd.Series) -> np.ndarray:
        return (1 + r).cumprod().values

    def _max_dd(r: pd.Series) -> float:
        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        return float(dd.min())

    def _cagr(r: pd.Series) -> float:
        n     = len(r)
        total = float((1 + r).prod())
        return total ** (_ANNUAL / n) - 1 if n > 0 else float("nan")

    def _sharpe(r: pd.Series) -> float:
        return (float(r.mean()) * _ANNUAL / (float(r.std()) * np.sqrt(_ANNUAL))
                if r.std() > 0 else float("nan"))

    equity = pd.DataFrame({
        "strategy": _cum(strat_ret),
        "buyhold":  _cum(bh_ret),
        "weight":   weights.values,
    }, index=common)

    stats = {
        "CAGR_strat":    _cagr(strat_ret),
        "CAGR_bh":       _cagr(bh_ret),
        "MaxDD_strat":   _max_dd(strat_ret),
        "MaxDD_bh":      _max_dd(bh_ret),
        "Sharpe_strat":  _sharpe(strat_ret),
        "Sharpe_bh":     _sharpe(bh_ret),
        "n_days":        len(common),
        "pct_invested":  float(weights.mean()),
    }

    return {"equity": equity, "stats": stats}