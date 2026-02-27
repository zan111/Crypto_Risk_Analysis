"""
Social Sentiment Engine
=======================
Fetches and aggregates social/news sentiment for crypto coins.

Sources (all optional — graceful fallback if keys missing):
  - NewsAPI       : free tier, get key at https://newsapi.org/
  - Reddit (PRAW) : read-only app, get keys at https://old.reddit.com/prefs/apps
  - X/Twitter     : requires tweepy + Bearer Token from developer.x.com

Set credentials via environment variables (do NOT hardcode):
    NEWS_API_KEY      — your NewsAPI key
    REDDIT_CLIENT_ID  — Reddit app client id
    REDDIT_SECRET     — Reddit app client secret
    X_BEARER_TOKEN    — Twitter/X Bearer token

Each source is tried independently; missing credentials are skipped cleanly.

Returns aggregated daily scores (-1 negative … +1 positive) per coin.
"""

import os
import time
import pickle
import requests
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import nltk

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import praw
    _PRAW_OK = True
except ImportError:
    _PRAW_OK = False

try:
    import tweepy
    _TWEEPY_OK = True
except ImportError:
    _TWEEPY_OK = False

_SIA = SentimentIntensityAnalyzer()

# Cache lives next to this file
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_cache.pkl")
_CACHE_TTL_SEC = 900   # 15-minute cache — extended for Top 10 (more API calls)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _score(text: str) -> float:
    """Return VADER compound score in [-1, +1]."""
    return _SIA.polarity_scores(str(text))["compound"]


def _load_cache() -> dict:
    try:
        with open(_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass


# =============================================================================
# SOURCE FETCHERS
# =============================================================================

def _fetch_newsapi(query: str, api_key: str, page_size: int = 50) -> pd.DataFrame:
    """
    Fetch recent news headlines from NewsAPI.
    Free tier: 100 req/day, English only, last 30 days.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    rows = []
    for a in articles:
        text = (a.get("title") or "") + " " + (a.get("description") or "")
        ts = a.get("publishedAt")
        rows.append({
            "source": "news",
            "text": text.strip(),
            "ts": pd.to_datetime(ts, utc=True) if ts else pd.NaT,
            "url": a.get("url", ""),
            "title": a.get("title", ""),
        })
    return pd.DataFrame(rows)


def _fetch_reddit(subreddit: str, client_id: str, client_secret: str,
                  limit: int = 100) -> pd.DataFrame:
    """Fetch hot posts from a subreddit via PRAW (read-only, no login needed)."""
    if not _PRAW_OK:
        raise ImportError("praw not installed — run: pip install praw")
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="defi_risk_tool/1.0 (by /u/anonymous)",
    )
    rows = []
    for post in reddit.subreddit(subreddit).hot(limit=limit):
        text = (post.title or "") + " " + (post.selftext or "")
        rows.append({
            "source": "reddit",
            "text": text.strip(),
            "ts": pd.to_datetime(post.created_utc, unit="s", utc=True),
            "url": f"https://reddit.com{post.permalink}",
            "title": post.title,
        })
    return pd.DataFrame(rows)


def _fetch_x(query: str, bearer_token: str, max_results: int = 100) -> pd.DataFrame:
    """Fetch recent tweets via Twitter/X v2 API (tweepy)."""
    if not _TWEEPY_OK:
        raise ImportError("tweepy not installed — run: pip install tweepy")
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    # Exclude retweets for cleaner signal
    q = f"({query}) -is:retweet lang:en"
    resp = client.search_recent_tweets(
        query=q, max_results=min(max_results, 100),
        tweet_fields=["created_at", "text"],
    )
    rows = []
    if resp and resp.data:
        for t in resp.data:
            rows.append({
                "source": "x",
                "text": t.text,
                "ts": pd.to_datetime(t.created_at, utc=True),
                "url": f"https://x.com/i/web/status/{t.id}",
                "title": t.text[:80],
            })
    return pd.DataFrame(rows)


# =============================================================================
# AGGREGATION
# =============================================================================

def _aggregate(df: pd.DataFrame, bucket: str = "1D") -> pd.DataFrame:
    """
    Score each row with VADER then bucket by day.
    Returns DataFrame with columns: sentiment (mean), volume (count), std.
    """
    if df.empty:
        return pd.DataFrame(columns=["sentiment", "volume", "std"])
    df = df.copy()
    df["score"] = df["text"].map(_score)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.dropna(subset=["ts"])
    agg = (
        df.set_index("ts")
        .resample(bucket)["score"]
        .agg(sentiment="mean", volume="count", std="std")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)
    return agg


# =============================================================================
# PUBLIC API
# =============================================================================

# Keyword map: coin symbol → search terms
COIN_QUERIES = {
    "BTC": "bitcoin OR #bitcoin OR $BTC",
    "ETH": "ethereum OR #ethereum OR $ETH",
    "SOL": "solana OR #solana OR $SOL",
}

REDDIT_SUBS = {
    "BTC": "Bitcoin+CryptoCurrency",
    "ETH": "ethereum+CryptoCurrency",
    "SOL": "solana+CryptoCurrency",
}


def get_sentiment(
    coins=("BTC", "ETH", "SOL"),
    force_refresh: bool = False,
    news_api_key: str | None = None,
    reddit_client_id: str | None = None,
    reddit_secret: str | None = None,
    x_bearer_token: str | None = None,
) -> dict:
    """
    Fetch and aggregate social sentiment for each coin.

    Credentials fall back to environment variables if not passed:
        NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_SECRET, X_BEARER_TOKEN

    Returns
    -------
    dict keyed by coin symbol, each value is a dict with:
        "by_day"   : pd.DataFrame with sentiment/volume/std per day
        "current"  : float — latest day mean sentiment
        "sources"  : list of source names that were fetched
        "headlines": list of {title, url, score} — latest 6 headlines (NewsAPI)
        "error"    : str or None
    """
    # Resolve credentials
    news_key    = news_api_key    or os.getenv("NEWS_API_KEY")
    reddit_id   = reddit_client_id or os.getenv("REDDIT_CLIENT_ID")
    reddit_sec  = reddit_secret    or os.getenv("REDDIT_SECRET")
    x_token     = x_bearer_token  or os.getenv("X_BEARER_TOKEN")

    # Cache check
    cache = _load_cache()
    cache_key = "sentiment_v2"
    if (
        not force_refresh
        and cache_key in cache
        and (datetime.now(tz=timezone.utc) - cache[cache_key]["ts"]).total_seconds() < _CACHE_TTL_SEC
    ):
        return cache[cache_key]["data"]

    results = {}

    for coin in coins:
        query = COIN_QUERIES.get(coin, coin)
        frames = []
        sources = []
        headlines = []
        errors = []

        # --- NewsAPI ---
        if news_key:
            try:
                df_n = _fetch_newsapi(query, news_key)
                if not df_n.empty:
                    df_n["score"] = df_n["text"].map(_score)
                    frames.append(df_n[["source", "text", "ts", "score"]])
                    sources.append("news")
                    headlines = (
                        df_n.sort_values("ts", ascending=False)
                        .head(6)[["title", "url", "score"]]
                        .to_dict("records")
                    )
            except Exception as e:
                errors.append(f"NewsAPI: {e}")

        # --- Reddit ---
        if reddit_id and reddit_sec:
            try:
                df_r = _fetch_reddit(REDDIT_SUBS.get(coin, "CryptoCurrency"),
                                     reddit_id, reddit_sec)
                if not df_r.empty:
                    df_r["score"] = df_r["text"].map(_score)
                    frames.append(df_r[["source", "text", "ts", "score"]])
                    sources.append("reddit")
            except Exception as e:
                errors.append(f"Reddit: {e}")

        # --- X/Twitter ---
        if x_token:
            try:
                df_x = _fetch_x(query, x_token)
                if not df_x.empty:
                    df_x["score"] = df_x["text"].map(_score)
                    frames.append(df_x[["source", "text", "ts", "score"]])
                    sources.append("x")
            except Exception as e:
                errors.append(f"X: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined["ts"] = pd.to_datetime(combined["ts"], utc=True)
            combined = combined.dropna(subset=["ts"])
            by_day = (
                combined.set_index("ts")
                .resample("1D")["score"]
                .agg(sentiment="mean", volume="count", std="std")
                .reset_index()
            )
            by_day["std"] = by_day["std"].fillna(0)
            current = float(by_day["sentiment"].iloc[-1]) if not by_day.empty else 0.0
        else:
            by_day = pd.DataFrame(columns=["ts", "sentiment", "volume", "std"])
            current = 0.0

        results[coin] = {
            "by_day": by_day,
            "current": current,
            "sources": sources,
            "headlines": headlines,
            "error": "; ".join(errors) if errors else None,
        }

    # Save cache
    cache[cache_key] = {"ts": datetime.now(tz=timezone.utc), "data": results}
    _save_cache(cache)
    return results


def sentiment_vs_returns(sentiment_data: dict, coin_data: dict) -> pd.DataFrame:
    """
    Align daily sentiment scores with coin returns for correlation analysis.

    Returns a DataFrame with columns: date, coin, sentiment, return_1d.
    """
    rows = []
    for coin, s in sentiment_data.items():
        if coin not in coin_data or s["by_day"].empty:
            continue
        bd = s["by_day"].copy()
        bd["ts"] = pd.to_datetime(bd["ts"]).dt.tz_localize(None).dt.normalize()
        ret = coin_data[coin]["returns"].dropna().copy()
        ret.index = ret.index.normalize()
        for _, row in bd.iterrows():
            d = row["ts"]
            ret_val = ret.get(d, np.nan)
            rows.append({
                "date": d,
                "coin": coin,
                "sentiment": row["sentiment"],
                "return_1d": float(ret_val) if not pd.isna(ret_val) else np.nan,
                "volume": row["volume"],
            })
    return pd.DataFrame(rows).dropna(subset=["sentiment", "return_1d"])


# =============================================================================
# POSITIONING LAYER — Sentiment Sensitivity Analysis
# =============================================================================

def sentiment_index_daily(sentiment_data: dict) -> pd.DataFrame:
    """
    Compute a single daily Sentiment Index across all coins.

    Returns DataFrame with columns: date, sentiment_index, sentiment_z
    The z-score measures how far today's sentiment deviates from its
    rolling 14-day mean — a "sentiment shock" detector.
    """
    frames = []
    for coin, s in sentiment_data.items():
        bd = s["by_day"].copy()
        if bd.empty:
            continue
        bd["ts"] = pd.to_datetime(bd["ts"]).dt.tz_localize(None).dt.normalize()
        bd = bd.set_index("ts")[["sentiment"]]
        bd.columns = [coin]
        frames.append(bd)

    if not frames:
        return pd.DataFrame(columns=["date", "sentiment_index", "sentiment_z"])

    combined = pd.concat(frames, axis=1).sort_index()
    combined["sentiment_index"] = combined.mean(axis=1)

    roll_mean = combined["sentiment_index"].rolling(14, min_periods=3).mean()
    roll_std = combined["sentiment_index"].rolling(14, min_periods=3).std()
    combined["sentiment_z"] = (combined["sentiment_index"] - roll_mean) / (roll_std + 1e-10)

    result = combined[["sentiment_index", "sentiment_z"]].reset_index()
    result.columns = ["date", "sentiment_index", "sentiment_z"]
    return result.dropna()


def sentiment_sensitivity(
    sentiment_data: dict,
    coin_data: dict,
    risk_index: pd.Series | None = None,
) -> dict:
    """
    Measure how price and volatility respond to sentiment shocks,
    optionally conditioned on risk regime.

    Returns
    -------
    dict with keys:
        "price_per_sentiment"  — β coefficient: return per unit sentiment change
        "vol_response"         — DataFrame: {regime, avg_sent_shock, avg_vol, n_days}
        "regime_impact"        — str: one-line deliverable sentence
        "daily_data"           — DataFrame: aligned daily data for charting
    """
    si = sentiment_index_daily(sentiment_data)
    if si.empty:
        return {
            "price_per_sentiment": float("nan"),
            "vol_response": pd.DataFrame(),
            "regime_impact": "Insufficient sentiment data.",
            "daily_data": pd.DataFrame(),
        }

    # Build portfolio return series
    coins = list(coin_data.keys())
    rets = pd.DataFrame({c: coin_data[c]["returns"] for c in coins}).dropna()
    port_ret = rets.mean(axis=1)
    port_ret.index = port_ret.index.normalize()

    # 5-day realised vol
    port_vol_5d = port_ret.rolling(5).std() * np.sqrt(365)

    # Align
    si = si.set_index("date")
    common = si.index.intersection(port_ret.index).intersection(port_vol_5d.dropna().index)
    if len(common) < 5:
        return {
            "price_per_sentiment": float("nan"),
            "vol_response": pd.DataFrame(),
            "regime_impact": "Not enough overlapping data between sentiment and price.",
            "daily_data": pd.DataFrame(),
        }

    df = pd.DataFrame({
        "sentiment_index": si.loc[common, "sentiment_index"],
        "sentiment_z": si.loc[common, "sentiment_z"],
        "return_1d": port_ret.reindex(common),
        "vol_5d": port_vol_5d.reindex(common),
    }).dropna()

    # 1. Price per unit sentiment (linear regression β)
    if len(df) > 3:
        z = np.polyfit(df["sentiment_z"], df["return_1d"] * 100, 1)
        beta = z[0]  # pct return per z-score unit of sentiment shock
    else:
        beta = float("nan")

    # 2. Volatility response conditioned on risk regime
    vol_resp_rows = []
    regime_impact_str = "Insufficient data for regime conditioning."

    if risk_index is not None and not risk_index.empty:
        risk_idx = risk_index.copy()
        risk_idx.index = risk_idx.index.normalize()
        df["risk_score"] = risk_idx.reindex(df.index)
        df = df.dropna(subset=["risk_score"])

        def _regime_label(s):
            if s < 30: return "Low Risk"
            if s < 60: return "Transition"
            return "High Stress"

        df["regime"] = df["risk_score"].apply(_regime_label)

        # Negative sentiment shocks only (z < -0.5)
        neg = df[df["sentiment_z"] < -0.5]

        for regime in ["Low Risk", "Transition", "High Stress"]:
            sub = neg[neg["regime"] == regime]
            if len(sub) > 0:
                vol_resp_rows.append({
                    "Regime": regime,
                    "Avg Sent. Shock (z)": f"{sub['sentiment_z'].mean():.2f}",
                    "Avg Fwd Vol (ann.)": f"{sub['vol_5d'].mean():.1%}",
                    "Avg Return (%)": f"{sub['return_1d'].mean() * 100:+.2f}%",
                    "N Days": len(sub),
                })

        # Build the deliverable sentence
        low_r = neg[neg["regime"] == "Low Risk"]["return_1d"].mean() * 100 if len(neg[neg["regime"] == "Low Risk"]) > 0 else 0
        high_r = neg[neg["regime"] == "High Stress"]["return_1d"].mean() * 100 if len(neg[neg["regime"] == "High Stress"]) > 0 else 0
        if abs(low_r) > 0 and abs(high_r) > 0:
            ratio = abs(high_r) / abs(low_r) if abs(low_r) > 0.001 else float("nan")
            regime_impact_str = (
                f"In High Stress regimes, negative sentiment shocks cause "
                f"{ratio:.1f}× larger downside ({high_r:+.2f}% avg) vs "
                f"Low Risk ({low_r:+.2f}% avg)."
            )
        else:
            regime_impact_str = "Not enough regime-conditioned sentiment shock data yet."
    else:
        # No risk index — still compute unconditional stats
        neg = df[df["sentiment_z"] < -0.5]
        pos = df[df["sentiment_z"] > 0.5]
        for label, sub in [("Neg Shock (z < -0.5)", neg), ("Pos Shock (z > 0.5)", pos)]:
            if len(sub) > 0:
                vol_resp_rows.append({
                    "Regime": label,
                    "Avg Sent. Shock (z)": f"{sub['sentiment_z'].mean():.2f}",
                    "Avg Fwd Vol (ann.)": f"{sub['vol_5d'].mean():.1%}",
                    "Avg Return (%)": f"{sub['return_1d'].mean() * 100:+.2f}%",
                    "N Days": len(sub),
                })

    return {
        "price_per_sentiment": beta,
        "vol_response": pd.DataFrame(vol_resp_rows) if vol_resp_rows else pd.DataFrame(),
        "regime_impact": regime_impact_str,
        "daily_data": df.reset_index(),
    }
