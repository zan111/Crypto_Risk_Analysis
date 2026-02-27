"""
API credentials — local use only, never commit to version control.
"""
import os

NEWS_API_KEY      = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID  = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET     = os.getenv("REDDIT_SECRET")
X_BEARER_TOKEN    = os.getenv(
    "X_BEARER_TOKEN",
)

# =============================================================================
# COIN SETS  (CoinGecko IDs mapped to display symbols)
# =============================================================================

COINS_TOP3 = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

COINS_TOP5 = {
    **COINS_TOP3,
    "BNB": "binancecoin",
    "XRP": "ripple",
}

COINS_TOP10 = {
    **COINS_TOP5,
    "ADA":  "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT":  "polkadot",
    "LINK": "chainlink",
}

# Ordered list of all supported symbols (used for UI display ordering)
ALL_COIN_SYMBOLS = list(COINS_TOP10.keys())

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

N_SIMS_QUICK = 300    # Quick Mode — fast interactive feedback
N_SIMS_FULL  = 1000   # Full Mode  — higher accuracy, slower
HORIZON_DAYS = 90     # Default simulation horizon

# =============================================================================
# CACHE TTLs  (seconds)
# =============================================================================

PRICE_CACHE_TTL     = 1_800   # 30 min — CoinGecko historical prices
SENTIMENT_CACHE_TTL = 900     # 15 min — news / Reddit / X scores
