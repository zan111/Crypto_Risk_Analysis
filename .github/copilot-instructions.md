<!-- .github/copilot-instructions.md - guidance for AI coding agents -->
# Copilot / AI Agent Instructions — DeFi Risk Tool

Purpose
- Help AI coding agents become productive quickly when modifying or extending this repo.

Quick mental model
- This is a local, single-user Dash app that computes a composite crypto systemic risk score
  by combining liquidity, regime, leverage, and sentiment signals. The UI (`app.py`) is just
  a thin Dash layer; most logic lives in the statistical modules below.

Core modules (follow these when changing behavior)
- `fetch_data.py` — single source of truth for historical prices, volumes and the local price cache (30min TTL). Example: `get_multi_coin_data('top5')` returns dict {symbol: DataFrame} with `price`, `returns`, `log_returns`, `volume_pct_rank`.
- `preprocess.py` — data munging utilities used before analytics (use same column names as `fetch_data` produces).
- `risk_engine.py` — numerical/statistics core (rolling vol/corr, PCA, VaR, portfolio decomposition). Functions expect the `coin_data` dict shape returned by `get_multi_coin_data()`.
- `risk_analysis.py` — higher-level helpers: beta calculation, regime detection, liquidity multipliers.
- `systemic_risk.py` — composes signals into the 0–100 index and contains backtests / lead-lag helpers.
- `social_sentiment.py` — multi-source sentiment extraction; optional APIs (NewsAPI, Reddit, X/Twitter).
- `app.py` — Dash callbacks, figure builders and orchestration. Keep UI changes isolated here.

Dataflow and integration points
- Typical pipeline: `get_multi_coin_data()` → `preprocess` → analytic calls in `risk_engine`/`risk_analysis` → `systemic_risk.compute_systemic_index()` → results rendered by `app.py`.
- Simulation models are implemented both in `fetch_data.py` and `app.py` (Monte Carlo variants). Prefer the `risk_engine` implementations for shared numeric routines and unit tests.
- Cache: `price_cache.pkl` is used by `fetch_data` — avoid bypassing it unless `force_refresh=True`.

Important conventions
- API keys live in `config.py` and are loaded from environment variables when available. Do NOT hardcode secrets; follow existing pattern (env fallback in `config.py`).
- Function contracts: most analytic functions accept `coin_data` as a dict of DataFrames with columns `price`, `returns`, `log_returns`, and sometimes `volume_pct_rank`. Preserve these column names when adding preprocessing steps.
- Numeric code must avoid side-effects (no plotting) — plotting belongs in `visualization.py` or `app.py` figure builders. This keeps analytics testable.
- Use annualisation constants and TTLs defined in `config.py` (e.g., `PRICE_CACHE_TTL`, `N_SIMS_QUICK`, `N_SIMS_FULL`).

Testing and running locally
- Recommended environment commands (from README):
```bash
conda create -n crypto_risk python=3.11
conda activate crypto_risk
pip install -r requirements.txt  # or install deps from README
python app.py
# open http://127.0.0.1:8050
```
- Many numeric functions are pure and easy to unit-test by providing small `coin_data` fixtures (two coins x short date range). Mock CoinGecko responses when testing `fetch_data`.

Repo-specific patterns to follow
- When adding a new analytic function, document expected input (`coin_data` shape) and output (DataFrame/Series/dict). See `risk_engine.compute_return_stats()` for a reference docstring style.
- Use `ensure_psd()` in `risk_engine.py` whenever constructing covariance matrices from sample returns to avoid numerical issues.
- For regime-sensitive behavior, prefer reusing `detect_volatility_regime()` / `beta_regime_summary()` rather than ad-hoc thresholds — these centralize regime logic.
- Plots for Dash should return Plotly `Figure` objects (see `fig_fan`, `fig_hist` in `app.py`). Avoid creating side-effecting Matplotlib figures for Dash pages.

Notes about external integrations
- CoinGecko via `pycoingecko` — primary pricing provider; rate limits avoided by local TTL cache.
- Sentiment depends on third-party APIs (NewsAPI, Reddit via `praw`, X/Twitter). These are optional; code gracefully falls back when keys are missing.

When creating PRs
- Keep numerical changes and UI changes in separate commits where possible (small diffs are easier to review).
- Add a short reproducible example in the PR description (commands to run a small simulation with `N_SIMS_QUICK`).

If you are uncertain
- Inspect `README.md` and `MATH_EXPLAINED.md` for the conceptual intent behind algorithms before changing formulas.

Questions / Feedback
- If anything in these instructions is unclear or missing, tell me which file or behavior you want more detail about and I will iterate.
