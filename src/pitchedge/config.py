"""Central configuration, loaded from environment variables.

All secrets and environment-specific settings live here and ONLY here, loaded
from a gitignored `.env` (see `.env.example`). No other module reads os.environ
directly and no secret is ever hardcoded.

Units / conventions:
  * Times are UTC everywhere internally; convert only at the display edge.
  * Probabilities live in [0, 1].
  * All randomness takes an explicit seed; ``RANDOM_SEED`` is the project default.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load `.env` from the project root if present. Real environment variables take
# precedence over the file, so deployment env (CI, prod) overrides local files.
load_dotenv()


def _get_str(name: str, default: str) -> str:
    """Return env var ``name`` as a string, falling back to ``default``."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    """Return env var ``name`` parsed as an int, falling back to ``default``."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    """Return env var ``name`` parsed as a float, falling back to ``default``."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return float(raw)


# --- Database -----------------------------------------------------------------
# SQLAlchemy URL using the psycopg (v3) driver. Default matches docker-compose.
DB_URL: str = _get_str(
    "DB_URL",
    "postgresql+psycopg://pitchedge:pitchedge@localhost:5432/pitchedge",
)

# --- Anthropic (narrative content engine) -------------------------------------
ANTHROPIC_API_KEY: str = _get_str("ANTHROPIC_API_KEY", "")
# Default narrative model; override in `.env` to trade quality for cost.
NARRATIVE_MODEL: str = _get_str("NARRATIVE_MODEL", "claude-sonnet-4-6")

# --- Telegram (content distribution) ------------------------------------------
TELEGRAM_BOT_TOKEN: str = _get_str("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = _get_str("TELEGRAM_CHAT_ID", "")
# Public join link for dashboard CTA (channel or t.me/... URL).
TELEGRAM_JOIN_URL: str = _get_str("TELEGRAM_JOIN_URL", "")

# --- Public dashboard deploy --------------------------------------------------
# When set (or when a bundled `data/snapshot/` exists), the Streamlit app serves
# a read-only snapshot instead of querying Postgres. This is how the locked
# pre-tournament board is published (e.g. Streamlit Community Cloud), where no
# Postgres is reachable. Empty -> live DB mode.
DASHBOARD_SNAPSHOT_DIR: str = _get_str("DASHBOARD_SNAPSHOT_DIR", "")
# External email-capture target (Google Form / Tally / Formspree URL) used on
# the public landing when there is no writable subscribers DB. Empty -> the
# email form is hidden in snapshot mode (we never silently drop submissions).
SUBSCRIBE_FORM_URL: str = _get_str("SUBSCRIBE_FORM_URL", "")

# Endpoint the in-app email form POSTs to on the public deploy (e.g. a Formspree
# form action, Tally, or a Google Form `formResponse` URL). When set, the
# landing keeps the in-app form + "you're on the list" success state and submits
# server-side over HTTP. ``SUBSCRIBE_EMAIL_FIELD`` is the form field name the
# endpoint expects (Formspree/Tally: "email"; Google Forms: "entry.<id>").
SUBSCRIBE_POST_URL: str = _get_str("SUBSCRIBE_POST_URL", "")
SUBSCRIBE_EMAIL_FIELD: str = _get_str("SUBSCRIBE_EMAIL_FIELD", "email")

# --- Reproducibility ----------------------------------------------------------
# Project-wide default seed. Simulations must be reproducible given inputs.
RANDOM_SEED: int = _get_int("RANDOM_SEED", 20260611)

# --- Model blend weight -------------------------------------------------------
# w in p_blend = w * p_model + (1 - w) * p_market.
# PROVENANCE: chosen by minimizing out-of-sample log loss in the Phase 4
# backtest (docs/BUILD_PLAN.md), NOT a magic number. The default here is a
# provisional placeholder until that backtest runs; it is expected to land low
# because the de-vigged market is the strongest single input we have.
BLEND_W: float = _get_float("BLEND_W", 0.00)

# Published standalone model temperature (``daily_disagreement``, etc.).
# 1.0 = raw Dixon-Coles. Phase 4 refit ~1.035 on n=208; production publishes raw.
MODEL_TEMPERATURE: float = _get_float("MODEL_TEMPERATURE", 1.0)

# --- Phase 4 backtest ---------------------------------------------------------
BACKTEST_ODDS_XLSX: str = _get_str(
    "BACKTEST_ODDS_XLSX",
    "data/backtest/WorldCup2022.xlsx",
)
BACKTEST_REPORT_PATH: str = _get_str("BACKTEST_REPORT_PATH", "backtest_report.md")
BACKTEST_RELIABILITY_PNG: str = _get_str(
    "BACKTEST_RELIABILITY_PNG",
    "reports/reliability_diagram.png",
)
# Cached Euro/Copa historical odds (CSV). Populated by scripts/fetch_euro_copa_odds.py
# when the Odds API key has a paid plan with /v4/historical access.
BACKTEST_EURO_COPA_CACHE: str = _get_str(
    "BACKTEST_EURO_COPA_CACHE",
    "data/backtest/euro_copa_odds_cache.csv",
)

# --- Data ingest paths --------------------------------------------------------
# Kaggle martj42 international results (results.csv inside the dataset folder).
HISTORY_CSV_PATH: str = _get_str(
    "HISTORY_CSV_PATH",
    "data/kaggle/international-results/results.csv",
)
# User-provided World Cup 2026 teams and fixtures CSVs.
TEAMS_CSV_PATH: str = _get_str("TEAMS_CSV_PATH", "data/wc2026/teams.csv")
FIXTURES_CSV_PATH: str = _get_str("FIXTURES_CSV_PATH", "data/wc2026/fixtures.csv")
ANNEX_C_JSON_PATH: str = _get_str("ANNEX_C_JSON_PATH", "data/wc2026/annex_c.json")

# --- Monte Carlo tournament sim (Phase 5) -------------------------------------
N_SIMS: int = _get_int("N_SIMS", 50_000)

# --- the-odds-api.com (v4) ----------------------------------------------------
ODDS_API_KEY: str = _get_str("ODDS_API_KEY", "")
ODDS_API_BASE_URL: str = _get_str("ODDS_API_BASE_URL", "https://api.the-odds-api.com")
# TODO: confirm the sport key for WC 2026 on your plan (GET /v4/sports).
ODDS_API_SPORT_KEY: str = _get_str("ODDS_API_SPORT_KEY", "soccer_fifa_world_cup")
ODDS_API_REGIONS: str = _get_str("ODDS_API_REGIONS", "us,uk,eu")

# --- Elo (international ratings prior) ----------------------------------------
ELO_INITIAL: float = _get_float("ELO_INITIAL", 1500.0)
ELO_K: float = _get_float("ELO_K", 20.0)
ELO_HOME_ADV: float = _get_float("ELO_HOME_ADV", 100.0)
ELO_RATING_DIVISOR: float = _get_float("ELO_RATING_DIVISOR", 400.0)

# --- Dixon-Coles --------------------------------------------------------------
DC_XI: float = _get_float("DC_XI", 0.005)  # time-decay per year; tune in Phase 4
DC_RECENCY_YEARS: float = _get_float("DC_RECENCY_YEARS", 6.0)
DC_MIN_MATCHES: int = _get_int("DC_MIN_MATCHES", 30)
DC_MAX_GOALS: int = _get_int("DC_MAX_GOALS", 10)
DC_RHO_MAX_ITER: int = _get_int("DC_RHO_MAX_ITER", 100)
DC_TAU_FLOOR: float = _get_float("DC_TAU_FLOOR", 1e-10)
