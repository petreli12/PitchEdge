"""Load pre-match odds for held-out tournament backtests.

Primary source: football-data.co.uk World Cup xlsx (market average H/D/A columns).
Optional: the-odds-api.com historical endpoint for Euro / Copa when ``ODDS_API_KEY``
is set (paid plan).
"""

from __future__ import annotations

import logging
import unicodedata
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from pitchedge import config
from pitchedge.model.devig import devig_proportional

log = logging.getLogger(__name__)

UTC = timezone.utc

# Map Kaggle / football-data naming differences (keys are normalized).
TEAM_NAME_ALIASES: dict[str, str] = {
    "usa": "united states",
    "united states": "united states",
    "korea republic": "south korea",
    "south korea": "south korea",
    "czech republic": "czechia",
    "czechia": "czechia",
    "turkey": "turkiye",
    "turkiye": "turkiye",
    "ivory coast": "cote d'ivoire",
    "cote d'ivoire": "cote d'ivoire",
    "cape verde": "cabo verde",
    "cabo verde": "cabo verde",
    "bosnia and herzegovina": "bosnia & herzegovina",
    "bosnia & herzegovina": "bosnia & herzegovina",
    "republic of ireland": "ireland",
    "ireland": "ireland",
}


def normalize_team_name(name: str) -> str:
    """Normalization key for cross-source team matching."""
    text = name.strip()
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    key = without_marks.casefold()
    return TEAM_NAME_ALIASES.get(key, key)


def match_key(home_team: str, away_team: str, match_date: date) -> tuple[str, str, str]:
    """Stable join key for odds ↔ results."""
    return (
        normalize_team_name(home_team),
        normalize_team_name(away_team),
        match_date.isoformat(),
    )


def _parse_avg_odds_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract ``H-Avg`` / ``D-Avg`` / ``A-Avg`` (or ``H_Avg`` variants)."""
    col_map = {str(c).lower().replace("_", "-"): c for c in frame.columns}
    h = col_map.get("h-avg")
    d = col_map.get("d-avg")
    a = col_map.get("a-avg")
    if h is None or d is None or a is None:
        raise ValueError(f"missing H/D/A average odds columns in {list(frame.columns)}")
    out = frame.copy()
    out["home_odds"] = pd.to_numeric(out[h], errors="coerce")
    out["draw_odds"] = pd.to_numeric(out[d], errors="coerce")
    out["away_odds"] = pd.to_numeric(out[a], errors="coerce")
    return out


def load_football_data_sheet(
    xlsx_path: str | Path,
    sheet_name: str,
) -> pd.DataFrame:
    """Parse one tournament sheet from football-data.co.uk World Cup workbook."""
    path = Path(xlsx_path)
    raw = pd.read_excel(path, sheet_name=sheet_name)
    odds = _parse_avg_odds_columns(raw)
    odds["Date"] = pd.to_datetime(odds["Date"]).dt.date
    rows: list[dict[str, Any]] = []
    for rec in odds.to_dict(orient="records"):
        ho = rec.get("home_odds")
        do = rec.get("draw_odds")
        ao = rec.get("away_odds")
        if ho is None or do is None or ao is None:
            continue
        if pd.isna(ho) or pd.isna(do) or pd.isna(ao):
            continue
        home = str(rec["Home"]).strip()
        away = str(rec["Away"]).strip()
        d = rec["Date"]
        ph, pdw, pa = devig_proportional(float(ho), float(do), float(ao))
        rows.append(
            {
                "date": d,
                "home_team": home,
                "away_team": away,
                "home_odds": float(ho),
                "draw_odds": float(do),
                "away_odds": float(ao),
                "p_home": ph,
                "p_draw": pdw,
                "p_away": pa,
                "source": "football-data",
                "match_key": match_key(home, away, d),
            }
        )
    return pd.DataFrame(rows)


def load_football_data_world_cups(
    xlsx_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load odds for World Cup 2018 and 2022 sheets."""
    path = Path(xlsx_path or config.BACKTEST_ODDS_XLSX)
    if not path.is_file():
        raise FileNotFoundError(
            f"football-data odds workbook not found: {path}. "
            "Download WorldCup2022.xlsx from football-data.co.uk into data/backtest/."
        )
    parts = [
        load_football_data_sheet(path, "WorldCup2018"),
        load_football_data_sheet(path, "WorldCup2022"),
    ]
    return pd.concat(parts, ignore_index=True)


def _extract_h2h_from_event(
    event: dict[str, Any],
    *,
    home_team: str,
    away_team: str,
) -> tuple[float, float, float] | None:
    """Pick first bookmaker h2h market and return decimal odds."""
    home_key = normalize_team_name(home_team)
    away_key = normalize_team_name(away_team)
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            prices: dict[str, float] = {}
            for outcome in market["outcomes"]:
                label = str(outcome["name"])
                key = (
                    normalize_team_name("draw")
                    if label.lower() == "draw"
                    else normalize_team_name(label)
                )
                prices[key] = float(outcome["price"])
            ho = prices.get(home_key)
            do = prices.get(normalize_team_name("draw"))
            ao = prices.get(away_key)
            if ho and do and ao and ho > 1 and do > 1 and ao > 1:
                return (float(ho), float(do), float(ao))
    return None


def fetch_odds_api_historical_for_match(
    *,
    sport_key: str,
    match_date: date,
    home_team: str,
    away_team: str,
    api_key: str | None = None,
    regions: str = "uk,eu",
) -> tuple[float, float, float] | None:
    """Query historical odds snapshot on ``match_date`` (UTC noon) and match teams."""
    key = api_key or config.ODDS_API_KEY
    if not key:
        return None
    snapshot_time = datetime.combine(match_date, time(12, 0), tzinfo=UTC).isoformat().replace(
        "+00:00", "Z"
    )
    url = (
        f"{config.ODDS_API_BASE_URL.rstrip('/')}/v4/historical/sports/{sport_key}/odds"
    )
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "decimal",
        "date": snapshot_time,
    }
    try:
        resp = requests.get(url, params=params, timeout=30.0)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("odds-api historical failed sport=%s date=%s: %s", sport_key, match_date, exc)
        return None

    events = payload.get("data") or []
    home_key = normalize_team_name(home_team)
    away_key = normalize_team_name(away_team)
    for event in events:
        eh = normalize_team_name(str(event.get("home_team", "")))
        ea = normalize_team_name(str(event.get("away_team", "")))
        if {eh, ea} != {home_key, away_key}:
            continue
        triple = _extract_h2h_from_event(
            event, home_team=home_team, away_team=away_team
        )
        if triple:
            return triple
    return None


def odds_api_historical_available(api_key: str | None = None) -> bool:
    """Return False when the key is on a free plan (historical endpoint 401)."""
    key = api_key or config.ODDS_API_KEY
    if not key:
        return False
    url = (
        f"{config.ODDS_API_BASE_URL.rstrip('/')}/v4/historical/sports/"
        "soccer_uefa_european_championship/odds"
    )
    try:
        resp = requests.get(
            url,
            params={
                "apiKey": key,
                "regions": "uk",
                "markets": "h2h",
                "date": "2024-06-14T12:00:00Z",
            },
            timeout=30.0,
        )
        if resp.status_code == 401:
            payload = resp.json()
            if payload.get("error_code") == "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN":
                log.warning(
                    "Odds API historical odds require a paid plan "
                    "(HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN). "
                    "Euro/Copa 2024 market rows will be missing until you upgrade "
                    "or populate %s via scripts/fetch_euro_copa_odds.py.",
                    config.BACKTEST_EURO_COPA_CACHE,
                )
                return False
        resp.raise_for_status()
        return True
    except Exception as exc:
        log.warning("Odds API historical probe failed: %s", exc)
        return False


def enrich_odds_from_odds_api(
    matches: list[dict[str, Any]],
    *,
    sport_key: str,
    api_key: str | None = None,
    historical_ok: bool | None = None,
) -> pd.DataFrame:
    """Fetch historical odds for each match row (slow; uses API credits)."""
    if historical_ok is False:
        return pd.DataFrame()
    if historical_ok is None and not odds_api_historical_available(api_key):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for m in matches:
        triple = fetch_odds_api_historical_for_match(
            sport_key=sport_key,
            match_date=m["date"],
            home_team=m["home_team"],
            away_team=m["away_team"],
            api_key=api_key,
        )
        if triple is None:
            continue
        ho, do, ao = triple
        ph, pdw, pa = devig_proportional(ho, do, ao)
        rows.append(
            {
                "date": m["date"],
                "home_team": m["home_team"],
                "away_team": m["away_team"],
                "home_odds": ho,
                "draw_odds": do,
                "away_odds": ao,
                "p_home": ph,
                "p_draw": pdw,
                "p_away": pa,
                "source": "odds-api-historical",
                "match_key": match_key(m["home_team"], m["away_team"], m["date"]),
            }
        )
    return pd.DataFrame(rows)


def load_euro_copa_odds_cache(
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load cached Euro/Copa odds CSV written by ``fetch_euro_copa_odds``."""
    path = Path(cache_path or config.BACKTEST_EURO_COPA_CACHE)
    if not path.is_file():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for rec in raw.to_dict(orient="records"):
        d = rec.get("date") or rec.get("Date")
        if hasattr(d, "date"):
            d = d.date()
        elif isinstance(d, str):
            d = pd.to_datetime(d).date()
        home = str(rec.get("home_team") or rec.get("Home") or rec.get("HomeTeam")).strip()
        away = str(rec.get("away_team") or rec.get("Away") or rec.get("AwayTeam")).strip()
        if "p_home" in rec and rec["p_home"] == rec["p_home"]:
            ph, pdw, pa = float(rec["p_home"]), float(rec["p_draw"]), float(rec["p_away"])
        else:
            ho = float(rec["home_odds"])
            do = float(rec["draw_odds"])
            ao = float(rec["away_odds"])
            ph, pdw, pa = devig_proportional(ho, do, ao)
        rows.append(
            {
                "date": d,
                "home_team": home,
                "away_team": away,
                "home_odds": rec.get("home_odds"),
                "draw_odds": rec.get("draw_odds"),
                "away_odds": rec.get("away_odds"),
                "p_home": ph,
                "p_draw": pdw,
                "p_away": pa,
                "source": "euro-copa-cache",
                "match_key": match_key(home, away, d),
            }
        )
    return pd.DataFrame(rows)


def build_backtest_odds_table(
    *,
    xlsx_path: str | Path | None = None,
    use_odds_api: bool = True,
    euro_copa_matches: list[dict[str, Any]] | None = None,
    euro_copa_cache_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge football-data WC odds with Euro/Copa cache and/or odds-api rows.

    Returns ``(odds_df, meta)`` where ``meta`` reports join coverage for the report.
    """
    meta: dict[str, Any] = {
        "euro_copa_matches_requested": len(euro_copa_matches or []),
        "euro_copa_cache_rows": 0,
        "odds_api_rows": 0,
        "odds_api_historical_blocked": False,
    }
    frames: list[pd.DataFrame] = [load_football_data_world_cups(xlsx_path)]
    cache_df = load_euro_copa_odds_cache(euro_copa_cache_path)
    if not cache_df.empty:
        meta["euro_copa_cache_rows"] = len(cache_df)
        frames.append(cache_df)
    historical_ok: bool | None = None
    if use_odds_api and config.ODDS_API_KEY and euro_copa_matches:
        from pitchedge.eval.tournaments import HELD_OUT_TOURNAMENTS

        historical_ok = odds_api_historical_available()
        meta["odds_api_historical_blocked"] = not historical_ok
        by_slug = {t.slug: t for t in HELD_OUT_TOURNAMENTS}
        for slug in ("euro_2024", "copa_2024"):
            tourn = by_slug[slug]
            if not tourn.odds_api_sport_key:
                continue
            subset = [m for m in euro_copa_matches if m.get("tournament_slug") == slug]
            if not subset:
                continue
            log.info(
                "Fetching odds-api historical odds for %s (%d matches)...",
                tourn.label,
                len(subset),
            )
            api_df = enrich_odds_from_odds_api(
                subset,
                sport_key=tourn.odds_api_sport_key,
                historical_ok=historical_ok,
            )
            if not api_df.empty:
                meta["odds_api_rows"] += len(api_df)
                frames.append(api_df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["match_key"], keep="first")
    meta["total_odds_rows"] = len(combined)
    return combined, meta


def lookup_market_probs(
    odds_df: pd.DataFrame,
    *,
    home_team: str,
    away_team: str,
    match_date: date,
) -> tuple[float, float, float] | None:
    """Return de-vigged market ``(p_home, p_draw, p_away)`` or None."""
    key = match_key(home_team, away_team, match_date)
    hit = odds_df[odds_df["match_key"] == key]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return (float(row["p_home"]), float(row["p_draw"]), float(row["p_away"]))
