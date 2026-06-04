"""Adapter for the-odds-api.com (v4) -> ``odds_snapshots``.

HTTP is isolated in ``fetch_odds()`` so tests can inject JSON fixtures without a
live call. See https://the-odds-api.com/liveapi/guides/v4/

**TODO — confirm before production ingest:**
  * ``ODDS_API_SPORT_KEY``: the exact sport key for WC 2026 on your API plan
    (e.g. ``soccer_fifa_world_cup`` — list via GET /v4/sports).
  * Team name matching: API ``home_team`` / ``away_team`` strings must align with
    ``teams.name`` in your fixtures CSV (spelling, accents, "USA" vs "United States").
  * ``regions`` / ``bookmakers`` query params: which books you want snapshotted.
  * Whether the draw outcome is always labeled ``"Draw"`` in the h2h market for
    your chosen region (some books may differ).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

import requests

from pitchedge import config, db

log = logging.getLogger(__name__)

UTC = timezone.utc

# Outcome label the-odds-api uses for the draw in h2h markets (verify per region).
DRAW_OUTCOME_NAME = "Draw"

INSERT_ODDS_SQL = """
INSERT INTO odds_snapshots (
    fixture_id, book, captured_utc, home_odds, draw_odds, away_odds
) VALUES (
    :fixture_id, :book, :captured_utc, :home_odds, :draw_odds, :away_odds
)
ON CONFLICT (fixture_id, book, captured_utc) DO NOTHING
"""


class OddsHttpClient(Protocol):
    """Minimal HTTP client surface for ``fetch_odds`` (mockable in tests)."""

    def get(self, url: str, *, params: Mapping[str, Any], timeout: float) -> Any:
        ...


def fetch_odds(
    *,
    sport_key: str | None = None,
    api_key: str | None = None,
    regions: str | None = None,
    markets: str = "h2h",
    odds_format: str = "decimal",
    base_url: str | None = None,
    client: OddsHttpClient | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """GET /v4/sports/{sport}/odds and return the parsed JSON list of events.

    Parameters mirror the-odds-api v4 query string. Raises on HTTP errors.
    """
    key = api_key or config.ODDS_API_KEY
    if not key:
        raise ValueError("ODDS_API_KEY is not set")

    sport = sport_key or config.ODDS_API_SPORT_KEY
    url = f"{(base_url or config.ODDS_API_BASE_URL).rstrip('/')}/v4/sports/{sport}/odds"
    params: dict[str, Any] = {
        "apiKey": key,
        "regions": regions or config.ODDS_API_REGIONS,
        "markets": markets,
        "oddsFormat": odds_format,
    }
    http = client or requests
    response = http.get(url, params=params, timeout=timeout)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"expected list of events from odds API, got {type(payload)}")
    return payload


def _normalize_team(name: str) -> str:
    """Case-insensitive match key; strips accents so Curacao matches Curaçao."""
    import unicodedata

    text = name.strip()
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_marks.casefold()


def _parse_utc_timestamp(value: str) -> datetime:
    """Parse ISO-8601 timestamps from the API (always stored as UTC)."""
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _extract_h2h_decimal(
    outcomes: list[dict[str, Any]],
    *,
    home_team: str,
    away_team: str,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(home_odds, draw_odds, away_odds)`` from an h2h outcomes list."""
    home_odds: float | None = None
    draw_odds: float | None = None
    away_odds: float | None = None
    home_key = _normalize_team(home_team)
    away_key = _normalize_team(away_team)

    for outcome in outcomes:
        name = _normalize_team(str(outcome["name"]))
        price = float(outcome["price"])
        if name == home_key:
            home_odds = price
        elif name == away_key:
            away_odds = price
        elif name == _normalize_team(DRAW_OUTCOME_NAME):
            draw_odds = price

    return home_odds, draw_odds, away_odds


def parse_odds_events(
    events: list[dict[str, Any]],
    fixture_lookup: Mapping[tuple[str, str], int],
    *,
    default_captured_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    """Map API events to ``odds_snapshots`` insert rows.

    ``fixture_lookup`` maps ``(home_team_name, away_team_name)`` as returned by the
    API to our ``fixtures.fixture_id``. Events with no lookup entry are skipped
    (logged at INFO).
    """
    captured_default = default_captured_utc or datetime.now(UTC)
    rows: list[dict[str, Any]] = []

    for event in events:
        home_team = str(event["home_team"])
        away_team = str(event["away_team"])
        lookup_key = (_normalize_team(home_team), _normalize_team(away_team))
        fixture_id = fixture_lookup.get(lookup_key)
        if fixture_id is None:
            log.info(
                "odds: no fixture for %s vs %s; skipping",
                home_team,
                away_team,
            )
            continue

        for bookmaker in event.get("bookmakers", []):
            book_key = str(bookmaker["key"])
            last_update = bookmaker.get("last_update")
            captured = (
                _parse_utc_timestamp(last_update)
                if last_update
                else captured_default
            )

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                home_odds, draw_odds, away_odds = _extract_h2h_decimal(
                    market.get("outcomes", []),
                    home_team=home_team,
                    away_team=away_team,
                )
                rows.append(
                    {
                        "fixture_id": fixture_id,
                        "book": book_key,
                        "captured_utc": captured,
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                    }
                )

    return rows


def build_fixture_lookup_from_db(
    *,
    db_url: str | None = None,
) -> dict[tuple[str, str], int]:
    """Build ``(home_name, away_name)`` -> ``fixture_id`` from ``fixtures`` + ``teams``."""
    sql = """
        SELECT f.fixture_id,
               COALESCE(NULLIF(TRIM(th.odds_name), ''), th.name) AS home_name,
               COALESCE(NULLIF(TRIM(ta.odds_name), ''), ta.name) AS away_name
        FROM fixtures f
        JOIN teams th ON f.home_id = th.team_id
        JOIN teams ta ON f.away_id = ta.team_id
        WHERE f.home_id IS NOT NULL AND f.away_id IS NOT NULL
    """
    rows = db.fetch_all(sql, db_url=db_url)
    lookup: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (_normalize_team(row["home_name"]), _normalize_team(row["away_name"]))
        lookup[key] = row["fixture_id"]
    return lookup


def ingest_odds_snapshots(
    events: list[dict[str, Any]] | None = None,
    *,
    fixture_lookup: Mapping[tuple[str, str], int] | None = None,
    db_url: str | None = None,
    fetch_live: bool = False,
) -> tuple[int, int]:
    """Write odds snapshots idempotently.

    Pass ``events`` (or set ``fetch_live=True`` to call the API). Returns
    ``(attempted, inserted)``.
    """
    if events is None:
        if not fetch_live:
            raise ValueError("provide events=... or fetch_live=True")
        events = fetch_odds()

    lookup = fixture_lookup or build_fixture_lookup_from_db(db_url=db_url)
    rows = parse_odds_events(events, lookup)
    attempted = len(rows)
    inserted = db.execute(INSERT_ODDS_SQL, rows, db_url=db_url) if rows else 0
    inserted = max(inserted, 0)
    log.info("odds ingest: attempted=%d inserted=%d", attempted, inserted)
    return attempted, inserted


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    attempted, inserted = ingest_odds_snapshots(fetch_live=True)
    print(f"odds: attempted={attempted} inserted={inserted}")


if __name__ == "__main__":
    main()
