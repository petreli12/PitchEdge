"""Tests for the-odds-api adapter (fixture JSON only, no live HTTP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pitchedge import db
from pitchedge.ingest import fixtures, odds

FIXTURES_DIR = Path(__file__).parent / "fixtures"
ODDS_SAMPLE = FIXTURES_DIR / "odds_api_sample.json"
TEAMS_CSV = FIXTURES_DIR / "wc_teams_48.csv"
FIXTURES_CSV = FIXTURES_DIR / "wc_fixtures_104.csv"

class _MockResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _MockClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.last_params: dict[str, Any] | None = None

    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> _MockResponse:
        self.last_params = params
        return _MockResponse(self.payload)


@pytest.fixture
def odds_events() -> list[dict[str, Any]]:
    return json.loads(ODDS_SAMPLE.read_text())


def test_normalize_team_matches_accented_api_names():
    from pitchedge.ingest.odds import _normalize_team

    assert _normalize_team("Curaçao") == _normalize_team("Curacao")
    assert _normalize_team("Bosnia & Herzegovina") == _normalize_team(
        "Bosnia & Herzegovina"
    )


def test_parse_odds_extracts_decimal_h2h(odds_events):
    lookup = {("team 1", "team 2"): 1}
    rows = odds.parse_odds_events(odds_events, lookup)
    pinnacle = [r for r in rows if r["book"] == "pinnacle"][0]
    assert pinnacle["fixture_id"] == 1
    assert pinnacle["home_odds"] == pytest.approx(2.10)
    assert pinnacle["draw_odds"] == pytest.approx(3.25)
    assert pinnacle["away_odds"] == pytest.approx(3.40)


def test_parse_odds_skips_unmapped_events(odds_events):
    lookup = {("team 1", "team 2"): 1}
    rows = odds.parse_odds_events(odds_events, lookup)
    books = {r["book"] for r in rows}
    assert "pinnacle" in books
    assert "bet365" in books
    assert all(r["fixture_id"] == 1 for r in rows)


def test_fetch_odds_uses_mock_client(odds_events):
    client = _MockClient(odds_events)
    result = odds.fetch_odds(
        sport_key="soccer_fifa_world_cup",
        api_key="test-key",
        client=client,
    )
    assert result == odds_events
    assert client.last_params["apiKey"] == "test-key"
    assert client.last_params["oddsFormat"] == "decimal"


def test_odds_ingest_idempotent(empty_ingest_tables, odds_events):
    fixtures.ingest_teams(TEAMS_CSV)
    fixtures.ingest_fixtures(FIXTURES_CSV)
    lookup = odds.build_fixture_lookup_from_db()

    odds.ingest_odds_snapshots(odds_events, fixture_lookup=lookup)
    first = db.fetch_one("SELECT count(*) AS n FROM odds_snapshots")["n"]
    assert first == 2  # pinnacle + bet365 h2h for fixture 1; unmapped event skipped

    odds.ingest_odds_snapshots(odds_events, fixture_lookup=lookup)
    second = db.fetch_one("SELECT count(*) AS n FROM odds_snapshots")["n"]
    assert second == first


def test_odds_snapshot_dedup_on_natural_key(empty_ingest_tables, odds_events):
    fixtures.ingest_teams(TEAMS_CSV)
    fixtures.ingest_fixtures(FIXTURES_CSV)
    lookup = odds.build_fixture_lookup_from_db()
    rows = odds.parse_odds_events(odds_events, lookup)
    odds.ingest_odds_snapshots(odds_events, fixture_lookup=lookup)
    before = db.fetch_one("SELECT count(*) AS n FROM odds_snapshots")["n"]
    db.execute(odds.INSERT_ODDS_SQL, rows)
    after = db.fetch_one("SELECT count(*) AS n FROM odds_snapshots")["n"]
    assert before == after
