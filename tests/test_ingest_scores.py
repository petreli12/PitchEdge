"""Tests for the the-odds-api /scores results adapter (fixture JSON, no live HTTP)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from pitchedge import db
from pitchedge.ingest import fixtures as fixtures_ingest
from pitchedge.ingest import scores
from pitchedge.score import score_finished_matches

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCORES_SAMPLE = FIXTURES_DIR / "scores_api_sample.json"
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
def score_events() -> list[dict[str, Any]]:
    return json.loads(SCORES_SAMPLE.read_text())


def test_parse_completed_scores_extracts_goals(score_events):
    parsed = scores.parse_completed_scores(score_events)
    # 2 mapped + 1 unmapped are completed with scores; the not-started one is skipped.
    assert len(parsed) == 3
    first = parsed[0]
    assert first.home_team == "Team 1"
    assert first.away_team == "Team 2"
    assert (first.home_goals, first.away_goals) == (3, 1)
    assert first.commence_utc.date() == date(2026, 6, 12)


def test_parse_skips_not_completed(score_events):
    parsed = scores.parse_completed_scores(score_events)
    assert all(p.home_team != "Team 1" or p.away_team != "Team 4" for p in parsed)


def test_parse_skips_missing_numeric_score():
    events = [
        {
            "completed": True,
            "home_team": "Team 1",
            "away_team": "Team 2",
            "commence_time": "2026-06-12T17:00:00Z",
            "scores": [
                {"name": "Team 1", "score": None},
                {"name": "Team 2", "score": "1"},
            ],
        }
    ]
    assert scores.parse_completed_scores(events) == []


def test_match_fixture_detects_swap():
    index = [
        scores.FixtureEntry(
            fixture_id=2,
            match_date=date(2026, 6, 13),
            home_history_name="Team 3",
            away_history_name="Team 4",
            home_names=frozenset({"team 3"}),
            away_names=frozenset({"team 4"}),
        )
    ]
    direct = scores.match_fixture(index, "team 3", "team 4", date(2026, 6, 13))
    swapped = scores.match_fixture(index, "team 4", "team 3", date(2026, 6, 13))
    assert direct == (index[0], False)
    assert swapped == (index[0], True)
    assert scores.match_fixture(index, "team 9", "team 4", date(2026, 6, 13)) is None


def test_fetch_scores_uses_mock_client(score_events):
    client = _MockClient(score_events)
    result = scores.fetch_scores(
        sport_key="soccer_fifa_world_cup",
        api_key="test-key",
        days_from=3,
        client=client,
    )
    assert result == score_events
    assert client.last_params["apiKey"] == "test-key"
    assert client.last_params["daysFrom"] == 3
    assert client.last_params["dateFormat"] == "iso"


def test_fetch_scores_omits_days_from_when_zero(score_events):
    client = _MockClient(score_events)
    scores.fetch_scores(api_key="k", days_from=0, client=client)
    assert "daysFrom" not in (client.last_params or {})


def test_sync_marks_fixtures_final_with_correct_orientation(
    empty_ingest_tables, score_events
):
    fixtures_ingest.ingest_teams(TEAMS_CSV)
    fixtures_ingest.ingest_fixtures(FIXTURES_CSV)

    matched, marked = scores.sync_from_scores_api(score_events)
    assert matched == 2  # the two mapped completed games (unmapped skipped)
    assert marked == 2

    statuses = {
        r["fixture_id"]: r["status"]
        for r in db.fetch_all(
            "SELECT fixture_id, status FROM fixtures WHERE fixture_id IN (1, 2)"
        )
    }
    assert statuses == {1: "final", 2: "final"}

    # Fixture 2 came in swapped (API home Team 4 lost 0-2); the fixture's home is
    # Team 3, so the stored result must read Team 3 won 2-0 (orientation fixed).
    from pitchedge.ingest.team_ids import team_name_to_id

    rows = db.fetch_all(
        "SELECT home_id, away_id, home_goals, away_goals FROM raw_results "
        "WHERE date = :d",
        {"d": date(2026, 6, 13)},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["home_id"] == team_name_to_id("Team 3")
    assert row["away_id"] == team_name_to_id("Team 4")
    assert (row["home_goals"], row["away_goals"]) == (2, 0)


def test_sync_is_idempotent(empty_ingest_tables, score_events):
    fixtures_ingest.ingest_teams(TEAMS_CSV)
    fixtures_ingest.ingest_fixtures(FIXTURES_CSV)

    scores.sync_from_scores_api(score_events)
    before = db.fetch_one("SELECT count(*) AS n FROM raw_results")["n"]
    second_matched, second_marked = scores.sync_from_scores_api(score_events)
    after = db.fetch_one("SELECT count(*) AS n FROM raw_results")["n"]
    assert before == after
    # Already-final fixtures are not re-marked on a second pass.
    assert second_marked == 0


def test_synced_results_enable_scoring(empty_ingest_tables, score_events):
    """End-to-end: synced finals let score.py write prediction_scores."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import text

    from pitchedge.predict import INSERT_PREDICTION_SQL, build_prediction_rows
    from pitchedge.model.dixon_coles import DixonColesModel

    fixtures_ingest.ingest_teams(TEAMS_CSV)
    fixtures_ingest.ingest_fixtures(FIXTURES_CSV)

    model = DixonColesModel(
        attack={i: 0.0 for i in range(1, 60)},
        defense={i: 0.0 for i in range(1, 60)},
        home_adv=0.0,
        rho=0.0,
        xi=0.0,
    )
    ko = datetime(2026, 6, 12, 17, 0, tzinfo=timezone.utc)
    fix = {
        "fixture_id": 1,
        "home_id": 1,
        "away_id": 2,
        "kickoff_utc": ko,
        "home_name": "Team 1",
        "away_name": "Team 2",
    }
    rows = build_prediction_rows(
        fix,
        model,
        {1: 1, 2: 2},
        model_version="v1",
        predicted_utc=ko - timedelta(days=1),
        market_probs=(0.33, 0.34, 0.33),
    )
    for row in rows:
        db.execute(INSERT_PREDICTION_SQL, row)

    scores.sync_from_scores_api(score_events)
    attempted, inserted = score_finished_matches()
    assert attempted == len(rows)
    assert inserted == len(rows)

    outcome = db.fetch_all(
        "SELECT DISTINCT outcome FROM prediction_scores ps "
        "JOIN match_predictions mp ON mp.id = ps.prediction_id "
        "WHERE mp.fixture_id = 1"
    )
    assert {r["outcome"] for r in outcome} == {"H"}  # Team 1 won 3-1
