"""Tests for historical results ingest (no live Kaggle call in CI)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pitchedge import db
from pitchedge.ingest import history
from pitchedge.ingest.team_ids import team_name_to_id

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HISTORY_SAMPLE = FIXTURES_DIR / "history_sample.csv"

# Spot-check: scores verified against martj42 / standard football records.
SPOT_CHECK = (
    {
        "date": "1872-11-30",
        "home_team": "Scotland",
        "away_team": "England",
        "home_score": 0,
        "away_score": 0,
        "tournament": "Friendly",
    },
    {
        "date": "1930-07-13",
        "home_team": "France",
        "away_team": "Mexico",
        "home_score": 4,
        "away_score": 1,
        "tournament": "FIFA World Cup",
    },
    {
        "date": "1930-07-30",
        "home_team": "Uruguay",
        "away_team": "Argentina",
        "home_score": 4,
        "away_score": 2,
        "tournament": "FIFA World Cup",
    },
)


def test_team_name_to_id_is_stable():
    assert team_name_to_id("Spain") == team_name_to_id("Spain")
    assert team_name_to_id("Spain") != team_name_to_id("France")


def test_frame_to_rows_maps_columns():
    frame = history.load_history_frame(HISTORY_SAMPLE)
    rows = history.frame_to_rows(frame)
    assert len(rows) == 5
    first = rows[0]
    assert first["home_goals"] == 0
    assert first["away_goals"] == 0
    assert first["competition"] == "Friendly"
    assert first["neutral"] is False


@pytest.mark.parametrize("match", SPOT_CHECK)
def test_spot_check_scores_in_frame(match):
    frame = history.load_history_frame(HISTORY_SAMPLE)
    subset = frame[
        (frame["date"].astype(str) == match["date"])
        & (frame["home_team"] == match["home_team"])
        & (frame["away_team"] == match["away_team"])
    ]
    assert len(subset) == 1
    row = subset.iloc[0]
    assert int(row["home_score"]) == match["home_score"]
    assert int(row["away_score"]) == match["away_score"]
    assert row["tournament"] == match["tournament"]


def test_history_ingest_is_idempotent(empty_ingest_tables):
    path = HISTORY_SAMPLE
    history.ingest_history(path)
    count_after_first = db.fetch_one("SELECT count(*) AS n FROM raw_results")["n"]

    history.ingest_history(path)
    count_after_second = db.fetch_one("SELECT count(*) AS n FROM raw_results")["n"]

    assert count_after_first == 5
    assert count_after_second == count_after_first


def test_history_dedup_on_natural_key(empty_ingest_tables):
    """Duplicate logical row (same date, teams, competition) inserts once."""
    history.ingest_history(HISTORY_SAMPLE)
    before = db.fetch_one("SELECT count(*) AS n FROM raw_results")["n"]

    frame = history.load_history_frame(HISTORY_SAMPLE)
    rows = history.frame_to_rows(frame)
    db.execute(history.INSERT_SQL, rows)

    after = db.fetch_one("SELECT count(*) AS n FROM raw_results")["n"]
    assert before == after == 5


def test_spot_check_rows_in_database(empty_ingest_tables):
    history.ingest_history(HISTORY_SAMPLE)
    for match in SPOT_CHECK:
        home_id = team_name_to_id(match["home_team"])
        away_id = team_name_to_id(match["away_team"])
        row = db.fetch_one(
            """
            SELECT home_goals, away_goals, competition
            FROM raw_results
            WHERE date = :date
              AND home_id = :home_id
              AND away_id = :away_id
              AND competition = :competition
            """,
            {
                "date": match["date"],
                "home_id": home_id,
                "away_id": away_id,
                "competition": match["tournament"],
            },
        )
        assert row is not None
        assert row["home_goals"] == match["home_score"]
        assert row["away_goals"] == match["away_score"]
