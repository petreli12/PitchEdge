"""Tests for WC teams and fixtures ingest."""

from __future__ import annotations

from pathlib import Path

from pitchedge import db
from pitchedge.ingest import fixtures

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEAMS_CSV = FIXTURES_DIR / "wc_teams_48.csv"
FIXTURES_CSV = FIXTURES_DIR / "wc_fixtures_104.csv"


def test_teams_csv_has_48_rows_across_12_groups():
    frame = fixtures.load_teams_frame(TEAMS_CSV)
    assert len(frame) == 48
    assert frame["group_label"].nunique() == 12
    assert all(frame.groupby("group_label").size() == 4)


def test_fixtures_csv_has_104_rows():
    frame = fixtures.load_fixtures_frame(FIXTURES_CSV)
    assert len(frame) == 104


def test_teams_and_fixtures_ingest_idempotent(empty_ingest_tables):
    fixtures.ingest_teams(TEAMS_CSV)
    fixtures.ingest_fixtures(FIXTURES_CSV)

    teams_n = db.fetch_one("SELECT count(*) AS n FROM teams")["n"]
    fix_n = db.fetch_one("SELECT count(*) AS n FROM fixtures")["n"]
    assert teams_n == 48
    assert fix_n == 104

    fixtures.ingest_teams(TEAMS_CSV)
    fixtures.ingest_fixtures(FIXTURES_CSV)

    assert db.fetch_one("SELECT count(*) AS n FROM teams")["n"] == teams_n
    assert db.fetch_one("SELECT count(*) AS n FROM fixtures")["n"] == fix_n


def test_team_insert_dedup_on_pk(empty_ingest_tables):
    fixtures.ingest_teams(TEAMS_CSV)
    rows = fixtures.teams_frame_to_rows(fixtures.load_teams_frame(TEAMS_CSV))
    db.execute(fixtures.INSERT_TEAM_SQL, rows)
    assert db.fetch_one("SELECT count(*) AS n FROM teams")["n"] == 48
