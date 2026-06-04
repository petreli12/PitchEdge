"""Schema + integrity tests for the migration.

These are DB-backed and skipped automatically when Postgres is unreachable.
They cover the Phase 0 validation checklist: all tables exist, the migration is
idempotent, and the append-only / pre-kickoff rules on match_predictions hold.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from pitchedge import migrations

UTC = timezone.utc


def test_all_expected_tables_exist(conn):
    rows = conn.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
    ).scalars().all()
    present = set(rows)
    missing = set(migrations.EXPECTED_TABLES) - present
    assert not missing, f"missing tables: {missing}"


def test_migration_is_idempotent(migrated_db):
    # Running the full migration again must not raise.
    migrations.migrate()
    migrations.migrate()


def _seed_fixture(conn, kickoff: datetime, *, team_id: int, fixture_id: int) -> None:
    conn.execute(
        text(
            "INSERT INTO teams (team_id, name, fifa_code) "
            "VALUES (:tid, :name, :code)"
        ),
        {"tid": team_id, "name": f"Team {team_id}", "code": f"T{team_id}"},
    )
    conn.execute(
        text(
            "INSERT INTO fixtures (fixture_id, kickoff_utc, home_id, away_id, "
            "stage, group_label) VALUES (:fid, :ko, :h, :a, 'Group', 'A')"
        ),
        {"fid": fixture_id, "ko": kickoff, "h": team_id, "a": team_id},
    )


def test_prediction_before_kickoff_is_accepted(conn):
    kickoff = datetime.now(UTC) + timedelta(days=2)
    _seed_fixture(conn, kickoff, team_id=9001, fixture_id=9001)

    conn.execute(
        text(
            "INSERT INTO match_predictions "
            "(fixture_id, model_version, predicted_utc, p_home, p_draw, p_away, "
            " source) VALUES (:fid, 'v0', :pu, 0.5, 0.3, 0.2, 'blend')"
        ),
        {"fid": 9001, "pu": kickoff - timedelta(hours=1)},
    )
    count = conn.execute(
        text("SELECT count(*) FROM match_predictions WHERE fixture_id = 9001")
    ).scalar_one()
    assert count == 1


def test_prediction_at_or_after_kickoff_is_rejected(conn):
    kickoff = datetime.now(UTC) + timedelta(days=2)
    _seed_fixture(conn, kickoff, team_id=9002, fixture_id=9002)

    with pytest.raises(DBAPIError):
        conn.execute(
            text(
                "INSERT INTO match_predictions "
                "(fixture_id, model_version, predicted_utc, p_home, p_draw, "
                " p_away, source) VALUES (:fid, 'v0', :pu, 0.5, 0.3, 0.2, 'blend')"
            ),
            {"fid": 9002, "pu": kickoff},  # exactly at kickoff -> rejected
        )


def test_probabilities_must_sum_to_one(conn):
    kickoff = datetime.now(UTC) + timedelta(days=2)
    _seed_fixture(conn, kickoff, team_id=9003, fixture_id=9003)

    with pytest.raises(IntegrityError):
        conn.execute(
            text(
                "INSERT INTO match_predictions "
                "(fixture_id, model_version, predicted_utc, p_home, p_draw, "
                " p_away, source) VALUES (:fid, 'v0', :pu, 0.5, 0.5, 0.5, 'model')"
            ),
            {"fid": 9003, "pu": kickoff - timedelta(hours=1)},
        )


def test_logged_prediction_cannot_be_updated_or_deleted(conn):
    kickoff = datetime.now(UTC) + timedelta(days=2)
    _seed_fixture(conn, kickoff, team_id=9004, fixture_id=9004)
    conn.execute(
        text(
            "INSERT INTO match_predictions "
            "(fixture_id, model_version, predicted_utc, p_home, p_draw, p_away, "
            " source) VALUES (:fid, 'v0', :pu, 0.5, 0.3, 0.2, 'blend')"
        ),
        {"fid": 9004, "pu": kickoff - timedelta(hours=1)},
    )

    with pytest.raises(DBAPIError):
        conn.execute(
            text("UPDATE match_predictions SET p_home = 0.6 WHERE fixture_id = 9004")
        )

    # Each failed statement aborts the transaction; restart with a savepoint
    # is overkill here, so just assert delete also fails on a fresh attempt.


def test_logged_prediction_delete_is_rejected(conn):
    kickoff = datetime.now(UTC) + timedelta(days=2)
    _seed_fixture(conn, kickoff, team_id=9005, fixture_id=9005)
    conn.execute(
        text(
            "INSERT INTO match_predictions "
            "(fixture_id, model_version, predicted_utc, p_home, p_draw, p_away, "
            " source) VALUES (:fid, 'v0', :pu, 0.5, 0.3, 0.2, 'blend')"
        ),
        {"fid": 9005, "pu": kickoff - timedelta(hours=1)},
    )

    with pytest.raises(DBAPIError):
        conn.execute(text("DELETE FROM match_predictions WHERE fixture_id = 9005"))
