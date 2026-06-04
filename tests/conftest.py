"""Shared pytest fixtures.

DB-backed tests are gated behind a connectivity check: if Postgres is not
reachable (e.g. ``make db-up`` was not run), those tests are skipped rather than
failing, so ``make test`` always collects and runs.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from pitchedge import config, db, migrations


def _db_reachable() -> bool:
    """Fast probe: a short connect timeout so the suite skips quickly when
    Postgres is down instead of blocking on a dropped connection."""
    probe = create_engine(
        config.DB_URL, connect_args={"connect_timeout": 3}, future=True
    )
    try:
        with probe.connect():
            return True
    except Exception:
        return False
    finally:
        probe.dispose()


@pytest.fixture(scope="session")
def migrated_db() -> None:
    """Ensure the schema is applied once per test session, or skip if no DB."""
    if not _db_reachable():
        pytest.skip("Postgres not reachable; run `make db-up` to enable DB tests")
    migrations.migrate()


TRUNCATE_INGEST_SQL = """
TRUNCATE TABLE
    odds_snapshots,
    prediction_scores,
    match_predictions,
    sim_results,
    team_ratings,
    fixtures,
    teams,
    raw_results
RESTART IDENTITY CASCADE
"""


def _assert_test_database() -> None:
    """Hard guard: refuse to truncate unless connected to a ``*_test`` database.

    The ingest tables are destructive to clear. This guard makes it impossible
    for a truncating fixture to run against the live ``pitchedge`` database even
    if someone invokes ``pytest`` directly with a live ``DB_URL`` (the earlier
    data wipe happened exactly this way). ``make test`` points at
    ``pitchedge_test``; anything else aborts the run loudly.
    """
    from pitchedge import db

    rows = db.fetch_all("SELECT current_database() AS name")
    dbname = str(rows[0]["name"]) if rows else ""
    if not dbname.endswith("_test"):
        pytest.fail(
            f"refusing to truncate ingest tables on database '{dbname}': "
            "DB-backed tests must run against a '*_test' database. "
            "Use `make test` (TEST_DB_URL=pitchedge_test) or set DB_URL to a "
            "test database. This guard protects the live `pitchedge` data."
        )


@pytest.fixture
def empty_ingest_tables(migrated_db):
    """Clear ingest-related tables at test **start** so integration tests are isolated.

    Does NOT truncate after the test. A previous teardown truncate here wiped
    real dev data when ``make test`` ran after ``make ingest-*`` on the same DB.
    """
    from pitchedge import db

    _assert_test_database()
    db.execute_script(TRUNCATE_INGEST_SQL)
    yield


@pytest.fixture
def conn(migrated_db):
    """A connection wrapped in a transaction that is always rolled back, so DB
    tests leave no residue."""
    engine = db.get_engine()
    connection = engine.connect()
    trans = connection.begin()
    try:
        yield connection
    finally:
        trans.rollback()
        connection.close()
